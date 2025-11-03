import os
import random
import time
from collections import deque
from pathlib import Path
from random import Random

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.optim import SGD, Muon

import wandb
from model import GPT, ModelConfig, rms_norm

torch.set_num_threads(1)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(0)
np.random.seed(0)
BATCH_SIZE = 512 + 256 + 128
LR = 512e-5
MOMENTUM = 0.95
EPS = 1e-8
WEIGHT_DECAY = 1e-2
WARMUP = 100
LOSS_QUEUE_LEN = 200
NUM_ATTN_HEADS = 16
SEQ_LEN = 128
HIDDEN_DIM = 1024
INTERMEDIATE_DIM = HIDDEN_DIM * 4
MAX_TRAINING_STEPS = 10_000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    run = wandb.init(
        project="baseline",
        name="muon-run",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "eps": EPS,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "num_warmup_step": WARMUP,
            "loss_queue_length": LOSS_QUEUE_LEN,
            "num_attention_heads": NUM_ATTN_HEADS,
            "intermediate_dim": INTERMEDIATE_DIM,
            "seq_len": SEQ_LEN,
        },
    )

    model_config = ModelConfig(
        hidden_dim=HIDDEN_DIM,
        num_attention_heads=NUM_ATTN_HEADS,
        intermediate_dim=INTERMEDIATE_DIM,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        net = GPT(config=model_config).to(device=DEVICE)
        net = torch.compile(net, dynamic=False, fullgraph=True)
        optimizer = Optimizer(net, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    print(net)
    print(f"Total parameters: {net.num_parameters():.3E}")
    step = 0
    start_time = time.perf_counter()
    start_training = start_time

    losses = deque(maxlen=LOSS_QUEUE_LEN)

    for step in range(MAX_TRAINING_STEPS + 1):
        lr = get_lr(step)
        optimizer.set_lr(lr)
        optimizer.zero_grad()
        data = get_batch_data(
            "data", "train", SEQ_LEN + 1, batch_size=BATCH_SIZE, step=step
        )
        data = data.to(dtype=torch.long, device=DEVICE)
        x, y = data[:, :-1], data[:, 1:]

        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            loss = loss_fn(net, x, y)

        loss.backward()
        grad_norm = compute_grad_norm(net.parameters())

        optimizer.step()
        losses.append(loss.data)
        pplx = torch.exp(loss)
        if step % 100 == 0:
            end_time = time.perf_counter()
            duration = end_time - start_time
            elapsed_time = end_time - start_training
            start_time = end_time
            avg_loss = sum(losses) / len(losses)
            avg_pplx = torch.exp(avg_loss).item()
            print(
                f"step: {step:>5}  pplx: {avg_pplx:.5f}  loss: {avg_loss:.5f}  duration: {duration:.1f}s  elapsed_time: {elapsed_time:.1f}s"
            )
            run.log(
                {
                    "avg_loss": avg_loss.item(),
                    "avg_pplx": avg_pplx,
                    "step": step,
                    "lr": lr,
                    "duration": duration,
                    "elapsed_time": elapsed_time,
                },
                commit=False,
            )
        run.log(
            {
                "loss": loss.item(),
                "pplx": pplx.item(),
                "step": step,
                "gradnorm": grad_norm,
                "lr": lr,
            }
        )
    end_training = time.perf_counter()
    print(f"DURATION = {end_training-start_training:.3f}s")
    run.finish()


def get_batch_data(data_path, split, seq_lenth, batch_size, *, step):
    data_path = Path(data_path) / f"{split}.bin"
    N = data_path.stat().st_size
    data = torch.from_file(data_path.as_posix(), dtype=torch.uint8, size=N)
    batch_data = []
    L = data.shape[0]
    for i in range(batch_size):
        rng = Random(step * batch_size + i)
        left = rng.randint(0, L - seq_lenth)
        right = left + seq_lenth
        batch_data.append(data[left:right])
    return torch.stack(batch_data, dim=0)


def loss_fn(net, x, y):
    logits = net(x).float()
    L = logits.shape[-1]
    xe_loss = cross_entropy(logits.view(-1, L), y.reshape(-1))
    return xe_loss


def get_lr(it):
    if it < WARMUP:
        return LR * (it + 1) / (WARMUP + 1)
    return LR


@torch.no_grad()
def compute_grad_norm(params):
    return sum(p.grad.square().sum() for p in params).sqrt()


class EmbedOptimizer(SGD):
    """Normalizing embedding grad by its RMS norm"""

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0):
        assert momentum > 0
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def step(self):
        """Perform a single optimization step."""
        for group in self.param_groups:
            params = []
            grads = []
            momentum_buffer_list = []
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            dampening = group["dampening"]
            lr = group["lr"]
            self._init_group(group, params, grads, momentum_buffer_list)
            for i, param in enumerate(params):
                buf = momentum_buffer_list[i]
                grad = grads[i]

                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                if buf is None:
                    buf = grad.detach().clone()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                update = rms_norm(buf)
                param.data.add_(update, alpha=-lr)
            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer


class Optimizer:
    def __init__(self, model, lr, momentum, weight_decay):
        super().__init__()
        matrix_params = []
        for name, p in model.named_parameters():
            if name.endswith("_proj.weight"):
                matrix_params.append(p)
        self.muon = Muon(
            matrix_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            adjust_lr_fn="original",
        )
        self.embed_optimizer = EmbedOptimizer(
            model.embed.parameters(),
            lr=lr,
            momentum=momentum,
        )

    def zero_grad(self, set_to_none=False):
        self.embed_optimizer.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.embed_optimizer.step()
        self.muon.step()

    def set_lr(self, lr):
        for param_group in self.embed_optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in self.muon.param_groups:
            param_group["lr"] = lr


if __name__ == "__main__":
    main()
