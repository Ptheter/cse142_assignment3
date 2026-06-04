"""Training utilities: batching and text generation."""

from __future__ import annotations

import torch

from transformer_lm.nn_utils import softmax


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input-target pairs from a 1-D token array.

    Args:
        data: 1-D tensor of token IDs.
        batch_size: Number of examples per batch.
        context_length: Number of tokens in each sequence.
        device: Device to place tensors on.

    Returns:
        ``(x, y)`` both of shape ``(batch_size, context_length)``.
    """

    if len(data) <= context_length:
        raise ValueError(
            "data must contain at least context_length + 1 tokens"
        )
    data = data.to(torch.long)
    max_start = len(data) - context_length - 1

    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,),
    )

    x = torch.stack([
        data[i : i + context_length]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1 : i + context_length + 1]
        for i in starts
    ])

    return x.to(device), y.to(device)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    context_length: int | None = None,
) -> list[int]:
    """Autoregressively generate tokens from a language model.

    Args:
        model: Maps ``(B, T)`` integer input to ``(B, T, vocab_size)`` logits.
        prompt_ids: Starting token IDs.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        context_length: Maximum context window (defaults to ``model.context_length``).

    Returns:
        List of token IDs (prompt + generated).
    """
    if context_length is None:
        context_length = model.context_length

    tokens = list(prompt_ids)

    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        x = torch.tensor(
            [tokens[-context_length:]],
            dtype=torch.long,
            device=device,
        )

        logits = model(x)           # (1, T, vocab_size)
        logits = logits[:, -1, :]   # (1, vocab_size)

        logits = logits / temperature

        probs = softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1,
        ).item()

        tokens.append(next_token)

    return tokens