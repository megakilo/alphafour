import numpy as np
import torch


def load_checkpoint(path, map_location=None, allow_unsafe_fallback=False):
    """
    Prefer PyTorch's safe weights-only loading path. Fall back to the full
    unpickler only for trusted local checkpoints when explicitly allowed.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not allow_unsafe_fallback:
            raise

        print(
            f"Falling back to trusted checkpoint load for {path}. "
            f"Only use this with local checkpoints you trust. ({exc})"
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def serialize_replay_buffer(replay_buffer):
    serialized = []
    for iteration_examples in replay_buffer:
        serialized_iteration = []
        for board, pi, value in iteration_examples:
            serialized_iteration.append(
                {
                    "board": torch.as_tensor(np.ascontiguousarray(board), dtype=torch.int8),
                    "pi": torch.as_tensor(np.ascontiguousarray(pi), dtype=torch.float32),
                    "value": float(value),
                }
            )
        serialized.append(serialized_iteration)
    return serialized


def deserialize_replay_buffer(serialized_replay_buffer):
    replay_buffer = []
    for iteration_examples in serialized_replay_buffer:
        restored_iteration = []
        for example in iteration_examples:
            if isinstance(example, dict):
                board = example["board"].cpu().numpy().astype(np.int8, copy=False)
                pi = example["pi"].cpu().numpy().astype(np.float32, copy=False)
                value = float(example["value"])
            else:
                board, pi, value = example
                board = np.asarray(board, dtype=np.int8)
                pi = np.asarray(pi, dtype=np.float32)
                value = float(value)

            restored_iteration.append((board, pi, value))
        replay_buffer.append(restored_iteration)
    return replay_buffer
