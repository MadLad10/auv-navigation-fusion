"""Recreate the NumPy inference test fixture with optional PyTorch."""
from pathlib import Path

import numpy as np
import torch

from auv_navigation_fusion.learned_velocity import default_path


def main():
    with np.load(default_path()) as weights:
        gru = torch.nn.GRU(15, 32, batch_first=True)
        head = torch.nn.Linear(32, 3)
        gru.load_state_dict({"weight_ih_l0": torch.from_numpy(weights["w_ih"]),
                             "weight_hh_l0": torch.from_numpy(weights["w_hh"]),
                             "bias_ih_l0": torch.from_numpy(weights["b_ih"]),
                             "bias_hh_l0": torch.from_numpy(weights["b_hh"])})
        head.load_state_dict({"weight": torch.from_numpy(weights["w_out"]),
                              "bias": torch.from_numpy(weights["b_out"])})
    features = np.random.default_rng(19).normal(0, 0.5, (40, 15)).astype(np.float32)
    with torch.no_grad():
        _, hidden = gru(torch.from_numpy(features)[None])
        prediction = head(hidden[-1]).numpy()[0]
    destination = Path(__file__).resolve().parents[1] / "tests/fixtures/gru_reference.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, normalized_features=features, torch_prediction=prediction)
    print(destination)


if __name__ == "__main__":
    main()
