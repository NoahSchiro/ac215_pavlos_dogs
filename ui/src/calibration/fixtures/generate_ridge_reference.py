"""Generate golden cases with PR #60's unmodified Python reference.

Usage: python generate_ridge_reference.py /path/to/pinned/calibration.py
Requires NumPy only. The source blob is checked before executing it.
"""

import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

COMMIT = "5340538d798e30fe2131cbebb54f38e00eafd8c9"
BLOB = "c5f29ed13e981c9f291fda79d83510144f4fd93d"
SOURCE = "model/src/training/calibration.py"


def main() -> None:
    reference = Path(sys.argv[1])
    content = reference.read_bytes()
    actual_blob = hashlib.sha1(
        f"blob {len(content)}\0".encode() + content
    ).hexdigest()
    if actual_blob != BLOB:
        raise ValueError(f"Expected reference blob {BLOB}, got {actual_blob}")
    fit_ridge = runpy.run_path(str(reference))["fit_ridge"]
    predictions = [
        [0.12, 0.22], [0.48, 0.15], [0.83, 0.31], [0.2, 0.64],
        [0.55, 0.57], [0.91, 0.79], [0.35, 0.88], [0.72, 0.46],
    ]
    targets = [
        [0.18, 0.24], [0.49, 0.12], [0.81, 0.2], [0.3, 0.65],
        [0.61, 0.52], [0.94, 0.68], [0.48, 0.84], [0.75, 0.36],
    ]
    probes = np.array([[0.15, 0.85], [0.7, 0.2], [-0.3, 1.4]])
    inputs = [
        (f"asymmetric-lambda-{lamb:g}", predictions, targets, lamb)
        for lamb in (0.0, 0.1, 2.0, 1e12)
    ]
    inputs += [
        ("single-sample", [[0.2, 0.3]], [[0.5, 0.7]], 0.1),
        ("constant-predictions", [[0.2, 0.3]] * 3,
         [[0.1, 0.2], [0.5, 0.7], [0.9, 0.8]], 0.1),
        ("collinear-predictions", [[0.1, 0.2], [0.3, 0.6], [0.4, 0.8]],
         [[0.2, 0.1], [0.6, 0.5], [0.8, 0.9]], 0.1),
        ("nearly-collinear-regularized", [[0.1, 0.2], [0.3, 0.6], [0.4, 0.800000000001]],
         [[0.2, 0.1], [0.6, 0.5], [0.8, 0.9]], 0.1),
    ]
    cases = []
    for name, x, y, lamb in inputs:
        fit = fit_ridge(np.array(x), np.array(y), lamb=lamb)
        cases.append({
            "name": name, "predictions": x, "targets": y, "lambda": lamb,
            "weight": fit.weight.tolist(), "intercept": fit.intercept.tolist(),
            "applied": fit.apply(probes).tolist(),
        })
    fixture = {
        "source": f"https://github.com/NoahSchiro/ac215_pavlos_dogs/blob/{COMMIT}/{SOURCE}",
        "sourceBlob": BLOB, "numpyVersion": np.__version__,
        "absoluteTolerance": 1e-12, "probes": probes.tolist(), "cases": cases,
    }
    Path(__file__).with_name("ridge-reference.json").write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    )
    print(f"Generated {len(cases)} cases with NumPy {np.__version__}")


if __name__ == "__main__":
    main()
