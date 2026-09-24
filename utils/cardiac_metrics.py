"""Cardiac functional metrics for CAMUS and EchoNet evaluation."""
from pathlib import Path

import numpy as np

from .config import CAMUS_TASK, ECHONET_TASK, TASKS


def require_ef_dependencies():
    try:
        from scipy import stats  # noqa: F401
        from skimage.measure import find_contours, label, regionprops  # noqa: F401
    except ImportError as error:
        raise RuntimeError("--compute_ef requires scipy and scikit-image; install requirements.txt") from error


def compute_left_ventricle_volumes(*args, **kwargs):
    # Defer optional EF dependencies until EF is requested.
    from .compute_ef import compute_left_ventricle_volumes as calculate
    return calculate(*args, **kwargs)


def compute_left_ventricle_volumes_single_plane(*args, **kwargs):
    from .compute_ef import compute_left_ventricle_volumes_single_plane as calculate
    return calculate(*args, **kwargs)


def summarize_cardiac_values(ground_truth, predictions):
    """test_npz: signed bias, error std, Pearson r, and Wilcoxon when n >= 10."""
    gt = np.asarray(ground_truth, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if gt.ndim != 1 or gt.shape != pred.shape:
        raise ValueError("Cardiac ground truth and predictions must be aligned 1D arrays")
    result = dict(count=int(gt.size), mae=None, bias=None, std=None, corr=None,
                  wilcoxon_signed_rank_test=None)
    if not gt.size:
        return result
    errors = pred - gt
    result["mae"] = float(np.abs(errors).mean())
    result["bias"] = float(np.mean(errors))
    result["std"] = float(np.std(errors))
    if gt.size >= 2 and gt.std() > 0 and pred.std() > 0:
        result["corr"] = float(np.corrcoef(gt, pred)[0, 1])
    if gt.size >= 10:
        if np.all(errors == 0):
            result["wilcoxon_signed_rank_test"] = {"statistic": 0.0, "pvalue": 1.0}
        else:
            from scipy import stats
            test = stats.wilcoxon(gt, pred)
            result["wilcoxon_signed_rank_test"] = {
                "statistic": float(test.statistic), "pvalue": float(test.pvalue),
            }
    return result


class CardiacMetrics:
    def __init__(self, task=CAMUS_TASK):
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}")
        require_ef_dependencies()
        self.single_plane = task == ECHONET_TASK
        self.metric_names = ("ef",) if self.single_plane else ("ef", "edv", "esv")
        self.patients = {}
        self.ground_truth = {}

    def add_sample(self, image_name, predictions, spacing, ef, edv, esv):
        if self.single_plane:
            patient, view = Path(image_name).stem, "4CH"
        else:
            parts = Path(image_name).stem.split("_")
            if len(parts) < 2 or parts[1].upper() not in ("2CH", "4CH"):
                raise ValueError(f"Expected CAMUS patient_2CH/4CH filename, got {image_name!r}")
            patient, view = parts[0], parts[1].upper()
        predictions = np.asarray(predictions)
        if predictions.ndim != 3 or predictions.shape[0] < 2:
            raise ValueError("EF requires endpoint predictions in [T, H, W] format")
        views = self.patients.setdefault(patient, {})
        if view in views:
            raise ValueError(f"Duplicate {view} video for {patient}")
        views[view] = {
            "ed": predictions[0].astype(np.uint8),
            "es": predictions[-1].astype(np.uint8),
            "spacing": np.asarray(spacing).reshape(-1)[:2][::-1].copy(),
        }
        # Match the reference's per-patient metadata assignment on each sample.
        self.ground_truth[patient] = {"ef": float(ef), "edv": float(edv), "esv": float(esv)}

    def compute(self):
        records, skipped, failures = [], [], []
        for patient, views in self.patients.items():
            required = {"4CH"} if self.single_plane else {"2CH", "4CH"}
            missing = sorted(required - views.keys())
            if missing:
                skipped.append({"patient": patient, "reason": f"Missing view(s): {', '.join(missing)}"})
                continue
            swapped = False
            try:
                for view, data in views.items():
                    spacing = data["spacing"]
                    if spacing.size != 2 or not np.isfinite(spacing).all() or np.any(spacing <= 0):
                        raise ValueError(f"Invalid pixel spacing in {view}")
                a4c = views["4CH"]
                if self.single_plane:
                    edv, esv = compute_left_ventricle_volumes_single_plane(
                        a4c_ed=a4c["ed"], a4c_es=a4c["es"], a4c_voxelspacing=a4c["spacing"],
                    )
                else:
                    a2c = views["2CH"]
                    edv, esv = compute_left_ventricle_volumes(
                        a2c_ed=a2c["ed"], a2c_es=a2c["es"], a2c_voxelspacing=a2c["spacing"],
                        a4c_ed=a4c["ed"], a4c_es=a4c["es"], a4c_voxelspacing=a4c["spacing"],
                    )
                if not np.isfinite([edv, esv]).all():
                    raise ValueError("Non-finite ventricular volumes")
                if esv > edv:
                    edv, esv = esv, edv
                    swapped = True
                ef = round(100.0 * (edv - esv) / edv, 2) if edv > 1e-8 else 0.0
            except Exception as error:
                if not self.single_plane:
                    skipped.append({"patient": patient, "reason": str(error)})
                    continue
                # eval_echonet in the reference retains failed cases with EF=0.
                failures.append({"patient": patient, "reason": str(error), "fallback_ef": 0.0})
                ef, edv, esv = 0.0, None, None
            predicted = {"ef": float(ef), "edv": edv, "esv": esv}
            records.append({
                "patient": patient,
                "gt": {key: self.ground_truth[patient][key] for key in self.metric_names},
                "pred": {key: float(predicted[key]) for key in self.metric_names},
                "volumes_swapped": swapped,
            })
        units = {"ef": "percent (errors in percentage points)", "edv": "mL", "esv": "mL"}
        result = {
            "source": "utils/cardiac_metrics.py and utils/compute_ef.py",
            "method": "single_plane_pca_simpson" if self.single_plane else "biplane_simpson",
            "metric_names": list(self.metric_names),
            "units": {key: units[key] for key in self.metric_names},
            "patients_total": len(self.patients), "patients_evaluated": len(records),
            "patients_skipped": len(skipped), "patients": records, "skipped": skipped,
            "failures": failures,
        }
        if self.single_plane:
            result["note"] = (
                "EchoNet uses the reference PCA single-plane calculation and evaluates EF only; "
                "placeholder spacing does not calibrate EDV/ESV in mL. "
                "Failed calculations are retained as EF=0, matching test_npz."
            )
        for key in self.metric_names:
            result[key] = summarize_cardiac_values(
                [record["gt"][key] for record in records], [record["pred"][key] for record in records],
            )
        return result
