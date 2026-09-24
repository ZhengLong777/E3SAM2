from dataclasses import dataclass
from pathlib import Path


CAMUS_FULL_TASK = "CAMUS_Video_Full"
CAMUS_SEMI_TASK = "CAMUS_Video_Semi"
CAMUS_TASK = CAMUS_SEMI_TASK
ECHONET_TASK = "EchoNet_Video"
CAMUS_TASKS = (CAMUS_FULL_TASK, CAMUS_SEMI_TASK)
TASKS = (*CAMUS_TASKS, ECHONET_TASK)
CAMUS_SPLIT = "7-1-2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_CAMUS_DATA_PATH = (
    DATA_ROOT / "CAMUS_Processed" / "Edge_Kernel_Size3_dilated_size3"
)
DEFAULT_ECHONET_DATA_PATH = (
    DATA_ROOT / "EchoNet_Processed" / "Resized128_Edge_Kernel_Size3" / "echocycle"
)


def resolve_data_settings(
    task: str,
    camus_data_path: str | Path = DEFAULT_CAMUS_DATA_PATH,
    echo_data_path: str | Path = DEFAULT_ECHONET_DATA_PATH,
    data_override: str | Path | None = None,
) -> tuple[Path, int]:
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")

    if data_override is not None:
        data_path = Path(data_override)
    elif task == ECHONET_TASK:
        data_path = Path(echo_data_path)
    else:
        data_path = Path(camus_data_path)

    image_size = 128 if task == ECHONET_TASK else 256
    return data_path, image_size


def resolve_supervised_frames(task: str, frame_length: int) -> tuple[int, ...]:
    if task == CAMUS_FULL_TASK:
        return tuple(range(frame_length))
    if task in (CAMUS_SEMI_TASK, ECHONET_TASK):
        return (0, -1)
    raise ValueError(f"Unsupported task: {task}")


def resolve_evaluation_frames(task: str, frame_length: int) -> tuple[int, ...]:
    if task in CAMUS_TASKS:
        return tuple(range(frame_length))
    if task == ECHONET_TASK:
        return (0, -1)
    raise ValueError(f"Unsupported task: {task}")


@dataclass
class FinalConfig:
    data_path: Path
    output_dir: Path = Path("runs")
    task: str = CAMUS_TASK
    image_size: int = 256
    frame_length: int = 10
    batch_size: int = 8
    epochs: int = 310
    base_lr: float = 1e-4
    warmup_iterations: int = 250
    weight_decay: float = 0.1
    mask_loss_weight: float = 0.8
    edge_loss_weight: float = 0.2
    mask_loss: str = "bce_dice"
    positive_weight: float = 2.0
    regularization_weight: float = 1e-2
    regularization_start_epoch: int = 30
    prediction_threshold: float = 0.6
    seed: int = 1234
    num_workers: int = 8
    supervised_frames: tuple[int, ...] = (0, -1)
    evaluation_frames: tuple[int, ...] = tuple(range(10))
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    def __post_init__(self):
        self.supervised_frames = resolve_supervised_frames(self.task, self.frame_length)
        self.evaluation_frames = resolve_evaluation_frames(self.task, self.frame_length)
