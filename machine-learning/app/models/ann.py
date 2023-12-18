from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

from ann.ann import Ann
from app.schemas import ndarray_f32, ndarray_i32, ndarray_i64

from ..config import log, settings


class AnnSession:
    """
    Wrapper for ANN to be drop-in replacement for ONNX session.
    """

    def __init__(self, model_path: Path):
        tuning_file = Path(settings.cache_folder) / "gpu-tuning.ann"
        with open(tuning_file, "a"):
            pass
        self.ann = Ann(tuning_level=3, tuning_file=tuning_file.as_posix())
        log.info("Loading ANN model %s ...", model_path)
        cache_file = model_path.with_suffix(".anncache")
        save = False
        if not cache_file.exists():
            save = True
            with open(cache_file, mode="a"):
                pass

        self.model = self.ann.load(
            model_path.as_posix(),
            save_cached_network=save,
            cached_network_path=cache_file.as_posix(),
        )
        log.info("Loaded ANN model with ID %d", self.model)

    def __del__(self) -> None:
        self.ann.unload(self.model)
        log.info("Unloaded ANN model %d", self.model)
        self.ann.destroy()

    def get_inputs(self) -> List[AnnNode]:
        shapes = self.ann.input_shapes[self.model]
        return [AnnNode(None, s) for s in shapes]

    def get_outputs(self) -> List[AnnNode]:
        shapes = self.ann.output_shapes[self.model]
        return [AnnNode(None, s) for s in shapes]

    def run(self, output_names: List[str] | None, input_feed: Any, run_options: Any = None) -> List[ndarray_f32]:
        inputs = [*input_feed.values()]
        return self.ann.execute(self.model, inputs)


class AnnNode(NamedTuple):
    name: str | None
    shape: Tuple[int, ...]
