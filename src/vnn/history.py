from typing import Iterator


class History:
    def __init__(self, metrics: list[str]):
        self.metrics = metrics
        self._history: list[dict[str, float]] = list()

    def __len__(self) -> int:
        return len(self._history)

    def __iter__(self) -> Iterator[dict[str, float]]:
        return iter(self._history)

    def __getitem__(self, idx: int) -> dict[str, float]:
        return self._history[idx]

    def save(self, **metrics: float) -> None:
        """Save metrics."""
        if set(metrics) != set(self.metrics):
            raise ValueError(f"Expected metrics {self.metrics}, got {list(metrics)}")

        self._history.append(metrics)

    def last(self) -> dict[str, float]:
        """Return last saved metrics."""
        return self._history[-1].copy()
