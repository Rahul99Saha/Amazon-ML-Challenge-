"""Timestamped, immediately-flushed progress logging (visible live in Kaggle job logs)."""
import logging
import sys
import time

_FMT = "%(asctime)s %(name)-10s %(message)s"


def get(name: str) -> logging.Logger:
    log = logging.getLogger(f"ber.{name}")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def _fmt_s(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Progress:
    """Logs `what: done/total (pct) elapsed ETA extra` at most every `every` seconds."""

    def __init__(self, log: logging.Logger, what: str, total: int, every: float = 60.0):
        self.log, self.what, self.total, self.every = log, what, max(total, 1), every
        self.done, self.t0, self.last = 0, time.time(), 0.0

    def step(self, n: int = 1, **extra) -> None:
        self.done += n
        now = time.time()
        if now - self.last >= self.every or self.done >= self.total:
            self.last = now
            el = now - self.t0
            eta = el / self.done * (self.total - self.done) if self.done else 0
            info = "  ".join(f"{k}={v}" for k, v in extra.items())
            self.log.info(f"{self.what}: {self.done:,}/{self.total:,} ({100 * self.done / self.total:.0f}%) "
                          f"elapsed {_fmt_s(el)} ETA {_fmt_s(eta)}  {info}")


class timed:
    """Context manager: logs start and duration of a block."""

    def __init__(self, log: logging.Logger, what: str):
        self.log, self.what = log, what

    def __enter__(self):
        self.t0 = time.time()
        self.log.info(f"{self.what} ...")
        return self

    def __exit__(self, *exc):
        self.log.info(f"{self.what} done in {_fmt_s(time.time() - self.t0)}")
