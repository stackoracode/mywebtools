"""Entry point: ``python -m mywebtools`` (single process, single event loop)."""

from __future__ import annotations

import logging


def main() -> None:
    import uvicorn

    from .config import settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # One worker on purpose: the job queue lives in this process's memory.
    uvicorn.run("mywebtools.app:app", host=settings.host, port=settings.port, workers=1, log_level="info",
                proxy_headers=True, timeout_keep_alive=30)


if __name__ == "__main__":
    main()
