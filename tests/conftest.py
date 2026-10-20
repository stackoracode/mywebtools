import os
import tempfile
from pathlib import Path

# Isolate every test run: data dir, .env file, no module-level app.
_tmp = Path(tempfile.mkdtemp(prefix="mwt-test-"))
os.environ["MWT_DATA_DIR"] = str(_tmp / "data")
os.environ["MWT_ENV_FILE"] = str(_tmp / ".env")
os.environ["MWT_NO_APP"] = "1"
os.environ.pop("APP_PASSWORD", None)
os.environ.pop("DUIX_API_KEY", None)

import pytest  # noqa: E402

from mywebtools.config import settings  # noqa: E402

settings.ensure_dirs()


@pytest.fixture
def data_dir() -> Path:
    return settings.data_dir
