"""Command line entry point."""

import sys

import uvicorn

from .api import create_app
from .config import DEFAULT_CONFIG_PATH, load_config

LISTEN_HOST = "127.0.0.1"


def main() -> None:
    """Run the service on the loopback address."""
    # The config path is a fixed argument from the unit definition, never a
    # request value, so it is not attacker-controlled input.
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    uvicorn.run(create_app(config), host=LISTEN_HOST, port=config.listen_port)


if __name__ == "__main__":
    main()