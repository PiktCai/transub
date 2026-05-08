import sys
import os
import types

if getattr(sys, '_MEIPASS', None):
    sys.path.insert(0, sys._MEIPASS)
    stub = types.ModuleType("transub")
    stub.__path__ = [os.path.join(sys._MEIPASS, "transub")]
    stub.__package__ = "transub"
    sys.modules["transub"] = stub

from transub.server import start_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18789
    start_server(host="127.0.0.1", port=port)
