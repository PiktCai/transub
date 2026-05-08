import sys

from transub.server import start_server

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18789
    start_server(host="127.0.0.1", port=port)
