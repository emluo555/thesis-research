"""
serve.py - Simple HTTP server for the Data Explorer.

Serves the DataExplorer/ directory on localhost with correct MIME types.

Usage:
    python serve.py [--port 8080]
"""

import argparse
import http.server
import os
import socketserver

PORT = 35365


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".json": "application/json",
        ".js": "application/javascript",
        ".css": "text/css",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description="Data Explorer HTTP server")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        print(f"\n  Data Explorer running at: http://localhost:{args.port}/\n")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
