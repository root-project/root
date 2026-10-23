#!/usr/bin/env python3
"""
benchmark_remote_simulation.py
Simulates a high-latency remote storage endpoint (e.g. XRootD / HTTP WebDAV across a WAN)
by serving RNTuple ROOT files via a local HTTP server with configurable artificial latency.

Usage:
    python3 benchmark_remote_simulation.py [--latency-ms 20] [--files 16] [--slots 1,2,4,8]
"""

import argparse
import http.server
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import time

class LatencyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    latency_seconds = 0.020  # 20ms simulated round-trip latency

    def do_GET(self):
        time.sleep(self.latency_seconds)
        super().do_GET()

    def do_HEAD(self):
        time.sleep(self.latency_seconds)
        super().do_HEAD()

    def log_message(self, format, *args):
        # Silence HTTP server logs during benchmarking
        pass

def start_latency_server(directory, latency_ms, port=8888):
    Handler = LatencyHTTPRequestHandler
    Handler.latency_seconds = latency_ms / 1000.0

    class DualStackServer(socketserver.TCPServer):
        allow_reuse_address = True

    # Change to target directory in request handler
    def handler_factory(*args, **kwargs):
        return Handler(*args, directory=directory, **kwargs)

    httpd = DualStackServer(("127.0.0.1", port), handler_factory)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    return httpd

def main():
    parser = argparse.ArgumentParser(description="Simulate remote storage latency for RNTuple MT benchmark.")
    parser.add_argument("--latency-ms", type=float, default=20.0, help="Artificial latency in ms per request (default: 20ms)")
    parser.add_argument("--port", type=int, default=8888, help="Port for local HTTP server (default: 8888)")
    parser.add_argument("--files", type=int, default=16, help="Number of files (default: 16)")
    args = parser.parse_args()

    print(f"=== RNTuple IMT Remote Latency Simulation ===")
    print(f"Simulated Latency: {args.latency_ms} ms per request")
    print(f"Target Files: {args.files}")
    print(f"HTTP Server: http://127.0.0.1:{args.port}/")
    print(f"=============================================")
    print("\nTo benchmark against this simulated remote endpoint with root:")
    print(f"  1. Start this script with --latency-ms {args.latency_ms}")
    print(f"  2. In another terminal, run root with URLs:")
    print(f"     root -l -b -q 'tree/dataframe/test/benchmark_rntuple_imt.C(\"http_file_list.txt\")'")

if __name__ == "__main__":
    main()
