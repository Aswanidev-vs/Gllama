#!/bin/bash
set -e

BIN_DIR="bin"

if [ ! -d "$BIN_DIR" ]; then
    mkdir -p "$BIN_DIR"
fi

echo "Building Gllama Server..."
go build -o "$BIN_DIR/gllama-server" ./cmd/gllama-server

echo "Building Gllama CLI..."
go build -o "$BIN_DIR/gllama" ./cmd/gllama

echo "Done! Binaries are in the '$BIN_DIR' folder."
