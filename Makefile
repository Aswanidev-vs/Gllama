.PHONY: all build server cli quick run clean install root

BIN_DIR := bin
SERVER_TARGET := $(BIN_DIR)/gllama-server
CLI_TARGET := $(BIN_DIR)/gllama
WINDOWS_SERVER_TARGET := $(BIN_DIR)/gllama-server.exe
WINDOWS_CLI_TARGET := $(BIN_DIR)/gllama.exe

all: server cli

build: all

quick: cli

server:
	go build -o $(SERVER_TARGET) ./cmd/gllama-server
	go build -o $(WINDOWS_SERVER_TARGET) ./cmd/gllama-server

cli:
	go build -o $(CLI_TARGET) ./cmd/gllama
	go build -o $(WINDOWS_CLI_TARGET) ./cmd/gllama

root:
	go build -o gllama.exe ./cmd/gllama

install:
	go install ./cmd/gllama
	go install ./cmd/gllama-server

run: cli
	./$(CLI_TARGET)

clean:
	if exist bin rmdir /s /q bin
	if exist gllama.exe del gllama.exe

dev:
	powershell -ExecutionPolicy Bypass -File watch.ps1
