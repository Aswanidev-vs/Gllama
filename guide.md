# Gllama Guide

This guide shows the current Gllama CLI flow and the main commands you can run from the repository root on Windows PowerShell.

## Build

Build the CLI:

```powershell
go build -o .\bin\gllama.exe .\cmd\gllama
```

Build the server:

```powershell
go build -o .\bin\gllama-server.exe .\cmd\gllama-server
```

If Go tries to write outside the workspace and fails on cache updates, the Gllama binaries may still be produced successfully. The confirmed CLI binary path is:

```text
G:\Gllama\bin\gllama.exe
```

## Setup

Download required `llama.cpp` runtime binaries:

```powershell
.\bin\gllama.exe setup
```

## Pull A Model

Download a model from Hugging Face into the local `models/` directory:

```powershell
.\bin\gllama.exe pull unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
```

You can also specify the Hugging Face repo with `--hf`:

```powershell
.\bin\gllama.exe pull --hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
```

## Run A Model

Run directly from a Hugging Face repo reference:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello"
```

Run a local GGUF file:

```powershell
.\bin\gllama.exe run .\models\gemma-4-E2B-it-Q4_K_M.gguf "Explain Go interfaces"
```

Run in interactive mode:

```powershell
.\bin\gllama.exe run .\models\gemma-4-E2B-it-Q4_K_M.gguf -i
```

## Serve

Start the HTTP server with the CLI backend:

```powershell
.\bin\gllama.exe serve --backend cli
```

Start the HTTP server with the CGO backend:

```powershell
.\bin\gllama.exe serve --backend cgo
```

You can also set a custom port:

```powershell
.\bin\gllama.exe serve --backend cli --port 11432
```

## List Models

List registered or local models:

```powershell
.\bin\gllama.exe list
```

## Common Runtime Flags

These options work with `run` and `pull` where relevant.

Basic tuning:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" -n 256 -t 8 -c 4096 -b 512
```

GPU layers:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --gpu-layers 99
```

Flash Attention:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --flash-attn on
```

## KV Cache Optimization Flags

Set a single cache type for both K and V:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --kv-cache-type q8_0
```

Set K and V cache types separately:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --kv-cache-type-k q8_0 --kv-cache-type-v f16
```

Enable KV offload:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --kv-offload
```

Disable KV offload:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --no-kv-offload
```

Enable unified KV:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --kv-unified
```

Use full SWA cache:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --swa-full
```

## TurboQuant Profiles

Gllama currently exposes TurboQuant-style presets as performance profiles:

- `lite`
- `q8`
- `q4`

Example:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --turboquant lite
```

What they currently do:

- `lite`: enables Flash Attention, enables KV offload, raises default batch size, biases KV cache toward `K=q8_0` and `V=f16`
- `q8`: same tuning direction but biases both K and V to `q8_0`
- `q4`: same tuning direction but biases both K and V to `q4_0`

Explicit user flags still win over the preset.

Example with explicit overrides:

```powershell
.\bin\gllama.exe run unsloth/gemma-4-E2B-it-GGUF:Q4_K_M "Hello" --turboquant lite --kv-cache-type-k q4_0 --kv-cache-type-v q8_0 --flash-attn on
```

## Notes

- `pull` and direct `run` still use `llama.cpp` underneath for actual downloads and inference execution in the CLI path.
- The CGO backend exists and can be used through `serve --backend cgo`.
- The current TurboQuant support is a Gllama-side tuning preset, not a full custom-kernel TurboQuant implementation.
- The HTTP API is available through `gllama serve`.

