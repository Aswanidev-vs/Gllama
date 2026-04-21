

📄 PRODUCT REQUIREMENTS DOCUMENT (PRD)


---

🧠 1. Product Overview

Name

Gllama

Tagline

> Developer-first Go runtime and bindings for local LLM inference powered by llama.cpp




---

🎯 Objective

Gllama is a modular, developer-friendly runtime + binding layer that:

Runs LLMs locally or via server

Provides a simple API (like Ollama)

Exposes advanced controls (GPU, KV cache, batching)

Enables experimentation (TurboQuant-like optimizations)


---

🧭 Vision

> 🧠 Build the most flexible and controllable Go interface for llama.cpp, combining:



bindings

runtime engine

API layer

performance tuning


---

👥 2. Target Users

Primary

Go developers building AI apps

Backend engineers

Students building local AI systems

Secondary

Researchers experimenting with inference

Indie developers building AI tools


---

🚀 3. Core Features


---

3.1 Engine Core (Core Brain)

Model load / unload

Inference (generate + stream)

Backend abstraction (CGO + HTTP/gRPC)

Central control logic


---

3.2 Go Binding Layer

Direct integration with llama.cpp via CGO

High-performance inference

Low-level control access


---

3.3 API Layer (Ollama-style)

Endpoints

POST /api/generate
POST /api/stream
GET  /api/models
POST /api/models/load
POST /api/models/unload


---

3.4 Backend Support

CGO backend

In-process

Lowest latency

HTTP backend

Local or remote

Easy deployment

gRPC backend (optional)

Structured communication

scalable


---

3.5 Performance Controls

Expose both simple and advanced control:

Simple

{ "mode": "fast" }

Advanced

{
"gpu_layers": 18,
"threads": 8,
"batch_size": 512,
"ctx_size": 4096
}


---

3.6 KV Cache System

FP16 (default)

Q8 / Q4 quantization

TurboQuant-lite (phase 1)

Full TurboQuant (future)


---

3.7 Model Management

Local model registry

Load/unload models dynamically

Track active model


---

3.8 Developer Experience

Clean Go SDK

CLI tool

Config-based tuning

Debug metrics


---

🧠 4. Non-Functional Requirements

Requirement	Goal

Performance	Near-native llama.cpp
Modularity	Plug-and-play components
Extensibility	Easy KV / backend extension
Stability	Safe API + backend isolation
Developer UX	Simple + powerful


---

🏗️ 5. System Architecture

[ CLI / App / SDK ]
                     ↓
                [ Engine ]
                     ↓
         ┌───────────┴───────────┐
         │                       │
   [ CGO Backend ]       [ HTTP/gRPC Backend ]
         │                       │
         └───────────┬───────────┘
                     ↓
           llama.cpp (GPU/CPU)

                ↑
           [ API Layer ]
         (wrapper over Engine)


---

📁 6. Project Structure (Industry Standard)


---

Root Layout

gllama/
├── cmd/                # entrypoints
├── internal/           # core logic
├── pkg/                # public SDK
├── api/                # API layer
├── configs/            # configuration
├── models/             # model metadata
├── scripts/            # build scripts
├── docs/               # documentation
└── go.mod


---

📦 Detailed Structure


---

1. cmd/



cmd/
├── gllama-server/
│   └── main.go
├── gllama-cli/
│   └── main.go


---

2. internal/engine/



internal/engine/
├── engine.go
├── generate.go
├── stream.go
├── model_manager.go
├── options.go
├── config.go


---

3. internal/backend/



internal/backend/
├── backend.go
├── cgo/
│   └── backend.go
├── http/
│   └── backend.go
├── grpc/
│   └── backend.go


---

4. internal/kv/



internal/kv/
├── interface.go
├── fp16.go
├── q8.go
├── q4.go
├── turboquant.go


---

5. internal/model/



internal/model/
├── registry.go
├── loader.go
├── metadata.go


---

6. api/



api/
├── http/
│   ├── server.go
│   ├── routes.go
│   ├── handlers.go
├── grpc/
│   ├── server.go
│   ├── proto/


---

7. pkg/



pkg/
├── client/
│   ├── client.go
│   ├── http.go
│   ├── grpc.go


---

🧩 7. Core Interfaces


---

Backend Interface

type Backend interface {
LoadModel(path string, opts Options) error
Generate(prompt string, opts Options) (string, error)
Stream(prompt string, opts Options, cb func(string)) error
Unload() error
Capabilities() Capabilities
}


---

Engine

type Engine struct {
backend Backend
kv       KVCacheStrategy
}


---

KV Cache

type KVCacheStrategy interface {
Encode([]float32) []byte
Decode([]byte) []float32
}


---

🌐 8. API Design


---

Generate

POST /api/generate


---

Stream

POST /api/stream


---

Models

GET /api/models


---

Load

POST /api/models/load


---

🧠 9. Development Roadmap


---

Phase 1 (MVP)

HTTP API

llama.cpp CLI integration

basic inference


---

Phase 2

streaming

model management

GPU tuning


---

Phase 3

CGO backend

KV cache tuning

Add GPU tuning + KV system

---

Phase 4

TurboQuant-lite


---

Phase 5

Full TurboQuant

Custom kernels


---

🔥 10. Design Principles


---

1. Engine-first architecture


2. API as thin wrapper


3. Backend abstraction


4. Developer-first design


5. Performance-focused




---

🏁 Final Summary

Gllama is:

> 🧠 A Go-first binding + runtime system for llama.cpp that combines simplicity, control, and performance

