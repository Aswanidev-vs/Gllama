# Gllama 

A high-performance LLM inference engine and server, powered by `llama.cpp` with native Go/CGO bindings.

## 🚀 Features

- **Blazing Fast CGO Backend**: Direct integration with `llama.cpp` for near-native inference speeds.
- **OpenAI Compatible API**: Drop-in replacements for Chat Completions and Embeddings.
- **Advanced Sampling**: Support for Top-K, Top-P (Nucleus), Min-P, and Temperature.
- **Semantic Embeddings**: Extract high-dimensional vectors for RAG and search.
- **Real-time Telemetry**: Monitoring of TTFT (Time To First Token) and TPS (Tokens Per Second).
- **Resource Safe**: Immediate inference cancellation upon client disconnect.
- **CLI Fallback**: Flexible backend architecture supporting both integrated CGO and external CLI calls.

## 🛠️ Getting Started

### Prerequisites
- Go 1.21+
- GCC / MinGW (for Windows CGO)
- `llama.cpp` static libraries

### Building
```powershell
go build -v -o gllama-server.exe ./cmd/gllama-server
```

### Running the Server
```powershell
./gllama-server.exe -backend cgo
```

## 📡 API Endpoints

### Chat Completions
**POST** `/v1/chat/completions`

```json
{
  "model": "llama-3",
  "messages": [{"role": "user", "content": "Tell me a joke."}],
  "temperature": 0.7,
  "top_p": 0.9,
  "min_p": 0.05
}
```

### Embeddings
**POST** `/v1/embeddings`

```json
{
  "input": "The future of AI is agentic.",
  "model": "nomic-embed"
}
```

## 📊 Performance Monitoring
Gllama includes built-in telemetry appended to responses:
`[Telemetry: 124 tokens, 45.2 t/s, TTFT: 120ms]`

## 📜 License
MIT
