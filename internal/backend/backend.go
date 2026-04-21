package backend

import (
	"context"
)

// Options represents common inference options
type Options struct {
	Model         string                 `json:"model"`
	Prompt        string                 `json:"prompt"`
	System        string                 `json:"system,omitempty"`
	Template      string                 `json:"template,omitempty"`
	Stream        bool                   `json:"stream,omitempty"`
	MaxTokens     int                    `json:"max_tokens,omitempty"`
	Temperature   float64                `json:"temperature,omitempty"`
	TopP          float64                `json:"top_p,omitempty"`
	TopK          int                    `json:"top_k,omitempty"`
	MinP          float64                `json:"min_p,omitempty"`
	Stop          []string               `json:"stop,omitempty"`
	Seed          int                    `json:"seed,omitempty"`
	GPU           bool                   `json:"gpu,omitempty"`
	GPULayers     int                    `json:"gpu_layers,omitempty"`
	Threads       int                    `json:"threads,omitempty"`
	ContextSize   int                    `json:"context_size,omitempty"`
	BatchSize     int                    `json:"batch_size,omitempty"`
	KVCacheType   string                 `json:"kv_cache_type,omitempty"`   // legacy shortcut for both K and V
	KVCacheTypeK  string                 `json:"kv_cache_type_k,omitempty"` // f16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl
	KVCacheTypeV  string                 `json:"kv_cache_type_v,omitempty"` // f16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl
	FlashAttention string                `json:"flash_attention,omitempty"` // auto, on, off
	KVOffload     *bool                  `json:"kv_offload,omitempty"`
	SWAFull       *bool                  `json:"swa_full,omitempty"`
	KVUnified     *bool                  `json:"kv_unified,omitempty"`
	TurboQuant    string                 `json:"turboquant,omitempty"` // off, lite, q8, q4
	HFRepo        string                 `json:"hf_repo,omitempty"` // Hugging Face repository
	HFFile        string                 `json:"hf_file,omitempty"` // Hugging Face file
	ExtraArgs     map[string]interface{} `json:"extra_args,omitempty"`
}

// NormalizeOptions applies Gllama-side performance presets before execution.
// This is intentionally conservative: explicit user values always win.
func NormalizeOptions(opts Options) Options {
	mode := opts.TurboQuant
	if mode == "" || mode == "off" {
		return opts
	}

	if opts.FlashAttention == "" {
		opts.FlashAttention = "on"
	}
	if opts.KVOffload == nil {
		enabled := true
		opts.KVOffload = &enabled
	}
	if opts.BatchSize == 0 {
		opts.BatchSize = 1024
	}

	switch mode {
	case "lite":
		if opts.KVCacheTypeK == "" && opts.KVCacheType == "" {
			opts.KVCacheTypeK = "q8_0"
		}
		if opts.KVCacheTypeV == "" && opts.KVCacheType == "" {
			opts.KVCacheTypeV = "f16"
		}
	case "q8":
		if opts.KVCacheTypeK == "" && opts.KVCacheType == "" {
			opts.KVCacheTypeK = "q8_0"
		}
		if opts.KVCacheTypeV == "" && opts.KVCacheType == "" {
			opts.KVCacheTypeV = "q8_0"
		}
	case "q4":
		if opts.KVCacheTypeK == "" && opts.KVCacheType == "" {
			opts.KVCacheTypeK = "q4_0"
		}
		if opts.KVCacheTypeV == "" && opts.KVCacheType == "" {
			opts.KVCacheTypeV = "q4_0"
		}
	}

	return opts
}

// Response represents a completion response
type Response struct {
	ID      string `json:"id,omitempty"`
	Model   string `json:"model,omitempty"`
	Created int64  `json:"created,omitempty"`
	Content string `json:"content,omitempty"`
	Done    bool   `json:"done,omitempty"`
	
	// Progress fields
	Status     string  `json:"status,omitempty"`     // e.g. "Downloading", "Generating"
	Percentage float64 `json:"percentage,omitempty"` // 0-100
	Speed      string  `json:"speed,omitempty"`      // e.g. "1.2 MB/s"
	TotalSize  string  `json:"total_size,omitempty"` // e.g. "3.2 GB"

	// Telemetry fields
	TPS           float64 `json:"tps,omitempty"`            // Tokens per second
	TTFT          float64 `json:"ttft_ms,omitempty"`        // Time to first token (ms)
	TokenCount    int     `json:"token_count,omitempty"`    // Total tokens generated
	TotalDuration float64 `json:"total_duration_ms,omitempty"` // Total generation time (ms)
}

// Capabilities defines what the backend supports
type Capabilities struct {
	SupportsStreaming bool
	SupportsGPU       bool
	SupportsBatching  bool
}

// Backend defines the interface for LLM inference engines
type Backend interface {
	// LoadModel loads a model into memory
	LoadModel(ctx context.Context, path string, opts Options) error
	
	// UnloadModel unloads the current model
	UnloadModel(ctx context.Context) error
	
	// Generate performs a single completion
	Generate(ctx context.Context, opts Options) (*Response, error)
	
	// Stream performs a streaming completion, calling callback for each fragment
	Stream(ctx context.Context, opts Options, cb func(*Response) error) error

	// Embed generates a vector embedding for the input text
	Embed(ctx context.Context, opts Options) ([]float32, error)
	
	// Capabilities returns the backend's supported features
	Capabilities() Capabilities
}

// EmbeddingResponse represents an embedding result
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}

type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type Usage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
