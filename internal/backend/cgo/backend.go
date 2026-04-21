package cgo

/*
#cgo CFLAGS: -I../../../llama.cpp/include -I../../../llama.cpp/ggml/include
#cgo LDFLAGS: -L../../../llama.cpp/build/src -L../../../llama.cpp/build/ggml/src -lllama -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a -lgomp -lstdc++

#include <stdlib.h>
#include <string.h>
#include "llama.h"

// Helper function to tokenize
static int tokenize_helper(const struct llama_vocab * vocab, const char * text, llama_token * tokens, int n_max_tokens, bool add_special) {
    return llama_tokenize(vocab, text, strlen(text), tokens, n_max_tokens, add_special, true);
}

// Helper to add token to batch
static void batch_add(struct llama_batch * batch, llama_token id, llama_pos pos, const llama_seq_id * seq_ids, int n_seq_ids, bool logits) {
    int i = batch->n_tokens;
    batch->token[i] = id;
    batch->pos[i]   = pos;
    batch->n_seq_id[i] = n_seq_ids;
    for (int j = 0; j < n_seq_ids; j++) {
        batch->seq_id[i][j] = seq_ids[j];
    }
    batch->logits[i] = logits ? 1 : 0;
    batch->n_tokens++;
}

// Build a sampling chain based on options
static struct llama_sampler * build_sampler(int top_k, float top_p, float min_p, float temp, uint32_t seed) {
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(sparams);

    if (temp > 0.0f) {
        if (top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
        if (top_p < 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0.0f) llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }

    return chain;
}
*/
import "C"
import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/Aswanidev-vs/Gllama/internal/backend"
)

type CGOBackend struct {
	mu    sync.Mutex
	model *C.struct_llama_model
	ctx   *C.struct_llama_context
	
	// Telemetry
	startTime time.Time
	firstTokenTime time.Duration
	tokenCount int
}

func NewCGOBackend() *CGOBackend {
	C.llama_backend_init()
	return &CGOBackend{}
}

func (b *CGOBackend) LoadModel(ctx context.Context, path string, opts backend.Options) error {
	opts = backend.NormalizeOptions(opts)

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.model != nil {
		b.unloadModel()
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	mparams := C.llama_model_default_params()
	mparams.n_gpu_layers = C.int32_t(opts.GPULayers)

	b.model = C.llama_model_load_from_file(cPath, mparams)
	if b.model == nil {
		return fmt.Errorf("failed to load model from %s", path)
	}

	cparams := C.llama_context_default_params()
	cparams.n_ctx = C.uint32_t(opts.ContextSize)
	if cparams.n_ctx == 0 {
		cparams.n_ctx = 2048 // default
	}
	cparams.n_batch = C.uint32_t(opts.BatchSize)
	if cparams.n_batch == 0 {
		cparams.n_batch = 512 // default
	}
	cparams.n_threads = C.int32_t(opts.Threads)
	cparams.n_threads_batch = C.int32_t(opts.Threads)
	cparams.offload_kqv = true

	switch strings.ToLower(opts.FlashAttention) {
	case "on":
		cparams.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_ENABLED
	case "off":
		cparams.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_DISABLED
	}

	if opts.KVOffload != nil {
		cparams.offload_kqv = C.bool(*opts.KVOffload)
	}

	if opts.SWAFull != nil {
		cparams.swa_full = C.bool(*opts.SWAFull)
	}

	if opts.KVUnified != nil {
		cparams.kv_unified = C.bool(*opts.KVUnified)
	}

	// KV Cache Types
	typeK, typeV := kvCacheTypes(opts)
	if ggmlType, ok := ggmlTypeForName(typeK); ok {
		cparams.type_k = ggmlType
	}
	if ggmlType, ok := ggmlTypeForName(typeV); ok {
		cparams.type_v = ggmlType
	}

	b.ctx = C.llama_init_from_model(b.model, cparams)
	if b.ctx == nil {
		C.llama_model_free(b.model)
		b.model = nil
		return fmt.Errorf("failed to create context")
	}

	// Enable embeddings if not already enabled (though params should handle it)
	C.llama_set_embeddings(b.ctx, true)

	return nil
}

func kvCacheTypes(opts backend.Options) (string, string) {
	typeK := opts.KVCacheTypeK
	typeV := opts.KVCacheTypeV

	if opts.KVCacheType != "" {
		if typeK == "" {
			typeK = opts.KVCacheType
		}
		if typeV == "" {
			typeV = opts.KVCacheType
		}
	}

	return strings.ToLower(typeK), strings.ToLower(typeV)
}

func ggmlTypeForName(name string) (C.enum_ggml_type, bool) {
	switch strings.ToLower(name) {
	case "", "f16", "fp16":
		return C.GGML_TYPE_F16, true
	case "f32":
		return C.GGML_TYPE_F32, true
	case "bf16":
		return C.GGML_TYPE_BF16, true
	case "q8_0":
		return C.GGML_TYPE_Q8_0, true
	case "q4_0":
		return C.GGML_TYPE_Q4_0, true
	case "q4_1":
		return C.GGML_TYPE_Q4_1, true
	case "q5_0":
		return C.GGML_TYPE_Q5_0, true
	case "q5_1":
		return C.GGML_TYPE_Q5_1, true
	case "iq4_nl":
		return C.GGML_TYPE_IQ4_NL, true
	default:
		return C.GGML_TYPE_F16, false
	}
}

func (b *CGOBackend) UnloadModel(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.unloadModel()
	return nil
}

func (b *CGOBackend) unloadModel() {
	if b.ctx != nil {
		C.llama_free(b.ctx)
		b.ctx = nil
	}
	if b.model != nil {
		C.llama_model_free(b.model)
		b.model = nil
	}
}

func (b *CGOBackend) Generate(ctx context.Context, opts backend.Options) (*backend.Response, error) {
	var result strings.Builder

	err := b.Stream(ctx, opts, func(resp *backend.Response) error {
		result.WriteString(resp.Content)
		return nil
	})

	if err != nil {
		return nil, err
	}

	return &backend.Response{
		ID:      fmt.Sprintf("gen-%d", time.Now().Unix()),
		Model:   opts.Model,
		Created: time.Now().Unix(),
		Content: result.String(),
		Done:    true,
	}, nil
}

func (b *CGOBackend) Stream(ctx context.Context, opts backend.Options, cb func(*backend.Response) error) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.ctx == nil {
		return fmt.Errorf("no model loaded")
	}

	b.startTime = time.Now()
	b.tokenCount = 0
	b.firstTokenTime = 0

	tokens, err := b.tokenize(opts.Prompt, true)
	if err != nil {
		return err
	}

	// KV Cache clear/init
	C.llama_memory_clear(C.llama_get_memory(b.ctx), true)

	// Decode prompt
	n_past := C.int(0)
	batch := C.llama_batch_init(C.int(len(tokens)), 0, 1)
	defer C.llama_batch_free(batch)

	seq_id := C.llama_seq_id(0)
	for i, token := range tokens {
		C.batch_add(&batch, token, C.llama_pos(i), &seq_id, 1, i == len(tokens)-1)
	}

	if C.llama_decode(b.ctx, batch) != 0 {
		return fmt.Errorf("initial decode failed")
	}
	n_past += C.int(len(tokens))

	vocab := C.llama_model_get_vocab(b.model)

	// Build sampler
	seed := uint32(opts.Seed)
	if seed == 0 {
		seed = uint32(time.Now().UnixNano())
	}
	smpl := C.build_sampler(C.int(opts.TopK), C.float(opts.TopP), C.float(opts.MinP), C.float(opts.Temperature), C.uint32_t(seed))
	defer C.llama_sampler_free(smpl)

	// Generation loop
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 128
	}

	for i := 0; i < maxTokens; i++ {
		// Check for context cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Sample the next token using the sampler chain
		nextToken := C.llama_sampler_sample(smpl, b.ctx, -1)
		
		// Record TTFT
		if b.tokenCount == 0 {
			b.firstTokenTime = time.Since(b.startTime)
		}
		b.tokenCount++

		if C.llama_vocab_is_eog(vocab, nextToken) {
			break
		}

		// Convert token to piece
		piece := b.tokenToPiece(vocab, nextToken)
		
		if err := cb(&backend.Response{Content: piece}); err != nil {
			return err
		}

		// Prepare next batch
		batch.n_tokens = 0
		C.batch_add(&batch, nextToken, C.llama_pos(n_past), &seq_id, 1, true)
		
		if C.llama_decode(b.ctx, batch) != 0 {
			return fmt.Errorf("decoding failed")
		}
		n_past++
		
		// Accept the token in the sampler
		C.llama_sampler_accept(smpl, nextToken)
	}

	// Final telemetry in the last response
	duration := time.Since(b.startTime)
	tps := float64(b.tokenCount) / duration.Seconds()
	
	cb(&backend.Response{
		Done: true,
		Content: fmt.Sprintf("\n\n[Telemetry: %d tokens, %.2f t/s, TTFT: %v]", b.tokenCount, tps, b.firstTokenTime),
	})
	return nil
}

func (b *CGOBackend) Embed(ctx context.Context, opts backend.Options) ([]float32, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.ctx == nil {
		return nil, fmt.Errorf("no model loaded")
	}

	tokens, err := b.tokenize(opts.Prompt, true)
	if err != nil {
		return nil, err
	}

	// KV Cache clear
	C.llama_memory_clear(C.llama_get_memory(b.ctx), true)

	// Batch for embeddings
	batch := C.llama_batch_init(C.int(len(tokens)), 0, 1)
	defer C.llama_batch_free(batch)

	seq_id := C.llama_seq_id(0)
	for i, token := range tokens {
		// For embeddings, we typically want output for the last token or all tokens if pooling is used
		// By default, llama.cpp handles pooling based on llama_context_params.
		// Setting logits=true for tokens that should contribute to the output.
		C.batch_add(&batch, token, C.llama_pos(i), &seq_id, 1, true)
	}

	if C.llama_decode(b.ctx, batch) != 0 {
		return nil, fmt.Errorf("embedding decode failed")
	}

	n_embd := int(C.llama_model_n_embd(b.model))
	
	// Get the embedding from the last token (index -1)
	embdPtr := C.llama_get_embeddings_ith(b.ctx, -1)
	if embdPtr == nil {
		return nil, fmt.Errorf("failed to get embeddings")
	}

	// Copy to Go slice
	result := make([]float32, n_embd)
	for i := 0; i < n_embd; i++ {
		result[i] = float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(embdPtr)) + uintptr(i)*4)))
	}

	return result, nil
}


func (b *CGOBackend) tokenToPiece(vocab *C.struct_llama_vocab, token C.llama_token) string {
	buf := make([]byte, 128)
	n := C.llama_token_to_piece(vocab, token, (*C.char)(unsafe.Pointer(&buf[0])), C.int(len(buf)), 0, true)
	if n < 0 {
		return ""
	}
	return string(buf[:n])
}

func (b *CGOBackend) Capabilities() backend.Capabilities {
	return backend.Capabilities{
		SupportsStreaming: true,
		SupportsGPU:       true,
		SupportsBatching:  true,
	}
}

func (b *CGOBackend) tokenize(text string, addSpecial bool) ([]C.llama_token, error) {
	vocab := C.llama_model_get_vocab(b.model)
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	nTokens := len(text) + 2
	tokens := make([]C.llama_token, nTokens)
	
	res := C.tokenize_helper(vocab, cText, (*C.llama_token)(unsafe.Pointer(&tokens[0])), C.int(nTokens), C.bool(addSpecial))
	if res < 0 {
		return nil, fmt.Errorf("tokenization failed")
	}
	return tokens[:res], nil
}
