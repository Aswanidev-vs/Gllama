package http

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/Aswanidev-vs/Gllama/internal/backend"
	"github.com/Aswanidev-vs/Gllama/internal/engine"
)

type Handler struct {
	Engine *engine.Engine
}

func NewHandler(e *engine.Engine) *Handler {
	return &Handler{Engine: e}
}

// Gllama Generate Handler
func (h *Handler) HandleGenerate(w http.ResponseWriter, r *http.Request) {
	var opts backend.Options
	if err := json.NewDecoder(r.Body).Decode(&opts); err != nil {
		h.error(w, "invalid request", http.StatusBadRequest)
		return
	}

	if opts.Stream {
		h.streamGenerate(w, r, opts)
		return
	}

	resp, err := h.Engine.Generate(r.Context(), opts)
	if err != nil {
		h.error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(resp)
}

func (h *Handler) streamGenerate(w http.ResponseWriter, r *http.Request, opts backend.Options) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		h.error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	err := h.Engine.Stream(r.Context(), opts, func(resp *backend.Response) error {
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	})

	if err != nil {
		json.NewEncoder(w).Encode(backend.Response{
			Done:    true,
			Content: fmt.Sprintf("\nerror: %s\n", err.Error()),
		})
	}
}

// OpenAI-Compatible Handlers
type ChatCompletionRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Stream      bool    `json:"stream"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	TopK        int     `json:"top_k"`
	MinP        float64 `json:"min_p"`
	MaxTokens   int     `json:"max_tokens"`
	Seed        int     `json:"seed"`
}

type ChatCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

func (h *Handler) HandleOpenAIChat(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.error(w, "invalid request", http.StatusBadRequest)
		return
	}

	// Map OpenAI request to Gllama options
	prompt := ""
	for _, m := range req.Messages {
		prompt += fmt.Sprintf("%s: %s\n", m.Role, m.Content)
	}
	prompt += "assistant: "

	opts := backend.Options{
		Model:       req.Model,
		Prompt:      prompt,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		MinP:        req.MinP,
		MaxTokens:   req.MaxTokens,
		Seed:        req.Seed,
	}

	if req.Stream {
		h.streamOpenAIChat(w, r, opts)
		return
	}

	resp, err := h.Engine.Generate(r.Context(), opts)
	if err != nil {
		h.error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	oaResp := ChatCompletionResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: resp.Created,
		Model:   resp.Model,
	}
	oaResp.Choices = append(oaResp.Choices, struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	}{
		Message: struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{Role: "assistant", Content: resp.Content},
		FinishReason: "stop",
	})

	json.NewEncoder(w).Encode(oaResp)
}

func (h *Handler) streamOpenAIChat(w http.ResponseWriter, r *http.Request, opts backend.Options) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		h.error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	err := h.Engine.Stream(r.Context(), opts, func(resp *backend.Response) error {
		if resp.Done {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return nil
		}

		chunk := map[string]interface{}{
			"id":      resp.ID,
			"object":  "chat.completion.chunk",
			"created": resp.Created,
			"model":   resp.Model,
			"choices": []map[string]interface{}{
				{
					"delta": map[string]string{
						"content": resp.Content,
					},
					"finish_reason": nil,
					"index":         0,
				},
			},
		}

		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", string(data))
		flusher.Flush()
		return nil
	})

	if err != nil {
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
	}
}

type EmbeddingRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"` // string or []string
}

func (h *Handler) HandleOpenAIEmbed(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.error(w, "invalid request", http.StatusBadRequest)
		return
	}

	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = append(inputs, v)
	case []interface{}:
		for _, x := range v {
			if s, ok := x.(string); ok {
				inputs = append(inputs, s)
			}
		}
	}

	if len(inputs) == 0 {
		h.error(w, "input is required", http.StatusBadRequest)
		return
	}

	var dataset []backend.EmbeddingData
	for i, input := range inputs {
		vec, err := h.Engine.Embed(r.Context(), backend.Options{
			Model:  req.Model,
			Prompt: input,
		})
		if err != nil {
			h.error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		dataset = append(dataset, backend.EmbeddingData{
			Object:    "embedding",
			Embedding: vec,
			Index:     i,
		})
	}

	resp := backend.EmbeddingResponse{
		Object: "list",
		Data:   dataset,
		Model:  req.Model,
		Usage: backend.Usage{
			PromptTokens: 0, // token counting not implemented yet
			TotalTokens:  0,
		},
	}

	json.NewEncoder(w).Encode(resp)
}

func (h *Handler) HandleListModels(w http.ResponseWriter, r *http.Request) {
	models := h.Engine.ListModels()
	json.NewEncoder(w).Encode(map[string]interface{}{
		"models": models,
	})
}

func (h *Handler) HandleLoadModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.error(w, "invalid request", http.StatusBadRequest)
		return
	}

	if err := h.Engine.LoadModel(r.Context(), req.Name); err != nil {
		h.error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "model loaded"})
}

func (h *Handler) HandleUnloadModel(w http.ResponseWriter, r *http.Request) {
	if err := h.Engine.UnloadModel(r.Context()); err != nil {
		h.error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "success", "message": "model unloaded"})
}

func (h *Handler) error(w http.ResponseWriter, msg string, code int) {
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
