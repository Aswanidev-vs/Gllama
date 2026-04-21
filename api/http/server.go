package http

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/Aswanidev-vs/Gllama/internal/engine"
)

type Server struct {
	addr    string
	engine  *engine.Engine
	handler *Handler
	server  *http.Server
}

func NewServer(port int, e *engine.Engine) *Server {
	h := NewHandler(e)
	addr := fmt.Sprintf(":%d", port)

	mux := http.NewServeMux()
	s := &Server{
		addr:    addr,
		engine:  e,
		handler: h,
	}

	s.registerRoutes(mux)

	s.server = &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  5 * time.Minute,
		WriteTimeout: 5 * time.Minute,
	}

	return s
}

func (s *Server) Start() error {
	fmt.Printf("Gllama server starting on %s\n", s.addr)
	return s.server.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.server.Shutdown(ctx)
}

func (s *Server) registerRoutes(mux *http.ServeMux) {
	// Gllama API
	mux.HandleFunc("/api/generate", s.handler.HandleGenerate)
	mux.HandleFunc("/api/models", s.handler.HandleListModels)
	mux.HandleFunc("/api/models/load", s.handler.HandleLoadModel)
	mux.HandleFunc("/api/models/unload", s.handler.HandleUnloadModel)

	// OpenAI API
	mux.HandleFunc("/v1/chat/completions", s.handler.HandleOpenAIChat)
	mux.HandleFunc("/v1/embeddings", s.handler.HandleOpenAIEmbed)

	// Health check
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})
}
