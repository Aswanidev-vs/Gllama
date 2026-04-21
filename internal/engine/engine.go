package engine

import (
	"context"
	"fmt"
	"sync"

	"github.com/Aswanidev-vs/Gllama/internal/backend"
)

type Engine struct {
	mu            sync.RWMutex
	activeBackend backend.Backend
	activeModel   string
	models        map[string]string       // model name -> path
	modelConfigs  map[string]*ModelConfig // model name -> config
}

func NewEngine() *Engine {
	return &Engine{
		models:       make(map[string]string),
		modelConfigs: make(map[string]*ModelConfig),
	}
}

func (e *Engine) SetBackend(b backend.Backend) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.activeBackend = b
}

func (e *Engine) RegisterModel(name, path string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.models[name] = path
}

func (e *Engine) RegisterConfig(conf *ModelConfig) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.modelConfigs[conf.Name] = conf
	if conf.Path != "" {
		e.models[conf.Name] = conf.Path
	}
}

func (e *Engine) LoadModel(ctx context.Context, name string) error {
	e.mu.Lock()
	path, ok := e.models[name]
	conf, hasConf := e.modelConfigs[name]
	b := e.activeBackend
	e.mu.Unlock()

	// If no local path but HFRepo is in options (handled by caller), 
	// we just pass it to the backend. But LoadModel specifically 
	// needs to handle the registration check if it's a local model.
	if !ok {
		// For HF models, they might not be registered yet. 
		// We'll let the backend handle the loading logic.
		// Mapping name to path as just the name for HF.
		path = name 
	}

	if b != nil {
		opts := backend.Options{Model: name}
		if hasConf {
			opts = conf.Options
			opts.Model = name
		}
		if err := b.LoadModel(ctx, path, opts); err != nil {
			return err
		}
	}

	e.mu.Lock()
	e.activeModel = name
	e.mu.Unlock()
	fmt.Printf("Model loaded: %s\n", name)
	return nil
}

func (e *Engine) UnloadModel(ctx context.Context) error {
	e.mu.Lock()
	b := e.activeBackend
	e.mu.Unlock()

	if b != nil {
		if err := b.UnloadModel(ctx); err != nil {
			return err
		}
	}

	e.mu.Lock()
	e.activeModel = ""
	e.mu.Unlock()
	fmt.Println("Model unloaded")
	return nil
}

func (e *Engine) Generate(ctx context.Context, opts backend.Options) (*backend.Response, error) {
	e.mu.RLock()
	b := e.activeBackend
	modelName := opts.Model
	if modelName == "" {
		modelName = e.activeModel
	}
	modelPath, ok := e.models[modelName]
	e.mu.RUnlock()

	if b == nil {
		return nil, fmt.Errorf("no backend configured")
	}

	if modelName == "" && opts.HFRepo == "" {
		return nil, fmt.Errorf("no model loaded or specified")
	}

	// Auto-load if not the active model
	if modelName != "" && modelName != e.activeModel && opts.HFRepo == "" {
		if err := e.LoadModel(ctx, modelName); err != nil {
			return nil, err
		}
	}

	if ok {
		opts.Model = modelPath
	} else if opts.HFRepo != "" {
		// pass through HF
	} else if !ok {
		return nil, fmt.Errorf("model %s not registered", modelName)
	}

	return b.Generate(ctx, opts)
}

func (e *Engine) Stream(ctx context.Context, opts backend.Options, cb func(*backend.Response) error) error {
	e.mu.RLock()
	b := e.activeBackend
	modelName := opts.Model
	if modelName == "" {
		modelName = e.activeModel
	}
	modelPath, ok := e.models[modelName]
	e.mu.RUnlock()

	if b == nil {
		return fmt.Errorf("no backend configured")
	}

	if modelName == "" && opts.HFRepo == "" {
		return fmt.Errorf("no model loaded or specified")
	}

	// Auto-load if not the active model
	if modelName != "" && modelName != e.activeModel && opts.HFRepo == "" {
		if err := e.LoadModel(ctx, modelName); err != nil {
			return err
		}
	}

	if ok {
		opts.Model = modelPath
	} else if opts.HFRepo != "" {
		// pass through HF
	} else if !ok {
		return fmt.Errorf("model %s not registered", modelName)
	}

	return b.Stream(ctx, opts, cb)
}

func (e *Engine) Embed(ctx context.Context, opts backend.Options) ([]float32, error) {
	e.mu.RLock()
	b := e.activeBackend
	modelName := opts.Model
	if modelName == "" {
		modelName = e.activeModel
	}
	modelPath, ok := e.models[modelName]
	e.mu.RUnlock()

	if b == nil {
		return nil, fmt.Errorf("no backend configured")
	}

	if modelName == "" && opts.HFRepo == "" {
		return nil, fmt.Errorf("no model loaded or specified")
	}

	if modelName != "" && modelName != e.activeModel && opts.HFRepo == "" {
		if err := e.LoadModel(ctx, modelName); err != nil {
			return nil, err
		}
	}

	if ok {
		opts.Model = modelPath
	}

	return b.Embed(ctx, opts)
}

func (e *Engine) ListModels() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var list []string
	for name := range e.models {
		list = append(list, name)
	}
	return list
}
