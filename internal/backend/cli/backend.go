package cli

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/Aswanidev-vs/Gllama/internal/backend"
	"github.com/Aswanidev-vs/Gllama/internal/wrapper"
)

type CLIBackend struct {
	BinaryPath string
	CurrentCmd *exec.Cmd
}

func NewCLIBackend(binaryPath string) *CLIBackend {
	return &CLIBackend{
		BinaryPath: binaryPath,
	}
}

func (b *CLIBackend) LoadModel(ctx context.Context, path string, opts backend.Options) error {
	// For CLI backend, we don't necessarily keep a persistent process
	// unless we use llama-server. But for Phase 1, we will verify the path.
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("model file not found: %w", err)
	}
	return nil
}

func (b *CLIBackend) UnloadModel(ctx context.Context) error {
	if b.CurrentCmd != nil && b.CurrentCmd.Process != nil {
		return b.CurrentCmd.Process.Kill()
	}
	return nil
}

func (b *CLIBackend) Generate(ctx context.Context, opts backend.Options) (*backend.Response, error) {
	args := b.buildArgs(opts)
	cmd := exec.CommandContext(ctx, b.BinaryPath, args...)
	cmd.Env = modelsEnv()

	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("error running llama-cli: %w\nOutput: %s", err, string(output))
	}

	result := string(output)
	// llama-cli output parsing logic would go here.
	// For now, we return the raw output or a simplified version.
	// Note: llama-cli often prints logs to stderr and text to stdout.

	return &backend.Response{
		ID:      fmt.Sprintf("gen-%d", time.Now().Unix()),
		Model:   opts.Model,
		Created: time.Now().Unix(),
		Content: result,
		Done:    true,
	}, nil
}

func (b *CLIBackend) Stream(ctx context.Context, opts backend.Options, cb func(*backend.Response) error) error {
	args := b.buildArgs(opts)
	// Add flags for streaming if llama-cli supports them or just pipe stdout
	cmd := exec.CommandContext(ctx, b.BinaryPath, args...)
	cmd.Env = modelsEnv()

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}

	if err := cmd.Start(); err != nil {
		return err
	}

	// Goroutine to handle stderr (progress parsing)
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			percent, speed, ok := wrapper.ParseProgress(line)
			if ok {
				cb(&backend.Response{
					Status:     "Downloading",
					Percentage: percent,
					Speed:      speed,
				})
			}
		}
	}()

	startTime := time.Now()
	var firstTokenTime time.Time
	tokenCount := 0

	reader := bufio.NewReader(stdout)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		tokenCount++
		if firstTokenTime.IsZero() {
			firstTokenTime = time.Now()
		}

		resp := &backend.Response{
			ID:      fmt.Sprintf("gen-%d", time.Now().Unix()),
			Model:   opts.Model,
			Created: time.Now().Unix(),
			Content: line,
			Status:  "Generating",
			Done:    false,
		}

		if err := cb(resp); err != nil {
			return err
		}
	}

	if err := cmd.Wait(); err != nil {
		return err
	}

	totalDuration := time.Since(startTime)
	var ttft float64
	if !firstTokenTime.IsZero() {
		ttft = float64(firstTokenTime.Sub(startTime).Milliseconds())
	}
	tps := float64(tokenCount) / totalDuration.Seconds()

	// Send final response with telemetry
	return cb(&backend.Response{
		Done:          true,
		TokenCount:    tokenCount,
		TPS:           tps,
		TTFT:          ttft,
		TotalDuration: float64(totalDuration.Milliseconds()),
	})
}

func (b *CLIBackend) Embed(ctx context.Context, opts backend.Options) ([]float32, error) {
	return nil, fmt.Errorf("embeddings not supported in CLI backend (use CGO backend)")
}

func (b *CLIBackend) Capabilities() backend.Capabilities {
	return backend.Capabilities{
		SupportsStreaming: true,
		SupportsGPU:       true,
		SupportsBatching:  true,
	}
}

func (b *CLIBackend) buildArgs(opts backend.Options) []string {
	return wrapper.BuildArguments(opts)
}

// modelsEnv returns the current environment with HF_HOME redirected to the
// project's models folder so downloads don't go to the default C drive cache.
func modelsEnv() []string {
	modelsDir := resolveModelsDir()
	env := os.Environ()
	filtered := env[:0]
	for _, e := range env {
		key := strings.ToUpper(e)
		if strings.HasPrefix(key, "HF_HOME=") || strings.HasPrefix(key, "HUGGINGFACE_HUB_CACHE=") {
			continue
		}
		filtered = append(filtered, e)
	}
	filtered = append(filtered, "HF_HOME="+modelsDir)
	filtered = append(filtered, "HUGGINGFACE_HUB_CACHE="+modelsDir)
	return filtered
}

func resolveModelsDir() string {
	if execPath, err := os.Executable(); err == nil {
		execDir := filepath.Dir(execPath)
		projectRoot := filepath.Dir(execDir)

		candidates := []string{
			filepath.Join(projectRoot, "models"),
			filepath.Join(execDir, "models"),
		}

		for _, candidate := range candidates {
			if info, err := os.Stat(candidate); err == nil && info.IsDir() {
				return candidate
			}
		}

		modelsDir := filepath.Join(projectRoot, "models")
		os.MkdirAll(modelsDir, 0755)
		return modelsDir
	}

	modelsDir, _ := filepath.Abs("models")
	os.MkdirAll(modelsDir, 0755)
	return modelsDir
}
