package cli

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
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

	reader := bufio.NewReader(stdout)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
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

	// Send final response
	return cb(&backend.Response{Done: true})
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
