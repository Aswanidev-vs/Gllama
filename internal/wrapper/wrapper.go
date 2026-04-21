package wrapper

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/Aswanidev-vs/Gllama/internal/backend"
)

var (
	// Support various formats: [ 14%] ( 1.4 MB/s) or 14% (1.4 MB/s)
	ProgressRegex = regexp.MustCompile(`(\d+)%.*?\(([\d\.]+\s*[M|K]B/s)\)`)
)

// BuildArguments converts backend options into llama-cli flags
func BuildArguments(opts backend.Options) []string {
	opts = backend.NormalizeOptions(opts)

	var args []string

	modelPath := opts.Model
	if modelPath == "" && opts.HFRepo != "" {
		modelPath = defaultHFModelPath(opts)
	}
	if modelPath != "" {
		args = append(args, "-m", modelPath)
	}

	if opts.Prompt != "" {
		args = append(args, "-p", opts.Prompt)
	}

	if opts.MaxTokens > 0 {
		args = append(args, "-n", strconv.Itoa(opts.MaxTokens))
	}

	if opts.Temperature > 0 {
		args = append(args, "--temp", fmt.Sprintf("%f", opts.Temperature))
	}

	if opts.Threads > 0 {
		args = append(args, "-t", strconv.Itoa(opts.Threads))
	}

	if opts.BatchSize > 0 {
		args = append(args, "-b", strconv.Itoa(opts.BatchSize))
	}

	if opts.ContextSize > 0 {
		args = append(args, "-c", strconv.Itoa(opts.ContextSize))
	}

	if opts.FlashAttention != "" {
		args = append(args, "-fa", opts.FlashAttention)
	}

	if opts.Seed != 0 {
		args = append(args, "--seed", strconv.Itoa(opts.Seed))
	}

	if opts.GPULayers > 0 {
		args = append(args, "-ngl", strconv.Itoa(opts.GPULayers))
	} else if opts.GPU {
		args = append(args, "-ngl", "99")
	}

	if opts.HFRepo != "" {
		args = append(args, "-hf", opts.HFRepo)
		hfFile := opts.HFFile
		if hfFile == "" {
			if strings.Contains(strings.ToLower(opts.HFRepo), "gemma") {
				hfFile = "gemma-4-E2B-it-Q4_K_M.gguf"
			} else {
				hfFile = "Q4_K_M.gguf"
			}
		}
		args = append(args, "-hff", hfFile)
	}

	cacheTypeK, cacheTypeV := kvCacheTypes(opts)
	if cacheTypeK != "" {
		args = append(args, "-ctk", cacheTypeK)
	}
	if cacheTypeV != "" {
		args = append(args, "-ctv", cacheTypeV)
	}

	if opts.KVOffload != nil {
		if *opts.KVOffload {
			args = append(args, "-kvo")
		} else {
			args = append(args, "-nkvo")
		}
	}

	if opts.SWAFull != nil {
		if *opts.SWAFull {
			args = append(args, "--swa-full")
		} else {
			args = append(args, "--no-swa-full")
		}
	}

	if opts.KVUnified != nil {
		if *opts.KVUnified {
			args = append(args, "--kv-unified")
		} else {
			args = append(args, "--no-kv-unified")
		}
	}

	return args
}

func kvCacheTypes(opts backend.Options) (string, string) {
	cacheTypeK := opts.KVCacheTypeK
	cacheTypeV := opts.KVCacheTypeV

	if opts.KVCacheType != "" {
		if cacheTypeK == "" {
			cacheTypeK = opts.KVCacheType
		}
		if cacheTypeV == "" {
			cacheTypeV = opts.KVCacheType
		}
	}

	return cacheTypeK, cacheTypeV
}

func defaultHFModelPath(opts backend.Options) string {
	fileName := opts.HFFile
	if fileName == "" {
		if strings.Contains(strings.ToLower(opts.HFRepo), "gemma") {
			fileName = "gemma-4-E2B-it-Q4_K_M.gguf"
		} else {
			fileName = "Q4_K_M.gguf"
		}
	}

	modelsDir := "models"

	if execPath, err := os.Executable(); err == nil {
		execDir := filepath.Dir(execPath)
		projectRoot := filepath.Dir(execDir)

		candidates := []string{
			filepath.Join(projectRoot, "models"),
			filepath.Join(execDir, "models"),
		}

		for _, candidate := range candidates {
			if info, err := os.Stat(candidate); err == nil && info.IsDir() {
				modelsDir = candidate
				break
			}
		}
	}

	return filepath.Join(modelsDir, fileName)
}

// ParseProgress searches a line for llama-cli download progress
func ParseProgress(line string) (float64, string, bool) {
	matches := ProgressRegex.FindStringSubmatch(line)
	if len(matches) == 3 {
		percentage, _ := strconv.ParseFloat(matches[1], 64)
		speed := matches[2]
		return percentage, speed, true
	}
	return 0, "", false
}
