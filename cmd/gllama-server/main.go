package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/Aswanidev-vs/Gllama/api/http"
	"github.com/Aswanidev-vs/Gllama/internal/backend/cgo"
	"github.com/Aswanidev-vs/Gllama/internal/backend/cli"
	"github.com/Aswanidev-vs/Gllama/internal/engine"
)

func findLlamaCLI(override string) string {
	// 1. Get executable directory
	execPath, _ := os.Executable()
	execDir := filepath.Dir(execPath)
	projectRoot := filepath.Dir(execDir)

	exeSuffix := ""
	if os.PathSeparator == '\\' {
		exeSuffix = ".exe"
	}
	llamaCliName := "llama-cli" + exeSuffix

	// 2. Prioritize local deps folder (Gllama/bin/deps)
	// Check relative to executable and relative to CWD
	localPaths := []string{
		filepath.Join(execDir, "deps", llamaCliName),
		filepath.Join(projectRoot, "bin", "deps", llamaCliName),
		filepath.Join("bin", "deps", llamaCliName),
	}

	for _, p := range localPaths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	// 3. Fallback to override if it exists
	if override != "" && override != "llama.cpp/build/bin/llama-cli.exe" {
		if _, err := os.Stat(override); err == nil {
			return override
		}
	}

	// 4. Final attempt: look in PATH
	if p, err := exec.LookPath(llamaCliName); err == nil {
		return p
	}

	return override
}

func main() {
	port := flag.Int("port", 11432, "Port to run the server on")
	backendType := flag.String("backend", "cli", "Backend type: cli or cgo")
	llamaPathFlag := flag.String("llama-path", "llama.cpp/build/bin/llama-cli.exe", "Path to llama-cli executable")
	modelDir := flag.String("models", "models", "Directory containing models")
	configsDir := flag.String("configs", "configs", "Directory containing model YAML configs")
	flag.Parse()

	llamaPath := findLlamaCLI(*llamaPathFlag)

	// Initialize Engine
	eng := engine.NewEngine()

	// Initialize Backend
	if *backendType == "cgo" {
		cgoBackend := cgo.NewCGOBackend()
		eng.SetBackend(cgoBackend)
	} else {
		cliBackend := cli.NewCLIBackend(llamaPath)
		eng.SetBackend(cliBackend)
	}

	// Handle default paths relative to executable if they don't exist in CWD
	execPath, _ := os.Executable()
	execDir := filepath.Dir(execPath)
	projectRoot := filepath.Dir(execDir)

	if _, err := os.Stat(*modelDir); os.IsNotExist(err) {
		altPath := filepath.Join(projectRoot, "models")
		if _, err := os.Stat(altPath); err == nil {
			*modelDir = altPath
		}
	}

	if _, err := os.Stat(*configsDir); os.IsNotExist(err) {
		altPath := filepath.Join(projectRoot, "configs")
		if _, err := os.Stat(altPath); err == nil {
			*configsDir = altPath
		}
	}

	modelConfigs, err := engine.LoadConfigsFromDir(*configsDir)
	if err == nil {
		for _, conf := range modelConfigs {
			eng.RegisterConfig(conf)
			fmt.Printf("Registered config: %s\n", conf.Name)
		}
	} else {
		fmt.Printf("Warning: Could not read configs directory %s: %v\n", *configsDir, err)
	}

	// Register models in the directory (simple scan for .gguf files)
	files, err := os.ReadDir(*modelDir)
	if err == nil {
		for _, f := range files {
			if !f.IsDir() && (filepath.Ext(f.Name()) == ".gguf") {
				name := f.Name()[:len(f.Name())-5]
				path := filepath.Join(*modelDir, f.Name())
				eng.RegisterModel(name, path)
				fmt.Printf("Registered model: %s -> %s\n", name, path)
			}
		}
	} else {
		fmt.Printf("Warning: Could not read models directory %s: %v\n", *modelDir, err)
	}

	// Start Server
	server := http.NewServer(*port, eng)

	go func() {
		if err := server.Start(); err != nil {
			log.Fatalf("Server failed: %v", err)
		}
	}()

	// Wait for interrupt signal
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop

	fmt.Println("\nShutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 5)
	defer cancel()
	server.Shutdown(ctx)
}
