package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Aswanidev-vs/Gllama/internal/backend"
	"github.com/Aswanidev-vs/Gllama/internal/deps"
	"github.com/Aswanidev-vs/Gllama/internal/wrapper"
)

const (
	ColorReset  = "\033[0m"
	ColorBold   = "\033[1m"
	ColorCyan   = "\033[36m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorGray   = "\033[90m"
)

func printLogo() {
	banner := `
   %s ██████╗  ██╗      ██╗      █████╗ ███╗   ███╗ █████╗ 
   %s██╔════╝  ██║      ██║     ██╔══██╗████╗ ████║██╔══██╗
   %s██║  ███╗ ██║      ██║     ███████║██╔████╔██║███████║
   %s██║   ██║ ██║      ██║     ██╔══██║██║╚██╔╝██║██╔══██║
   %s╚██████╔╝ ███████╗ ███████╗██║  ██║██║ ╚═╝ ██║██║  ██║
   %s ╚═════╝  ╚══════╝ ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
`
	fmt.Printf(banner,
		ColorCyan,
		ColorCyan,
		ColorCyan,
		ColorBlue,
		ColorBlue,
		ColorCyan,
	)

	// divider
	fmt.Printf("\n   %s──────────────────────────────────────────────────────────%s\n\n",
		ColorBlue, ColorReset)

	// tagline
	fmt.Printf("   %sGo-first bindings for llama.cpp %s• %s%s%s\n\n",
		ColorGray,
		ColorYellow,
		ColorBold+ColorGreen,
		"v1.0.0",
		ColorReset,
	)

}

func printUsage() {
	fmt.Printf("\n%sUsage:%s gllama <command> [options]\n\n", ColorCyan, ColorReset)
	fmt.Printf("%sCommands:%s\n", ColorCyan, ColorReset)
	fmt.Printf("  %srun%s       Run a model (default)\n", ColorGreen, ColorReset)
	fmt.Printf("  %spull%s      Download a model from Hugging Face\n", ColorGreen, ColorReset)
	fmt.Printf("  %slist%s      List registered models\n", ColorGreen, ColorReset)
	fmt.Printf("  %srm%s        Remove a downloaded model\n", ColorGreen, ColorReset)
	fmt.Printf("  %sps%s        List active models on server\n", ColorGreen, ColorReset)
	fmt.Printf("  %sserve%s     Start the Gllama server\n", ColorGreen, ColorReset)
	fmt.Printf("  %stq%s        Run with TurboQuant optimization\n", ColorGreen, ColorReset)
	fmt.Printf("  %ssetup%s     Setup dependencies\n", ColorGreen, ColorReset)
	fmt.Printf("  %shelp%s      Show this help message\n", ColorGreen, ColorReset)
	fmt.Println()
	fmt.Printf("%sOptions:%s\n", ColorCyan, ColorReset)
	flag.PrintDefaults()
	fmt.Println()
}

func printDetailedHelp() {
	printLogo()
	fmt.Printf("\n%sGllama - The Go-First CLI for llama.cpp%s\n", ColorBold+ColorCyan, ColorReset)
	fmt.Printf("%s──────────────────────────────────────────────────────────%s\n\n", ColorBlue, ColorReset)

	fmt.Printf("%sManagement Commands:%s\n", ColorCyan, ColorReset)
	fmt.Printf("  %spull <repo>%s      Download a model from Hugging Face\n", ColorGreen, ColorReset)
	fmt.Printf("  %slist%s               List your locally downloaded models\n", ColorGreen, ColorReset)
	fmt.Printf("  %srm <model>%s        Remove a model file from disk\n", ColorGreen, ColorReset)
	fmt.Printf("  %sps%s                 Show currently active models on the server\n", ColorGreen, ColorReset)
	fmt.Printf("  %ssetup%s              Install or update llama.cpp dependencies\n", ColorGreen, ColorReset)
	fmt.Println()

	fmt.Printf("%sExecution Commands:%s\n", ColorCyan, ColorReset)
	fmt.Printf("  %srun <model>%s       Start an interactive chat with a model\n", ColorGreen, ColorReset)
	fmt.Printf("  %stq <mode>%s         Run with TurboQuant (lite, q8, q4)\n", ColorGreen, ColorReset)
	fmt.Printf("  %sserve%s              Start the OpenAI-compatible Gllama API server\n", ColorGreen, ColorReset)
	fmt.Println()

	fmt.Printf("%sTurboQuant Modes (Optimization Presets):%s\n", ColorCyan, ColorReset)
	fmt.Printf("  %slite%s   Balanced Speed. Uses Q8_0 keys and F16 values for fast response.\n", ColorYellow, ColorReset)
	fmt.Printf("  %sq8%s     High Precision. Uses Q8_0 for both K/V. Saves ~50%% memory.\n", ColorYellow, ColorReset)
	fmt.Printf("  %sq4%s     Max Savings. Uses Q4_0 for both K/V. Best for 8GB GPUs/RAM.\n", ColorYellow, ColorReset)
	fmt.Println()

	fmt.Printf("%sManual Optimization (Advanced):%s\n", ColorCyan, ColorReset)
	fmt.Printf("  Override any preset using these manual flags:\n")
	fmt.Printf("  %s-ctk <type>%s    Set KV Cache Key type (f16, q8_0, q4_0, iq4_nl)\n", ColorGreen, ColorReset)
	fmt.Printf("  %s-ctv <type>%s    Set KV Cache Value type (f16, q8_0, q4_0, iq4_nl)\n", ColorGreen, ColorReset)
	fmt.Printf("  %s-fa <on|off>%s   Toggle Flash Attention for supported hardware\n", ColorGreen, ColorReset)
	fmt.Printf("  %s-ngl <N>%s       Manually set number of layers to offload to GPU\n", ColorGreen, ColorReset)
	fmt.Println()

	fmt.Printf("%sInteractive Shortcuts:%s\n", ColorCyan, ColorReset)
	fmt.Printf("  %s/exit%s              Quit the interactive session\n", ColorGray, ColorReset)
	fmt.Printf("  %s/regen%s             Regenerate the last response\n", ColorGray, ColorReset)
	fmt.Printf("  %s/clear%s             Clear the chat history\n", ColorGray, ColorReset)
	fmt.Println()

	fmt.Printf("%sOptions:%s\n", ColorCyan, ColorReset)
	flag.PrintDefaults()
	fmt.Println()
}

type Spinner struct {
	stopChan chan bool
	doneChan chan bool
	message  string
}

func NewSpinner(message string) *Spinner {
	return &Spinner{
		stopChan: make(chan bool),
		doneChan: make(chan bool),
		message:  message,
	}
}

func (s *Spinner) Start() {
	frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	go func() {
		i := 0
		for {
			select {
			case <-s.stopChan:
				fmt.Print("\r\033[K")
				s.doneChan <- true
				return
			default:
				fmt.Printf("\r%s%s%s %s ", ColorCyan, frames[i%len(frames)], ColorReset, s.message)
				i++
				time.Sleep(80 * time.Millisecond)
			}
		}
	}()
}

func (s *Spinner) Stop() {
	s.stopChan <- true
	<-s.doneChan
}

func main() {
	var (
		serverAddr  string
		model       string
		prompt      string
		stream      bool
		maxTokens   int
		temperature float64
		threads     int
		hfRepo      string
		hfFile      string
		interactive bool
		turboQuant  string
	)

	flag.StringVar(&serverAddr, "server", "http://localhost:11432", "Gllama server address")
	flag.StringVar(&serverAddr, "s", "http://localhost:11432", "Gllama server address (shorthand)")
	flag.StringVar(&model, "model", "", "Model name")
	flag.StringVar(&model, "m", "", "Model name (shorthand)")
	flag.StringVar(&prompt, "prompt", "", "Prompt to generate from")
	flag.StringVar(&prompt, "p", "", "Prompt to generate (shorthand)")
	flag.BoolVar(&stream, "stream", true, "Stream the output")
	flag.IntVar(&maxTokens, "n", 0, "Number of tokens to predict")
	flag.Float64Var(&temperature, "temp", 0.8, "Temperature")
	flag.IntVar(&threads, "threads", 0, "Number of threads")
	flag.StringVar(&hfRepo, "hf", "", "Hugging Face repository")
	flag.StringVar(&hfFile, "hff", "", "Hugging Face file name")
	flag.BoolVar(&interactive, "interactive", false, "Run in interactive mode")
	flag.StringVar(&turboQuant, "tq", "", "TurboQuant mode (lite, q8, q4)")

	flag.Usage = printUsage
	flag.Parse()

	// Show logo and usage if no arguments and no flags are set
	if flag.NArg() == 0 && model == "" && hfRepo == "" && !interactive {
		printLogo()
		printUsage()
		return
	}

	command := "run"
	if flag.NArg() > 0 {
		switch flag.Arg(0) {
		case "run", "pull", "list", "rm", "ps", "serve", "tq", "setup", "help":
			command = flag.Arg(0)
			
			if command == "tq" {
				if flag.NArg() >= 2 {
					turboQuant = flag.Arg(1)
					if flag.NArg() >= 3 && model == "" {
						model = flag.Arg(2)
					}
				}
			} else if command == "run" {
				// Handle "gllama run tq lite model.gguf"
				if flag.NArg() >= 2 && flag.Arg(1) == "tq" {
					if flag.NArg() >= 3 {
						turboQuant = flag.Arg(2)
						if flag.NArg() >= 4 && model == "" {
							model = flag.Arg(3)
						}
					}
				} else if model == "" && flag.NArg() >= 2 {
					model = flag.Arg(1)
				}
			} else {
				// Standard command handling (list, ps, etc)
				if model == "" && flag.NArg() >= 2 {
					model = flag.Arg(1)
				}
			}
		default:
			// If the first arg looks like a .gguf file, treat it as a model
			if strings.HasSuffix(flag.Arg(0), ".gguf") && model == "" {
				model = flag.Arg(0)
			} else if prompt == "" {
				prompt = strings.Join(flag.Args(), " ")
			}
		}
	}

	// Resolve bare model filename to full path in models/ directory
	if model != "" && !filepath.IsAbs(model) {
		if _, err := os.Stat(model); os.IsNotExist(err) {
			// Try resolving from models directory
			modelsDir := resolveModelsDir()
			candidate := filepath.Join(modelsDir, model)
			if !strings.HasSuffix(candidate, ".gguf") {
				candidate += ".gguf"
			}
			if _, err := os.Stat(candidate); err == nil {
				model = candidate
			}
		}
	}

	if command == "setup" {
		if err := deps.EnsureDependencies(true); err != nil {
			fmt.Printf("%sError:%s Setup failed: %v\n", ColorYellow, ColorReset, err)
			os.Exit(1)
		}
		fmt.Printf("%sSuccess!%s Gllama is ready to go.\n", ColorGreen, ColorReset)
		return
	}

	if command == "list" {
		listModels(serverAddr)
		return
	}

	if command == "ps" {
		listActiveModels(serverAddr)
		return
	}

	if command == "help" {
		printDetailedHelp()
		return
	}

	if command == "serve" {
		startServer()
		return
	}

	if command == "rm" {
		if flag.NArg() < 2 {
			fmt.Printf("%sError:%s Please specify a model to remove.\n", ColorYellow, ColorReset)
			return
		}
		removeModel(flag.Arg(1))
		return
	}

	if command == "pull" {
		if hfRepo == "" && flag.NArg() >= 2 {
			hfRepo = flag.Arg(1)
		}
		if hfRepo == "" {
			fmt.Printf("%sError:%s Please specify a Hugging Face repo to pull.\n", ColorYellow, ColorReset)
			return
		}

		llamaCliPath := findLlamaCli()
		if llamaCliPath == "" {
			fmt.Printf("%sError:%s llama-cli not found. Please run 'gllama setup' first.\n", ColorYellow, ColorReset)
			return
		}
		opts := backend.Options{
			HFRepo: hfRepo,
			HFFile: hfFile,
		}
		runDirect(llamaCliPath, opts)
		return
	}

	if command == "tq" {
		if turboQuant == "" {
			fmt.Printf("%sError:%s Please specify a TurboQuant mode (lite, q8, q4).\n", ColorYellow, ColorReset)
			fmt.Printf("Usage: gllama tq <mode> <model>\n")
			return
		}
		if model == "" {
			fmt.Printf("%sError:%s Please specify a model for TurboQuant execution.\n", ColorYellow, ColorReset)
			return
		}
	}

	// Default/Run behavior
	executeRun(serverAddr, model, prompt, stream, maxTokens, temperature, threads, hfRepo, hfFile, interactive, turboQuant)
}

func executeRun(serverAddr, model, prompt string, stream bool, maxTokens int, temperature float64, threads int, hfRepo, hfFile string, interactive bool, turboQuant string) {
	llamaCliPath := findLlamaCli()

	// Always prefer direct execution if available locally
	if llamaCliPath != "" {
		opts := backend.Options{
			Model:       model,
			Prompt:      prompt,
			Stream:      stream,
			MaxTokens:   maxTokens,
			Temperature: temperature,
			Threads:     threads,
			HFRepo:      hfRepo,
			HFFile:      hfFile,
			TurboQuant:  turboQuant,
		}
		runDirect(llamaCliPath, opts)
		return
	}

	// If server is not running and we don't have a local engine, try starting server
	if isLocalAddr(serverAddr) && !isServerRunning(serverAddr) {
		fmt.Printf("%sServer not detected.%s Starting gllama-server...\n", ColorGray, ColorReset)
		spin := NewSpinner("Warming up")
		spin.Start()
		if err := startLocalServer(); err != nil {
			spin.Stop()
			fmt.Printf("%sError:%s Failed to start server: %v\n", ColorYellow, ColorReset, err)
			return
		}

		// Wait for server to respond
		started := false
		for i := 0; i < 20; i++ {
			time.Sleep(500 * time.Millisecond)
			if isServerRunning(serverAddr) {
				spin.Stop()
				fmt.Println(ColorGreen + "✓ Server ready." + ColorReset)
				started = true
				break
			}
		}
		if !started {
			spin.Stop()
			fmt.Printf("%sWarning:%s Server timed out. Try running 'gllama serve' manually.\n", ColorYellow, ColorReset)
			return
		}
	}

	opts := backend.Options{
		Model:       model,
		Prompt:      prompt,
		Stream:      stream,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		Threads:     threads,
		HFRepo:      hfRepo,
		HFFile:      hfFile,
		TurboQuant:  turboQuant,
	}

	if interactive || (prompt == "" && hfRepo == "") {
		runInteractive(serverAddr, model, stream, maxTokens, temperature, hfRepo, hfFile, turboQuant)
		return
	}

	generate(serverAddr, opts)
}

func generate(serverAddr string, opts backend.Options) {
	url := serverAddr + "/api/generate"
	body, _ := json.Marshal(opts)

	spin := NewSpinner("Thinking")
	spin.Start()

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		spin.Stop()
		fmt.Printf("%sError:%s %v\n", ColorYellow, ColorReset, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		spin.Stop()
		out, _ := io.ReadAll(resp.Body)
		fmt.Printf("%sError (%d):%s %s\n", ColorYellow, resp.StatusCode, ColorReset, string(out))
		return
	}

	spin.Stop()
	if opts.Stream {
		decoder := json.NewDecoder(resp.Body)
		isDownloading := false
		var lastResp *backend.Response
		for {
			var r backend.Response
			if err := decoder.Decode(&r); err != nil {
				if err == io.EOF {
					break
				}
				break
			}

			if r.Status == "Downloading" {
				if !isDownloading {
					isDownloading = true
					fmt.Println()
				}
				renderProgressBar(r.Percentage, r.Speed)
				continue
			}

			if isDownloading {
				isDownloading = false
				fmt.Println()
			}

			if r.Content != "" {
				fmt.Print(r.Content)
			}
			lastResp = &r
			if r.Done {
				break
			}
		}
		fmt.Println()
		if lastResp != nil && lastResp.Done {
			fmt.Printf("\n%s\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500%s\n", ColorGray, ColorReset)
			fmt.Printf("%sTPS:%s %.2f %s|%s %sTTFT:%s %.2fms %s|%s %sTokens:%s %d\n",
				ColorCyan, ColorReset, lastResp.TPS, ColorGray, ColorReset,
				ColorCyan, ColorReset, lastResp.TTFT, ColorGray, ColorReset,
				ColorCyan, ColorReset, lastResp.TokenCount)
		}
	} else {
		var r backend.Response
		json.NewDecoder(resp.Body).Decode(&r)
		fmt.Println(r.Content)
		fmt.Printf("\n%sStats:%s TPS: %.2f | TTFT: %.2fms | Tokens: %d\n", ColorCyan, ColorReset, r.TPS, r.TTFT, r.TokenCount)
	}
}

func listModels(serverAddr string) {
	url := serverAddr + "/api/models"
	resp, err := http.Get(url)
	if err != nil {
		// Check local models folder if server down
		modelsDir := resolveModelsDir()
		files, _ := os.ReadDir(modelsDir)
		fmt.Printf("%sLocal Models (%s):%s\n", ColorBold, modelsDir, ColorReset)
		for _, f := range files {
			if !f.IsDir() && strings.HasSuffix(f.Name(), ".gguf") {
				fmt.Printf(" \u2022 %s\n", f.Name())
			}
		}
		return
	}
	defer resp.Body.Close()

	var data struct {
		Models []string `json:"models"`
	}
	json.NewDecoder(resp.Body).Decode(&data)

	fmt.Printf("%sModels registered on server:%s\n", ColorBold, ColorReset)
	for _, m := range data.Models {
		fmt.Printf(" \u2022 %s%s%s\n", ColorGreen, m, ColorReset)
	}
}

func listActiveModels(serverAddr string) {
	// Simple server check
	if !isServerRunning(serverAddr) {
		fmt.Println("Server is not running.")
		return
	}
	fmt.Println("Server is active at " + serverAddr)
}

func removeModel(name string) {
	modelsDir := resolveModelsDir()
	path := filepath.Join(modelsDir, name)
	if !strings.HasSuffix(path, ".gguf") {
		path += ".gguf"
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		fmt.Printf("%sError:%s Model %s not found in %s\n", ColorYellow, ColorReset, name, modelsDir)
		return
	}

	err := os.Remove(path)
	if err != nil {
		fmt.Printf("%sError:%s Failed to remove model: %v\n", ColorYellow, ColorReset, err)
		return
	}
	fmt.Printf("%sSuccess:%s Removed %s\n", ColorGreen, ColorReset, name)
}

func startServer() {
	printLogo()
	fmt.Printf("%sStarting Gllama Server...%s\n", ColorBold, ColorReset)

	execPath, _ := os.Executable()
	execDir := filepath.Dir(execPath)
	serverName := "gllama-server"
	if strings.Contains(strings.ToLower(os.Getenv("OS")), "windows") {
		serverName += ".exe"
	}

	cmd := exec.Command(filepath.Join(execDir, serverName))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()
}

func isLocalAddr(addr string) bool {
	return strings.Contains(addr, "localhost") || strings.Contains(addr, "127.0.0.1")
}

func isServerRunning(addr string) bool {
	u, err := url.Parse(addr)
	if err != nil {
		return false
	}
	conn, err := net.DialTimeout("tcp", u.Host, 200*time.Millisecond)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func startLocalServer() error {
	execPath, err := os.Executable()
	if err != nil {
		return err
	}
	execDir := filepath.Dir(execPath)

	serverName := "gllama-server"
	if strings.Contains(strings.ToLower(os.Getenv("OS")), "windows") || strings.Contains(strings.ToLower(execPath), ".exe") {
		serverName = "gllama-server.exe"
	}

	searchPaths := []string{
		filepath.Join(execDir, serverName),
		serverName,
		filepath.Join(".", serverName),
		filepath.Join("cmd", "gllama-server", serverName),
		filepath.Join(filepath.Dir(execDir), "cmd", "gllama-server", serverName),
	}

	var serverCmd string
	for _, path := range searchPaths {
		if _, err := os.Stat(path); err == nil {
			serverCmd = path
			break
		}
	}

	if serverCmd == "" {
		if path, err := exec.LookPath(serverName); err == nil {
			serverCmd = path
		}
	}

	if serverCmd == "" {
		return fmt.Errorf("could not find gllama-server binary")
	}

	cmd := exec.Command(serverCmd)
	return cmd.Start()
}

// findLlamaCli searches for the llama-cli binary in multiple locations,
// allowing gllama to be run from the project root or from PATH.
func findLlamaCli() string {
	binaryName := "llama-cli"
	if strings.Contains(strings.ToLower(os.Getenv("OS")), "windows") {
		binaryName += ".exe"
	}

	var searchPaths []string

	// 1. Relative to executable (works for ./bin/gllama or installed in go/bin)
	if execPath, err := os.Executable(); err == nil {
		execDir := filepath.Dir(execPath)
		searchPaths = append(searchPaths, filepath.Join(execDir, "deps", binaryName))
		// Also check same dir as executable
		searchPaths = append(searchPaths, filepath.Join(execDir, binaryName))
	}

	// 2. Relative to CWD (works for running from project root)
	if cwd, err := os.Getwd(); err == nil {
		searchPaths = append(searchPaths, filepath.Join(cwd, "bin", "deps", binaryName))
		searchPaths = append(searchPaths, filepath.Join(cwd, binaryName))
	}

	for _, p := range searchPaths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	// 3. System PATH (fallback for global installation)
	if path, err := exec.LookPath(binaryName); err == nil {
		return path
	}

	return ""
}

// isLlamaCppBannerLine returns true if the line is part of the llama.cpp
// startup banner that should be suppressed in favour of Gllama branding.
func isLlamaCppBannerLine(line string) bool {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return true // blank lines within the banner region
	}
	lower := strings.ToLower(trimmed)
	// Backend loading noise
	if strings.Contains(lower, "load_backend:") ||
		strings.Contains(lower, "load_model:") ||
		strings.Contains(lower, "loading model") ||
		strings.Contains(lower, "main: interactive mode") ||
		strings.Contains(lower, "main: llama_model_loader") ||
		strings.Contains(lower, "system_info:") ||
		strings.HasPrefix(lower, "main: ") {
		return true
	}
	// ASCII art (box-drawing / block characters used in llama.cpp logo)
	if strings.ContainsAny(line, "█░▒▓▄▀╗╚╝╔║═╠╣╬╩╦") {
		return true
	}
	// Build / model / modalities info lines
	if (strings.HasPrefix(lower, "build") ||
		strings.HasPrefix(lower, "model") ||
		strings.HasPrefix(lower, "modalities") ||
		strings.HasPrefix(lower, "version")) && strings.Contains(line, ":") {
		return true
	}
	// Available commands section
	if strings.Contains(lower, "available commands:") {
		return true
	}
	if strings.HasPrefix(trimmed, "/") {
		// Suppress any /command help lines
		cmds := []string{"/exit", "/regen", "/clear", "/read", "/glob", "/help"}
		for _, c := range cmds {
			if strings.HasPrefix(trimmed, c) {
				return true
			}
		}
	}
	return false
}

// printGllamaRunBanner prints the Gllama branding for an interactive run session.
func printGllamaRunBanner(modelName string) {
	banner := `
   %s ██████╗  ██╗      ██╗      █████╗ ███╗   ███╗ █████╗ 
   %s██╔════╝  ██║      ██║     ██╔══██╗████╗ ████║██╔══██╗
   %s██║  ███╗ ██║      ██║     ███████║██╔████╔██║███████║
   %s██║   ██║ ██║      ██║     ██╔══██║██║╚██╔╝██║██╔══██║
   %s╚██████╔╝ ███████╗ ███████╗██║  ██║██║ ╚═╝ ██║██║  ██║
   %s ╚═════╝  ╚══════╝ ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
`
	fmt.Fprintf(os.Stderr, banner,
		ColorCyan, ColorCyan, ColorCyan,
		ColorBlue, ColorBlue, ColorCyan,
	)
	fmt.Fprintf(os.Stderr, "\n   %s──────────────────────────────────────────────────────────%s\n\n",
		ColorBlue, ColorReset)
	fmt.Fprintf(os.Stderr, "   %sGo-first bindings for llama.cpp %s• %s%s%s\n\n",
		ColorGray, ColorYellow, ColorBold+ColorGreen, "v1.0.0", ColorReset)

	if modelName != "" {
		displayName := filepath.Base(modelName)
		fmt.Fprintf(os.Stderr, "   %smodel%s    : %s\n", ColorGray, ColorReset, displayName)
	}

	fmt.Fprintf(os.Stderr, "\n   %savailable commands:%s\n", ColorGray, ColorReset)
	fmt.Fprintf(os.Stderr, "     %s/exit%s or %sCtrl+C%s    stop or exit\n", ColorGreen, ColorReset, ColorGreen, ColorReset)
	fmt.Fprintf(os.Stderr, "     %s/regen%s             regenerate the last response\n", ColorGreen, ColorReset)
	fmt.Fprintf(os.Stderr, "     %s/clear%s             clear the chat history\n", ColorGreen, ColorReset)

	fmt.Fprintf(os.Stderr, "\n   %sgllama commands:%s\n", ColorGray, ColorReset)
	fmt.Fprintf(os.Stderr, "     %spull%s <repo>        download models from hugging face\n", ColorGreen, ColorReset)
	fmt.Fprintf(os.Stderr, "     %slist%s               list your local models\n", ColorGreen, ColorReset)
	fmt.Fprintf(os.Stderr, "     %srm%s <model>         remove a downloaded model\n", ColorGreen, ColorReset)
	fmt.Fprintf(os.Stderr, "     %sps%s                 list active models on server\n", ColorGreen, ColorReset)
	fmt.Fprintf(os.Stderr, "     %stq%s <mode>          run with turboquant (lite, q8, q4)\n", ColorGreen, ColorReset)
	fmt.Fprintf(os.Stderr, "     %sserve%s              start the gllama api server\n", ColorGreen, ColorReset)
	fmt.Fprintf(os.Stderr, "     %ssetup%s              re-run dependency setup\n", ColorGreen, ColorReset)
	fmt.Fprintln(os.Stderr)
}

type EphemeralFilter struct {
	dest          io.Writer
	buf           bytes.Buffer
	isRaw         bool
	mu            sync.Mutex
	isThinking    bool
	thinkingLines int
}

func (f *EphemeralFilter) Write(p []byte) (n int, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// High-speed fast-pass for the chat phase
	if f.isRaw && !f.isThinking {
		if bytes.IndexByte(p, '[') == -1 && bytes.IndexByte(p, '<') == -1 && 
		   !bytes.Contains(p, []byte("Thinking Process")) {
			return f.dest.Write(p)
		}
	}

	remaining := p
	for len(remaining) > 0 {
		if !f.isRaw {
			// Phase 1: Startup/Banner Filtering
			idx := bytes.IndexByte(remaining, '\n')
			promptIdx := bytes.Index(remaining, []byte(">"))

			if promptIdx != -1 && (idx == -1 || promptIdx < idx) {
				// Found a potential prompt
				f.buf.Write(remaining[:promptIdx+1])
				if bytes.HasSuffix(f.buf.Bytes(), []byte(">")) || bytes.HasSuffix(f.buf.Bytes(), []byte("> ")) {
					f.isRaw = true
					f.dest.Write([]byte("\r\033[K" + ColorBold + ColorGreen + "> " + ColorReset))
					f.buf.Reset()
					return len(p), nil // Switch to raw for the rest of this chunk
				}
				remaining = remaining[promptIdx+1:]
				continue
			}

			if idx != -1 {
				f.buf.Write(remaining[:idx+1])
				line := f.buf.String()
				if !isLlamaCppBannerLine(line) {
					f.dest.Write([]byte(line))
				}
				f.buf.Reset()
				remaining = remaining[idx+1:]
			} else {
				f.buf.Write(remaining)
				break
			}
		} else {
			// Phase 2: Chat/Thinking Processing
			// Check for end-thinking marker even on partial lines
			lowerRem := strings.ToLower(string(remaining))
			markers := []string{"[end thinking]", "</thought>", "end thinking:", "[end]", "[ prompt:", "\n\n"}
			foundEnd := false
			tagLen := 0
			endIdx := -1
			
			for _, m := range markers {
				if idx := strings.Index(lowerRem, m); idx != -1 {
					if m == "\n\n" && f.thinkingLines < 2 {
						continue
					}
					foundEnd = true
					endIdx = idx
					tagLen = len(m)
					break
				}
			}

			if f.isThinking && foundEnd {
				f.isThinking = false
				// RESTORE CURSOR and CLEAR DOWN (Professional Collapse)
				f.dest.Write([]byte("\033[u\033[J"))
				f.thinkingLines = 0
				
				if strings.Contains(lowerRem[endIdx:endIdx+tagLen], "[ prompt:") {
					// Keep footer
				} else {
					remaining = remaining[endIdx+tagLen:]
					continue
				}
			}

			idx := bytes.IndexByte(remaining, '\n')
			if idx == -1 {
				// Partial line
				if f.isThinking {
					if strings.Contains(lowerRem, "[start thinking]") || strings.Contains(lowerRem, "<thought>") || 
					   strings.Contains(lowerRem, "thinking process:") {
						f.isThinking = true
						f.thinkingLines = 0
						// SAVE CURSOR POSITION
						f.dest.Write([]byte("\033[s"))
						f.dest.Write([]byte(fmt.Sprintf("%s(thinking...)\n%s", ColorGray, ColorReset)))
						f.thinkingLines++
						break 
					}
					f.dest.Write([]byte(ColorGray + string(remaining) + ColorReset))
				} else {
					f.dest.Write(remaining)
				}
				break
			}

			line := remaining[:idx+1]
			lower := strings.ToLower(string(line))
			
			if strings.Contains(lower, "[start thinking]") || strings.Contains(lower, "<thought>") || 
			   strings.Contains(lower, "thinking process:") {
				f.isThinking = true
				f.thinkingLines = 0
				// SAVE CURSOR POSITION
				f.dest.Write([]byte("\033[s"))
				f.dest.Write([]byte(fmt.Sprintf("%s(thinking...)\n%s", ColorGray, ColorReset)))
				f.thinkingLines++
			} else if f.isThinking {
				f.dest.Write([]byte(ColorGray + string(line) + ColorReset))
				f.thinkingLines++
			} else {
				f.dest.Write(line)
			}
			remaining = remaining[idx+1:]
		}
	}
	return len(p), nil
}

func runDirect(binaryPath string, opts backend.Options) {
	printGllamaRunBanner(opts.Model)

	args := wrapper.BuildArguments(opts)
	cmd := exec.Command(binaryPath, args...)
	cmd.Env = setModelsEnv()

	stdoutFilter := &EphemeralFilter{dest: os.Stdout}
	stderrFilter := &EphemeralFilter{dest: os.Stderr}

	cmd.Stdin = os.Stdin
	cmd.Stdout = stdoutFilter
	cmd.Stderr = stderrFilter

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 130 {
			return
		}
		fmt.Fprintf(os.Stderr, "\n%sNote:%s session ended (%v)\n", ColorGray, ColorReset, err)
	}
}

func runInteractive(serverAddr string, model string, stream bool, maxTokens int, temperature float64, hfRepo string, hfFile string, turboQuant string) {
	printLogo()
	fmt.Printf("%sInteractive Mode%s | Server: %s\n", ColorBold, ColorReset, serverAddr)
	fmt.Println(ColorGray + "Type 'exit' to quit." + ColorReset)
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("%s>%s ", ColorGreen, ColorReset)
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()
		if input == "exit" || input == "quit" {
			break
		}
		if input == "" {
			continue
		}

		opts := backend.Options{
			Model:       model,
			Prompt:      input,
			Stream:      stream,
			MaxTokens:   maxTokens,
			Temperature: temperature,
			HFRepo:      hfRepo,
			HFFile:      hfFile,
			TurboQuant:  turboQuant,
		}
		generate(serverAddr, opts)
		fmt.Println()
	}
}

func setModelsEnv() []string {
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
	// 1. Check CWD first (most common for local dev)
	if cwd, err := os.Getwd(); err == nil {
		candidate := filepath.Join(cwd, "models")
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
	}

	// 2. Check relative to executable
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
		// Fallback: create in project root of executable
		modelsDir := filepath.Join(projectRoot, "models")
		os.MkdirAll(modelsDir, 0755)
		return modelsDir
	}

	// 3. Last resort fallback
	modelsDir, _ := filepath.Abs("models")
	os.MkdirAll(modelsDir, 0755)
	return modelsDir
}

func renderProgressBar(percentage float64, speed string) {
	width := 30
	completed := int(float64(width) * (percentage / 100.0))
	if completed > width {
		completed = width
	}
	bar := strings.Repeat("\u2588", completed) + strings.Repeat("\u2591", width-completed)
	fmt.Printf("\r  %sPulling:%s [%s] %.1f%% (%s)  ", ColorCyan, ColorReset, bar, percentage, speed)
}
