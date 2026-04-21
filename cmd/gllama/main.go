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
	logo := `
  %s________  %s.____    %s.____       %s_____      %s_____      %s_____   
 %s/  _____/  %s|    |   %s|    |     %s/  _  \    %s/     \    %s/  _  \  
%s/   \  ___  %s|    |   %s|    |    %s/  /_\  \  %s/  \ /  \  %s/  /_\  \ 
%s\    \_\  \ %s|    |___%s|    |___%s/    |    \%s/    Y    \%s/    |    \
 %s\______  / %s|_______%s|_______%s\____|__  /%s\____|__  /%s\____|__  /
        %s\/         %s\/       %s\/        %s\/         %s\/         %s\/ 
`
	fmt.Printf(logo, 
		ColorCyan, ColorCyan, ColorCyan, ColorCyan, ColorCyan, ColorCyan,
		ColorCyan, ColorCyan, ColorCyan, ColorCyan, ColorCyan, ColorCyan,
		ColorBlue, ColorBlue, ColorBlue, ColorBlue, ColorBlue, ColorBlue,
		ColorBlue, ColorBlue, ColorBlue, ColorBlue, ColorBlue, ColorBlue,
		ColorCyan, ColorCyan, ColorCyan, ColorCyan, ColorCyan, ColorCyan,
		ColorCyan, ColorCyan, ColorCyan, ColorCyan, ColorCyan, ColorCyan)
	fmt.Printf("\n    %sGo-first bindings for llama.cpp %s| %sv1.0.0%s\n\n", ColorGray, ColorBlue, ColorGreen, ColorReset)
}

func printUsage() {
	fmt.Printf("%sUsage:%s gllama <command> [options]\n\n", ColorBold, ColorReset)
	fmt.Printf("%sCommands:%s\n", ColorBold, ColorReset)
	fmt.Printf("  %srun%s       Run a model (default)\n", ColorGreen, ColorReset)
	fmt.Printf("  %spull%s      Download a model from Hugging Face\n", ColorGreen, ColorReset)
	fmt.Printf("  %slist%s      List registered models\n", ColorGreen, ColorReset)
	fmt.Printf("  %srm%s        Remove a downloaded model\n", ColorGreen, ColorReset)
	fmt.Printf("  %sps%s        List active models on server\n", ColorGreen, ColorReset)
	fmt.Printf("  %sserve%s     Start the Gllama server\n", ColorGreen, ColorReset)
	fmt.Printf("  %stq%s        Run with TurboQuant optimization\n", ColorGreen, ColorReset)
	fmt.Printf("  %ssetup%s     Setup dependencies\n", ColorGreen, ColorReset)
	fmt.Println()
	fmt.Printf("%sOptions:%s\n", ColorBold, ColorReset)
	flag.PrintDefaults()
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
		case "run", "pull", "list", "rm", "ps", "serve", "tq", "setup":
			command = flag.Arg(0)
		default:
			if prompt == "" {
				prompt = strings.Join(flag.Args(), " ")
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
	}

	if command == "tq" {
		if turboQuant == "" {
			turboQuant = "lite" // Default to lite if tq command used without flag
		}
	}

	// Default/Run behavior
	executeRun(serverAddr, model, prompt, stream, maxTokens, temperature, threads, hfRepo, hfFile, interactive, turboQuant)
}

func executeRun(serverAddr, model, prompt string, stream bool, maxTokens int, temperature float64, threads int, hfRepo, hfFile string, interactive bool, turboQuant string) {
	if isLocalAddr(serverAddr) && !isServerRunning(serverAddr) {
		if err := deps.EnsureDependencies(false); err != nil {
			fmt.Printf("%sError:%s Gllama requires llama.cpp to run.\n", ColorYellow, ColorReset)
			os.Exit(1)
		}

		fmt.Printf("%sServer not detected.%s Starting gllama-server...\n", ColorGray, ColorReset)
		spin := NewSpinner("Warming up")
		spin.Start()
		if err := startLocalServer(); err != nil {
			spin.Stop()
			fmt.Printf("%sWarning:%s Failed to start server: %v\n", ColorYellow, ColorReset, err)
			
			// Fallback to runDirect only if server fails to start
			execPath, _ := os.Executable()
			execDir := filepath.Dir(execPath)
			llamaCliPath := filepath.Join(execDir, "deps", "llama-cli")
			if strings.Contains(strings.ToLower(os.Getenv("OS")), "windows") {
				llamaCliPath += ".exe"
			}
			if _, err := os.Stat(llamaCliPath); err == nil {
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
		} else {
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
				fmt.Printf("%sWarning:%s Server started but timed out. Trying to continue...\n", ColorYellow, ColorReset)
			}
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

func runDirect(binaryPath string, opts backend.Options) {
	args := wrapper.BuildArguments(opts)
	cmd := exec.Command(binaryPath, args...)
	cmd.Env = setModelsEnv()
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		os.Exit(1)
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

func renderProgressBar(percentage float64, speed string) {
	width := 30
	completed := int(float64(width) * (percentage / 100.0))
	if completed > width {
		completed = width
	}
	bar := strings.Repeat("\u2588", completed) + strings.Repeat("\u2591", width-completed)
	fmt.Printf("\r  %sPulling:%s [%s] %.1f%% (%s)  ", ColorCyan, ColorReset, bar, percentage, speed)
}