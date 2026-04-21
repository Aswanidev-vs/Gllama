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
	frames := []string{"|", "/", "-", "\\"}
	go func() {
		i := 0
		for {
			select {
			case <-s.stopChan:
				fmt.Print("\r\033[K")
				s.doneChan <- true
				return
			default:
				fmt.Printf("\r%s %s ", frames[i%len(frames)], s.message)
				i++
				time.Sleep(100 * time.Millisecond)
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
	)

	flag.StringVar(&serverAddr, "server", "http://localhost:11432", "Gllama server address")
	flag.StringVar(&serverAddr, "s", "http://localhost:11432", "Gllama server address (shorthand)")

	flag.StringVar(&model, "model", "", "Model name")
	flag.StringVar(&model, "m", "", "Model name (shorthand)")

	flag.StringVar(&prompt, "prompt", "", "Prompt to generate from")
	flag.StringVar(&prompt, "p", "", "Prompt to generate (shorthand)")

	flag.BoolVar(&stream, "stream", true, "Stream the output")
	flag.BoolVar(&stream, "st", true, "Stream the output (shorthand)")

	flag.IntVar(&maxTokens, "n-predict", 0, "Number of tokens to predict")
	flag.IntVar(&maxTokens, "n", 0, "Number of tokens to predict (shorthand)")

	flag.Float64Var(&temperature, "temp", 0.8, "Temperature")

	flag.IntVar(&threads, "threads", 0, "Number of threads")
	flag.IntVar(&threads, "t", 0, "Number of threads (shorthand)")

	flag.StringVar(&hfRepo, "hf", "", "Hugging Face repository (e.g. repo:file or just repo)")
	flag.StringVar(&hfFile, "hff", "", "Hugging Face file name")

	flag.BoolVar(&interactive, "interactive", false, "Run in interactive mode")
	flag.BoolVar(&interactive, "i", false, "Run in interactive mode (shorthand)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s \"What is Go?\"\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -m llama3 -p \"Tell me a joke\"\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M\n", os.Args[0])
	}

	flag.Parse()

	if flag.NArg() > 0 && flag.Arg(0) == "setup" {
		if err := deps.EnsureDependencies(true); err != nil {
			fmt.Printf("Setup failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Setup complete! You can now use Gllama.")
		return
	}

	if prompt == "" && flag.NArg() > 0 {
		prompt = strings.Join(flag.Args(), " ")
	}

	if isLocalAddr(serverAddr) && !isServerRunning(serverAddr) {
		if err := deps.EnsureDependencies(false); err != nil {
			fmt.Printf("Error: Gllama requires llama.cpp to run, but it could not be found or downloaded.\n")
			fmt.Printf("Detail: %v\n", err)
			fmt.Println("You can try running './bin/gllama.exe setup' for a guided installation.")
			os.Exit(1)
		}

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
			}
			runDirect(llamaCliPath, opts)
			return
		}

		fmt.Printf("Server not detected at %s. Attempting to start gllama-server...\n", serverAddr)
		spin := NewSpinner("Starting server...")
		spin.Start()
		if err := startLocalServer(); err != nil {
			spin.Stop()
			fmt.Printf("Warning: Failed to start server automatically: %v\n", err)
			fmt.Println("Please start gllama-server manually and try again.")
		} else {
			for i := 0; i < 20; i++ {
				time.Sleep(500 * time.Millisecond)
				if isServerRunning(serverAddr) {
					spin.Stop()
					fmt.Println("Server started successfully.")
					break
				}
			}
			if !isServerRunning(serverAddr) {
				spin.Stop()
				fmt.Println("Warning: Server started but not responding yet.")
			}
		}
	}

	if hfRepo != "" && hfFile == "" {
		if parts := strings.Split(hfRepo, ":"); len(parts) == 2 {
			hfRepo = parts[0]
			hfFile = parts[1]
		}
	}

	if interactive || (prompt == "" && hfRepo == "") {
		runInteractive(serverAddr, model, stream, maxTokens, temperature, hfRepo, hfFile)
		return
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
	}

	generate(serverAddr, opts)
}

func runDirect(binaryPath string, opts backend.Options) {
	args := wrapper.BuildArguments(opts)
	cmd := exec.Command(binaryPath, args...)

	// Preserve llama.cpp terminal behavior exactly, including native
	// Hugging Face download progress rendering.
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "\nllama-cli exited with error: %v\n", err)
		os.Exit(1)
	}
}

func generate(serverAddr string, opts backend.Options) {
	url := serverAddr + "/api/generate"
	body, _ := json.Marshal(opts)

	spin := NewSpinner("Generating...")
	spin.Start()

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		spin.Stop()
		fmt.Printf("Error connecting to server: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		spin.Stop()
		out, _ := io.ReadAll(resp.Body)
		fmt.Printf("Error from server (%d): %s\n", resp.StatusCode, string(out))
		os.Exit(1)
	}

	if opts.Stream {
		spin.Stop()
		decoder := json.NewDecoder(resp.Body)
		isDownloading := false
		for {
			var r backend.Response
			if err := decoder.Decode(&r); err != nil {
				if err == io.EOF {
					break
				}
				buffered, _ := io.ReadAll(io.MultiReader(decoder.Buffered(), resp.Body))
				if len(buffered) > 0 {
					fmt.Printf("\nServer Error: %s\n", string(buffered))
				} else {
					fmt.Printf("\nError decoding stream: %v\n", err)
				}
				break
			}

			if r.Status == "Downloading" {
				if !isDownloading {
					isDownloading = true
					spin.Stop()
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

			if r.Done {
				break
			}
		}
		fmt.Println()
	} else {
		var r backend.Response
		json.NewDecoder(resp.Body).Decode(&r)
		fmt.Println(r.Content)
	}
}

func runInteractive(serverAddr string, model string, stream bool, maxTokens int, temperature float64, hfRepo string, hfFile string) {
	fmt.Printf("Gllama Interactive Mode (Server: %s)\n", serverAddr)
	fmt.Println("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
	if model != "" {
		fmt.Printf("Model: %s\n", model)
	}
	if hfRepo != "" {
		fmt.Printf("HF Repo: %s, File: %s\n", hfRepo, hfFile)
	}
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
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
		}
		generate(serverAddr, opts)
		fmt.Println()
	}
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
		return fmt.Errorf("could not find gllama-server binary in common locations or PATH")
	}

	exeSuffix := ""
	if strings.Contains(strings.ToLower(os.Getenv("OS")), "windows") {
		exeSuffix = ".exe"
	}
	llamaCliName := "llama-cli" + exeSuffix
	llamaCliPath := filepath.Join(execDir, "deps", llamaCliName)
	if _, err := os.Stat(llamaCliPath); err != nil {
		llamaCliPath = llamaCliName
	}

	cmd := exec.Command(serverCmd, "--llama-path", llamaCliPath)
	cmd.Stdout = nil
	cmd.Stderr = nil

	if err := cmd.Start(); err != nil {
		return err
	}

	return nil
}

func renderProgressBar(percentage float64, speed string) {
	width := 30
	completed := int(float64(width) * (percentage / 100.0))
	if completed > width {
		completed = width
	}

	bar := strings.Repeat("#", completed) + strings.Repeat(" ", width-completed)
	fmt.Printf("\rDownloading: [%s] %.1f%% (%s)  ", bar, percentage, speed)
}