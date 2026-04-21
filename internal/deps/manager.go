package deps

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	RepoOwner = "ggml-org"
	RepoName  = "llama.cpp"
	DepsDir   = "bin/deps"
)

var progressSteps = []string{"", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"}

// EnsureDependencies checks for llama-cli and downloads it if missing
func EnsureDependencies(forceInteractive bool) error {
	exeSuffix := ""
	if runtime.GOOS == "windows" {
		exeSuffix = ".exe"
	}
	expectedPath := filepath.Join(DepsDir, "llama-cli"+exeSuffix)

	if _, err := os.Stat(expectedPath); err == nil {
		return nil
	}

	if !forceInteractive {
		fmt.Printf("Gllama needs llama.cpp binaries to function. Start automatic download? [Y/n]: ")
		var response string
		fmt.Scanln(&response)
		if strings.ToLower(response) == "n" {
			return fmt.Errorf("user declined dependency installation")
		}
	}

	fmt.Println("Fetching latest llama.cpp release info...")
	release, err := GetLatestRelease()
	if err != nil {
		return fmt.Errorf("failed to get release info: %v", err)
	}

	asset, err := ResolveAsset(release)
	if err != nil {
		return fmt.Errorf("failed to resolve asset: %v", err)
	}

	fmt.Printf("Downloading %s...\n", asset.Name)
	tmpZip := filepath.Join(os.TempDir(), asset.Name)

	err = DownloadAsset(asset, tmpZip, func(downloaded, total int64) {
		if total > 0 {
			percent := float64(downloaded) / float64(total) * 100
			bar := renderProgressBar(percent, 24)
			fmt.Printf("\r\033[KInstalling model: [%s] %6.2f%% (%.2f/%.2f MB)", bar, percent, float64(downloaded)/1024/1024, float64(total)/1024/1024)
		} else {
			fmt.Printf("\r\033[KInstalling model: %.2f MB", float64(downloaded)/1024/1024)
		}
	})
	fmt.Println()

	if err != nil {
		return fmt.Errorf("download failed: %v", err)
	}

	fmt.Printf("Extracting to %s...\n", DepsDir)
	if err := Extract(tmpZip, DepsDir); err != nil {
		return fmt.Errorf("extraction failed: %v", err)
	}

	os.Remove(tmpZip)

	return nil
}

type Release struct {
	TagName string  `json:"tag_name"`
	Assets  []Asset `json:"assets"`
}

type Asset struct {
	Name               string `json:"name"`
	BrowserDownloadURL string `json:"browser_download_url"`
}

// GetLatestRelease fetches info about the latest llama.cpp release
func GetLatestRelease() (*Release, error) {
	resp, err := http.Get(fmt.Sprintf("https://api.github.com/repos/%s/%s/releases/latest", RepoOwner, RepoName))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to fetch release info: %s", resp.Status)
	}

	var release Release
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return nil, err
	}
	return &release, nil
}

// ResolveAsset finds the most appropriate asset for the current platform
func ResolveAsset(release *Release) (*Asset, error) {
	osName := runtime.GOOS
	arch := runtime.GOARCH

	var target string
	switch osName {
	case "windows":
		target = "win-vulkan-x64.zip"
	case "linux":
		target = "ubuntu-x64.tar.gz"
	case "darwin":
		if arch == "arm64" {
			target = "macos-arm64.tar.gz"
		} else {
			target = "macos-x64.tar.gz"
		}
	default:
		return nil, fmt.Errorf("unsupported OS: %s", osName)
	}

	for _, asset := range release.Assets {
		if strings.Contains(asset.Name, target) {
			return &asset, nil
		}
	}

	if osName == "windows" {
		for _, asset := range release.Assets {
			if strings.Contains(asset.Name, "win-cpu-x64.zip") {
				return &asset, nil
			}
		}
	}

	return nil, fmt.Errorf("could not find a suitable asset for %s-%s", osName, arch)
}

// DownloadAsset downloads the asset and calls the progress callback
func DownloadAsset(asset *Asset, dest string, progress func(int64, int64)) error {
	resp, err := http.Get(asset.BrowserDownloadURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download asset: %s", resp.Status)
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	size := resp.ContentLength
	var downloaded int64

	buffer := make([]byte, 32*1024)
	for {
		n, err := resp.Body.Read(buffer)
		if n > 0 {
			out.Write(buffer[:n])
			downloaded += int64(n)
			if progress != nil {
				progress(downloaded, size)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}

	return nil
}

// Extract extracts the downloaded archive to the destination directory
func Extract(src, destDir string) error {
	if strings.HasSuffix(src, ".zip") {
		return extractZip(src, destDir)
	} else if strings.HasSuffix(src, ".tar.gz") {
		return extractTarGz(src, destDir)
	}
	return fmt.Errorf("unsupported archive format: %s", src)
}

func extractZip(src, dest string) error {
	r, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer r.Close()

	os.MkdirAll(dest, 0755)

	for _, f := range r.File {
		fpath := filepath.Join(dest, f.Name)
		if f.FileInfo().IsDir() {
			os.MkdirAll(fpath, os.ModePerm)
			continue
		}

		if err := os.MkdirAll(filepath.Dir(fpath), os.ModePerm); err != nil {
			return err
		}

		outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return err
		}

		rc, err := f.Open()
		if err != nil {
			outFile.Close()
			return err
		}

		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()

		if err != nil {
			return err
		}
	}
	return nil
}

func extractTarGz(src, dest string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)

	os.MkdirAll(dest, 0755)

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		target := filepath.Join(dest, header.Name)

		switch header.Typeflag {
		case tar.TypeDir:
			if _, err := os.Stat(target); err != nil {
				if err := os.MkdirAll(target, 0755); err != nil {
					return err
				}
			}
		case tar.TypeReg:
			f, err := os.OpenFile(target, os.O_CREATE|os.O_RDWR, os.FileMode(header.Mode))
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				f.Close()
				return err
			}
			f.Close()
		}
	}
	return nil
}

func renderProgressBar(percentage float64, width int) string {
	if width <= 0 {
		return ""
	}

	if percentage < 0 {
		percentage = 0
	}
	if percentage > 100 {
		percentage = 100
	}

	totalUnits := percentage / 100.0 * float64(width*8)
	fullBlocks := int(totalUnits) / 8
	remainder := int(totalUnits) % 8

	var bar strings.Builder
	bar.Grow(width * len("█"))

	for i := 0; i < width; i++ {
		switch {
		case i < fullBlocks:
			bar.WriteString("█")
		case i == fullBlocks && remainder > 0:
			bar.WriteString(progressSteps[remainder])
		default:
			bar.WriteString(" ")
		}
	}

	return bar.String()
}
