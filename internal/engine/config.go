package engine

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Aswanidev-vs/Gllama/internal/backend"
	"gopkg.in/yaml.v3"
)

// ModelConfig represents model-specific default options
type ModelConfig struct {
	Name    string          `yaml:"name"`
	Path    string          `yaml:"path"`
	Options backend.Options `yaml:"options"`
}

// LoadConfig loads a model configuration from a YAML file
func LoadConfig(path string) (*ModelConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var conf ModelConfig
	if err := yaml.Unmarshal(data, &conf); err != nil {
		return nil, fmt.Errorf("error parsing yaml: %w", err)
	}

	return &conf, nil
}

// LoadConfigsFromDir scans a directory for .yaml files and returns a map of configs
func LoadConfigsFromDir(dir string) (map[string]*ModelConfig, error) {
	configs := make(map[string]*ModelConfig)
	files, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return configs, nil
		}
		return nil, err
	}

	for _, f := range files {
		if !f.IsDir() && (filepath.Ext(f.Name()) == ".yaml" || filepath.Ext(f.Name()) == ".yml") {
			conf, err := LoadConfig(filepath.Join(dir, f.Name()))
			if err != nil {
				fmt.Printf("Warning: Could not load config %s: %v\n", f.Name(), err)
				continue
			}
			configs[conf.Name] = conf
		}
	}

	return configs, nil
}
