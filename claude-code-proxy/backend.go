package main

import (
	"net/http"
	"os"
	"strconv"
	"time"
)

type Backend struct {
	BaseURL    string
	AuthHeader string
	AuthValue  string
	ChatPath   string
	Client     *http.Client
}

func NewOllamaBackend() *Backend {
	return &Backend{
		BaseURL:    "https://ollama.com/v1",
		AuthHeader: "Authorization",
		AuthValue:  "Bearer " + os.Getenv("CCP_OLLAMA_API_KEY"),
		ChatPath:   "/chat/completions",
		Client:     &http.Client{Timeout: getTimeout()},
	}
}

func NewBifrostBackend() *Backend {
	baseURL := os.Getenv("CCP_BIFROST_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8081"
	}
	return &Backend{
		BaseURL:    baseURL,
		AuthHeader: "Authorization",
		AuthValue:  "Bearer " + os.Getenv("CCP_BIFROST_API_KEY"),
		ChatPath:   "/v1/chat/completions",
		Client:     &http.Client{Timeout: getTimeout()},
	}
}

func getTimeout() time.Duration {
	t := os.Getenv("CCP_REQUEST_TIMEOUT")
	if t == "" {
		return 120 * time.Second
	}
	sec, err := strconv.Atoi(t)
	if err != nil {
		return 120 * time.Second
	}
	return time.Duration(sec) * time.Second
}

func (b *Backend) FullURL() string {
	return b.BaseURL + b.ChatPath
}
