package main

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

func main() {
	logLevel := new(slog.LevelVar)
	switch os.Getenv("CCP_LOG_LEVEL") {
	case "debug":
		logLevel.Set(slog.LevelDebug)
	case "warn":
		logLevel.Set(slog.LevelWarn)
	case "error":
		logLevel.Set(slog.LevelError)
	default:
		logLevel.Set(slog.LevelInfo)
	}
	logger := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel}))
	slog.SetDefault(logger)

	var backend *Backend
	backendName := "ollama"
	switch os.Getenv("CCP_BACKEND") {
	case "bifrost":
		backend = NewBifrostBackend()
		backendName = "bifrost"
		slog.Info("starting ccp", "backend", "bifrost", "url", backend.BaseURL)
	default:
		backend = NewOllamaBackend()
		slog.Info("starting ccp", "backend", "ollama")
	}

	port := os.Getenv("CCP_PORT")
	if port == "" {
		port = "8080"
	}

	maxConc := 10
	if m := os.Getenv("CCP_MAX_CONCURRENT"); m != "" {
		if v, err := strconv.Atoi(m); err == nil && v > 0 {
			maxConc = v
		}
	}
	sem := make(chan struct{}, maxConc)

	mux := http.NewServeMux()
	var inFlight atomic.Int64

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "ok",
			"backend": backendName,
		})
	})

	mux.HandleFunc("/v1/messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"input_tokens": 0}`))
	})

	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
		default:
			status, body := BuildError(503, "too many concurrent requests")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			w.Write(body)
			return
		}

		inFlight.Add(1)
		defer inFlight.Add(-1)

		reqID := genID()
		start := time.Now()

		if r.Method != http.MethodPost {
			status, body := BuildError(405, "method not allowed")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			w.Write(body)
			return
		}

		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			status, body := BuildError(400, "failed to read request body")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			w.Write(body)
			return
		}

		if slog.Default().Enabled(context.Background(), slog.LevelDebug) {
			slog.Debug("request", "id", reqID, "body", string(bodyBytes))
		}

		var ar AnthropicRequest
		if err := json.Unmarshal(bodyBytes, &ar); err != nil {
			status, body := BuildError(400, "invalid request JSON")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			w.Write(body)
			return
		}

		modelName := MapModel(ar.Model)
		oaiReq, err := TranslateRequest(&ar, modelName)
		if err != nil {
			if _, ok := err.(*UnsupportedContentError); ok {
				status, body := BuildError(400, err.Error())
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				w.Write(body)
				return
			}
			status, body := BuildError(400, err.Error())
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			w.Write(body)
			return
		}

		reqJSON, err := json.Marshal(oaiReq)
		if err != nil {
			slog.Error("failed to marshal backend request", "id", reqID, "err", err)
			status, body := BuildError(502, "failed to build backend request")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			w.Write(body)
			return
		}

if slog.Default().Enabled(context.Background(), slog.LevelDebug) {
		slog.Debug("backend-request", "id", reqID, "body", string(reqJSON))
	}

	forwardHeaders := make(http.Header)
	for k, v := range r.Header {
		kl := strings.ToLower(k)
		if kl == "x-anthropic-billing-header" || kl == "anthropic-version" || strings.HasPrefix(kl, "anthropic-beta") {
			continue
		}
		forwardHeaders[k] = v
	}
	forwardHeaders.Set(backend.AuthHeader, backend.AuthValue)
	forwardHeaders.Set("Content-Type", "application/json")
	if rid := r.Header.Get("x-request-id"); rid != "" {
		forwardHeaders.Set("x-request-id", rid)
	} else {
		forwardHeaders.Set("x-request-id", reqID)
	}

	backendReq, err := http.NewRequestWithContext(r.Context(), "POST", backend.FullURL(), strings.NewReader(string(reqJSON)))
	if err != nil {
		slog.Error("failed to create backend request", "id", reqID, "err", err)
		status, body := BuildError(502, "failed to create backend request")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write(body)
		return
	}
	backendReq.Header = forwardHeaders

		backendResp, err := backend.Client.Do(backendReq)
		if err != nil {
			status, body := BuildError(502, "backend unreachable: "+err.Error())
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			w.Write(body)
			slog.Error("backend error", "id", reqID, "err", err, "duration_ms", time.Since(start).Milliseconds())
			return
		}
		defer backendResp.Body.Close()

		if backendResp.StatusCode >= 400 {
			errBody, readErr := io.ReadAll(backendResp.Body)
			if readErr != nil {
				errBody = []byte("failed to read error body")
			}
			slog.Error("backend error", "id", reqID, "status", backendResp.StatusCode, "body", string(errBody))
			status, body := BuildError(502, "backend returned "+strconv.Itoa(backendResp.StatusCode))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			w.Write(body)
			return
		}

		if ar.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			StreamTranslate(w, backendResp.Body, modelName, reqID)
		} else {
			respBody, err := io.ReadAll(backendResp.Body)
			if err != nil {
				slog.Error("failed to read backend response", "id", reqID, "err", err)
				status, body := BuildError(502, "failed to read backend response")
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				w.Write(body)
				return
			}
			var oaiResp OpenAIResponse
			if err := json.Unmarshal(respBody, &oaiResp); err != nil {
				slog.Error("parse error", "id", reqID, "body", string(respBody))
				status, body := BuildError(502, "failed to parse backend response")
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				w.Write(body)
				return
			}
			anthResp := TranslateResponse(&oaiResp, modelName)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(anthResp)
		}

		slog.Info("request", "id", reqID, "model", ar.Model, "backend_model", modelName,
			"stream", ar.Stream, "duration_ms", time.Since(start).Milliseconds())
	})

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		slog.Info("shutting down", "in_flight", inFlight.Load())
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := srv.Shutdown(ctx); err != nil {
			slog.Error("shutdown error", "err", err)
		}
	}()

	slog.Info("listening", "port", port)
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}
