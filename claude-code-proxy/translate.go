package main

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// ---- Anthropic request types ----

type AnthropicRequest struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	Messages      []AnthropicMessage `json:"messages"`
	System        json.RawMessage    `json:"system,omitempty"`
	Tools         []AnthropicTool    `json:"tools,omitempty"`
	ToolChoice    json.RawMessage    `json:"tool_choice,omitempty"`
	Stream        bool               `json:"stream"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Metadata      *struct {
		UserID string `json:"user_id"`
	} `json:"metadata,omitempty"`
}

type AnthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type AnthropicBlock struct {
	Type         string          `json:"type"`
	Text         string          `json:"text,omitempty"`
	ID           string          `json:"id,omitempty"`
	Name         string          `json:"name,omitempty"`
	Input        json.RawMessage `json:"input,omitempty"`
	ToolUseID    string          `json:"tool_use_id,omitempty"`
	Content      json.RawMessage `json:"content,omitempty"`
	CacheControl json.RawMessage `json:"cache_control,omitempty"`
}

type AnthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// ---- OpenAI request types ----

type OpenAIRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	Tools       []OpenAITool    `json:"tools,omitempty"`
	ToolChoice  interface{}     `json:"tool_choice,omitempty"`
	Stream      bool            `json:"stream"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
	User        string          `json:"user,omitempty"`
}

type OpenAIMessage struct {
	Role       string           `json:"role"`
	Content    interface{}      `json:"content,omitempty"`
	Reasoning  string           `json:"reasoning,omitempty"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type OpenAIToolCall struct {
	ID       string         `json:"id"`
	Index    int            `json:"index,omitempty"`
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

type OpenAIFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIToolFunc `json:"function"`
}

type OpenAIToolFunc struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ---- OpenAI response types ----

type OpenAIResponse struct {
	ID      string         `json:"id"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   OpenAIUsage    `json:"usage"`
}

type OpenAIChoice struct {
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type OpenAIStreamChunk struct {
	Choices []OpenAIStreamChoice `json:"choices"`
}

type OpenAIStreamChoice struct {
	Delta        OpenAIStreamDelta `json:"delta"`
	FinishReason *string           `json:"finish_reason"`
}

type OpenAIStreamDelta struct {
	Content   string           `json:"content,omitempty"`
	Reasoning string           `json:"reasoning,omitempty"`
	ToolCalls []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ---- Anthropic response types ----

type AnthropicResponse struct {
	ID         string               `json:"id"`
	Type       string               `json:"type"`
	Role       string               `json:"role"`
	Model      string               `json:"model"`
	Content    []AnthropicRespBlock `json:"content"`
	StopReason string               `json:"stop_reason"`
	Usage      AnthropicUsage       `json:"usage"`
}

type AnthropicRespBlock struct {
	Type     string      `json:"type"`
	Text     string      `json:"text,omitempty"`
	ID       string      `json:"id,omitempty"`
	Name     string      `json:"name,omitempty"`
	Input    interface{} `json:"input,omitempty"`
	Thinking string      `json:"thinking,omitempty"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type UnsupportedContentError struct {
	Type string
}

func (e *UnsupportedContentError) Error() string {
	return "unsupported content block type: " + e.Type
}

// TranslateRequest converts an Anthropic Messages API request into an
// OpenAI Chat Completions request, applying model mapping, system message
// extraction, message flattening, and tool/tool_choice translation.
func TranslateRequest(ar *AnthropicRequest, modelName string) (*OpenAIRequest, error) {
	sysMsgs, err := buildSystemMessages(ar.System)
	if err != nil {
		return nil, err
	}

	flatMsgs, err := flattenMessages(ar.Messages)
	if err != nil {
		return nil, err
	}

	merged := mergeAdjacentMessages(sysMsgs, flatMsgs)

	or := &OpenAIRequest{
		Model:       modelName,
		Messages:    merged,
		Stream:      ar.Stream,
		MaxTokens:   ar.MaxTokens,
		Temperature: ar.Temperature,
		TopP:        ar.TopP,
		Stop:        ar.StopSequences,
	}

	if ar.TopK != nil {
		slog.Warn("top_k is not supported by OpenAI API and was dropped", "top_k", *ar.TopK)
	}

	if ar.Metadata != nil && ar.Metadata.UserID != "" {
		or.User = ar.Metadata.UserID
	}

	if len(ar.Tools) > 0 {
		or.Tools = translateTools(ar.Tools)
		or.ToolChoice = translateToolChoice(ar.ToolChoice)
	}

	return or, nil
}

func buildSystemMessages(system json.RawMessage) ([]OpenAIMessage, error) {
	if system == nil || len(system) == 0 {
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(system, &s); err == nil {
		return []OpenAIMessage{{Role: "system", Content: s}}, nil
	}
	var blocks []AnthropicBlock
	if err := json.Unmarshal(system, &blocks); err != nil {
		return nil, fmt.Errorf("failed to parse system: %w", err)
	}
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	if len(parts) > 0 {
		return []OpenAIMessage{{Role: "system", Content: strings.Join(parts, "\n")}}, nil
	}
	return nil, nil
}

func flattenMessages(msgs []AnthropicMessage) ([]OpenAIMessage, error) {
	var result []OpenAIMessage
	for _, am := range msgs {
		var contentStr string
		if err := json.Unmarshal(am.Content, &contentStr); err == nil {
			result = append(result, OpenAIMessage{Role: am.Role, Content: contentStr})
			continue
		}
		var blocks []AnthropicBlock
		if err := json.Unmarshal(am.Content, &blocks); err != nil {
			return nil, fmt.Errorf("failed to parse content: %w", err)
		}
		for _, b := range blocks {
			if b.Type == "image" || b.Type == "document" {
				return nil, &UnsupportedContentError{Type: b.Type}
			}
		}
		switch am.Role {
		case "user":
			var userTexts []string
			for _, b := range blocks {
				if b.Type == "tool_result" {
					content := extractContentString(b.Content)
					result = append(result, OpenAIMessage{
						Role:       "tool",
						ToolCallID: b.ToolUseID,
						Content:    content,
					})
				} else if b.Type == "text" {
					userTexts = append(userTexts, b.Text)
				}
			}
			if len(userTexts) > 0 {
				result = append(result, OpenAIMessage{
					Role:    "user",
					Content: strings.Join(userTexts, "\n"),
				})
			}
		case "assistant":
			var textParts []string
			var toolCalls []OpenAIToolCall
			for _, b := range blocks {
				if b.Type == "text" {
					textParts = append(textParts, b.Text)
				} else if b.Type == "tool_use" {
					toolCalls = append(toolCalls, OpenAIToolCall{
						ID:   b.ID,
						Type: "function",
						Function: OpenAIFunction{
							Name:      b.Name,
							Arguments: string(b.Input),
						},
					})
				} else if b.Type != "thinking" {
					slog.Debug("dropping unknown content block type", "type", b.Type, "role", am.Role)
				}
			}
			msg := OpenAIMessage{Role: "assistant"}
			if len(textParts) > 0 {
				msg.Content = strings.Join(textParts, "\n")
			}
			if len(toolCalls) > 0 {
				msg.ToolCalls = toolCalls
				if msg.Content == nil {
					msg.Content = ""
				}
			}
			result = append(result, msg)
		}
	}
	return result, nil
}

func extractContentString(content json.RawMessage) string {
	if content == nil {
		return ""
	}
	var s string
	if json.Unmarshal(content, &s) == nil {
		return s
	}
	var blocks []AnthropicBlock
	if json.Unmarshal(content, &blocks) == nil {
		var parts []string
		for _, b := range blocks {
			if b.Type == "text" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	slog.Warn("extractContentString: unrecognized content format", "content", string(content))
	return ""
}

func mergeAdjacentMessages(sysMsgs, flatMsgs []OpenAIMessage) []OpenAIMessage {
	all := make([]OpenAIMessage, 0, len(sysMsgs)+len(flatMsgs))
	all = append(all, sysMsgs...)
	all = append(all, flatMsgs...)
	if len(all) <= 1 {
		return all
	}
	merged := []OpenAIMessage{all[0]}
	for i := 1; i < len(all); i++ {
		prev := &merged[len(merged)-1]
		curr := all[i]
		if prev.Role == curr.Role && prev.Role == "user" && prev.ToolCallID == "" && curr.ToolCallID == "" {
			prevContent, ok1 := prev.Content.(string)
			currContent, ok2 := curr.Content.(string)
			if ok1 && ok2 {
				prev.Content = prevContent + "\n" + currContent
			} else {
				merged = append(merged, curr)
			}
		} else {
			merged = append(merged, curr)
		}
	}
	return merged
}

func translateTools(tools []AnthropicTool) []OpenAITool {
	if len(tools) == 0 {
		return nil
	}
	oaiTools := make([]OpenAITool, len(tools))
	for i, t := range tools {
		oaiTools[i] = OpenAITool{
			Type: "function",
			Function: OpenAIToolFunc{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		}
	}
	return oaiTools
}

func translateToolChoice(tc json.RawMessage) interface{} {
	if tc == nil || len(tc) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(tc, &s) == nil {
		if s == "any" {
			return "required"
		}
		return s
	}
	var obj map[string]interface{}
	if json.Unmarshal(tc, &obj) == nil {
		typ, _ := obj["type"].(string)
		if typ == "any" {
			return "required"
		}
		if typ == "tool" {
			if name, ok := obj["name"].(string); ok {
				return map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name": name,
					},
				}
			}
		}
		slog.Debug("translateToolChoice: returning raw object", "type", typ)
		return obj
	}
	slog.Debug("translateToolChoice: unparseable tool_choice, returning nil", "raw", string(tc))
	return nil
}

// MapModel maps a Claude Code model name to the corresponding backend model.
// claude-opus-* maps to deepseek-v4-pro; everything else maps to deepseek-v4-flash.
func MapModel(claudeModel string) string {
	if strings.HasPrefix(claudeModel, "claude-opus") {
		return "deepseek-v4-pro"
	}
	return "deepseek-v4-flash"
}

// BuildError returns an Anthropic-shaped error response with the given status and message.
func BuildError(status int, msg string) (int, []byte) {
	body, err := json.Marshal(map[string]interface{}{
		"type": "error",
		"error": map[string]string{
			"type":    "api_error",
			"message": msg,
		},
	})
	if err != nil {
		slog.Error("BuildError marshal failed", "err", err)
		return status, []byte(`{"type":"error","error":{"type":"api_error","message":"internal error"}}`)
	}
	return status, body
}

func TranslateResponse(or *OpenAIResponse, modelName string) *AnthropicResponse {
	ar := &AnthropicResponse{
		ID:    "msg_" + genID(),
		Type:  "message",
		Role:  "assistant",
		Model: modelName,
		Usage: AnthropicUsage{
			InputTokens:  or.Usage.PromptTokens,
			OutputTokens: or.Usage.CompletionTokens,
		},
	}

	if len(or.Choices) == 0 {
		return ar
	}

	msg := or.Choices[0].Message

	if msg.Reasoning != "" {
		ar.Content = append(ar.Content, AnthropicRespBlock{
			Type:     "thinking",
			Thinking: msg.Reasoning,
		})
	}

	if textContent, ok := msg.Content.(string); ok && textContent != "" {
		ar.Content = append(ar.Content, AnthropicRespBlock{
			Type: "text",
			Text: textContent,
		})
	}

	for _, tc := range msg.ToolCalls {
		var inputObj interface{}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &inputObj); err != nil {
			slog.Warn("failed to parse tool call arguments", "tool", tc.Function.Name, "err", err)
			inputObj = map[string]interface{}{}
		}
		ar.Content = append(ar.Content, AnthropicRespBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: inputObj,
		})
	}

	ar.StopReason = mapFinishReason(or.Choices[0].FinishReason)
	return ar
}

func mapFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	case "content_filter":
		return "end_turn"
	default:
		return "end_turn"
	}
}

func StreamTranslate(w http.ResponseWriter, body io.Reader, modelName, reqID string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		slog.Error("response writer does not support flushing", "id", reqID)
		return
	}

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	type phase int
	const (
		phaseIdle phase = iota
		phaseReasoning
		phaseText
		phaseToolCall
		phaseDone
	)

	currentPhase := phaseIdle
	contentBlockIndex := 0

	emitSSE(w, "message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":      "msg_" + genID(),
			"type":    "message",
			"role":    "assistant",
			"model":   modelName,
			"content": []interface{}{},
		},
	})
	flusher.Flush()

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk OpenAIStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			slog.Debug("stream parse error", "id", reqID, "err", err)
			continue
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta
		finishReason := chunk.Choices[0].FinishReason

		if delta.Reasoning != "" {
			if currentPhase == phaseIdle {
				emitSSE(w, "content_block_start", map[string]interface{}{
					"type":  "content_block_start",
					"index": contentBlockIndex,
					"content_block": map[string]interface{}{
						"type":      "thinking",
						"thinking":  "",
						"signature": "",
					},
				})
				contentBlockIndex++
				currentPhase = phaseReasoning
			}
			if currentPhase == phaseReasoning {
				emitSSE(w, "content_block_delta", map[string]interface{}{
					"type":  "content_block_delta",
					"index": contentBlockIndex - 1,
					"delta": map[string]string{
						"type":     "thinking_delta",
						"thinking": delta.Reasoning,
					},
				})
			}
		}

		if delta.Content != "" {
			if currentPhase == phaseIdle || currentPhase == phaseReasoning {
				if currentPhase == phaseReasoning {
					emitSSE(w, "content_block_stop", map[string]interface{}{
						"type":  "content_block_stop",
						"index": contentBlockIndex - 1,
					})
				}
				emitSSE(w, "content_block_start", map[string]interface{}{
					"type":  "content_block_start",
					"index": contentBlockIndex,
					"content_block": map[string]interface{}{
						"type": "text",
						"text": "",
					},
				})
				contentBlockIndex++
				currentPhase = phaseText
			}
			if currentPhase == phaseText {
				emitSSE(w, "content_block_delta", map[string]interface{}{
					"type":  "content_block_delta",
					"index": contentBlockIndex - 1,
					"delta": map[string]string{
						"type": "text_delta",
						"text": delta.Content,
					},
				})
			}
		}

		if len(delta.ToolCalls) > 0 {
			if currentPhase == phaseIdle || currentPhase == phaseReasoning || currentPhase == phaseText {
				if currentPhase == phaseReasoning || currentPhase == phaseText {
					emitSSE(w, "content_block_stop", map[string]interface{}{
						"type":  "content_block_stop",
						"index": contentBlockIndex - 1,
					})
				}
				currentPhase = phaseToolCall
			}
			if currentPhase == phaseToolCall {
				for _, tc := range delta.ToolCalls {
					emitSSE(w, "content_block_start", map[string]interface{}{
						"type":  "content_block_start",
						"index": contentBlockIndex,
						"content_block": map[string]interface{}{
							"type":  "tool_use",
							"id":    tc.ID,
							"name":  tc.Function.Name,
							"input": map[string]interface{}{},
						},
					})
					contentBlockIndex++

					args := tc.Function.Arguments
					chunkSize := 64
					for i := 0; i < len(args); i += chunkSize {
						end := i + chunkSize
						if end > len(args) {
							end = len(args)
						}
						partial := args[i:end]
						emitSSE(w, "content_block_delta", map[string]interface{}{
							"type":  "content_block_delta",
							"index": contentBlockIndex - 1,
							"delta": map[string]string{
								"type":         "input_json_delta",
								"partial_json": partial,
							},
						})
					}

					emitSSE(w, "content_block_stop", map[string]interface{}{
						"type":  "content_block_stop",
						"index": contentBlockIndex - 1,
					})
				}
			}
		}

		if finishReason != nil && *finishReason != "" {
			if currentPhase == phaseReasoning || currentPhase == phaseText {
				emitSSE(w, "content_block_stop", map[string]interface{}{
					"type":  "content_block_stop",
					"index": contentBlockIndex - 1,
				})
			}

			emitSSE(w, "message_delta", map[string]interface{}{
				"type": "message_delta",
				"delta": map[string]interface{}{
					"stop_reason":   mapFinishReason(*finishReason),
					"stop_sequence": nil,
				},
			})

			emitSSE(w, "message_stop", map[string]interface{}{
				"type": "message_stop",
			})

			currentPhase = phaseDone
		}

		flusher.Flush()
	}

	if currentPhase != phaseDone {
		if contentBlockIndex > 0 && (currentPhase == phaseReasoning || currentPhase == phaseText) {
			emitSSE(w, "content_block_stop", map[string]interface{}{
				"type":  "content_block_stop",
				"index": contentBlockIndex - 1,
			})
		}
		emitSSE(w, "message_delta", map[string]interface{}{
			"type": "message_delta",
			"delta": map[string]interface{}{
				"stop_reason":   "end_turn",
				"stop_sequence": nil,
			},
		})
		emitSSE(w, "message_stop", map[string]interface{}{
			"type": "message_stop",
		})
		flusher.Flush()
	}

	if err := scanner.Err(); err != nil {
		slog.Error("stream read error", "id", reqID, "err", err)
	}
}

func emitSSE(w io.Writer, event string, data interface{}) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		slog.Error("failed to marshal SSE data", "event", event, "err", err)
		return
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(jsonData)); err != nil {
		slog.Error("failed to write SSE", "event", event, "err", err)
	}
}

func genID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		slog.Error("failed to generate random ID", "err", err)
		return fmt.Sprintf("fallback_%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}
