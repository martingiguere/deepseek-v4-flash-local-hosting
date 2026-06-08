package main

import (
	"encoding/json"
	"fmt"
	"strings"
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
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

type OpenAIFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAITool struct {
	Type       string          `json:"type"`
	Function   OpenAIFunction  `json:"function,omitempty"`
	Parameters json.RawMessage `json:"parameters,omitempty"`
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
		return nil, nil
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
			prevContent, _ := prev.Content.(string)
			currContent, _ := curr.Content.(string)
			prev.Content = prevContent + "\n" + currContent
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
			Function: OpenAIFunction{
				Name: t.Name,
			},
			Parameters: t.InputSchema,
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
		if typ, ok := obj["type"].(string); ok && typ == "tool" {
			if name, ok := obj["name"].(string); ok {
				return map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name": name,
					},
				}
			}
		}
	}
	return tc
}

func MapModel(claudeModel string) string {
	if strings.HasPrefix(claudeModel, "claude-opus") {
		return "deepseek-v4-pro"
	}
	return "deepseek-v4-flash"
}

func BuildError(status int, msg string) (int, []byte) {
	body, _ := json.Marshal(map[string]interface{}{
		"type": "error",
		"error": map[string]string{
			"type":    "api_error",
			"message": msg,
		},
	})
	return status, body
}
