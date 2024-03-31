package tokens

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pkoukk/tiktoken-go"
	"github.com/sashabaranov/go-openai"
)

type Counter struct {
	model     string
	tokenizer *tiktoken.Tiktoken
}

// NewCounter creates a new token counter for the specified model.
func NewCounter(model string) (*Counter, error) {
	tokenizer, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return nil, err
	}
	return &Counter{
		model:     model,
		tokenizer: tokenizer,
	}, nil
}

// CountTokens returns the number of tokens in a string.
func (c *Counter) CountTokens(txt string) int {
	tokens := c.tokenizer.Encode(txt, nil, nil)
	return len(tokens)
}

var (
	tokensPerReqMessage     = 3
	tokensPerToolCall    = 4
	tokensPerName        = 1
	tokensForMultiTool   = 13
)

// CountRequestTokens returns the number of tokens in a chat completion request.
func (c *Counter) CountRequestTokens(
	req openai.ChatCompletionRequest,
) int {
	var (
		count int
	)

	// Every reply is primed with `<|start|>assistant<|message|>` and this each
	// completion (vs message) carries an overhead of 3 tokens.
	count += 3

	if len(req.Tools) > 0 {
		// Insert tools into a system prompt. Choose the first system prompt,
		// or if there are none, create one and prepend it.
		var addedTools bool
		for i, message := range req.Messages {
			if message.Role == openai.ChatMessageRoleSystem {
				req.Messages[i].Content = fmt.Sprintf(
					"%s\n\n%s",
					message.Content,
					formatFunctionDefinitions(req.Tools),
				)
				addedTools = true
				break
			}
		}
		if !addedTools {
			req.Messages = append(
				[]openai.ChatCompletionMessage{{
					Role:    openai.ChatMessageRoleSystem,
					Content: formatFunctionDefinitions(req.Tools),
				}},
				req.Messages...,
			)
		}
	}

	for _, message := range req.Messages {
		count += tokensPerReqMessage
		count += c.CountMessageTokens(message)
	}

	// Requests with 2 or more tool messages have a different token count. The
	// reason for this is not yet understood.
	var toolMessages int
	for _, message := range req.Messages {
		if message.Role == openai.ChatMessageRoleTool {
			toolMessages++
		}
	}
	if toolMessages > 1 {
		count += tokensForMultiTool
	}

	return count
}

func (c *Counter) CountRespTokens(
	resp openai.ChatCompletionResponse,
) int {
	var (
		count int
	)

	for _, choice := range resp.Choices {
		count += c.CountMessageTokens(choice.Message)

		// Don't count the role.
		count -= c.CountTokens(choice.Message.Role)

		// Tool calls bump completion token count down by 1.
		if len(choice.Message.ToolCalls) > 0 {
			count -= 1
		}

		// Remove 3 when both content and tools are present.
		if len(choice.Message.ToolCalls) > 0 && choice.Message.Content != "" {
			count -= 3
		}
	}

	return count
}

// CountMessageTokens returns the number of tokens in a single message,
// regardless of it's role/type. This is especially useful for counting the
// tokens in a completion message.
func (c *Counter) CountMessageTokens(
	message openai.ChatCompletionMessage,
) int {
	var (
		count int
	)

	count += c.CountTokens(message.Role)
	if message.Role == openai.ChatMessageRoleTool {
		// Tool content, if it's JSON, is needs to be reformatted into the same
		// JSON style as tool call arguments.
		var contentJSON map[string]interface{}
		if err := json.Unmarshal([]byte(message.Content), &contentJSON); err != nil {
			count += c.CountTokens(message.Content)
		} else {
			stringified, _ := stringifyObject(contentJSON, true)
			count += c.CountTokens(stringified)
		}
	} else {
		count += c.CountTokens(message.Content)
	}

	for _, tc := range message.ToolCalls {
		count += tokensPerToolCall
		count += c.CountTokens(string(tc.Type))
		if tc.Type == openai.ToolTypeFunction {
			count += c.CountTokens(tc.Function.Name) * 2
			args, err := formatArguments(tc.Function.Arguments)
			if err != nil {
				continue
			}
			count += c.CountTokens(args)
		}
	}

	// Add 4 when both content and tools are present.
	if len(message.ToolCalls) > 0 && message.Content != "" {
		count += 4
	}

	if message.Name != "" {
		count += c.CountTokens(message.Name) + tokensPerName
	}

	return count
}

// CountToolTokens returns an estimated number of tokens in the provied set of
// tools. Tools are included in requests differently depending on the contents
// of the request, so this is an estimate.
func (c *Counter) CountToolTokens(tools []openai.Tool) int {
	txt := formatFunctionDefinitions(tools)
	tokens := c.tokenizer.Encode(txt, nil, nil)
	return len(tokens) + 3
}

func formatFunctionDefinitions(tools []openai.Tool) string {
	var lines []string
	lines = append(
		lines,
		"# Tools",
		"## functions",
		"namespace functions {",
	)

	for _, tool := range tools {
		function := tool.Function
		if function.Description != "" {
			lines = append(lines, fmt.Sprintf("// %s", function.Description))
		}

		paramsJSON, _ := json.Marshal(function.Parameters)
		var params map[string]interface{}
		json.Unmarshal(paramsJSON, &params)

		properties, ok := params["properties"].(map[string]interface{})
		if ok && len(properties) > 0 {
			lines = append(lines, fmt.Sprintf("type %s = (_: {", function.Name))
			lines = append(lines, formatObjectProperties(params, 0))
			lines = append(lines, "}) => any;")
		} else {
			lines = append(lines, fmt.Sprintf("type %s = () => any;", function.Name))
		}
	}

	lines = append(
		lines,
		"} // namespace functions",
	)

	return strings.Join(lines, "\n")
}

// formatObjectProperties formats the properties of a JSON object including
// handling of required fields.
func formatObjectProperties(p map[string]interface{}, indent int) string {
	properties, ok := p["properties"].(map[string]interface{})
	if !ok {
		return "" // No properties, return empty string
	}

	required, _ := p["required"].([]interface{})
	requiredFields := make(map[string]bool)
	for _, r := range required {
		if fieldName, ok := r.(string); ok {
			requiredFields[fieldName] = true
		}
	}

	var lines []string
	for key, prop := range properties {
		props, ok := prop.(map[string]interface{})
		if !ok {
			continue // Skip if the property is not a JSON object
		}

		description, _ := props["description"].(string)
		if description != "" {
			lines = append(lines, fmt.Sprintf("// %s", description))
		}

		question := "?"
		if _, isRequired := requiredFields[key]; isRequired {
			question = ""
		}

		formattedType := formatType(props, indent)
		lines = append(lines, fmt.Sprintf("%s%s:%s,", key, question, formattedType))
	}

	indentSpaces := strings.Repeat(" ", indent)
	for i, line := range lines {
		lines[i] = indentSpaces + line
	}

	return strings.Join(lines, "\n")
}

func formatType(props map[string]interface{}, indent int) string {
	typ, ok := props["type"].(string)
	if !ok {
		return ""
	}

	switch typ {
	case "string":
		if enum, ok := props["enum"].([]interface{}); ok {
			var enumValues []string
			for _, val := range enum {
				enumValues = append(enumValues, fmt.Sprintf("\"%s\"", val))
			}
			return strings.Join(enumValues, " | ")
		}
		return "string"

	case "array":
		if items, ok := props["items"].(map[string]interface{}); ok {
			return fmt.Sprintf("%s[]", formatType(items, indent))
		}
		return "any[]"

	case "object":
		if properties, ok := props["properties"].(map[string]interface{}); ok {
			return fmt.Sprintf("{\n%s\n}", formatObjectProperties(properties, indent+2))
		}
		return "{}"

	case "integer", "number":
		if enum, ok := props["enum"].([]interface{}); ok {
			var enumValues []string
			for _, val := range enum {
				enumValues = append(enumValues, fmt.Sprintf("%v", val)) // Assuming all enum values are numbers
			}
			return strings.Join(enumValues, " | ")
		}
		return "number"

	case "boolean":
		return "boolean"

	case "null":
		return "null"

	default:
		return ""
	}
}

// formatArguments formats a JSON string with custom value formatting.
func formatArguments(arguments string) (string, error) {
	var jsonObject map[string]interface{}
	if err := json.Unmarshal([]byte(arguments), &jsonObject); err != nil {
		return "", err
	}
	if len(jsonObject) == 0 {
		return "{}\n", nil
	}

	// Arguments seem to be formatted "typescript" parameter style, without
	// quotes around the keys.
	return stringifyObject(jsonObject, false)
}

func formatField(fieldName string, value interface{}, useQuotes bool) string {
	formattedValue, _ := formatValue(value)
	if useQuotes {
		return fmt.Sprintf("\"%s\": %s", fieldName, formattedValue)
	}
	return fmt.Sprintf("%s:%s", fieldName, formattedValue)
}

// stringifyObject returns a JSON-formatted string representation of the object.
func stringifyObject(jsonObject map[string]interface{}, useQuotes bool) (string, error) {
	if len(jsonObject) == 1 {
		for fieldName, value := range jsonObject {
			return fmt.Sprintf("{%s}\n", formatField(fieldName, value, useQuotes)), nil
		}
	}

	//lines := []string{"{"}
	var lines []string
	properties := []string{}
	for fieldName, value := range jsonObject {
		properties = append(properties, formatField(fieldName, value, useQuotes))
	}
	if len(properties) > 0 {
		lines = append(lines, strings.Join(properties, ", "))
	}
	return fmt.Sprintf("{%s}\n", strings.Join(lines, "\n")), nil
}

// formatValue returns a JSON-formatted string representation of the value.
func formatValue(value interface{}) (string, error) {
	switch v := value.(type) {
	case string, float64, float32, int, int64, int32, int16, int8, uint, uint64, uint32, uint16, uint8:
		return fmt.Sprintf("\"%v\"", v), nil
	case []interface{}:
		elements := []string{}
		for _, element := range v {
			switch e := element.(type) {
			case string, float64, float32, int, int64, int32, int16, int8, uint, uint64, uint32, uint16, uint8:
				elements = append(elements, fmt.Sprintf("\"%v\"", e))
			default:
				elements = append(elements, "\"\"")
			}
		}
		return "[" + strings.Join(elements, ",") + "]", nil
	}
	return "\"\"", nil // Default case for other types or null
}
