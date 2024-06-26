# Tokens

    go get github.com/chrisdinn/tokens

Tokens is a simple package for counting tokens in an OpenAI
ChatCompletionRequest.

It's important to know how many tokens you're using because that how OpenAI
charges for access to their models. 

## Why do I need to count my own tokens

OpenAI provides token counts for prompts and completions on synchronous calls
to Create ChatCompletion endpoint as the usage parameter. However, they do not
provide those totals for streaming calls to the same endpoint. To count tokens
for a streaming request, at least for now, you need to do it yourself.

## How does it work?

This package uses [tiktoken-go](https://github.com/pkoukk/tiktoken-go) for
tokenization. To count tokens accurately you need to recreated the strategy
used but not documented by OpenAI. Accuracy is derived from a few key insights:

- Tools are rendered as typescript functions with a very specific format.
- Tool call arguments are rendered in typescript args format (no quotes around
argument keys).
- Tool messages are rendered as stringified, indented JSON (yes quotes around
argument keys).
- Role isn't counted for completion messages.

There are still open questions:

- Why does more than one tool message add 13 unaccounted for tokens?

## Usage

```go
package main

import (
	"fmt"
	"github.com/chrisdinn/tokens"
)

func main() {
	// Create a new token counter
	tc := tokens.NewCounter()

	// Count tokens in a string.
	someString = "You string for/from model goes here."
	fmt.Printf("%q: %d\n", someString, tc.CountTokens(someString))


	// Count tokens in a ChatCompletionRequest.
	req := openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleSystem,
			Content: "This is a system message.",
		}, {
			Role:    openai.ChatMessageRoleUser,
			Content: "This is a user message.",
		}},
	},
	fmt.Printf("Req tokens: %d\n", req, tc.CountRequestTokens(req))

	// Count tokens in a ChatCompletionMessage.
	msg := openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleAssistant,
		ToolCalls: []openai.ToolCall{{
			ID:   "testcall_20240330",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "get_current_weather",
				Arguments: "{\"location\": \"Park City, UT\"}",
			},
		}},
	}
	fmt.Printf("Msg tokens: %d\n", msg, tc.CountMessageTokens(msg))

	// Count tool tokens on their own.
	tools := []openai.Tool{{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "get_current_weather",
			Description: "Get the current weather in a given location.",
			Parameters: jsonschema.Definition{
				Type: jsonschema.Object,
				Properties: map[string]jsonschema.Definition{
					"location": {
						Type:        jsonschema.String,
						Description: "The city and state, e.g. San Francisco, CA",
					},
					"unit": {
						Type: jsonschema.String,
						Enum: []string{"celcius", "fahrenheit"},
					},
					"date": {
						Type:        jsonschema.String,
						Description: "The date for which to get the weather.",
					},
				},
				Required: []string{"location"},
			},
		},
	}},
	fmt.Printf("Tools tokens: %d\n", tools, tc.CountToolTokens(tools))

}
```


