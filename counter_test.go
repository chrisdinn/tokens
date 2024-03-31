package tokens

import (
	"context"
	"encoding/json"
	"flag"
	"os"
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var openaiKey string

func TestMain(m *testing.M) {
	flag.StringVar(&openaiKey, "openai-key", "", "API key for testing")
	flag.Parse()

	os.Exit(m.Run())
}

var tests = []struct {
	name string
	in   openai.ChatCompletionRequest
	// wantCached allows us to notice when the model itself has changed.
	wantCached int
}{{
		name: "Single system message",
		in: openai.ChatCompletionRequest{
			Messages: []openai.ChatCompletionMessage{{
				Role:    openai.ChatMessageRoleSystem,
				Content: "This is a system message.",
			}},
		},
		wantCached: 13,
	}, {
		name: "System message and user message",
		in: openai.ChatCompletionRequest{
			Messages: []openai.ChatCompletionMessage{{
				Role:    openai.ChatMessageRoleSystem,
				Content: "This is a system message.",
			}, {
				Role:    openai.ChatMessageRoleUser,
				Content: "This is a user message.",
			}},
		},
		wantCached: 23,
	}, {
		name: "Assistant message no tools",
		in: openai.ChatCompletionRequest{
			Messages: []openai.ChatCompletionMessage{{
				Role:    openai.ChatMessageRoleAssistant,
				Content: "This is an assistant message.",
			}},
		},
		wantCached: 13,
	}, {
		name: "User message with name",
		in: openai.ChatCompletionRequest{
			Messages: []openai.ChatCompletionMessage{{
				Role:    openai.ChatMessageRoleSystem,
				Content: "This is a system message.",
			}, {
				Role:    openai.ChatMessageRoleUser,
				Content: "This is a user message.",
				Name:    "Chris",
			}},
		},
		wantCached: 25,
	}, {
		name: "User message without name",
		in: openai.ChatCompletionRequest{
			Messages: []openai.ChatCompletionMessage{{
				Role:    openai.ChatMessageRoleSystem,
				Content: "This is a system message.",
			}, {
				Role:    openai.ChatMessageRoleUser,
				Content: "This is a user message.",
			}},
		},
		wantCached: 23,
	}, {
	name: "User message with one tool",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Killington this weekend.",
		}},
		Tools: []openai.Tool{{
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
							Enum: []string{"celsius", "fahrenheit"},
						},
					},
					Required: []string{"location"},
				},
			},
		}},
	},
	wantCached: 84,
}, {
	name: "System and user message with one tool",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a well-respected meteorologist.",
		}, {
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Killington this weekend.",
		}},
		Tools: []openai.Tool{{
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
					},
					Required: []string{"location"},
				},
			},
		}},
	},
	wantCached: 94,
}, {
	name: "Request with two tools",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a well-respected meteorologist.",
		}, {
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Killington this weekend.",
		}},
		Tools: []openai.Tool{{
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
		}, {
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "integer_enum_example",
				Description: "An example function that takes an integer enum.",
				Parameters: json.RawMessage(`{
	  "type": "object",
	  "properties": {
	      "integer_enum": {
	          "type": "integer",
			  "enum": [1, 2, 3]
	      }
	  },
	  "required": ["integer_enum"]
	}`),
			},
		}},
	},
	wantCached: 141,
}, {
	name: "System and user message with tools",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a well-respected meteorologist.",
		}, {
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Killington this weekend.",
		}},
		Tools: []openai.Tool{{
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
					},
					Required: []string{"location"},
				},
			},
		}},
	},
	wantCached: 94,
}, {
	name: "Assistant message with tool call then tool message",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Breckenridge this weekend.",
		}, {
			Role:    openai.ChatMessageRoleAssistant,
			Content: "I can help with that.",
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240327",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Breckenridge, CO\"}",
				},
			}},
		}, {
			Role:       openai.ChatMessageRoleTool,
			Content:    "The weather in Breckenridge, CO is 38 degrees.",
			ToolCallID: "testcall_20240327",
		}},
	},
	wantCached: 70,
}, {
	name: "Assistant message with tool call then 1 tool content property",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Vail this weekend.",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240327",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Vail, CO\"}",
				},
			}},
		}, {
			Role:       openai.ChatMessageRoleTool,
			Content:    "{\"temperature\": \"35\"}",
			ToolCallID: "testcall_20240327",
		}},
	},
	wantCached: 50,
}, {
	name: "Assistant message with tool call then 2 tool content properties",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Park City this weekend.",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240330",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Park City, UT\"}",
				},
			}},
		}, {
			Role:       openai.ChatMessageRoleTool,
			Content:    "{\"location\": \"Park City, UT\", \"temperature\": \"45\"}",
			ToolCallID: "testcall_20240330",
		}},
	},
	wantCached: 59,
}, {
	name: "Assistant message with tool call then 3 tool content properties",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Whistler this weekend.",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240330",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Whistler, BC\"}",
				},
			}},
		}, {
			Role:       openai.ChatMessageRoleTool,
			Content:    "{\"location\": \"Whistler, BC\", \"format\": \"fahrenheit\", \"temperature\": \"45\"}",
			ToolCallID: "testcall_20240330",
		}},
	},
	wantCached: 69,
}, {
	name: "Assistant message with JSON tool with no content and no args",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "What's the weather?",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240328",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{}",
				},
			}},
		}, {
			Role:       openai.ChatMessageRoleTool,
			Content:    "{}",
			ToolCallID: "testcall_20240328",
		}},
	},
	wantCached: 33,
}, {
	name: "Assistant message with JSON tool content and no args",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "What's the weather?",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240328",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{}",
				},
			}},
		}, {
			Role:       openai.ChatMessageRoleTool,
			Content:    "{\"temperature\": \"45\"}",
			ToolCallID: "testcall_20240328",
		}},
	},
	wantCached: 38,
}, {
	name: "Assistant message with JSON tool content and one args",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Killington this weekend.",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240328",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Killington, VT\"}",
				},
			}},
		}, {
			Role:    openai.ChatMessageRoleTool,
			Content: "{\"location\": \"Killington, VT\", \"format\": \"fahrenheit\", \"temperature\": \"45\"}",

			ToolCallID: "testcall_20240328",
		}},
	},
	wantCached: 66,
}, {
	name: "Assistant message with JSON tool content and two args",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Killington this weekend.",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240328",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Killington, VT\", \"unit\": \"celsius\"}",
				},
			}},
		}, {
			Role:    openai.ChatMessageRoleTool,
			Content: "{\"location\": \"Killington, VT\", \"format\": \"fahrenheit\", \"temperature\": \"45\"}",

			ToolCallID: "testcall_20240328",
		}},
	},
	wantCached: 71,
}, {
	name: "Assistant message with JSON tool content and three args",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "I want to ski at Killington this weekend.",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240328",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Killington, VT\", \"unit\": \"celsius\", \"date\": \"2024-03-28\"}",
				},
			}},
		}, {
			Role:    openai.ChatMessageRoleTool,
			Content: "{\"location\": \"Killington, VT\", \"format\": \"fahrenheit\", \"temperature\": \"45\"}",

			ToolCallID: "testcall_20240328",
		}},
	},
	wantCached: 80,
}, {
	name: "Assistant message with two tool call then two tool messages",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "Should I ski at Killington or Tremblant this weekend?",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240328_A",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Killington, VT\"}",
				},
			}, {
				ID:   "testcall_20240328_B",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Mount Tremblant, QC\"}",
				},
			}},
		}, {
			Role:       openai.ChatMessageRoleTool,
			Content:    "The weather in Killington, VT is 45 degrees.",
			ToolCallID: "testcall_20240328_A",
		}, {
			Role:       openai.ChatMessageRoleTool,
			Content:    "The weather at Mount Tremblant, QC is 32 degrees.",
			ToolCallID: "testcall_20240328_B",
		}},
	},
	wantCached: 111,
}, {
	name: "Assistant message with two JSON tool content messages",
	in: openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: "Should I ski at Killington or Tremblant this weekend?",
		}, {
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{{
				ID:   "testcall_20240328_A",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Killington, VT\", \"unit\": \"fahrenheit\", \"date\": \"2024-03-28\"}",
				},
			}, {
				ID:   "testcall_20240328_B",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_current_weather",
					Arguments: "{\"location\": \"Mount Tremblant, QC\", \"unit\": \"celsius\", \"date\": \"2024-03-28\"}",
				},
			}},
		}, {
			Role:    openai.ChatMessageRoleTool,
			Content: "{\"location\": \"Killington, VT\", \"format\": \"fahrenheit\", \"temperature\": \"45\"}",

			ToolCallID: "testcall_20240328_A",
		}, {
			Role:       openai.ChatMessageRoleTool,
			Content:    "{\"location\": \"Mount Tremblant, QC\", \"format\": \"celsius\", \"temperature\": \"-1\"}",
			ToolCallID: "testcall_20240328_B",
		}},
	},
	wantCached: 159,
}}

func TestCountRequestTokens(t *testing.T) {
	client := openai.NewClient(openaiKey)
	counter, err := NewCounter("gpt-4-1106-preview")
	if err != nil {
		t.Fatalf("NewCounter: %v", err)
	}
	for _, tt := range tests {
		req := tt.in
		req.Model = openai.GPT4TurboPreview
		req.MaxTokens = 150
		req.Temperature = 0.0

		resp, err := client.CreateChatCompletion(context.Background(), req)
		if err != nil {
			t.Fatalf("%s: CreateChatCompletion: %v", tt.name, err)
		}

		gotPromptTokens := counter.CountRequestTokens(tt.in)
		wantPromptTokens := resp.Usage.PromptTokens

		if gotPromptTokens != wantPromptTokens {
			t.Errorf(
				"%s: token count got %d, want %d, diff %d",
				tt.name,
				gotPromptTokens,
				wantPromptTokens,
				gotPromptTokens-wantPromptTokens,
			)
		}

		if wantPromptTokens != tt.wantCached {
			t.Errorf(
				"%s: cached token count got %d, want %d, diff %d",
				tt.name,
				tt.wantCached,
				wantPromptTokens,
				tt.wantCached-wantPromptTokens,
			)
		}

		gotCompletionTokens := counter.CountRespTokens(resp)
		wantCompletionTokens := resp.Usage.CompletionTokens

		respJSON, _ := json.Marshal(resp)

		if gotCompletionTokens != wantCompletionTokens {
			t.Errorf(
				"%s: completion token count got %d, want %d, diff %d\n%s",
				tt.name,
				gotCompletionTokens,
				wantCompletionTokens,
				gotCompletionTokens-wantCompletionTokens,
				string(respJSON),
			)
		}

	}
}
