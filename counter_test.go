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

func TestCountRequestTokens(t *testing.T) {
	tests := []struct {
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
		name: "User message with one tool - A",
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
						},
						Required: []string{"location"},
					},
				},
			}},
		},
		wantCached: 84,
	}, {
		name: "User message with one tool - tool choice",
		in: openai.ChatCompletionRequest{
			Messages: []openai.ChatCompletionMessage{{
				Role:    openai.ChatMessageRoleUser,
				Content: "I want to ski at either Killington or Vail this weekend.",
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
			ToolChoice: openai.ToolChoice{
				Type: openai.ToolTypeFunction,
				Function: openai.ToolFunction{
					Name: "get_current_weather",
				},
			},
		},
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
		//		name: "Assistant message with JSON tool content and one args",
		//		in: openai.ChatCompletionRequest{
		//			Messages: []openai.ChatCompletionMessage{{
		//				Role:    openai.ChatMessageRoleUser,
		//				Content: "I want to ski at Killington this weekend.",
		//			}, {
		//				Role: openai.ChatMessageRoleAssistant,
		//				ToolCalls: []openai.ToolCall{{
		//					ID:   "testcall_20240328",
		//					Type: openai.ToolTypeFunction,
		//					Function: openai.FunctionCall{
		//						Name:      "get_current_weather",
		//						Arguments: "{\"location\": \"Killington, VT\"}",
		//					},
		//				}},
		//			}, {
		//				Role:    openai.ChatMessageRoleTool,
		//				Content: "{\"location\": \"Killington, VT\", \"format\": \"fahrenheit\", \"temperature\": \"45\"}",
		//
		//				ToolCallID: "testcall_20240328",
		//			}},
		//		},
		//		wantCached: 66,
		//	}, {
		//		name: "Assistant message with JSON tool content and two args",
		//		in: openai.ChatCompletionRequest{
		//			Messages: []openai.ChatCompletionMessage{{
		//				Role:    openai.ChatMessageRoleUser,
		//				Content: "I want to ski at Killington this weekend.",
		//			}, {
		//				Role: openai.ChatMessageRoleAssistant,
		//				ToolCalls: []openai.ToolCall{{
		//					ID:   "testcall_20240328",
		//					Type: openai.ToolTypeFunction,
		//					Function: openai.FunctionCall{
		//						Name:      "get_current_weather",
		//						Arguments: "{\"location\": \"Killington, VT\", \"unit\": \"celsius\"}",
		//					},
		//				}},
		//			}, {
		//				Role:    openai.ChatMessageRoleTool,
		//				Content: "{\"location\": \"Killington, VT\", \"format\": \"fahrenheit\", \"temperature\": \"45\"}",
		//
		//				ToolCallID: "testcall_20240328",
		//			}},
		//		},
		//		wantCached: 71,
		//	}, {
		//		name: "Assistant message with JSON tool content and three args",
		//		in: openai.ChatCompletionRequest{
		//			Messages: []openai.ChatCompletionMessage{{
		//				Role:    openai.ChatMessageRoleUser,
		//				Content: "I want to ski at Killington this weekend.",
		//			}, {
		//				Role: openai.ChatMessageRoleAssistant,
		//				ToolCalls: []openai.ToolCall{{
		//					ID:   "testcall_20240328",
		//					Type: openai.ToolTypeFunction,
		//					Function: openai.FunctionCall{
		//						Name:      "get_current_weather",
		//						Arguments: "{\"location\": \"Killington, VT\", \"unit\": \"celsius\", \"date\": \"2024-03-28\"}",
		//					},
		//				}},
		//			}, {
		//				Role:    openai.ChatMessageRoleTool,
		//				Content: "{\"location\": \"Killington, VT\", \"format\": \"fahrenheit\", \"temperature\": \"45\"}",
		//
		//				ToolCallID: "testcall_20240328",
		//			}},
		//		},
		//		wantCached: 80,
		//	}, {
		//		name: "Assistant message with two tool call then two tool messages",
		//		in: openai.ChatCompletionRequest{
		//			Messages: []openai.ChatCompletionMessage{{
		//				Role:    openai.ChatMessageRoleUser,
		//				Content: "Should I ski at Killington or Tremblant this weekend?",
		//			}, {
		//				Role: openai.ChatMessageRoleAssistant,
		//				ToolCalls: []openai.ToolCall{{
		//					ID:   "testcall_20240328_A",
		//					Type: openai.ToolTypeFunction,
		//					Function: openai.FunctionCall{
		//						Name:      "get_current_weather",
		//						Arguments: "{\"location\": \"Killington, VT\"}",
		//					},
		//				}, {
		//					ID:   "testcall_20240328_B",
		//					Type: openai.ToolTypeFunction,
		//					Function: openai.FunctionCall{
		//						Name:      "get_current_weather",
		//						Arguments: "{\"location\": \"Mount Tremblant, QC\"}",
		//					},
		//				}},
		//			}, {
		//				Role:       openai.ChatMessageRoleTool,
		//				Content:    "The weather in Killington, VT is 45 degrees.",
		//				ToolCallID: "testcall_20240328_A",
		//			}, {
		//				Role:       openai.ChatMessageRoleTool,
		//				Content:    "The weather at Mount Tremblant, QC is 32 degrees.",
		//				ToolCallID: "testcall_20240328_B",
		//			}},
		//		},
		//		wantCached: 111,
		//	}, {
		//		name: "Assistant message with two JSON tool content messages",
		//		in: openai.ChatCompletionRequest{
		//			Messages: []openai.ChatCompletionMessage{{
		//				Role:    openai.ChatMessageRoleUser,
		//				Content: "Should I ski at Killington or Tremblant this weekend?",
		//			}, {
		//				Role: openai.ChatMessageRoleAssistant,
		//				ToolCalls: []openai.ToolCall{{
		//					ID:   "testcall_20240328_A",
		//					Type: openai.ToolTypeFunction,
		//					Function: openai.FunctionCall{
		//						Name:      "get_current_weather",
		//						Arguments: "{\"location\": \"Killington, VT\", \"unit\": \"fahrenheit\", \"date\": \"2024-03-28\"}",
		//					},
		//				}, {
		//					ID:   "testcall_20240328_B",
		//					Type: openai.ToolTypeFunction,
		//					Function: openai.FunctionCall{
		//						Name:      "get_current_weather",
		//						Arguments: "{\"location\": \"Mount Tremblant, QC\", \"unit\": \"celsius\", \"date\": \"2024-03-28\"}",
		//					},
		//				}},
		//			}, {
		//				Role:    openai.ChatMessageRoleTool,
		//				Content: "{\"location\": \"Killington, VT\", \"format\": \"fahrenheit\", \"temperature\": \"45\"}",
		//
		//				ToolCallID: "testcall_20240328_A",
		//			}, {
		//				Role:       openai.ChatMessageRoleTool,
		//				Content:    "{\"location\": \"Mount Tremblant, QC\", \"format\": \"celsius\", \"temperature\": \"-1\"}",
		//				ToolCallID: "testcall_20240328_B",
		//			}},
		//		},
		//		wantCached: 159,
		//	}, {
		name: "Problem example from testing",
		in: openai.ChatCompletionRequest{
			Messages: []openai.ChatCompletionMessage{{
				Role:    openai.ChatMessageRoleSystem,
				Content: "You are a coy but friendly Star Fleet intelligence officer charged with helping\nthe user train and built autonomous LLM age\nnts. You are an agent of few words\nbut you insist on making those words count.\n\nYou are still learning the ropes, so you may not be able to\nanswer all\nquestions. Be patient with users who find that frustrating.\n\n\nImagine you're writing directly in a Slack channel, where the Markdown\nformatting rules are unique. In Slack, bold text must be created with\n*asterisks* not **. It's crucial to adhere strictly to Slack's\ncustom Markdown\nstyle for compatibility reasons. Please respond with text formatted using\nSlack's guidelines: *bold*, _italic_, ~strikethrough~, <URL|link text>, code\nwith backticks, and code blocks with triple backticks. Use :emoji: for emojis.\nRemember, Slack's format is essential here, differing from GitHub or\ntraditional Markdown. Your adherence to these specifics ensures messages\ndisplay correctly in Slack environments.\n\nToday is Saturday, Jun. 22, 2024.\n\n\nA user has joined the conversation, their message is below.\n\n\n\n",
			}, {
				Role:    openai.ChatMessageRoleUser,
				Content: "<@U06L3JP9PEU> What time does the first train leave Belleville for Toronto Monday morning?",
				Name:    "Chris",
			}, {
				Role: openai.ChatMessageRoleAssistant,
				ToolCalls: []openai.ToolCall{{
					ID:   "call_NMLeg8JBhzvdr2Q0YfyJAywC",
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      "web_search",
						Arguments: "{\"query\":\"first train Belleville to Toronto Monday morning schedule\"}",
					},
				}},
			}, {
				Role:       openai.ChatMessageRoleTool,
				Content:    "{\"searchParameters\":{\"q\":\"first train Belleville to Toronto Monday morning schedule\",\"type\":\"search\",\"engine\":\"google\"},\"organic\":[{\"title\":\"Train Belleville - Toronto prices from $39.63 - Virail\",\"link\":\"https://www.virail.com/train-belleville-toronto\",\"snippet\":\"The journey from Belleville to Toronto by train is 106.31 mi and takes 2 hr 37 min. There are 11 connections per day, with the first departure at 8:16 AM and ...\",\"position\":1},{\"title\":\"Train from Belleville to Toronto - VIA Rail Canada - Busbud\",\"link\":\"https://www.busbud.com/en/train-belleville-toronto/t/drbgr0-dpz88g\",\"snippet\":\"Train from Belleville to Toronto: Find schedules, Compare prices & Book VIA Rail Canada tickets.\",\"attributes\":{\"Missing\":\"morning | Show results with:morning\"},\"rating\":4.6,\"ratingCount\":12,\"currency\":\"$\",\"price\":29,\"position\":2},{\"title\":\"Belleville train station | VIA Rail\",\"link\":\"https://www.viarail.ca/en/explore-our-destinations/stations/ontario/belleville\",\"snippet\":\"Opening Hours. Station. Monday Tuesday Wednesday Thursday. 06h30 to 21h00. Friday Saturday Sunday. 07h45 to 21h00 ...\",\"attributes\":{\"Missing\":\"first | Show results with:first\"},\"position\":3},{\"title\":\"Train Schedule: Toronto-Ottawa-Montréal | VIA Rail\",\"link\":\"https://www.viarail.ca/en/plan/train-schedules/toronto-ottawa-montreal\",\"snippet\":\"Train Schedule: Toronto - Ottawa/Montréal · 10:53. Departure 10:57 · 14:05. Departure 14:10 · 14:58. Departure 15:01 ...\",\"position\":4},{\"title\":\"Belleville, ON to Toronto, ON train tickets from $28 (€24) - Omio\",\"link\":\"https://www.omio.com/trains/belleville-on/toronto-on\",\"snippet\":\"The first train from Belleville, ON to Toronto, ON leaves at 6: 36 PM. Plan your trip with the Journey Planner from Omio. What time does the last train from ...\",\"attributes\":{\"Missing\":\"morning | Show results with:morning\"},\"currency\":\"€\",\"price\":24,\"position\":5},{\"title\":\"Train Toronto - Belleville prices from $39.57 - Virail\",\"link\":\"https://www.virail.com/train-toronto-belleville\",\"snippet\":\"The first daily departure from Toronto to Belleville leaves at 7:07 AM, while the last journey of the day sets out at 7:50 PM. These are according to the ...\",\"position\":6},{\"title\":\"Belleville → Toronto Bus: from $17 | FlixBus, Rider Express, Megabus\",\"link\":\"https://www.busbud.com/en-ca/bus-belleville-toronto/r/drbgr0-dpz88g\",\"snippet\":\"Book your next bus ticket from Belleville to Toronto. Find schedules and the best prices online with Busbud. Enjoy your trip with FlixBus, Book A Ride, ...\",\"rating\":3.6,\"ratingCount\":140,\"currency\":\"$\",\"price\":17,\"position\":7},{\"title\":\"Via Rail unveils new early morning train, daily service between ...\",\"link\":\"https://ottawa.ctvnews.ca/via-rail-unveils-new-early-morning-train-daily-service-between-toronto-ottawa-1.6865986\",\"snippet\":\"The railway company says the new 641 train will leave Ottawa at 4:19 a.m. and arrive in Toronto at 8:48 a.m. The service will begin on May 27 ...\",\"date\":\"Apr 29, 2024\",\"position\":8},{\"title\":\"Belleville to Toronto Train - Tickets from $31 | Wanderu\",\"link\":\"https://www.wanderu.com/en-ca/train/ca-on/belleville/ca-on/toronto/\",\"snippet\":\"We respond within minutes to help you out. Belleville - Toronto Train Schedule. WedJun 19. ThuJun 20. FriJun 21. SatJun 22. SunJun 23. MonJun 24. TueJun 25.\",\"attributes\":{\"Missing\":\"morning | Show results with:morning\"},\"position\":9},{\"title\":\"VIA Rail Canada - Early birds, rejoice! We're pleased... - Facebook\",\"link\":\"https://m.facebook.com/viarailcanada/posts/847588387413026/\",\"snippet\":\"Early birds, rejoice! We're pleased to announce the launch of the new 641 early-morning departure which will operate between\",\"date\":\"Apr 29, 2024\",\"position\":10}],\"peopleAlsoAsk\":[{\"question\":\"How much is a train ticket from Belleville to Toronto?\",\"snippet\":\"It is possible to travel from Belleville to Toronto by train for as little as $50.63 or as much as $681.73. The best price for this journey is $50.63.\",\"title\":\"Trains Belleville - Toronto: times, prices and tickets starting from $50.63\",\"link\":\"https://www.virail.ca/train-belleville-toronto\"},{\"question\":\"How early to get to train station Canada?\",\"snippet\":\"We are recommending travellers to be at the station 45 minutes prior to departure if you are travelling in the Corridor and 1 hour prior for the long distance and regional services (90 minutes prior to departure for trains 1 and 2 out of Toronto and Vancouver).\",\"title\":\"How early must I check-in for departure? - VIA Rail\",\"link\":\"https://www.viarail.ca/en/plan/faq/plan-your-trip/how-early-must-i-check-in-departure\"},{\"question\":\"Does Belleville have a via rail station?\",\"snippet\":\"The Belleville railway station in Belleville, Ontario, Canada is served by Via Rail trains running from Toronto to Ottawa and Montreal. The station is staffed, with ticket sales, vending machines, telephones, washrooms, and wheelchair access to the station and trains.\",\"title\":\"Belleville station (Ontario) - Wikipedia\",\"link\":\"https://en.wikipedia.org/wiki/Belleville_station_(Ontario)\"},{\"question\":\"How much is a train ticket from Belleville to Montreal?\",\"snippet\":\"Daily Trains\\n16\\nEarliest and Latest Train Departures\\n8:36AM - 6:48PM\\nMinimum Price\\n$82\\nAverage Ticket Price\\n$124\\nMinimum Trip Duration\\n2h53m\",\"title\":\"Train from Belleville to Montreal - VIA Rail Canada - Busbud\",\"link\":\"https://www.busbud.com/en-ca/train-belleville-montreal/t/drbgr0-f25dvk\"}],\"relatedSearches\":[{\"query\":\"Via rail first train belleville to toronto monday morning schedule\"},{\"query\":\"First train belleville to toronto monday morning schedule pdf\"},{\"query\":\"First train belleville to toronto monday morning schedule price\"},{\"query\":\"VIA train Belleville to Toronto schedule\"},{\"query\":\"VIA Rail Belleville to Toronto\"},{\"query\":\"VIA Rail Belleville to Toronto schedule price\"},{\"query\":\"Train from Belleville to Toronto Airport\"},{\"query\":\"Belleville to Toronto bus\"}]}",
				ToolCallID: "call_NMLeg8JBhzvdr2Q0YfyJAywC",
			}},
		},
	}}

	//2024/06/22 10:08:44 - tokenCounter prompt: got token count 1966, estimated 480, delta: 1486

	client := openai.NewClient(openaiKey)
	models := []string{
		openai.GPT4o,
		//	openai.GPT4Turbo,
	}

	for _, model := range models {
		useModel := model
		counter, err := NewCounter(model)
		if err != nil {
			t.Fatalf("NewCounter: %v", err)
		}
		for _, tt := range tests {
			req := tt.in
			req.Model = useModel
			req.MaxTokens = 150
			req.Temperature = 0.0

			resp, err := client.CreateChatCompletion(context.Background(), req)
			if err != nil {
				t.Fatalf("%s: CreateChatCompletion: %v", tt.name, err)
			}

			gotPromptTokens := counter.CountRequestTokens(tt.in)
			wantPromptTokens := resp.Usage.PromptTokens

			if gotPromptTokens != wantPromptTokens {
				reqJSON, _ := json.MarshalIndent(req, "", "  ")
				t.Errorf(
					"%s - %s: prompt token count got %d, want %d, diff %d\n%s",
					tt.name,
					model,
					gotPromptTokens,
					wantPromptTokens,
					gotPromptTokens-wantPromptTokens,
					string(reqJSON),
				)
			}

			//			if wantPromptTokens != tt.wantCached {
			//				t.Errorf(
			//					"%s - %s: cached token count got %d, want %d, diff %d",
			//					tt.name,
			//					model,
			//					tt.wantCached,
			//					wantPromptTokens,
			//					tt.wantCached-wantPromptTokens,
			//				)
			//			}

			//			gotCompletionTokens := counter.CountResponseTokens(resp)
			//			wantCompletionTokens := resp.Usage.CompletionTokens
			//			if req.ToolChoice != nil {
			//				// When we specify tool choice, we get these completion tokens
			//				// credited.
			//				wantCompletionTokens += counter.countToolChoice(req.ToolChoice)
			//			}
			//
			//			if gotCompletionTokens != wantCompletionTokens {
			//				t.Errorf(
			//					"%s - %s: completion token count got %d, want %d, diff %d",
			//					tt.name,
			//					model,
			//					gotCompletionTokens,
			//					wantCompletionTokens,
			//					gotCompletionTokens-wantCompletionTokens,
			//				)
			//				respJSON, _ := json.MarshalIndent(resp, "", "  ")
			//				t.Log(string(respJSON))
			//			}
		}
	}
}

//func TestCountResponseTokens(t *testing.T) {
//	tests := []struct {
//		name  string
//		model string
//		in    openai.ChatCompletionResponse
//		want  int
//	}{{
//		name:  "Single complete message",
//		model: "gpt-4o-2024-05-13",
//		in: openai.ChatCompletionResponse{
//			ID:      "chatcmpl-9dJ4AhT4Nw5Z5gqDfjvw1ZNFo96YA",
//			Object:  "chat.completion",
//			Created: 1719155110,
//			Model:   "gpt-4o-2024-05-13",
//			Choices: []openai.ChatCompletionChoice{{
//				Index: 0,
//				Message: openai.ChatCompletionMessage{
//					Role:    openai.ChatMessageRoleAssistant,
//					Content: "That sounds like a fun plan! To help you prepare, it's important to check the current weather conditions at Killington, VT. Would you like me to get the current weather information for you?",
//				},
//				FinishReason: "stop",
//			}},
//			Usage: openai.Usage{
//				PromptTokens:     71,
//				CompletionTokens: 40,
//				TotalTokens:      111,
//			},
//			SystemFingerprint: "fp_5e6c71d4a8",
//		},
//		want: 40,
//	}, {
//		name:  "Single complete message with tool call",
//		model: "gpt-4o-2024-05-13",
//		in: openai.ChatCompletionResponse{
//			ID:      "chatcmpl-9dTOqwCsAwEKCL1NywpjNrydfgTnD",
//			Object:  "chat.completion",
//			Created: 1719194832,
//			Model:   "gpt-4o-2024-05-13",
//			Choices: []openai.ChatCompletionChoice{{
//				Index: 0,
//				Message: openai.ChatCompletionMessage{
//					Role:    openai.ChatMessageRoleAssistant,
//					Content: "Let's check the current weather at Killington, VT to help you decide if skiing this weekend is viable.",
//					ToolCalls: []openai.ToolCall{{
//						ID:   "call_XtkMPwjzUOnjvQYDbiQPi9ST",
//						Type: openai.ToolTypeFunction,
//						Function: openai.FunctionCall{
//							Name:      "get_current_weather",
//							Arguments: "{\"location\":\"Killington, VT\"}",
//						},
//					}},
//				},
//				FinishReason: "tool_calls",
//			}},
//			Usage: openai.Usage{
//				PromptTokens:     71,
//				CompletionTokens: 40,
//				TotalTokens:      111,
//			},
//			SystemFingerprint: "fp_3e7d703517",
//		},
//		want: 40,
//	}, {
//		name:  "Single complete message with tool call - variant",
//		model: "gpt-4o-2024-05-13",
//		in: openai.ChatCompletionResponse{
//			ID:      "chatcmpl-9dTXl2qLHZ7eqgq4cWjdq3sBL1Ujh",
//			Object:  "chat.completion",
//			Created: 1719195385,
//			Model:   "gpt-4o-2024-05-13",
//			Choices: []openai.ChatCompletionChoice{{
//				Index: 0,
//				Message: openai.ChatCompletionMessage{
//					Role:    openai.ChatMessageRoleAssistant,
//					Content: "That sounds like a lot of fun! Before planning your ski trip, let's check the weather at Killington, VT for this weekend.",
//					ToolCalls: []openai.ToolCall{{
//						ID:   "call_asoYc09Lgcm6K0HGHDgI4ECd",
//						Type: openai.ToolTypeFunction,
//						Function: openai.FunctionCall{
//							Name:      "get_current_weather",
//							Arguments: "{\"location\": \"Killington, VT\"}",
//						},
//					}},
//				},
//				FinishReason: "tool_calls",
//			}},
//			Usage: openai.Usage{
//				PromptTokens:     71,
//				CompletionTokens: 62,
//				TotalTokens:      133,
//			},
//			SystemFingerprint: "fp_3e7d703517",
//		},
//		want: 62,
//	}, {
//		name:  "Single complete message with two tool calls",
//		model: "gpt-4o-2024-05-13",
//		in: openai.ChatCompletionResponse{
//			ID:      "chatcmpl-9dJ4B30gKE8br1vjbqTxeHnSe3RRV",
//			Object:  "chat.completion",
//			Created: 1719155111,
//			Model:   "gpt-4o-2024-05-13",
//			Choices: []openai.ChatCompletionChoice{{
//				Index: 0,
//				Message: openai.ChatCompletionMessage{
//					Role:    openai.ChatMessageRoleAssistant,
//					Content: "I'll get the current weather for both Killington, VT, and Vail, CO to help you decide where to ski this weekend.",
//					ToolCalls: []openai.ToolCall{{
//						ID:   "call_XJVrmo98o69BpRDldhLqRCvi",
//						Type: openai.ToolTypeFunction,
//						Function: openai.FunctionCall{
//							Name:      "get_current_weather",
//							Arguments: "{\"location\": \"Killington, VT\", \"unit\": \"fahrenheit\"}",
//						},
//					}, {
//						ID:   "call_5n958M9taJkwvTFLidR5e29S",
//						Type: openai.ToolTypeFunction,
//						Function: openai.FunctionCall{
//							Name:      "get_current_weather",
//							Arguments: "{\"location\": \"Vail, CO\", \"unit\": \"fahrenheit\"}",
//						},
//					}},
//				},
//				FinishReason: "tool_calls",
//			}},
//			Usage: openai.Usage{
//				PromptTokens:     86,
//				CompletionTokens: 90,
//				TotalTokens:      176,
//			},
//			SystemFingerprint: "fp_888385ccad",
//		},
//		want: 90,
//	}, {
//		name:  "Simple assistant message",
//		model: "gpt-4o-2024-05-13",
//		in: openai.ChatCompletionResponse{
//			ID:      "chatcmpl-9dTiR4eT5KtboJ1M1O15AKKXTefan",
//			Object:  "chat.completion",
//			Created: 1719196047,
//			Model:   "gpt-4o-2024-05-13",
//			Choices: []openai.ChatCompletionChoice{{
//				Index: 0,
//				Message: openai.ChatCompletionMessage{
//					Role:    openai.ChatMessageRoleAssistant,
//					Content: "How can I assist you today?",
//				},
//				FinishReason: "stop",
//			}},
//			Usage: openai.Usage{
//				PromptTokens:     13,
//				CompletionTokens: 7,
//				TotalTokens:      20,
//			},
//			SystemFingerprint: "fp_5e6c71d4a8",
//		},
//		want: 7,
//	}}
//
//	for _, tt := range tests {
//		counter, err := NewCounter(tt.model)
//		if err != nil {
//			t.Fatalf("NewCounter: %v", err)
//		}
//
//		got := counter.CountResponseTokens(tt.in)
//		if got != tt.want {
//			t.Errorf(
//				"%s - %s: got %d, want %d, diff %d",
//				tt.name,
//				tt.model,
//				got,
//				tt.want,
//				got-tt.want,
//			)
//		}
//	}
//}
