# SmolAgent Height Comparison

A conversational agent built with [Huggingface Smolagents](https://github.com/huggingface/smolagents) that takes a user's height query, dynamically finds comparable famous/fictional figures online, and displays the results in a Gradio interface, including the agent's step-by-step reasoning.

---

**Try the live demo:**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Abbasid/HeightCompareAgent)

---

## Key Features

*   Parses height from natural language queries (e.g., "I am 180 cm", "compare 5ft 10in").
*   Uses `DuckDuckGoSearchTool` to dynamically search for people/characters of similar height across various categories (fictional, historical, athletes, etc.).
*   Extracts potential height matches directly from search result text.
*   Validates extracted heights for reasonableness.
*   Generates human-readable comparison statements.
*   Provides a Gradio web interface (`gr.ChatInterface`).
*   Displays the agent's intermediate "Thought:", "Code:", and "Observation:" steps in the UI for transparency.

## Development Insights & Technical Learnings

This project involved significant iteration, particularly around prompt engineering and UI integration:

1.  **System Prompt Challenges:**
    *   Initially, I attempted to guide the agent's complex, multi-step behavior (dynamic searching, extraction, validation) solely through detailed instructions within the `system_prompt`.
    *   This proved difficult; the agent often struggled to consistently follow the nuanced steps and maintain context based only on the system prompt, leading to unreliable results.

2.  **Success with Dynamic Task Generation ("Task Injection"):**
    *   The breakthrough came from shifting strategy. Instead of relying only on the static system prompt, a Python function (`create_height_comparison_task`) was created.
    *   This function takes the raw user query and programmatically wraps it within a detailed, step-by-step task description *before* the agent starts processing.
    *   Passing this dynamically generated, comprehensive task description to `agent.run()` was far more effective in reliably controlling the agent's workflow and ensuring all required steps (searching, parsing, validating, comparing) were executed correctly.

3.  **UI Integration - `GradioUI` vs `gr.ChatInterface`:**
    *   Considered using the built-in `smolagents.GradioUI` for simplicity. However, its standard behavior likely involves passing the raw user input directly to `agent.run()`.
    *   This presented a challenge for the "Task Injection" strategy, as there wasn't an obvious way to intercept the user input, run `create_height_comparison_task` to generate the detailed instructions, and *then* pass those instructions to the agent within the `GradioUI` framework without modifying the library itself.
    *   Using the core `gradio` library's `gr.ChatInterface` directly provided the necessary control. Its callback function pattern allowed for:
        *   Receiving the raw user query.
        *   Calling `create_height_comparison_task` to generate the full task instructions.
        *   Executing `agent.run()` with these complete instructions.
        *   Crucially, capturing the agent's verbose output (`stdout` when `verbosity_level=3`) during execution using `contextlib.redirect_stdout`.
        *   Returning both the captured agent steps and the final answer, formatted for clear display in the chat interface.

## Setup (if running in colab ensure your private key is in secrets)

1.  Install dependencies: `pip install smolagents[gradio,litellm] google-auth` (See `requirements.txt` if available).
2.  Configure API Keys: Requires API keys for the chosen LLM provider (e.g., OpenRouter, Google Gemini). These are typically loaded via `google.colab.userdata` in the notebook or environment variables.
3.  Run the main Python script/notebook cell containing the Gradio `iface.launch()` command.
