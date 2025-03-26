- **The Agent Workflow:**

Think → Act → Observe.

[Huggingface Agents](https://www.notion.so/Huggingface-Agents-1c171e99f55a8014a9c4e146a4944188?pvs=21)

![image.png](attachment:9ba3586c-986d-4276-97c5-d359b440c588:image.png)

- An Agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions (often via external tools) to fulfill tasks.
- design of the Tools is very important and has a great impact on the quality of your Agent
- Note that Actions are not the same as Tools. An Action, for instance, can involve the use of multiple Tools to complete.
- Instead of following rigid behavior trees, they can respond contextually, adapt to player interactions, and generate more nuanced dialogue. This flexibility helps create more lifelike, engaging characters that evolve alongside the player’s actions.
- Each LLM has some special tokens specific to the model, The most important of those is the End of sequence token (EOS).
- Decoding strategies
    - selecting highest value
    - beam search

- core of an agent library is to append information in the system prompt
- Do you see the problem? The answer was hallucinated by the model
- A: That’s correct! But this is in fact a UI abstraction. Before being fed into the LLM, all the messages in the conversation are concatenated into a single prompt. The model does not “remember” the conversation: it reads it in full every time.
- A: That’s correct! But this is in fact a UI abstraction. Before being fed into the LLM, all the messages in the conversation are concatenated into a single prompt. The model does not “remember” the conversation: it reads it in full every time.
- The LLM will generate text, in the form of code, to invoke that tool. It is the responsibility of the Agent to parse the LLM’s output,
- We could provide the Python source code as the specification of the tool for the LLM, but the way the tool is implemented does not matter. All that matters is its name, what it does, the inputs it expects and the output it provides.
- What Tools Are: Functions that give LLMs extra capabilities, such as performing calculations or accessing external data.
- How to Define a Tool: By providing a clear textual description, inputs, outputs, and a callable function.
- Why Tools Are Essential: They enable Agents to overcome the limitations of static model training, handle real-time tasks, and perform specialized actions.
- What Tools Are: Functions that give LLMs extra capabilities, such as performing calculations or accessing external data.
- How to Define a Tool: By providing a clear textual description, inputs, outputs, and a callable function.
- Why Tools Are Essential: They enable Agents to overcome the limitations of static model training, handle real-time tasks, and perform specialized actions.

![](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/ReAct.png)

- Actions are the concrete steps an AI agent takes to interact with its environment.
- Type of Agents:
    - JSON, Code, function-calling
- The Stop and Parse Approach
    - structured
    - stop generation
    - parse output
- **CodeAct**

![](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/code-vs-json-actions.png)

- thinking (Thought) → acting (Act) and observing (Observe).
- Actions bridge an agent’s internal reasoning and its real-world interactions by executing clear, structured tasks—whether through JSON, code, or function calls.

![](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/AgentCycle.gif)

- another day:
- Sometimes, predefined workflows are sufficient to fulfill user requests, and there is no real need for an agentic framework.
- Smolagents
    - CodaAgents
    - toolCallingAgents
    - tools
    - Retruevak Agebts
    - MultiAgent Systems
    - Vision and Browser Agents
        - Agents in smolagents operate as multi-step agents.
    - However, research shows that tool-calling LLMs work more effectively with code directly. This is a core principle of smolagents, as shown in the diagram above from Executable Code Actions Elicit Better LLM Agents.
    - A tool is mostly a function that an LLM can use in an agentic system. SHOULD BE A PYTHON CLASS
    - Then you can load the tool with load_tool() or create it with from_hub() and pass it to the tools parameter in your agent. Since running tools means running custom code, you need to make sure you trust the repository,
    - You can directly import a Space from the Hub as a tool using the Tool.from_space() method!
    - Beware of not adding too many tools to an agent: this can overwhelm weaker LLM engines.
    -
