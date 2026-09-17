# Calling a Local LLM from Python with Ollama

**A practical guide to Ollama's HTTP API, chat requests, streaming responses, and Python integration**

Running an LLM locally is surprisingly easy these days. With Ollama, you can download a model, run it on your machine, and start chatting with it from the terminal.

But if you're building an application, the terminal isn't really the interesting part.

At some point, your application needs to communicate with the model. You might want to build a chatbot, add an AI feature to an existing application, or use a local model as part of a larger RAG or agent workflow.

That's where Ollama's HTTP API comes in.

In this tutorial, I'll start by testing the API with Postman and then use Python to make the same request programmatically. I'll also look at streaming, because this is one of those things that seems confusing until you actually see what the server is returning.

For this walkthrough, I'm using Ollama locally with the `llama3.2` model.

---

## The setup

The overall setup is quite simple:

```text
Python application
       |
       | HTTP request
       v
Ollama server
       |
       v
llama3.2
```

There are three different things here, and keeping them separate makes troubleshooting easier.

The Python application is the client. Ollama is the server that exposes the API, and `llama3.2` is the model being used by that server.

Ollama's local API normally runs on:

```text
http://localhost:11434
```

So our Python application doesn't need to know anything about the internal model files. It just needs to know how to talk to the API.

For this example, we'll mainly use two endpoints:

```text
GET  /api/tags
POST /api/chat
```

The first one lets us see which models are available. The second one is what we'll use to send a conversation to the model.

---

## First, check whether the model is available

Before writing any Python code, I wanted to make sure the model was actually available locally.

Ollama provides the `/api/tags` endpoint for this:

```http
GET http://localhost:11434/api/tags
```

I tested this from Postman.

The response included my locally available model:

```json
{
  "models": [
    {
      "name": "llama3.2:latest",
      "model": "llama3.2:latest"
    }
  ]
}
```

The actual response contains more information than this, including the model size, digest, format, family and other model details.

![Checking locally available Ollama models using the /api/tags endpoint](images/ollama-tags.png)

This is a small request, but it's a useful first check.

For example, imagine your Python application is configured to use `llama3.2`, but the model hasn't actually been downloaded. You could spend ten minutes debugging the Python request when the real problem is simply that the model isn't there.

I generally prefer checking the infrastructure first and the application code second.

If the server isn't running, fix that.

If the model isn't available, fix that.

Only then start debugging the application.

---

## Sending our first chat request

Once I confirmed that the model was available, I used the `/api/chat` endpoint.

The endpoint is:

```http
POST http://localhost:11434/api/chat
```

Here's the request I used:

```json
{
  "model": "llama3.2",
  "messages": [
    {
      "role": "user",
      "content": "What is an AI agent?"
    }
  ],
  "stream": true
}
```

There isn't much to the request, but there are a couple of things worth understanding.

The `model` field tells Ollama which model should handle the request:

```json
"model": "llama3.2"
```

Then we have `messages`:

```json
"messages": [
  {
    "role": "user",
    "content": "What is an AI agent?"
  }
]
```

The chat API represents the conversation as a list of messages rather than just one prompt string.

That becomes useful when you have a real conversation. You can include previous messages as part of the request, so the model has some context about what has already been discussed.

The third field is the interesting one:

```json
"stream": true
```

This controls how the generated response is returned.

---

## So what exactly is streaming?

If you've used an AI chatbot, you've probably seen text appearing gradually on the screen.

The model starts generating an answer and the application starts displaying it before the complete answer has been generated.

That's essentially what we're talking about here.

With a non-streaming request, the flow is roughly:

```text
Application
     |
     |---- request ---->
     |
     |   model generates
     |   complete answer
     |
     |<--- full response
```

With streaming enabled, it looks more like:

```text
Application
     |
     |---- request ---->
     |
     |<--- chunk 1
     |<--- chunk 2
     |<--- chunk 3
     |<--- chunk 4
     |
```

The application can process those pieces as they arrive.

This is especially useful for chat applications. If the model takes several seconds to generate an answer, the user doesn't have to stare at an empty screen during that time.

---

## Seeing streaming in Postman

This was actually one of the more useful parts of testing the API manually.

I sent the request from Postman with:

```json
"stream": true
```

and instead of getting one complete answer, I could see individual JSON responses arriving.

For example, one response chunk looked like this:

```json
{
  "model": "llama3.2",
  "message": {
    "role": "assistant",
    "content": "An"
  },
  "done": false
}
```

The generated content in this particular chunk was just:

```text
An
```

The next chunk contains more content, and this continues while the model is generating the answer.

![Streaming response from Ollama /api/chat in Postman](images/ollama-chat-streaming.png)

The `done` field is useful here.

When we see:

```json
"done": false
```

the response hasn't finished yet.

Eventually, Ollama sends the final response with `done` set to `true`.

This is much easier to understand when you actually see the response rather than just reading that "the API supports streaming" in documentation.

---

# Calling the same API from Python

Now that the API works in Postman, let's do the same thing from Python.

I'll use the `requests` library because we don't need an Ollama-specific Python SDK for this example. We're just making a normal HTTP request.

Install it with:

```bash
pip install requests
```

Then create:

```text
ollama_client.py
```

For the first Python example, I'm going to turn streaming off.

That makes it easier to understand the basic request and response before adding the extra complexity.

```python
import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3.2",
    "messages": [
        {
            "role": "user",
            "content": "What is an AI agent?"
        }
    ],
    "stream": False
}

response = requests.post(
    url,
    json=payload
)

response.raise_for_status()

data = response.json()

print(data["message"]["content"])
```

That's basically all we need for a first working client.

The request is sent here:

```python
response = requests.post(
    url,
    json=payload
)
```

The `json` parameter takes our Python dictionary and sends it as JSON in the request body.

I also like to check the HTTP status immediately:

```python
response.raise_for_status()
```

Without that check, the program could continue after an unsuccessful HTTP request and produce a less useful error somewhere further down the code.

After the request succeeds, we convert the response to a Python dictionary:

```python
data = response.json()
```

The generated answer can then be accessed through:

```python
data["message"]["content"]
```

---

# Adding streaming to the Python client

Now let's make the Python version behave more like our Postman request.

There are two things to change.

First, we'll ask Ollama to stream the response:

```python
"stream": True
```

Second, we'll tell the Python HTTP client that we're going to consume the response incrementally.

Here's the complete version:

```python
import json
import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3.2",
    "messages": [
        {
            "role": "user",
            "content": "What is an AI agent?"
        }
    ],
    "stream": True
}

with requests.post(
    url,
    json=payload,
    stream=True
) as response:

    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)

            content = chunk["message"].get("content", "")

            print(
                content,
                end="",
                flush=True
            )
```

There are two different `stream=True` values in this code, which is worth calling out.

This one:

```python
"stream": True
```

is part of the JSON request sent to **Ollama**.

It tells Ollama to stream the generated response.

This one:

```python
stream=True
```

belongs to the Python `requests` library.

It tells the HTTP client that we want to consume the HTTP response progressively.

They are related, but they're operating at different layers.

That's a small distinction, but it becomes important when you're trying to understand why a streaming implementation behaves differently from a normal request.

---

# Reading the streamed response

The main difference between the streaming and non-streaming versions is how we consume the response.

Instead of doing:

```python
data = response.json()
```

we iterate over the incoming data:

```python
for line in response.iter_lines():
```

Each line contains JSON, so we parse it:

```python
chunk = json.loads(line)
```

Then we extract the generated content:

```python
content = chunk["message"].get("content", "")
```

Finally, we print it immediately:

```python
print(
    content,
    end="",
    flush=True
)
```

The `end=""` is there because we don't want every small response chunk to appear on a separate line.

`flush=True` makes the output visible immediately rather than waiting for Python's output buffer.

Put these together and you get the familiar effect of an AI response being typed out progressively.

---

# Why I tested the API with Postman first

You could obviously skip Postman and write the Python code immediately.

I still think testing the raw API first is useful.

When you're learning or debugging an integration, every additional layer makes it harder to identify where the problem is.

If this works:

```text
Postman
   |
   v
Ollama API
   |
   v
Model
```

but this doesn't:

```text
Python
   |
   v
Ollama API
   |
   v
Model
```

then the problem is probably somewhere in the Python request or response handling.

That's a much smaller problem to investigate.

Postman also gives you a convenient way to inspect the actual HTTP interaction:

* HTTP method
* URL
* request body
* response status
* response body
* streaming behavior

Once the request is known to work, reproducing it in code is much less mysterious.

---

# A small reusable Python wrapper

The previous example is fine for demonstrating the API, but I wouldn't repeat the complete HTTP request throughout a real application.

We can hide that detail behind a function:

```python
import requests


def chat(prompt, model="llama3.2"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


answer = chat(
    "Explain the difference between an AI agent and a chatbot."
)

print(answer)
```

Now the rest of the application doesn't need to know how the HTTP request is constructed.

It can simply call:

```python
answer = chat("Explain embeddings in simple terms.")
```

This is a very small abstraction, but it's useful because it separates application code from the details of communicating with the model.

As the application grows, that separation becomes more valuable.

---

# Troubleshooting the local setup

When working with a local LLM, an error doesn't always tell you exactly where the problem is.

I find it useful to check the components one at a time.

### Is Ollama running?

First check whether the server is reachable at:

```text
http://localhost:11434
```

If the server isn't running, the Python application obviously won't be able to connect to it.

### Is the model installed?

Use:

```http
GET http://localhost:11434/api/tags
```

and check whether the model you're requesting is actually listed.

### Does the API work from Postman?

Try the same `/api/chat` request manually.

If it works in Postman but fails in Python, compare the two requests.

Pay particular attention to:

```text
URL
HTTP method
model name
messages
stream
```

### Does the model name match?

This is easy to overlook.

For example, the model returned from `/api/tags` might be:

```text
llama3.2:latest
```

while your application might be configured with:

```text
llama3.2
```

Ollama commonly allows the shorter model name when it refers to the default tag, but when debugging, checking the exact model name returned by the server is still useful.

---

# Another common problem: port 11434 is already in use

While working with Ollama locally, you may run:

```bash
ollama serve
```

and get an error saying that port `11434` is already in use.

At first, that can look like an Ollama startup failure.

It isn't necessarily one.

If another Ollama process is already running and listening on that port, starting another server will fail because two processes can't normally listen on the same address and port at the same time.

Before changing anything, check whether the existing server is responding.

For example, in PowerShell:

```powershell
Invoke-WebRequest http://localhost:11434
```

If the server responds, you may already have a working Ollama instance.

This is a useful debugging habit in general:

> Don't assume that a failed startup command means the service itself is unavailable. Check the actual service state first.

---

# Where does Ollama fit into an AI application?

At this point, we have a Python program talking to an LLM.

But the Python program isn't an AI agent just because it calls an LLM.

It's also not automatically a RAG application.

Ollama is providing the model/inference layer.

A larger application could look something like this:

```text
                       User
                         |
                         v
                  Python Application
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
         Retrieval     Tools    Business Logic
             |           |           |
             +-----------+-----------+
                         |
                         v
                    Ollama API
                         |
                         v
                      LLM
```

For example, a RAG application might retrieve relevant information from a vector database and then send that context to the model through Ollama.

An agentic application might use the model to decide which tool should be called.

A simple chatbot might only need the model and some conversation history.

The important part is that Ollama doesn't have to be the entire application.

It's one component.

---

# What about LangChain or LangGraph?

Once an AI application becomes more complicated, developers often introduce frameworks such as LangChain or LangGraph.

They can handle or simplify things such as:

* prompt management
* tool calling
* retrieval
* conversation state
* workflow orchestration
* agentic workflows

There's nothing wrong with using those abstractions.

But I think it's useful to understand what happens underneath them.

If something goes wrong in a framework-based application, being comfortable with the underlying HTTP interaction gives you another way to investigate the problem.

You can ask:

* Which model is being called?
* What messages are being sent?
* Is streaming enabled?
* Is the model available?
* Is the Ollama server reachable?
* What does the raw response look like?

You don't need to avoid frameworks.

Just don't let the framework become a complete black box.

---

# When does running an LLM locally make sense?

Local inference isn't automatically better than a hosted LLM API.

It depends on what you're building.

Running a model locally can be useful when you're:

* experimenting with LLMs
* developing a prototype
* learning how LLM APIs work
* testing an AI application locally
* working with data that you want to keep within your local development environment
* experimenting with different open models

There are trade-offs, of course.

The hardware available on the machine matters. Model size, quantization, CPU/GPU capabilities, available memory and workload all affect performance.

Production deployments introduce another set of concerns:

* authentication
* concurrent requests
* resource limits
* monitoring
* logging
* timeouts
* model versioning
* network security
* infrastructure capacity

So I wouldn't treat local inference as a universal replacement for cloud-based model APIs.

For development and experimentation, though, it is a very practical way to understand what's happening underneath an LLM application.

---

# The mental model I use

After working through the API, this is probably the simplest way to think about the setup:

```text
             YOUR APPLICATION
                    |
                    | HTTP
                    v
              OLLAMA SERVER
                    |
                    v
                 LLM MODEL
```

For a normal request:

```text
Request
   |
   v
Ollama
   |
   v
Complete response
```

For a streaming request:

```text
Request
   |
   v
Ollama
   |
   +----> chunk
   +----> chunk
   +----> chunk
   +----> chunk
   |
   v
final response
```

Once you understand that, the Python code isn't particularly complicated.

It's just an HTTP client consuming the API exposed by the local inference server.

---

# Final thoughts

The interesting thing about Ollama isn't just that you can run a language model on your laptop.

For a developer, the more useful part is that you can treat that local model as a service.

I started by checking what was available through:

```http
GET /api/tags
```

Then I sent a conversational request through:

```http
POST /api/chat
```

Testing the request in Postman made the interaction much easier to see. I could inspect the actual JSON being sent and, more importantly, see what a streaming response looked like.

From there, the Python implementation was mostly a matter of reproducing the same HTTP request and handling the response.

That basic interaction is enough to become the model layer underneath something much larger.

A chatbot could use it.

A RAG application could use it.

An agentic workflow could use it.

And once you understand the API underneath those higher-level tools, debugging them becomes a lot less intimidating.

That's probably the biggest takeaway for me: **before adding another abstraction layer, understand what is happening underneath it.**

---

## Technical reference

The Ollama API documentation provides the endpoint definitions, request formats, response formats and streaming behavior used in this tutorial.

https://github.com/ollama/ollama/blob/main/docs/api.md
