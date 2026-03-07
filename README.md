<!-- HEADER -->
<p align="center">
    <a
        target="_blank"
        rel="noopener noreferrer"
        href="https://en.wikipedia.org/wiki/Panopticon"
    >
        <img
            alt="panopticon-header"
            width="350px"
            src="https://res.cloudinary.com/wemakeart/image/upload/v1772893045/github/panopticon/readme_panopticon_reowvb.png"
        >
    </a>
</p>

<!-- INTRODUCTION -->
# PANOPTICON

```🚧 This project is in its early stages and is currently under active development 🚧```

This project is a modular bricolage, purposefully integrating existing tools to expose a local codebase to a suite of language models. The strategy prioritizes cost efficiency by handling routine queries locally, reserving cloud-based models like Claude for high-complexity tasks. A significant advantage is the resulting 'always-live' documentation, which remains synchronized and queriable as the code evolves.

<!-- BODY -->
## 1. Tools Utilized

* [UV](https://docs.astral.sh/uv)
* [Ollama](https://ollama.com)
* [Python](https://www.python.org)
* [LanceDB](https://docs.lancedb.com)
* [Continue Extension](https://www.continue.dev)
* [Nomic Embed Text](https://ollama.com/library/nomic-embed-text:latest) (Any viable embeddings model)
* [DeepSeek Coder V2 Lite](https://ollama.com/library/deepseek-coder-v2:lite) (Any viable language model/s)
* [Visual Studio Code](https://code.visualstudio.com) / [Cursor](https://cursor.com/home) (Any branch of VSC should work)

## 2. Getting Started

```👨🏾‍🔧 It is assumed that you already have UV, Python, Ollama, VS Code IDE (or equivalent) & Continue Extension installed. If not, please install them before proceeding. 👩🏾‍🔧```

**2.1 Install the dependencies -** ```uv sync```

**2.2 Pull required models:**

* *Ensure ollama is running*
* ```ollama pull deepseek-coder-v2:lite```
* ```ollama pull nomic-embed-text```

**2.3 Replace the placeholder - ```"your/path/to/main.py"``` - with your PATH in the Continue MCP-Server config file here - ```.continue\mcpServers\pano.yaml```**

**2.4 Ensure that your Continue local-config contents match the - ```continue_local_example.yaml``` - config**

<img align="left" alt="panopticon-continue-local-config" width="100%" src="https://res.cloudinary.com/wemakeart/image/upload/v1772910901/github/panopticon/pano-continue-local-config_kjxms6.png"></img>
<sub>Example - Cursor IDE</sub>

**2.5 Index the codebase -** ```uv run main.py index```

**2.6 You should now be able to see your MCP Server and Query your codebase:** (see below)

<table>
    <tr>
        <th>2.6.a MCP Connection</th>
        <th>2.6.b Ollama Models</th>
    </tr>
    <tr width="1px">
        <td>
            <img align="left" alt="panopticon-mcp-server" width="100%" src="https://res.cloudinary.com/wemakeart/image/upload/v1772915061/github/panopticon/pano-continue-mcp-server-success_xhndfi.png"></img>
        </td>
        <td>
            <img align="left" alt="panopticon-mcp-server" width="100%" src="https://res.cloudinary.com/wemakeart/image/upload/v1772915061/github/panopticon/pano-continue-llm-success_vdlpc4.png"></img>
        </td>
    </tr>
    <tr>
        <th>2.6.c Successful Query to Codebase</th>
    </tr>
    <tr>
        <td>
            <img align="left" alt="panopticon-mcp-server" width="100%" src="https://res.cloudinary.com/wemakeart/image/upload/v1772916285/github/panopticon/pano-continue-success_w98aew.png"></img>
        </td>
    </tr>
</table>

## 3. In-Progress Features

🟧 Switch to a file-change based indexing strategy <br>
🟧 Improve usability to allow drop-in setup for any project <br>
🟧 Evaluate other models for performance & usability *(for example - `qwen3-embedding:0.6b` with `qwen3.5:9b`)* <br>
