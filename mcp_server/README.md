# BioAI Unified MCP Server

Model Context Protocol (MCP) server that integrates BioAI Unified's multi-model QA capabilities with AI coding assistants.

## What is This?

This MCP server allows AI coding tools like **Claude Code**, **Gemini CLI**, and **Codex CLI** to use BioAI Unified for:

- **Code Analysis**: Get multi-model consensus on code quality, security issues, and bugs
- **Fix Validation**: Review proposed fixes before applying them
- **Content Generation**: Generate code or documentation with built-in quality assurance
- **Reasoning Control**: Configure thinking depth for complex analysis tasks

Instead of trusting a single AI's analysis, BioAI Unified runs your code through multiple AI models and only approves results when they reach consensus.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r mcp_server/requirements.txt
```

### 2. Ensure BioAI Unified is Running

```bash
# In the project root
python main.py
# Server starts at http://localhost:8000
```

### 3. Add to Your AI Coding Tool

#### Automatic Installation (Recommended)

Use the installer scripts in the project root:

**Windows:**
```cmd
install_mcp.bat
```

**Linux/macOS:**
```bash
chmod +x install_mcp.sh
./install_mcp.sh
```

The scripts automatically detect the correct absolute path and register the MCP server with Claude Code.

To uninstall:
```bash
./install_mcp.sh --uninstall   # Linux/macOS
install_mcp.bat --uninstall    # Windows
```

#### Manual Installation

> **Important**: MCP servers require **absolute paths**. Variables like `${PROJECT_ROOT}` are NOT supported by Claude Code. The only supported variables are system environment variables (`${HOME}`, `${USER}`, etc.) and `${CLAUDE_PLUGIN_ROOT}` (for plugins only).
>
> Relative paths will fail because the working directory is not guaranteed when Claude launches the server.

**Claude Code CLI:**
```bash
# Replace /path/to/BioAI_Unified with your actual path
claude mcp add bioai-unified -- python /path/to/BioAI_Unified/mcp_server/bioai_mcp_server.py

# Windows example:
claude mcp add bioai-unified -- python C:\Projects\BioAI_Unified\mcp_server\bioai_mcp_server.py

# Linux/macOS example:
claude mcp add bioai-unified -- python /home/user/BioAI_Unified/mcp_server/bioai_mcp_server.py
```

**Gemini CLI** (`~/.gemini/settings.json`):
```json
{
  "mcpServers": {
    "bioai-unified": {
      "command": "python",
      "args": ["/path/to/BioAI_Unified/mcp_server/bioai_mcp_server.py"]
    }
  }
}
```

**Codex CLI** (`~/.codex/config.toml`):
```toml
[features]
rmcp_client = true

[mcp_servers.bioai-unified]
command = "python"
args = ["/path/to/BioAI_Unified/mcp/bioai_mcp_server.py"]
```

> **Note**: Replace `/path/to/BioAI_Unified` with your actual installation path.

## Available Tools

### `bioai_analyze_code`

Analyze code for bugs, security issues, performance problems, and best practices violations.

```
Arguments:
  - code (required): The code to analyze
  - language: Programming language (auto-detected if not specified)
  - context: Additional context about the code
  - focus_areas: Array of areas to focus on (security, performance, bugs, style, all)

  # Reasoning Configuration (optional):
  - generator_model: Override generator model
  - qa_models: Override QA models list
  - reasoning_effort: "none" | "low" | "medium" | "high" (for GPT-5/O1/O3)
  - thinking_budget_tokens: integer >= 1024 (for Claude models)
  - qa_reasoning_effort: Reasoning effort for QA models

Returns:
  - Analysis with issues, severity levels, and recommendations
  - Overall quality score (1-10)
  - Consensus from multiple AI models
```

**Example usage in Claude Code:**
```
> Analyze this code for security issues using BioAI with high reasoning

[Claude calls bioai_analyze_code with reasoning_effort: "high"]

BioAI Analysis Results:
- Score: 7.5/10
- Issues Found: 3
  - [CRITICAL] SQL Injection vulnerability at line 45
  - [HIGH] Hardcoded API key at line 12
  - [MEDIUM] Missing input validation at line 30
```

### `bioai_review_fix`

Review a proposed code fix before applying it.

```
Arguments:
  - original_code (required): The original code with the issue
  - proposed_fix (required): The proposed fixed code
  - issue_description (required): What issue is being fixed
  - language: Programming language

  # Reasoning Configuration (optional):
  - generator_model: Override generator model
  - reasoning_effort: Reasoning depth for analysis
  - qa_reasoning_effort: Reasoning depth for QA evaluation

Returns:
  - Verdict: approve / reject / needs_changes
  - Whether fix solves the issue
  - Whether fix introduces new issues
  - Security and performance impact assessment
```

**Example usage:**
```
> Before applying this fix, validate it with BioAI

[Claude calls bioai_review_fix]

BioAI Review:
- Verdict: APPROVE
- Score: 8.5/10
- Solves Issue: Yes
- New Issues: None
- Security Impact: Positive (fixes vulnerability)
```

### `bioai_generate_with_qa`

Generate content with multi-model quality assurance.

```
Arguments:
  - prompt (required): The generation prompt
  - content_type: Type of content (code, documentation, article, json)
  - qa_criteria: Custom QA criteria array
  - min_score: Minimum score to approve (default: 7.5)
  - max_iterations: Maximum generation attempts (default: 3)

  # Reasoning Configuration (optional):
  - generator_model: Override generator model
  - qa_models: Override QA models list
  - reasoning_effort: Reasoning depth for generation
  - thinking_budget_tokens: Thinking tokens for Claude
  - qa_reasoning_effort: Reasoning depth for QA

Returns:
  - Generated content (only if approved)
  - Final score
  - QA summary
```

### `bioai_check_health`

Check if BioAI Unified API is available.

### `bioai_list_models`

List available AI models organized by provider.

### `bioai_get_config`

Get current MCP server configuration including default models and reasoning settings.

## Configuration

All configuration is via environment variables:

### Connection Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `BIOAI_API_URL` | `http://localhost:8000` | BioAI Unified API URL |
| `BIOAI_API_KEY` | (empty) | API key for authenticated access |
| `BIOAI_TIMEOUT` | `300` | Request timeout in seconds |
| `BIOAI_POLL_INTERVAL` | `2.0` | Polling interval for results |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BIOAI_GENERATOR_MODEL` | `gpt-5.2` | Default generator model |
| `BIOAI_QA_MODELS` | `claude-opus-4-5-20251101,z-ai/glm-4.7,gemini-3-pro-preview` | Comma-separated QA models |
| `BIOAI_GRAN_SABIO_MODEL` | `claude-opus-4-5-20251101` | Model for conflict resolution |

### Reasoning Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BIOAI_GENERATOR_REASONING` | `medium` | Default reasoning effort for generator (`none`, `low`, `medium`, `high`) |
| `BIOAI_QA_REASONING` | `medium` | Default reasoning effort for QA models |
| `BIOAI_GRAN_SABIO_REASONING` | `high` | Default reasoning effort for Gran Sabio |
| `BIOAI_THINKING_BUDGET` | `0` | Default thinking budget for Claude models (0 = auto) |

### Reasoning Effort Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `none` | No extended reasoning | Simple tasks, fast responses |
| `low` | Light reasoning | Routine code reviews |
| `medium` | Balanced reasoning | Most analysis tasks (default) |
| `high` | Deep reasoning | Complex security analysis, critical fixes |

**Example with custom configuration:**
```bash
BIOAI_API_URL=https://api.bioai.example.com \
BIOAI_API_KEY=sk-xxx \
BIOAI_GENERATOR_MODEL=gpt-5.2 \
BIOAI_GENERATOR_REASONING=high \
BIOAI_THINKING_BUDGET=8192 \
python bioai_mcp_server.py
```

## Using with Remote BioAI (SaaS)

If you're using a hosted BioAI Unified instance (run from BioAI_Unified directory):

```bash
claude mcp add bioai-unified \
  --env BIOAI_API_URL=https://api.bioai.example.com \
  --env BIOAI_API_KEY=your-api-key \
  --env BIOAI_GENERATOR_REASONING=high \
  -- python mcp_server/bioai_mcp_server.py
```

## Per-Call Reasoning Override

The AI client can override reasoning settings per-call:

```json
{
  "tool": "bioai_analyze_code",
  "arguments": {
    "code": "def vulnerable(): ...",
    "focus_areas": ["security"],
    "reasoning_effort": "high",
    "qa_reasoning_effort": "high"
  }
}
```

This allows the AI to request deeper analysis for complex or security-critical code.

## Windows Notes

On Windows, use the `install_mcp.bat` script for automatic setup.

For manual installation, use absolute paths:

```cmd
claude mcp add bioai-unified -- python C:\path\to\BioAI_Unified\mcp_server\bioai_mcp_server.py
```

If Python is not in your PATH, use the full Python path:

```cmd
claude mcp add bioai-unified -- C:\Python311\python.exe C:\path\to\BioAI_Unified\mcp_server\bioai_mcp_server.py
```

## Troubleshooting

### "Connection refused" error

Ensure BioAI Unified is running:
```bash
curl http://localhost:8000/
```

### "MCP SDK not installed"

Install the MCP package:
```bash
pip install mcp
```

### Timeout errors

Increase the timeout for complex analyses:
```bash
BIOAI_TIMEOUT=600 python bioai_mcp_server.py
```

### Models not found

Check available models:
```bash
curl http://localhost:8000/models
```

### Check current configuration

Use the `bioai_get_config` tool to see active settings.

## How It Works

```
+---------------------+     stdio      +---------------------+     HTTP      +---------------------+
|  Claude Code        | -------------> |  MCP Server         | ------------> |  BioAI Unified      |
|  Gemini CLI         |                |  (this script)      |               |  localhost:8000     |
|  Codex CLI          | <------------- |                     | <------------ |  or remote URL      |
+---------------------+                +---------------------+               +---------------------+
         |                                      |                                      |
         |  "Analyze this code"                 |                                      |
         |  + reasoning_effort: high            |                                      |
         |                                      |            +-------------------------+
         |                                      |            |  1. GPT-5.2             |
         |                                      |            |     generates analysis  |
         |                                      |            |     (with reasoning)    |
         |                                      |            |                         |
         |                                      |            |  2. Claude Opus 4.5     |
         |                                      |            |     + GLM-4.7           |
         |                                      |            |     + Gemini 3 Pro      |
         |                                      |            |     evaluate (QA)       |
         |                                      |            |     (with reasoning)    |
         |                                      |            |                         |
         |                                      |            |  3. Consensus score     |
         |  Analysis with 8.5/10 score          |            |     calculated          |
         | <------------------------------------+------------+-------------------------+
```

## License

Same license as BioAI Unified (MIT).
