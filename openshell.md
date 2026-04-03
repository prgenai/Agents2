# NVIDIA OpenShell — Technical Documentation

---

## Table of Contents

1. [What is OpenShell?](#1-what-is-openshell)
2. [Why It Exists — The Problem It Solves](#2-why-it-exists--the-problem-it-solves)
3. [Two Core Parts](#3-two-core-parts)
   - 3.1 [Gateway — The Control Room](#31-gateway--the-control-room)
   - 3.2 [Sandbox — Where the Agent Lives](#32-sandbox--where-the-agent-lives)
4. [Full Architecture at a Glance](#4-full-architecture-at-a-glance)
5. [Four Protection Layers](#5-four-protection-layers)
   - 5.1 [Layer 1 — Filesystem Isolation (Landlock LSM)](#51-layer-1--filesystem-isolation-landlock-lsm)
   - 5.2 [Layer 2 — System Call Filtering (seccomp BPF)](#52-layer-2--system-call-filtering-seccomp-bpf)
   - 5.3 [Layer 3 — Network Isolation (Network Namespace)](#53-layer-3--network-isolation-network-namespace)
   - 5.4 [Layer 4 — Outbound Traffic Control (HTTP Proxy + OPA)](#54-layer-4--outbound-traffic-control-http-proxy--opa)
6. [How AI Model Calls Work (Inference Routing)](#6-how-ai-model-calls-work-inference-routing)
7. [Request Lifecycle — Step by Step](#7-request-lifecycle--step-by-step)
8. [Real Example — Trading Analysis Agent](#8-real-example--trading-analysis-agent)
   - 8.1 [Sandbox Definition](#81-sandbox-definition)
   - 8.2 [Policy File](#82-policy-file)
   - 8.3 [Running the Agent](#83-running-the-agent)
   - 8.4 [What Each Layer Does](#84-what-each-layer-does)
9. [OpenClaw Risks and NemoClaw Protection](#9-openclaw-risks-and-nemoclaw-protection)
   - 9.1 [Risks Without OpenShell](#91-risks-without-openshell)
   - 9.2 [What NemoClaw Adds](#92-what-nemoclaw-adds)
10. [Sources](#10-sources)

---

## 1. What is OpenShell?

OpenShell is an open-source sandbox platform that runs AI agents safely. Released by NVIDIA, it isolates AI agents from your system using kernel-level security features. This means the agent can do its job — read files, call APIs, make model inferences — without being able to steal your data, exfiltrate secrets, or break into other parts of your machine.

OpenShell works with any AI agent framework that can run inside a container. It has been tested with OpenClaw, Claude, GPT-4, and custom agents. The key insight is simple: don't trust the agent to police itself. Lock it down at the kernel level instead.

> **Diagram to insert here:** `diagram-0` — High-level overview showing Agent → Sandbox → Gateway → External World

![Diagram](./diagrams/diagram-0.svg)

---

## 2. Why It Exists — The Problem It Solves

AI agents are powerful but dangerous. They can access files, call APIs, run code, and talk to the internet. By default, they run with your full permissions. If an agent gets compromised — through a malicious plugin, a prompt injection attack, or a bug in the agent framework — it has the keys to your kingdom.

Real incidents show this is not theoretical. OpenClaw had 2 CVEs that let malicious plugins steal credentials. Over 30,000 instances of vulnerable agent frameworks were exposed publicly on GitHub. Alibaba's ROME agent was found exfiltrating user data. Malicious Slack bot plugins have logged chat messages. This is happening now.

Most solutions try to solve this in software — guardrails in prompts, model safety training, plugin vetting. These are all good, but they share the same weakness. If the attacker controls what the agent sees or can manipulate the agent's reasoning, these defenses fall apart.

OpenShell takes a different approach. It does not ask the agent to be good. It makes it impossible for a bad agent to do damage.

> **Diagram to insert here:** `diagram-1` — Side-by-side comparison: Without OpenShell vs. With OpenShell

![Diagram](./diagrams/diagram-1.svg)

---

## 3. Two Core Parts

OpenShell has two separate pieces: a Gateway and a Sandbox. They run as separate processes on your machine. This separation is deliberate.

The **Gateway** is the control plane. It runs with your privileges and keeps your secrets. It decides what the sandbox can do. It holds your API keys, applies your policies, and logs everything. You control the Gateway.

The **Sandbox** is the data plane. It is where the agent actually runs. At creation time, the Gateway applies kernel-level restrictions: file access limits, network isolation, system call filtering. Once the agent starts, these locks cannot be undone from inside the sandbox. The agent cannot grant itself new permissions.

Setting it up takes one command. The Gateway and Sandbox communicate over a secure API. Everything the agent does is logged and auditable.

> **Diagram to insert here:** `diagram-2` — Two-panel view of Gateway (Control Plane) and Sandbox (Data Plane) with their components

![Diagram](./diagrams/diagram-2.svg)

### 3.1 Gateway — The Control Room

The Gateway runs on your machine with your permissions. It handles:

- **Sandbox Lifecycle** — creates, starts, stops, and destroys sandboxes
- **Policy Engine** — runs Open Policy Agent (OPA) to evaluate rules
- **Credential Store** — holds your API keys, database passwords, and other secrets
- **Auth Gate** — decides which requests from the sandbox are allowed
- **Audit Log** — records every action the agent takes
- **Operator TUI** — gives you a text interface to control everything

The agent never touches any of this. It cannot read the policy file, steal credentials, or modify logs.

### 3.2 Sandbox — Where the Agent Lives

The Sandbox is an isolated execution environment. When it starts, the Gateway applies:

- **File access rules** — locks down which paths the agent can read or write
- **System call filters** — blocks dangerous operations like ptrace, mount, and kexec
- **Network isolation** — agent gets its own network namespace, separate from yours
- **Credential injection** — when the agent calls inference.local, the Gateway swaps the agent's credentials with real API keys before the call goes out

The sandbox cannot change these restrictions once it is running.

---

## 4. Full Architecture at a Glance

Before diving into each layer, here is the full picture. Everything you need to know about how OpenShell works fits into this view.

> **Diagram to insert here:** `diagram-2-5` — Full architecture showing all components: Gateway internals, Sandbox internals, four kernel layers, inference.local, and External World

![Diagram](./diagrams/diagram-2-5.svg)

| Part | Where it runs | What it does |
|------|---------------|--------------|
| **Policy Engine (OPA)** | Gateway | Evaluates rules from policy.yaml, decides if requests are allowed |
| **Credential Store** | Gateway | Holds API keys and secrets, never exposed to sandbox |
| **Auth Gate** | Gateway | Checks every request from sandbox before allowing it |
| **Audit Log** | Gateway | Records all sandbox activity, queryable after execution |
| **Operator TUI** | Gateway | Text interface for humans to control the system |
| **AI Agent** | Sandbox | Your agent code running inside the isolated environment |
| **Landlock LSM** | Sandbox | Kernel module that enforces file access rules |
| **seccomp BPF** | Sandbox | Kernel subsystem that filters system calls |
| **Network Namespace** | Sandbox | Kernel feature that gives the sandbox its own network stack |
| **HTTP Proxy** | Sandbox | Intercepts outbound connections, enforces network policy |
| **inference.local** | Sandbox | Virtual endpoint that routes LLM calls through the Gateway |
| **Cloud LLM** | Internet | Claude, GPT-4, or other cloud models — agent credentials replaced before call |
| **Approved APIs** | Internet | External services you explicitly allow in the policy |
| **Blocked Endpoints** | Internet | Everything else — rejected by HTTP Proxy + OPA |

---

## 5. Four Protection Layers

OpenShell uses defence in depth. There are four independent layers, each using a different kernel mechanism. If an attacker bypasses one, the others still protect you.

### 5.1 Layer 1 — Filesystem Isolation (Landlock LSM)

Landlock is a Linux Security Module — a kernel extension that checks file operations. When the sandbox starts, the Gateway tells the kernel exactly which paths the agent can read and write. The agent cannot see /etc, /home, /root, or ~/.ssh. It can only access /sandbox and /tmp.

These rules are locked in at creation time. The agent running inside cannot grant itself access to new paths. Even if it calls chmod or tries to trick the kernel, Landlock still says no.

> **Diagram to insert here:** `diagram-3a` — Filesystem tree showing allowed paths (green) vs. blocked paths (red)

![Diagram](./diagrams/diagram-3a.svg)

### 5.2 Layer 2 — System Call Filtering (seccomp BPF)

seccomp stands for "secure computing mode." It is a kernel subsystem that filters system calls — the low-level operations that programs use to interact with the OS. When the sandbox starts, the Gateway installs a seccomp filter that runs for every system call made by the agent.

The filter is written in BPF bytecode and runs in the kernel. It blocks ptrace (attaching to other processes), mount (mounting filesystems), clone with NEWUSER (creating hidden user namespaces), perf_event_open (performance monitoring abuse), and kexec (loading a new kernel). These operations could escalate privileges or let the agent escape.

Because the filter runs in kernel space, the agent cannot remove it, patch it, or work around it. It can only use the system calls that are explicitly allowed.

> **Diagram to insert here:** `diagram-3b` — Allowed syscalls (green) vs. blocked syscalls (red) with the seccomp BPF filter in the middle

![Diagram](./diagrams/diagram-3b.svg)

### 5.3 Layer 3 — Network Isolation (Network Namespace)

A network namespace is a kernel feature that gives the sandbox its own isolated network stack. The agent's network is completely separate from your machine's network. It cannot see your WiFi, your LAN, or your loopback device.

The only way out of the network namespace is through a single exit point: the HTTP Proxy running inside the sandbox. All outbound connections go through this proxy, which enforces the network policy.

> **Diagram to insert here:** `diagram-3c` — Two bubbles: Host Network (grey) and Sandbox Network (green), showing the wall between them and the single proxy exit point

![Diagram](./diagrams/diagram-3c.svg)

### 5.4 Layer 4 — Outbound Traffic Control (HTTP Proxy + OPA)

Every outbound HTTP connection from the agent is intercepted by the HTTP CONNECT proxy. The proxy checks the destination hostname and the HTTP method against rules defined in policy.yaml. These rules are evaluated by Open Policy Agent (OPA).

For example, you can say: "Agent can call market-api.internal on port 443, GET only, from python3 only." Any other request — different hostname, wrong port, wrong method, wrong binary — gets blocked and logged immediately.

Unlike the kernel layers, the network policy can be updated while the sandbox is running. Hot-reload support means you can tighten or loosen rules without restarting.

> **Diagram to insert here:** `diagram-3d` — Flow diagram showing request → proxy intercepts → OPA evaluates → allow/deny decision with policy.yaml example

![Diagram](./diagrams/diagram-3d.svg)

---

## 6. How AI Model Calls Work (Inference Routing)

When the agent needs to call an AI model, it makes a request to `inference.local`. This is a virtual hostname that only exists inside the sandbox. The Gateway is listening on that name.

The agent includes its own credentials in the request. The Gateway strips out whatever the agent sent, injects the real API key from the Credential Store, and forwards the request to the actual model endpoint.

This means the agent never needs to know the real API key. The request can go to Claude, GPT-4, or a local LLM. From the external service's perspective, the request came from you, not from an untrusted agent. The agent's credentials never leave the sandbox.

> **Diagram to insert here:** `diagram-4` — Three-step pipeline: Strip agent token → Inject real credential → Route to approved LLM provider

![Diagram](./diagrams/diagram-4.svg)

---

## 7. Request Lifecycle — Step by Step

Here is the journey of a single request from the agent, end to end.

> **Diagram to insert here:** `diagram-5` — 9-step flowchart from agent request through all four layers to final response and audit log

![Diagram](./diagrams/diagram-5.svg)

| Step | Layer | What happens |
|------|-------|--------------|
| 1 | Agent | Agent tries to do something: open a file, make an API call, run a command |
| 2 | Landlock LSM | Kernel checks if the file path is allowed |
| 3 | seccomp BPF | Kernel checks if the system operation is safe |
| 4 | Network Namespace | Kernel routes the request through the sandbox's isolated network |
| 5 | HTTP Proxy | Proxy intercepts the outbound connection |
| 6 | OPA Rules | Policy engine checks the destination and method against policy.yaml |
| 7a | Allowed path | Request matches a rule — continues to destination or inference.local |
| 7b | Blocked path | Request matches no rule — rejected, written to audit log |
| 8 | Response | If allowed: response returns to the agent through the same path |
| 9 | Audit Log | Everything is recorded, allowed or blocked |

---

## 8. Real Example — Trading Analysis Agent

Let's walk through a real example. An agent reads internal market data, generates trading analysis, and summarises it using a cloud AI model. The market data must never leave your network. API keys stay in the Gateway. The agent is fully sandboxed.

### 8.1 Sandbox Definition

The agent runs in a container. Here is the Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /sandbox

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .

RUN mkdir -p /workspace/market-data /workspace/reports

CMD ["python", "agent.py"]
```

The agent code:

```python
import requests

# Read market data from internal API
response = requests.get('https://market-api.internal:443/latest-prices')
market_data = response.json()

analysis = f"Market summary: {len(market_data)} records loaded"

# Call cloud AI for summary
ai_response = requests.post(
    'https://inference.local/v1/messages',
    json={
        'model': 'claude-3-5-sonnet-20241022',
        'messages': [{'role': 'user', 'content': f'Summarize: {analysis}'}]
    }
)

summary = ai_response.json()['content'][0]['text']

with open('/workspace/reports/summary.txt', 'w') as f:
    f.write(summary)

print("Analysis complete")
```

### 8.2 Policy File

The policy file is YAML. It defines exactly what the sandbox can and cannot do:

```yaml
version: "1"

filesystem_policy:
  read_paths:
    - /sandbox
    - /workspace/market-data
    - /workspace/reports
  read_write_paths:
    - /workspace/reports
    - /tmp

process:
  allowed_binaries:
    - /usr/local/bin/python3
    - /usr/bin/python3
    - /bin/sh
    - /usr/bin/curl

network_policies:
  - name: internal-market-api
    destination:
      hostname: market-api.internal
      port: 443
    http_methods: [GET]
    allowed_binaries: [python3]

  - name: inference-local-calls
    destination:
      hostname: inference.local
      port: 443
    http_methods: [POST]
    allowed_binaries: [python3]

  - name: deny-all-else
    destination:
      hostname: "*"
      port: "*"
    http_methods: []
    allowed_binaries: []
```

This policy allows:
- Read access to /sandbox and market data paths
- Write access to /workspace/reports and /tmp only
- Only python3, curl, and shell as allowed binaries
- GET to market-api.internal and POST to inference.local
- Everything else is blocked

### 8.3 Running the Agent

```bash
# Start the Gateway
openshell gateway start --config gateway.yaml

# Create a provider pointing to your local Docker
openshell provider create docker \
  --name local-docker \
  --socket /var/run/docker.sock

# Create the sandbox with the policy
openshell sandbox create \
  --provider local-docker \
  --image trading-agent:latest \
  --policy policy.yaml \
  --name trading-bot
```

The Gateway reads the policy, sets up Landlock, installs the seccomp filter, creates the network namespace, starts the HTTP proxy, then launches the container.

### 8.4 What Each Layer Does

**Landlock** — The agent cannot see /etc, /home, or /root. It only sees /workspace/market-data and /workspace/reports. Any attempt to read /etc/passwd is blocked at the kernel.

**seccomp** — The agent cannot call ptrace() to inspect other processes, cannot mount() a filesystem, and cannot use clone() with NEWUSER to escalate privileges.

**Network Namespace** — The agent's network is isolated. All HTTP requests go through the proxy automatically.

**HTTP Proxy + OPA** — GET to market-api.internal is allowed. Any call to example.com or any unrecognised host is blocked immediately.

**Inference Routing** — When the agent calls inference.local, the Gateway replaces the agent's API key with the real one from the Credential Store. The request goes to Claude. Market data never leaves your network.

> **Diagram to insert here:** `diagram-6` — Full trading agent flow inside the OpenShell boundary, showing sandbox actions on the left and Gateway enforcement on the right

![Diagram](./diagrams/diagram-6.svg)

---

## 9. OpenClaw Risks and NemoClaw Protection

OpenClaw is NVIDIA's agent framework. It is powerful, flexible, and widely used. But by default, it runs with no isolation — the agent has your full permissions and can read any file, connect to any server, and load any plugin.

### 9.1 Risks Without OpenShell

| Category | Risk | Impact |
|----------|------|--------|
| **Files** | Agent can read and write any file on your machine | Private keys, customer data, source code — all exposed |
| **Network** | Agent can connect to any server | Data exfiltrated, internal services reachable as if it's you |
| **API Keys** | Agent sees all credentials in environment variables | Keys stolen and used to impersonate you |
| **Plugins** | Any plugin code runs with your permissions | Malicious plugin steals credentials, loads malware |
| **AI Calls** | Raw data sent to cloud AI with your account | Customer PII and proprietary data exposed in logs |
| **Logs** | Execution logs contain sensitive data | Accidentally committed to git or exposed in error reports |

### 9.2 What NemoClaw Adds

NemoClaw is NVIDIA's OpenClaw + OpenShell wrapper. It bundles OpenClaw inside an OpenShell sandbox with sensible defaults. You run one command:

```bash
nemoclaw run --agent my-agent.yaml --policy default.yaml
```

NemoClaw adds several features on top of base OpenShell:

- **PII Redaction** — automatically removes personal information before sending data to cloud AI
- **Deny-All Network Policy** — nothing is allowed by default; you explicitly add what the agent can reach
- **Intent Verification** — for critical operations, the agent must explain what it is doing before the Gateway allows it
- **Cisco AI Defense Skill Vetting** — integrates with Cisco's AI Defense to vet plugin code before the agent loads it

> **Diagram to insert here:** `diagram-7` — Before/after comparison: OpenClaw alone (risks exposed) vs. NemoClaw + OpenShell (all risks addressed)

![Diagram](./diagrams/diagram-7.svg)

---

## 10. Sources

- [Overview of NVIDIA OpenShell](https://docs.nvidia.com/openshell/latest/about/overview.html)
- [How OpenShell Works](https://docs.nvidia.com/openshell/latest/about/architecture.html)
- [About Gateways and Sandboxes](https://docs.nvidia.com/openshell/latest/sandboxes/index.html)
- [Customize Sandbox Policies](https://docs.nvidia.com/openshell/latest/sandboxes/policies.html)
- [Configure Inference Routing](https://docs.nvidia.com/openshell/latest/inference/configure.html)
- [Policy Schema Reference](https://docs.nvidia.com/openshell/latest/reference/policy-schema.html)
- [NVIDIA OpenShell GitHub](https://github.com/NVIDIA/OpenShell)
- [NVIDIA NemoClaw GitHub](https://github.com/NVIDIA/NemoClaw)
