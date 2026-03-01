# Envoy AI Gateway — Research Analysis

---

## Why This Matters

AI usage across engineering teams is growing fast. Right now, every team manages their own connection to AI providers — their own credentials, their own limits, their own logging. There is no central control.

This creates real risk:

- A single team can run up a large bill with no visibility until it is too late
- If an API key is compromised, there is no single place to revoke it
- There is no consistent audit trail of what was sent to which model and when
- Different teams use different providers in different ways — no standardisation

The question is not whether we need a central control point for AI traffic. We do. The question is **which tool is the right fit** and **how much do we need right now**.

---

## What is Envoy AI Gateway?

Envoy AI Gateway is a **cluster-level proxy** that sits between your applications and your AI providers. Every AI request from every application in the cluster flows through it.

It is a Kubernetes-native tool. You configure it using standard Kubernetes resource files (CRDs), the same way you configure networking or access policies today.

**What it does:**

- **Token budgets per team** — set a daily or monthly token limit for each team or namespace. When the limit is hit, requests are blocked automatically.
- **Centralised credential management** — apps never hold API keys. Envoy injects the right credentials per request. One place to rotate, one place to revoke.
- **Schema translation** — an app can send a standard OpenAI-format request, and Envoy translates it on the fly to Bedrock format, Vertex format, or any other provider. No code changes in the app.
- **Full observability** — every token used, every request made, exposed to Prometheus and Grafana out of the box.
- **Policy-driven routing** — route traffic by model, by team, by cost, or by availability using Kubernetes CRD config.

> 📎 *Insert* **diagram-1-envoy-architecture.drawio** *here*

---

## The Istio Analogy — Why This Pattern Is Familiar

If you have used Istio in OpenShift or Kubernetes, Envoy AI Gateway will feel immediately familiar. The pattern is identical.

**Istio** manages and controls traffic between microservices. You define routing rules, rate limits, and security policies using Kubernetes CRDs — and Istio enforces them at the network layer, without touching application code.

**Envoy AI Gateway** does the same thing — but for AI model traffic instead of microservice traffic. You define token budgets, credential policies, and routing rules using Kubernetes CRDs — and the gateway enforces them on every AI request.

Both are built on the same underlying engine: **Envoy Proxy**.

| Concept | Istio calls it | Envoy AI Gateway calls it |
|---|---|---|
| Entry point | Gateway | AI Gateway |
| Routing rule | VirtualService | AIGatewayRoute |
| Traffic policy | RateLimitPolicy | BackendTrafficPolicy |
| Rate limiting unit | Requests / second | Tokens / day per team |
| Underlying engine | Envoy Proxy | Envoy Proxy |

The key difference: Istio measures traffic in **requests per second**. Envoy AI Gateway measures it in **tokens per day** — because that is what drives AI cost.

> 📎 *Insert* **diagram-2-istio-vs-envoy.drawio** *here — this is the centrepiece diagram*

---

## What is LiteLLM?

LiteLLM is an **application-level proxy** for AI model access. It is developer-friendly, quick to deploy, and easy to configure using a simple YAML file.

At enterprise scale, LiteLLM runs as a standalone proxy service — all applications route through it, rather than calling AI providers directly. This is the same deployment pattern as Envoy AI Gateway.

**What it does:**

- Routes requests to the right model based on config
- Falls back to a backup model if the primary fails
- Retries failed requests automatically
- Tracks spend per API key
- Supports 100+ model providers through a unified API

> 📎 *Insert* **diagram-3-litellm.drawio** *here*

---

## Envoy AI Gateway vs LiteLLM

Both tools solve the same core problem: centralise AI traffic rather than letting every app connect directly to providers. The difference is **where they sit** and **what level of control they give you**.

| | Envoy AI Gateway | LiteLLM |
|---|---|---|
| **Layer** | Infrastructure / Cluster | Application / Developer |
| **Deployment** | Kubernetes-native, CRD-driven | Single pod, YAML config |
| **Audience** | Platform engineers | Developer teams |
| **Rate limiting** | Tokens per team / namespace | Spend per API key |
| **Credential management** | Injected by gateway — apps hold no keys | Stored in LiteLLM config |
| **Schema translation** | Yes — OpenAI, Bedrock, Vertex, and more | Yes — via unified API |
| **Observability** | Prometheus + Grafana, built in | Basic spend logging |
| **Policy enforcement** | Kubernetes CRDs — reviewed and audited | YAML config — developer-managed |
| **Complexity to adopt** | Higher — requires platform team involvement | Lower — a team can self-serve |
| **Best for** | Enterprise-wide governance | Quick team-level adoption |

> 📎 *Insert* **diagram-4-combined.drawio** *here*

---

## When to Use Which

**Choose Envoy AI Gateway when:**

- You need a single auditable control point for all AI usage across the organisation
- Token budgets and chargebacks need to be enforced at the platform level
- Security requires that no application pod ever holds an AI API key
- You are building a shared AI platform that multiple teams will use

**Choose LiteLLM when:**

- A team needs AI routing and fallback quickly without a platform dependency
- The priority is developer productivity — simple YAML, fast iteration
- You are at an early stage and want to prove the pattern before investing in platform infrastructure

**Both together (Enterprise pattern):**

In a mature setup, these two tools are not competitors — they work at different layers. Envoy AI Gateway enforces platform-level governance at the cluster boundary. LiteLLM can sit inside the application layer handling model routing and fallback. Traffic passes through Envoy first (governance) and then through LiteLLM (routing).

---

## Executive Summary

Envoy AI Gateway is an infrastructure-level control plane for AI traffic — the same way Istio is a control plane for microservice traffic. It gives the platform team full visibility and control over every AI call made in the cluster, without touching application code.

LiteLLM is a developer-friendly proxy that teams can self-serve today. It solves the immediate problem of routing, fallback, and spend tracking at the application level.

**The strategic direction is Envoy AI Gateway.** It aligns with how we already manage infrastructure in Kubernetes, it follows a pattern engineers already know from Istio, and it puts governance where it belongs — at the platform layer, not inside each application.

LiteLLM is a practical bridge. Teams can adopt it now and continue using it for application-level routing even after Envoy AI Gateway is in place.

---

*Research by [Your Name] — [Date]*
