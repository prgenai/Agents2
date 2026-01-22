import { tool, Agent, AgentInputItem, Runner, withTrace } from "@openai/agents";
import { z } from "zod";
import { OpenAI } from "openai";
import { runGuardrails } from "@openai/guardrails";


// Tool definitions
const getRetentionOffers = tool({
  name: "getRetentionOffers",
  description: "Retrieve possible retention offers for a customer",
  parameters: z.object({
    customer_id: z.string(),
    account_type: z.string(),
    current_plan: z.string(),
    tenure_months: z.integer(),
    recent_complaints: z.boolean()
  }),
  execute: async (input: {customer_id: string, account_type: string, current_plan: string, tenure_months: integer, recent_complaints: boolean}) => {
    // TODO: Unimplemented
  },
});

// Shared client for guardrails and file search
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Guardrails definitions
const jailbreakGuardrailConfig = {
  guardrails: [
    { name: "Jailbreak", config: { model: "gpt-5-nano", confidence_threshold: 0.7 } }
  ]
};
const context = { guardrailLlm: client };

function guardrailsHasTripwire(results: any[]): boolean {
    return (results ?? []).some((r) => r?.tripwireTriggered === true);
}

function getGuardrailSafeText(results: any[], fallbackText: string): string {
    for (const r of results ?? []) {
        if (r?.info && ("checked_text" in r.info)) {
            return r.info.checked_text ?? fallbackText;
        }
    }
    const pii = (results ?? []).find((r) => r?.info && "anonymized_text" in r.info);
    return pii?.info?.anonymized_text ?? fallbackText;
}

async function scrubConversationHistory(history: any[], piiOnly: any): Promise<void> {
    for (const msg of history ?? []) {
        const content = Array.isArray(msg?.content) ? msg.content : [];
        for (const part of content) {
            if (part && typeof part === "object" && part.type === "input_text" && typeof part.text === "string") {
                const res = await runGuardrails(part.text, piiOnly, context, true);
                part.text = getGuardrailSafeText(res, part.text);
            }
        }
    }
}

async function scrubWorkflowInput(workflow: any, inputKey: string, piiOnly: any): Promise<void> {
    if (!workflow || typeof workflow !== "object") return;
    const value = workflow?.[inputKey];
    if (typeof value !== "string") return;
    const res = await runGuardrails(value, piiOnly, context, true);
    workflow[inputKey] = getGuardrailSafeText(res, value);
}

async function runAndApplyGuardrails(inputText: string, config: any, history: any[], workflow: any) {
    const guardrails = Array.isArray(config?.guardrails) ? config.guardrails : [];
    const results = await runGuardrails(inputText, config, context, true);
    const shouldMaskPII = guardrails.find((g) => (g?.name === "Contains PII") && g?.config && g.config.block === false);
    if (shouldMaskPII) {
        const piiOnly = { guardrails: [shouldMaskPII] };
        await scrubConversationHistory(history, piiOnly);
        await scrubWorkflowInput(workflow, "input_as_text", piiOnly);
        await scrubWorkflowInput(workflow, "input_text", piiOnly);
    }
    const hasTripwire = guardrailsHasTripwire(results);
    const safeText = getGuardrailSafeText(results, inputText) ?? inputText;
    return { results, hasTripwire, safeText, failOutput: buildGuardrailFailOutput(results ?? []), passOutput: { safe_text: safeText } };
}

function buildGuardrailFailOutput(results: any[]) {
    const get = (name: string) => (results ?? []).find((r: any) => ((r?.info?.guardrail_name ?? r?.info?.guardrailName) === name));
    const pii = get("Contains PII"), mod = get("Moderation"), jb = get("Jailbreak"), hal = get("Hallucination Detection"), nsfw = get("NSFW Text"), url = get("URL Filter"), custom = get("Custom Prompt Check"), pid = get("Prompt Injection Detection"), piiCounts = Object.entries(pii?.info?.detected_entities ?? {}).filter(([, v]) => Array.isArray(v)).map(([k, v]) => k + ":" + v.length), conf = jb?.info?.confidence;
    return {
        pii: { failed: (piiCounts.length > 0) || pii?.tripwireTriggered === true, detected_counts: piiCounts },
        moderation: { failed: mod?.tripwireTriggered === true || ((mod?.info?.flagged_categories ?? []).length > 0), flagged_categories: mod?.info?.flagged_categories },
        jailbreak: { failed: jb?.tripwireTriggered === true },
        hallucination: { failed: hal?.tripwireTriggered === true, reasoning: hal?.info?.reasoning, hallucination_type: hal?.info?.hallucination_type, hallucinated_statements: hal?.info?.hallucinated_statements, verified_statements: hal?.info?.verified_statements },
        nsfw: { failed: nsfw?.tripwireTriggered === true },
        url_filter: { failed: url?.tripwireTriggered === true },
        custom_prompt_check: { failed: custom?.tripwireTriggered === true },
        prompt_injection: { failed: pid?.tripwireTriggered === true },
    };
}
const ClassificationAgentSchema = z.object({ classification: z.enum(["return_item", "cancel_subscription", "get_information"]) });
const classificationAgent = new Agent({
  name: "Classification agent",
  instructions: `Classify the user’s intent into one of the following categories: \"return_item\", \"cancel_subscription\", or \"get_information\".

1. Any device-related return requests should route to return_item.
2. Any retention or cancellation risk, including any request for discounts should route to cancel_subscription.
3. Any other requests should go to get_information.`,
  model: "gpt-4.1-mini",
  outputType: ClassificationAgentSchema,
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const returnAgent = new Agent({
  name: "Return agent",
  instructions: `Offer a replacement device with free shipping.
`,
  model: "gpt-4.1-mini",
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const retentionAgent = new Agent({
  name: "Retention Agent",
  instructions: "You are a customer retention conversational agent whose goal is to prevent subscription cancellations. Ask for their current plan and reason for dissatisfaction. Use the get_retention_offers to identify return options. For now, just say there is a 20% offer available for 1 year.",
  model: "gpt-4.1-mini",
  tools: [
    getRetentionOffers
  ],
  modelSettings: {
    temperature: 1,
    topP: 1,
    parallelToolCalls: true,
    maxTokens: 2048,
    store: true
  }
});

const informationAgent = new Agent({
  name: "Information agent",
  instructions: `You are an information agent for answering informational queries. Your aim is to provide clear, concise responses to user questions. Use the policy below to assemble your answer.

Company Name: HorizonTel Communications Industry: Telecommunications Region: North America
📋 Policy Summary: Mobile Service Plan Adjustments
Policy ID: MOB-PLN-2025-03 Effective Date: March 1, 2025 Applies To: All residential and small business mobile customers
Purpose: To ensure customers have transparent and flexible options when modifying or upgrading their existing mobile service plans.
🔄 Plan Changes & Upgrades
Eligibility: Customers must have an active account in good standing (no outstanding balance > $50).
Upgrade Rules:
Device upgrades are permitted once every 12 months if the customer is on an eligible plan.
Early upgrades incur a $99 early-change fee unless the new plan’s monthly cost is higher by at least $15.
Downgrades: Customers can switch to a lower-tier plan at any time; changes take effect at the next billing cycle.
CS Rep Tip: When customers request plan changes, confirm their next billing cycle and remind them that prorated charges may apply. Always check for active device installment agreements before confirming a downgrade.
💰 Billing & Credits
Billing Cycle: Monthly, aligned with the activation date.
Credit Adjustments:
Overcharges under $10 are automatically credited to the next bill.
For amounts >$10, open a “Billing Adjustment – Tier 2” ticket for supervisor review.
Refund Policy:
Refunds are issued to the original payment method within 7–10 business days.
For prepaid accounts, credits are applied to the balance—no cash refunds.
CS Rep Tip: If a customer reports a billing discrepancy within 30 days, you can issue an immediate one-time goodwill credit (up to $25) without manager approval.
🛜 Network & Outage Handling
Planned Maintenance: Customers receive SMS alerts for outages >1 hour.
Unplanned Outages:
Check the internal “Network Status Dashboard” before escalating.
If multiple customers in a region report the same issue, tag the ticket as “Regional Event – Network Ops.”
Compensation: Customers experiencing service interruption exceeding 24 consecutive hours are eligible for a 1-day service credit upon request.
📞 Retention & Cancellations
Notice Period: 30 days for postpaid accounts; immediate for prepaid.
Retention Offers:
Agents may offer up to 20% off the next 3 billing cycles if the customer cites “cost concerns.”
Retention codes must be logged under “RET-SAVE20.”
Cancellation Fee:
Applies only to term contracts (usually $199 flat rate).
Fee waived for verified relocation to non-serviceable area.
CS Rep Tip: Before processing a cancellation, review alternative retention offers—customers frequently stay when offered a temporary discount or bonus data package.
🧾 Documentation Checklist for CS Reps
Verify customer ID and account number.
Check account standing (billing, contracts, upgrades).
Record all interactions in the CRM ticket.
Confirm next billing cycle date for any changes.
Apply standard note template:
“Customer requested [plan/billing/support] change. Informed of applicable fees, next cycle adjustment, and confirmation reference #[ticket].”
⚠️ Compliance & Privacy
All interactions must comply with CCPA and FCC privacy standards.
Do not record or store personal payment information outside the secure billing system.
Use the “Secure Verification Flow” for identity confirmation before discussing account details.
🧠 Example`,
  model: "gpt-4.1-mini",
  modelSettings: {
    temperature: 1,
    topP: 1,
    maxTokens: 2048,
    store: true
  }
});

const approvalRequest = (message: string) => {

  // TODO: Implement
  return true;
}

type WorkflowInput = { input_as_text: string };


// Main code entrypoint
export const runWorkflow = async (workflow: WorkflowInput) => {
  return await withTrace("customer service", async () => {
    const state = {

    };
    const conversationHistory: AgentInputItem[] = [
      { role: "user", content: [{ type: "input_text", text: workflow.input_as_text }] }
    ];
    const runner = new Runner({
      traceMetadata: {
        __trace_source__: "agent-builder",
        workflow_id: "wf_697115611aec8190b882c987dcfb4c5f07f3abe8d861069c"
      }
    });
    const guardrailsInputText = workflow.input_as_text;
    const { hasTripwire: guardrailsHasTripwire, safeText: guardrailsAnonymizedText, failOutput: guardrailsFailOutput, passOutput: guardrailsPassOutput } = await runAndApplyGuardrails(guardrailsInputText, jailbreakGuardrailConfig, conversationHistory, workflow);
    const guardrailsOutput = (guardrailsHasTripwire ? guardrailsFailOutput : guardrailsPassOutput);
    if (guardrailsHasTripwire) {
      return guardrailsOutput;
    } else {
      const classificationAgentResultTemp = await runner.run(
        classificationAgent,
        [
          ...conversationHistory
        ]
      );
      conversationHistory.push(...classificationAgentResultTemp.newItems.map((item) => item.rawItem));

      if (!classificationAgentResultTemp.finalOutput) {
          throw new Error("Agent result is undefined");
      }

      const classificationAgentResult = {
        output_text: JSON.stringify(classificationAgentResultTemp.finalOutput),
        output_parsed: classificationAgentResultTemp.finalOutput
      };
      if (classificationAgentResult.output_parsed.classification == "return_item") {
        const returnAgentResultTemp = await runner.run(
          returnAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...returnAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!returnAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const returnAgentResult = {
          output_text: returnAgentResultTemp.finalOutput ?? ""
        };
        const approvalMessage = "Does this work for you?";

        if (approvalRequest(approvalMessage)) {
            const endResult = {
              message: "Your return is on the way."
            };
            return endResult;
        } else {
            const endResult = {
              message: "What else can I help you with?"
            };
            return endResult;
        }
      } else if (classificationAgentResult.output_parsed.classification == "cancel_subscription") {
        const retentionAgentResultTemp = await runner.run(
          retentionAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...retentionAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!retentionAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const retentionAgentResult = {
          output_text: retentionAgentResultTemp.finalOutput ?? ""
        };
      } else if (classificationAgentResult.output_parsed.classification == "get_information") {
        const informationAgentResultTemp = await runner.run(
          informationAgent,
          [
            ...conversationHistory
          ]
        );
        conversationHistory.push(...informationAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!informationAgentResultTemp.finalOutput) {
            throw new Error("Agent result is undefined");
        }

        const informationAgentResult = {
          output_text: informationAgentResultTemp.finalOutput ?? ""
        };
      } else {
        return classificationAgentResult;
      }
    }
  });
}
