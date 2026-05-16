"""
Cost Estimation Agents
Specialized LangChain agents for automotive repair cost estimation.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys — try local .env first, then fall back to multi-agent-research-system/.env
load_dotenv(Path(__file__).parent / ".env")

# Import local tools
from tools import web_search, scrape_url

from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── LLM Setup ───────────────────────────────────────────────
def get_llm():
    """Returns Groq LLM if API key exists, else falls back to Gemini."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            print("[INFO] Using Groq (Llama 3.3) as the primary LLM.")
            return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        except Exception as e:
            print(f"[WARN] Groq initialization failed: {e}. Falling back to Gemini.")
    
    print("[INFO] Using Gemini as the fallback LLM.")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

llm = get_llm()


# ─── Agents ──────────────────────────────────────────────────
def build_search_agent():
    """Search agent that finds part prices and repair costs."""
    return create_react_agent(model=llm, tools=[web_search])


def build_reader_agent():
    """Reader agent that scrapes automotive pricing sites."""
    return create_react_agent(model=llm, tools=[scrape_url])


# ─── Cost Writer Chain ────────────────────────────────────────
cost_writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert automotive repair cost estimator working for an insurance company.
Your job is to generate accurate, itemized repair cost estimates based on detected vehicle damage.

Always provide TWO pricing tiers:
- OEM (Original Equipment Manufacturer) — premium quality, higher cost
- Aftermarket — budget-friendly, widely available

Use ₹ (Indian Rupees) for pricing. Be specific with numbers based on the research data provided.
Format your output in clean markdown with tables."""),

    ("human", """Generate a detailed repair cost estimate for the following vehicle:

**Vehicle:** {vehicle_year} {vehicle_brand} {vehicle_model}
**Detected Damage:** {damages}

**Research Data (use this for pricing):**
{research}

---

Output the estimate using this EXACT structure:

## 🚗 VEHICLE REPAIR COST ESTIMATE

**Vehicle:** {vehicle_year} {vehicle_brand} {vehicle_model}
**Detected Damage:** {damages}

---

## 📋 ITEMIZED REPAIR COSTS

| Damage Part | OEM Price (₹) | Aftermarket Price (₹) | Labor Cost (₹) | Total OEM (₹) | Total Aftermarket (₹) |
|-------------|:---:|:---:|:---:|:---:|:---:|
[Fill one row for each damaged part with realistic prices]

---

## 💰 TOTAL COST SUMMARY

| Option | Parts Cost | Labor Cost | **Grand Total** |
|--------|:---:|:---:|:---:|
| 🏆 OEM (Premium) | ₹X | ₹Y | **₹Z** |
| 💡 Aftermarket (Budget) | ₹X | ₹Y | **₹Z** |

---

## 🔧 REPAIR RECOMMENDATIONS

- [Specific recommendation 1 based on the damage type]
- [Specific recommendation 2]
- [Priority repairs vs deferrable repairs]

---

## 📍 SUGGESTED REPAIR CENTERS

- **Authorized Service Center** — Best for OEM parts & warranty coverage
- **Multi-brand Workshop** — Good for aftermarket options at 40-60% lower cost
- [Any specific recommendation based on city if available]

---

## ⏱️ ESTIMATED REPAIR TIMELINE

[Estimate days required for each damage type and overall]
"""),
])

cost_writer_chain = cost_writer_prompt | llm | StrOutputParser()


# ─── Cost Critic Chain ────────────────────────────────────────
cost_critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior automotive insurance adjuster with 15+ years of experience across India.
Your role is to review cost estimates and flag any unrealistic pricing — both overestimates and underestimates.
Be concise, factual, and helpful to the vehicle owner."""),

    ("human", """Review this repair cost estimate for a **{vehicle}** with **{damages}**.

**Estimate to Review:**
{report}

---

Provide your validation in this EXACT format:

## ✅ COST VALIDATION REPORT

**Verdict:** [APPROVED ✅ / NEEDS REVIEW ⚠️ / ADJUSTED 🔄]

**Realistic Price Range:** ₹[MINIMUM] – ₹[MAXIMUM]

**Confidence Level:** [High / Medium / Low] — [one sentence reason]

---

**Assessment:**
[2-3 sentences on whether the estimate is realistic for this vehicle segment and damage type in the Indian market]

---

**Red Flags (if any):**
- [Overpriced / underpriced item if found, or "None identified"]

---

**💡 Owner's Recommendation:**
[One clear, actionable recommendation — e.g., "Get 3 quotes before accepting", "OEM is worth it for this part", etc.]
"""),
])

cost_critic_chain = cost_critic_prompt | llm | StrOutputParser()
