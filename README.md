# Gideon – ERP-Powered AI Assistant

Gideon is an AI-driven assistant integrated into Odoo ERP to automate financial analysis, document intelligence, and inventory optimization in secure on-premise environments.  
It leverages **Llama 3 (8B)** via Ollama, **RAG pipelines**, a modular **Docker-based architecture**, and **ERP-aware LLM function calling** to deliver enterprise-ready automation.

📄 **Accepted for publication at IEEE PuneCon 2025**

---

## 🚀 Key Features

### 💰 AI-Powered Expense Analysis
- Analyzes **overall, approved, and pending** expenses  
- Detects **monthly, quarterly, and yearly** financial patterns  
- Generates **interpretable insights** for data-driven decision-making  

---

### 📁 Document Intelligence + Conversational Querying
- Supports **PDF, Word, Excel, and text** formats  
- Extracts **text, tables, and structured data**  
- Uses **embeddings + semantic search** for RAG-driven responses  
- Provides **citation-backed answers** to natural language questions  

---

### 📊 Inventory Intelligence
- Performs **ABC Classification** for value-based inventory segmentation  
- Generates **product reorder recommendations** using demand + lead times  
- Computes **inventory turnover ratios** to identify slow/fast-moving items  

---

### 📦 ERP Data Mapping
- Maps extracted data into **Sales, HR, Inventory, Accounting** modules  
- Includes **user confirmation checkpoints** to prevent incorrect entries  
- Ensures **auditability and traceability** across workflows  

---

## 🏗️ System Architecture

Gideon is built using a fully containerized, modular architecture:

- **Odoo 18 ERP**
- **PostgreSQL 13** (primary ERP database)
- **ChromaDB** (vector embeddings storage)
- **Ollama + Llama 3 8B** (on-prem LLM inference)
- **RAG Pipeline** for enterprise document intelligence
- **LLM Integration Base & Easy AI Chat** modules for function calling and embeddings

Deployment uses **Docker Compose**, providing:

- Scalability  
- Reproducibility  
- Secure on-prem LLM operations  

---

## 🛠 Tech Stack

- **Languages:** Python  
- **Frameworks:** Odoo, LangChain, FastAPI (optional)  
- **Models:** Llama 3 8B (Ollama)  
- **Databases:** PostgreSQL, ChromaDB  
- **Tools:** Docker, Docker Compose, RAG Pipelines  

---

## 📚 Research Contribution

This project introduces:

- A modular **ERP-focused AI integration framework**  
- A multi-stage **enterprise document analysis pipeline**  
- **Hybrid LLM routing** with audit-ready workflows  

📄 **Accepted for publication at IEEE PuneCon 2025**

---

## ▶️ Usage

Once running:

1. Open **Odoo ERP** in the browser  
2. Navigate to the **AI Assistant** module  
3. Upload documents, ask questions, or trigger expense/inventory analysis  
4. Review AI-generated insights directly inside Odoo  

---

## 💡 Motivation

Traditional ERP systems lack intelligent automation for unstructured documents,
financial summaries, and inventory decision-making.  
Gideon bridges this gap by integrating LLM intelligence directly inside ERP workflows,
running fully on-prem to meet enterprise privacy requirements.

---

## 🧪 Future Enhancements

- Voice-enabled conversational interface  
- Financial forecasting using time-series models  
- Support for multi-LLM routing (Claude, GPT, Phi)  
- Zero-shot document classification for automation  
- RAG caching and performance optimization  
