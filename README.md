# EmbedSLR
embedslr = Quick publication ranking + bibliometric metrics for systematic literature reviews.

Open‑source toolkit that speeds up **screening and validation of publications**
in systematic literature reviews using modern embedding models.

# EmbedSLR &nbsp;🚀


> **EmbedSLR** is a concise Python framework that performs **deterministic, embedding‑based ranking** of publications and a **bibliometric audit** (keywords, authors, citations) to speed up the screening phase in systematic literature reviews.

* Fully reproducible – no stochastic LLM components  
* Five interchangeable embedding back‑ends (local SBERT, OpenAI, Cohere, Jina, Nomic)  
* **Wizard** (interactive CLI) and **Colab GUI** for zero‑config onboarding  
* Generates a ready‑to‑share `biblio_report.html` dashboard  

---

## ✨ Quick start (Google Colab)

```bash
!pip install git+https://github.com/s-matysik/EmbedSLR_.git
from embedslr.colab_app import run
run()

