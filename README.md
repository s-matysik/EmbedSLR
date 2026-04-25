# EmbedSLR &nbsp;🚀


> **EmbedSLR** is a concise Python framework that performs **deterministic, embedding‑based ranking** of publications and a **bibliometric audit** (keywords, authors, citations) to speed up the screening phase in systematic literature reviews.

* Fully reproducible – no stochastic LLM components  
* Five interchangeable embedding back‑ends (local SBERT, OpenAI, Cohere, Jina, Nomic)  
* **Wizard** (interactive CLI) and **Colab GUI** for zero‑config onboarding  
* Generates a ready‑to‑share `biblio_report.txt` dashboard  

---


---

## ✨ Quick start (Google Colab)

```bash
!pip install git+https://github.com/s-matysik/EmbedSLR.git
from embedslr.colab_app import run
run()

```

## 📝 Citing

If you use **EmbedSLR** in scientific work, please cite us:

```bibtex
{
   title   = {EmbedSLR: an open-source python framework for efficient embedding-based screening and bibliometric validation in systematic literature review},
  author  = {Matysik, S., Wiśniewska, J., Frankowski, P.K.},
  year    = {2025},
  journal = {SoftwareX},
  volume = {32},
  url     = {https://doi.org/10.1016/j.softx.2025.102416}
}
