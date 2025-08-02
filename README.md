# EmbedSLR &nbsp;🚀


> **EmbedSLR** is a concise Python framework that performs **deterministic, embedding‑based ranking** of publications and a **bibliometric audit** (keywords, authors, citations) to speed up the screening phase in systematic literature reviews.

* Fully reproducible – no stochastic LLM components  
* Five interchangeable embedding back‑ends (local SBERT, OpenAI, Cohere, Jina, Nomic)  
* **Wizard** (interactive CLI) and **Colab GUI** for zero‑config onboarding  
* Generates a ready‑to‑share `biblio_report.txt` dashboard  

---

## ✨ Quick start (Google Colab)

```bash
!pip install git+https://github.com/s-matysik/EmbedSLR_.git
from embedslr.colab_app import run
run()

## 📝 Citing

If you use **EmbedSLR** in scientific work, please cite our accompanying *Software X* article:

```bibtex
@article{Matysik2025,
  title   = {EmbedSLR - an open Python framework for deterministic embeddingbased screening and bibliometric validation in systematic literature reviews },
  author  = {Matysik, S., Wiśniewska, J., Frankowski P.K.},
  journal = {SoftwareX},
  year    = {2025},
  note    = {in press},
  doi     = {10.1016/j.softx.2025.XXXXXX},
  url     = {https://doi.org/10.1016/j.softx.2025.XXXXXX}
}
