# Example Dataset and Results
This folder contains an export from Scopus
data_Scopus_CSR_influence_consumer_behavior.csv
generated on 26 January 2025, containing 597 records matching the query:

(“CSR influence” OR “Corporate Social Responsibility influence”) AND “consumer behavior”

The dataset was processed using EmbedSLR 1.0.0 in Google Colab, with the SBERT backend and the model all-distilroberta-v1.
The bibliometric audit was limited to the Top 30 resultsfor the research question at hand: 

Does corporate social responsibility (CSR) influence consumer behavior?

The computation took approximately 4 minutes to complete.
Output files are available in the examples/results/ directory.

This process can be replicated, and the results independently verified, when testing the software.

## Installation Google Colab 

```bash
!pip install git+https://github.com/s-matysik/EmbedSLR.git
from embedslr.colab_app import run
run()
