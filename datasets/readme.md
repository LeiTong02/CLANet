# AstraZeneca Global Cell Bank Brightfield Image Dataset 
## Background 
Cell line authentication is a critical aspect of biomedical research to ensure accurate and reliable results. Traditional methods, such as short tandem repeat (STR) analysis, have limitations due to their cost and time requirements. In recent years, deep neural networks and computer vision techniques have shown great promise in cost-effective cellular image analysis. However, the lack of centralized datasets raises questions about whether cell image classification can replace or support cell line authentication. Considering the advantages of brightfield imaging, including simplified setup, the ability to track cell genealogies in long-term experiments, and its low cost, we aim to automate the cell line authentication process using deep learning and computer vision techniques based on brightfield cell images.

## Dataset Description 
To facilitate our research, we have curated a dataset consisting of sample images from the registered cell lines in the AstraZeneca Global Cell Bank (AZGCB). Our focus primarily centers around commonly used cancer cell lines, selected based on the frequency of requests for these cell lines. The dataset includes brightfield images collected during normal cell growth experiments where no compounds or medicines were used. The base medium was supplemented with 10% Fetal Bovine Serum (Sigma) and 1× GlutaMAX (Gibco), unless otherwise specified. Currently, the dataset comprises 187,990 brightfield images from 44 different cell lines across 104 experimental batches, as listed in Table 1.


## Data Collection 
Fig. 1 provides an example of our data collection process. Cells were thawed and seeded into flasks (Corning) at a density of 0.5-2×10^6 cells per flask. The flasks were then placed in the Incucyte S3 system (Essen Bioscience, Sartorius), and brightfield images were captured from various locations within the flask at regular intervals ranging from 1 to 8 hours. This process was conducted over a time period of 3 to 18 days. All images were exported in JPEG format with a size of 1408×1040 (96×96dpi).



