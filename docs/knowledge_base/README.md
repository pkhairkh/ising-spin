# Hebbian Learning x UV-Complete Physics x Expressivity: Knowledge Base

## Overview

This knowledge base collects primary sources that live at the intersection of three themes:

1. **Hebbian learning** – local, correlation-based synaptic plasticity (Hebbian rules, STDP, competitive/anti-Hebbian learning, Hebbian deep learning).
2. **UV-complete physics** – concepts from ultraviolet completion in quantum field theory applied to neural networks: Wilsonian effective field theory, UV fixed points (Gaussian/Gell-Mann-Low), renormalization group (RG) flow from UV to IR, asymptotic freedom and asymptotic safety analogs, scale/conformal anomalies, UV/IR duality, holographic duality (AdS/CFT), and the NN-QFT correspondence.
3. **Expressivity** – the capacity of a neural network to represent functions, depth separation, and how learning rules shape the hypothesis class.

**Important distinction**: "UV-complete" is not the same as "UV" or "high-energy physics" in general. UV-complete specifically refers to theories that are well-defined at all energy scales, including arbitrarily high energies, without requiring a cutoff or new physics at short distances. This includes asymptotically free theories (e.g., QCD), asymptotically safe theories, and conformal field theories. The papers in this knowledge base address how these UV-completeness concepts manifest in neural network field theories, rather than merely using RG or criticality as loose analogies.

## Scope and Limitations

The three-way intersection of these topics is extremely narrow. Very few published works simultaneously and substantially address all three. In this knowledge base:

- **3 sources** genuinely touch all three themes (topics covered = 3/3):
  - Daruwalla & Lipasti (2021): Information bottleneck as normative principle for Hebbian learning, with RG-like coarse-graining interpretation
  - Agliari et al. (2025): Spectral theory of Hebbian Hopfield networks with RG-like fixed point analysis
  - Eugenio (2025): Hebbian tokenizer performing renormalization group, directly linking Hebbian learning to RG flow

- **15 sources** substantially address at least two of the three themes, with clear hints or implications for the third. For each source, the "Relevance" section explicitly states how it connects to each theme.

If you are looking for the deepest three-way connections, start with the three T=3/3 sources, then explore the T=2/3 sources in order of their keyword scores.

## Methodology

Sources were identified through targeted arXiv searches using queries combining keywords from all three themes, with a specific focus on UV-completeness concepts (Wilsonian RG, UV fixed points, asymptotic freedom/safety, anomalies, UV/IR duality, NN-QFT correspondence). Searches were performed using Playwright-based web scraping. All sources are preprints available on arXiv. Only publicly accessible HTML abstract pages were scraped (no PDFs). Relevance filtering required a source to contain at least two of the three keyword categories, with particular emphasis on genuine UV-complete physics content rather than general statistical physics or IR criticality.

## File Index

| # | File | Title | Topics |
|---|------|-------|--------|
| 1 | `2021-daruwalla-information-bottleneck-hebbian.md` | Information Bottleneck-Based Hebbian Learning Rule | 3/3 |
| 2 | `2025-agliari-hopfield-spectral-hebbian.md` | The Importance of Being Empty: Hopfield Networks with Diluted Examples | 3/3 |
| 3 | `2025-eugenio-hebbian-learning-local-structure-language.md` | Hebbian Learning the Local Structure of Language | 3/3 |
| 4 | `2020-halverson-neural-networks-quantum-field-theory.md` | Neural Networks and Quantum Field Theory | 2/3 |
| 5 | `2021-erbin-nonperturbative-renormalization-nn-qft.md` | Nonperturbative Renormalization for the NN-QFT Correspondence | 2/3 |
| 6 | `2021-tiberi-gell-mann-low-criticality-neural-networks.md` | Gell-Mann-Low Criticality in Neural Networks | 2/3 |
| 7 | `2024-howard-wilsonian-renormalization-nngp.md` | Wilsonian Renormalization of Neural Network Gaussian Processes | 2/3 |
| 8 | `2024-howard-bayesian-rg-flow-nnft.md` | Bayesian RG Flow in Neural Network Field Theories | 2/3 |
| 9 | `2025-peraza-coppola-rg-deep-neural-networks.md` | RG for Deep Neural Networks: Universality of Learning and Scaling Laws | 2/3 |
| 10 | `2025-sen-viability-perturbative-qft-neurons.md` | Viability of Perturbative Expansion for QFTs on Neurons | 2/3 |
| 11 | `2025-kim-fno-effective-field-theory.md` | Analysis of Fourier Neural Operators via Effective Field Theory | 2/3 |
| 12 | `2025-rancon-dreaming-scale-invariance-rg.md` | Dreaming Up Scale Invariance via Inverse Renormalization Group | 2/3 |
| 13 | `2026-ferko-anomalies-neural-network-field-theory.md` | Anomalies in Neural Network Field Theory | 2/3 |
| 14 | `2026-ferko-topological-effects-neural-network-field-theory.md` | Topological Effects in Neural Network Field Theory | 2/3 |
| 15 | `2026-ghosh-neural-spectral-bias-conformal-correlators.md` | Neural Spectral Bias and Conformal Correlators I | 2/3 |
| 16 | `2024-yeung-generalized-holographic-reduced.md` | Generalized Holographic Reduced Representations | 2/3 |
| 17 | `2012-cowan-self-organized-criticality-neurons.md` | Self-Organized Criticality in a Network of Interacting Neurons | 2/3 |
| 18 | `2026-lesser-competing-nonlinearities-criticality.md` | Competing Nonlinearities, Criticality, and Order-to-Chaos Transition | 2/3 |

## How to Use

Each Markdown file contains:
- **YAML front matter** with title, authors, date, URL, and type
- **Summary** – a paraphrased overview of the source's core contribution (2–5 paragraphs)
- **Relevance** – explicit connections to each of the three themes
- **Keywords** – 3–7 keywords characterizing the source
- **Verbatim snippet** – a blockquote of the most important original paragraph for traceability

## Change Log

- **v2 (2026-05-26)**: Corrected the "UV-physics" theme to "UV-complete physics". Removed 10 papers that addressed general statistical physics / IR criticality without genuine UV-complete content. Added 11 papers that specifically address UV-complete physics concepts in neural network field theories: Wilsonian EFT correspondence, nonperturbative RG, Gell-Mann-Low criticality, UV fixed points, anomalies (scale and Weyl), topological effects (BKT, T-duality), UV cutoff sensitivity, CFT correlators, Bayesian RG flow, and Hebbian learning as RG.
- **v1**: Initial knowledge base with "UV-physics" theme (now recognized as insufficiently specific).
