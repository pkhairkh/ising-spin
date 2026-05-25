---
title: "The Importance of Being Empty: A Spectral Approach to Hopfield Neural Networks with Diluted Examples"
authors: "Elena Agliari, Alberto Fachechi, Domenico Luongo"
date: "2025-03-19"
url: "https://arxiv.org/abs/2503.15353"
type: "preprint"
---

## Summary

This paper investigates Hopfield neural networks in which the Hebbian coupling matrix is constructed from patterns that may be partially incomplete—containing blank or "empty" entries that simulate missing or dropped data. The authors consider three learning regimes: storing definite ground-truth patterns, learning from supervised (labeled) examples, and learning from unsupervised (unlabeled) examples. In each case, the examples are noisy versions of underlying ground-truth patterns, and the blank entries model real-world data sparsity. By exploiting and extending the Marchenko-Pastur theorem from random matrix theory, the authors derive the spectral distribution of the coupling matrices in all three scenarios. This spectral knowledge then allows them to analytically characterize the stability and attractiveness of stored patterns, as well as the network's generalization capabilities.

A surprising and practically significant finding is that the presence of blank entries can actually *improve* network performance under specific conditions, suggesting that deliberate data sparsification could serve as a beneficial training strategy. This counter-intuitive result is corroborated by extensive Monte Carlo simulations and holds even for structured (non-random) datasets. The authors further demonstrate that the Hebbian coupling matrix, when built from sparse examples, can be recovered as the fixed point of a gradient-descent algorithm with dropout, drawing a formal connection between Hebbian learning and modern regularized optimization.

The work is notable for its use of spectral methods from statistical physics to provide exact analytical results about the representational properties of Hebbian networks. The connection between dropout regularization and the fixed point of the Hebbian matrix is particularly intriguing, as it suggests that the common practice of randomly dropping network units during training has a natural interpretation in terms of Hebbian learning from incomplete data.

## Relevance

- **Hebbian learning**: The entire analysis is built around Hebbian coupling matrices in Hopfield networks. The paper studies how Hebbian learning from incomplete (diluted) examples affects pattern storage, stability, and retrieval, making Hebbian plasticity the central mechanism under investigation.

- **UV-complete physics**: The Marchenko-Pastur theorem and random matrix theory are core tools from statistical physics. The fixed-point analysis of the coupling matrix connects to RG-like ideas—specifically, the paper shows that the Hebbian matrix emerges as a stable fixed point of a dropout-based gradient flow, echoing how RG fixed points characterize stable coarse-grained descriptions.

- **Expressivity**: The paper directly addresses the capacity and generalization capabilities of Hebbian networks. By characterizing the spectral distribution of the coupling matrix, it provides a quantitative account of how many patterns can be stably stored and how well the network generalizes from noisy, incomplete examples—questions that lie at the heart of network expressivity.

## Keywords

- Hebbian learning
- Hopfield networks
- Random matrix theory
- Marchenko-Pastur theorem
- Generalization
- Dropout regularization

## Verbatim Snippet

> Finally, we demonstrate that the Hebbian matrix, built on sparse examples, can be recovered as the fixed point of a gradient descent algorithm with dropout, over a suitable loss function.
