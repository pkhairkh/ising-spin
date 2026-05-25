---
title: "Renormalization Group for Deep Neural Networks: Universality of Learning and Scaling Laws"
authors: "Gorka Peraza Coppola, Moritz Helias, Zohar Ringel"
date: "2025-10-29"
url: "https://arxiv.org/abs/2510.25553"
type: "preprint"
---

## Summary

This paper develops a renormalization group (RG) framework to analyze self-similarity and its breakdown in learning curves for a class of weakly non-linear (non-lazy) neural networks trained on power-law distributed data. The work is motivated by the observation that power laws and weak forms of universality pervade both natural datasets and deep learning models, suggesting that RG ideas—which were developed to understand self-similarity and universality in physical systems—should be applicable to deep learning. The authors show that this is indeed the case, but with important differences from conventional perturbative RG that arise from features specific to neural networks.

Two features often neglected in standard treatments—spectrum discreteness and lack of translation invariance—lead to both quantitative and qualitative departures from conventional perturbative RG. In particular, the concept of scaling intervals naturally replaces that of scaling dimensions. In conventional RG, scaling dimensions classify operators by how they transform under dilatations of continuous space; in the neural network context, the discreteness of the spectrum (arising from finite data or finite network size) means that scaling behavior is only defined over finite intervals rather than being an asymptotic property. Despite these differences, the framework retains key RG features: it enables the classification of perturbations as relevant or irrelevant, and reveals a form of universality at large data limits.

The most significant finding is the identification of a Gaussian Process-like UV fixed point that governs universality in the large-data limit. This fixed point is the analog of the Gaussian (free) fixed point in quantum field theory: it describes the theory at the UV (microscopic) end of the RG flow, where the network's behavior is governed by the simplest possible statistics (a Gaussian process). Perturbations away from this fixed point correspond to non-Gaussian corrections that arise from finite network width or non-linear training dynamics. The classification of these perturbations as relevant or irrelevant determines which features of the learning curve survive under RG flow and which are washed out, providing a principled understanding of universality in deep learning.

## Relevance

- **Hebbian learning**: The RG framework developed in this paper is directly applicable to understanding Hebbian learning dynamics. Hebbian rules are local and operate on individual synaptic connections, corresponding to microscopic (UV) updates. The RG flow from the UV to the IR determines which features of these local updates survive at the macroscopic (network) level. The classification of perturbations as relevant or irrelevant provides a criterion for whether specific Hebbian learning rules will have a lasting impact on network behavior or will be renormalized away under coarse-graining. The identification of the GP-like UV fixed point suggests that Hebbian learning rules that are consistent with the GP limit (i.e., that preserve the Gaussian structure of the network at large width) will be UV-natural in the sense of technical naturalness.

- **UV-complete physics**: The identification of a Gaussian Process-like UV fixed point is the paper's central contribution to UV-complete physics. In quantum field theory, a UV fixed point is a scale-invariant theory that governs the high-energy (short-distance) behavior of the system. The GP-like fixed point plays this role for neural networks: it governs the behavior of the network at the microscopic scale (large data, small perturbations). The RG flow away from this fixed point generates the effective description at coarser scales (smaller data, larger perturbations). The replacement of scaling dimensions by scaling intervals is a significant departure from standard QFT that reflects the discrete, finite nature of neural network spectra, and may have implications for whether neural network field theories can be UV-complete in the traditional sense.

- **Expressivity**: The RG framework provides a field-theoretic understanding of expressivity through the lens of universality and relevant perturbations. Near the UV fixed point, the expressivity of the network is minimal (the GP limit has limited representational capacity). Relevant perturbations away from the fixed point—those that grow under RG flow—increase the expressivity of the effective theory at coarser scales. The universality at the large-data limit implies that different architectures with the same relevant perturbations will have the same expressive capacity, providing a principled explanation for why many different neural network architectures achieve similar performance on the same tasks.

## Keywords

- Renormalization group
- UV fixed point
- Gaussian process limit
- Scaling laws
- Universality
- Scaling intervals

## Verbatim Snippet

> The framework retains key RG features: it enables the classification of perturbations as relevant or irrelevant, and reveals a form of universality at large data limits, governed by a Gaussian Process-like UV fixed point.
