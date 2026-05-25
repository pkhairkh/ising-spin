---
title: "Information Bottleneck-Based Hebbian Learning Rule Naturally Ties Working Memory and Synaptic Updates"
authors: "Kyle Daruwalla, Mikko Lipasti"
date: "2021-11-24"
url: "https://arxiv.org/abs/2111.13187"
type: "preprint"
---

## Summary

This work proposes a biologically plausible alternative to backpropagation for training deep neural networks, grounded in the information bottleneck (IB) principle. The authors decompose the IB-derived weight update into two components: a purely local, Hebbian term that depends only on the current input sample, and a global, modulatory signal computed from a batch of samples. The key insight is that this global modulatory signal can itself be learned by an auxiliary circuit that operates as a working-memory reservoir, thereby linking the capacity of working memory directly to the effective batch size used during learning. Evaluated on both synthetic benchmarks and image classification tasks such as MNIST, the rule demonstrates that greater working-memory capacity translates into improved learning performance. The paper positions itself as a first step toward understanding the mechanistic role of memory in synaptic plasticity.

The central contribution lies in bridging two historically separate ideas: the information bottleneck as a normative principle for layer-wise learning in deep networks, and Hebbian plasticity as a biologically grounded update rule. Previous IB-based approaches required batches of data to estimate mutual-information quantities, making them biologically implausible. By showing that only a small auxiliary reservoir is needed to approximate the batch-dependent modulatory signal, the authors provide a concrete neural mechanism that could, in principle, be realized in cortical microcircuits. The work thus opens a pathway for designing spiking network architectures where memory and learning are co-dependent rather than independently specified.

From a broader theoretical perspective, the paper implicitly touches on questions of representational capacity: the quality of the learned representation is constrained by the interplay between the local Hebbian component and the capacity of the working-memory circuit that supplies the modulatory signal. This suggests that the expressivity of the network is not merely a function of depth and width, but is shaped by the learning rule itself—a theme that resonates with ideas from statistical physics about how local rules can give rise to global order.

## Relevance

- **Hebbian learning**: The paper's core contribution is a Hebbian local learning rule derived from the information bottleneck. The weight update is decomposed into a purely Hebbian term (pre-post correlation on the current sample) and a global modulatory signal, making the rule biologically plausible while retaining IB-optimal compression behavior.

- **UV-complete physics**: The information bottleneck principle, which governs the learning objective, has deep connections to renormalization group flow and coarse-graining in statistical physics. The IB objective trades off compression (analogous to RG coarse-graining) against prediction accuracy, and the layer-wise training it induces can be viewed as a kind of RG-like successive compression through the network layers.

- **Expressivity**: The paper highlights how the capacity of the working-memory reservoir directly constrains the quality of learned representations. The expressivity of the overall system is thus shaped by the learning rule's ability to approximate batch-level statistics, suggesting that representational power is not purely architectural but also depends on the plasticity mechanism.

## Keywords

- Hebbian learning
- Information bottleneck
- Working memory
- Biologically plausible learning
- Layer-wise training
- Spiking neural networks

## Verbatim Snippet

> Our work takes a different approach by decomposing the weight update into a local and global component. The local component is Hebbian and only depends on the current sample. The global component computes a layer-wise modulatory signal that depends on a batch of samples. We show that this modulatory signal can be learned by an auxiliary circuit with working memory (WM) like a reservoir.
