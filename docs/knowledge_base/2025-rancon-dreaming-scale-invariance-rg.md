---
title: "Dreaming Up Scale Invariance via Inverse Renormalization Group"
authors: "Adam Rançon, Ulysse Rançon, Tomislav Ivek, Ivan Balog"
date: "2025-06-04"
url: "https://arxiv.org/abs/2506.04016"
type: "preprint"
---

## Summary

This paper investigates whether minimal neural networks can learn to invert the renormalization group (RG) coarse-graining procedure in the two-dimensional Ising model. Inverting the RG—reconstructing microscopic configurations from coarse-grained states—is formally impossible at the level of deterministic configurations. However, the authors show that it can be approached probabilistically: machine learning models can reconstruct scale-invariant *distributions* without relying on microscopic input. Remarkably, even neural networks with as few as three trainable parameters can learn to generate critical configurations, faithfully reproducing the scaling behavior of observables such as magnetic susceptibility, heat capacity, and Binder cumulants.

A real-space renormalization group analysis of the generated configurations confirms that these minimal models capture not only scale invariance but also reproduce nontrivial eigenvalues of the RG transformation. This means that the networks are not merely mimicking superficial statistical properties but are genuinely learning the RG-relevant structure of the critical distribution. Perhaps even more surprisingly, the authors find that increasing network complexity by introducing multiple layers offers no significant benefit over the minimal model. This finding suggests that simple local rules—akin to those generating fractal structures—are sufficient to encode the universality of critical phenomena.

The paper's implications extend beyond the specific Ising model. By demonstrating that inverse RG can be performed by extremely simple networks, it suggests that the information needed to characterize critical distributions is highly compressible, and that the neural network is essentially learning a compact encoding of the RG flow. This creates opportunities for efficient generative models of statistical ensembles in physics and raises the question of whether similar compressibility holds for other universality classes.

## Relevance

- **Hebbian learning**: While the paper does not explicitly employ Hebbian learning rules, the finding that extremely simple, nearly parameter-free networks can learn inverse RG maps suggests that even very basic local learning mechanisms could, in principle, acquire these mappings. The minimal parameter count is consistent with the kind of simple correlation-based (Hebbian) updates that could shape a network's connectivity to encode scale-invariant distributions.

- **UV-complete physics**: This is the paper's central theme. The entire study is framed in terms of the renormalization group, scale invariance, and critical phenomena. The networks learn to invert the RG flow (going from IR/UV to UV), reproduce RG eigenvalues, and generate configurations at the critical point of the Ising model—a canonical example from statistical field theory.

- **Expressivity**: The paper makes a striking statement about representational efficiency: a three-parameter network suffices to represent the inverse RG map for the 2D Ising critical distribution. This implies that the effective complexity (or expressivity requirement) of the inverse RG map is extremely low, at least for this universality class, and that deeper or wider networks offer no advantage in representing this particular function.

## Keywords

- Inverse renormalization group
- Scale invariance
- Ising model
- Critical phenomena
- Neural network generative models
- Universality classes

## Verbatim Snippet

> We demonstrate that even neural networks with as few as three trainable parameters can learn to generate critical configurations, reproducing the scaling behavior of observables such as magnetic susceptibility, heat capacity, and Binder ratios. Surprisingly, we find that increasing network complexity by introducing multiple layers offers no significant benefit.
