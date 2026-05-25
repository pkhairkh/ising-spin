---
title: "Topological Effects in Neural Network Field Theory"
authors: "Christian Ferko, James Halverson, Vishnu Jejjala, Brandon Robinson"
date: "2026-04-02"
url: "https://arxiv.org/abs/2604.02313"
type: "preprint"
---

## Summary

This paper extends the neural network field theory (NN-FT) construction to topological settings by including discrete parameters that label topological quantum numbers. The NN-FT framework formulates field theory as a statistical ensemble of fields defined by a network architecture and a density on its parameters. By incorporating topological data—such as winding numbers—into the network's parameter space, the authors show that NN-FT can capture topological phenomena that are invisible to perturbative analysis but play a crucial role in determining the theory's physical behavior.

The paper demonstrates two major topological results within the NN-FT framework. First, it recovers the Berezinskii-Kosterlitz-Thouless (BKT) transition, including both the spin-wave critical line at low temperatures and the proliferation of vortices at high temperatures. The BKT transition is a topological phase transition driven by the unbinding of vortex-antivortex pairs, and it is one of the most important examples of a phase transition that cannot be understood within Landau's symmetry-breaking framework. Its recovery in NN-FT shows that neural network field theories can capture non-perturbative, topological phenomena that go beyond the reach of the Feynman diagram expansion.

Second, the paper verifies the T-duality of the bosonic string within the NN-FT framework. T-duality is a remarkable symmetry of string theory that relates a string compactified on a circle of radius R to one compactified on a circle of radius 1/R, exchanging momentum and winding modes. The verification includes showing invariance under the exchange of momentum and winding on S^1, the transformation of the sigma model couplings according to the Buscher rules on constant toroidal backgrounds, the enhancement of the current algebra at the self-dual radius, and non-geometric T-fold transition functions. T-duality is a UV/IR duality—it relates the UV behavior of one description to the IR behavior of another—and its verification in NN-FT demonstrates that neural network field theories can capture this fundamental aspect of string-theoretic UV completion.

## Relevance

- **Hebbian learning**: The inclusion of topological parameters in NN-FT has implications for understanding how Hebbian learning interacts with topological features of the loss landscape. Topological defects such as vortices represent metastable states that can trap the dynamics; Hebbian learning rules, being local and gradient-like, may or may not be able to overcome these topological barriers depending on their specific form. The BKT transition in NN-FT provides a concrete example where the interplay between local dynamics (analogous to Hebbian updates) and topological defects (vortices) determines the global behavior of the system, suggesting that similar phenomena may arise in biological neural networks with Hebbian plasticity.

- **UV-complete physics**: The topological effects studied in this paper are directly relevant to UV completion. The BKT transition is a non-perturbative phenomenon that cannot be captured by any finite order of perturbation theory, demonstrating that UV-complete theories must account for topological effects. T-duality is a UV/IR duality that is central to string theory's status as a UV-complete theory of quantum gravity: it shows that the distinction between UV and IR physics is not absolute but depends on the duality frame. The verification of T-duality in NN-FT means that neural network field theories can encode this UV/IR mixing, providing a concrete setting for studying how UV-complete behavior emerges from the network's parameter space structure.

- **Expressivity**: Topological effects constrain and enrich the expressivity of neural network field theories. The discrete topological quantum numbers (winding numbers) are global features that cannot be captured by local perturbative expansions but are essential for the full representational capacity of the theory. The BKT transition demonstrates that the hypothesis class of the NN-FT changes qualitatively as a function of the temperature (or, equivalently, the width of the network): at low temperatures, the theory is constrained by topological order, while at high temperatures, vortex proliferation destroys this constraint. This provides a field-theoretic understanding of phase-dependent expressivity in neural networks.

## Keywords

- Topological effects
- BKT transition
- T-duality
- Winding numbers
- UV/IR duality
- Neural network field theory

## Verbatim Snippet

> We recover the Berezinskii-Kosterlitz-Thouless transition, including the spin-wave critical line and the proliferation of vortices at high temperatures. We also verify the T-duality of the bosonic string, showing invariance under the exchange of momentum and winding on S^1.
