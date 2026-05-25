---
title: "Neural Spectral Bias and Conformal Correlators I: Introduction and Applications"
authors: "Kausik Ghosh, Sidhaarth Kumar, Vasilis Niarchos, Andreas Stergiou"
date: "2026-04-20"
url: "https://arxiv.org/abs/2604.18686"
type: "preprint"
---

## Summary

This paper demonstrates that simple feed-forward neural networks can accurately compute correlation functions of conformal field theories (CFTs) on a line, establishing a striking alignment between the inductive bias of gradient-based training and the mathematical structure of conformal field theories. CFTs are quantum field theories that are invariant under conformal transformations (scale, rotation, and special conformal transformations), and they play a central role in modern theoretical physics as UV-complete fixed points of the renormalization group flow. The fact that neural networks can learn CFT correlators with remarkable accuracy suggests a deep connection between the spectral bias of gradient-based optimization and the smoothness properties of conformal correlators.

The key methodological innovation is a minimal-data approach: by optimizing a neural network solely on crossing symmetry and providing only the scaling dimension of the leading non-trivial operator and the correlator's value at a single anchor point, the network can reconstruct target physical correlators to within a few percent accuracy. Crossing symmetry is the fundamental consistency condition of CFTs, expressing the equivalence of different operator product expansion channels. The fact that this single constraint, combined with minimal data, suffices to determine the correlator reflects the extraordinary rigidity of CFTs—precisely the property that makes them UV-complete.

The authors establish the robustness of this approach across a broad class of theories and dimensions, including generalized free fields, contact and one-loop Witten diagrams in AdS2 (which are holographic correlators from the AdS/CFT correspondence), unitary and non-unitary 2d minimal models, the 3d Ising model (a paradigmatic example of a critical theory), and half-BPS correlators in 4d N=4 super-Yang-Mills theory (the most symmetric quantum field theory known), together with several thermal two-point functions including those of the 3d Ising model. The authors argue that the remarkable alignment between neural networks and CFTs stems from the spectral bias of gradient-based training, which heavily favors smooth functions. They ground this connection by analyzing the smoothness of conformal correlators using fractional Sobolev semi-norms, Chebyshev spectral decompositions, and a measure based on curvature.

## Relevance

- **Hebbian learning**: While the paper uses gradient-based training rather than Hebbian learning specifically, the spectral bias phenomenon it identifies has direct implications for Hebbian rules. Spectral bias—the tendency of gradient-based training to learn low-frequency (smooth) functions before high-frequency (rough) ones—is also a natural property of Hebbian learning, which is fundamentally a correlation-based rule that captures the dominant statistical modes of the input. The smoothness of CFT correlators means that they are naturally learnable by any method with a spectral bias toward smooth functions, including Hebbian learning. This suggests that Hebbian learning may be particularly well-suited for learning representations that have conformal symmetry, a property that is directly connected to UV completeness.

- **UV-complete physics**: CFTs are UV-complete theories by definition: they are scale-invariant fixed points of the RG flow that are well-defined at all energy scales. The paper's demonstration that neural networks can accurately learn CFT correlators means that neural networks can capture UV-complete physics. The crossing symmetry constraint, which is the key to the minimal-data approach, is a consistency condition that is necessary for the UV completeness of the CFT. The inclusion of holographic correlators (Witten diagrams in AdS) connects to the AdS/CFT correspondence, which provides a UV completion of the CFT through its dual gravitational description in anti-de Sitter space. The verification on 4d N=4 super-Yang-Mills—the gold standard of UV-complete quantum field theories—firms up the connection between neural network expressivity and UV-complete physics.

- **Expressivity**: The paper provides a concrete demonstration of how the inductive bias of the training algorithm (spectral bias toward smooth functions) shapes the effective expressivity of the network. Networks with spectral bias are particularly expressive for smooth functions (including CFT correlators) but may have limited expressivity for rough functions. The smoothness analysis using Sobolev semi-norms and Chebyshev decompositions quantifies this bias and provides a mathematical characterization of the effective hypothesis class. The success of the minimal-data approach suggests that the expressivity of the network, when combined with physical constraints (crossing symmetry), is sufficient to determine UV-complete correlators from very limited information.

## Keywords

- Conformal field theory
- Spectral bias
- Crossing symmetry
- UV-complete correlators
- AdS/CFT correspondence
- Smoothness analysis

## Verbatim Snippet

> We demonstrate that simple feed-forward neural networks can accurately compute correlation functions of conformal field theories on a line. Strikingly, by optimising a NN solely on crossing symmetry and providing only the scaling dimension of the leading non-trivial operator and the correlator's value at a single anchor point, we can reconstruct target physical correlators to within a few percent.
