---
title: "Wilsonian Renormalization of Neural Network Gaussian Processes"
authors: "Jessica N. Howard, Ro Jefferson, Anindita Maiti, Zohar Ringel"
date: "2024-05-09"
url: "https://arxiv.org/abs/2405.06008"
type: "preprint"
---

## Summary

This paper develops a practical approach to performing Wilsonian renormalization group (RG) analysis in the context of Gaussian Process (GP) regression, providing a concrete implementation of the RG framework for neural networks that goes beyond structural analogies. The key idea is to systematically integrate out the unlearnable modes of the GP kernel—those modes that carry too little signal relative to noise to be learned from data—thereby obtaining an RG flow in which the data sets the infrared (IR) scale. This is a direct implementation of the Wilsonian philosophy: starting from a UV-complete description (the full GP kernel with all modes), integrate out high-frequency (unlearnable) modes to obtain an effective description at the scale set by the data.

In simple cases, the RG flow results in a universal flow of the ridge (regularization) parameter, which becomes input-dependent in richer scenarios where non-Gaussianities are included. The ridge parameter in GP regression plays the role of a mass term in the field theory, and its flow under RG determines how the effective theory changes as modes are integrated out. The universality of the ridge flow in simple cases is a direct manifestation of the universality that RG analysis predicts: different microscopic theories (different kernel functions) flow to the same effective description at long wavelengths when they share the same relevant operators.

The paper makes a crucial conceptual contribution by establishing a natural connection between RG flow and the distinction between learnable and unlearnable modes. In the Wilsonian picture, the RG flow separates the theory into UV modes (which are integrated out) and IR modes (which are kept). In the learning context, this corresponds to separating the GP into unlearnable modes (which are too noisy to be resolved by the data) and learnable modes (which can be reliably inferred). This correspondence is not merely metaphorical; the mathematical structures are identical. The RG flow provides a principled criterion for determining which features of the data can be learned and which are effectively noise, a question of fundamental importance for understanding the generalization properties of neural networks.

## Relevance

- **Hebbian learning**: The distinction between learnable and unlearnable modes is directly relevant to Hebbian learning. Hebbian rules are local and correlation-based; they can only learn modes that are statistically resolvable from the data. The RG framework provides a formal criterion for which modes fall into this category, offering a principled way to understand the limitations and capabilities of Hebbian learning. The universal flow of the ridge parameter suggests that Hebbian learning, like GP regression, may exhibit universal behavior across different network architectures when viewed through the RG lens.

- **UV-complete physics**: The paper provides a concrete implementation of Wilsonian RG for neural networks, which is the foundational framework for understanding UV completion. In the Wilsonian paradigm, a UV-complete theory is one that can be defined at arbitrarily short distances (high energies); the RG flow from the UV theory to the effective IR description is obtained by integrating out high-energy modes. Here, the full GP kernel (including all modes, learnable and unlearnable) plays the role of the UV-complete theory, and the data sets the IR cutoff below which modes cannot be resolved. The paper demonstrates that this Wilsonian program can be carried out explicitly for neural network GPs, providing a template for asking UV-completeness questions about more general (non-Gaussian) neural network field theories.

- **Expressivity**: The RG flow of the ridge parameter and its input dependence directly characterize the effective expressivity of the model. After integrating out unlearnable modes, the remaining effective theory can only represent functions that are smooth relative to the data resolution. The RG flow thus quantifies how expressivity is reduced as one moves from the UV (all modes) to the IR (learnable modes only). The identification of potential universality classes in the RG flow suggests that the expressivity of wide classes of neural networks may be governed by a small number of RG-fixed-point behaviors, rather than depending on the details of the architecture.

## Keywords

- Wilsonian renormalization group
- Gaussian process regression
- Learnable vs. unlearnable modes
- Ridge parameter flow
- UV-complete theory
- Universality classes

## Verbatim Snippet

> We systematically integrate out the unlearnable modes of the GP kernel, thereby obtaining an RG flow of the GP in which the data sets the IR scale. In addition to being analytically tractable, this approach goes beyond structural analogies between RG and neural networks by providing a natural connection between RG flow and learnable vs. unlearnable modes.
