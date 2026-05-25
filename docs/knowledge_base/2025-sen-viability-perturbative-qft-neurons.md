---
title: "Viability of Perturbative Expansion for Quantum Field Theories on Neurons"
authors: "Srimoyee Sen, Varun Vaidya"
date: "2025-08-05"
url: "https://arxiv.org/abs/2508.03810"
type: "preprint"
---

## Summary

This paper examines a critical question at the intersection of quantum field theory and neural networks: whether neural network architectures that break statistical independence of parameters can serve as viable tools for simulating local quantum field theories at finite neuron number. While the infinite neuron number limit allows single-layer neural networks to exactly reproduce quantum field theory results, the practical utility of these architectures depends on their performance at finite N, where the 1/N expansion provides systematic corrections to the infinite-N results. The authors investigate this question using scalar phi-fourth theory in d Euclidean dimensions as a concrete test case.

The central finding is that the renormalized O(1/N) corrections to two- and four-point correlators yield perturbative series that are sensitive to the ultraviolet cutoff. This UV sensitivity means that the corrections depend on the highest frequency modes included in the neural network's representation, and that the convergence of the perturbative series is weak—higher-order corrections do not systematically decrease in magnitude. This is a fundamental problem because it means that the neural network's ability to approximate the QFT depends critically on how the UV cutoff is handled, and that naive implementations may require impractically large neuron numbers to achieve accurate results.

The authors propose a modification to the neural network architecture to improve convergence and discuss constraints on the parameters of the theory and the scaling of N (the number of neurons) that allow extraction of accurate field theory results. The modification involves introducing a UV regulator that suppresses the contribution of high-frequency modes in a controlled manner, analogous to the regularization schemes used in continuum quantum field theory (such as dimensional regularization or Pauli-Villars regularization). The scaling analysis reveals that N must scale with the UV cutoff in a specific way to maintain accuracy, providing concrete guidance for the design of neural network architectures for QFT simulation.

## Relevance

- **Hebbian learning**: While the paper does not directly address Hebbian learning, its analysis of UV sensitivity in neural network field theories has important implications for local learning rules. Hebbian learning operates by modifying synaptic weights based on local correlations, and these correlations depend on the UV (high-frequency) content of the network's activity. If the UV content is poorly controlled—as this paper shows can happen in finite-N neural networks—then Hebbian learning may be unstable or unreliable. The proposed architectural modifications that improve UV behavior could therefore be important for ensuring that Hebbian learning operates in a well-defined regime.

- **UV-complete physics**: This paper is directly about UV behavior in neural network field theories. The UV sensitivity of the 1/N corrections is precisely the kind of problem that the concept of UV completion addresses: a theory is UV-complete if it is well-defined at arbitrarily high energies (or, equivalently, arbitrarily short distances) without requiring the introduction of a cutoff. The paper shows that neural network QFTs are not automatically UV-complete; the UV sensitivity of the perturbative expansion means that the theory depends on the cutoff in an uncontrolled way. The proposed architectural modification represents a step toward UV completion by introducing a controlled UV regulator, but the question of whether a fully UV-complete neural network QFT exists remains open.

- **Expressivity**: The UV sensitivity of the 1/N corrections constrains the expressivity of neural network QFTs. If the network's ability to represent field-theoretic observables depends on the UV cutoff in an uncontrolled way, then the effective hypothesis class is not well-defined. The proposed modification that improves convergence effectively restricts the hypothesis class to functions that are well-behaved in the UV, which is a necessary condition for the network to faithfully represent the target QFT. The scaling constraints on N provide a quantitative relationship between the network size and the expressivity required to achieve a given level of accuracy.

## Keywords

- Quantum field theory on neurons
- UV cutoff sensitivity
- 1/N expansion
- Perturbative corrections
- UV completion
- Scalar field theory

## Verbatim Snippet

> We find that the renormalized O(1/N) corrections to two- and four-point correlators yield perturbative series which are sensitive to the ultraviolet cut-off and therefore have a weak convergence.
