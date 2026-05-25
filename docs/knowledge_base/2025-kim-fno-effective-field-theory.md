---
title: "Analysis of Fourier Neural Operators via Effective Field Theory"
authors: "Taeyoung Kim"
date: "2025-07-29"
url: "https://arxiv.org/abs/2507.21833"
type: "preprint"
---

## Summary

This paper applies the framework of effective field theory (EFT) to analyze Fourier Neural Operators (FNOs), a class of neural architectures that have become leading surrogates for solving PDEs. The author develops a systematic EFT analysis in infinite-dimensional function space, deriving closed recursion relations for the layer kernel and four-point vertex. These recursions are then examined in three practical settings: analytic activations, scale-invariant cases, and architectures with residual connections. A key finding is that nonlinear activations inevitably couple low-frequency inputs to high-frequency modes that would otherwise be discarded by spectral truncation, a phenomenon confirmed experimentally. This frequency transfer mechanism explains how FNOs can capture nontrivial features despite their spectral compression.

For wide networks, the paper derives explicit criticality conditions on the weight initialization ensemble that ensure input perturbations maintain a uniform scale across depth. At criticality, small perturbations neither vanish nor explode as they propagate through layers, which is directly analogous to the critical point in a statistical-physics model where correlation lengths diverge. The theoretically predicted ratio of kernel perturbations matches experimental measurements, validating the EFT framework. The paper further translates criticality theory into a practical matched-initialization (calibration) procedure, demonstrating on the PDEBench Burgers benchmark that the calibrated FNO achieves more stable optimization, faster convergence, and lower test error than a vanilla FNO.

The results collectively quantify how nonlinearity endows neural operators with the ability to capture features beyond the truncated spectral basis, provide concrete criteria for hyperparameter selection through criticality analysis, and explain why scale-invariant activations and residual connections enhance feature learning. The paper represents a compelling instance where the language and tools of high-energy physics—effective field theory, criticality, and scale invariance—directly inform the design and understanding of neural architectures.

## Relevance

- **Hebbian learning**: While this paper does not explicitly address Hebbian plasticity, its criticality analysis of weight initialization has implications for understanding what kinds of local learning dynamics are stable. The framework could, in principle, be extended to study how Hebbian-like weight updates interact with critical initialization conditions, making it a relevant background reference for the intersection.

- **UV-complete physics**: This is a central theme of the paper. The entire analytical framework is built on effective field theory, criticality conditions, and scale invariance—concepts directly imported from high-energy and statistical physics. The derivation of recursion relations for kernels and vertices mirrors the renormalization group approach, and the criticality conditions are analogous to tuning a system to a second-order phase transition.

- **Expressivity**: The paper directly addresses the representational capacity of FNOs. It shows that nonlinearity enables the network to represent functions beyond the truncated spectral basis, and that the expressivity is maximized near the critical initialization point. This provides a physics-inspired account of depth-dependent expressivity in neural operators.

## Keywords

- Effective field theory
- Fourier Neural Operators
- Criticality
- Scale invariance
- Spectral methods
- Neural operator theory

## Verbatim Snippet

> For wide networks, we derive explicit criticality conditions on the weight initialization ensemble that ensure small input perturbations maintain a uniform scale across depth, and we confirm experimentally that the theoretically predicted ratio of kernel perturbations matches the measurements.
