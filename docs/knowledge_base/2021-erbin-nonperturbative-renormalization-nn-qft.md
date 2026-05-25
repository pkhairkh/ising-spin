---
title: "Nonperturbative Renormalization for the Neural Network-QFT Correspondence"
authors: "Harold Erbin, Vincent Lahoche, Dine Ousmane Samary"
date: "2021-08-03"
url: "https://arxiv.org/abs/2108.01403"
type: "preprint"
---

## Summary

This paper extends the neural network–quantum field theory (NN-QFT) correspondence established by Halverson, Maiti, and Stoner by moving beyond perturbative analysis and developing a nonperturbative renormalization group framework for neural networks. The original correspondence maps the infinite-width limit of neural networks to a free field theory, with finite-width corrections treated as perturbative interactions. However, perturbation theory is only reliable near the Gaussian (free) fixed point and breaks down for strongly interacting networks far from the infinite-width limit. Erbin, Lahoche, and Samary address this limitation by employing the Wetterich-Morris exact renormalization group equation, which provides a nonperturbative tool for tracking the RG flow across the entire space of couplings, including regimes where perturbation theory fails.

The authors tackle several subtle conceptual issues that arise when applying field-theoretic renormalization to neural networks. First, they discuss the concepts of locality and power-counting in the NN context. In conventional quantum field theory, locality refers to interactions that depend on fields at the same spacetime point, and power-counting determines which operators are relevant, marginal, or irrelevant based on their mass dimension. For neural networks, inputs need not have a permutation symmetry (unlike spacetime coordinates), and the usual notions of locality may not hold. However, the authors argue that the renormalization group provides natural notions of locality and scaling even in this non-standard setting, because the RG organizes operators by their scaling behavior under coarse-graining, regardless of whether they are local in a geometric sense.

A particularly important contribution is the observation that data components may not have permutation symmetry—a common feature in real datasets where different input features have different statistical properties. In this case, the authors argue that random tensor field theories (rather than standard scalar field theories) provide a natural generalization of the NN-QFT correspondence. Random tensor models generalize matrix models to higher-rank tensors and have been developed to study quantum gravity and the melonic large-N limit, providing a natural language for neural networks with non-permutation-symmetric data.

The paper's major practical result is that changing the standard deviation of the neural network weight distribution can be interpreted as a renormalization flow in the space of networks. This provides a concrete, tunable parameter that drives the RG flow, analogous to changing the energy scale in a physical system. The authors focus on translation-invariant kernels and provide preliminary numerical results demonstrating the viability of the nonperturbative approach.

## Relevance

- **Hebbian learning**: While the paper does not directly study Hebbian learning, its nonperturbative RG framework is essential for understanding the UV behavior of networks trained with local learning rules. Hebbian learning operates in the regime of finite-width (interacting) networks, far from the Gaussian fixed point. The nonperturbative RG provides the tools to analyze how Hebbian-induced correlations flow under RG, which is beyond the reach of perturbative approaches.

- **UV-complete physics**: The paper's central contribution is providing a nonperturbative renormalization framework for the NN-QFT correspondence. The Wetterich-Morris equation is a workhorse of nonperturbative RG in high-energy physics, used to study asymptotic safety, UV fixed points, and phase diagrams beyond perturbation theory. By importing this tool into the NN context, the paper opens the door to asking whether neural network field theories can be UV-complete in the nonperturbative sense—whether they possess non-trivial UV fixed points analogous to the asymptotic safety scenario in quantum gravity. The identification of weight standard deviation as an RG flow parameter provides a concrete knob for exploring the UV structure of the theory.

- **Expressivity**: The nonperturbative RG framework allows for a more complete characterization of the hypothesis class of neural networks. Near the Gaussian fixed point, expressivity is limited (the free theory has minimal representational capacity). However, nonperturbative RG can reveal new fixed points and phases that are invisible to perturbation theory, potentially identifying regimes of enhanced expressivity that arise from strong interactions. The connection to random tensor theories also suggests that networks with structured (non-permutation-symmetric) data may have different expressivity properties than those with unstructured data, a distinction that emerges naturally from the field-theoretic framework.

## Keywords

- Nonperturbative renormalization group
- Wetterich-Morris equation
- Neural network-QFT correspondence
- Random tensor field theory
- Power-counting
- UV fixed points

## Verbatim Snippet

> A major result of our analysis is that changing the standard deviation of the neural network weight distribution can be interpreted as a renormalization flow in the space of networks.
