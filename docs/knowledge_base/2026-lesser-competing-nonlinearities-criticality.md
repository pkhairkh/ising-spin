---
title: "Competing Nonlinearities, Criticality, and Order-to-Chaos Transition in Deep Networks"
authors: "Omri Lesser, Debanjan Chowdhury"
date: "2026-05-06"
url: "https://arxiv.org/abs/2605.05294"
type: "preprint"
---

## Summary

This paper introduces the concept of statistical activation mixtures as a controlled mechanism for navigating the phase diagram of deep neural networks. The key idea is simple but powerful: each neuron independently and randomly draws its activation function from a two-component distribution with mixing fraction p. When applied to a mixture of Tanh and Swish activations, this construction produces a continuous phase transition that is sharp in the depth scaling of the preactivation variance. Below a critical mixing fraction p_c, the network is in a variance-collapsing phase; above p_c, it enters a variance-inflating phase. At exactly p_c, the network acquires statistical scale invariance—depth-independent variance—without sacrificing the smoothness of the activation functions.

This result resolves a longstanding tension in the deep learning theory literature: achieving scale-invariant signal propagation previously required the non-smooth ReLU family of activations, which renders networks ill-suited to curvature-based optimizers, physics-informed architectures, and neural-network quantum states. By showing that smooth activation mixtures can also achieve scale invariance, the paper opens new design possibilities. The transition is corroborated through multiple complementary analyses: variance propagation, parallel and perpendicular susceptibilities, and Lyapunov exponents. Training experiments on real datasets reveal non-monotonic test performance as a function of p, with an optimum near the theoretically predicted critical point, confirming that the initialization-level phase transition has direct consequences for learned representations.

Furthermore, the paper shows that the quenched activation disorder introduced by the statistical mixture acts as a structural regularizer, suppressing memorization of corrupted labels while preserving generalization. This establishes statistical activation mixtures as a principled, physics-inspired tool for controlling the depth scaling, representational properties, and generalization behavior of deep networks.

## Relevance

- **Hebbian learning**: While this paper does not directly study Hebbian plasticity, its framework for analyzing criticality and scale invariance at initialization is directly relevant to understanding the conditions under which local, Hebbian-like learning rules can function effectively. Networks initialized at criticality exhibit signal propagation properties that facilitate gradient flow, which is also essential for Hebbian learning rules that rely on pre- and post-synaptic correlations.

- **UV-complete physics**: This is a central theme. The paper employs effective field theory of signal propagation, identifies universality classes of activations, discovers a continuous phase transition, and characterizes scale invariance and criticality—all concepts imported directly from statistical physics and RG theory. The susceptibilities and Lyapunov exponents used to characterize the transition are standard tools from the physics of disordered systems.

- **Expressivity**: The paper directly addresses how the phase of the network (order vs. chaos, variance-collapsing vs. inflating) affects the depth scaling of representations and thus the expressive power. Networks at the critical point maintain depth-independent signal propagation, which is a necessary condition for deep networks to fully exploit their representational capacity.

## Keywords

- Criticality
- Phase transition
- Scale invariance
- Effective field theory
- Activation mixtures
- Universality classes

## Verbatim Snippet

> At p_c, the network acquires statistical scale invariance, with depth-independent variance, without sacrificing smoothness. This resolves a longstanding tension, where scale-invariant propagation has previously required the non-smooth ReLU family.
