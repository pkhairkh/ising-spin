---
title: "Self-Organized Criticality in a Network of Interacting Neurons"
authors: "J D Cowan, J Neuman, W van Drongelen"
date: "2012-09-18"
url: "https://arxiv.org/abs/1209.3829"
type: "preprint"
---

## Summary

This paper presents an analysis of a simple neural network that exhibits self-organized criticality (SOC), demonstrating how scale-free dynamics can emerge from the interplay of network architecture and synaptic learning rules. The network combines a basic neural circuit with an excitatory feedback loop that generates bistability, along with an anti-Hebbian synapse in its input pathway. Using methods from statistical field theory, the authors formulate the stochastic dynamics of the network as the action of a path integral, which they then analyze using renormalization group (RG) methods.

The RG analysis reveals that the network exhibits hysteresis as it switches between two stable states, each losing stability at a saddle-node bifurcation. In the neighborhood of these bifurcations, the fluctuations have the signature of directed percolation—a well-known universality class in non-equilibrium statistical mechanics. Thus, the network states undergo the neural analog of a phase transition in the universality class of directed percolation. The network is shown to replicate precisely the behavior of the original sand-pile model of Bak, Tang, and Wiesenfeld, which is the canonical example of self-organized criticality.

This paper is a landmark in the application of RG methods to neural networks, showing that a biologically motivated network with anti-Hebbian plasticity naturally self-organizes to a critical point. The use of statistical field theory and RG to analyze the network provides a rigorous foundation for understanding the emergence of scale-free dynamics in neural systems and connects the study of neural criticality to the broader physics literature on non-equilibrium phase transitions.

## Relevance

- **Hebbian learning**: Anti-Hebbian plasticity is a central component of the model. The anti-Hebbian synapse in the input pathway, combined with the excitatory feedback loop, drives the network to self-organize to criticality. This demonstrates that specific forms of Hebbian (and anti-Hebbian) learning can act as mechanisms for tuning neural circuits to critical operating points.

- **UV-complete physics**: The paper directly applies statistical field theory, path integrals, and renormalization group analysis to a neural network. The identification of directed percolation as the universality class of the network's critical behavior is a major contribution, connecting neural criticality to a well-studied class of non-equilibrium phase transitions in statistical physics.

- **Expressivity**: While the paper does not explicitly address expressivity, the connection between criticality and computational capability is implied. Networks at criticality exhibit maximal dynamical range and sensitivity to inputs, which relates to their representational and computational capacity. The self-organized nature of the criticality suggests that the learning rule itself shapes the computational properties of the network.

## Keywords

- Self-organized criticality
- Renormalization group
- Anti-Hebbian plasticity
- Directed percolation
- Statistical field theory
- Phase transition

## Verbatim Snippet

> Using the methods of statistical field theory, we show how one can formulate the stochastic dynamics of such a network as the action of a path integral, which we then investigate using renormalization group methods. The results indicate that the network exhibits hysteresis in switching back and forward between its two stable state.
