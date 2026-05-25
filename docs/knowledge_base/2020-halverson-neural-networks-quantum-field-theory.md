---
title: "Neural Networks and Quantum Field Theory"
authors: "James Halverson, Anindita Maiti, Keegan Stoner"
date: "2020-08-19"
url: "https://arxiv.org/abs/2008.08601"
type: "preprint"
---

## Summary

This foundational paper establishes a rigorous correspondence between neural networks and Wilsonian effective field theory (EFT), providing the theoretical backbone for understanding neural networks through the lens of quantum field theory. The key insight is that many asymptotic neural networks—those in the infinite-width limit—are drawn from Gaussian processes, which are the direct analog of non-interacting (free) field theories. Moving away from the asymptotic limit by considering finite-width networks yields non-Gaussian processes, corresponding to turning on particle interactions in the field-theoretic picture. This allows for the computation of neural network output correlation functions using Feynman diagrams, a central tool of perturbative quantum field theory.

The correspondence goes deeper than mere analogy. The authors show that minimal non-Gaussian process likelihoods are determined by the most relevant non-Gaussian terms, as classified by the flow in their coefficients induced by the Wilsonian renormalization group. This is directly analogous to how, in particle physics, the RG determines which operators are relevant, marginal, or irrelevant at a given energy scale. The result is a direct connection between overparameterization and simplicity: as the network width grows, the likelihood becomes increasingly Gaussian, meaning that overparameterized networks are governed by the simplest (free) field theory. This provides a field-theoretic explanation for why overparameterized neural networks often generalize well despite having far more parameters than necessary.

The paper further shows that whether the coupling coefficients in the EFT are constants or functions can be understood in terms of Gaussian process limit symmetries, following 't Hooft's principle of technical naturalness. Technical naturalness states that a parameter can be small without fine-tuning if setting it to zero enhances the symmetry of the theory. In the neural network context, this means that certain structural features of the network are protected by symmetries of the GP limit, ensuring their stability under RG flow. The formalism is valid for any architecture that becomes a GP in an asymptotic limit—a property preserved under certain types of training—which gives it broad applicability across the deep learning landscape.

## Relevance

- **Hebbian learning**: While the paper does not directly address Hebbian plasticity, its Wilsonian EFT framework provides the theoretical infrastructure for understanding how local learning rules (such as Hebbian updates) interact with the field-theoretic structure of the network. The RG classification of operators into relevant and irrelevant directions in coupling space directly informs which features of a Hebbian learning rule would survive under coarse-graining and which would be washed out. This is essential for understanding whether Hebbian learning can serve as a UV-complete training mechanism.

- **UV-complete physics**: This is the paper's central contribution. By mapping neural networks to Wilsonian EFTs, the authors establish that the infinite-width GP limit is the free (Gaussian) UV fixed point of the theory, and finite-width corrections correspond to turning on interactions. The RG flow from the UV (infinite-width, free theory) to the IR (finite-width, interacting theory) is the Wilsonian paradigm for understanding how a UV-complete theory (the free GP limit) generates effective descriptions at lower scales. The concept of technical naturalness from high-energy physics is directly imported, providing a criterion for which deformations of the GP limit are UV-natural.

- **Expressivity**: The paper provides a field-theoretic account of expressivity through its analysis of the RG flow of coupling coefficients. The most relevant non-Gaussian terms in the EFT expansion dominate the likelihood, effectively constraining the hypothesis class. Overparameterization, in this view, does not increase expressivity without bound; instead, it drives the network toward the UV fixed point (free theory), which has minimal expressivity but maximal simplicity. Expressivity emerges from the relevant deformations away from this fixed point, providing a RG-organized understanding of how width and depth trade off in determining representational capacity.

## Keywords

- Wilsonian effective field theory
- Neural network field theory
- Gaussian process limit
- Renormalization group
- Technical naturalness
- Feynman diagrams

## Verbatim Snippet

> Moving away from the asymptotic limit yields a non-Gaussian process and corresponds to turning on particle interactions, allowing for the computation of correlation functions of neural network outputs with Feynman diagrams. Minimal non-Gaussian process likelihoods are determined by the most relevant non-Gaussian terms, according to the flow in their coefficients induced by the Wilsonian renormalization group.
