---
title: "Gell-Mann-Low Criticality in Neural Networks"
authors: "Lorenzo Tiberi, Jonas Stapmanns, Tobias Kühn, Thomas Luu, David Dahmen, Moritz Helias"
date: "2021-10-05"
url: "https://arxiv.org/abs/2110.01859"
type: "preprint"
---

## Summary

This paper presents a renormalized theory of a prototypical neural field theory—the stochastic Wilson-Cowan equation—revealing that its critical structure is of the Gell-Mann-Low type, the archetypal form of a renormalizable quantum field theory. The Gell-Mann-Low equation describes how coupling constants flow under renormalization in a theory where interactions are marginally irrelevant: nonlinear couplings vanish under RG flow toward the Gaussian fixed point, but only logarithmically slowly, meaning they remain effective on most physically relevant scales. This structure is fundamentally different from the Wilson-Fisher fixed point that governs most critical phenomena in statistical mechanics, where couplings flow to a non-trivial interacting fixed point.

The authors compute the flow of couplings parameterizing interactions on increasing length scales in the Wilson-Cowan neural field theory. Despite surface similarities with the Kardar-Parisi-Zhang (KPZ) model—a paradigmatic non-equilibrium field theory—the neural field theory exhibits Gell-Mann-Low rather than KPZ renormalization. In KPZ theory, nonlinearities are marginally relevant and grow under RG flow; in the Wilson-Cowan theory, nonlinearities are marginally irrelevant and flow to zero. This distinction has profound consequences for the computational properties of the network: the logarithmic persistence of nonlinear interactions means that the system maintains a balance between linearity (optimal for information storage, as linear networks have well-defined attractors) and nonlinearity (required for computation, as nonlinear transformations enable the network to process and transform information).

The paper argues that this critical structure implements a desirable trade-off between linearity and nonlinearity that is optimal for biological computation. Purely linear networks can store information reliably but cannot perform complex computations; purely nonlinear networks can compute but suffer from chaotic dynamics and unreliable information storage. The Gell-Mann-Low structure provides a principled explanation for why neural systems might operate near criticality: not because of a Wilson-Fisher type phase transition (which would require fine-tuning to a critical point), but because of the robust, logarithmic persistence of marginally irrelevant interactions that naturally maintain the system in a computationally optimal regime.

## Relevance

- **Hebbian learning**: The Wilson-Cowan equation is a classical model of neural population dynamics that naturally incorporates Hebbian-like synaptic coupling. The renormalization of these couplings under RG flow directly informs how Hebbian learning rules affect the network's behavior across scales. The marginally irrelevant flow of the nonlinear coupling means that Hebbian-induced correlations persist logarithmically across scales, providing a mechanism by which local Hebbian plasticity can have long-range effects on network dynamics without requiring fine-tuning.

- **UV-complete physics**: The Gell-Mann-Low structure is one of the foundational concepts of UV-complete quantum field theory. It describes the behavior of marginally irrelevant couplings that flow toward the Gaussian UV fixed point but do so only logarithmically. This is precisely the structure found in quantum electrodynamics (QED), where the electric charge flows to zero at high energies but remains non-negligible at experimentally accessible scales. The paper's identification of this structure in a neural field theory provides a direct bridge between UV-complete physics and neural computation, showing that the same RG structure that governs the high-energy behavior of QED also governs the multi-scale dynamics of cortical circuits.

- **Expressivity**: The Gell-Mann-Low critical structure provides a novel perspective on network expressivity. Expressivity is not maximized at a non-trivial interacting fixed point (as in Wilson-Fisher criticality) but rather emerges from the logarithmic persistence of marginally irrelevant interactions. This means that the effective expressivity of the network is scale-dependent: it is highest at intermediate scales where nonlinear effects are significant, and diminishes at both very small scales (where the theory is approximately Gaussian) and very large scales (where nonlinearities have been renormalized away). This scale-dependent expressivity is a direct consequence of the UV-complete (Gaussian) nature of the fixed point.

## Keywords

- Gell-Mann-Low criticality
- Wilson-Cowan equation
- Renormalization group
- Marginally irrelevant couplings
- Gaussian fixed point
- Neural field theory

## Verbatim Snippet

> Despite similarities with the Kardar-Parisi-Zhang model, the theory is of a Gell-Mann-Low type, the archetypal form of a renormalizable quantum field theory. Here, nonlinear couplings vanish, flowing towards the Gaussian fixed point, but logarithmically slowly, thus remaining effective on most scales.
