---
title: "Anomalies in Neural Network Field Theory"
authors: "Christian Ferko, Samuel Frank, James Halverson, Vishnu Jejjala"
date: "2026-05-12"
url: "https://arxiv.org/abs/2605.12488"
type: "preprint"
---

## Summary

This paper derives Schwinger-Dyson equations and Ward identities in neural network field theory (NN-FT) and uses them to study anomalies—quantum mechanical violations of classical symmetries that are among the deepest and most subtle phenomena in theoretical physics. NN-FT formulates field theory in terms of a network architecture and a probability density on its parameters, and the authors show that the standard tools of quantum field theory—Schwinger-Dyson equations (which encode the constraints of the path integral) and Ward identities (which express the consequences of symmetries)—have natural analogs in this setting.

A key technical innovation is the identification of a conserved parameter space current that characterizes symmetries and how they break. This current is relevant even in non-local NN-FTs, where the standard notion of a local symmetry current does not apply because the network's parameter space does not have the structure of physical spacetime. However, the parameter space current can recover local currents in the case of a local Lagrangian by an appropriate fiber-wise average, demonstrating that the formalism reduces to the standard QFT results in the appropriate limit.

The paper applies this machinery to a remarkable range of problems spanning both machine learning and physics. In machine learning, the formalism is applied to feedforward networks and the attention mechanism of transformers, revealing symmetry structures in these architectures that were not previously appreciated. In physics, the authors use the NN-FT framework to study the U(1) symmetry for a complex scalar, the scale anomaly in four-dimensional massless phi-fourth theory, the Weyl anomaly for the bosonic string (including a new computation of the critical dimension), and examples involving discrete topological data such as winding numbers and T-duality. The scale anomaly—the quantum mechanical violation of scale invariance—is directly related to the question of UV completeness, as it determines whether a classically scale-invariant theory remains well-defined at all scales or requires UV completion.

## Relevance

- **Hebbian learning**: The anomaly analysis in NN-FT has implications for understanding the symmetry structure of networks trained with Hebbian learning rules. Hebbian learning is a correlation-based rule that respects certain symmetries (such as permutation symmetry among neurons) and breaks others (such as detailed balance). The Ward identity formalism developed in this paper provides a systematic way to identify which symmetries of the network are preserved by Hebbian learning and which are anomalously broken. The application to feedforward networks and attention mechanisms suggests that the symmetry-breaking patterns induced by different learning rules may be characterizable in terms of anomalies, providing a field-theoretic classification of learning rules.

- **UV-complete physics**: The study of anomalies is directly relevant to UV completion. In quantum field theory, anomalies provide crucial constraints on which theories can be UV-complete: a theory with a gauge anomaly (an anomalous breaking of a gauge symmetry) cannot be consistently quantized and is therefore not UV-complete. The scale anomaly studied in this paper determines whether a classically scale-invariant theory remains conformally invariant at the quantum level, which is a necessary condition for the theory to be UV-complete as a conformal field theory. The Weyl anomaly computation for the bosonic string—which yields the critical dimension—is one of the most celebrated results in string theory and is directly connected to the UV consistency of the theory. By reproducing and extending these results in the NN-FT framework, the paper demonstrates that neural network field theories can capture the same UV-completeness constraints as conventional QFTs.

- **Expressivity**: Anomalies constrain the hypothesis class of the neural network by imposing symmetry conditions that the learned function must satisfy. If a symmetry of the network architecture is anomalously broken, then the effective hypothesis class is larger than the classically symmetric one, because the network can represent functions that violate the classical symmetry. Conversely, if a symmetry is preserved (non-anomalous), then the hypothesis class is constrained to symmetric functions. The identification of anomaly structure in feedforward networks and attention mechanisms provides a field-theoretic characterization of the expressivity of these architectures, showing how representational capacity is determined by the interplay of classical symmetries and their quantum (finite-width) corrections.

## Keywords

- Anomalies
- Schwinger-Dyson equations
- Ward identities
- Scale anomaly
- Weyl anomaly
- Neural network field theory

## Verbatim Snippet

> We derive Schwinger-Dyson equations and Ward identities in NN-FT and utilize them to study anomalies. The equations depend on a conserved parameter space current that characterizes symmetries and how they break.
