---
title: "Bayesian RG Flow in Neural Network Field Theories"
authors: "Jessica N. Howard, Marc S. Klinger, Anindita Maiti, Alexander G. Stapleton"
date: "2024-05-27"
url: "https://arxiv.org/abs/2405.17538"
type: "preprint"
---

## Summary

This paper unifies two powerful frameworks—the Neural Network Field Theory (NNFT) correspondence and the Bayesian Renormalization Group (BRG)—to create a new framework called BRG-NNFT that enables systematic exploration of the space of neural networks and statistical field theories. The NNFT correspondence maps neural network architectures into the space of statistical field theories, while the BRG is an information-theoretic coarse-graining scheme that generalizes the principles of the exact renormalization group (ERG) to arbitrarily parameterized probability distributions, including those of neural networks.

The central insight of the paper is the dual interpretation of flows in the space of field theories. Neural network training dynamics induce a flow in the space of SFTs from the information-theoretic IR toward the UV: as training progresses, the network learns to capture finer-grained (more UV-like) features of the data distribution. Conversely, applying an information-shell coarse-graining to the trained network's parameters induces a flow from the information-theoretic UV toward the IR: as coarse-graining proceeds, fine-grained parameter information is integrated out, yielding a simpler effective description. This dual flow structure is the field-theoretic realization of the Wilsonian paradigm, where the UV theory (microscopic description) and the IR theory (macroscopic description) are related by RG flow.

A critical result is that when the information-theoretic cutoff scale coincides with a standard momentum scale, the BRG is equivalent to the ERG. This equivalence bridges the information-theoretic and physics perspectives on coarse-graining, showing that the same mathematical structure underlies both the statistical description of neural network parameters and the physical description of field theories. The authors demonstrate the BRG-NNFT correspondence on two analytically tractable examples: first, they construct BRG flows for trained infinite-width neural networks of arbitrary depth with generic activation functions; second, they show that for architectures with a single infinitely-wide layer, scalar outputs, and generalized cos-net activations, BRG coarse-graining corresponds exactly to the momentum-shell ERG flow of a free scalar SFT.

## Relevance

- **Hebbian learning**: The BRG-NNFT framework provides a natural language for understanding Hebbian learning through the lens of UV-complete field theory. Hebbian learning is a local, correlation-based update rule that progressively refines the network's parameters based on activity correlations. In the BRG-NNFT picture, this corresponds to an IR-to-UV flow: the network starts with a coarse (IR) description and progressively resolves finer (UV) features through local parameter updates. The information-theoretic nature of BRG is particularly apt, as Hebbian learning can be viewed as an information-theoretic process that captures statistical dependencies between pre- and post-synaptic activity.

- **UV-complete physics**: This paper is centrally about UV completion. The IR-to-UV flow induced by training is the neural network analog of reconstructing a UV-complete theory from its effective IR description—a fundamental problem in high-energy physics. The equivalence of BRG and ERG when the information-theoretic and momentum scales coincide provides a rigorous connection between the neural network RG flow and the standard Wilsonian RG flow of quantum field theories. This means that questions about UV completion in QFT can be translated into questions about training dynamics in neural networks, and vice versa. The identification of the free scalar SFT as the fixed point of the BRG flow for cos-net architectures demonstrates that these networks are UV-complete in the sense that their training dynamics flow toward a well-defined UV theory.

- **Expressivity**: The BRG-NNFT framework provides a field-theoretic characterization of expressivity through the structure of the RG flow. The number and nature of fixed points in the BRG flow determine the space of accessible effective theories, which in turn determines the representational capacity of the network. Networks whose BRG flow has multiple fixed points can access a richer space of effective theories than those with a single fixed point, implying greater expressivity. The analytic tractability of the BRG-NNFT correspondence for infinite-width networks provides explicit formulas for the flow of coupling constants, enabling quantitative predictions about how expressivity depends on architecture and training dynamics.

## Keywords

- Bayesian renormalization group
- Neural network field theory
- IR-to-UV flow
- Information-theoretic coarse-graining
- Wilsonian RG
- UV completion

## Verbatim Snippet

> NN training dynamics can be interpreted as inducing a flow in the space of SFTs from the information-theoretic IR toward the UV. Conversely, applying an information-shell coarse graining to the trained network's parameters induces a flow in the space of SFTs from the information-theoretic UV toward the IR. When the information-theoretic cutoff scale coincides with a standard momentum scale, BRG is equivalent to ERG.
