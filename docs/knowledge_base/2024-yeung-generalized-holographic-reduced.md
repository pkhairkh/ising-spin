---
title: "Generalized Holographic Reduced Representations"
authors: "Calvin Yeung, Zhuowen Zou, Mohsen Imani"
date: "2024-05-15"
url: "https://arxiv.org/abs/2405.09689"
type: "preprint"
---

## Summary

This paper introduces Generalized Holographic Reduced Representations (GHRR), an extension of Fourier Holographic Reduced Representations (FHRR) within the Hyperdimensional Computing (HDC) paradigm. HDC is a brain-inspired computing framework that operates on high-dimensional, pseudo-random vectors and seeks to bridge connectionist and symbolic approaches to AI: it allows explicit specification of representational structure (as in symbolic methods) while retaining the flexibility and robustness of connectionist approaches. The key limitation of existing HDC implementations is that their binding operations are commutative, which prevents the encoding of ordered, compositional structures such as sequences or trees.

GHRR addresses this limitation by introducing a flexible, non-commutative binding operation. This enables the encoding of complex data structures with explicit order information while preserving HDC's desirable properties of robustness and transparency. The authors prove the theoretical properties of GHRR, demonstrate its adherence to fundamental HDC properties, explore its kernel and binding characteristics, and conduct empirical experiments. The results show that GHRR achieves flexible non-commutativity, enhanced decoding accuracy for compositional structures, and improved memorization capacity compared to FHRR.

The paper is positioned at the intersection of representation theory and neural computation. The "holographic" in the name refers to the distributed, hologram-like nature of the representations, where information is spread across all dimensions rather than localized. This holographic encoding is conceptually related to ideas from physics about how information can be distributed in a holographic manner (as in the AdS/CFT correspondence), though the paper works in a discrete, high-dimensional algebraic setting rather than a continuous field-theoretic one.

## Relevance

- **Hebbian learning**: While GHRR does not explicitly use Hebbian learning, HDC as a paradigm has deep historical connections to associative memory and Hebbian-style learning. The binding and bundling operations in HDC are effectively correlation-based operations, and the memorization capacity improvements in GHRR could, in principle, be realized through Hebbian-like plasticity mechanisms in a neural implementation.

- **UV-complete physics**: The "holographic" nature of the representations echoes the holographic principle from high-energy physics (AdS/CFT correspondence), where information about a volume is encoded on its boundary. While the paper operates in a discrete mathematical setting rather than a field-theoretic one, the conceptual parallel is explicit in the naming and motivation.

- **Expressivity**: The paper directly addresses representational capacity through its analysis of memorization capacity and decoding accuracy. The non-commutative binding operation expands the class of compositional structures that can be represented, effectively increasing the expressivity of the HDC framework.

## Keywords

- Holographic reduced representations
- Hyperdimensional computing
- Non-commutative binding
- Compositional structures
- Memorization capacity
- Distributed representations

## Verbatim Snippet

> GHRR introduces a flexible, non-commutative binding operation, enabling improved encoding of complex data structures while preserving HDC's desirable properties of robustness and transparency.
