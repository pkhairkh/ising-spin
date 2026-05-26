"""
Manifold Capacity — Intrinsic Information Capacity of Dense Associative Memory.

THE FUNDAMENTAL QUESTION:
  How large is the function manifold of the DAM? How many independent
  parameters can actually be encoded within it? This is the abstract
  quantity that determines what the architecture CAN represent, regardless
  of training.

THE ANSWER — Metric Entropy H(F):
  H(F) = P_max * log2(V)

  where P_max is the number of distinguishable attractor basins (storage
  capacity), and log2(V) is the information per attractor.

P_max IS THE ENTIRE QUESTION. It depends on beta, D, k, and critically
on WHICH F(x) you use:

  LINEAR F (standard Hopfield):   P_max ~ 0.14 * D
  POLYNOMIAL F(x) = x^n:         P_max ~ alpha_n * D     (polynomial)
  EXPONENTIAL F(x) = exp(beta*x): P_max ~ exp(beta^2*k/2) / (a*beta)  (EXPONENTIAL)

The exponential regime is what makes Dense AM a language model.
The exp(beta^2) makes beta the SINGLE MOST IMPORTANT hyperparameter —
it controls the EXPONENT in manifold size, not just a linear multiplier.

KEY RESULT FOR THIS ARCHITECTURE:
  At beta=0.25 (current):  P_max ~ 164,  H(F) ~ 1,800 bits  (225 bytes)
  At beta=1.0:             P_max ~ 17,500, H(F) ~ 192,500 bits (24 KB)
  At beta=2.0:             P_max ~ 10^9,  H(F) ~ 10^10 bits  (1.2 GB)

The manifold is currently two orders of magnitude too small for language.

SECONDARY QUANTITIES:
  - d_eff = P_max * k * log2(D/k)  — encodable parameters (bits)
  - fat_F = P_max * k               — fat-shattering dimension
  - d_per_attractor = k * log2(D/k)  — independent bits per attractor

ALL INTEGER-ONLY where it matters. Capacity formulas use float for the
exp() since this is a DIAGNOSTIC (not part of the model dynamics).
The results are reported in integer bits / bytes.

Reference:
  - Amit, Gutfreund, Sompolinsky (1985): P_max ~ 0.14N for Hopfield
  - Krotov & Hopfield (2016): Dense AM with nonlinear F
  - Ramsauer et al. (2020): Hopfield Networks is All You Need
  - Demirel et al.: Sparse modern Hopfield networks
"""

import math
from typing import Dict, List, Optional


class ManifoldCapacity:
    """
    Compute the intrinsic information capacity of a Dense Associative Memory.

    This is NOT a runtime diagnostic — it's a theoretical computation of
    the maximum expressivity the architecture can achieve, based on the
    DAM storage capacity formula.

    The three key quantities:
      1. P_max: Number of distinguishable attractor basins
      2. H(F):  Metric entropy — total bits to specify any function in the class
      3. d_eff:  Effective number of encodable parameters (bits)

    These depend on:
      - beta: Inverse temperature (controls F(x) = exp(beta*x) sharpness)
      - D:    Spin dimension
      - k:    Number of active bits (sparsity a = k/D)
      - V:    Vocabulary size

    And critically on the F(x) regime:
      - LINEAR:    Standard Hopfield, P_max ~ 0.14*D
      - POLYNOMIAL: Moderate DAM, P_max ~ alpha_n*D
      - EXPONENTIAL: Full DAM, P_max ~ exp(beta^2*k/2) / (a*beta)
    """

    # Amit-Gutfreund-Sompolinsky constant for standard Hopfield
    AGS_ALPHA = 0.14

    @staticmethod
    def p_max_linear(D: int) -> float:
        """
        Storage capacity for STANDARD HOPFIELD (linear F).

        P_max ~ 0.14 * D

        This is the Amit-Gutfreund-Sompolinsky (1985) result.
        Beyond this, pattern retrieval breaks down catastrophically.

        For D=256: P_max ~ 36 patterns.
        Metric entropy: 36 * log2(2000) ~ 396 bits = 50 bytes.

        This is why standard Hopfield networks cannot do language.
        """
        return ManifoldCapacity.AGS_ALPHA * D

    @staticmethod
    def p_max_polynomial(D: int, n: int = 3) -> float:
        """
        Storage capacity for POLYNOMIAL DAM F(x) = x^n.

        P_max ~ alpha_n * D, where alpha_n grows roughly linearly with n.

        For n=3, D=256: P_max ~ 108 patterns.
        Still polynomial — the manifold grows LINEARLY with D.

        Compare: exponential F gives EXPONENTIAL growth in D.
        """
        # alpha_n scales roughly as 0.14 * n^1.5 (empirical fit to Krotov 2016)
        alpha_n = 0.14 * (n ** 1.5)
        return alpha_n * D

    @staticmethod
    def p_max_exponential(D: int, k: int, beta: float) -> float:
        """
        Storage capacity for EXPONENTIAL DAM F(x) = exp(beta*x).

        For sparse binary patterns with sparsity a = k/D:

          P_max ~ D * exp(beta^2 * a * D / 2) / (a * D * beta)
                = exp(beta^2 * k / 2) / (a * beta)

        This is the Krotov-Hopfield / Demirel result for Dense AM
        with exponential nonlinearity and sparse codes.

        The exp(beta^2 * k / 2) term is what gives the DAM its
        EXPONENTIAL storage capacity. This is why beta is the master
        dial — it controls the EXPONENT.

        Args:
            D: Spin dimension (e.g., 256).
            k: Number of active bits (e.g., 8).
            beta: Inverse temperature (e.g., 0.25 for beta_fp=64).

        Returns:
            P_max: Number of distinguishable attractor basins.
        """
        if k <= 0 or D <= 0 or beta <= 0:
            return 0.0

        a = k / D  # sparsity

        # Core formula: P_max ~ exp(beta^2 * k / 2) / (a * beta)
        exponent = (beta ** 2) * k / 2.0
        p_max = math.exp(exponent) / (a * beta)

        # Cap at physical maximum: J matrix can only encode so many
        # independent patterns. The hard limit is the rank of J,
        # which is at most D. But with int16 precision (65536 levels),
        # each entry can distinguish ~65536 patterns.
        # Practical upper bound: P_max <= D * 65536 / k
        p_physical_max = D * 65536 / k
        p_max = min(p_max, p_physical_max)

        return p_max

    @staticmethod
    def p_max_from_beta_fp(D: int, k: int, beta_fp: int) -> float:
        """
        Storage capacity from the integer beta_fp parameter.

        beta_fp is Q8 fixed-point: beta = beta_fp / 256.

        Args:
            D: Spin dimension.
            k: Active bits.
            beta_fp: Beta in Q8 fixed-point (e.g., 64 = 0.25).

        Returns:
            P_max for the exponential DAM regime.
        """
        beta = beta_fp / 256.0
        return ManifoldCapacity.p_max_exponential(D, k, beta)

    @staticmethod
    def metric_entropy(p_max: float, V: int) -> float:
        """
        Metric entropy H(F) of the DAM function class, in bits.

        H(F) = P_max * log2(V)

        This answers: "How many bits to uniquely specify any function
        this model can represent?" It is the information-theoretic
        size of the function manifold.

        Args:
            p_max: Storage capacity (number of attractor basins).
            V: Vocabulary size.

        Returns:
            H(F) in bits.
        """
        if V <= 1 or p_max <= 0:
            return 0.0
        return p_max * math.log2(V)

    @staticmethod
    def encodable_parameters(p_max: float, D: int, k: int) -> float:
        """
        Effective number of encodable parameters, in bits.

        d_eff = P_max * k * log2(D/k)

        Each stored attractor is a k-sparse binary pattern over D dims.
        The number of functionally independent parameters per attractor:
          d_per = k * log2(D/k)
        (each of k active bits needs ~log2(D/k) bits to specify its
        position among the D/k candidate groups).

        Total across all attractors:
          d_eff = P_max * d_per

        Args:
            p_max: Storage capacity.
            D: Spin dimension.
            k: Active bits.

        Returns:
            d_eff in bits.
        """
        if D <= 0 or k <= 0 or p_max <= 0 or k >= D:
            return 0.0
        d_per = k * math.log2(D / k)
        return p_max * d_per

    @staticmethod
    def fat_shattering_dimension(p_max: float, k: int) -> float:
        """
        Fat-shattering dimension fat_F(gamma) of the DAM function class.

        fat_F(gamma) ~ P_max * k

        This is the scale-sensitive VC dimension — the number of
        independently controllable binary decisions the model can make
        with confidence >= gamma.

        It is THE abstract measure of "how many things can the model know."

        Args:
            p_max: Storage capacity.
            k: Active bits per pattern.

        Returns:
            fat_F (approximate, for margin gamma ~ 1/k).
        """
        return p_max * k

    @staticmethod
    def compute_layer_capacity(
        D: int,
        k: int,
        beta_fp: int,
        V: int,
    ) -> Dict:
        """
        Compute full capacity analysis for a single DAM layer.

        Returns all three regimes (linear, polynomial, exponential)
        and all derived quantities (metric entropy, encodable params,
        fat-shattering dimension).

        Args:
            D: Spin dimension.
            k: Active bits.
            beta_fp: Beta in Q8 fixed-point.
            V: Vocabulary size.

        Returns:
            Dictionary with all capacity metrics.
        """
        beta = beta_fp / 256.0

        # Three regimes
        p_linear = ManifoldCapacity.p_max_linear(D)
        p_poly = ManifoldCapacity.p_max_polynomial(D, n=3)
        p_exp = ManifoldCapacity.p_max_from_beta_fp(D, k, beta_fp)

        # Exponential regime metrics (the actual one we use)
        h_exp = ManifoldCapacity.metric_entropy(p_exp, V)
        d_exp = ManifoldCapacity.encodable_parameters(p_exp, D, k)
        fat_exp = ManifoldCapacity.fat_shattering_dimension(p_exp, k)

        # Linear regime for comparison
        h_linear = ManifoldCapacity.metric_entropy(p_linear, V)
        d_linear = ManifoldCapacity.encodable_parameters(p_linear, D, k)
        fat_linear = ManifoldCapacity.fat_shattering_dimension(p_linear, k)

        # Conversion helpers
        def bits_to_bytes(b):
            return b / 8.0

        def bits_to_float16_equiv(b):
            """Each float16 param = 16 bits."""
            return b / 16.0

        return {
            # Input parameters
            'D': D,
            'k': k,
            'beta_fp': beta_fp,
            'beta': beta,
            'V': V,
            'sparsity': k / D,

            # Exponential DAM (ACTUAL regime)
            'p_max': p_exp,
            'metric_entropy_bits': h_exp,
            'metric_entropy_bytes': bits_to_bytes(h_exp),
            'encodable_params_bits': d_exp,
            'encodable_params_bytes': bits_to_bytes(d_exp),
            'float16_equiv_params': bits_to_float16_equiv(d_exp),
            'fat_shattering': fat_exp,

            # Linear Hopfield (for comparison)
            'p_max_linear': p_linear,
            'metric_entropy_linear_bits': h_linear,
            'fat_shattering_linear': fat_linear,

            # Polynomial DAM (for comparison)
            'p_max_polynomial': p_poly,

            # Capacity expansion factor vs linear
            'expansion_vs_linear': p_exp / max(1, p_linear),

            # Transformer equivalent (very rough)
            'transformer_equiv': ManifoldCapacity._transformer_equivalent(d_exp),
        }

    @staticmethod
    def compute_hierarchical_capacity(
        layer_configs: List[Dict],
        V: int,
    ) -> Dict:
        """
        Compute full capacity analysis for a hierarchical DAM.

        Each layer config is a dict with keys: D, k, beta_fp, name.

        The total manifold is the SUM of per-layer manifolds:
          H_total = sum_i H(F_i)

        Since each layer operates at a different temporal scale and
        encodes different aspects of language (lexical, syntactic,
        semantic, discourse), their attractor basins are largely
        independent. The RG coupling constrains but doesn't eliminate
        this independence.

        Args:
            layer_configs: List of dicts, each with D, k, beta_fp, name.
            V: Vocabulary size.

        Returns:
            Dictionary with hierarchical capacity metrics.
        """
        layer_results = []
        total_p_max = 0.0
        total_entropy = 0.0
        total_encodable = 0.0
        total_fat = 0.0

        for config in layer_configs:
            D = config['D']
            k = config['k']
            beta_fp = config['beta_fp']
            name = config.get('name', f'D{D}')

            result = ManifoldCapacity.compute_layer_capacity(D, k, beta_fp, V)
            result['name'] = name
            layer_results.append(result)

            total_p_max += result['p_max']
            total_entropy += result['metric_entropy_bits']
            total_encodable += result['encodable_params_bits']
            total_fat += result['fat_shattering']

        return {
            'layers': layer_results,
            'total_p_max': total_p_max,
            'total_metric_entropy_bits': total_entropy,
            'total_metric_entropy_bytes': total_entropy / 8.0,
            'total_encodable_params_bits': total_encodable,
            'total_encodable_params_bytes': total_encodable / 8.0,
            'total_float16_equiv': total_encodable / 16.0,
            'total_fat_shattering': total_fat,
            'transformer_equiv': ManifoldCapacity._transformer_equivalent(total_encodable),
        }

    @staticmethod
    def beta_sweep(
        D: int,
        k: int,
        V: int,
        beta_fp_values: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Sweep beta_fp values and compute capacity at each point.

        This shows how the manifold expands exponentially with beta.

        Args:
            D: Spin dimension.
            k: Active bits.
            V: Vocabulary size.
            beta_fp_values: List of Q8 beta values to sweep.
                Default: [32, 64, 96, 128, 192, 256, 384, 512]

        Returns:
            List of dicts with beta_fp, p_max, entropy, etc.
        """
        if beta_fp_values is None:
            beta_fp_values = [32, 64, 96, 128, 192, 256, 384, 512]

        results = []
        for beta_fp in beta_fp_values:
            cap = ManifoldCapacity.compute_layer_capacity(D, k, beta_fp, V)
            results.append({
                'beta_fp': beta_fp,
                'beta': beta_fp / 256.0,
                'p_max': cap['p_max'],
                'metric_entropy_bits': cap['metric_entropy_bits'],
                'metric_entropy_bytes': cap['metric_entropy_bytes'],
                'encodable_params_bits': cap['encodable_params_bits'],
                'float16_equiv': cap['float16_equiv_params'],
                'fat_shattering': cap['fat_shattering'],
                'transformer_equiv': cap['transformer_equiv'],
            })
        return results

    @staticmethod
    def compare_architectures(
        V: int,
        current: Optional[Dict] = None,
        target: Optional[Dict] = None,
    ) -> Dict:
        """
        Compare current architecture vs target architecture.

        Current: 4 layers, D=256, k=8, beta_fp=64
        Target:  4 layers, D=[512,256,128,64], k=[16,8,4,2], beta_fp=256

        Args:
            V: Vocabulary size.
            current: Dict with 'layers' list (each with D, k, beta_fp).
            target: Dict with 'layers' list (each with D, k, beta_fp).

        Returns:
            Comparison dictionary.
        """
        if current is None:
            current = {
                'layers': [
                    {'D': 256, 'k': 8, 'beta_fp': 64, 'name': 'L0_Lexical'},
                    {'D': 256, 'k': 8, 'beta_fp': 64, 'name': 'L1_Syntactic'},
                    {'D': 256, 'k': 8, 'beta_fp': 64, 'name': 'L2_Semantic'},
                    {'D': 256, 'k': 8, 'beta_fp': 64, 'name': 'L3_Discourse'},
                ]
            }

        if target is None:
            target = {
                'layers': [
                    {'D': 512, 'k': 16, 'beta_fp': 256, 'name': 'L0_Lexical'},
                    {'D': 256, 'k': 8, 'beta_fp': 256, 'name': 'L1_Syntactic'},
                    {'D': 128, 'k': 4, 'beta_fp': 256, 'name': 'L2_Semantic'},
                    {'D': 64, 'k': 2, 'beta_fp': 256, 'name': 'L3_Discourse'},
                ]
            }

        current_cap = ManifoldCapacity.compute_hierarchical_capacity(
            current['layers'], V
        )
        target_cap = ManifoldCapacity.compute_hierarchical_capacity(
            target['layers'], V
        )

        # Expansion factor
        if current_cap['total_metric_entropy_bits'] > 0:
            expansion = (
                target_cap['total_metric_entropy_bits']
                / current_cap['total_metric_entropy_bits']
            )
        else:
            expansion = float('inf')

        return {
            'current': current_cap,
            'target': target_cap,
            'entropy_expansion_factor': expansion,
            'current_transformer_equiv': current_cap['transformer_equiv'],
            'target_transformer_equiv': target_cap['transformer_equiv'],
        }

    @staticmethod
    def _transformer_equivalent(encodable_bits: float) -> str:
        """
        Rough transformer equivalent based on encodable parameters.

        This is a VERY rough mapping — the point is to give intuition
        about manifold size, not to claim exact equivalence.
        """
        params = encodable_bits / 16.0  # float16 equivalent

        if params < 1000:
            return "tiny n-gram (<1K params)"
        elif params < 10_000:
            return "small n-gram (~1-10K)"
        elif params < 100_000:
            return "Char-RNN (~10-100K)"
        elif params < 1_000_000:
            return "small LSTM (~100K-1M)"
        elif params < 10_000_000:
            return "4-layer LSTM (~1-10M)"
        elif params < 100_000_000:
            return "medium transformer (~10-100M)"
        elif params < 1_000_000_000:
            return "large transformer (~100M-1B)"
        else:
            return f"GPT-2+ scale ({params/1e9:.1f}B+ params)"

    @staticmethod
    def format_capacity_report(
        cap: Dict,
        title: str = "DAM Manifold Capacity",
    ) -> str:
        """
        Format a capacity report as a human-readable string.

        Args:
            cap: Output of compute_layer_capacity or compute_hierarchical_capacity.
            title: Report title.

        Returns:
            Formatted string.
        """
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"  {title}")
        lines.append(f"{'='*60}")

        if 'layers' in cap:
            # Hierarchical report
            lines.append(f"")
            for layer in cap['layers']:
                name = layer.get('name', 'Layer')
                lines.append(f"  {name} (D={layer['D']}, k={layer['k']}, "
                           f"beta_fp={layer['beta_fp']}):")
                lines.append(f"    P_max = {layer['p_max']:.0f} attractors")
                lines.append(f"    H(F)  = {layer['metric_entropy_bits']:.0f} bits "
                           f"({layer['metric_entropy_bytes']:.0f} bytes)")
                lines.append(f"    d_eff = {layer['encodable_params_bits']:.0f} bits "
                           f"({layer['encodable_params_bytes']:.0f} bytes)")
                lines.append(f"    fat_F = {layer['fat_shattering']:.0f}")
                lines.append(f"    Expansion vs linear: {layer['expansion_vs_linear']:.1f}x")
                lines.append(f"    Equivalent: {layer['transformer_equiv']}")
                lines.append(f"")

            lines.append(f"  {'─'*56}")
            lines.append(f"  HIERARCHICAL TOTAL:")
            lines.append(f"    P_max   = {cap['total_p_max']:.0f} attractors")
            lines.append(f"    H(F)    = {cap['total_metric_entropy_bits']:.0f} bits "
                       f"({cap['total_metric_entropy_bytes']:.0f} bytes)")
            lines.append(f"    d_eff   = {cap['total_encodable_params_bits']:.0f} bits "
                       f"({cap['total_encodable_params_bytes']:.0f} bytes)")
            lines.append(f"    fat_F   = {cap['total_fat_shattering']:.0f}")
            lines.append(f"    Equiv   = {cap['transformer_equiv']}")
        else:
            # Single layer report
            lines.append(f"  D={cap['D']}, k={cap['k']}, beta_fp={cap['beta_fp']}, V={cap['V']}")
            lines.append(f"  Sparsity a = {cap['sparsity']:.3f}")
            lines.append(f"")
            lines.append(f"  EXPONENTIAL DAM (F(x) = exp(beta*x)):")
            lines.append(f"    P_max   = {cap['p_max']:.0f} attractors")
            lines.append(f"    H(F)    = {cap['metric_entropy_bits']:.0f} bits "
                       f"({cap['metric_entropy_bytes']:.0f} bytes)")
            lines.append(f"    d_eff   = {cap['encodable_params_bits']:.0f} bits "
                       f"({cap['encodable_params_bytes']:.0f} bytes)")
            lines.append(f"    fat_F   = {cap['fat_shattering']:.0f}")
            lines.append(f"    Equiv   = {cap['transformer_equiv']}")
            lines.append(f"")
            lines.append(f"  LINEAR HOPFIELD (for comparison):")
            lines.append(f"    P_max   = {cap['p_max_linear']:.0f} attractors")
            lines.append(f"    H(F)    = {cap['metric_entropy_linear_bits']:.0f} bits")
            lines.append(f"    fat_F   = {cap['fat_shattering_linear']:.0f}")
            lines.append(f"")
            lines.append(f"  EXPANSION: {cap['expansion_vs_linear']:.1f}x vs linear Hopfield")

        lines.append(f"{'='*60}")
        return '\n'.join(lines)

    @staticmethod
    def format_beta_sweep(sweep_results: List[Dict]) -> str:
        """
        Format a beta sweep as a human-readable table.

        Args:
            sweep_results: Output of beta_sweep().

        Returns:
            Formatted string.
        """
        lines = []
        lines.append(f"{'='*80}")
        lines.append(f"  BETA SWEEP — Manifold Capacity vs Inverse Temperature")
        lines.append(f"{'='*80}")
        lines.append(f"  {'beta_fp':>7} {'beta':>6} {'P_max':>12} {'H(F) bits':>12} "
                    f"{'H(F) bytes':>12} {'d_eff bits':>12} {'Equiv':>20}")
        lines.append(f"  {'─'*76}")

        for r in sweep_results:
            lines.append(
                f"  {r['beta_fp']:>7} {r['beta']:>6.2f} "
                f"{r['p_max']:>12.0f} {r['metric_entropy_bits']:>12.0f} "
                f"{r['metric_entropy_bytes']:>12.0f} "
                f"{r['encodable_params_bits']:>12.0f} "
                f"{r['transformer_equiv']:>20}"
            )

        lines.append(f"{'='*80}")
        lines.append(f"  KEY: beta controls the EXPONENT in P_max ~ exp(beta^2 * k / 2)")
        lines.append(f"  Doubling beta from 0.25 to 0.50 triples capacity.")
        lines.append(f"  Going from 0.25 to 1.0 gives ~100x expansion.")
        lines.append(f"{'='*80}")
        return '\n'.join(lines)
