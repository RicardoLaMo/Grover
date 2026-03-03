# THORN Math

## 1) Routing distribution
- Equation:
  `pi_{i,h,m} = softmax_m(logits_{i,h,m})`
- Code mapping:
  - Router: `thorn/routing/router.py`
  - Simplex assertion: `thorn/debug.py::assert_probability_simplex`

## 2) Routed adjacency mixture
- Equation:
  `A_tilde_{ij}^{(i,h)} = sum_m pi_{i,h,m} * A_{ij}^{(m)}`
- Code mapping:
  - Attention layer: `thorn/layers/routed_attention.py`
  - Union graph masks/weights: `thorn/contracts.py::UnifiedNeighborhood`

## 3) Attention normalization over neighbors
- Equation (per node `i` and head `h`):
  `alpha_{j->i}^{(h)} = softmax_{j in N(i)}( score_{j->i}^{(h)} )`
- Code mapping:
  - `RoutedNeighborhoodAttention._edge_softmax`
  - Test invariant: `tests/test_router_attention.py::test_attention_normalizes_over_incoming_neighbors`

## 4) Alignment surrogate signal
- Surrogate definition:
  neighborhood-overlap agreement (Jaccard) between incoming neighbor sets across two views.
- Code mapping:
  - `thorn/alignment/surrogate.py::neighborhood_overlap_score`
  - `thorn/alignment/interface.py::align_views`

## 5) LID observer feature (Levina-Bickel MLE)
- Equation:
  `LID(x) = - ( (1/(k-1)) * sum_{j=1}^{k-1} log(r_j/r_k) )^{-1}`
- Code mapping:
  - `thorn/observers/lid.py`

## 6) Drift diagnostics
- Routing shift:
  `|| mean(pi_test_early) - mean(pi_test_late) ||_1`
- Collapse score:
  `max_m mean_{i,h}(pi_{i,h,m})`
- Code mapping:
  - `thorn/train/harness.py::_drift_metrics`
