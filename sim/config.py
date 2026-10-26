"""
Central experiment configuration.

NOTE ON SEED COUNT: the assignment sheet asks for >=100 unique task sets per
configuration. The simulator benchmarks at ~15-20ms/run worst case, so the
full campaign (28 configurations x 100 seeds = 2800 runs) completes in well
under a minute -- we use the full spec value of 100 directly.
"""

NUM_SEEDS = 100         # random task sets averaged per configuration (spec: >=100)
HORIZON_MS = 100000.0     # simulation time horizon per run, in ms
MAX_PARTITION_ATTEMPTS = 200  # regenerate task set this many times if WFD infeasible

# Fixed defaults used when a parameter isn't the one being swept
DEFAULT_UP = 0.5
DEFAULT_UA = 0.2
DEFAULT_M = 4
