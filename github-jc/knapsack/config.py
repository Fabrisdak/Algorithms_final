"""
Configuration constants and hyperparameters for the 0/1 Knapsack project.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts", "output")

# Random seed for reproducibility
RANDOM_SEED = 42

# ML hyperparameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 15
MLP_HIDDEN_LAYERS = (128, 64)
MLP_MAX_ITER = 500
TRAINING_INSTANCES = 2000
TRAIN_TEST_SPLIT = 0.2

# Experiment settings
REPORT_SIZE_TIERS = [
    {"n": 20, "W": 50},
    {"n": 50, "W": 100},
    {"n": 100, "W": 200},
]
REPORT_INSTANCES_PER_FAMILY = 20

# Chart settings
CHART_DPI = 300
CHART_FIGSIZE = (10, 6)
SEABORN_PALETTE = "colorblind"
