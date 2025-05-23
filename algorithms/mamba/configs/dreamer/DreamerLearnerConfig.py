import torch

from mamba.agent.learners.DreamerLearner import DreamerLearner
from mamba.configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerLearnerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_LR = 2e-4
        self.ACTOR_LR = 5e-4
        self.VALUE_LR = 5e-4
        self.CAPACITY = 250000
        self.MIN_BUFFER_SIZE = 500
        self.MODEL_EPOCHS = 60
        self.EPOCHS = 4
        self.PPO_EPOCHS = 5
        self.MODEL_BATCH_SIZE = 40
        self.BATCH_SIZE = 40
        self.SEQ_LENGTH = 20
        self.N_SAMPLES = 1
        self.TARGET_UPDATE = 1
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.GRAD_CLIP = 100.0
        self.HORIZON = 15
        self.ENTROPY = 0.001
        self.ENTROPY_ANNEALING = 0.99998
        self.GRAD_CLIP_POLICY = 100.0

    def create_learner(self, wandb_config):
        return DreamerLearner(self, wandb_config)
