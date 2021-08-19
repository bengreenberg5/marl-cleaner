from utils import RAY_DIR

class Agent:
    def __init__(self, policy_name, run_name, agent_num):
        self.run_name = run_name
        self.agent_num = agent_num
        self.name = f"{run_name}:{agent_num}"
        self.results_dir = f"{RAY_DIR}/{run_name}"
        self.config = dill.load(open(f"{self.results_dir}/params.pkl", "rb"))
        self.policy_name = self.config["policy_config"][agent_num]

    def create_trainer(self, checkpoint_num):
        checkpoint_path = f"{self.results_dir}/checkpoint_{str(checkpoint_num).zfill(6)}/checkpoint-{checkpoint_num}"
        trainer = trainer_from_config(self.config, results_dir="tmp")
        trainer.restore(checkpoint_path)
        self.trainer = trainer
        return trainer

