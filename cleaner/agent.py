from utils import RAY_DIR


class Agent(object):
    def __init__(self, policy_name):
        assert policy_name in ["ppo", "dqn"], f"unknown policy name: {policy_name}"
        self.policy_name = policy_name

    @staticmethod
    def from_checkpoint(policy_name, run_name, checkpoint_num):
        agent = Agent(policy_name)
        agent.run_name = run_name
        agent.agent_num = agent_num
        agent.config = dill.load(open(f"{self.results_dir}/params.pkl", "rb"))
        agent.create_trainer()

    def prepare_to_run(self, run_name, agent_num):
        self.results_dir = f"{RAY_DIR}/{run_name}"
        self.run_name = run_name
        self.agent_num = agent_num
        self.name = f"{run_name}:{agent_num}"

    def create_trainer(self):
        checkpoint_path = f"{self.results_dir}/checkpoint_{str(checkpoint_num).zfill(6)}/checkpoint-{checkpoint_num}"
        trainer = trainer_from_config(self.config, results_dir="tmp")
        trainer.restore(checkpoint_path)
        self.trainer = trainer
        return trainer
