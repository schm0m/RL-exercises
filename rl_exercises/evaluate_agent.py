import hydra
from train_agent import make_env, evaluate


@hydra.main("rl_exercises/configs", "eval", version_base="1.1")
def evaluate_agent(cfg):
    env = make_env(cfg.env_name)
    # TODO: make agent
    agent = None
    agent.load(cfg.policy_path)
    return_mean = evaluate(env, agent, cfg.num_episodes)
    print(return_mean)


if __name__ == "__main__":
    evaluate_agent()
