import gymnasium as gym
import gym_cutting_stock
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2310393_2311514_2310273_2311428.policy2310393_2311514_2310273_2311428 import Policy2310393_2311514_2310273_2311428

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",
)

NUM_EPISODES = 100

if __name__ == "__main__":
    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # Uncomment the following code to test your policy
    # Reset the environment
    observation, info = env.reset(seed=42)
    print(info)

    policy2310393_2311514_2310273_2311428 = Policy2310393_2311514_2310273_2311428()
    for _ in range(200):
        action = policy2310393_2311514_2310273_2311428.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)

        if terminated or truncated:
            observation, info = env.reset()

env.close()
