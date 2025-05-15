import gym
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os

class Policy:
    def __init__(self, input_size, hidden_size, output_size):
        self.shapes = [(input_size, hidden_size), (hidden_size,),
                       (hidden_size, hidden_size), (hidden_size,),
                       (hidden_size, output_size), (output_size,)]
        self.params = self._init_params()

    def _init_params(self):
        return [np.random.randn(*shape) * 0.1 for shape in self.shapes]

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def forward(self, x):
        x = np.array(x, dtype=np.float32)
        w1, b1, w2, b2, w3, b3 = self.params
        x = np.tanh(x @ w1 + b1)
        x = np.tanh(x @ w2 + b2)
        return np.tanh(x @ w3 + b3)

    def act(self, obs):
        obs = np.array(obs, dtype=np.float32)
        return self.forward(obs)


def evaluate(env, policy, episodes=2, render=False):
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render:
                env.render()
    return total_reward / episodes


def evolution_strategy(env_name='Walker2d-v4', pop_size=50, sigma=0.05, alpha=0.02, iterations=300):
    env = gym.make(env_name, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Policy(obs_dim, 64, act_dim)
    reward_history = []

    for iteration in range(iterations):
        params = policy.get_params()
        noise = [[np.random.randn(*p.shape) for p in params] for _ in range(pop_size)]

        rewards = []
        for i in range(pop_size):
            new_params = [p + sigma * eps for p, eps in zip(params, noise[i])]
            policy.set_params(new_params)
            r = evaluate(env, policy)
            rewards.append(r)

        rewards = np.array(rewards)
        A = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        for i in range(len(params)):
            update = np.zeros_like(params[i])
            for j in range(pop_size):
                update += A[j] * noise[j][i]
            params[i] += alpha / (pop_size * sigma) * update

        policy.set_params(params)
        test_reward = evaluate(env, policy)
        reward_history.append(test_reward)
        print(f"Iter {iteration + 1}/{iterations}: Reward {test_reward:.2f}")

    env.close()
    return policy, reward_history


def record_video(env_name, policy, video_path="C:\\Users\\Usuario\\Downloads\\HalfCheetah-v4.mp4"):
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()

    frames = []
    done = False

    while not done:
        frame = env.render()
        frames.append(frame)

        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    imageio.mimsave(video_path, frames, fps=30)
    print(f"🎥 Video guardado en {video_path}")


def save_results(policy, rewards, path="C:\\Users\\Usuario\\Desktop\\Challenge_reproducibility\\results"):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "paramsHalfCheetah-v4.npy"), policy.get_params())
    np.save(os.path.join(path, "rewardsHalfCheetah-v4.npy"), rewards)
    print(f"✅ Resultados guardados en carpeta: {path}")


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Iteración")
    plt.ylabel("Recompensa promedio")
    plt.title("Entrenamiento ES en MuJoCo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rewards_plot.png")
    plt.show()


if __name__ == "__main__":
    env_name = "HalfCheetah-v4"  # Puedes cambiarlo por "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", etc.
    policy, rewards = evolution_strategy(env_name=env_name, iterations=100)

    save_results(policy, rewards)
    plot_rewards(rewards)
    record_video(env_name, policy)
