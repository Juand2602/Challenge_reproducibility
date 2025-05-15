import gym
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

# Política MLP simple
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


# Evaluación individual (para multiprocessing)
def evaluate_candidate(args):
    env_name, base_params, noise_i, sigma = args
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = Policy(obs_dim, 32, act_dim)

    new_params = [p + sigma * n for p, n in zip(base_params, noise_i)]
    policy.set_params(new_params)

    total_reward = 0
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    return total_reward


# Algoritmo ES con paralelización
def evolution_strategy(env_name='HalfCheetah-v4', pop_size=200, sigma=0.1, alpha=0.02, iterations=100):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = Policy(obs_dim, 32, act_dim)
    reward_history = []

    for iteration in range(iterations):
        params = policy.get_params()
        noise = [[np.random.randn(*p.shape) for p in params] for _ in range(pop_size)]

        # Preparar argumentos para evaluación paralela
        args_list = [(env_name, params, eps, sigma) for eps in noise]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            rewards = pool.map(evaluate_candidate, args_list)

        rewards = np.array(rewards)
        A = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        # Actualizar parámetros
        for i in range(len(params)):
            update = sum(A[j] * noise[j][i] for j in range(pop_size))
            params[i] += alpha / (pop_size * sigma) * update

        policy.set_params(params)

        # Evaluación del modelo actual
        test_reward = evaluate_candidate((env_name, params, [np.zeros_like(p) for p in params], 0.0))
        reward_history.append(test_reward)
        print(f"Iteración {iteration + 1}/{iterations}: Recompensa promedio = {test_reward:.2f}")

    env.close()
    return policy, reward_history


def record_video(env_name, policy, video_path="C:\\Users\\Usuario\\Downloads\\Swimmer-v4Para.mp4"):
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
    np.save(os.path.join(path, "paramsSwimmer-v4Para.npy"), policy.get_params())
    np.save(os.path.join(path, "rewardsSwimmer-v4Para.npy"), rewards)
    print(f"✅ Resultados guardados en carpeta: {path}")


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Iteración")
    plt.ylabel("Recompensa promedio")
    plt.title("Entrenamiento ES en MuJoCo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Swimmer-v4Para.png")
    plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Requerido en Windows
    env_name = "Swimmer-v4"  # Puedes cambiar por HalfCheetah-v4, Walker2d-v4, etc.
    policy, rewards = evolution_strategy(env_name=env_name, pop_size=200, iterations=100)
    save_results(policy, rewards)
    plot_rewards(rewards)
    record_video(env_name, policy)