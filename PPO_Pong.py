import gym
import imageio
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gym.wrappers import AtariPreprocessing, FrameStack

# Crear entorno con render_mode
def make_env():
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, 4)
    return env

# Grabar video de la política entrenada
def record_video(model, env, path="ppo_pong.mp4", max_frames=500):
    obs = env.reset()[0]
    frames = []
    done = False
    count = 0
    while not done and count < max_frames:
        frame = env.render()
        frames.append(frame)
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        count += 1
    env.close()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=30)
    print(f"Video guardado en {path}")

# Evaluación de la política
def evaluate_policy(model, env, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return rewards

if __name__ == "__main__":
    env = make_env()

    # Crear modelo PPO en CPU
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    # Entrenar (menos pasos por simplicidad)
    model.learn(total_timesteps=150_000)

    # Guardar modelo entrenado
    model.save("ppo_pong")

    # Evaluar desempeño
    rewards = evaluate_policy(model, env)
    avg_reward = sum(rewards) / len(rewards)
    print(f"Recompensa promedio en evaluación: {avg_reward:.2f}")

    # Graficar recompensas
    plt.plot(rewards, marker='o')
    plt.title("Recompensa por episodio (evaluación PPO)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total")
    plt.grid(True)
    plt.savefig("rewards_pong.png")
    plt.show()

    # Grabar video
    record_video(model, make_env(), path="C:/Users/Franchesco/Desktop/challenge/results/ppo_pong.mp4")