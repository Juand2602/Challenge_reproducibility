import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gym

env = gym.make("HalfCheetah-v4")
model = PPO("MlpPolicy", env, verbose=1)

# Entrenar
model.learn(total_timesteps=5000000)

# Guardar
model.save("ppo_halfcheetah")

# Evaluar en varios episodios y guardar recompensas
num_episodes = 20
rewards = []

for _ in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    rewards.append(total_reward)

# Mostrar recompensa promedio
print(f"Recompensa promedio en {num_episodes} episodios: {sum(rewards)/num_episodes:.2f}")

# Graficar
plt.plot(rewards, marker='o')
plt.title("Recompensa total por episodio (Evaluación PPO)")
plt.xlabel("Episodio")
plt.ylabel("Recompensa total")
plt.grid(True)
plt.show()
