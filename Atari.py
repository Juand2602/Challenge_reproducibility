import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time

# --- PASO 1: Importar tu EvolutionStrategyAgent de Fase 1 ---
# Asegúrate de que el archivo 'es_agent_core_fase1_conciso.py' (o como lo hayas llamado)
# esté en el mismo directorio o en tu PYTHONPATH.
try:
    from Integracion_Red_Neuronal import EvolutionStrategyAgent 
except ImportError:
    print("ADVERTENCIA CRÍTICA: No se pudo importar 'EvolutionStrategyAgent'.")
    print("Asegúrate de que el archivo de la Fase 1 ('es_agent_core_fase1_conciso.py') esté accesible.")
    EvolutionStrategyAgent = None # Para que el resto del script no falle inmediatamente

# --- Red MLP para MuJoCo ---
class MuJoCoMLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, action_space):
        super(MuJoCoMLPPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, act_dim)
        # Registrar buffers para action_scale y action_bias para que se muevan con .to(device)
        self.register_buffer('action_scale', torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_logits = self.fc_out(x)
        # Escalar la acción: tanh produce [-1,1], luego escalar y trasladar al rango del entorno
        action_scaled = torch.tanh(action_logits) * self.action_scale + self.action_bias
        return action_scaled

    def get_action(self, observation, device="cpu"):
        if not isinstance(observation, torch.Tensor): 
            observation = torch.tensor(observation, dtype=torch.float32)
        if observation.ndim == 1: 
            observation = observation.unsqueeze(0) # Añadir dimensión de batch
        observation = observation.to(device)
        self.to(device) # Asegurar que la red esté en el dispositivo correcto
        with torch.no_grad(): 
            action = self.forward(observation)
        return action.cpu().numpy().flatten()

    def get_flattened_parameters(self):
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_flattened_parameters(self, flat_params):
        offset = 0
        for p in self.parameters():
            s, numel = p.shape, p.numel()
            chunk = flat_params[offset : offset + numel].reshape(s)
            p.data = torch.from_numpy(chunk.astype(np.float32)).to(p.device)
            offset += numel

# --- Función Principal para Ejecutar un Experimento MuJoCo ---
def main_mujoco_experiment(env_id="InvertedPendulum-v4", 
                           num_generations=200, 
                           population_size=64,
                           learning_rate=0.02, 
                           noise_std_dev=0.05, 
                           weight_decay=0.001, 
                           seed=42, 
                           device_str="cpu",
                           print_every_generations=10):

    if EvolutionStrategyAgent is None:
        print("Error: EvolutionStrategyAgent no está disponible. Saliendo.")
        return

    print(f"\n--- Iniciando Experimento MuJoCo ---")
    print(f"Entorno: {env_id}, Generaciones: {num_generations}, Población: {population_size}, LR: {learning_rate}, RuidoStd: {noise_std_dev}")
    device = torch.device(device_str)

    try:
        env_dims_check = gym.make(env_id)
        obs_dim = env_dims_check.observation_space.shape[0]
        act_dim = env_dims_check.action_space.shape[0]
        act_space_details = env_dims_check.action_space
        env_dims_check.close()
    except Exception as e:
        print(f"Error al crear el entorno MuJoCo '{env_id}': {e}")
        print("Asegúrate de que MuJoCo esté instalado ('pip install gymnasium[mujoco]') y configurado.")
        return
        
    policy_net = MuJoCoMLPPolicy(obs_dim, act_dim, act_space_details).to(device)

    agent = EvolutionStrategyAgent(
        policy_network=policy_net,
        env_name=env_id,
        learning_rate=learning_rate,
        noise_std_dev=noise_std_dev,
        population_size=population_size,
        weight_decay_coeff=weight_decay,
        fitness_shaping=True,
        seed=seed,
        device=device_str
    )
    print(f"Agente ES para {env_id} creado con {agent.num_params} parámetros.")
    
    start_train_time = time.time()
    agent.train(num_generations=num_generations, print_every=print_every_generations)
    end_train_time = time.time()
    print(f"Entrenamiento para '{env_id}' completado. Tiempo: {end_train_time - start_train_time:.2f}s.")

    # Evaluación final
    print(f"\nEvaluando política final para {env_id}...")
    final_rewards = []
    num_evals = 10
    final_params_copy = agent.policy_network.get_flattened_parameters().copy()
    for i in range(num_evals):
        eval_s = seed + num_generations + i 
        # Asumimos que agent._evaluate_fitness maneja la creación del entorno y el sembrado
        reward = agent._evaluate_fitness(final_params_copy, eval_seed=eval_s)
        final_rewards.append(reward)
    if final_rewards:
        print(f"Resultado final {env_id}: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f} en {num_evals} episodios.")

if __name__ == "__main__":
    DEVICE_CHOICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Configuración para el experimento MuJoCo ---
    MUJOCO_ENV = "InvertedPendulum-v4" # Otros: "Hopper-v4", "Walker2d-v4"
    MUJOCO_GENS = 200
    MUJOCO_POP = 64
    MUJOCO_LR = 0.02
    MUJOCO_NOISE_STD = 0.05
    MUJOCO_WD = 0.001

    main_mujoco_experiment(
        env_id=MUJOCO_ENV,
        num_generations=MUJOCO_GENS,
        population_size=MUJOCO_POP,
        learning_rate=MUJOCO_LR,
        noise_std_dev=MUJOCO_NOISE_STD,
        weight_decay=MUJOCO_WD,
        seed=42,
        device_str=DEVICE_CHOICE,
        print_every_generations=10
    )
