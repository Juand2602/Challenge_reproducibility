import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import time

# --- PASO 1: Importar tu EvolutionStrategyAgent de Fase 1 ---
try:
    from Integracion_Red_Neuronal import EvolutionStrategyAgent
except ImportError:
    print("ADVERTENCIA CRÍTICA: No se pudo importar 'EvolutionStrategyAgent'.")
    print("Asegúrate de que el archivo de la Fase 1 ('es_agent_core_fase1_conciso.py') esté accesible.")
    EvolutionStrategyAgent = None

# --- Preprocesamiento y Red CNN para Atari ---
def make_atari_env(env_id, frame_stack=4, seed=None, render_mode_str=None, custom_frame_skip=1):
    env = gym.make(env_id, render_mode=render_mode_str)
    # Aplicar preprocesamiento estándar de Atari
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, 
                             frame_skip=custom_frame_skip, noop_max=30, scale_obs=True)
    # Apilar frames
    env = FrameStack(env, num_stack=frame_stack)
    if seed is not None:
        env.action_space.seed(seed) # Sembrar espacio de acción
        # La semilla principal para el PRNG del entorno se establece en env.reset(seed=seed)
    return env

class AtariCNNPolicy(nn.Module):
    def __init__(self, num_actions, input_channels=4):
        super(AtariCNNPolicy, self).__init__()
        self.input_channels = input_channels
        # Arquitectura CNN (Nature DQN / A3C)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Función para calcular la dimensión de salida de las capas convolucionales
        def conv_output_dim(h_in=84, w_in=84): # Asumir entrada 84x84
            with torch.no_grad(): # No se necesitan gradientes para este cálculo
                # Crear una observación ficticia
                dummy_obs = torch.zeros(1, self.input_channels, h_in, w_in)
                # Pasar por las capas convolucionales
                x = F.relu(self.conv1(dummy_obs))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                return int(np.prod(x.size())) # Producto de todas las dimensiones
        
        self.fc_input_dim = conv_output_dim()
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc_out = nn.Linear(512, num_actions) # Salida de logits para acciones discretas
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2)) # Inicialización común en RL
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # Asegurar que la entrada sea float y esté normalizada a [0,1]
        # AtariPreprocessing con scale_obs=True ya debería devolver [0,1] float.
        # FrameStack puede devolver LazyFrames que se convierten a uint8 al hacer np.array().
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        # Si ya es float pero no está en [0,1] (por si acaso)
        elif x.ndim == 4 and x.max() > 1.0 and x.min() >=0.0 : # (N,C,H,W)
             x = x / 255.0
        elif x.ndim == 3 and x.max() > 1.0 and x.min() >=0.0 : # (C,H,W)
             # Añadir batch dim temporalmente para la normalización
             x = (x.unsqueeze(0) / 255.0).squeeze(0)


        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Aplanar para la capa densa
        x = F.relu(self.fc1(x))
        action_logits = self.fc_out(x)
        return action_logits

    def get_action(self, observation, device="cpu"):
        # La observación de FrameStack (AtariPreprocessing) es un LazyFrames.
        # Convertirlo a un array NumPy primero, luego a Tensor.
        observation_np = np.array(observation) if not isinstance(observation, np.ndarray) else observation
        
        # Asegurar que tenga la forma (C, H, W) y añadir batch_dim -> (1, C, H, W)
        if observation_np.ndim == 3: # Asume (C, H, W)
            obs_tensor = torch.from_numpy(observation_np).unsqueeze(0).to(device, dtype=torch.float32)
        elif observation_np.ndim == 4: # Asume ya (N, C, H, W)
            obs_tensor = torch.from_numpy(observation_np).to(device, dtype=torch.float32)
        else:
            raise ValueError(f"Forma de observación Atari inesperada: {observation_np.shape}")
        
        self.to(device) # Asegurar que la red esté en el dispositivo
        with torch.no_grad():
            logits = self.forward(obs_tensor)
            action = torch.argmax(logits, dim=1).item() # Tomar la acción con el mayor logit
        return action

    def get_flattened_parameters(self):
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_flattened_parameters(self, flat_params):
        offset = 0
        for p in self.parameters():
            s, numel = p.shape, p.numel()
            chunk = flat_params[offset : offset + numel].reshape(s)
            p.data = torch.from_numpy(chunk.astype(np.float32)).to(p.device)
            offset += numel

# --- Función Principal para Ejecutar un Experimento Atari ---
def main_atari_experiment(env_id="PongNoFrameskip-v4", 
                          num_generations=100, 
                          population_size=32,
                          learning_rate=0.005, 
                          noise_std_dev=0.02, 
                          weight_decay=0.01, 
                          seed=42, 
                          device_str="cpu",
                          print_every_generations=5,
                          frame_stack_count=4,
                          custom_fs=1):

    if EvolutionStrategyAgent is None:
        print("Error: EvolutionStrategyAgent no está disponible. Saliendo.")
        return

    print(f"\n--- Iniciando Experimento Atari ---")
    print(f"Entorno: {env_id}, Generaciones: {num_generations}, Población: {population_size}, LR: {learning_rate}, RuidoStd: {noise_std_dev}")
    device = torch.device(device_str)

    try:
        # Crear entorno de prueba para obtener dimensiones
        env_dims_check = make_atari_env(env_id, frame_stack=frame_stack_count, seed=seed, custom_frame_skip=custom_fs)
        num_actions = env_dims_check.action_space.n
        # obs_shape es (stack_size, height, width)
        input_channels = env_dims_check.observation_space.shape[0] 
        env_dims_check.close()
    except Exception as e:
        print(f"Error al crear el entorno Atari '{env_id}': {e}")
        print("Asegúrate de tener las ROMs de Atari instaladas ('pip install gymnasium[atari] gymnasium[accept-rom-license]').")
        return

    policy_net = AtariCNNPolicy(num_actions, input_channels=input_channels).to(device)

    # Para Atari, el agente ES necesitará una forma de crear el entorno preprocesado
    # Se puede modificar el agente para aceptar una función `env_creator_func`
    # o manejar la creación del entorno de Atari de forma especial si `env_name` es de Atari.
    # Por simplicidad, asumiremos que tu agente ES puede ser modificado o que
    # su `_evaluate_fitness` puede detectar y crear el entorno Atari correctamente.
    # Una forma es pasar una función lambda para crear el entorno:
    env_creator = lambda name, s: make_atari_env(name, frame_stack=frame_stack_count, seed=s, custom_frame_skip=custom_fs)

    agent = EvolutionStrategyAgent(
        policy_network=policy_net,
        env_name=env_id, # El agente lo usará, idealmente con env_creator
        learning_rate=learning_rate,
        noise_std_dev=noise_std_dev,
        population_size=population_size,
        weight_decay_coeff=weight_decay,
        fitness_shaping=True,
        seed=seed,
        device=device_str
        # env_creator_func=env_creator # Si modificaste tu agente para esto
    )
    # Si no pasas env_creator_func, asegúrate que tu agente ES en _evaluate_fitness
    # use make_atari_env(self.env_name, frame_stack=..., seed=eval_seed, custom_frame_skip=...)
    # cuando detecte que self.env_name es un entorno Atari.

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
        # Aquí también, el agente debe ser capaz de crear el entorno Atari correctamente para la evaluación
        reward = agent._evaluate_fitness(final_params_copy, eval_seed=eval_s)
        final_rewards.append(reward)
    if final_rewards:
        print(f"Resultado final {env_id}: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f} en {num_evals} episodios.")


if __name__ == "__main__":
    DEVICE_CHOICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Configuración para el experimento Atari ---
    ATARI_ENV = "PongNoFrameskip-v4" # Otros: "BreakoutNoFrameskip-v4"
    ATARI_GENS = 100                 # Aumentar significativamente para Atari (ej. 500-2000+)
    ATARI_POP = 32                   # Atari puede ser más lento por episodio
    ATARI_LR = 0.005                 # Suele ser más bajo para CNNs
    ATARI_NOISE_STD = 0.02
    ATARI_WD = 0.01
    ATARI_FRAME_STACK = 4
    ATARI_FRAME_SKIP = 1             # Para el experimento de invarianza de frame-skip, este valor se variaría.

    main_atari_experiment(
        env_id=ATARI_ENV,
        num_generations=ATARI_GENS,
        population_size=ATARI_POP,
        learning_rate=ATARI_LR,
        noise_std_dev=ATARI_NOISE_STD,
        weight_decay=ATARI_WD,
        seed=42,
        device_str=DEVICE_CHOICE,
        print_every_generations=5,
        frame_stack_count=ATARI_FRAME_STACK,
        custom_fs=ATARI_FRAME_SKIP
    )
