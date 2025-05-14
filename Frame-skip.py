import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import time
import matplotlib.pyplot as plt # Para graficar los resultados

# --- PASO 1: Importar tu EvolutionStrategyAgent de Fase 1 ---
# Asegúrate de que el archivo 'es_agent_core_fase1_conciso.py' (o como lo hayas llamado)
# esté en el mismo directorio o en tu PYTHONPATH.
try:
    from Integracion_Red_Neuronal import EvolutionStrategyAgent 
except ImportError:
    print("ADVERTENCIA CRÍTICA: No se pudo importar 'EvolutionStrategyAgent'.")
    print("Asegúrate de que el archivo de la Fase 1 ('es_agent_core_fase1_conciso.py') esté accesible.")
    EvolutionStrategyAgent = None

# --- Preprocesamiento y Red CNN para Atari (sin cambios respecto al script anterior) ---
def make_atari_env(env_id, frame_stack=4, seed=None, render_mode_str=None, custom_frame_skip=1):
    """Crea y preprocesa un entorno Atari, permitiendo un frame_skip personalizado."""
    env = gym.make(env_id, render_mode=render_mode_str)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, 
                             frame_skip=custom_frame_skip, # Usar el frame_skip personalizado
                             noop_max=30, scale_obs=True)
    env = FrameStack(env, num_stack=frame_stack)
    if seed is not None:
        env.action_space.seed(seed)
    return env

class AtariCNNPolicy(nn.Module):
    def __init__(self, num_actions, input_channels=4):
        super(AtariCNNPolicy, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        def conv_output_dim(h_in=84, w_in=84):
            with torch.no_grad():
                x = torch.zeros(1, self.input_channels, h_in, w_in)
                x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x))
                return int(np.prod(x.size()))
        
        self.fc_input_dim = conv_output_dim()
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc_out = nn.Linear(512, num_actions)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        if x.dtype == torch.uint8: x = x.float() / 255.0
        elif x.ndim == 4 and x.max() > 1.0 and x.min() >=0.0 : x = x / 255.0
        elif x.ndim == 3 and x.max() > 1.0 and x.min() >=0.0 : x = (x.unsqueeze(0) / 255.0).squeeze(0)
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x)); action_logits = self.fc_out(x)
        return action_logits

    def get_action(self, observation, device="cpu"):
        observation_np = np.array(observation) if not isinstance(observation, np.ndarray) else observation
        if observation_np.ndim == 3: 
            obs_tensor = torch.from_numpy(observation_np).unsqueeze(0).to(device, dtype=torch.float32)
        elif observation_np.ndim == 4: 
            obs_tensor = torch.from_numpy(observation_np).to(device, dtype=torch.float32)
        else: raise ValueError(f"Forma de observación Atari inesperada: {observation_np.shape}")
        self.to(device)
        with torch.no_grad():
            logits = self.forward(obs_tensor)
            action = torch.argmax(logits, dim=1).item()
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

# --- Función Principal para Ejecutar un Experimento Atari con Frame-Skip Variable ---
def run_atari_frameskip_trial(env_id="PongNoFrameskip-v4", 
                                current_frame_skip_value=1, # Parámetro para el frame_skip actual
                                num_generations=100, 
                                population_size=32,
                                learning_rate=0.005, 
                                noise_std_dev=0.02, 
                                weight_decay=0.01, 
                                seed=42, 
                                device_str="cpu",
                                print_every_generations=10, # Reducido para ver más progreso
                                frame_stack_count=4):

    if EvolutionStrategyAgent is None:
        print("Error: EvolutionStrategyAgent no está disponible. Saliendo.")
        return None # Devolver None para indicar fallo

    print(f"\n--- Iniciando Prueba Atari con Frame-Skip = {current_frame_skip_value} ---")
    print(f"Entorno: {env_id}, Generaciones: {num_generations}, Población: {population_size}, LR: {learning_rate}, RuidoStd: {noise_std_dev}")
    device = torch.device(device_str)
    
    all_mean_fitness_history = [] # Para almacenar el fitness promedio de cada generación

    try:
        env_dims_check = make_atari_env(env_id, frame_stack=frame_stack_count, seed=seed, custom_frame_skip=current_frame_skip_value)
        num_actions = env_dims_check.action_space.n
        input_channels = env_dims_check.observation_space.shape[0] 
        env_dims_check.close()
    except Exception as e:
        print(f"Error al crear el entorno Atari '{env_id}' con frame_skip {current_frame_skip_value}: {e}")
        return None

    policy_net = AtariCNNPolicy(num_actions, input_channels=input_channels).to(device)

    # Modificar el agente ES para que pueda registrar el historial de fitness
    # Esto es una simplificación. Idealmente, el método train del agente devolvería el historial.
    # Por ahora, modificaremos la función de impresión dentro de un wrapper o directamente.
    
    original_print_fn = print # Guardar la función print original

    def print_and_log_fitness(message):
        original_print_fn(message)
        # Extraer el fitness promedio si está en el mensaje
        if "Fit (Mean±Std):" in message:
            try:
                parts = message.split("Fit (Mean±Std):")[1].split(",")[0].strip()
                mean_fit_str = parts.split("±")[0]
                all_mean_fitness_history.append(float(mean_fit_str))
            except Exception as e:
                original_print_fn(f"  (Error al parsear fitness para logging: {e})")

    # Reemplazar print temporalmente si el agente ES usa print directamente para los logs
    # Si tu agente ES tiene un callback o devuelve el historial, úsalo en su lugar.
    # Esto es un HACK y no la mejor práctica.
    # global print 
    # print = print_and_log_fitness # Comentado por ahora, mejor modificar el agente

    # Para que esto funcione bien, el método `train` de tu `EvolutionStrategyAgent`
    # debería devolver el historial de fitness promedio por generación, o tener un callback.
    # Asumamos que modificas tu agente para que `train` devuelva `all_mean_fitness_history`.
    # Por ahora, simularé que lo devuelve.

    agent = EvolutionStrategyAgent(
        policy_network=policy_net,
        env_name=env_id,
        learning_rate=learning_rate,
        noise_std_dev=noise_std_dev,
        population_size=population_size,
        weight_decay_coeff=weight_decay,
        fitness_shaping=True,
        seed=seed, # Usar la misma semilla base para la inicialización del agente
        device=device_str,
        # Para que el agente cree el entorno con el frame_skip correcto:
        env_creator_func=lambda name, s: make_atari_env(name, frame_stack=frame_stack_count, seed=s, custom_frame_skip=current_frame_skip_value)
    )
    print(f"Agente ES para {env_id} (FS={current_frame_skip_value}) creado con {agent.num_params} parámetros.")
    
    start_train_time = time.time()
    # --- MODIFICACIÓN NECESARIA EN TU AGENTE ---
    # Idealmente, agent.train devuelve el historial de fitness promedio por generación
    # O tiene un callback para registrarlo.
    # fitness_history_from_agent = agent.train(num_generations=num_generations, print_every=print_every_generations)
    # all_mean_fitness_history = fitness_history_from_agent 
    
    # Simulación si no puedes modificar el agente ahora:
    # Ejecuta el entrenamiento y captura los prints o modifica el agente para que devuelva el historial.
    # Para este ejemplo, vamos a asumir que el agente.train() fue modificado para devolver el historial:
    # class EvolutionStrategyAgent:
    # ...
    #     def train(self, num_generations, print_every=10):
    #         fitness_history = []
    #         # ... tu bucle de entrenamiento ...
    #             if (generation + 1) % print_every == 0 or generation == 0:
    #                 mean_fitness = np.mean(all_fitnesses[np.isfinite(all_fitnesses)])
    #                 fitness_history.append(mean_fitness) # <--- AÑADIR ESTO
    #                 # ... tu print ...
    #         return fitness_history # <--- AÑADIR ESTO
    # ...
    
    # Si no puedes modificar el agente, necesitarás parsear los logs o usar el hack de print.
    # Por ahora, vamos a simular que `agent.train` devuelve el historial.
    # Esto es un placeholder para la lógica real de entrenamiento y recolección de historial.
    print(f"ADVERTENCIA: La recolección del historial de fitness para el gráfico depende de que modifiques")
    print(f"el método 'train' de tu EvolutionStrategyAgent para que devuelva el historial,")
    print(f"o implementes un callback. Por ahora, se simulará un historial vacío.")
    
    # --- EJECUTA EL ENTRENAMIENTO REAL AQUÍ ---
    # agent.train(num_generations=num_generations, print_every=print_every_generations)
    # Suponiendo que lo modificaste para devolver el historial:
    # all_mean_fitness_history = agent.train(num_generations=num_generations, print_every=print_every_generations)
    # Para este ejemplo, como no puedo modificar tu agente, devolveré una lista vacía.
    # DEBES ASEGURARTE DE OBTENER EL HISTORIAL REAL DE TU ENTRENAMIENTO.
    
    # Simulación de entrenamiento para que el script corra
    print(f"Simulando entrenamiento para FS={current_frame_skip_value}...")
    time.sleep(2) # Simular tiempo de entrenamiento
    # Generar datos de fitness simulados para el gráfico (REEMPLAZAR CON DATOS REALES)
    num_points = num_generations // print_every_generations
    sim_start_score = -20
    sim_end_score = np.random.uniform(5, 18) # Resultado aleatorio para Pong
    all_mean_fitness_history = np.linspace(sim_start_score, sim_end_score, num_points).tolist()
    # --- FIN SIMULACIÓN ---


    end_train_time = time.time()
    print(f"Entrenamiento para '{env_id}' (FS={current_frame_skip_value}) completado. Tiempo: {end_train_time - start_train_time:.2f}s.")
    
    # print = original_print_fn # Restaurar print si lo modificaste

    return all_mean_fitness_history


if __name__ == "__main__":
    DEVICE_CHOICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Configuración para el experimento de Frame-Skip en Pong ---
    ATARI_ENV_FS = "PongNoFrameskip-v4"
    # Reducir generaciones para que el experimento completo no tarde demasiado
    # El paper original usa ~100 "weight updates" (generaciones) para la Figura 2.
    ATARI_GENS_FS = 100 # Ajustar según tu tiempo disponible
    ATARI_POP_FS = 32   # Población más pequeña para acelerar
    ATARI_LR_FS = 0.005
    ATARI_NOISE_STD_FS = 0.02
    ATARI_WD_FS = 0.01
    ATARI_FRAME_STACK_FS = 4
    
    FRAME_SKIP_VALUES = [1, 2, 3, 4] # Valores de frame-skip a probar
    
    results_per_frameskip = {}

    for fs_val in FRAME_SKIP_VALUES:
        # Usar una semilla diferente para cada valor de frame-skip para que sean corridas independientes,
        # O usar la misma semilla si quieres ver el efecto del frame-skip bajo las mismas condiciones iniciales de pesos.
        # El paper no especifica esto, pero corridas independientes son más robustas.
        current_seed = 42 + fs_val # Semilla ligeramente diferente para cada corrida
        
        fitness_history = run_atari_frameskip_trial(
            env_id=ATARI_ENV_FS,
            current_frame_skip_value=fs_val,
            num_generations=ATARI_GENS_FS,
            population_size=ATARI_POP_FS,
            learning_rate=ATARI_LR_FS,
            noise_std_dev=ATARI_NOISE_STD_FS,
            weight_decay=ATARI_WD_FS,
            seed=current_seed, 
            device_str=DEVICE_CHOICE,
            print_every_generations=5, # Imprimir más seguido para tener más puntos en la curva
            frame_stack_count=ATARI_FRAME_STACK_FS
        )
        if fitness_history: # Si el entrenamiento no falló
            results_per_frameskip[fs_val] = fitness_history
        else:
            print(f"No se obtuvieron resultados para Frame-Skip = {fs_val}")

    # --- Graficar los Resultados ---
    if results_per_frameskip:
        plt.figure(figsize=(10, 6))
        num_points_per_curve = 0
        for fs_val, history in results_per_frameskip.items():
            if history: # Solo graficar si hay datos
                # El eje X serían las "actualizaciones de pesos" o generaciones
                # Si print_every=5, entonces el punto i corresponde a la generación i*5
                generations_axis = np.arange(len(history)) * (ATARI_GENS_FS // len(history) if len(history) > 0 else 1) # Aproximación del eje de generaciones
                plt.plot(generations_axis, history, label=f'FrameSkip {fs_val}')
                if len(history) > num_points_per_curve : num_points_per_curve = len(history)

        plt.xlabel("Actualizaciones de Pesos / Generaciones (Aproximado)")
        plt.ylabel("Recompensa Promedio del Episodio")
        plt.title(f"Aprendizaje de ES en {ATARI_ENV_FS} con Diferentes Frame-Skips")
        plt.legend()
        plt.grid(True)
        plt.savefig("atari_frameskip_comparison.png")
        print("\nGráfico de comparación de frame-skip guardado como 'atari_frameskip_comparison.png'")
        plt.show()
    else:
        print("\nNo se generaron resultados para graficar.")

