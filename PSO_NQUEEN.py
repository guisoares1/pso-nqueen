import random
import time

from matplotlib import pyplot as plt
import numpy as np

# Função de fitness para calcular os conflitos
def fitness(solution):
    n = len(solution)
    conflicts = 0

    # Verificar conflitos de linha (rainhas na mesma linha)
    conflicts += n - len(set(solution)) 

    for i in range(n):
        for j in range(i + 1, n):
            if abs(solution[i] - solution[j]) == abs(i - j):
                conflicts += 1

    return conflicts

def initialize_particles(n, num_particles):
    particles = []
    for _ in range(num_particles):
        particle = list(range(1, n + 1))
        random.shuffle(particle)
        particles.append(particle)
    print(particles)
    return particles

# Inicializar velocidades das partículas (pode começar com todas zero)
def initialize_velocities(n, num_particles):
    velocities = []
    for _ in range(num_particles):
        # Velocidade pode ser um vetor de n elementos, começando em 0
        velocity = [0] * n
        velocities.append(velocity)
    return velocities

def update_velocity_position(particle, velocity, pbest, gbest, W, C1, C2):
    n = len(particle)

    for i in range(n):
        r1 = random.random()
        r2 = random.random()

        # Atualizar a velocidade
        velocity[i] = W * velocity[i] + C1 * r1 * (pbest[i] - particle[i]) + C2 * r2 * (gbest[i] - particle[i])

    # Atualizar a posição da partícula usando a velocidade, sem limitar
    for i in range(n):
        new_pos = (particle[i] + int(velocity[i])) % n  # Garantir que fique no limite do tabuleiro
        # Trocar as rainhas de posição
        particle[i], particle[new_pos] = particle[new_pos], particle[i]

    return particle, velocity


# Selecionar o melhor global localmente (vizinho mais próximo)
def get_local_gbest(particles, fitness_scores, i, neighborhood_size):
    num_particles = len(particles)
    neighbors = []
    
    # Definir os vizinhos considerando uma topologia em anel
    for j in range(i - neighborhood_size, i + neighborhood_size + 1):
        neighbor_index = j % num_particles
        neighbors.append((particles[neighbor_index], fitness_scores[neighbor_index]))
    
    # Retornar a partícula com o menor fitness entre os vizinhos
    best_neighbor = min(neighbors, key=lambda x: x[1])
    return best_neighbor[0]

def pso(n, num_particles, max_iterations, neighborhood_size):
    W = 0.5  # Inércia
    C1 = 1.5  # Coeficiente cognitivo (individual)
    C2 = 2.0  # Coeficiente social (global)

    # Inicializar partículas e velocidades
    particles = initialize_particles(n, num_particles)
    velocities = initialize_velocities(n, num_particles)

    # Inicializar Pbest e os valores de fitness correspondentes
    pbest = particles[:]  # Melhor posição individual
    pbest_fitness = [fitness(p) for p in pbest]
    
    # Inicializar o fitness global (gbest local)
    fitness_scores = [fitness(p) for p in particles]

    # Iniciar contagem de tempo
    start_time = time.time()

    # Loop até atingir o critério de parada
    for iteration in range(max_iterations):
        for i in range(num_particles):
            
            gbest_local = get_local_gbest(particles, fitness_scores, i, neighborhood_size)

            particles[i], velocities[i] = update_velocity_position(particles[i], velocities[i], pbest[i], gbest_local, W, C1, C2)
            
            current_fitness = fitness(particles[i])
            
            if current_fitness < pbest_fitness[i]:
                pbest[i] = particles[i][:]
                pbest_fitness[i] = current_fitness

            fitness_scores[i] = current_fitness
        
        # Exibir o tempo e o melhor fitness a cada 100 iterações
        if (iteration + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            best_fitness = min(fitness_scores)
            print(f"Iteração {iteration + 1}: Melhor fitness = {best_fitness}, Tempo decorrido = {elapsed_time:.2f} segundos")
        
        if min(fitness_scores) == 0:
            break

    # Melhor solução global encontrada
    best_index = fitness_scores.index(min(fitness_scores))
    return particles[best_index], fitness_scores[best_index], iteration


def plot_board_with_matplotlib(n, solution):
    board = np.zeros((n, n))
    board[::2, ::2] = 1  # Quadrados brancos nas posições pares
    board[1::2, 1::2] = 1  # Quadrados brancos nas posições ímpares

    # Configura o gráfico
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(board, cmap='gray', interpolation='nearest')

    for row, col in enumerate(solution):
        # Ajusta para índices baseados em 0
        x = col - 1
        y = row
        # Desenha um círculo vermelho representando a rainha
        circle = plt.Circle((x, y), 0.4, color='red', fill=True)
        ax.add_patch(circle)

    # Ajusta os limites e o aspecto
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Inverte o eixo y para que a posição [0,0] fique no canto inferior esquerdo

    plt.show()


n = 16
num_particles = 10
max_iterations = 100000
neighborhood_size = 3  # Define o tamanho da vizinhança

best_solution, best_fitness, iteracoes = pso(n, num_particles, max_iterations, neighborhood_size)
print("Melhor solução encontrada:", best_solution)
print("Fitness da melhor solução:", best_fitness)
print("Iteração numero:", iteracoes)
#plot_board_with_matplotlib(n, best_solution)
