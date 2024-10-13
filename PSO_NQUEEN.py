import random
import time

# Função de fitness para calcular os conflitos
def fitness(solution):
    n = len(solution)
    conflicts = 0

    # Verificar conflitos de linha (rainhas na mesma linha)
    conflicts += n - len(set(solution))  # Penaliza se houver rainhas na mesma linha

    # Verificar conflitos nas diagonais
    for i in range(n):
        for j in range(i + 1, n):
            # Se estiver na mesma diagonal
            if abs(solution[i] - solution[j]) == abs(i - j):
                conflicts += 1

    return conflicts

# Inicializar as partículas (soluções)
def initialize_particles(n, num_particles):
    particles = []
    for _ in range(num_particles):
        # Cada partícula é uma solução aleatória para as N-Rainhas
        particle = list(range(1, n + 1))
        random.shuffle(particle)
        particles.append(particle)
    return particles

# Inicializar velocidades das partículas (pode começar com todas zero)
def initialize_velocities(n, num_particles):
    velocities = []
    for _ in range(num_particles):
        # Velocidade pode ser um vetor de n elementos, começando em 0
        velocity = [0] * n
        velocities.append(velocity)
    return velocities

# Atualizar a velocidade e a posição da partícula
def update_velocity_position(particle, velocity, pbest, gbest, W, C1, C2):
    n = len(particle)
    for i in range(n):
        r1 = random.random()  # Número aleatório entre 0 e 1
        r2 = random.random()

        # Atualizar velocidade
        velocity[i] = W * velocity[i] + C1 * r1 * (pbest[i] - particle[i]) + C2 * r2 * (gbest[i] - particle[i])
        
        # Limitar a velocidade para não sair do tabuleiro
        velocity[i] = max(min(velocity[i], n-1), -(n-1))

        # Atualizar a posição da partícula somando a velocidade
        particle[i] += int(velocity[i])

        # Manter a rainha dentro dos limites do tabuleiro (1 a n)
        if particle[i] < 1:
            particle[i] = 1
        elif particle[i] > n:
            particle[i] = n

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
            # Obter o melhor global localmente
            gbest_local = get_local_gbest(particles, fitness_scores, i, neighborhood_size)

            # Atualizar a partícula e sua velocidade
            particles[i], velocities[i] = update_velocity_position(particles[i], velocities[i], pbest[i], gbest_local, W, C1, C2)
            
            # Atualizar o fitness da partícula
            current_fitness = fitness(particles[i])

            # Atualizar o Pbest da partícula
            if current_fitness < pbest_fitness[i]:
                pbest[i] = particles[i][:]
                pbest_fitness[i] = current_fitness
            
            # Atualizar o fitness atual
            fitness_scores[i] = current_fitness
        
        # Exibir o tempo e o melhor fitness a cada 100 iterações
        if (iteration + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            best_fitness = min(fitness_scores)
            print(f"Iteração {iteration + 1}: Melhor fitness = {best_fitness}, Tempo decorrido = {elapsed_time:.2f} segundos")
        
        # Verificar se alguma partícula já tem fitness zero (solução sem conflitos)
        if min(fitness_scores) == 0:
            break

    # Melhor solução global encontrada
    best_index = fitness_scores.index(min(fitness_scores))
    return particles[best_index], fitness_scores[best_index]

# Testar o PSO com topologia em anel para o problema de 20 Rainhas
n = 8
num_particles = 10
max_iterations = 1000
neighborhood_size = 2  # Define o tamanho da vizinhança

best_solution, best_fitness = pso(n, num_particles, max_iterations, neighborhood_size)
print("Melhor solução encontrada:", best_solution)
print("Fitness da melhor solução:", best_fitness)
