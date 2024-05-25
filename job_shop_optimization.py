import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time

class JobShop:
    def __init__(self, jobs):
        self.jobs = jobs 
        self.num_jobs = len(jobs)
        self.num_machines = len(set(machine for job in jobs for machine, _ in job))
        
    def evaluate(self, schedule):
        machine_times = [0] * self.num_machines
        job_times = [0] * self.num_jobs
        
        for job_index, operations in schedule:
            for machine, time in self.jobs[job_index]:
                start_time = max(machine_times[machine], job_times[job_index])
                machine_times[machine] = start_time + time
                job_times[job_index] = start_time + time
                
        return max(job_times)
    
    def fitness(self, schedule):
        makespan = self.evaluate(schedule)
        return -makespan
    
    def generate_initial_solution(self):
        priorities = sorted(range(self.num_jobs), key=lambda x: sum(time for _, time in self.jobs[x]), reverse=True)
        return [(i, self.jobs[i]) for i in priorities]
    
    def get_neighbors(self, schedule):
        neighbors = []
        for i in range(len(schedule)):
            for j in range(i + 1, len(schedule)):
                neighbor = schedule[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors
    
    def hill_climbing(self):
        current_solution = self.generate_initial_solution()
        current_value = self.fitness(current_solution)
        
        while True:
            neighbors = self.get_neighbors(current_solution)
            neighbor_values = [(self.fitness(neighbor), neighbor) for neighbor in neighbors]
            best_neighbor_value, best_neighbor = min(neighbor_values, key=lambda x: x[0])
            
            if best_neighbor_value < current_value:
                current_solution, current_value = best_neighbor, best_neighbor_value
            else:
                break
        
        return current_solution, -current_value
    
    def simulated_annealing(self, initial_temperature, cooling_rate):
        current_solution = self.generate_initial_solution()
        current_value = self.fitness(current_solution)
        temperature = initial_temperature
        
        while temperature > 1:
            neighbors = self.get_neighbors(current_solution)
            next_solution = random.choice(neighbors)
            next_value = self.fitness(next_solution)
            
            delta = next_value - current_value
            
            if delta < 0 or random.uniform(0, 1) < math.exp(-delta / temperature):
                current_solution, current_value = next_solution, next_value
            
            temperature *= cooling_rate
        
        return current_solution, -current_value

    def visualize_schedule(self, schedule, title):
        fig, gnt = plt.subplots(figsize=(15, 10))
        gnt.set_title(title)
        gnt.set_xlabel('Time')
        gnt.set_ylabel('Machine')

        y_ticks = [10 * (i + 1) - 5 for i in range(self.num_machines)]
        y_tick_labels = [f'Machine {i}' for i in range(self.num_machines)]
        
        gnt.set_yticks(y_ticks)
        gnt.set_yticklabels(y_tick_labels)
        
        colors = plt.cm.get_cmap('tab10', self.num_jobs)
        job_labels = [f'Job {i}' for i in range(self.num_jobs)]

        machine_start_times = {i: 0 for i in range(self.num_machines)}

        for job_index, operations in schedule:
            job_start_time = 0
            for machine, time in operations:
                start_time = max(machine_start_times[machine], job_start_time)
                gnt.broken_barh(
                    [(start_time, time)], 
                    (10 * machine, 9), 
                    facecolors=(colors(job_index)),
                    edgecolor='black'
                )
                gnt.text(start_time + time / 2, 10 * machine + 4.5, f'{job_index}', 
                        ha='center', va='center', color='white', fontsize=12)
                machine_start_times[machine] = start_time + time
                job_start_time = start_time + time

        handles = [plt.Line2D([0], [0], color=colors(i), lw=4) for i in range(self.num_jobs)]
        gnt.legend(handles, job_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.tight_layout()
        plt.show()


##############  caso 1 
jobs = [
    [(0, 3), (1, 2), (2, 2), (3, 1)],
    [(0, 2), (2, 1), (1, 4), (3, 3)],
    [(1, 4), (2, 3), (0, 2), (3, 5)],
    [(2, 2), (3, 1), (0, 4), (1, 3)],
    [(0, 3), (2, 4), (3, 2), (1, 1)],
    [(1, 2), (0, 3), (3, 4), (2, 1)],
    [(2, 3), (3, 2), (0, 1), (1, 4)],
    [(0, 4), (1, 3), (2, 2), (3, 1)],
    [(1, 2), (0, 1), (3, 4), (2, 3)],
    [(2, 1), (3, 4), (0, 3), (1, 2)]
]

job_shop = JobShop(jobs)

# Hill Climbing
start_time = time.time()
hill_climbing_solution, hill_climbing_value = job_shop.hill_climbing()
end_time = time.time()
hill_climbing_duration = end_time - start_time
print("Solución Hill Climbing :", hill_climbing_solution)
print("Hill Climbing Makespan:", hill_climbing_value)
print("Hill Climbing Tiempo de Ejecución:", hill_climbing_duration, "segundos")
job_shop.visualize_schedule(hill_climbing_solution, "Hill Climbing Schedule")

# Simulated Annealing
initial_temperature = 1000
cooling_rate = 0.95
start_time = time.time()
sa_solution, sa_value = job_shop.simulated_annealing(initial_temperature, cooling_rate)
end_time = time.time()
sa_duration = end_time - start_time
print("Solución Simulated Annealing:", sa_solution)
print("Simulated Annealing Makespan:", sa_value)
print("Simulated Annealing Tiempo de Ejecución:", sa_duration, "segundos")
job_shop.visualize_schedule(sa_solution, "Simulated Annealing Schedule")

##############  caso 2 
# Generación de la matriz de trabajos 60 x 60
num_jobs = 60
num_machines = 60
jobs = [
    [(machine, random.randint(1, 10)) for machine in range(num_machines)]
    for _ in range(num_jobs)
]
job_shop = JobShop(jobs)
#Hill Climbing
start_time = time.time()
hc_solution, hc_makespan = job_shop.hill_climbing()
hc_duration = time.time() - start_time
print(f"Hill Climbing Solution: {hc_solution[:5]}...")  
print(f"Hill Climbing Makespan: {hc_makespan}")
print(f"Hill Climbing Duration: {hc_duration} seconds")

#Simulated Annealing
initial_temperature = 1000
cooling_rate = 0.95
start_time = time.time()
sa_solution, sa_makespan = job_shop.simulated_annealing(initial_temperature, cooling_rate)
sa_duration = time.time() - start_time
print(f"Simulated Annealing Solution: {sa_solution[:5]}...")  # Mostrar solo los primeros 5 trabajos para brevedad
print(f"Simulated Annealing Makespan: {sa_makespan}")
print(f"Simulated Annealing Duration: {sa_duration} seconds")


##############  caso 3 
# Generación de la matriz de trabajos 100 x 100
num_jobs = 100
num_machines = 100
jobs = [
    [(machine, random.randint(1, 10)) for machine in range(num_machines)]
    for _ in range(num_jobs)
]
job_shop = JobShop(jobs)
#Hill Climbing
start_time = time.time()
hc_solution, hc_makespan = job_shop.hill_climbing()
hc_duration = time.time() - start_time
print(f"Hill Climbing Solution: {hc_solution[:5]}...")  
print(f"Hill Climbing Makespan: {hc_makespan}")
print(f"Hill Climbing Duration: {hc_duration} seconds")

#Simulated Annealing
initial_temperature = 1000
cooling_rate = 0.95
start_time = time.time()
sa_solution, sa_makespan = job_shop.simulated_annealing(initial_temperature, cooling_rate)
sa_duration = time.time() - start_time
print(f"Simulated Annealing Solution: {sa_solution[:5]}...")  # Mostrar solo los primeros 5 trabajos para brevedad
print(f"Simulated Annealing Makespan: {sa_makespan}")
print(f"Simulated Annealing Duration: {sa_duration} seconds")
