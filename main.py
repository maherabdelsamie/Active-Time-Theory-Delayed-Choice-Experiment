import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
C = 299792458  # Speed of light in m/s
h = 6.62607015e-34  # Planck's constant in m^2 kg / s
delta_t = 5  # Time delay for the GlobalTime calculations

class GlobalTime:
    def __init__(self, use_ath=True, particles=[], cesium_atoms=[], quantum_systems=[]):
        self.phi_history = [0] * delta_t
        self.current_time = 0
        self.time_flow_rate = 1
        self.dt = 0.01
        self.time_flow_rates = []
        self.intrinsic_times = []
        self.use_ath = use_ath
        self.particles = particles
        self.cesium_atoms = cesium_atoms
        self.quantum_systems = quantum_systems

    def calculate_phi_derivative(self):
        if not self.use_ath:
            return 0
        # Stochastic generative influence on time
        phi_delayed = self.phi_history[-delta_t]
        s_t = np.random.normal(0, 1)
        
        # Influence based on particle states (adaptive faculty)
        states = np.array([p.state for p in self.particles])
        mean_state = np.mean(states, axis=0)
        variance = np.var(states, axis=0)

        # Combine stochastic and system-influenced effects on time
        phi_prime = s_t + 0.1 * np.linalg.norm(mean_state) + 0.05 * np.linalg.norm(variance)
        return phi_prime

    def update_phi(self, energy_density):
        # Improved time evolution with more dynamics and stochastic influence
        def phi_derivative(phi, energy_density):
            return np.tanh(energy_density) - 0.1 * phi + self.calculate_phi_derivative()

        # Use a 4th-order Runge-Kutta integration for more accurate time evolution
        k1 = self.dt * phi_derivative(self.phi_history[-1], energy_density)
        k2 = self.dt * phi_derivative(self.phi_history[-1] + 0.5 * k1, energy_density)
        k3 = self.dt * phi_derivative(self.phi_history[-1] + 0.5 * k2, energy_density)
        k4 = self.dt * phi_derivative(self.phi_history[-1] + k3, energy_density)
        phi_update = (k1 + 2*k2 + 2*k3 + k4) / 6

        new_phi = self.phi_history[-1] + phi_update
        self.phi_history.append(new_phi)
        if len(self.phi_history) > delta_t:
            self.phi_history.pop(0)

    def update_time_flow(self):
        # Time flow rate is influenced by phi, representing time's directive and adaptive faculties
        current_phi = self.phi_history[-1]
        self.time_flow_rate = 1 + 0.05 * np.tanh(current_phi) + np.random.uniform(-0.02, 0.02)  # Adding stochastic variability
        self.time_flow_rates.append(self.time_flow_rate)

    def update_current_time(self):
        if self.use_ath:
            self.current_time += self.time_flow_rate * self.dt
        else:
            self.current_time += self.dt
        self.intrinsic_times.append(self.current_time)

    def calculate_energy_density(self):
        total_energy = sum(0.5 * p.mass * np.linalg.norm(p.velocity)**2 + p.potential_energy for p in self.particles)
        energy_exchange = sum(p.energy_exchange for p in self.particles)
        total_energy += energy_exchange
        volume = 1
        energy_density = total_energy / volume

        # Update energy exchange for each particle
        for particle in self.particles:
            particle.energy_exchange *= 0.9  # Decay factor
            particle.energy_exchange += np.random.uniform(-0.05, 0.05) * self.phi_history[-1]

        for atom in self.cesium_atoms:
            atom.calculate_transitions(self, energy_density)

        return energy_density

    def update_all(self):
        energy_density = self.calculate_energy_density()
        self.update_phi(energy_density)
        self.update_time_flow()
        self.update_current_time()

        for particle in self.particles:
            particle.update_state(self.dt, self)
            particle.calculate_dilated_time(self.dt, self)

        for system in self.quantum_systems:
            system.evolve(self)


class QuantumParticle:
    def __init__(self, position, velocity, mass=1.0):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.mass = mass
        self.dilated_times = []
        self.potential_energy = 0
        self.energy_exchange = 0
        self.state = np.random.rand(3)  # Random initial state

    def update_state(self, dt, global_time):
        temporal_aperture = self.calculate_temporal_aperture(global_time)
        lorentz_factor_ath = 1 / np.sqrt(1 - np.linalg.norm(self.velocity)**2 / C**2) * (1 + global_time.phi_history[-1] * temporal_aperture)
        effective_dt = dt * lorentz_factor_ath
        self.position += self.velocity * effective_dt

        # Update energy exchange
        self.energy_exchange = np.random.uniform(-0.1, 0.1) * global_time.phi_history[-1]

        if isinstance(effective_dt, (int, float)):
            self.dilated_times.append(effective_dt)
        else:
            print(f"Non-numerical value detected for Particle's effective_dt: {effective_dt}")

    def calculate_temporal_aperture(self, global_time):
        local_phi_influence = self.calculate_local_phi_influence(global_time)
        temporal_aperture = np.exp(-local_phi_influence)
        
        if not isinstance(temporal_aperture, (int, float)):
            print(f"Non-numerical temporal_aperture detected: {temporal_aperture}")
            return 1
        return temporal_aperture

    def calculate_local_phi_influence(self, global_time):
        influence = np.zeros_like(self.velocity)
        for other_particle in global_time.particles:
            if other_particle is not self:
                direction = other_particle.position - self.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    influence += (direction / distance) * (1 / distance**2)
        average_influence = influence / len(global_time.particles) if global_time.particles else 0
        
        if not isinstance(average_influence, (int, float, np.ndarray)):
            print(f"Non-numerical average_influence detected: {average_influence}")
            return 0
        return np.linalg.norm(average_influence)

    def calculate_dilated_time(self, dt, global_time):
        temporal_aperture = self.calculate_temporal_aperture(global_time)
        lorentz_factor = 1 / np.sqrt(1 - np.linalg.norm(self.velocity)**2 / C**2)
        lorentz_factor_ath = lorentz_factor * (1 + global_time.phi_history[-1] * temporal_aperture) if global_time.use_ath else lorentz_factor
        dilated_time = dt * lorentz_factor_ath
        self.dilated_times.append(dilated_time)


class CesiumAtom:
    def __init__(self, initial_state):
        self.state = initial_state
        self.base_energy_levels = [
            0.0, 1.3859, 1.4546, 2.2981, 2.6986, 2.7210, 1.7977, 1.8098, 2.8007, 2.8060
        ]
        self.transition_frequencies = []

    def calculate_transitions(self, global_time, energy_density):
        phi_current = global_time.phi_history[-1]
        initial_state_energy = self.base_energy_levels[self.state]
        transition_energy = initial_state_energy * (1 + phi_current * energy_density)
        transition_frequency = transition_energy / h
        self.transition_frequencies.append(transition_frequency)


class QuantumSystem:
    def __init__(self, initial_state):
        self.state = initial_state
        self.measured_state = None
        self.measurement_time = None

    def evolve(self, global_time):
        phi = global_time.phi_history[-1]
        self.state = self.state * np.exp(1j * phi)

    def measure(self, global_time, measurement_type):
        self.measured_state = np.random.choice(['path_A', 'path_B'])
        self.measurement_time = global_time.current_time


class DelayedChoiceExperiment:
    def __init__(self, global_time):
        self.global_time = global_time
        self.quantum_system = QuantumSystem(initial_state=1)
        self.measurement_time = None
        self.measurement_type = None
        self.global_time.quantum_systems.append(self.quantum_system)

    def run(self, steps, measurement_step, measurement_type):
        for step in range(steps):
            self.global_time.update_all()
            if step == measurement_step:
                self.measurement_time = self.global_time.current_time
                self.measurement_type = measurement_type
                self.quantum_system.measure(self.global_time, self.measurement_type)

    def analyze_results(self):
        if self.measurement_time is None:
            return {
                'measured_state': None,
                'measurement_time': None,
                'phi_at_measurement': None
            }
        
        measurement_index = min(int(self.measurement_time / self.global_time.dt), len(self.global_time.phi_history) - 1)
        phi_at_measurement = self.global_time.phi_history[measurement_index]
        
        return {
            'measured_state': self.quantum_system.measured_state,
            'measurement_time': self.measurement_time,
            'phi_at_measurement': phi_at_measurement
        }

def plot_results(global_time, experiment_results):
    # Plot phi evolution and measurement time
    plt.figure(figsize=(10, 6))
    plt.plot(global_time.phi_history)
    plt.axvline(x=experiment_results['measurement_time'] / global_time.dt, color='r', linestyle='--', label='Measurement Time')
    plt.title('Phi Evolution and Measurement Time')
    plt.xlabel('Simulation Step')
    plt.ylabel('Phi')
    plt.legend()
    plt.show()

    # Plot time flow rates
    plt.figure(figsize=(10, 6))
    plt.plot(global_time.time_flow_rates)
    plt.title('Time Flow Rates Over Simulation')
    plt.xlabel('Simulation Step')
    plt.ylabel('Time Flow Rate')
    plt.show()

    # Plot intrinsic time
    plt.figure(figsize=(10, 6))
    plt.plot(global_time.intrinsic_times)
    plt.title('Intrinsic Time Over Simulation Steps')
    plt.xlabel('Simulation Step')
    plt.ylabel('Intrinsic Time')
    plt.show()

# Main simulation
if __name__ == "__main__":
    num_nodes = 100
    num_steps = 2000
    measurement_step = 1500

    particles = [QuantumParticle([1.0, 0.0, 0.0], np.random.uniform(-0.5, 0.5, 3) * C, mass=1.0) for _ in range(num_nodes)]
    cesium_atoms = [CesiumAtom(initial_state=np.random.randint(0, 4)) for _ in range(10)]
    global_time = GlobalTime(use_ath=True, particles=particles, cesium_atoms=cesium_atoms)

    experiment = DelayedChoiceExperiment(global_time)
    experiment.run(steps=num_steps, measurement_step=measurement_step, measurement_type='which_path')
    results = experiment.analyze_results()

    print("Delayed Choice Experiment Results:")
    print(f"Measured State: {results['measured_state']}")
    print(f"Measurement Time: {results['measurement_time']}")
    print(f"Phi at Measurement: {results['phi_at_measurement']}")

    # Plot the results
    plot_results(global_time, results)
