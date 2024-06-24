# Hydraulic-System-Analysis
import autograd.numpy as np
from autograd import jacobian

# Defining the flow rate balance equations at the nodes
def node_eq1(Flow):
    return Flow[0] + Flow[6] - 2.5

def node_eq2(Flow):
    return Flow[1] + Flow[4] - Flow[0]

def node_eq3(Flow):
    return -Flow[1] + Flow[2] + 0.5

def node_eq4(Flow):
    return -Flow[2] - Flow[3] + 1

def node_eq5(Flow):
    return Flow[3] - Flow[5] - Flow[4] + 1

# Defining the head loss equations in the loops
def loop_eq1(Flow):
    return 10153.188 * (Flow[0]**2) + 330507.429 * (Flow[4]**2) - 130570.836 * (Flow[5]**2) - 10328.357 * (Flow[6]**2)

def loop_eq2(Flow):
    return 130570.836 * (Flow[1]**2) + 43523.612 * (Flow[2]**2) - 130570.836 * (Flow[3]**2) - 330507.429 * (Flow[4]**2)

# Calculating Jacobians for the above equations
jacobian_node_eq1 = jacobian(node_eq1)
jacobian_node_eq2 = jacobian(node_eq2)
jacobian_node_eq3 = jacobian(node_eq3)
jacobian_node_eq4 = jacobian(node_eq4)
jacobian_node_eq5 = jacobian(node_eq5)
jacobian_loop_eq1 = jacobian(loop_eq1)
jacobian_loop_eq2 = jacobian(loop_eq2)

# Initial settings
tolerance = 0.001
max_iterations = 400
iteration_count = 0
error = np.inf

# Number of equations and unknowns
num_eqs = 7
num_vars = 7

# Initial guess for the flow rates
flow_guess = np.ones((num_vars, 1), dtype=float)

# Newton-Raphson iteration method
while np.any(abs(error) > tolerance) and iteration_count < max_iterations:
    # Evaluate the equations with the current guess
    evaluated_functions = np.array([
        node_eq1(flow_guess), node_eq2(flow_guess), node_eq3(flow_guess),
        node_eq4(flow_guess), node_eq5(flow_guess), loop_eq1(flow_guess), loop_eq2(flow_guess)
    ]).reshape(num_eqs, 1)
    
    # Flatten the initial guess for Jacobian computation
    flat_flow_guess = flow_guess.flatten()
    
    # Calculate the Jacobian matrix
    jacobian_matrix = np.array([
        jacobian_node_eq1(flat_flow_guess), jacobian_node_eq2(flat_flow_guess), jacobian_node_eq3(flat_flow_guess),
        jacobian_node_eq4(flat_flow_guess), jacobian_node_eq5(flat_flow_guess), jacobian_loop_eq1(flat_flow_guess),
        jacobian_loop_eq2(flat_flow_guess)
    ]).reshape(num_eqs, num_vars)
    
    # Determine the change in the guess using the Newton-Raphson formula
    delta_flow_guess = np.linalg.solve(jacobian_matrix, evaluated_functions)
    new_flow_guess = flow_guess - delta_flow_guess

    # Update the error and the guess
    error = new_flow_guess - flow_guess
    flow_guess = new_flow_guess

    # Print the current iteration and error
    print(f"Iteration {iteration_count}")
    print(f"Error: {error.flatten()}")
    print("--------------------------")

    iteration_count += 1

# Display the solution to the equations
print("Solution to the equations:")
for i, flow_rate in enumerate(new_flow_guess.flatten()):
    print(f"Flow_{chr(ord('A') + i)}: {flow_rate} m3/s")

# Parameters for calculating head loss
pipe_lengths = np.array([600, 600, 200, 600, 600, 200, 200])
pipe_diameters = np.array([0.25, 0.15, 0.10, 0.15, 0.15, 0.20, 0.20])
node_elevations = np.array([30, 25, 20, 20, 22, 25])
friction_factor = 0.02
initial_head_A = 15

# Calculating the head at each node
print(f"Head at node A is {initial_head_A} m")
current_head = initial_head_A
for i in range(5):
    head_loss = (8 * friction_factor * pipe_lengths[i] * (new_flow_guess[i][0])**2) / (9.81 * pipe_diameters[i]**5 * np.pi**2)
    if node_elevations[i] > node_elevations[i + 1]:
        current_head -= head_loss
    else:
        current_head += head_loss
    print(f"Head at node {chr(ord('A') + i + 1)} is {current_head} m")
