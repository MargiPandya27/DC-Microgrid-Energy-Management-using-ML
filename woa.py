import numpy as np
from sklearn.svm import SVR
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Iris dataset
#iris = load_iris()
#X, y = iris.data, iris.target

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#C=1.0
#epsilon = 0.2
#gamma = 'scale'
#
#svr = SVR(C=C, epsilon=epsilon, gamma=gamma)
#    
#    # Train the SVR model
#svr.fit(x_train, y_train)
#    
#    # Predict on the testing set
#y_pred = svr.predict(x_test)
#    
#    # Calculate the mean squared error as the fitness value
#mse = mean_squared_error(y_test, y_pred)
#
#print(f'Mean Squared Error: {mse}')

a = 0.5

# Define the objective function for SVR optimization
def objective_function(position):
    # Extract the hyperparameters from the position list
    C = 10 ** position[0]
    epsilon = 10 ** position[1]
    gamma = 10 ** position[2]
    
    # Create an SVR model with the hyperparameters
    svr = SVR(C=C, epsilon=epsilon, gamma=gamma)
    
    # Train the SVR model
    svr.fit(x_train, y_train)
    
    # Predict on the testing set
    y_pred = svr.predict(x_test)
    
    # Calculate the mean squared error as the fitness value
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

# Define the bounds for each hyperparameter
bounds = [(-3, 3), (-3, 3), (-3, 3)]  # For C, epsilon, gamma
num_dimensions = len(bounds)

# Define the WOA optimization function
def woa_optimization(objective_function, bounds, max_iter, num_whales):
    # Initialize the population of whales
    low=[b[0] for b in bounds]
    high=[b[1] for b in bounds]
    population = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(num_whales, len(bounds)))


    # Initialize the best solution and best fitness
    best_solution = None
    best_fitness = float('inf')
    
    # Iterate through each generation
    for iteration in range(max_iter):
        a = 2 - i*(2/max_iter)
        c = 2*np.exp(-4*i/max_iter)
        for i in range(num_whales):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2*a*r1 - a
            C = 2*r2
            b = 1
            l = (-1)**np.random.randint(1,3)
            p = np.random.rand()
            position = population[i]
            # Update the fitness value of the current whale
            current_fitness = objective_function(position)

            # Update the best solution if a better solution is found
            if current_fitness < best_fitness:
                best_solution = position
                best_fitness = current_fitness

            # Update the position of the current whale based on its distance to the best solution
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)
            A = 2 * a * r1 - a
            C = 2 * r2
            D = np.abs(C * best_solution - position)
            updated_whale_position = np.zeros(num_dimensions)

            for j in range(num_dimensions):
                if C[j] < 1:
                    updated_whale_position[j] = (
                        best_solution[j] - A[j] * D[j]
                    )  # Spiral updating mechanism
                else:
                    if D[j] < 0.5 * (high[j] - low[j]):
                        updated_whale_position[j] = (
                            position[j] + b * D[j]
                        )  # Encircling mechanism
                    else:
                        updated_whale_position[j] = (
                            position[j] - b * D[j]
                        )  # Encircling mechanism

                # Boundary handling
                lower_bound, upper_bound = bounds[j]
                if updated_whale_position[j] < lower_bound:
                    updated_whale_position[j] = lower_bound
                if updated_whale_position[j] > upper_bound:
                    updated_whale_position[j] = upper_bound

        # Update the position of the current whale
        position = updated_whale_position

    return best_solution, best_fitness

# Set the parameters for WOA optimization
max_iter = 100  # Number of iterations for WOA
num_whales = 10  # Number of whales in the population

# Call the WOA optimization function
best_solution, best_fitness = woa_optimization(objective_function, bounds, max_iter, num_whales)

# Extract the hyperparameters from the best solution
C = 10 ** best_solution[0]
epsilon = 10 ** best_solution[1]
gamma = 10 ** best_solution[2]

svr = SVR(C=C, epsilon=epsilon, gamma=gamma)
    
# Train the SVR model
svr.fit(x_train, y_train)
    
# Predict on the testing set
y_pred = svr.predict(x_test)
    
# Calculate the mean squared error as the fitness value
mse = mean_squared_error(y_test, y_pred)

print(f'Optimized hyperparameters (C, eplisom, gamma): {best_solution}')
print(f'Optimized Mean Squared Error: {mse}')