import numpy as np

# generates randomnumber everytime it runs
print(np.random.randint(1, 10))
print(np.random.randint(1, 10))

# Reset your random number generator to a known starting point every time it resets to sam enumber it will hgenrate same number
np.random.seed(42)
print(np.random.randint(1, 10))
np.random.seed(42)
print(np.random.randint(1, 10))

X = np.array([
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [6, 6]
])

X_train = X[:4]
X_test = X[4:]

print("X_train:")
print(X_train)

print("X_test:")
print(X_test)

first_point = X[:1]
second_point = X[1:2]
third_point = X[2:3]
selected_point = np.array([3, 4])
print("first_point:", first_point)
print("second_point:", second_point)
print("third_point:", third_point)

sum_square_of_difference = np.sum((second_point - first_point)**2)
print("Value of  difference of squares of sum =" , sum_square_of_difference)
print("Value of squareroot of sum=" , np.sqrt(sum_square_of_difference))

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

distances = [euclidean_distance(selected_point, first_point), euclidean_distance(selected_point, second_point), euclidean_distance(selected_point, third_point)]
print("distances:", distances)
# does not return sorted indices. It returns the indexes that would put the distances in ascending order.
indices = np.argsort(distances)
print("indices:", indices)