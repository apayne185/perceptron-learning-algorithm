import numpy as np
import matplotlib.pyplot as plt


def random_line():
  points = np.random.rand(2,2)
  slope = (points[1,1]-points[0,1]) / (points[1,0]-points[0,0])
  intercept = points[0,1] - slope * points[0,0]
  return slope, intercept


def dataset(size, slope, intercept):
  X = np.random.rand(size, 2) * 2 - 1    #points in [-1,1]  -->features
  y = np.sign(X[:, 1] - X[:, 0] + intercept)     #1 or -1  -->labels
  return X,y


def pla(X, y):
  # n_sample, n_feature = X.shape     # n_feature = x.shape[1]
  weight = np.zeros(X.shape[1])
  bias = 0
  updates = 0

  while True:
    misclassified = 0
    for i in range(len(y)):
      y_pred = np.sign(np.dot(weight, X[i])+bias)    #algoritm from slide 16
      if y_pred != y[i]:
        weight += y[i] * X[i]
        bias += y[i]
        updates +=1
        misclassified += 1
    if misclassified == 0:
      break       #converged
  return weight, bias, updates


# def convergence():

def plot(X, y, slope, intercept, weight, bias):
  plt.figure(figsize=(8,8))
  for i in range(len(y)):
    if y[i] ==1:
      plt.scatter(X[i,0], X[i,1], color='blue', label='+1' if i==0 else "")
    else:
      plt.scatter(X[i,0], X[i,1], color='red', label='-1' if i==0 else "")


  x_vals = np.linspace(-1,1,100)
  line = slope* x_vals +intercept
  percept_line = -(weight[0] / weight[1]) * x_vals - (bias / weight[1])
  plt.plot(x_vals, line, 'g--', label = 'Target Function')
  plt.plot(x_vals, percept_line, 'b-', label='Perceptron Hypothesis')

  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.xlim(-1, 1)
  plt.ylim(-1, 1)
  plt.legend()
  plt.grid()
  plt.title('PLA')
  plt.show()



if __name__ == "__main__":
  slope, intercept = random_line()
  X, y = dataset(size = 100, slope=slope, intercept=intercept)
  weight,bias,updates = pla(X,y)

  plot(X, y, slope, intercept, weight, bias)
  print("Updates before convergence: ", updates, "\n\n")


  large_slope, large_intercept = random_line()
  large_X, large_y = dataset(size = 1000, slope=large_slope, intercept=large_intercept)
  large_weight, large_bias, large_updates = pla(large_X, large_y)

  plot(large_X, large_y, large_slope, large_intercept, large_weight, large_bias)
  print("Updates before convergence: ", large_updates)




def r_ten(dim):
  weight= np.random.randn(dim)
  bias = np.random.randn()
  return weight, bias

def dataset_r10(size, weights, bias):
  X = np.random.uniform(-1,1, (size, len(weights)))
  y = np.sign(np.dot(X, weights)+ bias)
  return X,y


def pla_r10(X, y):
    weights = np.zeros(X.shape[1])
    bias = 0
    updates = 0

    while True:
        misclassified = 0
        for i in range(len(y)):
            y_pred = np.sign(np.dot(weights, X[i]) + bias)
            if y_pred != y[i]:
                weights += y[i] * X[i]
                bias += y[i]
                updates += 1
                misclassified += 1
        if misclassified == 0:
            break
    return weights, bias, updates


if __name__ == "__main__":
    dimension = 10
    target_weights, target_bias = r_ten(dim=dimension)
    X, y = dataset_r10(1000, target_weights, target_bias)
    learned_weight, learned_bias, num_updates = pla_r10(X, y)


    print("Updates before convergence: ", num_updates)
    print("Target weights:", target_weights)
    print("Learned weights: ", learned_weight)
    print("Target bias:", target_bias)
    print("Learned bias: ",learned_bias)