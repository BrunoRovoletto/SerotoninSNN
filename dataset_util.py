import IPython
assert IPython.version_info[0] >= 3, "Your version of IPython is too old, please update it to at least version 3."

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set fixed random seed for reproducibility
np.random.seed(2022)

class Synthetic_Dataset_Utils():

    def __init__(self, kernel_type = "RBF"):

        if(kernel_type == "RBF"):
          self.kernel = self.RBF_kernel

    def get_distance(self, x1,x2):

        return np.sqrt(np.sum((x1-x2)**2))

    def RBF_kernel(self, x1, x2, theta):
        """
        Define the multivariate kernel function.

        Parameters
        ----------
        x1 : (D,) np.ndarray
          A D-dimensional data point.
        x2 : (D,) np.ndarray
          Another D-dimensional data point.
        theta : (4,) np.ndarray
          The array containing the hyperparameters governing the kernel function.

        Returns
        -------
        float
          Value of the kernel function.
        """

        x1 = np.array([x1])
        x2 = np.array([x2])


        return theta[0] * np.exp(- theta[1]/2 * self.get_distance(x1, x2)**2 ) + theta[2] + theta[3] * x2@x1.reshape(-1,1)


    def compute_Gram_matrix(self, X, theta):
        N = X.shape[0]
        X_norm = np.sum(X**2, axis=1).reshape(-1,1)
        distance_squared = X_norm + X_norm.T - 2 * X @ X.T
        G = theta[0] * np.exp(- theta[1]/2 * distance_squared) + theta[2] + theta[3] * X @ X.T
        regularizer = np.eye(N)*1.e-8
        return G + regularizer

    def compute_Gram_matrix_GPU(self, X, theta):
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_torch = torch.from_numpy(X).to(device)
        theta_torch = torch.from_numpy(theta).to(device)
        N = X_torch.shape[0]
        X_norm = torch.sum(X_torch**2, dim=1).view(-1,1)
        distance_squared = X_norm + X_norm.T - 2 * X_torch @ X_torch.T
        K_torch = theta_torch[0] * torch.exp(- theta_torch[1]/2 * distance_squared) + theta_torch[2] + theta_torch[3] * X_torch @ X_torch.T
        regularizer = torch.eye(N, device=device)*1.e-8
        return K_torch + regularizer


    def generate_surfaces(self, theta, num_samples = 1, N = 21):

        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        X, Y = np.array(np.meshgrid(x, y))
        X = np.stack([X.ravel(), Y.ravel()], axis=1)
        S = X.shape[0]
        mu = np.zeros(S)
        K = self.compute_Gram_matrix(X, theta)
        curves = np.random.multivariate_normal(mu, K, size = num_samples)
        return [curve.reshape((N,N)) for curve in curves]

    def generate_surfaces_GPU(self, theta, num_samples=1,  N = 21):

        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        X, Y = np.array(np.meshgrid(x, y))
        X = np.stack([X.ravel(), Y.ravel()], axis=1)
        S = X.shape[0]
        mu = np.zeros(S)
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mu_torch = torch.from_numpy(mu).to(device)
        K_torch = self.compute_Gram_matrix_GPU(X, theta)
        mvn = torch.distributions.MultivariateNormal(mu_torch, covariance_matrix=K_torch)
        curves_torch = mvn.sample((num_samples,))
        curves = curves_torch.cpu().numpy()
        return [curve.reshape((N,N)) for curve in curves]


    def plot_curve(self, curve, title = "Curve"):

        curve = curve[0]

        N = curve.shape[1]
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)

        X, Y = np.array(np.meshgrid(x, y))


        ax = plt.figure().add_subplot(projection='3d')

        surf = ax.plot_surface(X, Y, curve, cmap='viridis')

        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()




    def build_tree(self, params, depth, max_depth, num_classes_per_level, std_multiplier, ranges):
      node = [params]  # The first element is the parameters at this node

      if depth >= max_depth:
          return node  # Return the node containing only parameters

      else:
          children = []

          for _ in range(num_classes_per_level):

              new_params = []
              for p_idx, parameter in enumerate(params):

                  std_dev = std_multiplier * ranges[p_idx] / ((num_classes_per_level) ** (depth + 1))
                  sampled_param = np.random.normal(loc=parameter, scale=std_dev)
                  while(sampled_param < 0):
                      sampled_param = np.random.normal(loc=parameter, scale=std_dev)
                  new_params.append(sampled_param)

              # Recursively build the child node
              child_node = self.build_tree(new_params, depth + 1, max_depth, num_classes_per_level, std_multiplier, ranges)
              children.append(child_node)

          node.append(children)  # Append the list of children to the node
          return node

    def get_parameters_by_index(self, tree, indices):
      node = tree
      if not isinstance(indices, tuple):
          indices = (indices,)
      for idx in indices:
          # Parameters are at node[0], children are at node[1]
          if len(node) <= 1:
              raise IndexError("No further children at this index")
          node = node[1][idx]  # Move to the child node at the given index
      return node[0]  # Return the parameters at the final node


    def get_parameters_by_depth(self, tree, depth):
      result = []
      if depth == 0:
          return [tree[0]]
      elif len(tree) > 1:
          for child in tree[1]:
              result.extend(self.get_parameters_by_depth(child, depth - 1))
      return result


    def _get_cat_idx(self, num_classes_per_level, max_depth):
      cat_idx = []

      for i in range(27):

        idx = []

        for j in reversed(range(max_depth)):

          z = math.floor(i/num_classes_per_level**j)
          idx.append( z % num_classes_per_level)

        cat_idx.append(tuple(idx))

      return tuple(cat_idx)

    def get_depth(self, tree):

        if len(tree) <= 1:  # No children, depth is 1
            return 0
        else:
            # Add 1 for the current level and calculate the max depth of the children
            return 1 + max(self.get_depth(child) for child in tree[1])

    def make_dataset(self, tree, num_samples_per_class, GPU = True, N=21):

        dataset = {}
        max_depth = self.get_depth(tree)
        num_classes_per_level = len(self.get_parameters_by_depth(tree, 1))

        leaves = self.get_parameters_by_depth(tree, max_depth)   # Actual final categories

        categories = {}
        curves = {}

        category_indexes = self._get_cat_idx(num_classes_per_level, max_depth)

        for i, leaf in enumerate(leaves):

          categories[category_indexes[i]] = tuple(leaf)


        for i, leaf in enumerate(leaves):

          if GPU:
            curves[category_indexes[i]] = [curve.reshape(N,N) for curve in self.generate_surfaces(leaf, num_samples = num_samples_per_class, N = N)]
          else:
            curves[category_indexes[i]] = [curve.reshape(N,N) for curve in self.generate_surfaces(leaf, num_samples = num_samples_per_class, N = N)]

        dataset["categories"] = categories
        dataset["curves"] = curves

        return dataset

    def get_curves(self, dataset, category):

        return dataset["categories"][category], dataset["curves"][category]




