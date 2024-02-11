import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

class AntColonyOptimization:
    """
    Реализация алгоритма Ant Colony Optimization, сочетающая в себе однопоточную и многопоточную реализацию
    """
    def __init__(self, num_points, num_ants, graph_range=100, alpha=1, beta=2, Q=100, pheromone_decay=0.1):
        """
        Конструктор, задаем константы и создаем случайный 2D граф
        """
        self.num_points = num_points
        self.num_ants = num_ants
        self.graph = np.random.rand(num_points, 2) * graph_range
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.pheromone_decay = pheromone_decay

    def _distance(self, p1, p2):
        # Вычисление евклидова расстояния между двумя точками
        return np.linalg.norm(p1 - p2)

    def _pheromone_update(self, best_path, best_distance):
        # Обновление феромонов на лучшем пути
        pheromone_increment = self.Q / best_distance
        for i in range(self.num_points - 1):
            self._pheromone_table[best_path[i], best_path[i + 1]] += pheromone_increment
        self._pheromone_table[best_path[-1], best_path[0]] += pheromone_increment

    def _pheromone_evaporation(self):
        # Испарение феромонов с заданным коэффициентом
        self._pheromone_table *= (1 - self.pheromone_decay)

    def display_graph(self, path=None):
        # Отображение графа с возможным оптимальным путем
        plt.figure()
        plt.scatter(self.graph[:, 0], self.graph[:, 1])
        if path:
            for i in range(len(path) - 1):
                plt.plot([self.graph[path[i], 0], self.graph[path[i + 1], 0]],
                         [self.graph[path[i], 1], self.graph[path[i + 1], 1]], 'k-')
            plt.plot([self.graph[path[-1], 0], self.graph[path[0], 0]],
                     [self.graph[path[-1], 1], self.graph[path[0], 1]], 'k-')
        plt.show()

    def _calculate_ant_path(self, ant_id):
        visited = [False] * self.num_points
        path = [ant_id]

        for _ in range(self.num_points - 1):
            current = path[-1]
            visited[current] = True

            probabilities = [(self._pheromone_table[current][i] ** self.alpha) * ((1 / self._distance(self.graph[current], self.graph[i])) ** self.beta)
                             for i in range(self.num_points) if not visited[i]]
            probabilities = [p / sum(probabilities) for p in probabilities]

            next_point = np.random.choice([i for i in range(self.num_points) if not visited[i]],
                                           p=probabilities)

            path.append(next_point)

        return path, sum(self._distance(self.graph[path[i]], self.graph[path[i + 1]]) for i in range(self.num_points - 1))

    def calculate_path(self, max_iterations):
        """
        Однопоточная реализация алгоритма
        """
        best_path = None
        best_distance = np.inf
        self._pheromone_table = np.ones((self.num_points, self.num_points))

        for _ in range(max_iterations):
            paths = [self._calculate_ant_path(i) for i in range(self.num_ants)]
            paths.sort(key=lambda x: x[1])

            if paths[0][1] < best_distance:
                best_path = paths[0][0]
                best_distance = paths[0][1]
                self._pheromone_update(best_path, best_distance)
            self._pheromone_evaporation()

        return best_path, best_distance

    def calculate_path_parallel(self, max_iterations):
        """
        Многопоточная реализация алгоритма
        """
        best_path = None
        best_distance = np.inf
        self._pheromone_table = np.ones((self.num_points, self.num_points))

        with Pool() as pool:
            for _ in range(max_iterations):
                paths = pool.map(self._calculate_ant_path, range(self.num_ants))
                paths.sort(key=lambda x: x[1])

                if paths[0][1] < best_distance:
                    best_path = paths[0][0]
                    best_distance = paths[0][1]
                    self._pheromone_update(best_path, best_distance)
                self._pheromone_evaporation()

        return best_path, best_distance

if __name__ == "__main__":
    num_points = 20
    num_ants = 20
    max_iterations = 100

    aco = AntColonyOptimization(num_points, num_ants)
    aco.display_graph()

    start_time1 = time.time()
    best_path, best_distance = aco.calculate_path(max_iterations)
    print(f"Best path: {best_path}")
    print(f"Best distance: {best_distance}")
    end_time1 = time.time()
    print(f"Time taken: {end_time1 - start_time1} seconds")
    aco.display_graph(best_path)

    print()

    start_time2 = time.time()
    best_path_parallel, best_distance_parallel = aco.calculate_path_parallel(max_iterations)
    print(f"Best path (parallel): {best_path_parallel}")
    print(f"Best distance (parallel): {best_distance_parallel}")
    end_time2 = time.time()
    print(f"Time taken (parallel): {end_time2 - start_time2} seconds")
    aco.display_graph(best_path_parallel)


    best_path1, best_distance1 = aco.calculate_path(100)
