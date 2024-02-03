import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time

class AntColonyOptimization:
    # Инициализация класса
    def __init__(self, num_points, num_ants, evaporation_rate, pheromone_deposit, alpha, beta):
        self.num_points = num_points  # Количество точек
        self.num_ants = num_ants  # Количество муравьев
        self.evaporation_rate = evaporation_rate  # Скорость испарения феромона
        self.pheromone_deposit = pheromone_deposit  # Количество феромона, откладываемого каждым муравьем
        self.alpha = alpha  # Вес феромонов
        self.beta = beta  # Вес расстояния
        self.points = np.random.rand(num_points, 2)  # Генерация случайных точек
        self.pheromones = np.ones((num_points, num_points))  # Матрица феромонов
        np.fill_diagonal(self.pheromones, 0)  # Обнуляем диагональ, так как муравьи не могут возвращаться в текущую точку

    # Расчет расстояния между двумя точками
    def distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    # Отображение графика с точками и путем
    def display_graph(self, path=None):
        plt.scatter(self.points[:,0], self.points[:,1])  # Отображение точек
        if path:  # Если найден путь
            for i in range(len(path)-1):
                plt.plot([self.points[path[i],0], self.points[path[i+1],0]], [self.points[path[i],1], self.points[path[i+1],1]], 'k-')  # Отображение пути
            plt.plot([self.points[path[-1],0], self.points[path[0],0]], [self.points[path[-1],1], self.points[path[0],1]], 'k-')  # Замыкающий отрезок пути
        plt.show()

    # Вычисление пути с использованием муравьиного алгоритма
    def calculate_path(self, max_iterations):
        best_path = None
        best_distance = float('inf')

        for _ in range(max_iterations):
            for ant in range(self.num_ants):  # Для каждого муравья
                visited = [0]  # Список посещенных точек, начинаем с первой точки
                current = 0  # Текущая точка
                unvisited = list(range(1, self.num_points))  # Список непосещенных точек

                while unvisited:  # Пока есть непосещенные точки
                    probabilities = [((self.pheromones[current, i] ** self.alpha) * ((1/self.distance(self.points[current], self.points[i])) ** self.beta)) for i in unvisited]  # Расчет вероятностей перехода в непосещенные точки
                    probabilities = probabilities / np.sum(probabilities)  # Нормализация вероятностей
                    next_point = np.random.choice(unvisited, p=probabilities)  # Выбор следующей точки по вероятностям

                    visited.append(next_point)  # Добавляем выбранную точку в посещенные
                    unvisited.remove(next_point)  # Удаляем выбранную точку из непосещенных
                    current = next_point  # Текущей точкой становится выбранная точка

                path_distance = sum([self.distance(self.points[visited[i]], self.points[visited[i+1]]) for i in range(len(visited)-1)]) + self.distance(self.points[visited[-1]], self.points[visited[0]])  # Расчет длины пути

                if path_distance < best_distance:  # Если длина пути лучше текущей лучшей
                    best_distance = path_distance  # Обновляем лучшую длину пути
                    best_path = visited  # Обновляем лучший найденный путь

                for i in range(len(visited)-1):  # Обновляем феромоны на пройденном пути
                    self.pheromones[visited[i], visited[i+1]] += self.pheromone_deposit/path_distance
                    self.pheromones[visited[i+1], visited[i]] += self.pheromone_deposit/path_distance
                self.pheromones[visited[-1], visited[0]] += self.pheromone_deposit/path_distance
                self.pheromones[visited[0], visited[-1]] += self.pheromone_deposit/path_distance

            self.pheromones *= (1 - self.evaporation_rate)  # Испарение феромонов

        return best_path, best_distance

    def calculate_ant_path(self):
        visited = [0]  # Список посещенных точек, начинаем с первой точки
        current = 0  # Текущая точка
        unvisited = list(range(1, self.num_points))  # Список непосещенных точек

        while unvisited:  # Пока есть непосещенные точки
            probabilities = [((self.pheromones[current, i] ** self.alpha) * (
                        (1 / self.distance(self.points[current], self.points[i])) ** self.beta)) for i in
                             unvisited]  # Расчет вероятностей перехода в непосещенные точки
            probabilities = probabilities / np.sum(probabilities)  # Нормализация вероятностей
            next_point = np.random.choice(unvisited, p=probabilities)  # Выбор следующей точки по вероятностям

            visited.append(next_point)  # Добавляем выбранную точку в посещенные
            unvisited.remove(next_point)  # Удаляем выбранную точку из непосещенных
            current = next_point  # Текущей точкой становится выбранная точка

        path_distance = sum([self.distance(self.points[visited[i]], self.points[visited[i + 1]]) for i in
                             range(len(visited) - 1)]) + self.distance(self.points[visited[-1]],
                                                                       self.points[visited[0]])  # Расчет длины пути

        for i in range(len(visited) - 1):  # Обновляем феромоны на пройденном пути
            self.pheromones[visited[i], visited[i + 1]] += self.pheromone_deposit / path_distance
            self.pheromones[visited[i + 1], visited[i]] += self.pheromone_deposit / path_distance
        self.pheromones[visited[-1], visited[0]] += self.pheromone_deposit / path_distance
        self.pheromones[visited[0], visited[-1]] += self.pheromone_deposit / path_distance

        self.pheromones *= (1 - self.evaporation_rate)  # Испарение феромонов
        return visited, path_distance

    def calculate_path_parallel(self, max_iterations):
        best_path = None
        best_distance = float('inf')
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.calculate_ant_path) for _ in range(max_iterations)]
            for future in futures:
                path, distance = future.result()
                if distance < best_distance:
                    best_distance = distance
                    best_path = path
        return best_path, best_distance

if __name__ == "__main__":
    num_points = 20
    num_ants = 10
    evaporation_rate = 0.1
    pheromone_deposit = 10
    alpha = 1
    beta = 5

    aco = AntColonyOptimization(num_points, num_ants, evaporation_rate, pheromone_deposit, alpha, beta)

    aco.display_graph()

    start_time1 = time.time()
    best_path1, best_distance1 = aco.calculate_path(100)
    end_time1 = time.time()

    print(f"Time taken: {end_time1 - start_time1} seconds")
    print(f"Distance: {best_distance1}")
    aco.display_graph(best_path1)

    start_time2 = time.time()
    best_path2, best_distance2 = aco.calculate_path_parallel(100)
    end_time2 = time.time()

    print(f"Time taken async: {end_time2 - start_time2} seconds")
    print(f"Distance: {best_distance2}")
    aco.display_graph(best_path2)

