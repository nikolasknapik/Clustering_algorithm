import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time

def calculate_distance(point1,point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def generate_first_points():
    n = 20
    minimum = -5000
    maximum = 5000
    points = set()

    while len(points) < n:
        x = random.randint(minimum,maximum)
        y = random.randint(minimum,maximum)
        point = (x,y)
        if point not in points:
            points.add(point)

    return list(points)

def generate_points(first_points):
    points = first_points[:]
    n = 20000
    offset = 100

    for i in range(n):
        point = random.choice(points)
        x_offset = random.randint(-offset, offset)
        y_offset = random.randint(-offset, offset)

        new_x = point[0] + x_offset
        new_y = point[1] + y_offset

        if new_x > 5000:
            new_x = 5000
        if new_x < -5000:
            new_x = -5000
        if new_y > 5000:
            new_y = 5000
        if new_y < -5000:
            new_y = -5000

        points.append((new_x, new_y))

    return points


def create_distance_matrix(points):
    points_array = np.array(points)
    num_points = len(points_array)
    distance_matrix = np.full((num_points, num_points), np.inf)  
    for i in range(num_points):
        for j in range(i + 1, num_points): 
            distance = np.linalg.norm(points_array[i] - points_array[j]) 
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  

    return distance_matrix


def find_closest_clusters(distance_matrix):
    min_dist = np.min(distance_matrix)
    closest_pair = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
    return closest_pair

def calculate_centroid(cluster_a, cluster_b, centroid_a, centroid_b):
    len_a = len(cluster_a)
    len_b = len(cluster_b)
    new_x = (centroid_a[0] * len_a + centroid_b[0] * len_b) / (len_a + len_b)
    new_y = (centroid_a[1] * len_a + centroid_b[1] * len_b) / (len_a + len_b)
    return (new_x, new_y)

def calculate_medoid(cluster):
    if len(cluster) == 1:
        return cluster[0]

    min_total_distance = float('inf')
    medoid = cluster[0]

    for point in cluster:
        total_distance = sum(calculate_distance(point, other_point) for other_point in cluster)
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            medoid = point

    return medoid


def update_distance_matrix(distance_matrix, new_cluster, removed_cluster, centroids):
    infinity = np.inf
    
    distance_matrix[removed_cluster, :] = infinity
    distance_matrix[:, removed_cluster] = infinity

    new_centroid = centroids[new_cluster]
    remaining_indices = list(centroids.keys()) 

    for i in remaining_indices:
        if i != new_cluster:
            other_centroid = centroids[i]
            new_distance = calculate_distance(new_centroid, other_centroid)
            distance_matrix[new_cluster, i] = new_distance
            distance_matrix[i, new_cluster] = new_distance

    return distance_matrix


def visualize_clusters(clusters, centroids, title="Vizualizácia zhlukov"):
    colors = plt.cm.get_cmap("tab10", len(clusters))
    plt.figure(figsize=(10, 8))

    for cluster_id, points in clusters.items():
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        plt.scatter(x_coords, y_coords, color=colors(cluster_id))
        centroid = centroids[cluster_id]
        plt.scatter(centroid[0], centroid[1], color='black', marker="x", s=100, edgecolor='black')

    plt.title(title)
    plt.xlabel("X súradnice")
    plt.ylabel("Y súradnice")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_avg_distance(cluster, center):
    total_distance = sum(calculate_distance(point, center) for point in cluster)
    return total_distance / len(cluster)


def calculate_success_rate(clusters, centroids):
    threshold = 500
    successful_clusters = 0

    for cluster_id, cluster_points in clusters.items():
        center = centroids[cluster_id]
        avg_distance = calculate_avg_distance(cluster_points, center)
        if avg_distance < threshold:
            successful_clusters += 1

    success_rate = (successful_clusters / len(clusters)) * 100
    return success_rate


start_time = time.time()
last_log_time = start_time 

first_points = generate_first_points()  
all_points = generate_points(first_points)
distance_matrix = create_distance_matrix(all_points)

clusters = {i: [point] for i, point in enumerate(all_points)}
centroids = {i: point for i, point in enumerate(all_points)}

num_clusters = len(clusters)
target_clusters = 10

visualize_clusters(clusters, centroids, title="Pred zhlukovaní")

use_centroids = True

while True:
    closest_pair = find_closest_clusters(distance_matrix)
    cluster_a, cluster_b = closest_pair

    clusters[cluster_a].extend(clusters[cluster_b])

    if use_centroids:
        new_value = calculate_centroid(
            clusters[cluster_a], clusters[cluster_b], centroids[cluster_a], centroids[cluster_b]
        )
    else:
        new_value = calculate_medoid(clusters[cluster_a])

    centroids[cluster_a] = new_value
    del clusters[cluster_b]
    del centroids[cluster_b]

    distance_matrix = update_distance_matrix(distance_matrix, cluster_a, cluster_b, centroids)
    

    success_rate = calculate_success_rate(clusters,centroids)
    if success_rate < 100:
        print("Break kvolu 500")
        break

    current_time = time.time()
    if current_time - last_log_time >= 30 * 60:  
        elapsed_minutes = (current_time - start_time) / 60
        print(f"Prešlo {elapsed_minutes:.2f} minút, stále pracujeme...")
        last_log_time = current_time


end_time = time.time()

execution = end_time - start_time
print(f"Program trval {(execution/60):.2f} minút")
visualize_clusters(clusters, centroids, title="Po zhlukovaní")

success_rate = calculate_success_rate(clusters, centroids)
print(f"Úspešnosť zhlukovania: {success_rate:.2f}% klastrov má priemernú vzdialenosť od stredu menšiu ako 500.")
