import heapq
import matplotlib.pyplot as plt
import networkx as nx

# Graph (Approximate road distances in km)

graph = {
    "Andheri Station": {
        "Vile Parle": 3,
        "Jogeshwari": 4
    },

    "Jogeshwari": {
        "Goregaon": 3
    },

    "Goregaon": {
        "Malad": 4
    },

    "Malad": {
        "Bandra": 11
    },

    "Vile Parle": {
        "Santacruz": 3
    },

    "Santacruz": {
        "Khar Road": 2
    },

    "Khar Road": {
        "Bandra": 2
    },

    "Bandra": {
        "Mahim": 3,
        "BKC": 4
    },

    "BKC": {
        "Sion": 5
    },

    "Sion": {
        "Dadar": 5
    },

    "Mahim": {
        "Dadar": 3
    },

    "Dadar": {
        "Prabhadevi": 2
    },

    "Prabhadevi": {
        "Siddhivinayak Temple": 1
    },

    "Siddhivinayak Temple": {}
}

# Heuristic values (Estimated distance to Siddhivinayak)

heuristics = {
    "Andheri Station": 18,
    "Jogeshwari": 17,
    "Goregaon": 19,
    "Malad": 22,
    "Vile Parle": 15,
    "Santacruz": 12,
    "Khar Road": 10,
    "Bandra": 7,
    "BKC": 6,
    "Sion": 5,
    "Mahim": 4,
    "Dadar": 2,
    "Prabhadevi": 1,
    "Siddhivinayak Temple": 0
}

# Node Positions

positions = {
    "Andheri Station": (2,10),
    "Jogeshwari": (0,9),
    "Goregaon": (-1,8),
    "Malad": (-2,7),

    "Vile Parle": (3,9),
    "Santacruz": (3,8),
    "Khar Road": (3,7),
    "Bandra": (3,6),

    "BKC": (5,5),
    "Sion": (5,3),

    "Mahim": (2,4),
    "Dadar": (2,2),
    "Prabhadevi": (2,1),
    "Siddhivinayak Temple": (2,0)
}

# A* Algorithm

def a_star(graph, heuristic, start, goal):

    pq = [(heuristic[start], start, [start], 0)]
    visited = set()

    while pq:

        f, current, path, g = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        if current == goal:
            return path, g

        for neighbor, cost in graph[current].items():

            if neighbor not in visited:

                new_g = g + cost
                new_f = new_g + heuristic[neighbor]

                heapq.heappush(
                    pq,
                    (new_f, neighbor, path + [neighbor], new_g)
                )

    return None, float("inf")


start = "Andheri Station"
goal = "Siddhivinayak Temple"

path, distance = a_star(graph, heuristics, start, goal)

print("Optimal Path:")
print(" -> ".join(path))
print("Total Distance:", distance, "km")

# Visualization

G = nx.DiGraph()

for node, neighbors in graph.items():
    for neighbor, weight in neighbors.items():
        G.add_edge(node, neighbor, weight=weight)

plt.figure(figsize=(11,8))

path_edges = list(zip(path, path[1:]))
normal_edges = [e for e in G.edges() if e not in path_edges]

nx.draw_networkx_nodes(G, positions,
                       node_color="skyblue",
                       node_size=2200)

nx.draw_networkx_edges(G, positions,
                       edgelist=normal_edges,
                       edge_color="gray",
                       arrows=True)

nx.draw_networkx_edges(G, positions,
                       edgelist=path_edges,
                       edge_color="orange",
                       width=3,
                       arrows=True)

labels = {n: f"{n}\nh={heuristics[n]}" for n in G.nodes()}
nx.draw_networkx_labels(G, positions, labels, font_size=8)

edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels)

plt.title("A* Search: Andheri Station → Siddhivinayak Temple")
plt.axis("off")
plt.show()
