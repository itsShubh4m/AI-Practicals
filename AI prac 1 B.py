from collections import deque

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}

visited = []
queue = deque()

def bfs(start):
    visited.append(start)
    queue.append(start)

    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for i in graph[node]:
            if i not in visited:
                visited.append(i)
                queue.append(i)

print("BFS Traversal:")
bfs('A')
