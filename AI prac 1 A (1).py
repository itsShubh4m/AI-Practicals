graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

visited = []
stack = []

def dfs(start):
    stack.append(start)

    while stack:
        node = stack.pop()

        if node not in visited:
            print(node, end=" ")
            visited.append(node)

            for i in reversed(graph[node]):
                stack.append(i)

print("DFS Traversal:")
dfs('A')
