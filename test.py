import heapq

test = []
heapq.heappush(test, (5,"victor"))
heapq.heappush(test, (1, "pickle"))
_, name = heapq.heappop(test)
print(name)
_, name = heapq.heappop(test)
print(name)
_, name = heapq.heappop(test)
print(name)
