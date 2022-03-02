import importlib.util

# import the compiled module
spec = importlib.util.spec_from_file_location("tenzing", "../build/src/tenzing-python.cpython-36m-x86_64-linux-gnu.so")
tenzing = importlib.util.module_from_spec(spec)

p = tenzing.NoOp("test")
print(type(p))
print(p.name())

graph = tenzing.Graph()
print(type(graph))
print(graph.vertex_size())
graph.start_then(p)
graph.then_finish(p)
print(graph.vertex_size())

state = tenzing.State(graph)
print(type(state))
