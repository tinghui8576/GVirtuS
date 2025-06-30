with open("testing.out") as f:
    lines = f.readlines()

functions = []
for line in lines:
    line = line.strip()
    if ":" in line:
        func = line.split(":", 1)[1].strip()
        functions.append(func)

unique_functions = sorted(set(functions))

for func in unique_functions:
    print(func)
