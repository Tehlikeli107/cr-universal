import os
for r, d, fs in os.walk(r"C:\Users\salih\Desktop"):
    for f in fs:
        if "graph9" in f:
            print(os.path.join(r, f))
