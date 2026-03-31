import os, glob
# Search for graph8.g6
for root, dirs, files in os.walk(r"C:\Users\salih\Desktop"):
    for f in files:
        if f == "graph8.g6":
            print(os.path.join(root, f))
