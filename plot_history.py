import sys

import matplotlib.pyplot as plt
import pandas as pd

# first argument is the file to load
fp = sys.argv[1] if len(sys.argv) > 1 else None

# load the csv file
# generation,max,avg,min
df = pd.read_csv(fp)

# plot the data
df.plot(x="generation", y=df.columns[1:])
plt.title("Fitness History")
plt.xlabel("Generation")
plt.ylabel("Fitness")


# if -s is passed, save the plot to fp.png
if "-s" in sys.argv:
    fp = fp.split(".")[0]
    plt.savefig(fp + ".png")
    print("Saved to", fp + ".png")
else:
    plt.show()
