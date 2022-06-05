import json
import sys
import numpy as np
import matplotlib.pyplot as plt

training_json = sys.argv[1]
train_run = json.load(open(training_json, "r"))["accuracy"]

plt.figure()
plt.plot(train_run[:100])
plt.xlabel("Iteration", size=14)
plt.ylabel("Accuracy (%)", size=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.title("Live Demo Training Results", size=16)
plt.savefig("live_demo/training_plot.png")

