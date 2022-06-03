## Implementing and Benchmarking a Fault-Tolerant Parameter Server for Distributed Machine Learning Applications

# CS244B Final Project - Anusri Pampari and Aman Patel

This repository contains all relevant code for our implementation of a parameter server system for distributed training of machine learning models. We also provide functionality to train a logistic regression model on the MNIST dataset with the task of predicting whether a number is even or odd. 

All code can be found in the `src/` folder. Below is a brief description of important files:
 - `consistent_hashing.py` - contains our implementation of consistent hashing, which we use to distribute weights among servers
 - `server.py` - defines the servers that form the core of the distributed training procedure
 - `worker.py` - defines the data workers that are associated with each server
 - `scheduler.py` - sets up a training run by defining servers and workers, and assigning model weights to servers
 - `models.py` - defines our Logistic Regression model along with code to evaluate it
 - `data_loader.py` - loads and formats MNIST data for our training experiments

The command to run training is `python src/main.py -ns [num_servers] -nw [num_workers] -o [output_path] -i [num_iterations] -c [checkpoint] -f [do_failure_test] -s [server_to_fail] -f_iter [iteration_to_fail] -ev_int [eval_interval]`. Brief descriptions of the parameters are below: 
- `num_servers` - number of servers to use (required)
- `num_workers` - number of workers per server (required)
- `output_path` - path to store training output (required)
- `num_iterations` - number of iterations (batches) to run training for (default 500)
- `checkpoint` - interval at which to checkpoint model weights and synchronize them across workers (default 1)
- `eval_interval` - interval at which to evaluate model accuracy on test set (default 5)
- `do_failure_test` - whether to test server failure or not (default False)
- `iteration_to_fail` - which iteration server failure will occur on (default 60, only relevant if `do_failure_test` is true)
- `server_to_fail` - id of server to fail (default "server4", only relevant if `do_failure_test` is true)

For completeness, training results that comprised Figure 3 are located in `results/`, and training results that comprised Figure 4 are in `results/fault_tolerance/failure_no_failure`. Notebooks to produce the plots in our paper are `notebooks/ConsistentHashing.ipynb` (Figure 2), `notebooks/figure3_plots.ipynb` (Figure 3), and `notebooks/fault_tolerant_plots.ipynb` (Figure 4).  





