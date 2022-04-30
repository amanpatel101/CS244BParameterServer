TODO -

(1) Distribute more weights in model parallel - currentl only 2 weights are distributed self.W and self.b
(2) Test the model parallel implementation on a different datasets (can try MNIST too not sure if the performance will be good - can try MNIST first and then move to other dataset)
(3) Integrate consistency hashing into the current implementation.

Evaluation TODO -

(1) How keys-server distribution is happening
(2) Profile bandwidth after model parallel - compare with data parallel only - Comment on how this effects training time and accuracy
(3) Simulate effect of checkpointing frequently versus after n epochs
(4) What happens when a server is added? - show change in bandwidth
(5) What happens when a server is killed? - go back to previous checkpoint - show lag in convergence
(6) What happens when a server is killed? - add step to replicate data across servers - compare with (5)


