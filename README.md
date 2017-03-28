# two-stage-condnet
Haha it's actually three stages, depending on how you count:
- (Pre)Train target
- Partitionning - RL
- Compute computation policy based on partition


#### Partitioning
hp : number of partition.
Options :
-> One bandit by neuron
-> Sequence to sequence
   input : sequence of weight
   output : sequence of labels
-> One big contextual bandit by layer (context : a neuron's weights)

objective : meta validation accuracy of 2.2, computation cost?


#### Computation policy
Implementation details : early stopping : nb de fois que validation loss a augmenté après une époque. tolerance = 5

objective func : training accuracy, lambda nb de neuronne activée

- Reinforce
- Linear actor critic
- Gumbell softmax as policy