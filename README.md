# two-stage-condnet

## Report
Read the report in (docs/report.pdf)[docs/report.pdf]


Haha it's actually three stages, depending on how you count:
- (Pre)Train target
- Partitionning - RL
- Compute computation policy based on partition

Terminology:
- 'Target net-F' is the main network doing some task 'F_theta: X->Y', whose computation is divided in nodes eta (e.g. neurons in a dense nnet). 
- 'Partition model-B' is the thing that assigns each node in the target net to one of k partitions, 'B_beta: Theta->[k]^{|eta|}
- 'Computation policy-pi' is the policy that performs conditional computation on the target network 'pi_omega X->{0,1}^{|eta|}
- 'Lazy model-f' is the resulting combination of the computation policy. 'f_{theta,omega}: X->Y'

note: the words "partition function" have a very strong meaning in energy based models, do not use!



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