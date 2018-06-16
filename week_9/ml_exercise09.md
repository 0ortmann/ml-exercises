### Exercise 09 Machine Learning 

#### Authors

- Felix Ortmann
- Ina Reis


#### Task 09.01

All plots are in the `./plots` folder. The file names are self descriptive.

It was possible to use very small networks for the first three data sets. We needed 1-2 features and 1-2 neurons. For the first set, we needed 2 layers but for set 2 and three a single layer was sufficient for very minimal error results.

Suprisingly, this did not even change when we added noise. The noise was misclassified - that is totally expectable. 

We had some trouble with set 4, though. It was difficult to find a matching network at all, yet alone a minimal one (see the corresponding plots). The minimal network we found has three layers with 6-4-2 neurons. When we added noise, set 4 got a lot harder to classify. The learning phase of the network was disturbed very much, during the first ~1500 epochs there was a lot of "jumping" in the classification output. But the network stabilized and yielded comparable good results to the network without noise. It just took significantly more epochs to train.