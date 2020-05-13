# SmashRNN (based on hierarchical attention)
It uses 3-Stage, 3 levels encoder (word_level, sent_level, para_level) to encode the input using bi-GRUs. Further , it uses Siamese network (of MashRNNs) to learn the similarity between the documents.
This is a modified implementation of original paper of SMASH-RNN using HAN(Hierarchical Attention Network).
It works for long size docs too.
As the elmo model itself uses bi-lstm so we use elmo representation directly instead of word_encoder.
Here, we are not using dot-product attention, instead we have used feed-forward attention. 
Binary Cross Entropy has been used as loss function.
The pytoch implementation is uploaded.
