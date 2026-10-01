Seed=1, steps=4000, lr=0.5, optimiser=SGD, loss=BCEWithLogits

| hidden activation | final loss | 4/4 correct? | early ||grad W1||_2 |
|---|---|---|---|
| sigmoid | 0.0124 | True | 0.001753 |
| tanh | 0.3473 | False | 0.008501 |
| relu | 0.6931 | False | 0.006358 |
