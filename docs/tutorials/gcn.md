# Graph Convolutional Networks

In this tutorial, you will create a [Graph Convolutional Network](https://arxiv.org/abs/1609.02907) (GCN) layer using <span style="font-variant:small-caps;">libmg</span>.

A GCN layer performs a graph convolution by multiplying the node features with a matrix of weights, normalized using the adjacency matrix with self loops 
and the corresponding degree matrix. The node features $Z \in \mathbb{R}^{N \times C}$ are computed according to the equation 

$$
Z = f(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}X\Theta)
$$

where $\tilde{A}$ is the adjacency matrix $A$ with the added self loops $\tilde{A} = A + I$, $\tilde{D}$ is its
degree matrix, $X \in \mathbb{R}^{N \times F}$ is the node features matrix, and $\Theta \in \mathbb{R}^{F \times C}$ is a matrix of trainable weights. The
function $f$ is the activation function, for example the $ReLU$ function $ReLU(x) = \max(0, x)$.

Since the term $A' = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$ is constant for each graph we can pre-compute it. The input graph will be
modified so that each node has a self loop and have edge labels corresponding to the entries of $A'$. 

We start by defining a `Dataset` that contains a single `Graph`. For this tutorial, we will simply obtain the graph on-the-fly using the `read` method. The 
node features are three-dimensional one-hot vectors of type `float32`. The adjacency matrix will have `uint8` binary entries and every node will feature a 
self loop.The edge features will be a single `float32` value obtained corresponding to the values of $A'$ above.

```python
import numpy as np
from scipy.sparse import coo_matrix
from libmg import Dataset, Graph
class MyDataset(Dataset):
    def read(self):
        X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0]],
                     dtype=np.float32)
        A = coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        ([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
                         [0, 1, 2, 1, 2, 3, 1, 2, 3, 3, 4, 1, 4])),
                       shape=(5, 5), dtype=np.float32)
        E = np.array([[0.3333333], [0.3333333], [0.3333333], [0.3333333],
                      [0.3333333], [0.40824828], [0.3333333], [0.3333333],
                      [0.40824828], [0.49999997], [0.49999997], [0.40824828],
                      [0.49999997]], dtype=np.float32)
        return [Graph(x=X, a=A, e=E)]
```



To implement the GCN, we define a $\psi$ function that implements the product of the node features $X$ with the weight matrix $\Theta$ followed 
by the application of the activation function $f$, i.e. the classic dense neural network layer. For this tutorial, we will have 5 outputs features, 
therefore we pass 5 to the `tf.keras.layers.Dense` constructor. Then we also need a $\varphi$ function that generates the 
messages as the hadamard product between the edge labels and the node labels and a $\sigma$ function that aggregates the messages by summing them.

```python
import tensorflow as tf
from libmg import PsiLocal, Phi, Sigma
dense = PsiLocal(tf.keras.layers.Dense(5))
prod = Phi(lambda i, e, j: i * e)
add = Sigma(lambda m, i, n, x: tf.math.segment_sum(m, i))
```

We can now create the compiler instance. We will use `dense` to refer to the $\psi$ function, while we use the symbols $*$ and $+$ to denote the 
$\psi$ function and the $\sigma$ function. The `CompilerConfig` object is created using the `xae_config` method (since we use the `SingleGraphLoader` for 
datasets consisting of a single graph and we have edge labels), and it specifies the types of the node labels, edge labels and the adjacency matrix. We do 
not specify any tolerance value since we will not be using fixpoints.

```python
from libmg import MGCompiler, CompilerConfig, NodeConfig, EdgeConfig

compiler = MGCompiler(psi_functions={'dense': dense},
                      phi_functions={'*': prod},
                      sigma_functions={'+': add},
                      config=CompilerConfig.xae_config(NodeConfig(tf.float32, 3),
                                                       EdgeConfig(tf.float32, 1),
                                                       tf.float32, {})
                      )
```
Now we can create our model, consisting of a single GCN layer, using the $\mu\mathcal{G}$ expression $\rhd_{+}^{*} ; \mathtt{dense}$:

```python
model = compiler.compile('|*>+ ; dense')
```
Normally, this model would be trained on the data on the target node labels (which we didn't define for the dataset above). In this tutorial we skip this part 
and show how the model can be run on the dataset we defined earlier. Since the dataset consists of a single graph, we will be using the `SingleGraphLoader`.

```python
from libmg import SingleGraphLoader
# We instantiate the dataset first
dataset = MyDataset()
loader = SingleGraphLoader(dataset)
```

We obtain our single graph instance, as a list of `Tensor` objects, by getting the next element from `loader.load()` and we pass it to our model using the 
`call` API:

```python
inputs = next(iter(loader.load()))
print(model.call(inputs))
```
We should obtain in output a `Tensor` similar to this:

```python
tf.Tensor(
[[-0.10205704 -0.12557599  0.21571322 -0.48293048  0.28344738]
 [-0.40618476 -0.14519213  0.22650823 -0.6907434   0.54866683]
 [-0.40618476 -0.14519213  0.22650823 -0.6907434   0.54866683]
 [-0.10484731 -0.05232301  0.08413776 -0.3304627   0.15030748]
 [-0.03938639 -0.111077    0.19549546 -0.32164502  0.22442521]],
shape=(5, 5), dtype=float32)
```

We can see that the model has trainable weights by calling `model.summary()`. You should obtain the following output:

```python
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 INPUT_X (InputLayer)           [(None, 3)]          0           []                               
                                                                                                  
 INPUT_A (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 INPUT_E (InputLayer)           [(None, 1)]          0           []                               
                                                                                                  
 post_image_Phi_Sigma (PostImag  (None, 3)           0           ['INPUT_X[0][0]',                
 e)                                                               'INPUT_A[0][0]',                
                                                                  'INPUT_E[0][0]']                
                                                                                                  
 function_application_dense (Fu  (None, 5)           15          ['post_image_Phi_Sigma[0][0]']   
 nctionApplication)                                                                               
                                                                                                  
==================================================================================================
Total params: 15
Trainable params: 15
Non-trainable params: 0
__________________________________________________________________________________________________
```

We have implemented a GCN layer, a trainable component of many GNN applications. Instead of modifying the adjacency matrix, which is not allowed in $\mu\mathcal{G}$, we have used edge labels to encode 
the same information. We have defined some functions and composed them according to the rules of $\mu\mathcal{G}$, obtaining a model that behaves in the 
same way as a GCN layer.