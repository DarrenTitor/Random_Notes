# chap5 - Vector Calculus

本书讨论的function通常input为$x\in \mathbb{R}^D$，output为f(x)。

$\mathbb{R}^D$称为domain of f，函数值的集合称为image/codomain of f。
![](Pasted%20image%2020221023164946.png)

![](Pasted%20image%2020221023165049.png)

![](Pasted%20image%2020221023165209.png)


Throughout this book, we assume that functions are differentiable.
![](Pasted%20image%2020221023165315.png)


## 5.1 Differentiation of Univariate Functions

![](Pasted%20image%2020221023165814.png)



![](Pasted%20image%2020221023165848.png)
The derivative of f points in the direction of steepest ascent of f.

### 5.1.1 Taylor Series
![](Pasted%20image%2020221023170411.png)

![](Pasted%20image%2020221023170527.png)
![](Pasted%20image%2020221023170614.png)
其中这个$f\in C^{\infty}$是无限次可微的意思。
![](Pasted%20image%2020221023170619.png)

通常，Taylor polynomial of degree n是对任意函数f的approximation。这里强调f是polynomial是因为多项式在有限次可微之后导数就会变成0，这时候当n足够大（且没有必要取到正无穷）时就可以得到对f的精确representation。
另外注意Taylor polynomial最终得到的是f这个函数本身，而不是$f(x_0)$这个函数值。

![](Pasted%20image%2020221023172534.png)


### 5.1.2 Differentiation Rules

![](Pasted%20image%2020221023172901.png)


## 5.2 Partial Differentiation and Gradients

The generalization of the derivative to functions of several variables is the gradient.

![](Pasted%20image%2020221023173424.png)

Partial Derivative：
注意这里gradient是一个row vector。
gradient/jacobian的行数是domain的维数，列数是codomain的维数。
![](Pasted%20image%2020221023174008.png)


很多人将gradient vector写作column vector，
本书写作row vector主要是为了multi-variate chain rule时计算方便，不需要额外留意维度。
![](Pasted%20image%2020221023174228.png)

### 5.2.1 Basic Rules of Partial Differentiation

![](Pasted%20image%2020221023180422.png)
![](Pasted%20image%2020221023180430.png)
上面的性质依然成立，但需要注意矩阵乘法没有交换律。

### 5.2.2 Chain Rule



![](Pasted%20image%2020221023183622.png)

![](Pasted%20image%2020221023183924.png)

![](Pasted%20image%2020221023184858.png)
注意上面式子中，gradient的形状，行数对应的时codomain，列数对应的是domain。
只要一直遵守上面的原则，chain rule就可以简单地写成矩阵连乘，而不用不停地transpose。
![](Pasted%20image%2020221023185052.png)


## 5.3 Gradients of Vector-Valued Functions

接下来我们推广到f的输出不是实数，而是vector的情况：

一个输出为m维向量的f，其实可以看做是m个输出为实数的函数f组成的column vector：
![](Pasted%20image%2020221023185955.png)

由此，向量f对某一个x的偏导也可以写为column vector：
![](Pasted%20image%2020221023190434.png)
向量f对所有x的偏导组成的就是一个矩阵：
![](Pasted%20image%2020221023190700.png)

![](Pasted%20image%2020221023191113.png)


***
jacobian的一个重要应用就是概率论中的变量替换（Section 6.7），称为change-of-variable method。
而变量替换带来的scaling要通过jacobian的determinant来揭示。
The amount of scaling due to the transformation of a variable is provided by the determinant.

物理意义上的scaling factor（不考虑flipping），就是determinant的绝对值。
在求这个scaling factor的时候我们可以观察一个unit单位体积，看经过线性变换之后体积变成了多少。
![](Pasted%20image%2020221024003709.png)

下面介绍两种方法来求这个scaling factor。假设我们想perform a variable transformation from $(b_1, b_2)$ to $(c_1, c_2)$，这两组都是2维空间的basis. 
然后在下面的例子中，这两组basis的取值如下图：
![](Pasted%20image%2020221024004145.png)

Approach 1
因为basis $(b_1, b_2)$是standard basis，所以basis change matrix就直接是变换后basis的坐标，也就是$(c_1, c_2)$组成的矩阵:
![](Pasted%20image%2020221024004528.png)
![](Pasted%20image%2020221024004544.png)
因此scaling factor就是determinant的绝对值。

Approach 2
上面的方法虽然简单，但是basis change matrix的这个选法只适用于线性变换。因此下面来介绍一种更通用的方法，基于偏导数来计算。

接下来我们从variable transformation的角度来看f，
f将一组相对于$(b_1, b_2)$的坐标映射到一组相对于$(c_1, c_2)$的坐标。
![](Pasted%20image%2020221024010052.png)
我们想观察经过f变换前后，单位体积的变化。说白了可以不严谨地表示成f(volume)/volume，这就让人联想到斜率，进而想到导数/gradient。
![](Pasted%20image%2020221024010441.png)

然后就能得到下面的计算：
![](Pasted%20image%2020221024010610.png)
Jacobian就反映了坐标的变换。
并且当变换时线性变换的时候，jacobian是exact的。
对于非线性变换，the Jacobian approximates this nonlinear transformation locally with a linear one.





上面的Jacobian determinant和variable transformations通常用于对random variables and probability distributions进行变换，在机器学习中非常重要。
在NN的语境下，通常会用于reparametrization trick，也叫infinite perturbation analysis。
![](Pasted%20image%2020221024011105.png)
***
下面总结一下当函数f的输入输出分别是scalar或vector的时候，gradient的维度：
![](Pasted%20image%2020221024011252.png)
![](Pasted%20image%2020221024011313.png)

$f(x)=Ax$，则f对x的gradient就是A：
![](Pasted%20image%2020221024012625.png)

下面是一个chain rule的例子：
![](Pasted%20image%2020221024012822.png)

![](Pasted%20image%2020221024013138.png)


## 5.4 Gradients of Matrices
We will encounter situations where we need to take gradients of matrices with respect to vectors (or other matrices), which results in a multidimensional tensor.

矩阵对矩阵求梯度，Jacobian的形状仍然遵守之前的规则。
For example, if we compute the gradient of an m× n matrix A with respect to a p × q matrix B, the resulting Jacobian would be $(m\times n)\times (p\times q)$, i.e., a four-dimensional tensor J, whose entries are given as $J_{ijkl} = \partial A_{ij}/\partial B_{kl}$.
![](Pasted%20image%2020221024222117.png)

实际操作中，需要注意
有时我们直接进行矩阵对矩阵的求导，得到$(m\times n)\times (p\times q)$的jacobian tensor；
有时我们将矩阵先reshape成mn维和pq维的vector，然后求导得到$mn\times pq$的jacobian matrix（然后如果有需要就再将其reshape回$(m\times n)\times (p\times q)$）。因为chain rule对于矩阵来说就只是直接连乘，对于tensor还要注意维度对不对，容易出错。 
![](Pasted%20image%2020221024223451.png)


下面是两个例子，仔细看也能看懂：
![](Pasted%20image%2020221025000602.png)
![](Pasted%20image%2020221025000609.png)
![](Pasted%20image%2020221025002456.png)
![](Pasted%20image%2020221025002503.png)



## 5.5 Useful Identities for Computing Gradients

![](Pasted%20image%2020221025180623.png)

tensor的trace和transpose：
the trace of a D×D×E×F tensor would be an E×F-dimensional matrix.
Similarly, when we “transpose” a tensor, we mean swapping the first two dimensions.


## 5.6 Backpropagation and Automatic Differentiation


### 5.6.1 Gradients in a Deep Network
![](Pasted%20image%2020221025183335.png)


### 5.6.2 Automatic Differentiation

It turns out that backpropagation is a special case of a general technique in numerical analysis called automatic differentiation.

![](Pasted%20image%2020221025183834.png)
![](Pasted%20image%2020221025183847.png)
![](Pasted%20image%2020221025184711.png)

![](Pasted%20image%2020221025184733.png)

![](Pasted%20image%2020221025185211.png)


## 5.7 Higher-Order Derivatives

## 5.8 Linearization and Multivariate Taylor Series


