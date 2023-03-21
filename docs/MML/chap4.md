# chap 4 - Matrix Decompositions

![](Pasted%20image%2020221015101710.png)

## 4.1 Determinant and Trace


Determinants are only defined for square matrices.
we write the determinant as det(A) or sometimes as |A|.

`n*n`方阵A的determinant是将A映射到实数的一个函数。

矩阵可逆<=>determinant不为0.
![](Pasted%20image%2020221015102243.png)
determinant一旦为0，说明经过线性变换之后，原本线性独立的basis变得线性相关了，以至于determinant所代表的signed volume变为了0。换句话说经过线性变换之后，dimension减小了。
一个向量空间降维之后，自然就没办法还原会原来的空间，因此determinant为0的时候矩阵不可逆。


下图中是n=1,2,3时，determinant的closed form，比较常用。
![](Pasted%20image%2020221015102400.png)
![](Pasted%20image%2020221015102818.png)


upper-triangular矩阵：
对角线下边全为0
lower-triangular矩阵：
对角线上边全为0
注意对角线不一定为0.
![](Pasted%20image%2020221015112157.png)

![](Pasted%20image%2020221015112558.png)

![](Pasted%20image%2020221015112645.png)

determinant的计算：
Laplace expansion
![](Pasted%20image%2020221015112949.png)
这个式子很难看懂，所以直接看下面的例子：
![](Pasted%20image%2020221015113023.png)
![](Pasted%20image%2020221015113032.png)

一些性质：
![](Pasted%20image%2020221015123106.png)

第二章提到：silimar matrix本质上描述的是同一个线性变换，只不过使用的是不同的basis，因此determinant相同。
![](Pasted%20image%2020221015123229.png)

![](Pasted%20image%2020221015123305.png)

由于上面最后三条性质，我们也可以用高斯消元法求determinant。
先将矩阵变成row-echelon form，当矩阵变成triangular form时，再利用对角线乘积。
![](Pasted%20image%2020221015123535.png)

下面这三个说的是一回事，都是在讲线性变换之后向量空间的dimension没有降低。
$det(A)\neq 0\iff rk(A)=n \iff A\,is \,invertible$.
![](Pasted%20image%2020221015131939.png)



trace：
trace=方阵的对角线元素之和。
![](Pasted%20image%2020221015132837.png)
![](Pasted%20image%2020221015132843.png)
trace的性质：
![](Pasted%20image%2020221015132944.png)

上面的最后一天我可以推广到多个矩阵连乘的trace：
只要将矩阵的顺序进行cyclic permutations，trace都保持不变。
![](Pasted%20image%2020221015133238.png)
由此可以得到以下的结论：
x和y都是n维向量，$xy^T$本身是一个`n*n`方阵，但这个方阵的trace可以转化为求$tr(x^Ty)$，等于x和y的dot product $x^Ty$。
![](Pasted%20image%2020221015133354.png)


接下来讨论basis的选择会不会影响trace的值：
chap2.7.2 提到，对线性变换A进行basis变换，可以得到$S^{-1}AS$ ，然后就有
![](Pasted%20image%2020221015134455.png)
说明basis的选择不会影响trace。


借助determinant和trace，接下来我们要将一个矩阵表示为一个多项式，这个多项式之后会经常用到
Taking together our understanding of determinants and traces we can now define an important equation describing a matrix A in terms of a polynomial,
Characteristic Polynomial
![](Pasted%20image%2020221015140207.png)
其中
![](Pasted%20image%2020221015140225.png)

这式子完全不知道咋来的。


## 4.2 Eigenvalues and Eigenvectors

![](Pasted%20image%2020221015140426.png)

As we will see, the eigenvalues of a linear mapping will tell us how a special set of vectors, the eigenvectors, is transformed by the linear mapping.

定义：
![](Pasted%20image%2020221015165427.png)

在很多地方，都是默认eigenvalue时降序排列的，但本书没有这个假设。
![](Pasted%20image%2020221015165501.png)



下面这些结论都是等价的：
![](Pasted%20image%2020221015165656.png)

Collinearity and Codirection
两向量方向相同称为codirection，
两向量方向相同或相反称为collinear。
![](Pasted%20image%2020221015165732.png)

假如x时A的eigenvector，则与x collinear的向量都是A的eigenvector。
![](Pasted%20image%2020221015170018.png)



$\lambda$是eigenvalue<=>$\lambda$是characteristic polynomial的一个root。（不知道有啥用）
![](Pasted%20image%2020221015170055.png)
=>
上面这个式子characteristic polynomial 是用来具体计算eigenvalue的。
我们不需要记住那一长串，只需要记住$p_A(\lambda)=det(A-\lambda I)$.
然后因为这个式子的root是eigenvalue，所以我们令$det(A-\lambda I)=0$，然后就能解出所有的$\lambda$.






然后这里定义了一个algebraic multiplicity，也不知道是干嘛的。
![](Pasted%20image%2020221015170355.png)

一个eigenvalue对应很多个eigenvector，我们将某一个eigenvalue $\lambda$所对应的所有eigenvector的span称为eigenspace of A with respect to λ 。记作$E_\lambda$.
A的所有eigenvalue（注意不是eigenvector）构成的集合称为eigenspectrum或者spectrum。
![](Pasted%20image%2020221015170904.png)


如果$\lambda$是A的一个eigenvalue，则$E_\lambda$是homogeneous system of linear equations $(A-\lambda I)x=0$的solution space。
![](Pasted%20image%2020221015171043.png)


几何意义上，如果一个eigenvector对应的是一个非零的eigenvalue，
说明eigenvector指向的这个方向是被线性变换“stretch”了的，而“stretch”的程度则由eigenvalue来描述。
![](Pasted%20image%2020221015171459.png)
我的理解：
观察$Ax=\lambda x$，在线性变换之后，x并没有被旋转，而是和原来的x成collinear关系。
也就是说eigenvector代表着线性变换在这个方向只有伸缩，没有旋转。


![](Pasted%20image%2020221015172129.png)


一些常用性质：
- transpose之后eigenvalue不变，但eigenvector可能改变。
- eigenspace $E_\lambda$ is the null space of $A-\lambda I$。
![](Pasted%20image%2020221015172327.png)

- similar matrices的eigenvalue相同。
因为similar matrices本身描述的是同一个线性变换，而eigenvalues记录的是只受到“stretch”的方向被“stretch”的程度，而这个“程度”与basis选择无关。所以eigenvalue相同。
由此我们得到了三个不随basis change而改变的量：
eigenvalues、determinant和trace。用它们来描述线性变换本身的性质。
![](Pasted%20image%2020221015172659.png)
- Symmetric, positive definite matrices的eigenvalue一定大于0
![](Pasted%20image%2020221015174559.png)
![](Pasted%20image%2020221015175023.png)


具体计算eigenvalues，eigenvectors的过程：
![](Pasted%20image%2020221015181131.png)
![](Pasted%20image%2020221015181200.png)
![](Pasted%20image%2020221015181212.png)


某一个eigenvalue对应的linear independent eigenvector的个数，就是这个eigenvalue的geometric multiplicity。
也就是说geometric multiplicity就是这个eigenvalue对应的eigenspace的dimensionality。
![](Pasted%20image%2020221015185837.png)

geometric multiplicity不会超过algebraic multiplicity。
An eigenvalue’s geometric multiplicity cannot exceed its algebraic multiplicity, but it may be lower.
![](Pasted%20image%2020221015185728.png)


注意：对于代表旋转操作的线性变换，因为没有哪一个方向是只经过“stretch”而没有经过旋转的，因此这样的矩阵只有complex eigenvalue，而没有实数eigenvalue。
![](Pasted%20image%2020221015190608.png)

如果一个矩阵的eigenvalues各不相同，则其对应的eigenvectors线性独立。
也就是说这些eigenvector可以视作basis。
![](Pasted%20image%2020221015191112.png)


Defective matrix:
这个概念还挺重要。
简单来说就是它线性独立的eigenvector个数小于n，或者说它的dimension of eigenspaces之和小于n。
![](Pasted%20image%2020221015192421.png)
non-defective matrix不一定有n distinct eigenvalues。
（也就是说
$n\,distinct\,eigenvalues \Rightarrow n\, linear\, independent\, eigenvectors$.
$n\, linear\, independent\, eigenvectors \nRightarrow n\,distinct\,eigenvalues$.
）
但defective matrix一定没有n distinct eigenvalues。


***
任意矩阵的$A^TA$一定是symmetric, positive semidefinite的。
当rk(A)=n时，$A^TA$一定是symmetric, positive definite的。（positive definite的部分之前证明过了）
![](Pasted%20image%2020221015194354.png)
证明：
![](Pasted%20image%2020221015194537.png)

***
Spectral Theorem
对于symmetric矩阵A，一定存在一组orthonormal 的eigenvectors作为basis，而且所有eigenvalue都为实数。
![](Pasted%20image%2020221015194913.png)

换句话说，
**symmetric矩阵的eigendecomposition一定存在。** 我们可以将A分解为$A=PDP^T$, where D is diagonal and the columns of P contain the eigenvectors.
![](Pasted%20image%2020221015231333.png)
我的理解：
$A=PDP^T$这个式子其实就是[chap2矩阵相似](All%20About%20Data%20Science/MML/chap2.md#^dbbab1)提到的basis变换$\hat{A}=S^{-1}AS$，这里要求S regular。
P的columns是orthonormal的eigenvectors，因此P是orthogonal matrix，P regular，因为P的column已经orthogonal了就一定linear independent。此外因为P是orthogonal的，有$P^T=P^{-1}$.
这时对比$A=PDP^T$和$\hat{A}=S^{-1}AS$，可以发现就是一回事。
P（eigenvectors）描述的其实相当于是一个basis change，而A和D是similar的，本质上描述的是同一个线性变换。
***
下面讨论一下eigenvalue, trace, determinant这三者之间的关系。
![](Pasted%20image%2020221015234848.png)
determinant不随basis变换而改变，A和D描述的又是同一个线性变换，那么det(A)=det(D)。
而D是diagonal矩阵，算出来的det(D)就等于上面这个式子。

![](Pasted%20image%2020221015235143.png)
trace也不随basis变换而改变，那么tr(A)=tr(D)。而tr(D)对角线上的元素就是eigenvalue，因此得到上面这个式子。


## 4.3 Cholesky Decomposition

Cholesky Decomposition可以类比于实数的开根号运算。

注意这里要求是正定矩阵，而不是半正定矩阵。
对称的正定矩阵可以分解为$A=LL^T$，其中L是一个lower triangular矩阵。
而且L唯一确定。
![](Pasted%20image%2020221022223721.png)

具体计算：
写出对应关系解方程就好，不怎么重要。
![](Pasted%20image%2020221022224001.png)
![](Pasted%20image%2020221022224049.png)

Cholesky decomposition在很多出现了symmetric positive definite matrices的场景都有应用。
- the covariance matrix of a multivariate Gaussian variable (see Section 6.5) is symmetric, positive definite. The Cholesky factorization of this covariance matrix allows us to generate samples from a Gaussian distribution
- It also allows us to perform a linear transformation of random variables, which is heavily exploited when computing gradients in deep stochastic models, such as the variational auto-encoder
- The Cholesky decomposition also allows us to compute determinants very efficiently.
- 
Cholesky decomposition可以用于求determinant。
![](Pasted%20image%2020221022224403.png)



## 4.4 Eigendecomposition and Diagonalization


diagonal matrix

![](Pasted%20image%2020221022224518.png)

determinant：product of its diagonal entries
a matrix power $D^k$：each diagonal element raised to the power k
the inverse $D^{-1}$：对角线元素取倒数
![](Pasted%20image%2020221022224557.png)

矩阵的Diagonalization是basis change we discussed in Section 2.7.2 and eigenvalues from Section 4.2的综合应用。


之前提到过如果有$D=P^{-1}AP$，则A与D similar。
现在我们只关注D是diagonal matrix、而且对角线上的元素都是A的eigenvalue的情况。


![](Pasted%20image%2020221022230646.png)



下面来证明diagonalization本质就是basis change，且选择的basis是A的eigenvectors。
![](Pasted%20image%2020221022225711.png)

上面对于diagonalization的定义，并没有提到eigenvalue和eigenvector。下面来证明**diagonalization带来的必然是eigenvalue和eigenvector**。
![](Pasted%20image%2020221022230136.png)
将$AP=PD$展开，然后令等式两边对应相等，正好能得到很多组$Ap_i=\lambda_i p_i$，因此解得的向量是eigenvector，解得的$\lambda$是eigenvalue。
![](Pasted%20image%2020221022230454.png)
这里我们要求P is invertible，也就是P is full rank，也就是说这些eigenvectors $p_i$线性独立，是一组basis。


那什么样的A才是diagonalizable的呢？
根据下面的定理，A is diagonalizable当且仅当A的eigenvectors线性独立。
![](Pasted%20image%2020221022231426.png)

eigenvectors线性独立也就是说A是non-defective的，也就是说只有non-defective才能被diagonalize。
![](Pasted%20image%2020221022231610.png)

而前面提到，根据spectum theorem，symmetric矩阵的eigenvector一定orthonormal。因此**symmetric矩阵一定能被diagonalized.**
![](Pasted%20image%2020221022231903.png)


Diagonalization的几何意义：
A和D描述的是同一个线性变换，
$e_i$和$p_i$是两组不同的basis，其中$p_i$和D搭配使用时，可以看到经过D变换之后，沿着$p_i$的方向都是只有被伸缩，而没有旋转。
$P$和$P^{-1}$都只是basis change，只起到旋转的作用。
![](Pasted%20image%2020221022232352.png)



Eigendecomposition的具体计算：
![](Pasted%20image%2020221022233202.png)
![](Pasted%20image%2020221022233224.png)

一些性质：
- 经过eigendecomposition，可以简化矩阵幂运算的计算量。![](Pasted%20image%2020221022233508.png)
- 简化determinant的计算量
![](Pasted%20image%2020221022233526.png)
![](Pasted%20image%2020221022233548.png)
（P是basis change，代表旋转，因此线性变换先后signed volume不变，determinant为1）.



## 4.5 Singular Value Decomposition

SVD很通用，因为任何时候都存在。
SVD has been referred to as the “fundamental theorem of linear algebra” (Strang, 1993) because it can be applied to all matrices, not only to square matrices, and it always exists.
而且SVD可以用来quantifies the change between the underlying geometry of these two vector spaces。
![](Pasted%20image%2020221022233902.png)


SVD Theorem
$A=U\Sigma V^T$，
其中U是`m*m`orthogonal 方阵，V是`n*n`orthogonal 方阵。
$\Sigma$是`m*n`diagonal矩阵且对角线元素非负（尽管不一定是方阵）。

![](Pasted%20image%2020221022234127.png)
其中$\Sigma$的对角线元素被称为singular values，
$u_i$称为left-singular vectors，
$v_j$称为right-singular vectors。
![](Pasted%20image%2020221022234538.png)
singular matrix $\Sigma$唯一确定。
![](Pasted%20image%2020221022234758.png)
注意$\Sigma$是`m*n`的，与A的shape相同。
由于不一定是方阵，$\Sigma$可能长这样：
![](Pasted%20image%2020221022234922.png)
或者是这样：
![](Pasted%20image%2020221022234946.png)

所有矩阵都存在SVD。
![](Pasted%20image%2020221022234955.png)


### 4.5.1 Geometric Intuitions for the SVD

we will discuss the SVD as sequential linear transformations performed on the bases.

SVD的思路跟eigendecomposition很类似。
![](Pasted%20image%2020221023000304.png)
- basis change via $V^T$.
- scaling and augmentation (or reduction) in dimensionality via the singular value matrix $\Sigma$.
- a second basis change via $U$.

![](Pasted%20image%2020221023001531.png)
![](Pasted%20image%2020221023001409.png)

SVD在V和W都有basis change，并且本身V和W就是两个不同的vector space。
相比之下，eigendicomposition是在相同的向量空间V中，并且使用使用的是同一组basis。
我的理解：
参照[矩阵等价&矩阵相似](All%20About%20Data%20Science/MML/chap2.md#^dbbab1)这里，可以看到eigendicomposition对应的是矩阵相似，A与D相似，并且在同一个向量空间、线性变换前后使用的是同一组basis。
而SVD更像是equivalent的条件设置。
![](Pasted%20image%2020221023001659.png)
![](Pasted%20image%2020221023003217.png)


### 4.5.2 Construction of the SVD
We will next discuss why the SVD exists and show how to compute it in detail.

对比eigendecompositionh和SVD的式子，
![](Pasted%20image%2020221023003359.png)
![](Pasted%20image%2020221023003406.png)
如果我们令
![](Pasted%20image%2020221023003443.png)
则SVD就是eigendecomposition。




接下来说明SVD为什么存在。

思路：
![](Pasted%20image%2020221023004611.png)


step 1：
已知symmetric矩阵一定可以diagonalize，
又知任意矩阵A的$A^TA$一定symmetric positive semidefinite。
所以$A^TA$一定可以diagonalize。
得到：
![](Pasted%20image%2020221023005107.png)

假设A的SVD存在，$A=U\Sigma V^T$，
则
![](Pasted%20image%2020221023005334.png)
对比上面两个式子，
![](Pasted%20image%2020221023005350.png)

![](Pasted%20image%2020221023005450.png)

可以得到：
对$A^TA$进行eigendecomposition，
得到的eigenvector就是A的right-singular vectors V，
得到的eigenvalue就是A的singular value$\Sigma$的平方。


step2：
同理，对$AA^T$进行相同的操作，可以得到
（注意这里$A^TA$是`n*n`的，$AA^T$是`m*m`的，）
![](Pasted%20image%2020221023005749.png)
对$AA^T$进行eigendecomposition，
得到的eigenvector就是A的left-singular vectors V，
得到的eigenvalue就是A的singular value$\Sigma$的平方。

step3：
![](Pasted%20image%2020221023010138.png)
上面两步中用到的$\Sigma$一定是相同的。
（这一步没看懂有啥用）

step4：
（这步中间有些细节不懂，就这样吧。）
现在我们借助对$A^TA$和$AA^T$的eigendecomposition，得到了A的$U$、$V^T$和$\Sigma$。如果只是只是为了求A的SVD，那我们现在已经算完了。
但我们现在另外再来探索一下U和V的关系。

首先我们观察到$Av_j$本身就是orthogonal的。
![](Pasted%20image%2020221023011909.png)

但我们想让U是orthonormal的，因此就对$Av_j$除以其norm得到单位向量：
![](Pasted%20image%2020221023012132.png)
至此得到u和Av的数值关系：
![](Pasted%20image%2020221023012243.png)

也就是
![](Pasted%20image%2020221023012342.png)
这个式子很像eigenvalue equation，但是注意等式两边的的向量是不同的。

注意上面式子中$v_i$是n维。$u_i$是m维。如果只看上面这个式子的话，感觉$v_i$和$u_i$中维度更高的那一方，只有一部分值受到了约束。
比如n=6， m=4，向量$v_i$比$u_i$长，上面的式子只展示了向量$v_i$的前4维等于啥，并没说后2维需要满足什么条件。但经过之前的SVD推导过程，我们知道$v_i$本身还需要是orthonormal的。

![](Pasted%20image%2020221023013317.png)

### 4.5.3 Eigenvalue Decomposition vs. Singular Value Decomposition
Eigenvalue Decomposition和SVD的对比：

- SVD始终存在。eigendecomposition要求矩阵是方阵，且eigenvectors是n维空间的basis。
- eigendecomposition的eigenvector不一定orthogonal。SVD的U和V中的向量一定是orthonormal的，因此描述的一定是旋转操作。
- SVD的U和V之间一般没有inverse关系（而且可能连shape都不一样）。eigendecomposition的$p$和$P^{-1}$成inverse关系。
- SVD的$\Sigma$一定是实数且非负。eigendecomposition中的eigenvalue不一定。
- 对于symmetric矩阵，eigendecomposition和SVD相同。
![](Pasted%20image%2020221023013912.png)
![](Pasted%20image%2020221023013920.png)


SVD相关的一些terminology and conventions：

![](Pasted%20image%2020221023015014.png)
SVD的另一种定义方法：
$\Sigma$是方阵，U和V是矩形。
![](Pasted%20image%2020221023015107.png)
![](Pasted%20image%2020221023015224.png)


![](Pasted%20image%2020221023015332.png)


## 4.6 Matrix Approximation

这部分等哪天有心情了再看，先看chap5


