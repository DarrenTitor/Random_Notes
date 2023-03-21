# chap2: Linear Algebra

除了通常意义上的几何意义上的vector，多项式也可以看作是vector：两个多项式相加可以得到多项式、多项式与标量相乘可以得到多项式。因为满足这两个条件，因此可以视作vector。
![](Pasted%20image%2020221009180204.png)
我们更关心的其实是n维实数空间上的vector，比几何意义上的vector更抽象一点：
![](Pasted%20image%2020221009180556.png)

In this book, we will focus on finitedimensional vector spaces, in which case there is a 1:1 correspondence between any kind of vector and $\mathbb{R}^n$.

在数学中我们关心closure这个概念，What is the set of all things that can result from my proposed operations? 
In the case of vectors: What is the set of vectors that can result by starting with a small set of vectors, and adding them to each other and scaling them? 
这样我们就可以得到**vector space**。


![](Pasted%20image%2020221009181108.png)

## 2.1 Systems of Linear Equations

通常，我们将
![](Pasted%20image%2020221010192837.png)
写作
![](Pasted%20image%2020221010192913.png)
这样的，矩阵相乘的形式。然后求解向量X。

In general, for a real-valued system of linear equations we obtain either
**no, exactly one, or infinitely many** solutions.


## 2.2 Matrices

![](Pasted%20image%2020221010193452.png)
![](Pasted%20image%2020221010193438.png)

矩阵的element-wise product称为hadamard product。
![](Pasted%20image%2020221010193641.png)

### 2.2.2 Inverse and Transpose
Unfortunately, not every matrix A possesses an inverse A−1. If this inverse does exist, A is called **regular/invertible/nonsingular**, otherwise singular/noninvertible.
![](Pasted%20image%2020221010194011.png)

当determinant=0时，矩阵不可逆。

一些重要性质：
![](Pasted%20image%2020221010194938.png)
![](Pasted%20image%2020221010194529.png)

对称矩阵的sum一定是对称的，但product不一定对称。
![](Pasted%20image%2020221010194615.png)

### 2.2.3 Multiplication by a Scalar

![](Pasted%20image%2020221010194801.png)


### 2.2.4 Compact Representations of Systems of Linear Equations

线性方程组
![](Pasted%20image%2020221010195118.png)
可以写作：

![](Pasted%20image%2020221010195151.png)
注意每个x所对应的是column，不是row。
统称为$Ax=b$。
$Ax$ is a (linear) combination of the columns of $A$.

## 2.3 Solving Systems of Linear Equations


### 2.3.1 Particular and General Solution
![](Pasted%20image%2020221010200623.png)
![](Pasted%20image%2020221010201438.png)
![](Pasted%20image%2020221010201458.png)


### 2.3.2 Elementary Transformations
elementary transformations在不改变solution的同时，可以对方程组进行简化。

![](Pasted%20image%2020221010201738.png)

![](Pasted%20image%2020221010205748.png)
row-echelon form的要求：
- 全0的row放到最下面
- 阶梯状矩阵，每一行从左起的第一个非0元素必须严格在上一行第一个非0元素的右边。
- 
![](Pasted%20image%2020221010202925.png)
首先把方程组写成augmented matrix。
然后转换成row-echelon form
![](Pasted%20image%2020221010202917.png)
![](Pasted%20image%2020221010203028.png)
![](Pasted%20image%2020221010204949.png)


![](Pasted%20image%2020221010204945.png)
![](Pasted%20image%2020221010205056.png)


借助row-echelon form可以帮助我们找到特解。
因为上面例子中的pivot为x1, x3, x4，
因此令augmented matrix中的这三列的线性组合等于向量b。

![](Pasted%20image%2020221010205122.png)
然后把free variable设为0，就得到了一个特解。
![](Pasted%20image%2020221010205359.png)


Reduced Row Echelon Form：
![](Pasted%20image%2020221010205535.png)
Reduced Row Echelon Form的要求：
- 全0的row放到最下面
- 严格呈阶梯状
- pivot全为1
- 每一**列**除了pivot，其余全为0



reduced row-echelon form可以帮助我们找通解。
![](Pasted%20image%2020221010205550.png)

Gaussian elimination就是一个将augmented matrix转化为reduced row-echelon form的算法。
![](Pasted%20image%2020221010223636.png)

![](Pasted%20image%2020221010224025.png)
![](Pasted%20image%2020221010224032.png)

### 2.3.3 The Minus-1 Trick
![](Pasted%20image%2020221010224234.png)

这是一个可以快速求出Ax=0的方法。（求出的结果记得要再加上特解，才是最终的通解。）

适用于reduced row-echelon form没有全0row的情况，
A为转换后得到的reduced row-echelon form，通常col数大于row数，
这时我们在A的**各处**增加若干行，将A补全成一个方阵。（不一定加在A的底部，可以插入到A的中间）
![](Pasted%20image%2020221010225143.png)
**使得方阵的对角线contains eather 1 or -1.**
![](Pasted%20image%2020221010225201.png)

![](Pasted%20image%2020221010225402.png)
![](Pasted%20image%2020221010225408.png)
此时，pivot -1所在的columns就是Ax=0的solution。
实际上这些column组成了Ax=0的解空间的basis，称为kernel或者null space。
例如：
![](Pasted%20image%2020221010225632.png)

矩阵求逆的算法：
之前求解线性方程组我们是求解$AX=b$中的X，
现在如果想算A的逆，其实就是求解$AX=I_n$.
这样一来我们只需要将$[A|I_n]$视作augmented matrix，然后将其转换为reduced row-echelon form。
![](Pasted%20image%2020221010230105.png)
这样一来，augmented matrix右边的部分就是我们想要的X，也就是$A^{-1}$。
![](Pasted%20image%2020221010230209.png)




### 2.3.4 Algorithms for Solving a System of Linear Equations
在之前的讨论中，我们都假设$Ax=b$有解。当方程组无解的时候，我们只能求其近似解，其中一种方式就是linear regression。


在某些特殊情况下，如果我们能找到$A^{-1}$，就能直接找到$Ax=b$的解，$x=A^{-1}b$。
However, this is only possible if A is a square matrix and invertible, which is often not the case.


在更多情况下，我们一般引入假设：
A has linearly independent columns，或者rank of A is identical to its column rank，或者A的行列式不等于0，

则此时$A^T\cdot A$可逆

![](Pasted%20image%2020221010231611.png)
这样得到的$x=(A^TA)^{-1}A^T$称为Moore-Penrose pseudo-inverse，which also corresponds to the minimum norm least-squares solution. （这也是最小二乘法的解）。

***

#### 证明$rk(A)=n\iff A^TA \, is \, invertible$.

^401219

证明$rk(A)=n\iff A^TA \, is \, invertible$.
（下面的证明需要用到chap2.7.3中的定理）
![](Pasted%20image%2020221014183337.png)
上面证明了$rk(A)=rk(A^TA)$，
接下来只需证明$rk(A^TA)=n\iff A^TA \, is \, invertible$。
也就是只需证明为什么 矩阵full rank <=>矩阵可逆。
这是chap2.6.2中的一个推论。
所以证明结束。

另外用上图中的思路其实可以证明：
$rk(A)=rk(A^T)=rk(A^TA)=rk(AA^T)$.
这是一个挺常用的结论。
就不证明了，了解思路就行。
***

disadvantage:
- 计算量大
- for reasons of numerical precision it is generally not recommended to compute the inverse or pseudo-inverse

其他求解线性方程组的方法：
- 高斯消元法尽管很重要，但对于变量个数巨大的场景不适用。
- In practice, systems of many linear equations are solved indirectly


## 2.4 Vector Spaces
这一部分，将借助group来正式定义vector。


### 2.4.1 Groups
![](Pasted%20image%2020221010234012.png)
group需要满足的条件：
- 是operation的closure
- operation满足结合律
- 有单位元
- 有逆元

如果group另外还满足交换律，则为**Abelian group**。
![](Pasted%20image%2020221010234311.png)

![](Pasted%20image%2020221010234520.png)

### 2.4.2 Vector Spaces

用group来定义vector space：
![](Pasted%20image%2020221010235642.png)

![](Pasted%20image%2020221010235935.png)
注意：
向量、矩阵和虚数的集合都是vector space，因为满足上面的性质。


### 2.4.3 Vector Subspaces

Intuitively, they are sets contained in the original vector space with the property that when we perform vector space operations on elements within this subspace, we will never leave it. In this sense, they are “closed”.

![](Pasted%20image%2020221012144528.png)
通常我们需要判断：
- 零元在子空间内
- 所有运算都在子空间内满足闭包

一些结论：
![](Pasted%20image%2020221012144931.png)
![](Pasted%20image%2020221012144959.png)

## 2.5 Linear Independence

现在有finite个vector，则它们的线性组合定义为：
向量0永远可以写成一组vector的线性组合，但这是trivial线性组合。
通常我们更关心线性组合系数不全为0的情况。
![](Pasted%20image%2020221012145251.png)

线性独立、线性相关是借助$0=\sum^k_{i=1}\lambda_ix_i$来定义的。
当这个式子只有trivial解，此时这些向量线性独立。
如果存在non-trivial解，则这些向量线性相关。
![](Pasted%20image%2020221012150110.png)

当set中有0向量、有两个相等向量、某一向量可以用其他向量表示时，整个set线性相关。
![](Pasted%20image%2020221012150810.png)

**实际判断线性相关通常使用Gaussian Elimination**：
首先我们得到row echelon form (the reduced row-echelon form is unnecessary here):
所有pivot column都与其**左边**的column线性独立。
所有non-pivot column都可以表示为其左边的pivot column的线性组合。
![](Pasted%20image%2020221012151207.png)


**All column vectors are linearly independent if and only if all columns are pivot columns.**
![](Pasted%20image%2020221012151341.png)
注意这里说的是所有column都是pivot column，我们不关心row。
![](Pasted%20image%2020221012152050.png)
比如说我们得到了这个row echelon form，就能得出结论这3个向量线性独立。


假设我们现在有k个线性独立的向量$b_i$，用这些向量组合出了m个新的向量（线性组合）$x_i$，则我可以把整个过程简写为矩阵乘法的形式：
![](Pasted%20image%2020221012152726.png)


然后就引出一个问题，得到的这一组新向量$x_i$是不是也是线性独立的？
还是沿用至之前的方法，我们令x的线性组合等于零向量，
![](Pasted%20image%2020221012153210.png)
然后看有没有non-trivial解。
![](Pasted%20image%2020221012153259.png)
然后发现，当且仅当对b进行线性组合的系数向量$\{\lambda_1,\lambda_2,...,\lambda_m\}$线性独立时，得到的这一组x才线性独立。

如果用k个线性独立向量表示m>k个新向量，则新向量有冗余，必定线性相关。
![](Pasted%20image%2020221012153537.png)

例子：
注意下面的$b_i$都是向量，而非标量。
![](Pasted%20image%2020221012153833.png)
![](Pasted%20image%2020221012153945.png)

## 2.6 Basis and Rank

如果一个vector space中的所有向量都可以用集合A中的向量线性组合得到，称A为V的generating set，V为A的span。
![](Pasted%20image%2020221012154656.png)

线性独立的generating set最小，称为这个向量空间V的一个basis。
![](Pasted%20image%2020221012164824.png)

注意：一组线性独立的向量不一定是basis。
![](Pasted%20image%2020221012165204.png)

对于finite-dimensional vector spaces V，V的维度就等于V的basis vector的个数。
![](Pasted%20image%2020221012175432.png)
（注意：vector space的维度是通过basis定义的。）
**区分vector的维度和vector space的维度。**
假如有一组向量是3维的，但它们只能张成一个2维平面，则这个张成空间是2维而不是3维。

![](Pasted%20image%2020221012165830.png)

如何计算一个subspace的basis：
![](Pasted%20image%2020221012165914.png)
将spanning vector写成矩阵A，A的echelon form pivot对应的column就是这个subspace的basis。


### 2.6.2 Rank

^e436c7

rank其实就是一个矩阵的linearly independent columns的个数。
![](Pasted%20image%2020221012170324.png)

![](Pasted%20image%2020221012170921.png)

对于`n*n` 的方阵，矩阵可逆当且仅当rank=n。
![](Pasted%20image%2020221012171710.png)

Ax=b有解当且仅当$rk(A)=rk(A|b)$.
![](Pasted%20image%2020221012171722.png)

![](Pasted%20image%2020221012172037.png)

![](Pasted%20image%2020221012172126.png)


## 2.7 Linear Mappings

现在有两个vector space V和W。我们想让V->W的映射可以保持向量所具有的“可以相加，可以标量乘法”的性质，满足这样条件的映射称为
linear mapping 或者 linear transformation。
![](Pasted%20image%2020221012172719.png)


矩阵本身就是一种线性变换。此外，之前提到矩阵还可以表示向量的集合。
可见矩阵有两种理解方式。
When working with matrices, we have to keep in mind what the matrix represents: a linear mapping or a collection of vectors. 




![](Pasted%20image%2020221012173946.png)
Injective: 不存在V中多个元素映射到W中同一元素的情况。
Surjective: W中所有元素都能通过V中的某一元素映射得到。

一旦一个映射满足Bijective，则这个映射存在对应的逆映射。
![](Pasted%20image%2020221012174336.png)


如果只考虑线性映射，则可以得到几个新概念：
![](Pasted%20image%2020221012174555.png)



![](Pasted%20image%2020221012175136.png)
对于两个finite-dimensional vector space V和W，
V and W are isomorphic if and only if dim(V) = dim(W).

V和W同构（线性映射+双射）<=> dim(V) = dim(W).

Intuitively, this means **that vector spaces of the same dimension are kind of the same thing**, as they can be transformed into each other without incurring any loss.
![](Pasted%20image%2020221012175920.png)


同时由上面这个定理，我们发现mxn的矩阵和mn维的向量其实是同构的，因为它们之间可以通过线性双射的映射互相转换。

一些性质：
![](Pasted%20image%2020221012180224.png)
![](Pasted%20image%2020221012180302.png)

### 2.7.1 Matrix Representation of Linear Mappings
接下来的讨论中，basis之间的顺序很重要，
B为向量空间V的n个basis组成的n-tuple，
称为V的ordered basis。
![](Pasted%20image%2020221012180614.png)

注意区分下面这3种notation。
![](Pasted%20image%2020221012180830.png)


因为V中的任意元素x都可以被basis唯一表示，而B又是basis的有序tuple，则对于每个x我们都可以用一系列系数组成的vector表示。
定义这个vector就是the coordinate vector/**coordinate** representation of x with respect to the ordered basis B.



Transformation Matrix
我们将V的每一个basis b 的**映射后的向量**$\Phi(b_j)$都表示为W的basis c的线性组合。
（注意不是用c表示b，而是用c表示映射之后的$\Phi(b_j)$。）
**换句话说，$b_j$是起点，是自变量。$\Phi(b_j)$是终点，是因变量。而我们想要用某种形式把这个映射过程记录下来，因此就记录下$\Phi(b_j)$相对于basis c的坐标。这样一来，我们就将整个映射的过程“翻译”成了相对于basis c的坐标，记录到矩阵A中。**
然后把对应的系数存到一个矩阵A中，这个A就记录着V->W这个映射的信息。
因为
![](Pasted%20image%2020221012183152.png)


因为我们是用c来表示b，表示所用的系数a，就是线性变换之后b的坐标。
**也就是说$\Phi(b_j)$的坐标就是A的第j列的向量。**
![](Pasted%20image%2020221012184737.png)


这个矩阵A可以用来把一个向量x相对于B的坐标“翻译”成相对于C的坐标。
![](Pasted%20image%2020221012190116.png)

![](Pasted%20image%2020221012190316.png)

### 2.7.2 Basis Change

接下来，来分析basis的选择对于变换矩阵A的影响。
![](Pasted%20image%2020221012204124.png)

![](Pasted%20image%2020221012205853.png)
![](Pasted%20image%2020221012210822.png)
证明就略过了，脚标太复杂，看不懂。
直观理解：
![](Pasted%20image%2020221012210932.png)

![](Pasted%20image%2020221012211819.png)
![](Pasted%20image%2020221012212124.png)
![](Pasted%20image%2020221012212412.png)

#### 矩阵等价、矩阵相似

^dbbab1

Equivalent：
上面的例子中的整个过程，其实就是在讲$A$和$\tilde{A}$其实是同一个线性变换。因此将其称之为equivalent。
![](Pasted%20image%2020221012212843.png)

Similar：
注意下面定义中$A$和$\tilde{A}$都是`n*n`方阵，为啥呢？
![](Pasted%20image%2020221012214104.png)
因为similar是equivalent的特殊情况。
equivalent的场景是V->W，
而similar是V->V，所以$A$和$\tilde{A}$都是V向V这个向量空间自己的映射。
而为什么S=T？

![](Pasted%20image%2020221012214900.png)

根据 [So we are only choosing _one_ basis for both sides. This restricts our freedom of action, but also preserves more properties of the matrix A.](https://math.stackexchange.com/a/2864164)，
上图其实画的不对，应该是下图这样。
V->V并不能得出S=T，
之所以S=T，是因为我们在矩阵相似的场景中引入了更强的假设：
linear transformation前后使用相同的basis，而不是两组不同的basis（B和C）。
这样一来自然S是同一个S。
![](Pasted%20image%2020221012222357.png)
根据上面的帖子，引入新的假设——
"restricts our freedom of action, but also preserves more properties of the matrix A. Where 矩阵相似只能推出 $rk(A)=rk(\tilde{A})$, now we get $det(A)=det(\tilde{A}$), $trace(A)=trace(\tilde{A}$) and the Eigenvalues of $A$ and $\tilde{A}$ coincide. "
虽然设定更严格，但同时带来了更多的性质。


![](Pasted%20image%2020221012223831.png)
![](Pasted%20image%2020221012223840.png)


### 2.7.3 Image and Kernel

假设有一个线性映射$\Phi:V\to W$，V和W为向量空间。
$ker(\Phi)$是V中被映射到W的零元的向量的集合。
$Im(\Phi)$是W中，所有可以经过映射得到的元素的集合 。
![](Pasted%20image%2020221012230351.png)
![](Pasted%20image%2020221012230403.png)
性质：
![](Pasted%20image%2020221012231010.png)

假设W是V经过线性变换A得到的，那么这个映射的image就是A的columns的span，称为column space。
![](Pasted%20image%2020221012232323.png)

![](Pasted%20image%2020221012232435.png)

kernel/null space对应Ax=0的通解。
![](Pasted%20image%2020221012232509.png)

kernel和image的dimention之和等于V的dimention。
![](Pasted%20image%2020221012232724.png)
上面这个定理的推论，懒得看了：
![](Pasted%20image%2020221012232848.png)



## 2.8 Affine Spaces

接下来来讨论一些不经过origin的空间。因为不包含零元，这些空间不再是向量空间。
![](Pasted%20image%2020221013100617.png)

ML语境下很多时候linear和affine是混用的。
![](Pasted%20image%2020221013100744.png)

### 2.8.1 Affine Subspaces

Affine Subspace：
V是向量空间，U是V的子空间。$x_0$是V的一个向量。
则$x_0$加上U中任一向量得到的结果的集合就称为affine subspace或者linear manifold。
U称为direction或direction space，
$x_0$称为support point。
在后面的章节称这样的subspace为hyerplane。
![](Pasted%20image%2020221013104115.png)
其实相当于把子空间U中的所有元素都平移了$x_0$。
U本身一定包含零元，但平移之后可能不包含。因此affine subspace不一定是vector space。
Examples of affine subspaces are points, lines, and planes in R3, which do not (necessarily) go through the origin.


下面我们来定义parameters：
（parameter是借助affine space定义的。）
假设有一个k-dim affine space $L=x_0+U$，我们可以用**U**的**ordered** basis $(b_1,b_2,...,b_k)$来uniquely表示这个affine space
![](Pasted%20image%2020221013105401.png)
这种表示称为parametric equation of L，所用到的系数称为parameters.


line，plane，hyperspace都是借助affine subspaces定义的：
![](Pasted%20image%2020221013111828.png)
![](Pasted%20image%2020221013111837.png)
plane：
U是两个线性独立的basis所张成的span，然后$x_0$与U相加得到的affine subspace就是一个平面。

![](Pasted%20image%2020221013112229.png)


Inhomogeneous systems of linear equations Ax=b对应的是一个维度为n-rk(A)的affine subspace，
homogeneous equation systems Ax = 0对应的是vector subspace。
![](Pasted%20image%2020221013114059.png)


### 2.8.2 Affine Mappings

![](Pasted%20image%2020221013190116.png)
![](Pasted%20image%2020221013190235.png)

affine mapping可以看作是线性映射$V\to W$和translation $W\to W$的组合。
![](Pasted%20image%2020221013190255.png)

affine mapping 组合之后还是affine mapping.
![](Pasted%20image%2020221013190416.png)
![](Pasted%20image%2020221013190458.png)




