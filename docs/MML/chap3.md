# chap3 Analytic Geometry
![](Pasted%20image%2020221013190749.png)



## 3.1 Norms
norm 是一个将向量映射为实数的函数。
norm需要满足可乘、三角不等式和非负这三个性质。
![](Pasted%20image%2020221013191557.png)

![](Pasted%20image%2020221013193045.png)
![](Pasted%20image%2020221013193051.png)


### 3.2 Inner Products

### 3.2.1 Dot Product
dot product只是inner product的一种形式。
![](Pasted%20image%2020221013193334.png)

### 3.2.2 General Inner Products

之前提到的mapping都只有一个argument，
接下来用到的映射$\Omega$ 有两个argument，并且对每个argument都是linear的，称为bilinear mapping。
![](Pasted%20image%2020221013193855.png)

假设$\Omega$ 是一个$V\times V \to R$ 的bilinear mapping，将同一个向量空间中的两个向量映射为一个实数，
当两个argument的顺序不重要的时候，称$\Omega$ 是symmetric的。
如果$\Omega$满足：只要x不是0向量，$\Omega(x,x)$就一定大于0；$\Omega(0,0)=0$. 就称$\Omega$是**positive definite**的。

![](Pasted%20image%2020221013194403.png)

Inner product 的定义：
A positive definite, symmetric bilinear mapping $\Omega$ 被称为V上的inner product。（正定+对称）.
![](Pasted%20image%2020221013195140.png)

V与inner product的组合称为inner product space。
当使用dot product作为inner product时，称V为Euclidean vector space。
![](Pasted%20image%2020221013195308.png)




### 3.2.3 Symmetric, Positive Definite Matrices

机器学习中，Symmetric, positive definite matrices很重要，而它们是通过inner product来定义的。

假设向量空间V带有一个inner product运算$\langle \cdot,\cdot\rangle$ ，并且有一组ordered basis B。
现有V中的两个向量x和y，则可以用B的线性组合表示这两个向量。
![](Pasted%20image%2020221013200835.png)
![](Pasted%20image%2020221013200902.png)
![](Pasted%20image%2020221013201008.png)
通过上面可以看出，
![](Pasted%20image%2020221013210527.png)
$\hat{x}$和$\hat{y}$是向量x和y相对于B的坐标，也就是说inner product本质上其实是A决定的。

The symmetry of the inner product also means that A is symmetric. 
innner product的对称决定了A也是对称的。

然后，因为inner product的定义
![](Pasted%20image%2020221013212554.png)
可以直接得出下面的式子：
$\forall x \in V\setminus \{0\}: x^TAx>0$.  
![](Pasted%20image%2020221013210819.png)



Symmetric, Positive Definite Matrix
就是通过上面这个式子定义的：
对于V中任意非0向量x都有$x^TAx>0$. 且如果A对称，
则A被称为Symmetric, Positive Definite.
![](Pasted%20image%2020221013213618.png)
也就是说，正定矩阵其实反映的是这个向量空间中所定义的inner product运算的性质。

![](Pasted%20image%2020221013213939.png)


如果已知A是一个symmetric, positive definite矩阵了（对称，对非零向量$x^TAx>0$），则$\hat{x}^TA\hat{y}$就是V上的一个inner product运算。
![](Pasted%20image%2020221013214321.png)

某一个映射是inner product，当且仅当
![](Pasted%20image%2020221013214552.png)
这个映射操作可以被表示为$\hat{x}^TA\hat{y}$且A为symmetric, positive definite矩阵。

一些推论：
![](Pasted%20image%2020221013215021.png)

A的对角线元素都>0.
因为令x为standard basis时，$x^TAx>0$其实就约束着A的对角线上的元素>0.
![](Pasted%20image%2020221013215116.png)


### 3.3 Lengths and Distances

所有inner product运算开根号之后都满足norm运算的要求，也就是说所有inner product开根号之后都是一个norm运算。
![](Pasted%20image%2020221013222228.png)
但是并不是所有norm都有对应的inner product，比如L1-norm。

定义distance：
![](Pasted%20image%2020221013222918.png)
![](Pasted%20image%2020221013222930.png)
![](Pasted%20image%2020221013223608.png)



尽管 inner product与metric都是将两个向量映射为一个实数，而且都是symmetric, positive definite的，也都满足三角不等式。
但是，它们的behavior正好相反
![](Pasted%20image%2020221013223546.png)

## 3.4 Angles and Orthogonality


inner products also capture the geometry of a vector space by defining the angle ω between two vectors.

由于柯西不等式：
![](Pasted%20image%2020221014004346.png)
因此有
![](Pasted%20image%2020221014004443.png)
由此就可以定义两个向量之间的夹角angle。
![](Pasted%20image%2020221014011909.png)


正交：
两个向量inner product为0称为orthogonal，
正交+两个向量都是单位向量，称为orthonormal。
![](Pasted%20image%2020221014133616.png)

下面这个例子中，两个向量在使用dot product时正交，
但使用
![](Pasted%20image%2020221014134138.png)
另一个inner product时不正交。
![](Pasted%20image%2020221014134050.png)
![](Pasted%20image%2020221014134058.png)


Orthogonal Matrix：
![](Pasted%20image%2020221014134436.png)
当矩阵的每一列都orthonormal的（注意不是orthogonal），向量与自己作inner product得到1，其余得到0。因此
![](Pasted%20image%2020221014134618.png)
然后就可以直接推出
![](Pasted%20image%2020221014134631.png)

注意：
根据上面的定义，虽然我们叫A“orthogonal matrix”，但其实A是“orthonormal”的，这只是命名的问题，有点怪。
注意以后提到“orthogonal matrix”，就隐含了每个column都是orthonormal的，是单位矩阵。

由下面这个式子可以看出，某一向量x在经过一个orthogonal matrix的变换之后，其norm不会改变，本质上是因为$A^TA=I$。
![](Pasted%20image%2020221014135014.png)


此外，由下面这个式子可以看出，对x、y同时施加orthogonal matrix的变换，变换前后两个向量的夹角也不会改变。
![](Pasted%20image%2020221014135416.png)

其实orthogonal matrix对应的是旋转变换（with the possibility of flips）.


## 3.5 Orthonormal Basis

之前我们提到n维向量空间需要n个线性独立的向量（basis），
现在我们增加两个约束，
basis正交，basis为单位向量（也就是说是orthonormal basis），看看能得出什么性质。

![](Pasted%20image%2020221014141543.png)

我们可以通过高斯消元法得到一个span的orthonormal basis，
使用的augmented matrix为$\hat{B}\hat{B}^T|\hat{B}$.
直观上也不难理解，我们用高斯消元法求得一个$\hat{B}^T$，而由于$\hat{B}^T$所处的位置，需要$\hat{B}\hat{B}^T=\hat{B}$，
也就是说$\hat{B}^T=\hat{B}^{-1}$，因此高斯消元得到的矩阵是正交矩阵，得到的向量是orthonormal basis。
![](Pasted%20image%2020221014141826.png)

## 3.6 Orthogonal Complement

Having defined orthogonality, we will now look at vector spaces that are orthogonal to each other. 


假设V是D维向量空间，U是V的M维子空间，
则V中那些与U中所有向量都正交的向量组成的集合（准确的说是子空间），就是U的orthogonal complement $U^\perp$.
并且它的维度是D-M。
![](Pasted%20image%2020221014142834.png)

并且V中的任意向量都可以用$U$的basis和$U^\perp$的basis（的并集）的线性组合来表示。
![](Pasted%20image%2020221014143647.png)

假如在三维空间中，U是一个二维平面，则与U正交的单位向量ω就是U的orthonormal complement这个子空间的basis。我们称ω是U的法向量（normal vector）。
![](Pasted%20image%2020221014144021.png)

### 3.7 Inner Product of Functions

In the following, we will look at an example of inner products of a different type of vectors: inner products of functions.
我们将向量推广到函数，来讨论function之间的inner product和function之间的orthogonal。

We can think of a vector $x\in \mathbb{R}^n$  as a function with n function values.
然后，将有限维向量推广到无限维，将离散的一个个argument推广到一个连续的定义域，
这样一来原本的inner product需要的求和就可以推广到求定积分。

![](Pasted%20image%2020221014144947.png)
![](Pasted%20image%2020221014145007.png)

当定积分结果为0，则称这两个函数为orthogonal.
与有限维向量求inner product不同，对两个函数求inner product求积分，结果有可能diverge（无穷大），因此这里需要数学上有其他要求。但本书不涉及这些。
![](Pasted%20image%2020221014145659.png)

![](Pasted%20image%2020221014145920.png)

## 3.8 Orthogonal Projections

we can project the original high-dimensional data onto a lower-dimensional feature space and work in this lower-dimensional space to learn more about the dataset and extract relevant patterns


projection的定义：
有点难理解，如果对V进行两次变换的结果，和进行一个变换的结果相同，则这个变换称为投影。
就比如将三维空间中的一些向量投影到xy平面上，投影一次的结果，和“先投影一次，再投影一次”的结果是相同的。
![](Pasted%20image%2020221014151529.png)

线性变换都可以用矩阵描述，
描述投影操作的矩阵称为projection matrices。
需要满足$P_\pi^2=P_\pi$.
![](Pasted%20image%2020221014151937.png)

下面我们讨论将inner product spaces投影到subspace中。
其中inner product我们默认全部使用dot product。
### 3.8.1 Projection onto One-Dimensional Subspaces (Lines)


首先我们有一条过原点的直线，basis为b。这条直线可以看作是basis b张成的1-dim subspace U。
当我们将向量x投影到U时，其实就是寻找投影到U之后的向量$\pi_U(x)\in U$中，哪一个距离x最近。
![](Pasted%20image%2020221014152622.png)



![](Pasted%20image%2020221014154935.png)
然后：
![](Pasted%20image%2020221014155246.png)

![](Pasted%20image%2020221014155614.png)
![](Pasted%20image%2020221014155856.png)

从上面的式子可以直接观察出$P_\pi$是symmetric的。

$P_\pi$可以将任意向量投影到b所张成的一维子空间（直线）上。

注意，投影的结果仍然是n维向量，而不是标量。
但是，我们其实可以相对于basis b直接用$\lambda$来表达投影后的结果。
![](Pasted%20image%2020221014160120.png)
![](Pasted%20image%2020221014160321.png)

chap4将会讲到，$\pi_U(x)$其实是矩阵$P_\pi$的一个特征向量，其对应的特征值是1.
![](Pasted%20image%2020221014160335.png)



### 3.8.2 Projection onto General Subspaces

下面我们从投影到1维直线U推广到投影到m维空间U。
假设U有一组ordered basis $B=(b_1,b_2,...,b_m)$，则将x投影到U中之后可以用B来uniquely得到$\pi_U(x)$的坐标：
![](Pasted%20image%2020221014172206.png)
![](Pasted%20image%2020221014172227.png)

然后我们想让$\pi_U(x)$与$x$越近越好，
这样一来就可以推出$\pi_U(x)-x$需要与B正交。
![](Pasted%20image%2020221014173105.png)

写成矩阵的形式就得到了下面的式子：
![](Pasted%20image%2020221014173327.png)
![](Pasted%20image%2020221014173602.png)
所以normal equation其实表示的就是投影操作。

并且因为$B=(b_1,b_2,...,b_m)$本身是basis，column一定线性独立，因此$B^TB$一定是可逆的。[证明](All%20About%20Data%20Science/MML/chap2.md#^401219)
因此可以解出投影之后向量的坐标$\lambda$。
![](Pasted%20image%2020221014185540.png)


![](Pasted%20image%2020221014185922.png)
![](Pasted%20image%2020221014185928.png)

有了$\lambda$，我们就可以表示出$\pi_U(x)$，最终表示出projection matrix $P_\pi$。
![](Pasted%20image%2020221014190057.png)

![](Pasted%20image%2020221014190435.png)


当Ax=b无解时，我们可以借助投影得到一个近似解。
Ax=b无解，说明b不位于A的column的span中。
这时我们可以借助投影找到A的span中离b最近的一个，也就是将b投影到A的span中。
如果这里的inner product使用的时dot product，则这个求解过程其实就是最小二乘法。
![](Pasted%20image%2020221014191327.png)

当B是Orthonormal Basis时，上面得到的$\lambda$的式子可以进一步简化。
![](Pasted%20image%2020221014191854.png)


### 3.8.3 Gram-Schmidt Orthogonalization

### 3.8.4 Projection onto Affine Subspaces
假设有affine space $L = x_0+U$，而我们想要把x投影到L上。
于是我们将其转化为把$x-x_0$投影到$L-x_0=U$的任务，这样一来就成了投影到向量空间的任务。投影结束之后，再加上$x_0$将结果翻译回L。
![](Pasted%20image%2020221014204414.png)

而这个转换的过程并不会影响x到L的距离。
![](Pasted%20image%2020221014204444.png)



## 3.9 Rotations

A rotation is a linear mapping (more specifically, an automorphism of rotation a Euclidean vector space) that rotates a plane by an angle θ about the origin。

### 3.9.3 Rotations in n Dimensions
三维空间中的旋转：固定住一个轴不动，将垂直于这个固定轴的平面进行旋转。
推广到n维空间就是固定住n-2维不动。
![](Pasted%20image%2020221014205028.png)
![](Pasted%20image%2020221014205408.png)
![](Pasted%20image%2020221014205413.png)





























