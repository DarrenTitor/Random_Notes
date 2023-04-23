
首先是task设定，我们已有变长seq “data/reference”，然后有一个new-coming 变长seq “observation”。
现在我们想计算这两个变长seq的距离（或者泛化地讲，likelihood$P(O|\lambda)$，$\lambda$表示model）。

一种方法是，用model来对data进行概括，说白了就是使其变成定长seq，这样一来就可以和observation进行比较。
- distance based: mean (1 centroid), k-means
- prob based: gaussian, GMM
其中mean和gaussian对应，k-mean和GMM对应，分别代表使用1个vector和k个vector的情况。
![](Pasted%20image%2020230328234337.png)
这样做的缺点是，我们将model逐个和observation $o_i$比较，也就是假设$o_i$之间是independent的，考虑到speech、text这些seq的特性，这么做不好。（人的reaction time是100ms，speech的frame是~10ms，因此frame之间高度dependent。）

HMM的目的：
- break the $o_i$ independence假设
- break the idea of “将所有known只用一个model表示”

***

因为现在我们想让data也是变长seq，因此就存在一个alignment的问题。
![](Pasted%20image%2020230329002227.png)

不同的alignment可以用不同的path表示。
在每个step计算data和observation的distance，对不同的path计算sum of distance，然后挑一个distance sum最小的当作最优的alignment。
可能的path数量很大，这里根据task的特性一般有两个约束：monotonic、不能skip。
然后这个alignment的过程称为“dynamic time warping”。
上面讲的是宏观上的reference pattern，但我们还得想办法对data进行表示。因为要的是变长seq，之前的GMM之类的就不能用了，需要找一个新的可以用的model。而选的这个model就是状态机。


***

![](Pasted%20image%2020230329005149.png)
**数学上，我们可以将state machine看作一个number seq generator。**
我们可以记录下每个触发状态转移的event，也可以记录一下都经历了哪些state，都行。

简单来说我们要用一个probabilistic state machine。我们将state machine的event改成prob的形式。
![](Pasted%20image%2020230329005508.png)
这个probablistic state machine就叫markov chain。

下面把这些写成式子：
![](Pasted%20image%2020230329010530.png)
在状态机中的每个state，我们只关心outward prob（②能去哪），而不关心是从什么state来到②的。换句话说每个state的prob只condition on前一个state。所以就有了$P(x)=\prod_{t} P\left(x_{t} \mid x_{t-1}\right)$.

另外，对比之前的GMM之类的model，因为之前那些假设$o_i$之间independent，也就是只关心$P(x)=\prod_{t} P\left(x_{t}\right)$，并没有condition on过去的state。现在用了state machine（markov chain），多考虑了一个state。

***
Gaussian和GMM是generative model，可以individually生成样本，但不能生成seq。
现在我们要将state machine的思想和gaussian结合起来，生成seq。

令state machine中的每一个state对应一个gaussian，有各自的参数（μ和σ），然后用state machine决定不同gaussianzhijiande转移。
![](Pasted%20image%2020230423233652.png)
当gaussian是multi-variate gaussian时，每个gaussian可以生成vector。由此，我们将markov model变成HMM，将number seq generator变成一个vector seq generator。
之所以hidden，是因为只看生成得到的observation seq的话，只能知道它是由一个gaussian生成的，但我们不能看出其具体的state，因此hidden。（对比，对于number seq generator，看到seq“11233”，我们直接能看出每个数字对应的state。）

***
我们用HMM这个model表示了data，现在需要表示how well the model fits the observation，也就是$P(O|\lambda_{THE})$，可以类比之前的“计算model与observation的距离”。
![](Pasted%20image%2020230423234712.png)
（在ASR中，要比较$P(O|\lambda_{THE})$和$P(O|\lambda_{A})$）

我们需要找出 state seq X，
$P(O|\lambda_{THE})=\Sigma_{X}P(O,X|\lambda_{THE})$
（对all possible state seqs求和）
（和之前穷举path sum其实是一回事）
![](Pasted%20image%2020230424000416.png)
![](Pasted%20image%2020230424000802.png)

***
HMM training: EM






















