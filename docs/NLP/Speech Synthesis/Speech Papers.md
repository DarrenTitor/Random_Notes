## 常用技术
### One TTS Alignment to Rule Them All
The alignment learning framework **combines the forward-sum algorithm, Viterbi algorithm, and an efficient static prior**. 

parallel (non-autoregressive) TTS models factor out durations from the decoding process, thereby requiring durations as input for each token. 

These models generally rely on external aligners [4] like the **Montreal Forced Aligner** (MFA) [10], or on durations extracted from a pre-trained autoregressive model(forced aligner)

non auto regressive的问题：
- dependency on external alignments
- poor training efficiency, require carefully engineered training schedules to prevent unstable learning
- difficult to extend to new languages either because pre-existing aligners are unavailable or their output does not exactly fit the desired format

模型：
![](Pasted%20image%2020230225185406.png)
输入：**encoded** text input和mel spectrogram
将文本长度N和mel frame个数T对齐。


objective的第一部分是likelihood，基于hidden markov model的forward sum。
![](Pasted%20image%2020230226002508.png)

然后要将text和mel spectrogram分别送入一个encoder，得到两个latent vector（不要求长度相同）。
![](Pasted%20image%2020230226002630.png)
然后对这两个latent vector计算**pairwise**的L2，得到一个距离的矩阵。
然后对这个矩阵D关于text维度求softmax。
![](Pasted%20image%2020230226002947.png)



然而得到的这个A_soft还不能用于alignment。我们想要明确的text和mel frame之间的映射，而不是概率，也就是说A应该是hard label，是binary的0或1，而不是01之间的小数。
![](Pasted%20image%2020230226003417.png)
我们对A_soft施加viterbi算法，得到一个单调的（monotonic）路径矩阵A_hard，然后通过KL divergence计算A_soft和A_hard之间的距离，最后将KL 散度与之间的forward sum的loss相加，作为训练的loss。
![](Pasted%20image%2020230226004117.png)


Alignment Acceleration

为了加速训练，为likelihood的部分加了一个prior，变成求posterior。
![](Pasted%20image%2020230226004344.png)




### Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi

这论文有点复杂，思想是GMM+HMM，具体的还没看。
下面这个简要的介绍来自MFA的文档：
https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/index.html
### Pipeline of training

The Montreal Forced Aligner by default goes through four primary stages of training. 
- The first pass of alignment uses **monophone** models, where each phone is modelled the same regardless of phonological context. 
- The second pass uses **triphone** models, where context on either side of a phone is taken into account for acoustic models. 
- The third pass performs LDA+MLLT (feature space Maximum Likelihood Linear Regression (fMLLR)) to learn a transform of the features that makes each phone’s features maximally different. 
- The final pass enhances the triphone model by taking into account **speaker differences,** and calculates a transformation of the mel frequency cepstrum coefficients (MFCC) features for each speaker. 

See the [Kaldi](http://kaldi-asr.org/) page on feature transformations for more detail on these final passes.





## 纯synthesis
### FASTSPEECH 2: FAST AND HIGH-QUALITY END-TOEND TEXT TO SPEECH
ICLR 2021

1) directly training the model with ground-truth target instead of the simplified output from teacher
2) introducing more variation information of speech (e.g., pitch, energy and more accurate duration) as conditional inputs. 


回顾FastSpeech：
![](Pasted%20image%2020221112235227.png)
![](Pasted%20image%2020230204174541.png)
![](Pasted%20image%2020230204174706.png)
![](Pasted%20image%2020230204174822.png)

![](Pasted%20image%2020230204175124.png)

encoder部分，使用一些额外的模型、方法分别得到pitch、duration和energy。
然后训练时将这些信息加到latent embedding中，同时用这些信息训练各自的predictor。
测试时直接用predictor预测出这些信息，加到latent embedding中。


### Mixer-TTS: Non-Autoregressive, Fast and Compact Text-to-Speech Model Conditioned on Language Model Embeddings
这篇的introduction部分写得很好，信息量很大。
Non-autoregressive models can generate speech two orders of magnitude faster than auto-regressive models with similar quality.
![](Pasted%20image%2020230223224301.png)

Glow-TTS [10] proposed a flowbased algorithm for unsupervised alignment training. This algorithm has been improved in RAD-TTS [11] and modified for non-autoregressive models in [12]. This new alignment framework greatly simplifies TTS training pipeline
这个12之后要看下。
R. Badlani, A. Lancucki, K. Shih, R. Valle, W. Ping, and B. Catanzaro, “One TTS alignment to rule them all,” arXiv:2108.10447, 2021.

![](Pasted%20image%2020230223224443.png)
上面解释了用external tool求pitch的重要性。
顺便提了一下用BERT的好处。

这篇文章是对计算机视觉的MLP-Mixer的迁移。
- backbone：MLP-Mixer，non auto regressive
- uses an explicit duration predictor, which is trained by the unsupervised alignment framework proposed in [12]
- has an explicit pitch predictor
- adds token embeddings from an external pre-trained LM to improve speech prosody and pronunciation. 
并且指出如果只用BERT的token，而不用BERT作inference的话，并没有增加多少计算量，但效果很明显。（本文使用的是huggingface的ALBERT。）
![](Pasted%20image%2020230223225540.png)




![](Pasted%20image%2020230224001354.png)
![](Pasted%20image%2020230224001341.png)
这里这个需要详细看一下12和这里的代码，没准能用到。
we **train the speech-to-text alignments jointly with the decoder** by using adaptation of unsupervised alignment algorithm [12] which was proposed in implementation of FastPitch 1.13
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch
![](Pasted%20image%2020230224001055.png)





















### Fastpitch: Parallel Text-to-Speech with Pitch Prediction
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch



















## 有关prosody的synthesis


### Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron

^7f06d5

2018
![](Pasted%20image%2020221112170436.png)
![](Pasted%20image%2020221112172311.png)


训练时，reference audio就是target audio。
为了避免copy问题，采用的解决方法是：
使用bottleneck结构，强制让模型学到低维representation。
inference时，reference audio无限制，不一定要和训练数据是同一speaker。

我的问题：
怎样设计才能让模型的不同module各自学到speaker和prosody？
（这篇文章的speaker部分写得好乱，不想看了）



计算prosody embedding的encoder，使用的是几层conv+通过stride进行downsample+GRU得到一个128dim的embedding。
然后再用几层fully connected layer将embedding调整到想要的维数。

prosody encoder的input形式也有讲究，使用的是mel-warped spectrum。
![](Pasted%20image%2020221112174621.png)

我们可以换个角度，把prosody encoder这个带有downsample的convnet看作是autoencoder的encoder，整个模型相当于输入audio输出audio的antoencoder。只不过decoder部分，也就是tacotron，还使用了text和speaker的信息。
![](Pasted%20image%2020221112175012.png)

另外，其实可以不只产生一个prosody embedding，而是可变长度，产生很多个。这样模型可以处理很长的句子，但也有坏处。
![](Pasted%20image%2020221112175815.png)

然后这篇文章提出了几个metric。
![](Pasted%20image%2020221112175856.png)




### ROBUST AND FINE-GRAINED PROSODY CONTROL OF END-TO-END SPEECH SYNTHESIS
2019
![](Pasted%20image%2020221112184334.png)
这篇文章是直接打上面[这篇](#^7f06d5)的。
two limitations
- controlling the prosody at a specific moment of generated speech is not clear.
为了引入可变长度个数个prosody embedding而不是只用一个。
- inter-speaker prosody transfer is not robust if the difference between the pitch range of the source speaker and the target speaker is significant
speaker之间如果pitch差太多，inference效果不好。

contribution：
1. 可变长度个prosody embedding，长度可以等于reference audio也可以等于text。
2. 对prosody embedding做normalization有好处。
![](Pasted%20image%2020221112185119.png)


介绍GST-Tacotron，
![](Pasted%20image%2020221112185746.png)

Method：
Variable-length prosody embedding
将prosody embedding的个数downsample到与text side（encoder attention长度）或speech side（decoder attention长度）相匹配。

prosody encoder中的convnet，第一层是用Coordconv来保留positional information。

![](Pasted%20image%2020221112193755.png)











### Cross-speaker Style Transfer with Prosody Bottleneck in Neural Speech Synthesis

现状limitation：
1. the style embedding extracted from single reference speech can hardly provide fine-grained and appropriate prosody information for arbitrary text to synthesize
2. the content/text, prosody, and speaker timbre are usually highly entangled, it’s therefore not realistic to expect a satisfied result when freely combining these components, such as to transfer speaking style between speakers.

contribution：
使用prosody bottleneck，disentangles the prosody from content and speaker timbre

基于transformer-TTS
没仔细看









### PROSOSPEECH: ENHANCING PROSODY WITH QUANTIZED VECTOR PRE-TRAINING IN TEXT-TO-SPEECH
(ICASSP 2022)
![](Pasted%20image%2020221112221728.png)
Based on FastSpeech

现状limitation：
1. some works use external tools to extract pitch contour. However, the extracted pitch has inevitable errors
2. Some works extract prosody attributes (e.g., pitch, duration and energy) from speech and model them separately. However, these prosody attributes are dependent on each other and produce natural prosody together.
3. Prosody has very high variability and varies from person to person and word to word. **It can be very difficult to shape the full distribution of prosody using the limited amount of high-quality TTS data.**


Method：
![](Pasted%20image%2020221112223127.png)
![](Pasted%20image%2020221112223139.png)


![](Pasted%20image%2020221112225516.png)
我认为的关键信息：
- 基于fastspeech，需要预测character对应的时长
- 从text中用encoder提取出text和phoneme的embedding
- 从reference audio的mel-spectrogram的低频部分中用encoder提取出embedding，称为LPV。
- 这篇工作的关键在于，training的时候，代表prosody的embedding LPV来自于ground truth的spectrogram；inference的时候，如果有reference audio，则LPV来自于prosody decoder；如果没有reference audio，则LPV来自于一个pretrained auto regressive predictor。这个predictor接收文字和speaker embedding，输出prosody embedding。
- LPV predictor是auto-regressive的，在训练的时候会使用auto-regressive模型常用的teacher forcing。
![](Pasted%20image%2020221112231339.png)
- pre-train
![](Pasted%20image%2020221112232606.png)
![](Pasted%20image%2020221112232623.png)




### Discourse-Level Prosody Modeling with a Variational Autoencoder for Non-Autoregressive Expressive Speech Synthesis
![](Pasted%20image%2020230205182112.png)
基于fastspeech2
![](Pasted%20image%2020230205182556.png)
#### 相关工作
Autoregressive：
Tacotron2 and Transformer-TTS
- However, these models suffer from the unsatisfactory robustness of the attention mechanism, especially when the training data is highly expressive.
- low inference efficiency.

Non-autoregressive：
FastSpeech and Parallel Tacotron

Acoustic：
Recently, acoustic models based on Normalizing Flows and Diffusion Probabilistic Models have also pushed the naturalness of speech synthesis to a new level.

one-to-many mapping带来的问题：
“**one-to-many mapping from phoneme sequences to acoustic features**, especially the prosody variations in expressive speech with multiple styles”

- Variational autoencoder (VAE)
- Another approach to address the one-to-many issue is utilizing the textual information in a context range wider than the current sentence, e.g., at the paragraph or discourse level. （说白了就是context。）
![](Pasted%20image%2020230205191126.png)
以往这两个思路用于autoregressive model，本文将其用于non autoregressive。

In this method, a VAE is combined with FastSpeech to extract phone-level latent prosody representations, i.e., prosody codes, from the fundamental frequency (F0), energy and duration of the speech. Then, a Transformer-based model is constructed to predict prosody codes, taking discourse-level linguistic features and BERT embeddings as input.

FastSpeech1：
usually results in over-smoothed output
In order to alleviate this problem, FastSpeech1 introduces an autoregressive teacher model for knowledge distillation

FastSpeech2：
![](Pasted%20image%2020230205193455.png)
![](Pasted%20image%2020230205193804.png)
FastSpeech2存在的问题：
- 需要额外的module
- 需要将pitch spectrogram用CWT进行插值，不太符合直觉。
- 使用frame level的pitch信息，范围不够大。
#### Method

- 将用其他方法提取得到的variation information (i.e., pitch, energy, and upsampled frame-level duration)送入VAE，而不是直接送入spectrogram。这样就可以专注于prosody，忽略其他细节，增强泛化能力。![](Pasted%20image%2020230205194854.png)
疑问：variation info从哪来？
答：The structures of the text encoder, duration predictor, length regulator and mel-spectrogram decoder follows the ones in FastSpeech1。

- VAE encoder 中sample得到的latent prosody embedding和Bert的text embedding进行concat，然后送入decoder。
![](Pasted%20image%2020230205195933.png)
    VAE 的elbo的KL散度需要用到这两个东西，记一下：
    ![](Pasted%20image%2020230205195555.png)

- VAE encoder为每个phone预测一个多维的μ，这个向量称为这个phone的prosody code of this phone。

- 使用文本的context：用pretrained BERT提取context的embedding，然后训练一个encoder，接受这些context embedding，预测之前得到的prosody code。
![](Pasted%20image%2020230205201353.png)
- 值得注意的是，真正inference的时候，是没有VAE的。VAE（reference encoder）只是用来生成prosody code，然后用这个prosody code来训练一个从context中预测prosody code的encoder。最后inference用的是这个context encoder预测出来的prosody信息。
![](Pasted%20image%2020230205201909.png)
#### 总结：
这篇文章中用VAE生成一个代表prosody的latent 分布，然后并没有直接用这个latent信息来inference。而是用这个latent信息训练了一个“输入context text，输出prosody latent 信息”的encoder。最后用这个encoder预测的prosody来指导fastspeech的inference。
**这个使用VAE进行latent建模，然后再用latent训练其他module的思路可以学习一下。没准可以把其中的哪一部分换成visual的encoder。**







### Improving Emotional Speech Synthesis by Using SUS-Constrained VAE and Text Encoder Aggregation
In recent studies, Variational AutoEncoder(VAE) [8] shows stronger capabilities in disentanglement, scaling and interpolation for expression modeling [9] and style control[10].


思想：
- 在VAE中，Instead of conventional KL-divergence regularizer, the new constraint expects the means of the embedding vectors are on the surface of the unit sphere while all dimensions have a uniform standard deviation. 也就是使用了一个新的约束/loss。
- 将emotion embedding作为query送入encoder而不是decoder。同时也使用一些其他text embedding作为encoder的query。这样就组成了一个**multi-query attention**。


It contains 
a self-attention-based text encoder, 
an RNN-based auto-regressive decoder, 
a GMMbased attention[14] bridging them, 
a VAE-based emotion encoder and 
an emotion classifier. 
WaveRNN[15] is adopted to convert mel spectrogram to waveforms.
![](Pasted%20image%2020230223213227.png)
#### SUS-constrained VAE

这里讲了一下VAE的posterior collapse的问题：
![](Pasted%20image%2020230223210537.png)
核心思想：
The critical problem here, we think, is the **distances between the means should be in the similar order with their standard deviations.**

If the distance between means is much bigger than their standard deviation, latent vectors will collapse to the means. Conversely, **if the distance between means is much smaller, latent vectors will collapse to be independent on the input.**
（感觉这里的independent是不是应该是dependent，没准写错了？）

Inspired by it, this paper restricts the means approaching to the Surface of the Unit Sphere (SUS) while set the standard deviations to be an appropriate constant for all dimensions, such as 1.
设std为常数，约束mean到单位圆的表面。


上面的思路的故事讲得很好，
放到实际代码中，其实就是相当于拉格朗日约束，约束latent vector的mean的norm为1.
然后手动设置std为一个常数。
![](Pasted%20image%2020230223210959.png)

#### Weighted Aggregation
multi-query attention
说白了就是把encoder中每个layer的输出拿过来送入同一个attention，计算一个整体的信息，来囊括各个尺度的信息。
说白了相当于object detection的FPN。
#### Combined multi-query
![](Pasted%20image%2020230223213359.png)
其实就是把emotion embedding也送入这个类似于FPN的multi-query attention。具体细节不看了。

-   这里还有一个参考的地方：展示对emotion的聚类（latent embedding）会用到t-SNE。
- 还有几个需要看的东西：
-   In recent studies, Variational AutoEncoder(VAE) [8] shows stronger capabilities in disentanglement, scaling and interpolation for expression modeling [9] and style control[10].
-   In our previous work[11], we utilize the contexts extracted form the stacked layers to do self-learned multi-query attention over an expressive corpus.
-   posterior collapse. Many attempts have been made to address this puzzle, such as annealing strategy in [10].
-   还有收藏夹里的几篇关于posterior collapse的文章。
-   GMMv2 attention
-   layer normalization
-   multi-query attention



## Video相关

### VisageSynTalk: Unseen Speaker Video-to-Speech Synthesis via Speech-Visage Feature Selection
ECCV 2022
这篇的任务是lip speech。

Nevertheless, video-to-speech synthesis is considered as challenging since it is expected to represent not only the speech content but also the identity characteristics (e.g., voice) of the speaker

![](Pasted%20image%2020221114223035.png)

![](Pasted%20image%2020221114232951.png)



![](Pasted%20image%2020221114223356.png)

视频经过visual encoder之后，用speech-visage feature selection module提取speech content信息，过滤掉speaker的外貌信息。
![](Pasted%20image%2020221115010414.png)
speech-visage feature selection module原文说的挺花哨，其实就是用lstm+softmax计算attention weight，用这些weight对encoder得到的visual feature计算embedding。

感觉下面这一步是这篇文章的关键。
**我们假设feature中的信息只有speech content和speaker外貌信息两种。
那么计算了speaker外貌的attention weight之后，1-weight得到的就是speech信息的weight。**
![](Pasted%20image%2020221115010505.png)

然后对visual和speech的feature都做multi-head attention。

systhesis部分：
![](Pasted%20image%2020221115011520.png)
类似style-transfer的思想。

这里使用的style transfer generator出自：'A style-based generator architecture for generative adversarial networks.' cvpr2019
将speech和visual的信息同时送入这个generator，其中使用visual信息来定义style。最终输出mel-spectrogram。

最后后处理的部分，先训练一个网络将mel-spectrogram转成linear spectrogram，然后施加Griffin-Lim algorithm 得到waveform。这里训练这个后处理网络的时候对应着一个reconstruction loss。

![](Pasted%20image%2020221115012352.png)

对speech和visual特征各自加上了一些子网络来指引之前的attention的训练。说白了就是相当于multi-task。

针对visual feature的多任务学习：
这里这个思路也很好：
![](Pasted%20image%2020221115012632.png)
除了对得到的visual feature进行一个multi class classification来预测speaker，我们还想让同一个speaker说不同内容的两个视频，抽取出来的visual feature尽可能类似。
也就是在classifier的输出logit和输入feature map层面各自计算一个loss。
![](Pasted%20image%2020221115012525.png)
最后，我们想让speech和visual的feature各司其职，也就是说，如果将speech content的feature输入进上面提到的预测speaker的classifer，我们希望其预测结果很差。也就是说我们希望这一部分的loss越大越好。
原文是通过reverse gradient实现的，我感觉其实直接在这个loss之前直接加个负号作用也差不多。
![](Pasted%20image%2020221115013532.png)
![](Pasted%20image%2020221115013615.png)

针对speech content feature的多任务学习：
![](Pasted%20image%2020221115014016.png)

这里有一个pretrain的模型，输入spectrogram，输出speaker id。
我们将原本模型输出的spectrogram送入这个pretrained model，得到的结果与ground truth speaker id算一个loss。

在此基础上，我们调换两个对应着不同speaker 的visual feature，同时使用原本各自的speech content feature，期待这个pretrain的模型可以预测出这两个speaker的身份互换了。


模型其实还接了一个GAN的discriminator。
换句话说，这个模型其实同时使用了对于mel-spectrogram的重构损失以及discriminator的分类损失。
![](Pasted%20image%2020221115014936.png)

#### 总结：
感觉VisageSynTalk中的这个将信息disentangle为w和1-w两部分的想法可能能用上。没准以后可以将video中的信息解耦为与speech相关的信息和与speech无关的信息。或者解耦为与speecher相关和无关的两部分，或者解耦为与event相关和无关的两部分。反正这个思路可以继续延伸。

### SVTS: Scalable Video-to-Speech Synthesis
INTERSPEECH 2022

![](Pasted%20image%2020221115210117.png)

It consists of a video-to-spectrogram predictor followed by a spectrogram-to-waveform synthesizer.

ResNet18+conformer network
+
pre-trained neural vocoder

contribution:
除了提出一个简单高效的模型之外，这篇文章的消融实验也挺值得注意。
![](Pasted%20image%2020221115210246.png)

conformer：
A. Gulati, J. Qin, C. Chiu, et al., “Conformer: Convolutionaugmented transformer for speech recognition,” in Interspeech, H. Meng, B. Xu, and T. F. Zheng, Eds., ISCA, 2020

![](Pasted%20image%2020221115210506.png)


![](Pasted%20image%2020221115213502.png)

![](Pasted%20image%2020221115213513.png)


























