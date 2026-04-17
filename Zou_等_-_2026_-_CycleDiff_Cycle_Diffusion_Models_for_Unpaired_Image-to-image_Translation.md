# CycleDiff: Cycle Diffusion Models for Unpaired Image-to-image Translation

Shilong Zou†, Yuhang Huang†, Renjiao Yi, Chenyang Zhu∗, Shixiang Wang, Xiangchao Zhang, Kai $\mathrm { { X u ^ { * } } }$ , Senior Member, IEEE CycleDiff.github.io 

Abstract—We introduce a diffusion-based cross-domain image translator in the absence of paired training data. Unlike GANbased methods, our approach integrates diffusion models to learn the image translation process, allowing for more coverable modeling of the data distribution and performance improvement of the cross-domain translation. However, incorporating the translation process within the diffusion process is still challenging since the two processes are not aligned exactly, i.e., the diffusion process is applied to the noisy signal while the translation process is conducted on the clean signal. As a result, recent diffusionbased studies employ separate training or shallow integration to learn the two processes, yet this may cause the local minimal of the translation optimization, constraining the effectiveness of diffusion models. To address the problem, we propose a novel joint learning framework that aligns the diffusion and the translation process, thereby improving the global optimality. Specifically, we propose to extract the image components with diffusion models to represent the clean signal and employ the translation process with the image components, enabling an endto-end joint learning manner. On the other hand, we introduce a time-dependent translation network to learn the complex translation mapping, resulting in effective translation learning and significant performance improvement. Benefiting from the design of joint learning, our method enables global optimization of both processes, enhancing the optimality and achieving improved fidelity and structural consistency. We have conducted extensive experiments on $\mathbf { R G B } {  } \mathbf { R G B }$ and diverse cross-modality translation tasks including RGB↔Edge, $\mathbf { R } \mathbf { G B } {  }$ Semantics and RGB Depth, showcasing better generative performances than the state of the arts. Especially, our method achieves the best FID score in widely-adopted tasks and outperforms the second-best method with an improved FID of 19.61 and 19.67 on Dog→Cat and Dog→Wild respectively. 

Index Terms—Diffusion model, Unpaired image-to-image translation. 

# I. INTRODUCTION

MAGE-to-image translation is an important and useful I task in computer vision and has been attracting increasing attention lately. Generalized image-to-image tasks include 

† Equal contribution. 

∗ Co-corresponding author 

S. Zou, Y. Huang, R. Yi and C. Zhu are with the College of Computer Science and Technology, National University of Defense Technology, Changsha 410073, China (e-mail: zoushilong@nudt.edu.cn;huangai@nudt.edu.cn; yirenjiao $@$ nudt.edu.cn;zhuchenyang07@nudt.edu.cn). Xiangchao Zhang and Shixiang Wang are with College of Future Information Technology, Fudan University, Shanghai 200438, China. 

K. Xu is with the School of Computer, National University of Defense Technology, Changsha 410073, China, also with the Xiangjiang Laboratory, Changsha 410013, China, and also with the Institute of AI for Industries, Chinese Academy of Sciences, Nanjing 211135, China (e-mail: kevin.kai.xu@gmail.com) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/f4b7adec232af314dc67af247f61dce5ed25f225de454b926442c724489ced63.jpg)



Edge


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/03c9544518a68cbb9b21c43876b5d4fbb901e855fa17c89c3a6645fc14b63479.jpg)



RGB


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/e6214f6488932ee179520d9a1b08d1b1b129684fcf99d9a2c4009bb7aab90a93.jpg)



Sema


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/2a0fa9ef5d29f93e1e4816d31e0136e0d83a127fab0c072ae7a505b912ca92f5.jpg)



Q


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/9ebd8ac0c514802aad39db478bf583cb1a3beade4b285f6cc9db9985df5e5e47.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/130911610ac75b11d235c5340ed35276fcaca3b907c5affc572fe53f37cffb2c.jpg)



Q


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/f052720077e35df9efc329940232c01920ac109f29f6829f7c737aca14ca3480.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/179f1a6d1288f242555161090407c3d1763ee24d827adad3720ec9c9c182225a.jpg)



Edge


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/299e23df21cdc62071be134e577627c957e95cec2b347e8ce0fe96d7230e3221.jpg)



RGB


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/d5b05a8b4692568c346969c05222856943a458c66b796bb9e38eddc5c5ac8da1.jpg)



Semanti


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/822a6cb728f9f8c09a0cc737caa2278a60ff81ff99a261deab960f16636bccb7.jpg)



RGB


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/bf37a1fbc34f5187ddc81dd861b5b2a4f113d1312fc158a3459b57367450e7c6.jpg)



Depth


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/6bcb086f5993901cca6ac1182f815bff9080641b9f0dc1dd40372ede224302f3.jpg)



→



RGB


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/79c4d8bb80633b64765e11427f7c05291484b9bfad1cb712f63d44966a04638b.jpg)



RGB


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/6884596d98a3b75db21434a29283b3dc17860d6a50f0b434c7a1614697bdae36.jpg)



Edge


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/9482f3c4893638c94d2aa1b8a2c25a717b89f82ee849d6e821ba030f9fa422a8.jpg)



RGB


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/3844692cdc069f56a5f5a8f5befa389872efd8e2a1b8c298bcbdf1cb3cb8b555.jpg)



Semantics


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/0f5d9a4c347d62879d17564e6b2465f759b93ac91c05babadff3488c496a2460.jpg)



RGB


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/bb0f1c397f3e093483a380001a2a0eb21743f4e71e89d6feebb706ef981b2781.jpg)



→



Depth



Fig. 1. The proposed CycleDiff consists of two domain-specific diffusion models and a cycle translator, and learns the diffusion and translation processes jointly. The cycle translator consists of two translation network used for performing cycle translation between two domains: $G _ { \phi } : { \mathcal { S } } \to { \mathcal { T } }$ and $F _ { \psi } : \mathcal { T }  \mathcal { S }$ . We employ the cycle consistency constrain to regularize the forward and backward translation mappings. Utilizing only unpaired images, CycleDiff can synthesize structure-consistent and photo-realistic results across different modalities of images.


image style translation [1, 2], edge detection [3, 4], semantic segmentation [5, 6], and depth estimation [7, 8]. Common to these tasks is the requirement of paired exemplars (e.g. rgb edge pairs) for training; obtaining such paired data is often effort intensive. For example in Fig. 1, the desired output is not even well-defined for male female transfiguration and the pixel-level semantic maps are difficult to annotate. Therefore, leveraging unpaired data solely holds great potential in alleviating the burden of data collection and facilitating the generation of a virtually endless stream of new paired data. 

The typical solution of unpaired image-to-image translation is CycleGAN [9], which utilizes two generators to conduct the two-sided translation and proposes the cycle-consistency loss to maintain structure consistency. The unpaired setting of CycleGAN inspires a series of GAN-based studies [10, 11] for unpaired image-to-image translation. However, they can hardly 

achieve photo-realistic results due to several reasons. Firstly, the GAN-based generator employs a succinct framework yet does not optimize the distribution loss [12], resulting in frequent collapse of image translation. Moreover, the complex image-to-image translation is regarded as a one-step mapping in GAN-based methods, i.e., they only call the model function once per generation. However, such one-step mapping always overfits the individual points of the data distribution rather than the entire data distribution. As a result, the modeling of the cross-domain translation is hard to cover the distributions of both domains, leading to unsatisfactory translation outcomes. This serves as motivation for exploring more robust solutions in unpaired image-to-image translation. 

Recent diffusion probabilistic models have demonstrated superior image generation capabilities to GANs, prompting increased interest in diffusion-based methods for unpaired image-to-image translation. EGSDE [13], CycleDiffusion [14] and SDDM [15] focus on revising the sampling equation of diffusion models for translating images between different domains. They only consider the optimization of diffusion but without explicit learning of translation, which hardly improves the cross-domain translation performance. UNIT-DDPM [16] introduces a learnable translation module between different domains to learn the translation process. However, this module is trained independently from the diffusion-based generation in each domain, which may still lead to suboptimal translation performance without joint optimization. Another line of work attempts to train the diffusion and translation processes simultaneously. SynDiff [17] trains the two processes jointly; however, the joint optimization is only effective for a single denoising step in the diffusion process, indicating suboptimal convergence. Consequently, developing a joint learning framework that involves deeper interaction between the two processes becomes essential to enhance both global optimality and generative quality. 

In this paper, we present Cycle Diffusion Models (CycleDiff) that incorporate the cycle-consistency learning process with the diffusion models. Different from the one-step mapping of GAN-based methods, we learn the translation process with a multi-step mapping, i.e., we conduct the translation process in each denoising step of diffusion models. In this way, our model is more suitable for modeling the entire data distribution of both domains, facilitating the cross-domain translation. Compared to previous diffusion-based methods, our approach benefits from deeper integration of joint diffusion and cycle translation, improving the global optimality and maintaining the structure consistency to a great extent. Moreover, our method can be easily applied to tackle cross-modality image translation such as $\mathrm { R G B } $ Edge and $\mathrm { R G B } $ Semantics. 

Specifically, as shown in Fig. 2, we propose to embed a cycle translator between the diffusion models of two different domains, resulting in a joint training framework. There are two crucial designs to make the cycle translator work well. On the one hand, we extract the image components with the diffusion models and conduct the cycle translation process on the image components. This implementation makes it possible to learn the diffusion and translation processes jointly. On the other hand, we introduce a time-dependent 

translation network that fits the multi-step translation mapping effectively, thus improving translation quality significantly.We have conducted extensive experiments on four types of tasks including $\mathbf { R G B } {  } \mathbf { R G B }$ , RGB↔Edge, RGB Semantics, and RGB Depth, achieving new state-of-the-art performances and demonstrating the superiority of the proposed method. 

Our contributions are summarized as follows: 

• We propose a novel joint learning framework for unpaired image-to-image translation that integrates cycleconsistent translation at every denoising step of the diffusion process. This deep integration enhances global optimality while improving generative quality. 

• We introduce two key techniques to facilitate the joint learning of translation and diffusion. 1) we extract the image component from diffusion models to align the input of the translation process, thus allowing for joint learning; 2) we introduce a time-dependent translation network to improve the translation performance significantly. 

• Our method can be easily extended to crossmodality image translation. Extensive experiments show that the proposed method outperforms the state-of-the-art methods on $\mathbf { \mathrm { R G B } } {  } \mathbf { \mathrm { R G B } }$ (including Cat↔Dog, Wild Dog, Male Female, Old Young, Summer Winter, Labe Cityscape, Map Satellite, Horse Zebra) and achieves impressive performances on cross-modality translation tasks, including RGB Edge, $\mathrm { R G B } $ Semantics, and RGB Depth. 

# II. RELATED WORK

Unpaired image-to-image translation. Currently, GANbased methods have achieved impressive results in the realm of paired image-to-image translation tasks [18]. Nonetheless, there is significant potential for improvement, particularly within the domain of unpaired image-to-image translation, which can be broadly categorized into two-sided and one-sided mapping approaches. For the former framework, the most widely adopted method is the cycle consistency constraint, which limits the translated image to be able to translate back by inverse mapping, including CycleGAN [9], DualGAN [10] and DiscoGAN [11]. Following this, there are several studies have been devoted to improving it. U-GA-IT [19] incorporates an attention mechanism to let the models focus on the more important regions distinguishing between domains via the auxiliary classifier. Santa [20] introduces the shortest path regularization to find a proper mapping between two different domains based on GAN. However, the GAN-based methods can easily trap in model collapse due to the succinct framework and one-step mapping. 

Diffusion probabilistic models. Recently, diffusion probabilistic models have gained popularity as a result of their powerful generative capability through iterative denoising in various fields like images [21, 22], graph [23] and speech [24]. DDPM [25] first proposed a denoising diffusion probabilistic model, which requires many sampling timesteps to generate high-quality results. Following this, DDIM [26] aimed to accelerate sampling with implicit probabilistic models. Later, [27] formulated the general SDE framework, which linked 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/1ebb0c94e3b010d18a99b86b83be9c36d4108c5b10b97a07402f4b8b7bea144d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/c7e165174d71e435b85e693437902b781de500f606cb250926b774bb3334ccca.jpg)



Fig. 2. The overall architecture of CycleDiff. CycleDiff comprises two parts: the diffusion models and the cycle translator. The diffusion models are employed to extract image components, which are then fed into the cycle translator for unpaired translation between two domains. The diffusion and translation processes are learned jointly.


the score-based generative model and diffusion probabilistic model. [28] incorporate auto-encoder with the diffusion model, allowing the generation of higher-resolution. 

Diffusion models for unpaired image-to-image translation. There are lots of work foucsing on the image-toimage translation task. ILVR [29] and SDEdit [30] leverage score-based diffusion model to guide the generation process based on the target domain images, but both ignore the source domain data. EGSDE [13] employs an energy function pretrained on the source and target domain to refine the inference process and maintain the faithfulness of translated images. SDDM [15] explicitly optimizes the intermediate generative distributions by decomposing the score function into the “denoising” part and “refinement” part, which achieves promising results. CycleDiffusion [14] defines a latent space for stochastic diffusion model, which enables the unpaired I2I translation, image editing, and plug-and-play guidance with the pretrained diffusion model. All the methods mentioned above lack unpair training between the source and target domain, making it hard to obtain better results for structure similarity. UNIT-DDPM [16] utilizes two diffusion models and two translation models using cycle consistency constraints to achieve unpaired image-to-image translation, yet without 

joint learning. UNSB [31] aims to learn the diffusion-based generation and translation processes jointly via introducing the neural Schrodinger Bridge, however, it focuses on onesided translation instead of cycle translation. SynDiff [17] also learns the diffusion and cycle translation processes jointly, yet it formulates the translation process as a one-step mapping operation performed exclusively on clean signals. Unlike the above methods, we learn the translation process with a multistep mapping, and integrate the translation process into each denoising step of diffusion models, enhancing the global optimality. 

# III. METHOD

CycleDiff comprises two main components as depicted in Fig. 2. In Sec. III-B, we introduce the overall framework of Cycle Diffusion Models, including joint translation in Sec. III-B1 and our proposed Time-dependent translation network in Sec. III-B2. Then we present the training process and the inference process of CycleDiff in Sec. III-C and Sec. III-D respectively. Finally, we describe the implementation details of the method in Sec. III-E. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/ff19189edabca650219e3edc7480ea8752d227874041381948d24f1c6802c42f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/29bad56b1ec6e0e27ea0457bebd87c463d8a02eea05886def35146c1ab0adc6e.jpg)



Fig. 3. Comparison of decoupled diffusion model and traditional diffusion model [25] (with $ { \mathbf { \cdot } } _ { \mathbf { X 0 } }$ prediction’). The decoupled diffusion model can isolate clean components from the noisy input, while the estimates of the traditional diffusion model are still noisy and not suitable for the subsequent transformation process.


# A. Problem Formulation

Our goal is to translate i. Given two sets of data twoand $s$ $\tau$ $\boldsymbol { S } = \{ x _ { i } ^ { S } \} _ { i = 1 } ^ { N }$ $\mathcal { T } = \{ x _ { i } ^ { T } \} _ { i = 1 } ^ { M }$ where $x _ { i } ^ { S }$ and $x _ { i } ^ { \mathcal { T } }$ denote the $i$ -th images from $s$ and $\tau$ respectively, we translate $x _ { i } ^ { S }$ to $\tau$ and $x _ { i } ^ { \mathcal { T } }$ to $s$ simultaneously. 

# B. Framework of Cycle Diffusion Models

1) Joint cycle translation and diffusion: Different from previous diffusion-based methods that only revise the sampling equation or separately train translation and diffusion networks, we propose to learn the two processes jointly, obtaining a unified optimization objective and improving the generative quality greatly. 

First and foremost, there is a significant issue that must be addressed regarding joint learning. Intuitively, to train the translation process, we need to input the original images from domains $s$ and $\tau$ and learn the mapping between the two domains. On the other hand, traditional diffusion models [25] require estimation of either noise or clean signals from noisy inputs; however, precise estimation remains challenging as noise and clean signals are intricately mixed in the diffusion process. As shown in the top panel of Fig. 3, the clean signal estimated by traditional diffusion models [25] lacks sufficient clarity for subsequent translation. Consequently, directly combining the two training processes without proper alignment between them results in suboptimal performance. 

To solve the problem, we propose to extract the image components from the diffusion model to represent the clean images of the two domains. In this way, we can use the image components as the input of the translation process, performing cycle translation and diffusion training jointly. Inspired by 

decoupled diffusion models [32], we utilize the denoising network to predict the gradient of the image attenuation at each time $t$ , which can be represented as the image component. 

Specifically, the forward diffusion process describes the image attenuation and the noise growth processes, which are formulated as: 

$$
\mathbf {x} _ {t} ^ {S} = \mathbf {x} _ {0} ^ {S} + \int_ {0} ^ {t} C _ {t} ^ {S} \mathrm {d} t + t \epsilon^ {S}, \tag {1}
$$

Here, $\textstyle \mathbf { x } _ { 0 } ^ { S } + \int _ { 0 } ^ { t } C _ { t } ^ { S } \mathrm { d } t$ denotes the image attenuation process and describes the increasing process of noise, and $\epsilon ^ { S } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is the standard normal noise. Essentially, $C _ { t } ^ { S }$ is the gradient of the image attenuation and the image must attenuate to zero when $t ~ = ~ 1$ , therefore, we can easily obtain $C _ { t } ^ { S }$ via $\begin{array} { r } { \mathbf { x } _ { 0 } ^ { S } + \int _ { 0 } ^ { 1 } C _ { t } ^ { S } \mathrm { d } t = \mathbf { \dot { 0 } } . } \end{array}$ , i.e., $C _ { t } ^ { S } = - \mathbf { x } _ { 0 } ^ { S }$ . Obviously, the image attenuation process is governed by $C _ { t } ^ { S }$ , thus we can use $C _ { t } ^ { S }$ as the image component. In the training process, we use the denoising network to estimate the image component additionally. In practice, the noisy image $\mathbf { x } _ { t } ^ { S }$ is fed into the denoising network together with the time step $t$ , outputting the predicted noise $\epsilon _ { \theta } ^ { S }$ and image component $C _ { t _ { \theta } } ^ { S }$ . This process can be formulated by: 

$$
C _ {t _ {\theta}} ^ {S}, \epsilon_ {\theta} ^ {S} = \operatorname {N e t} _ {\theta} ^ {S} \left(\mathbf {x} _ {t} ^ {S}, t\right), \tag {2}
$$

where $\pmb { \theta }$ represents the parameter of domain $s$ denoising U-Net. Note that we perform the same implementation in domain $\tau$ . 

After obtaining the image components $C _ { t _ { \theta } } ^ { S }$ and $C _ { t _ { \theta } } ^ { \mathcal { T } }$ at time $t$ , we propose a cycle translator to conduct the cycle translation process as shown in the middle of Fig. 2. The cycle translator includes two time-dependent translation networks $F _ { \psi }$ and $G _ { \phi }$ , which represents the mapping $\tau  s$ and $s  \tau$ respectively. Given the image component $C _ { t _ { \theta } } ^ { S }$ from domain $s$ , the time-dependent translation network $G _ { \phi }$ first translate it into domain $\tau$ , and then the translated image component is projected back to domain $s$ , resulting in a cycle translation process. We formulate this process by the following equation: 

$$
\hat {C} _ {t _ {\theta}} ^ {S} = F _ {\psi} \left(G _ {\phi} \left(C _ {t _ {\theta}} ^ {S}, t\right), t\right), \tag {3}
$$

In a similar way, the cycle translation of $\mathcal { T } \to \mathcal { S } \to \mathcal { T }$ can be written as: 

$$
\hat {C} _ {t _ {\theta}} ^ {\mathcal {T}} = G _ {\phi} \left(F _ {\psi} \left(\boldsymbol {C} _ {t _ {\theta}} ^ {\mathcal {T}}, t\right), t\right). \tag {4}
$$

2) Time-dependent translation network: Since we jointly learn the translation and the diffusion processes, we need to view the translation process as a multi-step mapping and train the cycle translator at each time t. To this end, we introduce a time-dependent translation network that fuses the time information into the feature of the image component. Concretely, the time-dependent translation network comprises a time-attention block and a ResNet[33]-based encoder-decoder architecture. In the time-attention block, the time step $t$ is first encoded via an MLP to obtain a time embedding $i$ . This embedding is then fused with the estimated image component $C _ { t _ { \theta } } ^ { S }$ using FiLM [34] mechanism, resulting in a fused feature. 

Then, a self-attention layer is adopted to enhance the fused feature. The process is represented as: 

$$
\dot {t} = \mathbf {M L P} (t),
$$

$$
\bar {C} _ {t _ {\theta}} ^ {\mathcal {S}} = \operatorname {R e s n e t B l o c k} \left(C _ {t _ {\theta}} ^ {\mathcal {S}}, \dot {t}\right), \tag {5}
$$

$$
\tilde {C} _ {t _ {\theta}} ^ {\mathcal {S}} = \operatorname {M H S A} (\bar {C} _ {t _ {\theta}} ^ {\mathcal {S}}),
$$

where $\mathbf { M L P } ( \cdot )$ and ResnetBlock(·), and MHSA(·) mean the MLP layer, Resnet block, and multi-head self-attention laye r. C¯S $\hat { C } _ { t _ { \theta } } ^ { S }$ and C˜Stθ $\boldsymbol { \tilde { C } } _ { t _ { \theta } } ^ { S }$ denote the intermediate output after Resnet block and the enhanced feature after self-attention respectively. Next, the enhanced feature $\boldsymbol { \tilde { C } } _ { t _ { \theta } } ^ { S }$ ˜S is fed into the encoder-decoder architecture that is composed of stacked Resnet blocks, producing the translated feature. Finally, a $7 \times 7$ convolution is applied to the translated feature, resulting in the translated image component. 

# C. Training

The training objective of CycleDiff consists of two main components: the diffusion model loss and the cycle translator loss. The diffusion loss is used to supervise the denoising network and extract the image component. The cycle translator loss includes three elements. Firstly, the adversarial loss and discriminator contrastive loss (DCL) ensure that the distributions of translated image components closely match those of the target domain. Secondly, the cycle consistency loss and perceptual loss encourage the mapping function to preserve crucial features of the source image components. Thirdly, the identity loss aims to further enhance the quality of the translated image components while preserving maximum background information. The detailed training process is illustrated in Alg. 1. 

1) Diffusion model loss: For the training of the decoupled diffusion model, we supervise the $\epsilon ^ { S }$ and $\mathbf { x } _ { 0 } ^ { S }$ simultaneously, thus, the training objective of the diffusion model $\mathcal { L } _ { d m }$ is represented as: 

$$
\mathcal {L} _ {d m} = \min  _ {\boldsymbol {\theta}} \mathbb {E} _ {q \left(\mathbf {x} _ {0} ^ {S}\right)} \mathbb {E} _ {q (\boldsymbol {\epsilon})} [ \| C _ {t _ {\boldsymbol {\theta}}} ^ {S} - C _ {t} ^ {S} \| ^ {2} + \| \boldsymbol {\epsilon} _ {\boldsymbol {\theta}} ^ {S} - \boldsymbol {\epsilon} ^ {S} \| ^ {2} ], \tag {6}
$$

wheand $\pmb { \theta }$ represents the parameter of denoising network, enote the output of the diffusion model. Note tha $C _ { t _ { \theta } } ^ { S }$ $\epsilon _ { \theta } ^ { S }$ domain $\tau$ employs the same loss. 

2) Cycle translator loss: Adversarial Loss To train the cycle translator, we employ the classical adversarial loss and two additional discriminators $D _ { \psi } ^ { S }$ and $D _ { \phi } ^ { \mathcal { T } }$ to restraint mapping function $F _ { \psi }$ and $G _ { \phi }$ . For the mapping function $G _ { \phi } : { \mathcal { S } } \to { \mathcal { T } }$ and its corresponding discriminator $D _ { \phi } ^ { \mathcal { T } }$ , the adversarial loss can be expressed as follows: 

$$
\begin{array}{l} \mathcal {L} _ {a d v} ^ {S} = \mathbb {E} _ {\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {T} \sim p _ {d a t a} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {T})} [ \log D _ {\phi} ^ {T} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {T}) ] \tag {7} \\ + \mathbb {E} _ {\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {S} \sim p _ {d a t a} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {S})} [ \log (1 - D _ {\phi} ^ {\mathcal {T}} (G _ {\phi} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {S}, t))) ], \\ \end{array}
$$

$G _ { \phi }$ tries to generate image components that are similar to the image components from domain $\tau$ , while $D _ { \phi } ^ { \mathcal { T } }$ aims nents $G _ { \phi } ( C _ { t _ { \theta } } ^ { S } , t )$ sh between the translated ima and real image components $\dot { C } _ { t _ { \theta } } ^ { \mathcal { T } }$ ompo-. The goal of $G _ { \phi }$ is to minimize the objective while competing with an adversary $D _ { \phi }$ , which attempts to maximize it, i.e. 


Algorithm 1 Training algorithm of CycleDiff.


1: Initialize $i = 0, N = \text{num\_iters}, lr$ ; denoising network parameters: $\theta$ ; cycle translator parameters: $\phi, \psi$ ; diffusion model $\mathbf{Net}_{\theta}^{\mathcal{S}}$ of domain $\mathcal{S}$ and $\mathbf{Net}_{\theta}^{\mathcal{T}}$ of domain $\mathcal{T}$ ; translation network $G_{\phi}$ for $\mathcal{S} \to \mathcal{T}$ and $F_{\psi}$ for $\mathcal{T} \to \mathcal{S}$ ;  
2: while $i < N$ do  
3: $x_0^S \in \mathcal{S}, x_0^T \in \mathcal{T}$ ;  
4: $t \sim \text{Uniform}(0, 1), \epsilon^S \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \epsilon^T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ;  
5: $\mathbf{x}_t^S = \mathbf{x}_0^S + \int_0^t C_t^S \mathrm{d}t + t\epsilon^S$ ;  
6: $\mathbf{x}_t^T = \mathbf{x}_0^T + \int_0^t C_t^T \mathrm{d}t + t\epsilon^T$ ;  
7: $C_{t_\theta}^S, \epsilon_\theta^S = \mathbf{Net}_{\theta}^S(\mathbf{x}_t^S, t), C_{t_\theta}^T, \epsilon_\theta^T = \mathbf{Net}_{\theta}^T(\mathbf{x}_t^T, t)$ ;  
8: $\hat{C}_{t_\theta}^S = F_{\psi}(G_{\phi}(C_{t_\theta}^S, t), t), \hat{C}_{t_\theta}^T = G_{\phi}(F_{\psi}(C_{t_\theta}^T, t), t)$ ;  
9: Calculate $\mathcal{L}_{dm}$ and $\mathcal{L}_{tra}$ ;  
10: $\theta, \phi, \psi \gets lr * \nabla_{\theta, \phi, \psi}(\mathcal{L}_{dm} + \mathcal{L}_{tra})$ ;  
11: $i = i + 1$ ;  
12: end while;  
13: return $\theta, \phi, \psi$ ; 


Algorithm 2 Inference algorithm of CycleDiff.


1: Initialize $t = 0, k = 0, N = \text{num\_steps}, s = 1/N, x_0^S \in S, \epsilon^S \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \mathbf{Net}_\theta^S, \mathbf{Net}_\theta^T, G_\phi, C_{list} = []$ ;  
2: $\mathbf{z}_t^S = \mathbf{z}_0^S + \int_0^t C_t^S \mathrm{d}t + t\epsilon^S$ ;  
3: $C_{t_\theta}^S, \epsilon_\theta^S = \mathbf{Net}_\theta^S(\mathbf{z}_t^S, t), C_{t_\theta}^T$ ;  
4: $t = t + s$ ;  
5: $C_{list}$ . append $(C_{t_\theta}^S)$ ;  
6: while $t <= 1$ do  
7: $\mathbf{z}_t^S = \mathbf{z}_0^S + \int_0^t C_{t_\theta}^S \mathrm{d}t + t\epsilon_\theta^S$ ;  
8: $C_{t_\theta}^S, \epsilon_\theta^S = \mathbf{Net}_\theta^S(\mathbf{z}_t^S, t)$ ;  
9: $\hat{C}_{t_\theta}^T = G_\phi(C_{t_\theta}^S, t)$ ;  
10: $C_{list}$ . append $(\hat{C}_{t_\theta}^T)$ ;  
11: $t = t + s$ ;  
12: end while;  
13: $\mathbf{x}_t^T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ;  
14: while $t > 0$ do  
15: $C_{t_\theta}^T, \epsilon_\theta^T = \mathbf{Net}_\theta^T(\mathbf{x}_t^T, t)$ ;  
16: $C_{t_\theta}^S = C_{list}$ . pop();  
17: $\mathbf{d} = C_{t_\theta}^S + \epsilon_\theta^T$ ;  
18: $\mathbf{x}_{t-s}^T = \mathbf{x}_t^T - \mathbf{d}(t-s)$ ;  
19: $\mathbf{x}_t^T = \mathbf{x}_{t-s}^T$ ;  
20: $t = t - s$ ;  
21: end while;  
22: return $\mathbf{x}_t^T$ ; 

$\begin{array} { r } { \operatorname* { m i n } _ { G _ { \phi } } \operatorname* { m a x } _ { D _ { \phi } ^ { \mathcal { T } } } \mathcal { L } _ { a d v } ^ { S } ( G _ { \phi } , D _ { \phi } ^ { \mathcal { T } } , C _ { t _ { \theta } } ^ { S } , C _ { t _ { \theta } } ^ { \mathcal { T } } ) } \end{array}$ . Similarly, the objec-and its discriminator $F _ { \psi } : \mathcal { T }  \mathcal { S }$ $D _ { \psi } ^ { S }$ as well: i.e. $\begin{array} { r }  \operatorname* { m i n } _ { F _ { \psi } \operatorname* { m a x } _ { D _ { \psi } ^ { s } } \mathcal { L } _ { a d v } ^ { \mathcal { T } } \left( F _ { \psi } , D _ { \psi } ^ { S } , C _ { t _ { \theta } } ^ { \mathcal { T } } , C _ { t _ { \theta } } ^ { S } \right) } \end{array}$ , C Stθ ). In summary, we can define the adversarial loss as: 

$$
\begin{array}{l} \mathcal {L} _ {a d v} = \mathcal {L} _ {a d v} ^ {S} \left(G _ {\phi}, D _ {\psi} ^ {\mathcal {T}}, C _ {t _ {\theta}} ^ {S}, C _ {t _ {\theta}} ^ {\mathcal {T}}\right) \\ + \mathcal {L} _ {a d v} ^ {\mathcal {T}} \left(F _ {\psi}, D _ {\psi} ^ {S}, C _ {t _ {\theta}} ^ {\mathcal {T}}, C _ {t _ {\theta}} ^ {S}\right). \end{array} \tag {8}
$$

DCL loss To stabilize the training procedure of the generator and make full use of the discriminator output, we introduce Discriminator Contrastive Loss (DCL) instead of just mapping the input image to a probability scalar via the discriminator. We first reshape the output of the discriminator $\hat { \mathbf { v } } ^ { T }$ to $N$ - 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/5895c39ceeba6c217cf27a8034446ba174245c1b7254a5bcf75915ef65254686.jpg)



Fig. 4. Qualitative comparisons on $\mathbf { R G B }  \mathbf { R G B }$ tasks with state-of-the-art methods. CycleDiff could achieve superior visual results for both realism and faithfulness across all tasks. For example, in the fourth row, our method effectively retains the features that are independent of the domain, such as the white ground, while eliminating those that are specific to the domain, such as the shape of the eyebrows and mouth.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/8e65383aa5f4667e7cf5be852cfe002b6c69f3db6233ca51d14750a8c4f94cfb.jpg)



Fig. 5. More visual results on additional datasets of CycleDiff. Our method could produce high fidelity results both on time-varying datasets and challenging artificial domain data.


dimensional feature vectors (here, we set $N = 7$ ). Then, we normalize each vector to prevent from space collapsing and get $\dot { \hat { \mathbf { v } } } ^ { \mathcal { T } }$ . The DCL loss can be formulated as follows: 

$$
\begin{array}{l} \mathcal {L} _ {d c l} ^ {\mathcal {T}} (\dot {\mathbf {v}} _ {i} ^ {\mathcal {T}}, \dot {\mathbf {v}} ^ {\mathcal {T}}, \dot {\mathbf {v}} _ {i -} ^ {\mathcal {T}}) = - \frac {1}{| \dot {\mathbf {v}} _ {i -} ^ {\mathcal {T}} |} \sum_ {\dot {\mathbf {v}} _ {j} ^ {\mathcal {T}} \in \dot {\mathbf {v}} _ {i -} ^ {\mathcal {T}}} \log \times \\ \frac {\exp (\dot {\mathbf {v}} _ {i} ^ {\mathcal {T}} \cdot \dot {\mathbf {v}} _ {j} ^ {\mathcal {T}} / \tau)}{\sum_ {\dot {\mathbf {v}} _ {k} ^ {\mathcal {T}} \in \dot {\mathbf {v}} ^ {\mathcal {T}}} \exp (\dot {\mathbf {v}} _ {i} ^ {\mathcal {T}} \cdot \dot {\mathbf {v}} _ {k} ^ {\mathcal {T}} / \tau) + \sum_ {\dot {\mathbf {v}} _ {k} ^ {\mathcal {T}} \in \dot {\mathbf {v}} _ {i -} ^ {\mathcal {T}}} \exp (\dot {\mathbf {v}} _ {i} ^ {\mathcal {T}} \cdot \dot {\mathbf {v}} _ {k} ^ {\mathcal {T}} / \tau)}, \end{array} \tag {9}
$$

where $\dot { \mathbf { v } } _ { i - } ^ { T }$ denotes those feature vectors are in $\dot { \mathbf { v } } ^ { \mathcal { T } }$ but not in $\dot { \mathbf { v } } _ { i } ^ { \mathcal { T } }$ , i.e., $\dot { \mathbf { v } } _ { i - } ^ { T } = \dot { \mathbf { v } } ^ { T } \backslash \dot { \mathbf { v } } _ { i } ^ { T }$ . $\tau = 0 . 1$ is the temperature, which scales the distance between different feature vectors. The DCL loss for $\mathcal { L } _ { d c l } ^ { S } ( \dot { \mathbf { v } } _ { i } ^ { S } , \dot { \mathbf { v } } ^ { S } , \dot { \mathbf { v } } _ { i - } ^ { S } )$ is calculated in a similar manner. 

Cycle Consistency Loss Despite the $L _ { a d v }$ could ensure the translated image component $\hat { C } _ { t _ { \theta } } ^ { \dagger }$ to be in the correct domain $\tau$ . However, this cannot encourage the translated image component $\hat { C } _ { t \theta } ^ { \mathcal { T } }$ to be similar to the source image component $C _ { t \theta } ^ { S }$ . To preserve important features of the source component, we introduce the cycle consistency loss to maintain the structure similarity. Specifically, for each image component $C _ { t _ { \theta } } ^ { S }$ from $s$ , the cycle translation encourages the source image component C S $C _ { t _ { \theta } } ^ { \bar { S } }$ to translate back to the source image component, i.e., $C _ { t _ { \theta } } ^ { \check { S } ^ { \sigma } }  G _ { \phi } ( C _ { t _ { \theta } } ^ { S } , t )  F _ { \psi } ( G _ { \phi } ( C _ { t _ { \theta } } ^ { S } , t ) , \bar { t } ) \approx C _ { t _ { \theta } } ^ { \bar { S } }$ t θ C St . The 

cycle consistency loss can be formulated as follows: 

$$
\begin{array}{l} \mathcal {L} _ {c y c} = \mathbb {E} _ {\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\boldsymbol {S}} \sim p _ {d a t a} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\boldsymbol {S}})} [ \| F _ {\psi} (G _ {\phi} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\boldsymbol {S}}, t), t) - \boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\boldsymbol {S}} \| _ {1} ] \\ + \mathbb {E} _ {\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}} \sim p _ {d a t a} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}})} [ \| G _ {\phi} (F _ {\psi} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}}, t), t) - \boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}} \| _ {1} ]. \tag {10} \\ \end{array}
$$

Perceptual Loss Considering the $\mathcal { L } _ { c y c }$ is not enough to recover all textural and structural information, we incorporate the perceptual loss [35] to preserve the source image component structure. We combine the loss of the high level feature by feeding the source image component $C _ { t _ { \theta } } ^ { S }$ low-and translated image component $F _ { \psi } ( G _ { \phi } ( C _ { t _ { \theta } } ^ { S } ) )$ into VGG16 [36] architecture. The perceptual loss is as follows: 

$$
\begin{array}{l} \mathcal {L} _ {l p s} = \mathbb {E} _ {\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {S}} \sim p _ {d a t a} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {S}})} [ \left\| \varphi (F _ {\psi} (G _ {\phi} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {S}}, t), t)) - \varphi (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {S}}) \right\| _ {2} ] \\ + \mathbb {E} _ {\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}} \sim p _ {d a t a} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}})} [ \| \varphi (G _ {\phi} (F _ {\psi} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}}, t), t)) - \varphi (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}}) \| _ {2} ], \tag {11} \\ \end{array}
$$

where $\varphi ( \cdot )$ is the VGG16 feature extractor from $1 { \sim } 5$ -th convolutional layers. 

Identity Loss To ensure approximate identity mappings when the target domain image components are fed into the generator, we introduce identity loss to maintain this mapping. Formally, the identity loss is defined as follows: 

$$
\begin{array}{l} \mathcal {L} _ {i d t} = \mathbb {E} _ {\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {S} \sim p _ {d a t a} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {S})} [ \| F _ {\psi} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {S}, t) - \boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {S} \| _ {1} ] \tag {12} \\ + \mathbb {E} _ {\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}} \sim p _ {d a t a} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}})} [ \| G _ {\phi} (\boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}}, t) - \boldsymbol {C} _ {t _ {\boldsymbol {\theta}}} ^ {\mathcal {T}} \| _ {1} ]. \\ \end{array}
$$


TABLE I QUANTITATIVE COMPARISON OF UNPAIRED IMAGE-TO-IMAGE TRANSLATION METHODS. THE BEST RESULTS ARE SHOWN IN BOLD, AND THE SECOND-BEST RESULTS ARE UNDERLINED. NOTE THAT THE KID METRIC IS MULTIPLIED BY 100.


<table><tr><td>Model</td><td>FID ↓</td><td>KID ↓</td><td>SSIM ↑</td><td>FID ↓</td><td>KID ↓</td><td>SSIM ↑</td></tr><tr><td></td><td colspan="3">Cat → Dog</td><td colspan="3">Dog → Cat</td></tr><tr><td>CycleGAN[9]</td><td>85.9</td><td>-</td><td>-</td><td>107.7</td><td>-</td><td>-</td></tr><tr><td>MUNIT[37]</td><td>104.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DRIT[38]</td><td>123.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Distance[39]</td><td>155.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SelfDistance[39]</td><td>144.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GCGAN[40]</td><td>96.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LSeSim[41]</td><td>72.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ITTR (CUT)[42]</td><td>68.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>StarGAN v2[43]</td><td>54.88 ± 1.01</td><td>-</td><td>0.27 ± 0.003</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CUT[44]</td><td>76.21</td><td>4.287</td><td>0.601</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Santa[20]</td><td>52.1</td><td>2.773</td><td>0.492</td><td>46.06</td><td>1.678</td><td>0.415</td></tr><tr><td>UNSB[31]</td><td>79.69</td><td>3.936</td><td>0.543</td><td>48.31</td><td>1.814</td><td>0.614</td></tr><tr><td>SynDiff[17]</td><td>148.79</td><td>9.926</td><td>0.514</td><td>171.52</td><td>12.051</td><td>0.579</td></tr><tr><td>ILVR[29]</td><td>74.37 ± 1.55</td><td>2.054 ± 0.231</td><td>0.363 ± 0.001</td><td>54.52 ± 1.34</td><td>3.298 ± 0.196</td><td>0.392 ± 0.001</td></tr><tr><td>SDEdit[30]</td><td>74.17 ± 1.01</td><td>2.357 ± 0.156</td><td>0.423 ± 0.001</td><td>59.44 ± 1.01</td><td>3.736 ± 0.134</td><td>0.425 ± 0.001</td></tr><tr><td>EGSDE[13]</td><td>70.16 ± 1.03</td><td>1.982 ± 0.147</td><td>0.411 ± 0.001</td><td>62.34 ± 1.54</td><td>3.191 ± 0.167</td><td>0.402 ± 0.001</td></tr><tr><td>SDDM[15]</td><td>62.29 ± 0.63</td><td>-</td><td>0.422 ± 0.001</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CycleDiffusion[14]</td><td>58.87</td><td>2.140</td><td>0.557</td><td>58.80</td><td>1.510</td><td>0.403</td></tr><tr><td>CycleDiff (Ours)</td><td>49.78 ± 0.43</td><td>1.487 ± 0.105</td><td>0.552 ± 0.001</td><td>26.45 ± 0.36</td><td>0.847 ± 0.096</td><td>0.585 ± 0.001</td></tr><tr><td></td><td colspan="3">Wild → Dog</td><td colspan="3">Dog → Wild</td></tr><tr><td>CycleGAN[9]</td><td>83.95</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CUT[44]</td><td>92.94</td><td>4.807</td><td>0.592</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Santa[20]</td><td>74.93</td><td>3.161</td><td>0.453</td><td>43.44</td><td>1.714</td><td>0.439</td></tr><tr><td>UNSB[31]</td><td>86.72</td><td>4.018</td><td>0.524</td><td>59.58</td><td>2.470</td><td>0.582</td></tr><tr><td>SynDiff[17]</td><td>133.08</td><td>6.653</td><td>0.201</td><td>156.53</td><td>8.732</td><td>0.168</td></tr><tr><td>ILVR[29]</td><td>75.33 ± 1.22</td><td>2.766 ± 0.180</td><td>0.287 ± 0.001</td><td>63.15 ± 1.14</td><td>2.918 ± 0.175</td><td>0.353 ± 0.001</td></tr><tr><td>SDEdit[30]</td><td>68.51 ± 0.65</td><td>2.653 ± 0.109</td><td>0.343 ± 0.001</td><td>84.58 ± 0.61</td><td>4.534 ± 0.120</td><td>0.411 ± 0.001</td></tr><tr><td>EGSDE[13]</td><td>59.75 ± 0.62</td><td>2.028 ± 0.114</td><td>0.343 ± 0.001</td><td>80.17 ± 0.53</td><td>4.061 ± 0.114</td><td>0.409 ± 0.001</td></tr><tr><td>SDDM[15]</td><td>57.38 ± 0.53</td><td>-</td><td>0.328 ± 0.001</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CycleDiffusion[14]</td><td>56.45</td><td>2.316</td><td>0.479</td><td>69.48</td><td>1.253</td><td>0.408</td></tr><tr><td>CycleDiff (Ours)</td><td>48.57 ± 0.46</td><td>1.203 ± 0.087</td><td>0.540 ± 0.001</td><td>23.77 ± 0.41</td><td>0.810 ± 0.069</td><td>0.588 ± 0.001</td></tr><tr><td></td><td colspan="3">Male → Female</td><td colspan="3">Female → Male</td></tr><tr><td>CycleGAN[9]</td><td>67.78</td><td>-</td><td>-</td><td>77.74</td><td>-</td><td>-</td></tr><tr><td>CUT[44]</td><td>45.03</td><td>-</td><td>-</td><td>47.66</td><td>-</td><td>-</td></tr><tr><td>Santa[20]</td><td>48.47</td><td>3.768</td><td>0.570</td><td>51.26</td><td>3.426</td><td>0.592</td></tr><tr><td>UNSB[31]</td><td>37.87</td><td>4.137</td><td>0.630</td><td>49.43</td><td>4.436</td><td>0.628</td></tr><tr><td>SynDiff[17]</td><td>108.10</td><td>10.88</td><td>0.568</td><td>117.15</td><td>11.31</td><td>0.527</td></tr><tr><td>ILVR[29]</td><td>46.12 ± 0.33</td><td>4.737 ± 0.142</td><td>0.510 ± 0.001</td><td>68.66 ± 0.37</td><td>6.397 ± 0.131</td><td>0.443 ± 0.001</td></tr><tr><td>SDEdit[30]</td><td>49.43 ± 0.47</td><td>4.621 ± 0.159</td><td>0.572 ± 0.000</td><td>77.56 ± 0.47</td><td>8.059 ± 0.126</td><td>0.497 ± 0.001</td></tr><tr><td>EGSDE[13]</td><td>45.12 ± 0.24</td><td>4.596 ± 0.096</td><td>0.512 ± 0.001</td><td>72.63 ± 0.19</td><td>7.094 ± 0.092</td><td>0.487 ± 0.001</td></tr><tr><td>SDDM[15]</td><td>44.37 ± 0.23</td><td>-</td><td>0.526 ± 0.001</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CycleDiffusion[14]</td><td>45.04</td><td>4.511</td><td>0.563</td><td>46.23</td><td>3.607</td><td>0.553</td></tr><tr><td>CycleDiff (Ours)</td><td>43.48 ± 0.18</td><td>3.642 ± 0.052</td><td>0.640 ± 0.001</td><td>44.29 ± 0.24</td><td>2.614 ± 0.047</td><td>0.652 ± 0.001</td></tr></table>

3) Full Optimization Objective of CycleDiff: The full optimization objective is composed of the diffusion loss and translation loss: 

$$
\begin{array}{l} \mathcal {L} = \lambda_ {1} \mathcal {L} _ {d m} + \mathcal {L} _ {t r a} \\ = \lambda_ {1} \mathcal {L} _ {d m} + \lambda_ {2} \mathcal {L} _ {a d v} + \lambda_ {3} \mathcal {L} _ {c y c} \tag {13} \\ + \lambda_ {4} \mathcal {L} _ {i d t} + \lambda_ {5} \mathcal {L} _ {l p s} + \lambda_ {6} \mathcal {L} _ {d c l}, \\ \end{array}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } , \lambda _ { 4 } , \lambda _ { 5 } , \lambda _ { 6 }$ are scale values, controlling the weight of different losses. Compared with CycleGAN, we conduct an ablation experiment on the newly added loss of $\mathcal { L } _ { l p s }$ and $\mathcal { L } _ { d c l }$ in Sec. IV-D. 

# D. Inference

Since we learn the translation process as a multi-step mapping, we need to call the translation network in the inference stage iteratively. Specifically, given an image from domain $S$ , its image component same time, we forward C S $C _ { t _ { \theta } } ^ { S }$ each denoising step. Atto the translation network $C _ { t _ { \theta } } ^ { S }$ into the translation network $G _ { \phi }$ together with the time step $t$ to produce $\hat { C } _ { t _ { \theta } } ^ { \mathcal { T } }$ . To generate the corresponding image of domain $\tau$ , we start from the normal distribution and conduct the iterative denoising process 


TABLE II QUANTITATIVE RESULTS ON ADDITIONAL DATASETS. THE BEST RESULTS ARE SHOWN IN BOLD, THE SECOND-BEST RESULTS ARE UNDERLINED, AND THE THIRD-BEST RESULTS ARE DOUBLE-UNDERLINED.


<table><tr><td rowspan="2">Model</td><td colspan="2">Old→Young</td><td colspan="2">Summer→Winter</td><td colspan="2">Label→Cityscape</td><td colspan="2">Map→Satellite</td><td colspan="2">Horse→Zebra</td></tr><tr><td>FID↓</td><td>KID↓</td><td>FID↓</td><td>KID↓</td><td>FID↓</td><td>KID↓</td><td>FID↓</td><td>KID↓</td><td>FID↓</td><td>KID↓</td></tr><tr><td>NOT[45]</td><td>-</td><td>-</td><td>185.5</td><td>8.732</td><td>221.3</td><td>19.76</td><td>224.9</td><td>16.59</td><td>104.3</td><td>5.012</td></tr><tr><td>CycleGAN[9]</td><td>43.5</td><td>-</td><td>84.9</td><td>1.022</td><td>76.3</td><td>3.532</td><td>54.6</td><td>3.430</td><td>77.2</td><td>1.957</td></tr><tr><td>MUNIT[37]</td><td>-</td><td>-</td><td>115.4</td><td>4.901</td><td>91.4</td><td>6.401</td><td>181.7</td><td>12.03</td><td>133.8</td><td>3.790</td></tr><tr><td>Distance[39]</td><td>-</td><td>-</td><td>97.2</td><td>2.843</td><td>81.8</td><td>4.410</td><td>98.1</td><td>5.789</td><td>72.0</td><td>1.856</td></tr><tr><td>GcGAN[40]</td><td>-</td><td>-</td><td>97.5</td><td>2.755</td><td>105.2</td><td>6.824</td><td>79.4</td><td>5.153</td><td>86.7</td><td>2.051</td></tr><tr><td>CUT[44]</td><td>44.2</td><td>-</td><td>84.3</td><td>1.207</td><td>56.4</td><td>1.611</td><td>56.1</td><td>3.301</td><td>45.5</td><td>0.541</td></tr><tr><td>SDEdit[30]</td><td>-</td><td>-</td><td>118.6</td><td>3.218</td><td>-</td><td>-</td><td>-</td><td>-</td><td>97.3</td><td>4.082</td></tr><tr><td>P2P[46]</td><td>-</td><td>-</td><td>99.7</td><td>2.626</td><td>-</td><td>-</td><td>-</td><td>-</td><td>60.9</td><td>1.093</td></tr><tr><td>Santa[20]</td><td>41.9</td><td>-</td><td>-</td><td>-</td><td>46.1</td><td>-</td><td>-</td><td>-</td><td>36.2</td><td>-</td></tr><tr><td>UNSB[31]</td><td>-</td><td>-</td><td>73.9</td><td>0.421</td><td>53.2</td><td>1.191</td><td>47.6</td><td>2.013</td><td>35.7</td><td>0.587</td></tr><tr><td>CycleDiff (Ours)</td><td>40.7</td><td>1.06</td><td>72.7</td><td>1.015</td><td>45.1</td><td>0.866</td><td>53.2</td><td>3.129</td><td>43.8</td><td>0.575</td></tr></table>

using the diffusion model of domain $\tau$ . Importantly, we replace the image component of domain $\tau$ with the translated image component CSt $C _ { t _ { \theta } } ^ { S ^ { - } }$ te at each time $t$ . In this way, we can utilize the diffusion model of domain $\tau$ to generate images corresponding to domain $s$ and vice versa. The detailed sampling process is shown in Alg. 2. 

# E. Implementation Details

Network architecture As shown at the bottom of Fig. 2, the proposed time-dependent translation network consists of the time-attention block and an encoder-decoder architecture. More specifically, the time-attention block includes an MLP layer, a ResNet block, and a self-attention layer, while the encoder-decoder architecture consists of three down-sampling layers, twelve residual blocks, and three up-sampling layers. The ablation on variants of translation network architecture in Sec. IV-D. We follow [28] to conduct the diffusion process in the latent space. For the design of the discriminator, we adopt the same discriminator framework as PatchGANs [18]. The architecture of the diffusion model is based on UNet [47]. Inspired by [32], our denoising UNet needs to output both the image component and the estimated noise. To accommodate this, the UNet comprises two decoders for predicting the image component $C$ and the noise component ϵ respectively. 

Training details We perform all experiments on a single NVIDIA A100 GPU. We select the least-square loss [48] as the adversarial loss. For training of the CycleDiff, the training process contains 100000 iterations. We warm up the diffusion models from scratch for the first 50000 iterations, then the diffusion and translation models are optimized together until the training stops. For training of the diffusion model, we employ an AdamW optimizer with a decaying learning rate (from $1 e ^ { - 4 }$ to $1 e ^ { - 5 }$ ). For the joint training of the cycle translator, we apply Adam optimizer with an initial learning rate of $2 e ^ { - 4 }$ . The batch size is set to 24 for all experiments. We utilize the exponential moving average (EMA) to stabilize the performance of models during training. 

For the perceptual loss, we adopt the commonly used VGG16 with the official pretrained weights. We calculate the L2 loss between the features extracted from the $1 { \sim } 5 \cdot$ -th convolutional layers of the translated and source image components. We then average them to obtain the final perceptual loss. We set $\lambda _ { 1 } = 5 e ^ { - 2 } , \lambda _ { 2 } = 1 , \lambda _ { 3 } = 1 0 , \lambda _ { 4 } = 5 , \lambda _ { 5 } = 0 . 5 , \lambda _ { 6 } = 0 . 0 2$ as the standard setting for all experiments. In the denoising process, we implement 100 diffusion steps for $\mathbf { R G B }  \mathbf { R G B }$ tasks, aligning with the SDDM method, and extend this to 200 steps for cross-modality tasks. The ablation study on the effect of additional denoising steps can be seen in Sec. IV-D. 

# IV. EXPERIMENT

# A. Experimental setup

Datasets. We evaluate our method on the following datasets with the resolution of $2 5 6 \times 2 5 6$ , except for RGB Depth task with size $3 7 5 \times 3 7 5$ : 

(1) AFHQ [49] is a high-quality animal faces dataset, comprising three domains: cat, dog, and wild. Each category has 500 testing images. We conduct experiments on four tasks: Cat→Dog, Dog→Cat, Wild→Dog and $\mathrm { D o g \mathrm {  W i l d } }$ . (2) CelebA-HQ [50] is a high-quality human face dataset and contains two categories, male and female. For each category, there are 1,000 images for validation. We conduct experiments on the dataset with Male Female and Female Male. (3) Edge2shoes [18] contains 50,025 shoe images and corresponding edge images. we use 49,825 images for training and 200 images for testing on $\mathrm { E d g e {  } R G B }$ and RGB→Edge tasks. (4) CelebAMask-HQ [51] is a high-quality human face segmentation dataset, consisting of 24,183 images for training and 2,824 images for testing. Each image has a corresponding segmentation mask of human facial attributes with 19 classes. (5) Virtual KITTI 2 [52] is a more photo-realistic version of virtual KITTI [53] dataset, which contains multiple sets of images such as RGB, depth, class segmentation and so on. The dataset is partitioned into training and testing subsets at a ratio of 1:0.14, with 1,827 images for training and 

299 images for testing. (6) Additional unpaired image-toimage tranlation benchmarks, including: Old Young [31], Summer Winter [9], Label Cityscape [54], Map Satellite [9] and Horse Zebra [9]. 

In particular, we conduct experiments on AFHQ and CelebA-HQ datasets for $\mathbf { R G B } {  } \mathbf { R G B }$ tasks. Furthermore, the Edge2shoes, CelebAMask-HQ and Virtual KITTI datasets are utilized for the cross-modality tasks of RGB↔Edge, RGB Semantics and $\begin{array} { r } { \mathrm { R G B }  } \end{array}$ Depth respectively. 

Evaluation Metrics. On $\mathbf { \mathrm { R G B } } {  } \mathbf { \mathrm { R G B } }$ , we evaluate our translated images under two metrics. To evaluate the realism between translated images and target domain images, we report Frechet Inception Score (FID) [55] and Kernel Inception Distance (KID) [56]. To quantify the faithfulness between source domain images and translated images, we report the Structural Similarity Index Measure (SSIM) [57]. Following prior work [3], we compute the Optimal Dataset Scale (ODS) and Optimal Image Scale (OIS) to evaluate the F-scores for general edge detection on RGB Edge. Following [58, 59], we use the mean F1-score and mean intersection over union (mIOU) overall categories excluding background to measure the semantics performance on RGB Semantics. Following [60], we adopt the Root Mean Squared Error (RMSE) metrics to estimate how well the predicted depths match the ground truth on RGB Depth. 

# B. Comparisons to State-of-the-art methods

We compare CycleDiff with several state-of-the-art imageto-image translation methods: GAN-based (CycleGAN [9], CUT [44], Santa [20]), SBDM-based (SDEdit [30], ILVR [29], EGSDE [13], SDDM [15], UNSB [31]) and SynDiff [17]. We report their performances across six tasks, detailed in Tab. I and Fig. 4. The results on $\mathrm { C a t } {  } \mathrm { D o g }$ , Wild Dog and Male Female are reported from [13], [15], [14] and [61]. The results on Dog→Cat, Dog→Wild, Female Male are reproduced by ourselves using their official codes. The results on Horse Zebra, Summer Winter, Lable Cityscape, Map Satellite and Old Young are reported from [31] and [20]. In particular, both ILVR and SDEdit employ 1000 denoising steps, CycleDiffusion utilizes 800 denoising steps, and EGSDE adopts 200 denoising steps. Notably, our method follows SDDM and utilizes only 100 denoising steps. 

Results on Cat↔Dog. As shown in Tab. I, CycleDiff outperforms all GAN-based methods by a large margin in the FID metric, which demonstrates the advantage of considering the image translation as a multi-step mapping. Especially, CycleDiff exceeds the second best method by 2.32 and 19.61 on Cat→Dog and Dog→Cat respectively. 

Note that the SSIM metric denotes the similarity between the translated image and the input image, and a higher score can not mean a better translation performance. For example, CUT gets the best SSIM but the translated image has little changes compared to the input image as shown in Fig. 4. Compared to diffusion-based methods, CycleDiff benefits from the joint learning of translation and diffusion, achieving the best FID score and a comparable SSIM to the state of the art. We also visualize the qualitative comparisons in Fig. 4. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/fdc904f60851b5d96ca25301892b9688f4d9240b97807539b63161ec3127d65e.jpg)



Fig. 6. Qualitative comparison on RGB Edge, RGB Semantics, RGB↔Depth with state-of-the-art methods. CycleDiff is capable of translating images between various modalities.


Our method exhibits more reasonable and structure-consistent outcomes, indicating the superiority of joint learning. 

Results on Wild Dog. As reported in the middle of Tab. I, our method obtains the best results on the FID metric and very competitive performances on the SSIM metric. Compared to GAN-based methods, we outperform CycleGAN and CUT by a large margin (35.38 and 44.37) on the FID metric for Wild Dog. Moreover, our method obtains the best metrics for both Wild Dog and Dog→Wild, i.e., CycleDiff exceeds the second best method by 7.88 and 19.67 for Wild Dog and Dog→Wild on FID metric. Besides, as shown in Fig. 4, CycleDiff can generate images possessing both structure consistency and high fidelity. 

Results on Male Female. The bottom of Tab. I reports the performance comparisons between CycleDiff and state-


TABLE III QUANTITATIVE COMPARISON ON CROSS-MODALITY TASKS WITH STATE-OF-THE-ART METHODS. THE BEST RESULTS USING DIFFUSION MODELS ARE IN BOLD.


<table><tr><td></td><td colspan="2">Edge→RGB</td><td colspan="2">RGB→Edge</td><td colspan="2">Semantics→RGB</td><td colspan="2">RGB→Semantics</td><td colspan="2">Depth→RGB</td><td>RGB→Depth</td></tr><tr><td></td><td>FID ↓</td><td>KID ↓</td><td>ODS ↑</td><td>OIS ↑</td><td>FID ↓</td><td>KID ↓</td><td>mean F1-score (%) ↑</td><td>mIOU (%) ↑</td><td>FID ↓</td><td>KID ↓</td><td>RMSE ↓</td></tr><tr><td>CycleGAN</td><td>119.90</td><td>7.025</td><td>0.580</td><td>0.589</td><td>101.05</td><td>8.947</td><td>21.78</td><td>14.65</td><td>226.34</td><td>21.897</td><td>1.35</td></tr><tr><td>CUT</td><td>99.57</td><td>5.010</td><td>0.331</td><td>0.331</td><td>107.16</td><td>9.338</td><td>17.55</td><td>11.73</td><td>233.71</td><td>24.054</td><td>1.51</td></tr><tr><td>CycleDiffusion</td><td>381.35</td><td>48.060</td><td>0.141</td><td>0.176</td><td>260.96</td><td>41.222</td><td>3.33</td><td>1.78</td><td>464.72</td><td>61.283</td><td>2.43</td></tr><tr><td>CycleDiff (Ours)</td><td>74.93</td><td>2.377</td><td>0.766</td><td>0.769</td><td>69.29</td><td>5.975</td><td>42.95</td><td>30.63</td><td>143.25</td><td>12.368</td><td>0.52</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/9942a250ea12aa8c251e065236327d2ed43b1ec9f84ef35f673b390ba912a83a.jpg)



Ours


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/5c93d47ec2e1641cedfb70ca0d68fbf6c52676ecf05d2f14695a6570ce547078.jpg)



w/o JT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/6bf3b7ee46c22bc4592270efea971aa285e41c81117204f367865e662aea5837.jpg)



w/o IP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/6a6dba938f34928c53d090062504201606590d5062b5cfb8f66692f96d1f2f1e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/69b05761d8bfc447ea55550b3a115d228bc3224f094284c05bc55028ce1baf0d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/fdf865f72e2a9c044b7f1a683bcc527752764634e63dd42f34eb07ee7464008f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/e9b1cc74d32853f2ca7b293a8c76b4f12e811d1b9ea45661fbe3c75733460b5a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/e8bc5ed41c725e28d17c5eb47424c35541fe683412f984c50180dc855aad7225.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/9f33dd0306f7330a2c2d372ba21bee6bce344fb5b4799595c9f08cb180f59262.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/50e85233221ec6f7b19b0984db4cc00ff28a5f431fdb8e9dbe99572e43169927.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/44e12d5686ab1194dd9026faba2b2ed760faab46660e646b06864a192258ca4a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/6a1e6a2ec87c30c220c47678791b35eb0a25d38f0e365e7555405082d4e391d9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/6d0acaf3809a2e22e2e82cd0709b810904aca0eccb2eea18cfabf095bb7f0a26.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/6a98931ec9a5611483908f5554a07c2a29620275d5fdf2bf7a758a13ba3b64ba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/0fb1a2296a6712500b7f6e8e4c5f9d520ba1d4e80a8f5414830bb8ee155aab34.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/246f5b468b63504de044a6958a30c6363f6d65a484e2710cf4b6a62e81e838d2.jpg)



Fig. 7. Ablation study on joint training, multi-step mapping and image component. ‘w/o JT’ and ‘w/o IP’ mean ‘without joint training’ and ‘without image component’ respectively.


of-the-art methods. CycleDiff achieves the best results on all metrics and exceeds the second best method by 1.94 for Female Male on the FID metric. Though the improvement in FID is insignificant compared to other diffusion-based methods, the visual results depicted at the top of Fig. 4 show that our method can maintain better structural consistency. This indicates that the proposed joint learning can improve the generative capability via the improved optimality, resulting in a higher structural similarity. 

Results on additional benchmarks. As shown in Tab. II, our method achieves the best FID scores on Old→Young, Summer Winter and Lable Cityscapes tasks and obtains competitive performance to state-of-the-art methods on Map Satellite and Horse Zebra tasks. As depicted in Fig. 5, CycleDiff exhibits exceptional generative power for unpaired image-to-image translation, not only on time-varying datasets such as Old Young and Summer Winter, but also on challenging artificial domain data, including Map Satellite and Labe Cityscape. 


TABLE IV QUANTITATIVE COMPARISON RESULTS OF ABLATION STUDIES ON ARCHITECTURES OF THE TRANSLATION NETWORK AND DENOISING STEPS. WE CONDUCT EXPERIMENTS ON SEMANTICS RGB. NOTE THAT THE KID METRIC IS MULTIPLIED BY 100.


<table><tr><td>Generator Architecture</td><td>FID ↓</td><td>KID ↓</td></tr><tr><td>UNet-arch</td><td>272.69</td><td>35.097</td></tr><tr><td>ResNet6-arch</td><td>80.87</td><td>9.124</td></tr><tr><td>ResNet9-arch</td><td>77.42</td><td>8.073</td></tr><tr><td>ResNet12-arch</td><td>69.29</td><td>5.975</td></tr><tr><td>Step (with ResNet12-arch)</td><td>FID ↓</td><td>KID ↓</td></tr><tr><td>Step 50</td><td>97.39</td><td>10.079</td></tr><tr><td>Step 100</td><td>69.95</td><td>6.294</td></tr><tr><td>Step 200</td><td>69.29</td><td>5.975</td></tr><tr><td>Step 500</td><td>66.83</td><td>5.694</td></tr><tr><td>Step 800</td><td>65.43</td><td>5.511</td></tr></table>

# C. Cross-modality Translation

To demonstrate the effectiveness of our method, we conduct a series of experiments on various modality translation tasks, including RGB↔Edge, RGB Semantics, and RGB Depth. Specifically, we compare CycleDiff with GAN-based methods including CycleGAN and CUT. We also reproduce the diffusion-based method CycleDiffusion for comparisons of cross-modality tasks and employ 200 denoising steps for generation. 

Results on RGB↔Edge Our method achieves more realistic visual results compared to CUT and CycleGAN on Edge→RGB, generating images more closely to ground truth. For RGB Edge, our method is even able to capture finer details than ground truth, as exemplified in the fourth row of the results. On the other hand, CycleDiffusion performs poorly on cross-modal tasks, highlighting the inherent difficulty of directly transferring between two different domains without joint training. 

Results on RGB Semantics Compared to CycleGAN, CycleDiff produces high-quality results with an improved FID score of 31.76. Concurrently, CycleDiff can generate more precise semantic labels in challenging datasets that encompass 19 categories on RGB Semantics, demonstrating that our method could align the different modalities with the help of joint training. 

Results on RGB Depth As shown in Tab. III, our method outperforms the state-of-the-art methods by 83.09 FID score on Depth $\scriptstyle  \mathrm { R G B }$ and achieves the best RMSE of 0.52 on RGB Depth. As illustrated at the bottom of Fig. 6, the results generated by our method are even close to the ground truth 


TABLE V QUANTITATIVE COMPARISON RESULTS OF ABLATION STUDIES ON THE LOSS COMPONENT, JOINT LEARNING AND TIME-DEPENDENT TRANSLATION NETWORK.


<table><tr><td>Model</td><td>Ldcl</td><td>Llps</td><td>FID ↓</td></tr><tr><td>w/o D</td><td>-</td><td>✓</td><td>29.13</td></tr><tr><td>w/o P</td><td>✓</td><td>-</td><td>28.21</td></tr><tr><td>Ours</td><td>✓</td><td>✓</td><td>26.45</td></tr><tr><td>w/o JT</td><td>✓</td><td>✓</td><td>52.81</td></tr><tr><td>Ours</td><td>✓</td><td>✓</td><td>26.45</td></tr><tr><td>w/o TD</td><td>✓</td><td>✓</td><td>54.31</td></tr><tr><td>w/ TD &amp; w/o SA</td><td>✓</td><td>✓</td><td>29.58</td></tr><tr><td>Ours</td><td>✓</td><td>✓</td><td>26.45</td></tr></table>

on RGB Depth and Depth RGB. 

# D. Ablation Studies

Effect of the joint training. To verify the advantages of joint training, we implement a separate training scheme for diffusion models and the cycle translator. In practice, we first train the diffusion models and then utilize the generated images by diffusion models to train the translation process. The setting without joint training is denoted as ‘w/o JT’. As reported in Tab. V, the joint training framework achieves an improvement of FID score by 26.36, enhancing the generated image quality and realistic greatly. As depicted in Fig. 7, the joint training manner could synthesize reasonable images with much higher fidelity and structural consistency, demonstrating its effectiveness. To further validate the effectiveness of our method, we conduct comparative analysis of feature distributions across different variants using Principal Component Analysis (PCA) and Kernel Density Estimation (KDE). As depicted in Fig. 9, the distribution generated by our method with joint training (w/ JT) closely aligns with the target domain (GT). As a comparison, the variant without joint training (w/o JT) shows a noticeable divergence from the target distribution, indicating that the proposed joint training strategy enhances global optimality. 

Effect of the image component. In the setting of joint learning, we additionally predict the image component via the denoising network, thus we can combine the translation and diffusion processes. To show the effect of the image component, we remove the image component, instead, we utilize the estimated noise to calculate a clean image via the forward equation of the diffusion model. In this way, we can use the calculated clean image to conduct the translation process. As shown in the rightmost column of Fig. 7, it can not synthesize the correct images without the image component, demonstrating the importance of extracting image components for effective domain translation. 

Effect of time-dependent translation network. To evaluate the effectiveness of the time-dependent translation network, we modify the time-dependent translation network, removing the time input and just using the encoder-decoder architecture to process the image component. We denote the settings without the time-dependent translation network as ‘w/o TD’. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/e51a16f1465c2f8d56358f500989641c74b9211c316ccb9cf6a50c97eafa4397.jpg)



Fig. 8. Comparison between different loss components and time-dependent translation network. ‘w/o $\mathbf { D } ^ { * }$ , ‘w/o $\mathbf { P } ^ { \ast }$ and ‘w/o TD’ denote ‘without $\mathcal { L } _ { d c l }$ ’, ‘without $\mathcal { L } _ { l p s } \mathrm { : }$ ’ and ‘w/o time-dependent translation network’ respectively.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/e3561bf724aab91c5c11b652891b63fa8f430c7cab2faf4a24431b02307352ab.jpg)



Fig. 9. Comparison of distribution under different variants: target domain data distribution, data distribution generated by our method w/ JT and our method w/o JT. We generate 500 images of the target domain on $\mathrm { D o g } {  } \mathrm { C a t }$ and adopt Inception-V3 [62] for feature extraction. We then apply Principal Component Analysis (PCA) [63] to reduce the feature dimensionality to two, and use Kernel Density Estimation (KDE) [64] to visualize the probability density of the data distribution.


In addition, we also remove the self-attention layer in the time-attention block to verify its effectiveness and denote this setting as ‘w/ TD & w/o SA’. As reported in Tab. V, the timedependent translation network significantly improves the FID by 27.86, which demonstrates its superiority. The self-attention layer could further enhance the fused feature, improving the FID metric by 3.13. We also depict the visual results in Fig. 8, revealing that the time-dependent translation network can yield more realistic and high-quality visual outcomes. 

Variants of the translation network. We find that the network layer of the encoder-decoder in the translation network has a significant impact on the generation performance. In practice, we ablate the layers of the Resnet blocks and replace the Resnet block with UNet block [65]. We name these variants UNet-arch, ResNet6-arch (6 layers), ResNet9-arch (9 layers), and ResNet12-arch (12 layers). As reported in Tab. IV, the ResNet12-arch achieves the best FID score of 69.29. The UNet-arch performs poorly, which indicates the operator of the skip concatenation mechanism may not be suitable for image translation tasks. 

Effect of the denoising steps. As shown in Tab. IV, we 


TABLE VI COMPARISON WITH THE STATE-OF-THE-ART METHODS IN TERMS OF INFERENCE SPEED AND CUDA MEMORY USAGE.


<table><tr><td>Method</td><td>sec/iter ↓</td><td>CUDA Memory (GB) ↓</td></tr><tr><td>CUT</td><td>0.24</td><td>2.91</td></tr><tr><td>ILVR</td><td>60</td><td>1.84</td></tr><tr><td>SDEdit</td><td>33</td><td>2.20</td></tr><tr><td>EGSDE</td><td>85</td><td>3.64</td></tr><tr><td>CycleDiffusion</td><td>46</td><td>4.23</td></tr><tr><td>CycleDiff (Ours)</td><td>4.9</td><td>9.31</td></tr></table>

ablate five different steps ranging from 50 to 800 to evaluate the effect of denoising steps. Furthermore, it surpasses the performance of CycleGAN on the cross-modal task in just 50 steps. Adding the denoising steps requires more time and computation, so we adopt 200 denoising steps as the standard settings for cross-modality image translation experiments. 

Effect of perceptual and DCL loss. To explore the effect of the perceptual loss and DCL loss, we analyze CycleDiff by comparing three different settings on Cat→Dog: 1) without DCL loss $\mathcal { L } _ { d c l }$ (w/o D), 2) without perceptual loss $\mathcal { L } _ { l p s }$ (w/o P), 3) with total loss (Ours). Tab. V shows that $\mathcal { L } _ { d c l }$ could play an important role in the translation process to ensure the data distribution of translated image components to be close to the target domain and improve the FID by 2.68. The $\mathcal { L } _ { l p s }$ increases the capability of the generator and produces an improvement of FID by 1.76. More specifically, the perceptual loss could preserve the texture information of the input image as shown in the second row of Fig. 8 and the DCL loss encourages the source image to be more similar to target domain images as depicted in the last row of Fig. 8. 

Comparison of computational cost. To evaluate the efficiency of the proposed method, we report the inference time and CUDA memory usage during generation with a batch size of 1, as presented in Tab. VI. Our method demonstrates significantly faster sampling speeds compared to existing state-of-the-art approaches, achieving nearly $7 \times$ acceleration over SDEdit and better performance among diffusion-based methods. 

# V. CONCLUSION AND FUTURE WORK

This paper introduces CycleDiff, a diffusion-based translator for unpaired image-to-image translation tasks. We are the first attempt to learn the diffusion and translation processes jointly. On the one hand, we propose to extract the image components from diffusion models to allow for joint learning. On the other hand, we introduce a time-dependent translation network to learn the translation process efficiently. Our method can be easily applied to cross-modality image translation tasks, including RGB→Edge, RGB Semantics and RGB Depth. We conduct experiments on five different public datasets, showcasing the great superiority. 

In the future, one direction that could be investigated is to extend our method to more domains, such as $\mathbf { R G B } {  } \mathbf { N o r m }$ , RGB Text, and medical image translation such as MRI↔CT. An additional promising research direction involves leveraging 

our cycle translation framework to bridge the sim-to-real domain gap in robotic learning applications. 

# VI. ACKNOWLEDGMENT

We thank the anonymous reviewers for their valuable comments. This work is supported in part by the NSFC (62522219), the Major Program of Xiangjiang Laboratory (23XJ01009), NSFC (62325211, 62132021, 62372457, 62572477). 

# REFERENCES



[1] E. Richardson, Y. Alaluf, O. Patashnik, Y. Nitzan, Y. Azar, S. Shapiro, and D. Cohen-Or, “Encoding in style: a stylegan encoder for image-to-image translation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 2287–2296. 





[2] Y. Wang, Z. Zhang, W. Hao, and C. Song, “Multidomain image-to-image translation via a unified circular framework,” IEEE Transactions on Image Processing, pp. 670–684, 2020. 





[3] Y. Ye, K. Xu, Y. Huang, R. Yi, and Z. Cai, “Diffusionedge: Diffusion probabilistic model for crisp edge detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2024, pp. 6675–6683. 





[4] Y. Liu, Z. Xie, and H. Liu, “An adaptive and robust edge detection method based on edge proportion statistics,” IEEE Transactions on Image Processing, pp. 5206–5215, 2020. 





[5] W. Zhou, J. Liu, J. Lei, L. Yu, and J.-N. Hwang, “Gmnet: Graded-feature multilabel-learning network for rgb-thermal urban scene semantic segmentation,” IEEE Transactions on Image Processing, pp. 7790–7802, 2021. 





[6] D. Wu, Z. Guo, A. Li, C. Yu, C. Gao, and N. Sang, “Conditional boundary loss for semantic segmentation,” IEEE Transactions on Image Processing, 2023. 





[7] X. Ye, X. Fan, M. Zhang, R. Xu, and W. Zhong, “Unsupervised monocular depth estimation via recursive stereo distillation,” IEEE Transactions on Image Processing, pp. 4492–4504, 2021. 





[8] X. Xu, Z. Chen, and F. Yin, “Multi-scale spatial attention-guided monocular depth estimation with semantic enhancement,” IEEE Transactions on Image Processing, pp. 8811–8822, 2021. 





[9] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2223– 2232. 





[10] Z. Yi, H. Zhang, P. Tan, and M. Gong, “Dualgan: Unsupervised dual learning for image-to-image translation,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2849–2857. 





[11] T. Kim, M. Cha, H. Kim, J. K. Lee, and J. Kim, “Learning to discover cross-domain relations with generative adversarial networks,” in International conference on machine learning, 2017, pp. 1857–1865. 





[12] G. Kurz, F. Pfaff, and U. D. Hanebeck, “Kullbackleibler divergence and moment matching for hyperspherical probability distributions,” in 2016 19th International Conference on Information Fusion (FUSION), 2016, pp. 2087–2094. 





[13] M. Zhao, F. Bao, C. Li, and J. Zhu, “Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations,” Advances in Neural Information Processing Systems, pp. 3609–3623, 2022. 





[14] C. H. Wu and F. De la Torre, “A latent space of stochastic diffusion models for zero-shot image editing and guidance,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7378–7387. 





[15] S. Sun, L. Wei, J. Xing, J. Jia, and Q. Tian, “Sddm: scoredecomposed diffusion models on manifolds for unpaired image-to-image translation,” in International Conference on Machine Learning, 2023, pp. 33 115–33 134. 





[16] H. Sasaki, C. G. Willcocks, and T. P. Breckon, “Unit-ddpm: Unpaired image translation with denoising diffusion probabilistic models,” arXiv preprint arXiv:2104.05358, 2021. 





[17] M. Ozbey, O. Dalmaz, S. U. Dar, H. A. Bedel, S¸ . ¨ Ozturk, ¨ A. Gung ¨ or, and T. C¸ ukur, “Unsupervised medical im- ¨ age translation with adversarial diffusion models,” IEEE Transactions on Medical Imaging, vol. 42, no. 12, pp. 3524–3539, 2023. 





[18] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-toimage translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1125–1134. 





[19] J. Kim, M. Kim, H. Kang, and K. Lee, “U-GAT-IT: unsupervised generative attentional networks with adaptive layer-instance normalization for image-to-image translation,” in 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020, 2020. 





[20] S. Xie, Y. Xu, M. Gong, and K. Zhang, “Unpaired imageto-image translation with shortest path regularization,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 10 177–10 187. 





[21] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, pp. 8780–8794, 2021. 





[22] O. Ozdenizci and R. Legenstein, “Restoring vision in ¨ adverse weather conditions with patch-based denoising diffusion models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 





[23] T. Luo, Z. Mo, and S. J. Pan, “Fast graph generation via spectral diffusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 





[24] Y.-J. Lu, Z.-Q. Wang, S. Watanabe, A. Richard, C. Yu, and Y. Tsao, “Conditional diffusion probabilistic model for speech enhancement,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 7402–7406. 





[25] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, pp. 6840–6851, 2020. 





[26] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in International conference on machine learning, 2021, pp. 8162–8171. 





[27] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021. 





[28] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695. 





[29] J. Choi, S. Kim, Y. Jeong, Y. Gwon, and S. Yoon, “ILVR: conditioning method for denoising diffusion probabilistic models,” in 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, 2021, pp. 14 347–14 356. 





[30] C. Meng, Y. He, Y. Song, J. Song, J. Wu, J. Zhu, and S. Ermon, “Sdedit: Guided image synthesis and editing with stochastic differential equations,” in The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022. 





[31] B. Kim, G. Kwon, K. Kim, and J. C. Ye, “Unpaired image-to-image translation via neural schrodinger ¨ bridge,” in The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. 





[32] Y. Huang, Z. Qin, X. Liu, and K. Xu, “Simultaneous image to zero and zero to noise: Diffusion models with analytical image attenuation,” arXiv preprint arXiv:2306.13720, 2023. 





[33] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778. 





[34] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018. 





[35] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, 2016, pp. 694–711. 





[36] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, Y. Bengio and Y. LeCun, Eds., 2015. 





[37] X. Huang, M.-Y. Liu, S. Belongie, and J. Kautz, “Multimodal unsupervised image-to-image translation,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 172–189. 





[38] H.-Y. Lee, H.-Y. Tseng, J.-B. Huang, M. Singh, and M.- H. Yang, “Diverse image-to-image translation via disentangled representations,” in Proceedings of the European 





conference on computer vision (ECCV), 2018, pp. 35–51. 





[39] S. Benaim and L. Wolf, “One-sided unsupervised domain mapping,” Advances in neural information processing systems, 2017. 





[40] H. Fu, M. Gong, C. Wang, K. Batmanghelich, K. Zhang, and D. Tao, “Geometry-consistent generative adversarial networks for one-sided unsupervised domain mapping,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 2427–2436. 





[41] C. Zheng, T.-J. Cham, and J. Cai, “The spatiallycorrelative loss for various image translation tasks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 16 407–16 417. 





[42] W. Zheng, Q. Li, G. Zhang, P. Wan, and Z. Wang, “Ittr: Unpaired image-to-image translation with transformers,” arXiv preprint arXiv:2203.16015, 2022. 





[43] Y. Choi, Y. Uh, J. Yoo, and J.-W. Ha, “Stargan v2: Diverse image synthesis for multiple domains,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8188–8197. 





[44] T. Park, A. A. Efros, R. Zhang, and J.-Y. Zhu, “Contrastive learning for unpaired image-to-image translation,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX 16, 2020, pp. 319–345. 





[45] A. Korotin, D. Selikhanovych, and E. Burnaev, “Neural optimal transport,” in The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 





[46] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or, “Prompt-to-prompt image editing with cross-attention control,” in The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 





[47] Y. Song and S. Ermon, “Improved techniques for training score-based generative models,” Advances in neural information processing systems, vol. 33, pp. 12 438–12 448, 2020. 





[48] X. Mao, Q. Li, H. Xie, R. Y. Lau, Z. Wang, and S. Paul Smolley, “Least squares generative adversarial networks,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2794–2802. 





[49] Y. Choi, Y. Uh, J. Yoo, and J. Ha, “Stargan v2: Diverse image synthesis for multiple domains,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020. Computer Vision Foundation / IEEE, 2020, pp. 8185–8194. 





[50] T. Karras, T. Aila, S. Laine, and J. Lehtinen, “Progressive growing of gans for improved quality, stability, and variation,” in 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings, 2018. 





[51] C.-H. Lee, Z. Liu, L. Wu, and P. Luo, “Maskgan: Towards diverse and interactive facial image manipulation,” in Proceedings of the IEEE/CVF conference on computer 





vision and pattern recognition, 2020, pp. 5549–5558. 





[52] Y. Cabon, N. Murray, and M. Humenberger, “Virtual kitti 2,” arXiv preprint arXiv:2001.10773, 2020. 





[53] A. Gaidon, Q. Wang, Y. Cabon, and E. Vig, “Virtual worlds as proxy for multi-object tracking analysis,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4340–4349. 





[54] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 3213– 3223. 





[55] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” in Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, 2017, pp. 6626–6637. 





[56] M. Binkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demystifying MMD gans,” in 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. 





[57] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004. 





[58] Y. Zheng, H. Yang, T. Zhang, J. Bao, D. Chen, Y. Huang, L. Yuan, D. Chen, M. Zeng, and F. Wen, “General facial representation learning in a visual-linguistic manner,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 18 697–18 709. 





[59] G. Te, Y. Liu, W. Hu, H. Shi, and T. Mei, “Edgeaware graph representation learning and reasoning for face parsing,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16, 2020, pp. 258–274. 





[60] V. Patil, C. Sakaridis, A. Liniger, and L. Van Gool, “P3depth: Monocular depth estimation with a piecewise planarity prior,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 1610–1621. 





[61] J. Han, M. Shoeiby, L. Petersson, and M. A. Armin, “Dual contrastive learning for unsupervised image-toimage translation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 746–755. 





[62] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 2818–2826. 





[63] H. Hotelling, “Analysis of a complex of statistical variables into principal components.” Journal of educational psychology, vol. 24, no. 6, p. 417, 1933. 





[64] V. A. Epanechnikov, “Non-parametric estimation of a 



multivariate probability density,” Theory of Probability & Its Applications, vol. 14, no. 1, pp. 153–158, 1969. [65] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, 2015, pp. 234–241. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/481da6f8ae734eebdc5e45ff311f18529487898c09179fbb94ca5f63ca365d49.jpg)


Shixiang Wang received the B.Eng. degree from Shanghai Jiao Tong University, China, in 2013 and M.Phil. degree from University of Chinese Academy of Sciences, China, and Ph. D degree from The Hong Kong Polytechnic University, Hong Kong, in 2019. After he received the Ph. D degree, he did the Postdoctoral work at the State Key Laboratory of Ultra-precision Machining Technology, Department of Industrial and Systems Engineering, The Hong Kong Polytechnic University. He is currently an assistant professor in Fudan University. His current 

research interests include advanced manufacturing technology, precision surface measurement and freeform characterization, etc. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/5c6e604bb5680fcdb99d16bd6b0a86a82246dcfe4fd0cb04ec190851db6f70ac.jpg)


Shilong Zou received the bachelor’s degree from the University of Dalian Maritime University, Dalian, 2023. He is currently pursuing the master’s degree at the National University of Defense Technology, Changsha, China. 

His research interests include computer vision and generative models. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/6bda02634b812c6ace3fa528998acb28e84f020ff4376f8614ab28f31cda74a5.jpg)


Xiangchao Zhang received the B.E degree in measurement technology from the University of Science and Technology of China in 2005 and the Ph. D. degree in precision measurement and instrumentation from the University of Huddersfield, UK in 2009. Since 2011, he was with the Department of Optical Science and Engineering, Fudan University as an associate professor and has been with the College of Future Information Technology, Fudan University as a full professor since December 2022. He is a senior member of SPIE, and a member of ISO 

TC213, IEEE and OSA. His research interests include optical measurement technology, micro/nano optics and image processing. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/acfbaa65f0f4fa1280cd2e3d89c655917bf87eab14fad09216de8f2117ac794c.jpg)


Yuhang Huang received the bachelor’s degree from the University of Shanghai for Science and Technology, Shanghai, China, in 2019, and the master’s degree from Shanghai University, Shanghai, China, in 2022. He is currently pursuing the Ph.D. degree at the National University of Defense Technology, Changsha, China. 

His research interests include computer vision, graphics, and generative models. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/2f89c43a40613bc7af44220467af79395f8b3ee2a9b686cf32690d005447a10d.jpg)


Kai Xu (Senior Member, IEEE) received the Ph.D. degree in computer science from the National University of Defense Technology (NUDT), Changsha, China, in 2011. From 2008 to 2010, he worked as a Visiting Ph.D. degree with the GrUVi Laboratory, Simon Fraser University, Burnaby, BC, Canada. He is currently a Professor with the School of Computer Science, NUDT. He is also an Adjunct Professor with Simon Fraser University. His current research interests include data-driven shape analysis and modeling, and 3-D vision and robot perception 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/122c38be1730ef54ae9a04678290bde02caacb04516aaaca773599f95cb91766.jpg)


Renjiao Yi is an Associate Professor at the School of Computer, National University of Defense Technology. She received her Ph.D. degree from Simon Fraser University in 2019. She is interested in inverse rendering, 3D scene understanding & editing, and related AR applications. 

and navigation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-17/5d1f7b22-295a-4beb-bbd1-99ca71ee272d/400f27bc94b693ef7f1a8e3561b41b9168a3ea9599ae497897506c5bd43cba59.jpg)


Chenyang Zhu is an Associate Professor at the School of Computer, National University of Defense Technology. The current directions of interest include data-driven shape analysis and modeling, 3D vision and, robot perception & navigation, etc. 