# EDGE-PRESERVING NOISE FOR DIFFUSION MODELS

Jente Vandersanden, Sascha Holl, Xingchang Huang, Gurprit Singh 

Max Planck Institute for Informatics, Germany 

{jvanders, sholl, xhuang, gsingh}@mpi-inf.mpg.de 

# ABSTRACT

Classical generative diffusion models learn an isotropic Gaussian denoising process, treating all spatial regions uniformly, thus neglecting potentially valuable structural information in the data. Inspired by the long-established work on anisotropic diffusion in image processing, we present a novel edge-preserving diffusion model that generalizes over existing isotropic models by considering a hybrid noise scheme. In particular, we introduce an edge-aware noise scheduler that varies between edgepreserving and isotropic Gaussian noise. We show that our model’s generative process converges faster to results that more closely match the target distribution. We demonstrate its capability to better learn the low-to-mid frequencies within the dataset, which plays a crucial role in representing shapes and structural information. Our edge-preserving diffusion process consistently outperforms state-of-the-art baselines in unconditional image generation. It is also particularly more robust for generative tasks guided by a shape-based prior, such as stroke-to-image generation. We present qualitative and quantitative results (FID and CLIP score) showing consistent improvements of up to $30 \%$ for both tasks. Our source code and supplementary content are available via the public domain edge-preservingdiffusion.mpi-inf.mpg.de. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/ea3a0e23d0ae00107268af290b40965c4e0fc755affe78dbcc4c78613649d8a9.jpg)



Figure 1: A classic isotropic diffusion process (top row) is compared to our hybrid edge-aware diffusion process (middle row) on the left side. We propose a hybrid noise (bottom row) that progressively changes from anisotropic $t = 0$ ) to isotropic noise $\mathit { t } = 4 9 9 $ ). We use our edgeaware noise for training and inference. On the right, we compare both noise schemes on the SDEdit framework (Meng et al., 2022) for stroke-based image generation. Our model consistently outperforms DDPM’s isotropic scheme, is more robust against visual artifacts and produces sharper outputs without missing structural details.


# 1 INTRODUCTION

Previous work on diffusion models mostly uses isotropic Gaussian noise to transform an unknown data distribution into a known distribution (e.g., normal distribution), which can be analytically sampled (Song and Ermon, 2019; Song et al., 2021; Ho et al., 2020; Kingma et al., 2021). Due to the 

isotropic nature of the noise, all regions in the data samples $\mathbf { x } _ { \mathrm { 0 } }$ are uniformly corrupted, regardless of the underlying structural content, which is typically distributed in a non-isotropic manner. During the backward process, the model is trained to learn an isotropic denoising process that ignores this potentially valuable non-isotropic information. In image processing literature (Elad et al., 2023), denoising is a well studied topic. Following the work by Perona and Malik (1990) structure-aware guidance has shown remarkable improvements in denoising. Since generative diffusion models can also be seen as denoisers, we ask ourselves: Can we enhance the effectiveness of the generative diffusion process by incorporating awareness of the structural content of the data samples in the underlying dataset? 

To explore our question, we introduce a new class of diffusion models that generalizes over existing isotropic models and explicitly learns a content-aware noise scheme. We call our noise scheme edge-preserving noise. It offers several benefits: First, it allows the backward generative process to converge more quickly to accurate predictions. Second, our edge-preserving model better captures the low-to-mid frequencies in the target dataset, which typically represent shapes and structural information. Consequently, we achieve improved results for unconditional image generation. Lastly, our model also demonstrates greater robustness and quality for generative tasks that rely on shapebased priors. 

To summarize, we make the following contributions: 

• We present a novel class of content-aware diffusion models and show how it is a generalization of existing isotropic diffusion models 

• We conduct a frequency analysis to better understand the modeling capabilities of our edge-preserving model. 

• We run extensive qualitative and quantitative experiments across a variety of datasets to validate the superiority of our model over existing models. 

• We observed consistent improvements in pixel space diffusion. We found that our model converges faster to more accurate predictions and better learns the low-to-mid frequencies of the target data, resulting in FID score improvements of up to $30 \%$ for unconditional image generation and most remarkably a more robust behaviour and better quality on generative tasks with a shape-based prior. 

# 2 RELATED WORK

Most existing diffusion-based generative models (Sohl-Dickstein et al., 2015; Song and Ermon, 2019; Song et al., 2021; Ho et al., 2020) corrupt data samples by adding noise with the same variance to all pixels. These generative models can generate diverse novel content when the noise variance is higher. On the contrary, noise with lower variance is known to preserve the underlying content of the data samples. Rissanen et al. (2023) introduced an inverse heat dissipation model (IHDM), which applies isotropic Gaussian blurring to corrupt images, which they show is equivalent to introducing non-isotropic noise in the frequency domain. Hoogeboom and Salimans (2023) improved IHDM by adding isotropic noise, achieving higher quality. More recently, Huang et al. (2024a) proposed the blue noise diffusion model (BNDM), using negatively correlated noise for enhanced visual quality and FID scores. While IHDM and BNDM incorporate non-isotropic noise, they do not explicitly account for structures present in the signal. 

Various efforts (Bansal et al., 2023; Daras et al., 2023) were made to develop non-isotropic noise models for diffusion processes. Dockhorn et al. (2022) proposed to use critically-damped Langevin diffusion where the data variable at any time is augmented with an additional ”velocity” variable. Noise is only injected in the velocity variable. Voleti et al. (2022) performed a limited study on the impact of isotropic vs non-isotropic Gaussian noise for a score-based model. The idea behind non-isotropic Gaussian noise is to use noise with different variance across image pixels. They use a non-diagonal covariance matrix to generate non-isotropic Gaussian noise, but their sample quality did not improve in comparison to the isotropic case. Yu et al. (2024) developed this idea further and proposed a Gaussian noise model that adds noise with non-isotropic variance to pixels. The variance is chosen based on how much a pixel or region needs to be edited. They demonstrated a positive impact on editing tasks. 

Our definition of anisotropy follows directly from the seminal work by Perona and Malik (1990) on anisotropic diffusion for image filtering. We apply a non-isotropic variance to pixels in an edge-aware manner, meaning that we suppress noise on edges. 

# 3 BACKGROUND

Generative diffusion processes. A generative diffusion model consists of two processes: the forward process transforms data samples $\mathbf { x } _ { \mathrm { 0 } }$ into samples $\mathbf { x } _ { T }$ that are distributed according to a well-known prior distribution, such as a normal distribution $\mathcal { N } ( 0 , I )$ . The corresponding backward process does exactly the opposite: it transforms samples $\mathbf { x } _ { T }$ into $\hat { \mathbf { x } } _ { 0 }$ , distributed according to the target distribution $p _ { 0 } ( \mathbf { x } )$ . This backward process involves predicting a vector quantity, interpretable as either noise or the gradient of the data distribution, which is precisely the task for which the generative diffusion model is trained. Previous works (Song and Ermon, 2019; Song et al., 2021; Ho et al., 2020; Kingma et al., 2021; Rissanen et al., 2023; Hoogeboom and Salimans, 2023) typically formulate the forward process as the following linear equation: 

$$
\mathbf {x} _ {t} = \gamma_ {t} \mathbf {x} _ {0} + \sigma_ {t} \boldsymbol {\epsilon} _ {t} \tag {1}
$$

here, $\mathbf { x } _ { t }$ is the data sample diffused up to time $t$ , $\mathbf { x } _ { \mathrm { 0 } }$ stands for the original data sample, $\epsilon _ { t }$ is a standard normal Gaussian noise, and the signal coefficient $\gamma _ { t }$ and noise coefficient $\sigma _ { t }$ determine the signal-to-noise ratio (SNR) $\left( \gamma _ { t } / \sigma _ { t } \right)$ . The SNR refers to the proportion of signal retained relative to the amount of injected noise. Note that $\gamma _ { t }$ and $\sigma _ { t }$ are both scalars. Previous works have made several different choices for $\gamma _ { t }$ and $\sigma _ { t }$ respectively, leading to different variants, each with their own advantages and limitations. 

Denoising probabilistic model. Following the probabilistic paradigm of Ho et al. (2020), we would like to introduce the posterior probability distributions of the general diffusion process described by Eq. (1). We will show the exact form that our forward and backward processes take in Section 4.1 and Section 4.3 respectively. For details and full derivations of the equations in this paragraph, we would like to refer to the appendix of Kingma et al. (2021). The isotropic diffusion process formulated in Eq. (1) has the following marginal distribution: 

$$
q \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right) = \mathcal {N} \left(\gamma \mathbf {x} _ {0}, \sigma_ {t} ^ {2} \boldsymbol {I}\right) \tag {2}
$$

Moreover, it has the following Markovian transition probabilities: 

$$
q \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {s}\right) = \mathcal {N} \left(\gamma_ {t | s} \mathbf {x} _ {s}, \sigma_ {t | s} ^ {2} \boldsymbol {I}\right) \tag {3}
$$

with the forward posteriof the noise coefficient) $\begin{array} { r } { \gamma _ { t | s } = \frac { \gamma _ { t } } { \gamma _ { s } } } \end{array}$ rward posterior variance (or square. For a Gaussian diffusion process, $\sigma _ { t | s } ^ { 2 } = \sigma _ { t } ^ { 2 } - \gamma _ { t | s } ^ { 2 } \sigma _ { s } ^ { 2 }$ $0 < s < t < T$ given that $q ( \mathbf { x } _ { s } | \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) \propto { \dot { q } } ( \mathbf { x } _ { t } | \mathbf { x } _ { s } ) q ( \mathbf { x } _ { s } | \mathbf { x } _ { 0 } )$ , one can analytically derive a backward process that is also Gaussian, and has the following marginal distribution: 

$$
q \left(\mathbf {x} _ {s} \mid \mathbf {x} _ {t}, \mathbf {x} _ {0}\right) = \mathcal {N} \left(\boldsymbol {\mu} _ {t \rightarrow s}, \sigma_ {t \rightarrow s} ^ {2} \boldsymbol {I}\right). \tag {4}
$$

The backward posterior variance $\sigma _ { t  s } ^ { 2 }$ has the following form: 

$$
\sigma_ {t \rightarrow s} ^ {2} = \left(\frac {1}{\sigma_ {s} ^ {2}} + \frac {\gamma_ {t | s} ^ {2}}{\sigma_ {t | s} ^ {2}}\right) ^ {- 1} \tag {5}
$$

and the backward posterior mean $\mu _ { t  s }$ is formulated as: 

$$
\boldsymbol {\mu} _ {t \rightarrow s} = \sigma_ {t \rightarrow s} ^ {2} \left(\frac {\gamma_ {t | s}}{\sigma_ {t | s} ^ {2}} \mathbf {x} _ {t} + \frac {\gamma_ {s}}{\sigma_ {s} ^ {2}} \mathbf {x} _ {0}\right). \tag {6}
$$

Samples can be generated by simulating the reverse Gaussian process with the posteriors in Eq. (5) and Eq. (6). A practical issue is that Eq. (6) itself depends on the unknown $\mathbf { x } _ { \mathrm { 0 } }$ , the sample we are trying to generate. To overcome this, one can instead approximate the analytic reverse process in which $\mathbf { x } _ { \mathrm { 0 } }$ is replaced by its approximator $\hat { \mathbf { x } } _ { 0 }$ , learned by a deep neural network $f _ { \theta } ( \pmb { x } _ { t } , t )$ . The network can learn to directly predict $\mathbf { x } _ { \mathrm { 0 } }$ given an $\mathbf { x } _ { t }$ (a sample with a level of noise that corresponds to time 

t), but previous work has shown that it is beneficial to instead optimize the network to learn the approximator $\hat { \boldsymbol { \epsilon } } _ { t }$ . $\hat { \boldsymbol { \epsilon } } _ { t }$ predicts the unscaled Gaussian white noise that was injected at time t. $\hat { \mathbf { x } } _ { 0 }$ can then be obtained via Eq. (7), which follows from Eq. (1). 

$$
\hat {\mathbf {x}} _ {0} = \frac {1}{\gamma_ {t}} \mathbf {x} _ {t} - \frac {\sigma_ {t}}{\gamma_ {t}} \hat {\boldsymbol {\epsilon}} _ {t} \tag {7}
$$

Edge-preserving filters in image processing. In this work, we aim to choose $\gamma _ { t }$ and $\sigma _ { t }$ such that we obtain a diffusion process that injects noise in a content-aware manner. To do this, we are inspired by the field of image processing, where a classic and effective technique for denoising is edge-preserved filtering via anisotropic diffusion (Weickert, 1998). To overcome the problem of destroying relevant structural information in the image when applying an isotropic filter, Perona and Malik (1990) instead propose an anisotropic diffusion process of the form: 

$$
\mathbf {x} _ {t} = \mathbf {x} _ {0} + \int_ {0} ^ {t} \mathbf {c} \left(\mathbf {x} _ {s}, s\right) \Delta \mathbf {x} _ {s} d s \tag {8}
$$

where the diffusion coefficient $\mathbf { c } ( \mathbf { x } _ { s } , s )$ takes the following form: 

$$
\mathbf {c} (\mathbf {x}, t) = \frac {1}{\sqrt {1 + \frac {| | \nabla \mathbf {x} _ {t} | |}{\lambda}}} \tag {9}
$$

where $| | \nabla \mathbf { x } | |$ is the gradient magnitude image, and $\lambda$ is the edge sensitivity. Intuitively, in the regions of the image where the gradient response is high (on edges), the diffusion coefficient will be smaller, and therefore the signal gets less distorted there. The edge sensitivity $\lambda$ determines how sensitive the diffusion coefficient is to the image gradient response. 

Inspired by the anisotropic diffusion coefficient presented in Eq. (9), we aim to design a linear diffusion process that incorporates edge-preserving noise. Our hope is that by doing this, the generative diffusion model will better learn the underlying geometrical structures of the target distribution, leading to a more effective generative denoising process. To obtain our content-aware linear diffusion process, we apply the idea of edge-preserved filtering to the noise term of Eq. (1). We cannot directly use (Perona and Malik, 1990)’s formulation because their time-dependent diffusion coefficient makes the process nonlinear. Instead, we make the coefficient depend only on $\mathbf { x } _ { \mathrm { 0 } }$ : 

$$
\mathbf {x} _ {t} = \gamma_ {t} \mathbf {x} _ {0} + \frac {b}{\sqrt {1 + \frac {| | \nabla \mathbf {x} _ {0} | |}{\lambda (t)}}} \boldsymbol {\epsilon} _ {t} \tag {10}
$$

Where $b$ is the noise coefficient’s numerator and can be chosen as desired. To study the impact of non-isotropic edge-preserving noise on the generative diffusion process, we chose our parameters√ √ $\gamma _ { t } = \sqrt { \bar { \alpha } _ { t } }$ and $b = \sqrt { 1 - \bar { \alpha _ { t } } }$ such that it closely matches the well-studied forward process of (Ho et al., 2020), but nothing prevents us from making different choices for $\gamma _ { t }$ and $b$ . Note that the noise coefficient in Eq. (1) becomes a tensor $\sigma _ { t }$ instead of a scalar $\sigma _ { t }$ for our process. Intuitively, we preserve edges by reducing noise based on the edges in the original image. In our formulation, we also consider $\lambda$ to be time-varying (more details in section Section 4.2). 

# 4 AN EDGE-PRESERVING GENERATIVE PROCESS

# 4.1 FORWARD HYBRID NOISE SCHEME

The forward edge-preserving process described in Eq. (10) in its pure form is not very meaningful in our setup. This is because if the edges are preserved all the way up to time $t = T$ , we end up with a rather complex distribution $p _ { T } ( x )$ that we cannot analytically take samples from. Instead, we would like to end up with a well-known prior distribution at time $t = T$ , such as the standard normal distribution. To achieve this, we instead consider the following hybrid forward process: 

$$
\mathbf {x} _ {t} = \gamma_ {t} \mathbf {x} _ {0} + \frac {b}{(1 - \tau (t)) \sqrt {1 + \frac {| | \nabla \mathbf {x} _ {0} | |}{\lambda (t)}} + \tau (t)} \boldsymbol {\epsilon} _ {t} \tag {11}
$$

The function $\tau ( t )$ now appearing in the denominator of the diffusion coefficient is the transition function. When $\tau ( t ) < 1$ , we obtain edge-preserving noise (the edge-preservation is stronger when $\dot { \boldsymbol { \tau } } ( t ) \approx 0 )$ . The turning point where $\tau ( t ) = 1$ is called the transition point $t _ { \Phi }$ . At the transition point,√ we switch over to isotropic noise with scalar noise coefficient √ $\sigma _ { t } = b$ (note that we chose $\gamma _ { t } = \sqrt { \bar { \alpha _ { t } } }$ and $b = \sqrt { 1 - \bar { \alpha _ { t } } } )$ . 

This approach allows us to flexibly design noise schedulers that start off with edge-preserving noise and towards the end of the forward process fall back to an isotropic diffusion coefficient. Practically, one can choose any function for $\tau ( t )$ , as long as it maps to [0; 1] and $\tau ( t ) = 1$ for $t$ in proximity to $T$ . We performed an ablation for different transition functions in Section 5.1. 

Observe how our diffusion process generalizes over existing isotropic processes: by setting $\tau ( t ) = 1$ constant, we simply obtain an isotropic process with signal coefficient $\gamma _ { t }$ and noise coefficient $\sigma _ { t } = b$ . Choosing any other non-constant function for $\tau ( t )$ leads to a hybrid diffusion process that consists of an edge-preserving stage and an isotropic stage (starting at $\tau ( t ) = 1 $ ). 

# 4.2 TIME-VARYING EDGE SENSITIVITY $\lambda ( t )$

The edge sensitivity parameter $\lambda$ controls the level of detail preserved along image edges. Very low values of (e.g. $\lambda = 1 e - 5 )$ will retain almost all fine details. The more we increase $\lambda$ , the less details will be preserved. When $\lambda$ becomes very high (e.g. $\lambda = 1$ ), the process becomes nearly isotropic. Our ablation study (Section 5.1) explores this effect in detail. We found that constant $\lambda$ -values harm sample quality: too low values results in unrealistic, ”cartoonish” images, while too high values diminish the effectiveness of the edge-preserving diffusion model, making the model behave almost like an isotropic process. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/657b43f33bb9ff4dfe4423a7d6fb3f5746c48441b3c42c457bfcbd612cd82210.jpg)


To overcome this, we instead consider a time-varying edge sensitivity $\lambda ( t )$ . We set an interval $[ \lambda _ { m i n } ; \lambda _ { m a x } ]$ that bounds the possible values for the time-varying edge sensitivity. The function that governs $\lambda ( t )$ within this interval can in theory again be chosen freely. We have so far experimented with a linear function and a sigmoid function. We experienced that a linear function for $\lambda ( t )$ resulted in higher sample quality and therefore used this function for our experiments. Additionally, we have attempted to optimize the interval $[ \lambda _ { m i n } ; \lambda _ { m a x } ]$ , but this led to unstable behaviour. 

# 4.3 BACKWARD PROCESS POSTERIORS AND TRAINING

Given our forward hybrid diffusion process introduced in Section 4.1, we can derive the actual formulations for the posterior mean $\mu _ { t  s }$ and variance $\sigma _ { t  s } ^ { 2 }$ for the corresponding backward process. To do this, we simply fill in Eq. (5) and Eq. (6) with our choices for the signal coefficient $\gamma _ { t }$ and vriance $\pmb { \sigma } _ { t } ^ { 2 }$ . Recall that we chose gain a tensor, contrary $\pmb { \sigma } _ { t } ^ { 2 }$ to be a tensor, which is why the backward posterior isotropic diffusion processes considered in previous $\pmb { \sigma } _ { t  s } ^ { 2 }$ 

We first introduce an auxiliary variable $\sigma ^ { 2 } ( t )$ , which represents the variance of our forward process at a given time $t$ . This is simply the square of our choice for the noise coefficient $\sigma _ { t }$ formulated in Eq. (11): 

$$
\boldsymbol {\sigma} ^ {2} (t) = \frac {1 - \bar {\alpha} _ {t}}{(1 - \tau (t)) ^ {2} \left(1 + \frac {| | \nabla \mathbf {x} _ {0} | |}{\lambda (t)}\right) + 2 \left((1 - \tau (t)) \sqrt {1 + \frac {| | \nabla \mathbf {x} _ {0} | |}{\lambda (t)}} \tau (t)\right) + \tau (t) ^ {2}} \tag {12}
$$

Here $\bar { \alpha _ { t } }$ has the same meaning as earlier described in Section 3. We now have the backward posterior variance $\sigma _ { t  s } ^ { 2 }$ : 

$$
\boldsymbol {\sigma} _ {t \rightarrow s} ^ {2} = \left(\frac {1}{\boldsymbol {\sigma} ^ {2} (t)} + \frac {\frac {\bar {\alpha} _ {t}}{\bar {\alpha} _ {s}}}{\boldsymbol {\sigma} ^ {2} (t) - \frac {\bar {\alpha} _ {t}}{\bar {\alpha} _ {s}} \boldsymbol {\sigma} ^ {2} (s)}\right) ^ {- 1} \tag {13}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/9f9927c0d1ba461a375aac56f6d9ebe4001b39e93c5e8f7d282c89589b100ab4.jpg)



Figure 2: We visually compare the impact of our edge-preserving noise on the generative process. In each column, we show predictions $\hat { \mathbf { x } } _ { 0 }$ at selected time steps. Our method converges significantly faster to a sharper and less noisy image than its isotropic counterpart. This is evident by the earlier emergence (from $t = 4 0 0$ ) of structural details like the pattern on the cat’s head, eyes, and whiskers with our approach.


and the backward posterior mean $\mu _ { t  s }$ 

$$
\boldsymbol {\mu} _ {t \rightarrow s} = \sigma_ {t \rightarrow s} ^ {2} \left(\frac {\frac {\sqrt {\bar {\alpha} _ {t}}}{\sqrt {\bar {\alpha} _ {s}}}}{\sigma^ {2} (t) - \frac {\bar {\alpha} _ {t}}{\bar {\alpha} _ {s}} \sigma^ {2} (s)} \mathbf {x} _ {t} + \frac {\sqrt {\bar {\alpha} _ {s}}}{\sigma^ {2} (s)} \mathbf {x} _ {0}\right) \tag {14}
$$

Given Eq. (13) and Eq. (14), the only unknown preventing us from simulating the Gaussian backward process is $\mathbf { x } _ { \mathrm { 0 } }$ . 

Note that $\mathbf { x } _ { \mathrm { 0 } }$ in our case depends on a non-isotropic noise. Therefore, we cannot just use an isotropic approximator $\hat { \boldsymbol { \epsilon } } _ { t }$ for the isotropic noise $\epsilon _ { t }$ to predict $\hat { \mathbf { x } } _ { 0 }$ via Eq. (7). Instead, we need a model that can predict the non-isotropic noise $\sigma _ { t } \epsilon _ { t }$ . We introduce the loss function that trains such an approximator: 

$$
\mathcal {L} = \left\| f _ {\theta} \left(\boldsymbol {x} _ {t}, t\right) - \boldsymbol {\sigma} _ {t} \boldsymbol {\epsilon} _ {t} \right\| ^ {2}. \tag {15}
$$

It is very similar to the loss function used in DDPM, with the difference that our model explicitly learns to predict the non-isotropic edge-preserving noise $( \sigma _ { t } \epsilon _ { t } )$ . In Appendix D, we show how our loss formulation can be adapted to approximate the negative log-likelihood. $f _ { \theta } ( \pmb { x } _ { t } , t )$ stands for the time-conditioned U-Net used to approximate the time-varying noise function. The visual difference between the backward process of an isotropic diffusion model (DDPM) and ours is shown in Fig. 2. Our formulation introduces a negligible overhead. The only additional computation that needs to be performed is the image gradient $\lvert \lvert \nabla \mathbf { x } _ { 0 } \rvert \rvert$ , which can be done very efficiently on modern GPUs. We have not noticed any significant difference in training times between vanilla DDPM and our method. 

# 5 EXPERIMENTS

Implementation details We provide the implementation details for our experiments in Appendix E. Please also find our training performance analysis on different frequency bands in Appendix B. 

Unconditional image generation We show unconditional image generation results in Fig. 3 and Appendix F. The corresponding FID metrics are listed in Table 1. We observe improvements w.r.t. all baselines both visually and quantitatively. While the visual improvement over DDPM is subtle, our model generally demonstrated greater robustness to artifacts. 

We attribute these improvements to the explicit training of our model on predicting the non-isotropic noise associated with the edges in the dataset. We also performed comparisons in the 


Table 1: Quantitative FID score comparisons for unconditional image generation among IHDM (Rissanen et al., 2023), DDPM (Ho et al., 2020), BNDM (Huang et al., 2024a) and our method across different datasets.


<table><tr><td>FID (↓)</td><td>CelebA(1282)</td><td>LSUN-Church(1282)</td><td>AFHQ-Cat(1282)</td></tr><tr><td>IHDM</td><td>89.67</td><td>119.34</td><td>53.86</td></tr><tr><td>DDPM</td><td>28.17</td><td>31.00</td><td>17.60</td></tr><tr><td>BNDM</td><td>26.35</td><td>29.86</td><td>14.54</td></tr><tr><td>Ours</td><td>26.15</td><td>23.17</td><td>13.06</td></tr></table>

latent space, which are listed in Table 2 in Appendix F. For latent space diffusion (CelebA(2562) 


Cat (128
2) CelebA (128
2) Church (128
2)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/dfa27cc80b96fddfde0c342bc538bc83b6d1443ea0976dfabac1225b7911077d.jpg)



IHDM (Rissanen et al.)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/75c68a2983a0626f0521cb878dd93528ac58044b1e75eccd9eda4b269771b1f1.jpg)



BNDM (Huang et al.)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/6b18f6972b62955fb8b309cd45d58b80befd8247ca46e210459cf21769d37b56.jpg)



DDPM (Ho et al.)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/306477fee1bb864f1f8b4ab43fc96d07b5b1993ecb65d911fad4bf35b1c552db.jpg)



Ours



Figure 3: We compare unconditionally generated samples for IHDM, BNDM and DDPM with our model. While qualitative improvements are subtle, ours performs consistently better quantitatively. Corresponding FID scores can be found in Table 1. Additional results are presented in the appendix.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/c690474988ad058783759fb1249ea3c91f30c65be9fa79d0d83615dbc2d39313.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/cb314a2a826d423bc34321c094cef5ce5433dcf0abf57730353dae03cef4fc4c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/c60ab4b9d810353e7636562911a8956689c8f341c257dd617a59e0ee04f3ac32.jpg)



Synthetic stroke painting


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/1060b79e7d1fd6e2e08837523930ac5dab584e54c62a9e6758060a5384bdd936.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/79164188ca6915d1ea03982a2fe704d25559c76865f5081993796a026fdd95df.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/66ef569ae710e871d6b8b6bf4b61f2d84773defa9079813e50317d799d0cf098.jpg)



BNDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/1f3da3089850b7c5ac507840d2b6c5392d8a4d885e620508b6bcd00d521b713d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/c79fc382bad150d02a90d1e7e120fc51d9ec64dc1149026c1ab777f142bc6132.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/19963f8da8e28ec00fe375e92d72ce4e676f79c48abeb12388d5f83674417f50.jpg)



DDPM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/e2c98754c9ea7835a8482f3187e193c160a8ecdb408a39ddf0a440e647473b3b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/d9c65e4a28c382f6e69206ac414c0033f1b44371511aff580a2d393c72394ac5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/4e51ec97f57a3348ae14b2a78f80b76c012dc41ac1970a534117f206db94df08.jpg)



Ours


<table><tr><td>FID (↓)</td><td>BNDM</td><td>DDPM</td><td>Ours</td></tr><tr><td>CelebA(1282)</td><td>68.0</td><td>45.80</td><td>39.08</td></tr><tr><td>Church(1282)</td><td>93.81</td><td>72.54</td><td>56.14</td></tr><tr><td>Cat(1282)</td><td>51.05</td><td>27.61</td><td>23.50</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/8dd535cba652c0228414e60899feb60cef4bf2802e46ab520a46cb9cdd523168.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/eaf055a83657781aef3d07c3cd9085fa83735ae4a56c42dcb50ab6bcd7fd9650.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/c584ee636e5f844f1a1865edd2405194b217be38008894fdfde1f15c35b485b1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/fa311ccee06ccda816ab4614b9a07cf1ebaeebd28d8c6889acbfc08d25e59392.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/52e50f85da46ac280b07cd71f359d358c0ca34a91c04eb0d35d3c103a949737a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/6f559e6dfb0a0f531ec34e96609caa17742614f8f83a587b699b4a42ab515092.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/9930c27346717dbf7464371489983a6764d10b5b8c40de86e826ab7fd7d9af4c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/e19c7d3812275048cba5013c7f9389ce6e95912e95bedcd92e7cdc587994696d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/c6f454ec3bf0e2d6ab64489e5c697f6efb097889593c0a80fcc2afa6f3c2960d.jpg)



Human painting



Ours



Figure 4: Left: Various diffusion models applied to the SDEdit framework (Meng et al., 2022) are shown. The leftmost column displays the stroke-based guide (via k-means clustering applied to an image), with the other three columns showing the model outputs. Overall, our model shows sharper details with less distortions compared to other models, leading to a better visual and quantitative performance. The corresponding FID scores are shown in the top right column. Right: Our model also effectively uses human-drawn paintings as shape guides, with particularly precise adherence to details, such as the orange patches on the cat’s fur, unlike DDPM (middle column).


and AFHQ- $\mathrm { C a t ( 5 1 2 ^ { 2 } ) } ,$ ), although our model is slightly outperformed on the FID metric, the visual quality of our samples is often comparable, and at times even superior (see Fig. 11 and Fig. 12 in Appendix F). This highlights the known limitations of FID, as it doesn’t always reliably capture visual quality (Liu et al., 2018). Our reported FID scores may be numerically higher than those in other works, but absolute comparisons between two different papers are unreliable unless the FID implementation, backbone, and training conditions are identical, as absolute FID values are highly implementation-dependent. Training $^ +$ inference time and memory consumptions of all methods are shown in Table 3 in the Appendix. 

Stroke-guided image generation (SDEdit) Motivated by the hope of improved adherence to shapebased priors like stroke paintings, we applied our edge-preserving diffusion model to SDEdit (Meng et al., 2022) for stroke-based image generation. Using k-means clustering, we converted 1000 training images into stroke paintings and fed them into SDEdit with different diffusion models as backbone, including BNDM (Huang et al., 2024a) and DDPM (Ho et al., 2020). With a hijack 

point of $0 . 5 5 T$ , we computed FID scores to measure which model best reconstructs the original image, given a stroke-based prior. Our model better preserves guiding priors, reducing artifacts and improving performance. Further evaluations on precision/recall (Tables 4 and 5) and CLIP-score (Table 6) confirm it maintains diversity and enhances semantic preservation compared to its isotropic counterpart DDPM. These results highlight the model’s potential for image-editing tasks, especially in scenarios where preserving geometric details is crucial. 

# 5.1 ABLATION STUDY

Impact of transition function $\tau ( t )$ . We have experimented with three different choices for the transition function $\tau ( t )$ : linear, cosine and sigmoid. While cosine and sigmoid show similar performance, we found that having a smooth linear transition function significantly improves the performance of the model. A qualititative and quantitative comparison between the choices is presented in the inline figure below. 

Impact of transition points $t _ { \Phi }$ . We have investigated the impact of the transition point $t _ { \Phi }$ on our method’s performance by considering 3 different diffusion schemes: $2 5 \%$ edge-preserving - $7 5 \%$ isotropic, $50 \%$ isotropic - $50 \%$ edge-preserving and $7 5 \%$ edge-preserving - $2 5 \%$ isotropic. A visual example for AFHQ-Cat $( 1 \bar { 2 } 8 ^ { 2 } )$ is presented in the inline figure on the right. We have experienced that there are limits to how far the transition point can be placed without sacrificing sample quality. Visually, we observe that the further the transition point is placed, the less details the model generates. The core shapes however stay intact. This is illustrated well by Fig. 7 in Appendix F. For the datasets we tested on, we found that the $5 0 \% - 5 0 \%$ diffusion scheme works best in terms of FID metric and visual sharpness. This again becomes apparent in Fig. 7: although the samples for $t _ { \Phi } = 0 . 2 5$ contain slightly more details, 

the samples for $t _ { \Phi } = 0 . 5$ are significantly sharper. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/b627edcb40332c10e612ac35806ad4fed94cc20b01b6fa1d7cdcb5a6eee801cb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/2fd04629f26c260155114ae56156a57ff8b390ab6279aad4b5bce5115f716165.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/60229e758bd62e2c76ad6c2ccfc4eb3af908145966ca61a37214a6812423df7e.jpg)


Impact of edge sensitivity $\lambda ( t )$ . As shown in the above inline figure, lower constant $\lambda ( t )$ values lead to less detailed, more flat, ”water-painting-style” samples. Intuitively, a lower $\lambda ( t )$ corresponds to stronger edge-preserving noise and our model is explicitly trained accordingly to better learn the core structural shapes instead of the high-frequency details that we typically find in interior regions. Our time-varying choice for $\lambda ( t )$ works better than other settings in our experiments, by effectively balancing the preservation of structural information across different granularities of detail. 

# 6 CONCLUSION

We introduced a new class of edge-preserving generative diffusion models that generalize isotropic models with negligible overhead. Our hybrid process consists of an edge-preserving phase, which maintains structural details, followed by an isotropic phase to ensure convergence to a known prior. This decoupled approach better captures low-to-mid frequencies and accelerates convergence to sharper predictions. It outperforms several state-of-the-art models on both unconditional and shapeguided generative tasks. Future work could explore extending our non-isotropic framework to video generation for better temporal consistency, as well as automating hyperparameter optimization. 

# REFERENCES



Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Cold diffusion: Inverting arbitrary image transforms without noise. Advances in Neural Information Processing Systems, 36, 2023. 





Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8188–8197, 2020. 





Giannis Daras, Mauricio Delbracio, Hossein Talebi, Alex Dimakis, and Peyman Milanfar. Soft diffusion: Score matching with general corruptions. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. 





Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Score-based generative modeling with criticallydamped langevin diffusion. In International Conference on Learning Representations, 2022. 





Mathias Eitz, James Hays, and Marc Alexa. How do humans sketch objects? ACM Trans. Graph. (Proc. SIGGRAPH), 31(4):44:1–44:10, 2012. 





Michael Elad, Bahjat Kawar, and Gregory Vaksman. Image denoising: The deep learning revolution and beyond—a survey paper. SIAM Journal on Imaging Sciences, 16(3):1594–1654, 2023. 





U. G. Haussmann and E. Pardoux. Time Reversal of Diffusions. The Annals of Probability, 14 (4):1188 – 1205, 1986. doi: 10.1214/aop/1176992362. URL https://doi.org/10.1214/ aop/1176992362. 





Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 





Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 





Emiel Hoogeboom and Tim Salimans. Blurring diffusion models. In The Eleventh International Conference on Learning Representations, 2023. 





Xingchang Huang, Corentin Salaun, Cristina Vasconcelos, Christian Theobalt, Cengiz Oztireli, and Gurprit Singh. Blue noise for diffusion models. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024a. 





Yi Huang, Jiancheng Huang, Jianzhuang Liu, Mingfu Yan, Yu Dong, Jiaxi Lyu, Chaoqi Chen, and Shifeng Chen. Wavedm: Wavelet-based diffusion models for image restoration. IEEE Transactions on Multimedia, 2024b. 





Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696–21707, 2021. 





Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 





Tuomas Kynka¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved ¨ precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019. 





Cheng-Han Lee, Ziwei Liu, Lingyun Wu, and Ping Luo. Maskgan: Towards diverse and interactive facial image manipulation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 





Shaohui Liu, Yi Wei, Jiwen Lu, and Jie Zhou. An improved evaluation framework for generative adversarial networks. arXiv preprint arXiv:1803.07474, 2018. 





Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. 





Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 





Pietro Perona and Jitendra Malik. Scale-space and edge detection using anisotropic diffusion. IEEE Transactions on pattern analysis and machine intelligence, 12(7):629–639, 1990. 





Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 





Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation. In The Eleventh International Conference on Learning Representations, 2023. 





Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 





Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 





Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 





Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. 





George Stein, Jesse Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L Caterini, Eric Taylor, and Gabriel Loaiza-Ganem. Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 





Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016. 





Vikram Voleti, Christopher Pal, and Adam M Oberman. Score-based denoising diffusion with non-isotropic gaussian noise models. In NeurIPS 2022 Workshop on Score-Based Methods, 2022. 





Joachim Weickert. Anisotropic diffusion in image processing, volume 1. Teubner Stuttgart, 1998. 





Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015. 





Xi Yu, Xiang Gu, Haozhi Liu, and Jian Sun. Constructing non-isotropic gaussian diffusion model using isotropic gaussian diffusion model for image editing. Advances in Neural Information Processing Systems, 36, 2024. 



# A RELATION TO SCORE-BASED GENERATIVE MODELING

# A.1 TRAINING OF A SCORE-BASED MODEL

Given any $\mathbb { R } ^ { d }$ -valued $( d \in \mathbb { N } ,$ ) forward process $( \mathbf { x } _ { t } ) _ { t \in [ 0 , T ] }$ such that $\mathbf { x } _ { \mathrm { 0 } }$ is distributed to a desired data distribution $\mu$ on $\mathbb { R } ^ { d }$ , a score-based model can be trained by minimizing the loss: 

$$
\mathcal {L} (\tilde {s}) := \int_ {0} ^ {T} \alpha (t) \int \mu (\mathrm {d} x) \operatorname {E} _ {x} \left[ \left\| \tilde {s} (t, \mathbf {x} _ {t}) \right\| ^ {2} + 2 \nabla_ {x} \cdot \tilde {s} (t, \mathbf {x} _ {t}) \right] \mathrm {d} t, \tag {16}
$$

where $T \in ( 0 , \infty )$ , $\alpha : [ 0 , T ]  [ 0 , \infty )$ is a suitable weighting function and $\tilde { s } : [ 0 , T ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the desired score estimate. The score is defined to be 

$$
s (t, \cdot) := \nabla \ln p _ {t}, \tag {17}
$$

where $p _ { t }$ denotes the density of $\mathbf { x } _ { t }$ with respect to the Lebesgue measure on $\mathbb { R } ^ { d }$ , which we assume to exist for all $t \in [ 0 , T ]$ . 

In order to ensure stability and convergence of the training, $\alpha ( t )$ is usually chosen to be inversely proportional to the expected squared norm: 

$$
\mathrm {E} \left[ \left\| s (t, \mathbf {x} _ {t}) \right\| ^ {2} \right] \tag {18}
$$

of the true score $s ( t , \cdot )$ . 

In practice, $\mathbf { x } _ { t }$ is often conditionally Gaussian given $\mathbf { x } _ { \mathrm { 0 } }$ . In that case, the suggested choice for $\alpha ( t )$ can be easily computed. In fact, the score of a Gaussian random variable with covariance matrix $\Sigma$ is given by: 

$$
\operatorname {t r} \left(\Sigma^ {- 1}\right). \tag {19}
$$

# A.2 SAMPLING IN A SCORE-BASED MODEL

Assuming that $( \mathbf { x } _ { t } ) _ { t \in [ 0 , T ] }$ is the solution of a stochastic differential equation (SDE) 

$$
\mathrm {d} \mathbf {x} _ {t} = b (t, \mathbf {x} _ {t}) \mathrm {d} t + \sigma (t, \mathbf {x} _ {t}) \mathrm {d} \mathbf {w} _ {t} \tag {20}
$$

for some drift $b : [ 0 , T ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , diffusion coefficient $\sigma : [ 0 , T ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d \times d }$ and Wiener process $\mathbf { \Psi } ( \mathbf { w } _ { t } ) _ { t \in [ 0 , T ] }$ , a mild condition (Haussmann and Pardoux, 1986) on the drift and diffusion coefficent are sufficient to show that the reverse process 

$$
\overline {{\mathbf {x}}} _ {t} := \mathbf {x} _ {T - t} \quad \text {f o r} t \in [ 0, T ] \tag {21}
$$

is the solution of an SDE as well. In fact, in that case, $( \overline { { \mathbf { x } } } _ { t } ) _ { t \in [ 0 , T ] }$ is the solution 

$$
\mathrm {d} \overline {{\mathbf {x}}} _ {t} = \bar {b} (t, \overline {{\mathbf {x}}} _ {t}) \mathrm {d} t + \bar {\sigma} (t, \overline {{\mathbf {x}}} _ {t}) \mathrm {d} \overline {{\mathbf {w}}} _ {t}, \tag {22}
$$

where 

$$
\bar {b} (t, x) := (\nabla_ {x} \cdot \Sigma) (T - t, x) + \Sigma (T - t, x) s (T - t, x) - b (T - t, x); \tag {23}
$$

$$
\bar {\sigma} (t, x) := \sigma (T - t, x) \tag {24}
$$

$$
\Sigma := \sigma \sigma^ {*} \tag {25}
$$

and $( \overline { { \mathbf { w } } } _ { t } ) _ { t \in [ 0 , T ] }$ is another Wiener process. Since, by assumption, $\overline { { \mathbf { x } } } _ { T } = \mathbf { x } _ { 0 }$ is distributed according to our data distribution $\mu$ , sampling from the data distribution can be achieved by simulating the SDE (22). In practice, the usually unknown score $s$ is replaced by the score estimate $\tilde { s }$ learned during the training process. 

# A.3 INTEGRATING OUR FORWARD PROCESS TO THE SCORE-BASED FRAMEWORK

We can immediately use our forward process (11) for score-based generative modeling. To do so, we can interpret the forward process (11) as the solution of the SDE: 

$$
\mathrm {d} \mathbf {y} _ {t} = \beta_ {t} \mathrm {d} t + \varsigma_ {t} \mathrm {d} \mathbf {w} _ {t}; \tag {26}
$$

$$
\mathbf {y} _ {0} = 0, \tag {27}
$$

where 

$$
\beta_ {t} := \frac {\mathrm {d}}{\mathrm {d} t} b _ {t} \mathbf {x} _ {0}; \tag {28}
$$

$$
\varsigma_ {t} := \sqrt {2 \tilde {\sigma} _ {t} \frac {\mathrm {d}}{\mathrm {d} t} \sigma_ {t}} \tag {29}
$$

and 

$$
b _ {t} := \sqrt {\bar {\alpha} _ {t}}; \tag {30}
$$

$$
\sigma_ {t} := \frac {\sqrt {1 - \bar {\alpha} _ {t}}}{(1 - \tau (t)) \sqrt {1 + \frac {| | \nabla \mathbf {x} _ {0} | |}{\lambda (t)}} + \tau (t)}; \tag {31}
$$

$$
\tilde {\sigma} _ {t} := \sigma_ {t} - \sigma_ {0}. \tag {32}
$$

However, it is more natural to translate our basic idea directly to an SDE and consider: 

$$
\mathrm {d} \mathbf {y} _ {t} = b _ {t} \mathrm {d} t + \sigma_ {t} \mathrm {d} \mathbf {w} _ {t}; \tag {33}
$$

$$
\mathbf {y} _ {0} = \mathbf {x} _ {0} \tag {34}
$$

instead. For the solution $( \mathbf { y } _ { t } ) _ { t \in [ 0 , T ] }$ of an SDE of the form (33), $\mathbf { y } _ { t }$ is conditionally Gaussian given $\mathbf { y } _ { 0 }$ . Assuming $\mathbf { y } _ { 0 }$ is distributed according to the target data distribution $\mu$ , we can use the general procedure described in Appendix A.1 and Appendix A.2 to train the score and sample from $\mu$ . 

# B FREQUENCY ANALYSIS OF TRAINING PERFORMANCE

To better understand our model’s capacity of modeling the target distribution, we conducted an analysis on its training performance for different frequency bands. Our setup is as follows, we create 5 versions of the AFHQ-Cat128 dataset, each with a different cutoff frequency. This corresponds to convoluting each image in the dataset with a Gaussian kernel of a specific standard deviation $\sigma$ , representing a frequency band. For each frequency band, we then trained our model for a fixed amount of 10000 training iterations. We place a model checkpoint at every 1000 iterations, so we can also investigate the evolution of the performance over this training time. We measure the performance by computing the FID score between 1000 generated samples (for that specific checkpoint) and the original dataset of the corresponding frequency band. A visualization of the analyzed results is presented in the inline figure on the right. We found that our model 

FID score evolution over training time, per cutoÆ frequency 


a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/a0126ce51e318eec40811c5e29e9c4d07da7cd607c925b9a364c96393d3cb91b.jpg)



b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/3dee707bb369bb79641a2bb5b7d76e1e8640ee5b87c0ad2cba6e14f1d1c394d7.jpg)



c)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/dbb2e044f8040bbeea49a3fb1af9916f8e3e7eb2ed2a697c0c3bdd45b38490fb.jpg)



d )


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/853aba03c0e0eb0942fa4b49986d230149c3eedeabd9959dc67a9c5e93cebbba.jpg)


is able to learn the low-to-mid frequencies of the dataset significantly better than the isotropic model (DDPM). The figure shows the evolution of FID score over the first 10,000 training iterations per frequency band (larger $\sigma$ values correspond to lower frequency bands). a) and $b$ ) show performance in terms of FID score of DDPM and our model, respectively. c) shows their difference (positive favors our method). $d$ ) visualizes the information in 2D for a more accurate comparison. Our model significantly outperforms in low-to-mid frequency bands (lower FID is better). 

# C MOTIVATION BEHIND OUR HYBRID NOISE PROCESS

A valid observation to make is that given our hybrid forward process with two distinct stages, the edges are preserved longer, but still lost in the end. How does longer preservation of edges help the generative process? A first thing to note is that the longer preservation of edges by itself does not have any impact, if we still let the model predict isotropic noise. Secondly, by modifying the forward process to be an edge-preserving one, the backward posterior formulation will also change and will rely on a non-isotropic variance, as discussed Eq. (13). It is the combination of edge-preserving noise, together with our structure-aware loss function that makes the model work. Furthermore, our 

frequency analysis (Appendix B) has quantitatively shown that our decoupling approach is beneficial to learning the low-to-mid frequencies of the target dataset. This is consistent with recent work on wavelet-based diffusion models (Huang et al., 2024b), that demonstrates it is advantageous to learn low-frequency content separately from high-frequency content in the wavelet domain, using two distinct modules in their architecture. Instead, we use two distinct diffusion stages, one that focuses on lower-frequency primary structural content (edge-preserving stage), and one that focuses on fine-grained high-frequency details (isotropic stage). 

# D HOW NEGATIVE LOG LIKELIHOOD CAN BE APPROXIMATED

In this section we explain how negative log-likelihood in the original DDPM Ho et al. (2020) paper can be approximated with our formulation. 

The denoising probabilistic model paradigm defined in the DDPM paper defines the loss by minimizing a variational upper bound on the negative log likelihood. Because our noise is still Gaussian, the derivation they make in Eq. (3) to (5) of their paper still holds for us. The difference however is that we are non-isotropically scaling our noise based on the image content. As a result, our methods differ on Eq. (8) in their paper. Instead, we end up with the following form: 

$$
L _ {t - 1} = \mathbb {E} _ {q} \left[ \Sigma^ {- 1} \left(\tilde {\mu_ {\mathbf {t}}} \left(\mathbf {x} _ {t}, \mathbf {x} 0\right) - \mu \theta \left(\mathbf {x} _ {t}, t\right)\right). \left(\tilde {\mu_ {\mathbf {t}}} \left(\mathbf {x} _ {t}, \mathbf {x} 0\right) - \mu \theta \left(\mathbf {x} _ {t}, t\right)\right) \right] \tag {35}
$$

In essence, for our formulation that considers non-isotropic Gaussian noise, we need to apply a different loss scaling for each pixel. 

Our formulation still provides an analytical variational upper bound to approximate the negative log-likelihood. While our heuristic loss function (Eq. (15)) already proved effective for approximating non-isotropic noise corresponding to structural content in the data, a more accurate KL-divergence loss would include the scaling discussed above. 

# E IMPLEMENTATION DETAILS OF EXPERIMENTS

We compare our method against three baselines, namely DDPM (Ho et al., 2020), IHDM (Rissanen et al., 2023) and BNDM (Huang et al., 2024a). The motivation for comparing with the latter two works is that they also consider a non-isotropic form of noise. 

We perform experiments on two settings: pixel-space diffusion following the setting of Ho et al. (2020); Rissanen et al. (2023) and latent-space diffusion following (Rombach et al., 2022) noted as LDM in Table 2, where the diffusion process runs in the latent space. We use the following datasets: CelebA $\cdot 1 2 8 ^ { 2 }$ , 30,000 training images) (Lee et al., 2020), AFHQ-Cat $\mathrm { 1 2 8 ^ { 2 } }$ , 5,153 training images) (Choi et al., 2020), Human-Sketch ( $1 2 8 ^ { 2 }$ , 20,000 training images) (Eitz et al., 2012) (see Fig. 5) and LSUN-Church $\mathrm { 1 2 8 ^ { 2 } }$ , 126,227 training images) (Yu et al., 2015) for pixel-space diffusion. For latent-space diffusion (Rombach et al., 2022), we tested on CelebA $( 2 5 6 ^ { 2 } )$ ) and AFHQ-Cat $( 5 1 2 ^ { 2 } )$ . 

We used a batch size of 64 for all experiments in image space, and a batch size of 128 for all experiments in latent space. We trained AFHQ-Cat $( 1 2 8 ^ { 2 } )$ for 1000 epochs, AFHQ-Cat $( 5 1 2 ^ { 2 } )$ (latent diffusion) for 1750 epochs, CelebA $( 1 2 8 ^ { 2 } )$ for 475 epochs, CelebA(2562) (latent diffusion) for 1000 epochs and LSUN-Church $( 1 2 8 ^ { 2 } )$ for 90 epochs for our method and all baselines we compare to. Our framework is implemented in Pytorch (Paszke et al., 2017). For the network architecture we adopt the 2D U-Net from Rissanen et al. (2023). We use $\mathrm { T } = 5 0 0$ discrete time steps for both training and inference, except for AFHQ-Cat $( 1 2 8 ^ { 2 } )$ , where we used $\mathrm { T } = 7 5 0$ . To optimize the network parameters, we use Adam optimizer (Kingma and Ba, 2014) with learning rate $\mathrm { i } e ^ { - 4 }$ for latent-space diffusion models and $2 e ^ { - \bar { 5 } }$ for pixel-space diffusion models. We trained all datasets on $2 \mathbf { x }$ NVIDIA Tesla A40. 

For our final results in image space, we used a linear scheme for $\lambda ( t )$ that linearly interpolates between $\lambda _ { m i n } = 1 e ^ { - 4 }$ and $\bar { \lambda _ { m a x } } = 1 e ^ { - 1 }$ . We used a transition point $t _ { \Phi } = 0 . 5$ and a linear transition function $\tau ( t )$ . For latent diffusion, we used $\lambda _ { m i n } = 1 e ^ { - 5 }$ and $\lambda _ { m a x } = 1 e ^ { - 1 }$ , with $t _ { \Phi } = 0 . 5$ and a linear $\tau ( t )$ . 

To evaluate the quality of generated samples, we consider FID (Heusel et al., 2017). using the implementation from Stein et al. (2024), with Inception-v3 network (Szegedy et al., 2016) as backbone. We generate 30k images to compute FID scores for unconditional generation for all datasets. 

# F ADDITIONAL RESULTS

In this section, we provide additional results and ablations. 

Table 2 shows quantitative FID comparisons using latent diffusion (Rombach et al., 2022) models on all the baselines. 

Figure 8, Figure 9, Figure 10, Figure 11 and Figure 12 show more generated samples and comparisons between IHDM, DDPM on all previously introduced datasets. In Fig. 5 we show samples for the Human-Sketch $( 1 2 8 ^ { 2 } )$ data set specifically. This dataset was of particular interest to us, given the images only consist of high-frequency, edge content. Although we observed that this data is remarkably challenging for all methods, our model is able to consistently deliver visually better results. Note that although we report FID scores for this data set, they are very inconsistent with the visual quality of the samples. This is likely due to the Inception-v3 backbone being designed for continuous image data, leading to highly unstable results when applied to high-frequency binary data. 

Figure 7 shows an additional visualization of the impact $t _ { \Phi }$ for the LSUN-Church $( 1 2 8 ^ { 2 } )$ dataset. $t _ { \Phi } = 0 . 5$ works best in terms of FID metric, consistent to the results shown in Section 5.1. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/d0bfa6385986f6a03d5a72b8793438d3af1f1cd9c2aa118f6ac166fd07ea0110.jpg)



Figure 5: Generated unconditional samples for the Human Sketch $( 1 2 8 ^ { 2 } )$ dataset (Eitz et al., 2012). All models were trained for an equal amount of 575 epochs. Note that the FID scores are inconsistent with visual quality. The cause for this is the Inception-v3 backbone, which is designed for continuous image data, leading to highly unstable results when applied to high-frequency binary data like handdrawn sketches.



Table 2: Quantitative FID score comparisons on latent diffusion models (Rombach et al., 2022) among IHDM (Rissanen et al., 2023), DDPM (Ho et al., 2020), BNDM (Huang et al., 2024a) and our method.


<table><tr><td>Unconditional FID (↓)</td><td>CelebA(2562, latent)</td><td>AFHQ-Cat(5122, latent)</td></tr><tr><td>IHDM</td><td>88.12</td><td>28.09</td></tr><tr><td>DDPM</td><td>7.87</td><td>22.86</td></tr><tr><td>BNDM</td><td>10.93</td><td>13.62</td></tr><tr><td>Ours</td><td>13.89</td><td>18.91</td></tr></table>

Table 3: Our measurements on time and memory consumptions are based on data resolution (128x128) and a batch size of 64. Note that BNDM and Flow Matching make use of less inference steps $( \mathrm { T } 0 2 5 0$ vs. $\mathrm { T } { = } 5 0 0$ for Ours, DDPM and Simple Diffusion), and therefore are expected to be faster for inference. Our setup consisted of 2 NVIDIA Quadro RTX 8000 GPUS. We see that timings and memory usage of Ours is very similar to DDPM, suggesting that the Sobel filter we apply to approximate $| | \nabla \mathbf { x } | |$ brings minimal overhead. 

<table><tr><td></td><td>Ours</td><td>DDPM</td><td>BNDM</td></tr><tr><td>Training time (seconds per iteration)</td><td>1.12</td><td>1.11</td><td>0.74</td></tr><tr><td>Inference time (to generate 1 batch)</td><td>301.5</td><td>277.5</td><td>77.2</td></tr><tr><td>Inference Memory (GB)</td><td>9.16</td><td>9.16</td><td>10.3</td></tr></table>

Table 4: Shape-guided image generation (based on SDEdit (Meng et al., 2022)): precision (metric for realism) and recall (metric for diversity) scores (Kynka¨anniemi et al., 2019) for isotropic model ¨ DDPM, and our edge-preserving model. We consistently outperform in terms of precision, and again closely match in terms of recall. 

<table><tr><td rowspan="2">Shape-guided image generation</td><td colspan="2">Ours</td><td colspan="2">DDPM</td></tr><tr><td>Precision (↑)</td><td>Recall (↑)</td><td>Precision (↑)</td><td>Recall (↑)</td></tr><tr><td>AFHQ-Cat(1282)</td><td>0.93</td><td>0.80</td><td>0.92</td><td>0.66</td></tr><tr><td>CelebA(1282)</td><td>0.65</td><td>0.46</td><td>0.53</td><td>0.53</td></tr><tr><td>LSUN-Church(1282)</td><td>0.87</td><td>0.46</td><td>0.84</td><td>0.50</td></tr></table>

Table 5: Unconditional image generation: precision (metric for realism) and recall (metric for diversity) scores for isotropic model DDPM, and our edge-preserving model. While we slightly get outperformed, we find that our edge-preserving model closely matches DDPM on both metrics. therefore we would argue that edge-preserving noise minimally impacts diversity. 

<table><tr><td rowspan="2">Unconditional image generation</td><td colspan="2">Ours</td><td colspan="2">DDPM</td></tr><tr><td>Precision (↑)</td><td>Recall (↑)</td><td>Precision (↑)</td><td>Recall (↑)</td></tr><tr><td>AFHQ-Cat(1282)</td><td>0.76</td><td>0.20</td><td>0.77</td><td>0.21</td></tr><tr><td>CelebA(1282)</td><td>0.90</td><td>0.16</td><td>0.92</td><td>0.17</td></tr><tr><td>LSUN-Church(1282)</td><td>0.65</td><td>0.33</td><td>0.47</td><td>0.38</td></tr></table>

Table 6: We provide additional comparison for our shape-guided generative task (Meng et al., 2022) evaluated using the CLIP metric (Radford et al., 2021). Our method consistently outperforms the baselines on this metric, indicating that the generated images are more semantically aligned with the ground-truths (the original images used to generate the stroke paintings). We show several examples (Fig. 4 and Fig. 6) where our model solves visual artifacts that are apparent with other baselines, which can improve the semantical meaning of the generated image. 

<table><tr><td>CLIP</td><td>Ours</td><td>DDPM</td></tr><tr><td>AFHQ-Cat(1282)</td><td>88.97</td><td>88.78</td></tr><tr><td>CelebA(1282)</td><td>61.15</td><td>61.02</td></tr><tr><td>LSUN-Church(1282)</td><td>64.32</td><td>62.57</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/6818f73170ae1add60a8ffbab265953a1b09780a4888e9cc3d122e50f6b7f647.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/5e1ea742fcb0cbd06e78291ffdaaf6385f5b263152e3f99e77fef39ebf879a45.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/d56004fc1617e71dc13d4751e900b44dd5617503958c4ffd97aa6a231ad0f49e.jpg)



Synthetic painting


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/77808163349bbcf3a87e3108eaa5a31ecbed79eb086d17bb8778dd8abdaa85e5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/ad4af5174fbe7161e5c5f72aab866798cc73ea63bef2bb6d285d6289cae85b08.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/963210612b24b97ca9ba90e47a32d718f6f865cd1a848d8b1e2b7c962ff88c9c.jpg)



BNDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/d1e4cdf779d49c9097541722c9a05007ea31a594a1702bfb995346abe03614f1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/e2c2a15db4015a73230d2b3c11b09d11b97a2b7740034a71e413332bc834eacb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/e71b814743175cc06111332ad80f74431177ffd4a282cbf4f3bdd286f52bb0a4.jpg)



DDPM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/bcb1301b3be841cffe8388399cc76c266d827a2059c1926686f006f29633eb6e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/19276c1ae95b640c3ce96f9947a0dc7223c785042f18198f272a34f67340d12d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/cf51200fd01afa218d4ab4077a1c3167648482dbb481b01f76ad27c1cdd81439.jpg)



Ours


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/e08e68eb5929c8553e1ee194f4043353c0caf6933017ae90a15b4d049494509d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/93f99def8751caa30192e99ae6ae86fa2189e0a151a4462bc7287819c7ba604e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/ffe634fd1e19d35a8ac9001345004a76c34a8cb898ecdc8e5dd8b88bd29e2fff.jpg)



Synthetic painting


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/d950a78c2e656c536cae8f3fa6d12919def9c8edde2b26bcf87f40043fdd0681.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/8bf97504027a623ea2a3306fbcdb57c7595e19782ef6d93de56f436901269e28.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/2044a0a3d2cbbf138570497bff2e04e75fc1d0625e0fa4e83cc5d7c5dad15232.jpg)



BNDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/37b2f85e936972d0339d8f70f490c317a82df7089165b52167602349c61f8986.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/935d0b54fccd866712f729b34fdf9b60e0f23141757c26aa26e026dcf26be0b6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/de1eff6fa2973a0f3a08561eeaea01c85d07b7678fe6d350af1540dbbad6dc90.jpg)



DDPM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/3083d71d7f9d2df4ac138c71a77a3b182f5882e8a39003c5128744e1e520980d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/8158f37645c4f80dc4c51328d5c4d6ae8851631931e584fc623fe0266254185f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/d255859f33a4eb51b6f16b74adae23d82582152f4a3c08b2f4a056531759b14a.jpg)



Ours



Figure 6: More samples for our model and other baselines applied to SDEdit (Meng et al., 2022). Note how our model is able to generate sharper results that suffer less from artifacts. Although BNDM can generate satisfactory results in certain cases (e.g., cat and church), it often deviates from the stroke painting guide, potentially producing outcomes that differ significantly from the user’s original intent. In contrast, our method closely follows the stroke painting guide, accurately preserving both shape and color.



FID: 39.31


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/c77e195f35f9509672145811a8c23d37fb71772517783a03ef8d095ab0d3d4f3.jpg)



0.25



35.71


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/9c909deb613c4450cb52ddd3fa9cbe201dcb78164a20fcc6374e652cadf181db.jpg)



86.41


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/0d2a848faeb6eece516be70d149d03c6b1e292cafa04a506c20a8718375c029c.jpg)



0.75



Figure 7: Impact of location of transition point $t _ { \Phi }$ on sample quality, shown for the LSUN-Church $( 1 \bar { 2 } 8 ^ { 2 } )$ dataset. If we place $t _ { \Phi }$ too far, the model happens to learn only the lowest frequencies and generates no details at all. Placing it too early leads to results that are less sharp. We found that by placing $t _ { \Phi }$ at $50 \%$ , we strike a good balance between the two, leading to better quantitative and qualitative results.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/e809266f82a5dfa7a5c6bbca9dba01613d5781c9c00a7f9df9b6a3f41bef2052.jpg)



IHDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/72e2c8bf576bdc6e8216d7d605982312155a156d2ba007b1aab54883078e5594.jpg)



BNDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/78c09af9e635753afd7262c44495e28dd1ae7f046ceaa3708ab4b121c93124ad.jpg)



DDPM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/74cba529748a5b87294021d36591ffd7476213a206d06b65d139b0ba953929b8.jpg)



Figure 8: More unconditional samples for IHDM, DDPM and our method on the AFHQ-Cat $( 1 2 8 ^ { 2 } )$ dataset. Although the difference between DDPM and our method is subtle, we consistently found that our approach captures geometric details more effectively (e.g., whiskers) and experiences fewer blurry artifacts (e.g., right sample in row 3, DDPM vs. Ours).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/f270ef9424f44b74e60ed53d9431c058e2ec76dbab403594b0640a8c9e173fa6.jpg)



IHDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/d486fd418fed5b3a1451a4cb663db8b1c75c28c86200d97b9dbd6f3b916e8e5e.jpg)



BNDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/7e26eb6c828f982e86c04d4bdf7601160cae10e2929db855d1b44740ea04e80d.jpg)



DDPM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/0a34945808b5266d969a733115cff6d7bc38c3218cb5e9773f06bd2a5831eb12.jpg)



Ours



Figure 9: More unconditional samples for IHDM, BNDM, DDPM and our method on the CelebA $( 1 \bar { 2 } 8 ^ { 2 } )$ dataset. While BNDM is only slightly outperformed by our model in terms of FID metric, its samples look noticeably different in terms of colors. We attribute this difference to the fact that BNDM simulates an ODE, where we in contrast simulate an SDE, possibly causing both methods to sample a different part of the manifold. In terms of visual quality the BNDM samples also show more artifacts, but it is known from previous work that FID score does not always well reflect percepted quality (Liu et al., 2018).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/67140bd61dac21f8df4fc4cf5bce1ee837017b84961eec91873f6b7df8229e77.jpg)



IHDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/ec2887de3aef6b2022d4a2adc652ef1d5c41e17f456e17f21103281586e13ee6.jpg)



BNDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/13dbe3f369de2e74602a22d141b4a925488b0f3802eb7878d750ab005ec1ab2d.jpg)



DDPM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/c842ae93e259ec904ecfcfc261c51acbbb52db4648c5e5868976f08e95977a7e.jpg)



Ours



Figure 10: More unconditional samples for IHDM, BNDM, DDPM and our method on the LSUN-Church $( 1 2 8 ^ { 2 } )$ dataset. lthough our results appear similar to DDPM’s, our method more effectively captures the geometric details of buildings and exhibits fewer artifacts, such as blurry regions, compared to DDPM.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/bfde6b477ca75cb9e2524705a67544d0a19463f0da4150171c1dbd69cbce41d4.jpg)



IHDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/ef75b68ca6bd3888b717f51a03a6eeae38efd3bf7783c38ef3bb7deb058bf009.jpg)



BNDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/fe8ee2331d312f0784c2b9aa806e90fe782e747bfc4e2028fd2ac755f081cacf.jpg)



DDPM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/1416c18e8d73a7aaa7adc8d19962bc2d3512de778f4b34bab4d6257f36646d14.jpg)



Ours



Figure 11: More unconditional samples for IHDM, DDPM and our method on the AFHQ-Cat $( 5 1 2 ^ { 2 }$ , LDM) dataset. All samples are generated via diffusion in latent space. Note that despite the deficit in FID score, our method is able to produce results of very similar perceptual quality.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/0298fd128d461202bf39fff24a57ea0d24b610536e4aaa4150c6b7cc68092dec.jpg)



IHDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/40a3eddb1846b4139110e4ed820a3250a47ca64c340397781b77c2f9649fcb38.jpg)



BNDM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/bb91a5953c629a3b61073c99aaf663771e55c9686cc2e148df7f591018234ec5.jpg)



DDPM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-04-22/a97b318a-4944-47ce-993e-aa155c513428/c88e3d3cac42ed97399a7d4d87af7a5c60e60f70adadd40aa6901f6bfc939d9a.jpg)



Ours



Figure 12: More unconditional samples for IHDM, DDPM and our method on the CelebA $( 2 5 6 ^ { 2 }$ , LDM) dataset. All samples are generated via diffusion in latent space. Although our method is slightly outperformed in terms of the FID metric, the visual quality of our samples is highly comparable to the baselines, and in some cases, even superior (e.g., third row of DDPM vs. Ours).
