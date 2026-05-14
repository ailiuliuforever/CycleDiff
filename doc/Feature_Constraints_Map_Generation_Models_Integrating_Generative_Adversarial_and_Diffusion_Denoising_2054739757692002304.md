Article 

# Feature Constraints Map Generation Models Integrating Generative Adversarial and Diffusion Denoising

Chenxing Sun 1,† , Xixi Fan 2,†, Xiechun Lu 3,4,5,6,* , Laner Zhou 2, Junli Zhao 1, Yuxuan Dong 7 and Zhanlong Chen 1,2,5,6 



Key Laboratory of Geological Survey and Evaluation of Ministry of Education, China University of Geosciences, Wuhan 430074, China; sunchenxing@cug.edu.cn (C.S.); chenzl@cug.edu.cn (Z.C.) 





2 School of Computer Science, China University of Geosciences, Wuhan 430074, China; 1202311226@cug.edu.cn (X.F.) 





3 Hubei Key Laboratory of Intelligent Vision Based Monitoring for Hydroelectric Engineering, China Three Gorges University, Yichang 443002, China 





4 College of Computer and Information Technology, China Three Gorges University, Yichang 443002, China 





5 Engineering Research Center of Natural Resource Information Management and Digital Twin Engineering Software, Ministry of Education, Wuhan 430074, China 





6 Hubei Key Laboratory of Intelligent Geo-Information Processing, China University of Geosciences, Wuhan 430074, China 





7 GeoScene Information Technology Co., Ltd., Beijing 100028, China 





* Correspondence: luxiechun@ctgu.edu.cn 





† These authors contributed equally to this work. 



# Abstract

The accelerated evolution of remote sensing technology has intensified the demand for real-time tile map generation, highlighting the limitations of conventional mapping approaches that rely on manual cartography and field surveys. To address the critical need for rapid cartographic updates, this study presents a novel multi-stage generative framework that synergistically integrates Generative Adversarial Networks (GANs) with Diffusion Denoising Models (DMs) for high-fidelity map generation from remote sensing imagery. Specifically, our proposed architecture first employs GANs for rapid preliminary map generation, followed by a cascaded diffusion process that progressively refines topological details and spatial accuracy through iterative denoising. Furthermore, we propose a hybrid attention mechanism that strategically combines channel-wise feature recalibration with coordinate-aware spatial modulation, enabling the enhanced discrimination of geographic features under challenging conditions involving edge ambiguity and environmental noise. Quantitative evaluations demonstrate that our method significantly surpasses established baselines in both structural consistency and geometric fidelity. This framework establishes an operational paradigm for automated, rapid-response cartography, demonstrating a particular utility in time-sensitive applications including disaster impact assessment, unmapped terrain documentation, and dynamic environmental surveillance. 

Keywords: generative mapping; remote sensing cartography; attention mechanism; image translation 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/43d9b8aa3456e1215fb2bc2a7d5767f468f37856ca2311b6488a8a5e58364164.jpg)


Academic Editor: Salah Bourennane 

Received: 30 May 2025 

Revised: 3 July 2025 

Accepted: 10 July 2025 

Published: 3 August 2025 

Citation: Sun, C.; Fan, X.; Lu, X.; 

Zhou, L.; Zhao, J.; Dong, Y.; Chen, Z. 

Feature Constraints Map Generation 

Models Integrating Generative 

Adversarial and Diffusion Denoising. 

Remote Sens. 2025, 17, 2683. https:// 

doi.org/10.3390/rs17152683 

Copyright: © 2025 by the authors. 

Licensee MDPI, Basel, Switzerland. 

This article is an open access article distributed under the terms and 

conditions of the Creative Commons 

Attribution (CC BY) license 

(https://creativecommons.org/ 

licenses/by/4.0/). 

# 1. Introduction

Tile maps, as digital representations of contemporary geographic information, provide real-time, high-precision spatial data and geographic features [1–3]. Widely adopted by online mapping platforms—such as Google Maps and Bing Maps—they have transformed the way we navigate and locate spatial positions [4–6]. Nevertheless, conventional tile map production still depends heavily on field surveys and manual drafting [7,8], rendering it incapable of satisfying the need for rapid updates in response to dynamic environmental changes. This dependency often yields outdated or imprecise map layers, impeding time-critical applications such as emergency response. Therefore, developing methods for rapidly generating tile maps from remote sensing imagery is essential. Such an approach significantly accelerates map production and ensures timely updates, especially in inaccessible or hazardous areas for ground-based surveys. 

Although end-to-end generative models have proven to be effective for various image synthesis tasks [9,10], they often struggle to translate remote sensing images into precise map tiles, especially in identifying and delineating geographic features such as forests, buildings, water bodies, and roads [11,12]. For instance, vegetation or urban structures often partially obscure road segments in aerial imagery, disrupting their visual continuity. Consequently, the generative network architecture must detect visible feature fragments and infer and reconstruct the obscured sections to maintain topological consistency [13]. This necessity involves designing a model that is both sensitive to the exposed geometry of geographic features and capable of predicting the trajectory and extent of occluded sections, thereby ensuring coherent and reliable map outputs [14]. 

Current map generation models, exemplified by MapGAN [11], SMAPGAN [15], and CreativeGAN [16], utilize GAN-based (Generative Adversarial Networks) image translation methods to achieve map creation from remote sensing imagery [10,17]. These methods require just a single forward pass for image generation, but the training process of GANs is notoriously unstable. This instability arises from the alternating need to train a generator and a discriminator [18]. The generator is responsible for producing target images, while the discriminator evaluates the authenticity of input images. During training, the discriminator often shows more effective learning results than the generator, leading to a significant skill disparity that hinders adversarial training [19]. The pronounced class imbalance, compounded by the intrinsic constraints of existing generator architectures, has precipitated a performance ceiling in GAN-based cartographic synthesis. Traditional remedies—such as enlarging the training corpus or lengthening the training schedule— have delivered only marginal gains in output fidelity, thereby underscoring the imperative for fundamentally novel architectural designs or algorithmic innovations [20–22]. 

Diffusion-based generation techniques (DMs) have recently emerged to overcome the coarse detail rendering and training instability inherent in GAN-based map generation. These methods generate data by incrementally removing noise, offering a controllable process that can precisely adjust the diversity and style of generated samples [23,24]. It usually results in improved detail, resolution, and diversity in generated images compared to GANs [25,26]. Moreover, the training process of diffusion models is based on a maximum likelihood estimation, eliminating the need for complex adversarial training and avoiding common GAN issues such as training instability and mode collapse [25,27]. However, diffusion models involve multiple iterative steps during generation, which result in slower sampling speeds and higher memory demands, posing challenges for the timeliness of real-time map generation [28,29]. In contrast, GANs achieve single-pass inference but suffer from mode collapse [20]. 

To address these challenges, we propose a hybrid adversarial–diffusion framework for rapid tile map generation from remote sensing imagery. Our generator embeds a diffusion-style noise injection and denoising module within a GAN backbone, preserving single-pass inference efficiency while enhancing fine-detail synthesis. We introduce a novel coordinate–channel attention mechanism to model spatial and feature interactions. Joint and ablation studies demonstrate their clear advantage over vanilla, channel-only, and coordinate-only attention schemes. Extensive experiments show that our model produces multi-scale tile maps with superior visual fidelity and achieves improved FID and SSIM scores compared to state-of-the-art baselines. 

The main contributions of this paper are as follows: 1. We developed a novel model for tile map generation by integrating GANs with diffusion models. This model combines the adversarial framework of GANs with the noise reduction mechanism of diffusion models, ensuring both rapid generation and higher quality in tile-based map creation. 2. We designed a specialized attention mechanism for converting remote sensing images into tile maps, incorporating both channel and coordinate attention. This mechanism enhances the model’s feature extraction and terrain representation capabilities by capturing the cross-channel relationships and spatial positioning information inherent in remote sensing imagery. 

The rest of this paper is organized as follows: Section 2 discusses related works on image translation, map generation, diffusion models, and attention mechanisms. Section 3 introduces the primary methods used in this study. Section 4 details the experimental procedures and results. Finally, the conclusions are summarized in Section 5. 

# 2. Related Works

# 2.1. GAN-Based Image Generation

Generative Adversarial Networks (GANs), introduced by Goodfellow et al. in 2014 [9], revolutionized image generation by establishing an adversarial framework between a generator and a discriminator. The generator aims to synthesize realistic images, while the discriminator learns to distinguish between real and synthetic data. This iterative competition drives the generator to produce increasingly convincing outputs. Early advancements, such as Deep Convolutional GAN (DCGAN) [30], stabilized training through the use of convolutional layers and batch normalization, while Wasserstein GAN (WGAN) [31] addressed instability via gradient penalty optimization. These innovations laid the groundwork for high-quality image synthesis, enabling GANs to transcend basic generation tasks and evolve into versatile tools for domain adaptation. 

Building on GANs, image-to-image (I2I) translation emerged as a paradigm for transforming images across domains while preserving semantic content. Unlike traditional methods that rely on manual feature engineering, GAN-based approaches automate end-toend translation by learning mappings between domains. For instance, CycleGAN achieved unpaired translation through cyclic consistency constraints [32], and Progressive GAN (Pro-GAN) enabled high-resolution synthesis via incremental training [33]. These breakthroughs expanded I2I applications—from style transfer to image restoration—by leveraging adversarial training to capture complex domain relationships [34–36]. Subsequent innovations, such as StyleGAN [37] and BigGAN [20], enhanced output fidelity through hierarchical style modulation and large-scale training, demonstrating the GANs’ remarkable capacity to create highly realistic synthetic images across various domains. 

The evolution of I2I reflects the broader progression of generative models. After GANs established adversarial training principles, architectures such as StyleGAN2 [38] refined texture details and spatial consistency, which are critical for photorealistic translation. Applications have extended beyond static images: M3GAN [39] adapted GANs for robust time-series anomaly detection, while the Cooperative Network for Time Series (CNTS) [40] integrated adversarial and cooperative learning for sequential data. These advances underscore how GAN frameworks have matured to address both visual and non-visual translation challenges. These advancements solidify the role of GANs as a paradigmdefining force in both artistic and scientific visualization domains, continually pushing the boundaries of synthetic image generation while maintaining the core adversarial learning mechanisms that form their theoretical foundation. 

# 2.2. Diffusion-Based Image Generation

The core concept of diffusion models is to transform data from a real distribution to a simplified distribution, such as a Gaussian distribution, through a stepwise noise process. During generation, the model learns the reverse process to recover data from noise, gradually denoising to generate new samples. The advantage of this approach is its ability to capture detailed information within complex data distributions. In image generation, diffusion models have been widely used in various tasks, including image synthesis, superresolution reconstruction, and image restoration. Unlike GANs, diffusion models excel in generation quality, especially when handling high-resolution images, as they typically produce more detailed and realistic images. In 2020, Ho and Jonathon Ho introduced Denoising Diffusion Probabilistic Models (DDPM) [24], applying “denoising” diffusion probabilistic models to image generation for the first time, achieving breakthroughs in the field. The same year, Jiaming Song et al. proposed Denoising Diffusion Implicit Models (DDIM) [29], which share the same training process as DDPM but eliminate the need to simulate a multi-step Markov chain during sample generation, significantly improving generation speed. 

Additionally, Cohen et al. introduced Latent Diffusion Models (LDMs) [17], which incorporate the diffusion process in the latent space instead of directly operating in the pixel space, reducing costs and enabling fast sampling and efficient training. Despite the advantages of diffusion models in quality generation, their generation process is often slow and computationally expensive, posing challenges for applications requiring quick responses. Moreover, training diffusion models are relatively complex and demand substantial computational resources and time. Recently, researchers have started exploring the potential of combining diffusion models with other generative techniques to overcome these limitations. 

# 2.3. Remote Sensing Map Generation

In recent years, GANs and I2I translation methods have been extensively applied to the generation and transformation of remote sensing images. Li et al. [11] introduced the MapGAN model, which utilizes GANs to generate network tile maps efficiently, producing various tile maps from remote sensing images and rendering matrices with notable accuracy. Concurrently, Zhang et al. [41] enhanced the map generation quality by integrating external geographic data and transforming geographic text data into images to complement satellite images during the conversion process. Moreover, Fu et al. [16] developed an end-to-end online map generation method to convert aerial images into digital maps automatically. Li et al. [12] proposed CscGAN, a conditional scale-consistent generative network for multi-level remote sensing image-to-map translation, utilizing a unified model to create corresponding tile maps at varying scales from multi-scale remote sensing images. To address domain discrepancies and content consistency in multi-layer translations from satellite images, Fu et al. [42] developed a hierarchical-aware fusion network, incorporating a Recursive Deep (RD) generator, a hierarchical classifier, a map element extractor, and a Multi-Level Fusion (MLF) generator. Recently, Sebaq et al. [43] and Liu et al. [44] investigated the incorporation of diffusion models in cartography, focusing on aligning text with imagery to generate high-resolution satellite images based on sequential text prompts. Alternatively, Tian et al. [45] proposed MapGen-Diff, an end-to-end image-to-map generator utilizing a denoising diffusion bridge model. This methodology employs a strategy inspired by Brownian motion to balance diversity and accuracy, alongside latent space compression and geometric consistency regularization, to enhance boundary clarity and color accuracy in generated maps. 

While GANs have advanced remote sensing map generation, their practical application is frequently impeded by persistent challenges such as training instability, mode collapse, and insufficient fidelity in detail-rich scenes. In contrast, diffusion models excel at generating images with high spatial consistency and fine-grained detail, effectively mitigating the artifact and quality issues common to GANs. However, the substantial computational overhead and low inference speed of diffusion models limit their utility in time-sensitive applications, such as real-time map generation. A hybrid GAN-diffusion framework, therefore, emerges as a highly compelling solution, engineered to leverage the architectural merits of both paradigms in a synergistic manner. This approach leverages the efficiency of GANs for rapid, coarse-structural generation while capitalizing on the meticulous refinement capabilities of diffusion models to enhance textural and geometric details. By doing so, the hybrid model strikes an optimal balance between generation efficiency and output quality, presenting a robust pathway for high-fidelity, real-time remote sensing applications. 

# 3. Method

# 3.1. Basement

# 3.1.1. GAN Framework

The Pix2Pix network model is the initial model applying GAN architecture to image translation [10]. It employs a Conditional Generative Adversarial Network (cGAN) structure consisting of a generator and a discriminator. The generator uses a U-net structure (Figure 1), where the encoder extracts features from the input image through convolution and downsampling operations, and the decoder restores the feature map to its original size using transposed convolutions and upsampling operations. Skip connections are introduced by directly linking the encoder’s layers to the corresponding layers in the decoder, allowing the decoder to better utilize multi-level feature information. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/29c5f4a8eecf130b686dd366c542da3568dd3ffd96e0463693febfd661b4dccc.jpg)



Figure 1. The structure of the Pix2Pix generator.


Unlike other GAN models, Pix2Pix uses the PatchGAN structure (Figure 2) as a discriminator. PatchGAN divides the image into multiple patches, each containing authenticity information corresponding to the image. It assesses the authenticity of each patch individually and ultimately determines the authenticity of the entire image. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/c8630b65182832cb90ba82835369200f18452f4f79892056f56b9b320648d588.jpg)



Figure 2. The structure of the Pix2Pix discriminator.


The adversarial process involves the generator creating images intended to deceive the discriminator while the discriminator endeavors to assess the authenticity of these images accurately. The loss function for this adversarial process is defined as follows: 

$$
L _ {G A N} (G, D) = E _ {x, y} [ \log D (x, y) ] + E _ {x, z} [ \log (1 - D (x, G (x, z))) ], \tag {1}
$$

L1 loss is introduced to evaluate the difference between generated and real images, enhancing the structural and content similarity of the generated images to the real ones. 

$$
L _ {L 1} (G) = E _ {x, y, z} [ | y - G (x, z) | _ {1} ], \tag {2}
$$

The total loss is a weighted sum of the adversarial and L1 losses, with the importance of each measured by the hyperparameter λ. 

$$
G ^ {*} = \arg \min _ {G} \max _ {D} L _ {G A N} (G, D) + \lambda L _ {L 1} (G), \tag {3}
$$

# 3.1.2. Diffusion Model

In diffusion models [24], the noise injection or diffusion process entails incrementally adding Gaussian noise $\epsilon _ { t } \sim N ( 0 , I )$ to the input data sample $x _ { 0 }$ . The state of the image $x _ { t }$ at each subsequent timestep is derived from the previous timestep $x _ { t - 1 }$ by computing $q \big ( x _ { t } | x _ { t - 1 } \big )$ . At step t, the data $x _ { t }$ resembles a Gaussian distribution. $\beta _ { t }$ is employed as a hyperparameter, increasing linearly from 0.0001 to 0.02, and $\alpha _ { t }$ is defined as $1 - \beta _ { t }$ . Based on the properties of a Markov chain, the relationship between the image states $x _ { t }$ and $x _ { t }$ −1 is given as follows: 

$$
x _ {t} = \sqrt {\alpha_ {t}} x _ {t - 1} + \sqrt {1 - \alpha_ {t}} \epsilon_ {t - 1}, \epsilon_ {t} \sim \mathcal {N} (0, I), \tag {4}
$$

The noise injection is crucial for the diffusion model to learn the subsequent denoising process. By understanding how noise is incrementally added, the model learns to infer the denoising process—specifically, deducing the image state at the previous timestep $x _ { t - 1 }$ from the current timestep $x _ { t } ,$ which involves calculating $q ( x _ { t - 1 } | x _ { t } )$ . This inference process continues iteratively, ultimately enabling the derivation of the image state at time $t = 0$ from $t = T$ , denoted as $q ( x _ { 0 } | x _ { T } )$ . This comprehensive learning facilitates the generation of an image from pure random noise. 

$$
x _ {t - 1} = \frac {1}{\sqrt {\alpha_ {t}}} \left(x _ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \overline {{{\alpha_ {t}}}}}} \epsilon_ {\theta}\right) + \sigma_ {q} z, (z \sim N (0, I)), \tag {5}
$$

# 3.2. Overview

To tackle the challenges of poor generation quality in GAN models and the slow, computationally intensive sampling process in diffusion models, we propose a hybrid framework that integrates denoising mechanisms into the adversarial generative architecture for rapid, high-quality tile map generation. As depicted in Figure 3, this framework consists of an attention-enhanced generator and a PatchGAN-based discriminator, both of which incorporate noise mechanisms and timesteps. The decision map output by the discriminator from the previous iteration is upsampled and injected as noise into the actual remote sensing imagery. By contaminating these images with noise and embedding a timestep t, the generator learns to remove the noise, thereby producing images with enhanced resolution and clarity. To effectively extract terrain features from remote sensing images, we have developed a novel attention mechanism that combines coordinate and channel information. Coordinate attention, sensitive to positional data, significantly enhances the map generation accuracy, while channel attention accentuates key geographic features, improving the clarity and emphasis of the generated tile maps. This innovative attention mechanism allows the model to adapt more effectively to the transformation of remote sensing imagery into tile maps. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/9aacff90cb1a9f64d8f3fe17698c18b062d48ce184120b8d39b65e0c1d75a1fa.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/606b491b4c6bc72f2bb26ff224ebc5a818a49b0be963daebb7600828c7f3a1db.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/e86c630e056486a68c77f88ba9dd07c2fef04ec8fb53b97d0ba7ae834ef9782d.jpg)



Basic Block MLP Add Multiply N Layer Norm Sigmoid


Figure 3. The overall framework structure consists of the following parts: The upper section contains the generator, which includes an encoder and decoder connected through residual links and utilizes time embedding to aid the denoising process. The middle section features the discriminator, which evaluates the authenticity of input images. The output patches are upsampled and then fed back into the generator for guidance. The bottom section presents a detailed diagram of the Neck Block within the generator and a comprehensive flowchart illustrating the attention mechanism throughout the network. 

Compared to traditional generative adversarial networks, which often encounter challenges like road blurring and discontinuity in generating regular geographic textures, our model utilizes noise injection and recovery to train the generator on high-frequency feature reconstruction (Figure 4). By introducing controllable noise at the input stage, the generator must learn the process of mapping from noisy distributions to clear structures during adversarial training, thereby maintaining road continuity and river topology fidelity in satellite-to-vector map conversion tasks. To address complex noise interferences such as cloud cover and atmospheric scattering in remote sensing data, the model incorporates adaptive denoising modules within the generator that segregate interference signals, preserving original terrain features and avoiding the information loss typical of manual denoising. Furthermore, the model employs a modified U-net architecture with integrated attention mechanisms when generating unstructured terrains like mountains and coastlines. This configuration effectively restrains terrain distortion and prevents structural misalignment in violation of geographic principles by applying global morphological constraints and local texture adaptation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/5a6e0e8aa9495bf37bd9ffadfd28854acdc8fe50760feb75cdc7f7fa1d35b7fa.jpg)



Figure 4. The model training process involves time t embedding, with effects shown at (t = 1, 20, 50, and 100), respectively. In this context, RS denotes the real input remote sensing imagery, and Noisy refers to the decision map output from the previous discriminator round, acting as the noise map that affects the remote sensing imagery. Consequently, Noisy RS is the contaminated map tile. Fake Map refers to the generated map tile, whereas Real Map denotes the actual map tile. Observations from the map tiles indicate that as t is embedded, the Noisy embedding shifts from random to systematic, and the level of contamination gradually decreases. The embedding of different t values enhances the generator’s denoising capabilities at various t states.


# 3.3. Diffusion-Fused Adversarial Generation

As illustrated in Figure 5, this framework incorporates diffusion-based noise addition and denoising mechanisms within a conventional adversarial generative architecture. The input layer takes real remote sensing imagery and the previous discriminator iteration’s decision map, producing corresponding map tiles as output. During training, the time parameter t (representing different epochs) is embedded through Fourier transforms and MLP layers, enabling the generator to control noise injection and removal adaptively across training stages. The generator learns progressive denoising strategies by conditioning the model on t, thereby capturing robust RS-Map mapping relationships through multi-stage feature refinement. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/4422a57a36368f3b865be990848d7865e99b5a1df44e2f8dea6d8c30cd4a6a07.jpg)



Downsample



Upsample



Basic Block



Conv2D Block



MLP



Add



A Attention



Fourier



Tanh



Layer Norm


Figure 5. Architecture of the main generator with time embedding: The core framework processes spatially concatenated inputs (remote sensing imagery and previous discriminator decision map) through an encoder-decoder pathway to synthesize tile maps. The corresponding key submodules include: (a) Downsample Block; (b) W block; (c) Output block; (d) Upsample block; (e) Neck Block; (f) Basic Block containing convolutional layer, channel normalization, and SiLU activation. 

The generator employs a U-net architecture with encoder–decoder symmetry. The encoder extracts hierarchical semantic features via convolutional layers and downsampling blocks, while the decoder reconstructs spatial details through deconvolution and upsampling, enhanced by skip connections that fuse multi-level encoder features. Crucially, both pathways integrate time-dependent operations: t is injected into all Basic Blocks to modulate feature normalization and noise intensity, ensuring synchronization between diffusion scheduling and feature learning. 

In the encoder, each Downsample Block contains four stacked Basic Blocks with residual connections. Each Basic Block sequentially processes features through convolution, normalization, and activation layers, followed by an attention mechanism and downsampling operations. Noise is adaptively added to features based on the current training stage t, with intensity governed by the embedded time parameters. A lightweight Neck Block between the encoder and decoder reduces channel dimensions to focus on essential features. Upsample Blocks mirror the encoder structure with four Basic Blocks in the decoder, utilizing residual connections and cross-level skip connections from the encoder. Attention mechanisms guide feature upsampling to recover the original image resolution while progressively preserving semantic consistency. 

In the specific noise injection process, our primary objective is to generate tile maps directly from real remote sensing images rather than crafting a tile map-style image from random noise, as in diffusion models. Consequently, this model’s noise addition process does not require transforming the input x into Gaussian-distributed random noise. Instead, we upsample the decision map and progressively inject it as noise into the input image x during each round (Figure 4). 

To facilitate noise injection, we designed an additional structure known as the W block. This structure uses the entire patch output from the previous round’s discriminator as input. The resulting output is combined with the image x from the source domain for image translation, which serves as the input to the generator. Here, $D _ { t }$ represents the decision map generated by discriminator D at the t-th epoch, and a learnable mapping is introduced as $W _ { t } = W ( D _ { t - 1 } )$ . Subsequently, $W _ { t }$ is integrated into the input image x via the method $x _ { t } = x + x W ( D _ { t - 1 } )$ . The model then employs a U-net structure within the generator network to learn how to remove this noise to produce the desired tile map. The specific composition is detailed below, where B refers to the Basic Block, as illustrated in Figure 5. 

$$
B (F) = \text { ReLU } (\text { BatchNorm } (\text { Conv } * F)), \tag {6}
$$

$$
W _ {t} = W (D _ {t - 1}) = \operatorname{Tanh} \left(\operatorname{Conv} * \left(U _ {u p} (B (D _ {t - 1}))\right) ^ {3}\right), \tag {7}
$$

Using the decision map output by PatchGAN as a noise source offers several advantages. The decision map provides insight into the generator’s output performance, which benefits the generator. This information aids the generator in identifying and improving areas requiring enhancement. In traditional GANs, a major challenge for achieving fast and stable training is the need to train a generator and a similarly proficient discriminator alternately. Training the generator is often more challenging than training the discriminator, frequently resulting in a highly capable discriminator that overwhelms the generator, leading to training failure. In this model, artificially supplying the generator with additional decision map information from the discriminator significantly reduces the generator’s training difficulty, thereby balancing the adversarial game more effectively. 

By injecting the decision map from the discriminator into the original image in the same manner, we combine the knowledge of GANs with entropy and infer that when decisions for each patch are independent, their joint entropy equals the sum of the entropies of each patch. 

$$
H (Y _ {1}, \dots , Y _ {N}) = \sum_ {i = 1} ^ {N} H (Y _ {i}), \tag {8}
$$

$$
H (Y _ {i}) = - D _ {i} \log D _ {i} - (1 - D _ {i}) \log (1 - D _ {i}), \tag {9}
$$

Assuming the discriminator consists of N patches, then the model’s $L _ { G A N }$ can be expressed as follows: 

$$
L _ {G A N} (G, D) = \frac {1}{N} \sum_ {i = 1} ^ {N} \tilde {L} _ {G A N} (G, D _ {i}), \tag {10}
$$

where $\stackrel { \sim } { L } _ { G A N } ( G , D _ { i } )$ represents the adversarial loss corresponding to each patch, when the output $D _ { i } = 0 . 5$ for each patch maximizes the joint distribution entropy, indicating that the GAN objective function reaches its theoretical optimal value of $- l o g ( 4 )$ , which suggests that the model has reached Nash equilibrium and training has converged. 

$$
L _ {G A N} = \frac {1}{N} N \left(\log \left(\frac {1}{2}\right) + \log \left(\frac {1}{2}\right)\right) = - \log (4), \tag {11}
$$

# 3.4. Geophysical Feature Attention Enhancement Mechanism

We developed a novel geophysical feature enhancement hybrid attention mechanism (Figure 6) to enhance the geophysical features necessary for mapping from remote sensing imagery. This mechanism effectively captures cross-channel relationships and spatial– positional information while delivering efficient feature representation in a lightweight architecture. It comprises two components: a channel attention mechanism [41] and a coordinate-based attention mechanism [42]. The workflow processes the input feature map $F \in R ^ { C \times H \times W }$ through these two attention mechanisms. The formula is presented as follows: 

$$
F ^ {\prime} = M _ {c h} (F) \otimes F, \tag {12}
$$

$$
F ^ {\prime \prime} = M _ {c o} \left(F ^ {\prime}\right) \otimes F ^ {\prime}, \tag {13}
$$

where $M _ { c h }$ represents the channel attention mechanism, and $M _ { c o }$ denotes the coordinate attention mechanism. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/87df13f7df27b609bcf0b84ea4c3d4b3889b0006c9f0a9881b8f26b28125bc8d.jpg)



Figure 6. Overall structure of the geophysical feature attention enhancement mechanism.


# 3.4.1. Channel Attention for Geophysical Feature Recognition

Channel attention mechanisms are pivotal in recognizing geophysical features on maps. They identify and emphasize key geographic elements such as roads, rivers, and buildings, thereby significantly enhancing translated maps’ visual clarity and informational prominence. By assigning weights to different channels, these mechanisms increase the model’s focus on important features and suppress channels with irrelevant or noisy features, improving the accuracy and efficiency of the generation process. Additionally, they dynamically adjust feature responses, enabling the network to better adapt to variations in feature representations, particularly in complex environments like vegetation occlusion or similar backgrounds, thus demonstrating stronger generalization capabilities. Furthermore, by integrating information across channels, these mechanisms enhance the model’s understanding of global context, ensuring the overall consistency and structural integrity of the map, which makes the translation results visually more accurate. 

As shown in Figure 7, the structure of the channel attention mechanism is as follows: first, each map of the input feature maps undergoes global average pooling and global max pooling to generate two different vectors, $F _ { a v g } ^ { C }$ and $F _ { m a x } ^ { C }$ . These vectors, each with dimensions $C \times 1 \times 1$ , are then passed through a convolutional layer Conv, ReLU activation, a 1 × 1 convolution, and finally, a Sigmoid function for normalization [46]. This process calculates the attention weight for each channel of the input feature maps. 

$$
\begin{array}{l} M _ {c h} (F) = \delta (M L P (A v g P o o l (F)) + M L P (M a x P o o l (F))) \\ = \delta \left(W _ {1} \left(W _ {0} \left(F _ {\text {avg}} ^ {C}\right)\right) + W _ {1} \left(W _ {0} \left(F _ {\text {max}} ^ {C}\right)\right)\right), \tag {14} \\ \end{array}
$$

where $M _ { c h } ( F )$ represents the channel attention map, δ denotes the Sigmoid function, and $M L P$ stands for multi-layer perceptron. $W _ { 0 } \in { R ^ { C / r \times C } }$ and $W _ { 0 } \in R ^ { C \times C / r }$ are the weight matrices of the multi-layer perceptron. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/e6b804406114d81c4e1c854546edea7d3302e935d2d2a6e428e484799fb7db5e.jpg)



Figure 7. Channel attention mechanism: The input features undergo max pooling and average pooling, followed by processing through an MLP (multi-layer perceptron), after which the weights are summed and activated.


# 3.4.2. Coordinate Attention for Spatial Relationships

Coordinate attention offers distinct advantages in map image translation. Firstly, it demonstrates strong adaptability, accommodating significant variations in map images due to scale, style, and complexity differences. Its sensitivity to positional information enables coordinate attention to adapt more effectively to diverse map image types. Secondly, it aligns closely with visual elements the human eye perceives, facilitating the generation of translation results that appear more natural and consistent with human visual habits. Furthermore, the coordinate attention’s flexible and lightweight nature allows easy integration into existing network architectures without significantly increasing computational load. More importantly, coordinating attention generates position-aware attention maps that preserve precise geospatial structural information by performing global pooling operations in horizontal and vertical directions. This capability enhances the model’s understanding and translation of geographical features across different map regions. 

The coordinate attention mechanism is composed of two steps, as illustrated in Figure 8: embedding coordinate information and generating coordinate attention. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/215637d53fe722fc4d168381fcc28233dcbdc4aea6cf7db98a1419b72afb6eb2.jpg)



Figure 8. Coordinate attention mechanism: The input features undergo average pooling along the X-axis and Y-axis, followed by concatenation, convolution, and activation.


For coordinate position embedding, each map of the input feature maps undergoes average pooling in the horizontal direction to obtain a $C \times H \times 1$ vector, and in the vertical direction to obtain ${ \mathfrak { a } } C \times 1 \times W$ vector. 

$$
z _ {c} ^ {h} (h) = \frac {1}{W} \sum_ {0 \leq i <   W} F _ {c} ^ {\prime} (h, i), \tag {15}
$$

$$
z _ {c} ^ {w} (w) = \frac {1}{H} \sum_ {0 \leq i <   H} F _ {c} ^ {\prime} (j, w), \tag {16}
$$

Subsequently, in the attention generation phase, one of the vectors is first transposed, and then the two vectors are concatenated. A 1 × 1 convolution is then used for transformation. 

$$
f = \delta (F _ {1} ([ z ^ {h}, z ^ {w} ])), \tag {17}
$$

Here, $F _ { 1 }$ denotes a $1 \times 1$ convolution, $[ . , . ]$ indicates concatenation along the spatial dimension, and δ represents an activation function. Then, the tensor $f$ is split along the spatial dimension into two separate tensors, $f ^ { h }$ and $f ^ { w }$ . Two $1 \times 1$ convolutions, $F _ { h }$ and $F _ { w } ,$ , are then used to transform the two feature maps to the same number of channels as the input $F ^ { ' }$ . 

$$
g ^ {h} = \delta (F _ {h} (f ^ {h})), \tag {18}
$$

$$
g ^ {w} = \delta (F _ {w} (f ^ {w})), \tag {19}
$$

$$
M _ {c o} \left(F ^ {\prime}\right) = g _ {c} ^ {h} (i) \times g _ {c} ^ {w} (j), \tag {20}
$$

# 3.5. Loss Function

The loss function of this model integrates GAN and L1 losses and is adjusted with the embedding of model time t: 

$$
L ^ {t} (G, D) = L _ {G A N} ^ {t} (G, D) + \lambda L _ {L 1} ^ {t} (G), \tag {21}
$$

The specific formulas for the GAN and L1 losses used in this model are as follows: 

$$
L _ {G A N} ^ {t} (G, D) = E _ {x _ {t}, y} [ \log D (x _ {t}, y) ] + E _ {x _ {t}, z} [ \log (1 - D (x _ {t}, G (x _ {t}, z))) ], \tag {22}
$$

$$
L _ {L 1} ^ {t} (G) = E _ {x _ {t}, y, z} [ | y - G (x _ {t}, z) | _ {1} ], \tag {23}
$$

Herein, the loss computation process influenced by time embedding is shown in the following equation: 

$$
x _ {t} = x _ {0} + x _ {0} W \left(D _ {t - 1}\right), \tag {24}
$$

The input to the generator receives interference from the noise $W ( D _ { t - 1 } )$ generated by the discriminator’s output at the previous time $D _ { t - 1 }$ . The parameter λ is used to balance the weights of the GAN adversarial loss and the L1 loss. During training, the model must learn how to remove noise to meet the adversarial requirements of the discriminator while ensuring that the generated images are similar in structure and detail to the real images, thereby achieving high-quality map generation. 

# 4. Experiment

The model undergoes the initial training and evaluation on the Maps dataset, with qualitative and quantitative comparisons against conventional image translation baselines. Subsequent validation employs the MLMG dataset, benchmarking performance against the existing SOTA model LACM, which addresses satellite map domain gaps via its cascaded architecture, integrating map element extractors and multi-level fusion generators. Finally, ablation studies modify the attention mechanism to evaluate its contribution in comparison to prevalent alternatives. 

# 4.1. Experimental Implementations

# 4.1.1. Datasets

This paper utilizes two different datasets: the Maps dataset and the MLMG dataset. The Maps dataset [10] comprises map tiles scraped from Google Maps, with images sized at $6 0 0 \times 6 0 0$ pixels. Its training and test datasets include 1096 and 1098 paired remote sensing images and corresponding tile maps. The MLMG dataset [47] is a large-scale dataset based on satellite images for multi-level map generation tasks. It consists of satellite images from Google Maps and the corresponding map images, divided into two parts: MLMG-US, covering regions of the United States, and MLMG-CN, covering regions of China. The dataset is meticulously organized into four levels of samples: levels 15, 16, 17, and 18. Details of the dataset and example tiles are presented in Table 1 and Figure 9, respectively. 


Table 1. Details of the number of training and test sets in the MLMG dataset for the US and CN regions at levels 15, 16, 17, and 18.


<table><tr><td>Level</td><td>Train-US</td><td>Train-CN</td><td>Test-US</td><td>Test-CN</td><td>Pixel Zoom</td></tr><tr><td>15</td><td>2000</td><td>2000</td><td>20</td><td>20</td><td>4.8 m/pixel</td></tr><tr><td>16</td><td>2000</td><td>2000</td><td>80</td><td>80</td><td>2.4 m/pixel</td></tr><tr><td>17</td><td>2000</td><td>2000</td><td>320</td><td>320</td><td>1.2 m/pixel</td></tr><tr><td>18</td><td>2000</td><td>2000</td><td>1280</td><td>1280</td><td>0.6 m/pixel</td></tr><tr><td>Total</td><td>8000</td><td>8000</td><td>1700</td><td>1700</td><td>-</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/e037191e5850725948b6e2695cd205bd8759547324dbe20ed7b6bb476c7defc8.jpg)



Figure 9. The MLMG dataset includes examples at levels 15, 16, 17, and 18, with the order from left to right being MLMG-CN and MLMG-US, collected, respectively, from urban areas in China and the United States.


# 4.1.2. Evaluation Metrics

We employ SSIM (Structural Similarity Index Measure), FID (Fréchet Inception Distance), and PSNR (Peak Signal-to-Noise Ratio) as three crucial evaluation metrics. These are standard indicators for measuring pixel-by-pixel geographic features and overall structural similarity in the experimental parts of the SOTA map generation methods, each evaluating image generation quality from distinct perspectives. SSIM [48] quantifies visual similarity between two images, considering luminance, contrast, and structural information, with values ranging from 0 to 1, where values closer to 1 signify higher similarity. Its advantage lies in its ability to simulate human visual perception and sensitivity to detail and texture variations. FID [49] assesses image quality by calculating the statistical distance between real and generated images within the Inception network’s feature space. Lower FID values indicate that generated images are statistically closer to real ones, implying higher image quality. PSNR is commonly used to evaluate image quality by quantifying the difference between original and generated images; higher PSNR values suggest a greater similarity and better generated image quality, whereas lower values indicate larger differences and poorer quality. However, its limitation is that it may not always correspond to the human visual perception system; for example, an image might appear distorted despite a high PSNR value. Thus, a comprehensive evaluation using SSIM, FID, and PSNR reliably assesses the quality of generated map tiles. 

# 4.1.3. Implementation Details

While training the model, we applied a consistent standard configuration across all models, including the use of the Adam optimizer with parameters β1 and β2 set to 0.5 and 0.999, respectively. We initialized the generator (based on the U-net-256 architecture) and the discriminator (patchGAN) with an initial learning rate of 0.0002. The learning rate was kept constant for the first 100 epochs and then linearly decreased to zero over the next 100 epochs, resulting in a total of 200 epochs of training. During training, a batch size of eight was used, along with instance normalization. The input images were resized to a resolution of 260 × 260, then randomly cropped to 256 × 256, and data augmentation strategies, including random jittering and horizontal flipping, were applied. 

As shown in Table 2, our model demonstrates a superior balance between efficiency and complexity in terms of computational performance for rapid-response cartography. With a minimal parameter count of ~35.8 M, it significantly reduces the computational overhead and resource footprint compared to heavyweight models like Stable Diffusion (~900 M). This streamlined architecture enables a high throughput of ~5.4 tiles per second, substantially outperforming other diffusion-based methods, including Stable Diffusion (~4.0 tiles per second) and the prohibitively slow DDPM (~1.0 tiles per second). While GANs exhibit a higher raw speed, our model achieves its efficiency while retaining the high-fidelity generation capabilities inherent to the diffusion paradigm. This optimal trade-off validates its practical applicability, enabling the swift generation of extensive geographic areas (~4.43 km2/min) and confirming its suitability for automated, rapidresponse cartography. 


Table 2. Performance and complexity comparison of generative models for map-tile synthesis.


<table><tr><td>Method</td><td>Parameters (M)</td><td>Time/1k Tiles (s)</td><td>Throughput (Tiles/s)</td></tr><tr><td>Pix2Pix [10]</td><td>~54</td><td>~98</td><td>~10.2</td></tr><tr><td>BigGAN [20]</td><td>~80</td><td>~125</td><td>~8.0</td></tr><tr><td>StyleGAN2 [38]</td><td>~150</td><td>~100</td><td>~10.0</td></tr><tr><td>DDPM [24] (1000 step)</td><td>~139</td><td>~1000</td><td>~1.0</td></tr><tr><td>SD [17] (40 step)</td><td>~900</td><td>~250</td><td>~4.0</td></tr><tr><td>Ours (200 step)</td><td>~35.8</td><td>~186</td><td>~5.4</td></tr></table>

The experiments were conducted on a workstation equipped with two NVIDIA Tesla T4 GPUs, offering a total of 32 GB of VRAM, and training for 200 epochs took approximately 7 h. 

# 4.2. Evaluation Results

# 4.2.1. Quantitative Evaluation

Table 3 compares our model with other map generation models using various evaluation metrics for the one-to-one map generation task across the entire test set. Our model surpasses conventional map generation models regarding SSIM, FID, and PSNR scores. A higher SSIM score indicates that images generated by our model exhibit greater visual structural similarity, with an improved preservation of details, contrast, and overall structure. A lower FID score implies that the feature distribution of the generated images is closer to that of real images, suggesting the generated images are more statistically realistic. 

A higher PSNR value also reflects a greater peak signal-to-noise ratio, indicating a lower noise level and better image quality. 


Table 3. Quantitative comparison of this model with some other common image translation models in map generation experiments of the Maps dataset.


<table><tr><td>Method</td><td>SSIM↑</td><td>FID↓</td><td>PSNR↑</td></tr><tr><td>Pix2pix [10]</td><td>0.631</td><td>216.557</td><td>24.535</td></tr><tr><td>Pix2pixHD [17]</td><td>0.726</td><td>129.901</td><td>21.243</td></tr><tr><td>CycleGAN [32]</td><td>0.724</td><td>123.618</td><td>25.929</td></tr><tr><td>Ours</td><td>0.817</td><td>39.832</td><td>27.912</td></tr></table>

The proposed model demonstrates superior performance, as evidenced by enhanced the Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Fréchet Inception Distance (FID) scores, which can be attributed to its innovative architectural design. The improvement in PSNR is primarily a result of the synergistic interplay between the progressive denoising mechanism of the diffusion model and the generative adversarial network (GAN). Specifically, the diffusion model iteratively refines image details through multi-step noise reduction, leading to generated images with higher pixel-wise fidelity to the ground truth. This is further augmented by a hybrid attention mechanism, integrating channel and coordinate attention, which enhances the spatial localization accuracy of geographic features, thereby significantly minimizing pixel-level errors. Concurrently, the outstanding SSIM performance stems from the hybrid attention mechanism’s capacity to jointly optimize cross-channel feature interdependencies and long-range spatial relationships, complemented by the diffusion process’s inherent ability to preserve multi-scale structural information. This synergy enables the generation of images with sharper edges and more defined textural contrasts that closely emulate real-world data, proving particularly effective in complex scenarios such as maintaining the coherent generation of roadways partially occluded by foliage. Furthermore, the notable improvement in the FID score underscores the model’s proficiency in capturing high-level semantic authenticity. This is reinforced by the U-net architecture’s multi-level feature fusion, which ensures semantic consistency across geographic elements at varying scales. 

To demonstrate the model’s effectiveness effectively, we adopted the SOTA experimental setup from LACM [47] and retrained our model using the MLMG dataset, comparing it with the model described in that paper. The experiments were conducted exclusively at the four levels of the MLMG-US dataset, with comparison to the Pix2pixHD [17], CycleGAN [32], SPADE [50], SelectionGAN [51], TSIT [52], LPTN [53], SMAPGAN [15], CreativeGAN [16], and LACM [47] equivalent methods. 

As shown in Table 4, comparative analysis reveals our model’s comprehensive superiority in both PSNR and FID metrics, especially excelling in PSNR due to its unique diffusion-enhanced adversarial framework combined with hybrid attention mechanisms. The diffusion model’s progressive denoising process enables the precise restoration of highfrequency details, such as road edges and building contours, whereas LACM’s reliance solely on GANs often results in blurring effects. Simultaneously, the hybrid attention mechanism (channel and coordinate) significantly enhances object localization and structural coherence through spatial and feature-wise synergistic reinforcement. The leading FID scores at Levels 17 and 18 indicate that the generated images closely align with real-world semantic distributions, thanks to the dynamic noise injection strategy. Visually, our model effectively addresses LACM’s issues of road discontinuities and jagged edges by utilizing U-net-based multi-scale fusion and diffusion refinement, along with the spatial and semantic recognition capabilities of our specialized hybrid attention mechanism, thereby demonstrating strong topological reasoning in complex occlusion scenarios. This dual mechanism of “diffusion-based texture enhancement and attention-guided correction” achieves a synergistic breakthrough in pixel-level accuracy (PSNR), semantic authenticity (FID), and structural naturalness (SSIM). 


Table 4. Quantitative comparison of this model with existing map generation methods across four levels (15–18) of the MLMG-US dataset, where bold and underlined indicate the best and second-best results, respectively.


<table><tr><td>Level</td><td>Method</td><td>FID↓</td><td>PSNR↑</td><td>Level</td><td>Method</td><td>FID↓</td><td>PSNR↑</td></tr><tr><td rowspan="10">15</td><td>CycleGAN [32]</td><td>312.14</td><td>20.725</td><td rowspan="10">16</td><td>CycleGAN [32]</td><td>237.79</td><td>22.545</td></tr><tr><td>Pix2pixHD [17]</td><td>331.10</td><td>20.908</td><td>Pix2pixHD [17]</td><td>206.16</td><td>22.785</td></tr><tr><td>SPADE [50]</td><td>459.11</td><td>20.468</td><td>SPADE [50]</td><td>351.68</td><td>22.865</td></tr><tr><td>SelectionGAN [51]</td><td>337.83</td><td>20.617</td><td>SelectionGAN [51]</td><td>272.04</td><td>22.702</td></tr><tr><td>TSIT [52]</td><td>284.17</td><td>20.540</td><td>TSIT [52]</td><td>219.99</td><td>22.543</td></tr><tr><td>LPTN [53]</td><td>351.61</td><td>21.327</td><td>LPTN [53]</td><td>323.64</td><td>23.454</td></tr><tr><td>SMAPGAN [15]</td><td>336.35</td><td>22.506</td><td>SMAPGAN [15]</td><td>292.50</td><td>24.819</td></tr><tr><td>CreativeGAN [16]</td><td>267.37</td><td>21.428</td><td>CreativeGAN [16]</td><td>193.56</td><td>22.947</td></tr><tr><td>LACM [47]</td><td>195.64</td><td>21.532</td><td>LACM [47]</td><td>154.18</td><td>23.488</td></tr><tr><td>Ours</td><td>213.921</td><td>23.129</td><td>Ours</td><td>193.26</td><td>24.854</td></tr><tr><td rowspan="10">17</td><td>CycleGAN [32]</td><td>167.12</td><td>23.076</td><td rowspan="10">18</td><td>CycleGAN [32]</td><td>138.92</td><td>23.715</td></tr><tr><td>Pix2pixHD [17]</td><td>171.39</td><td>23.878</td><td>Pix2pixHD [17]</td><td>110.35</td><td>24.801</td></tr><tr><td>SPADE [50]</td><td>295.67</td><td>24.000</td><td>SPADE [50]</td><td>224.64</td><td>25.330</td></tr><tr><td>SelectionGAN [51]</td><td>239.45</td><td>23.914</td><td>SelectionGAN [51]</td><td>194.97</td><td>25.442</td></tr><tr><td>TSIT [52]</td><td>137.50</td><td>23.264</td><td>TSIT [52]</td><td>123.45</td><td>24.167</td></tr><tr><td>LPTN [53]</td><td>253.00</td><td>24.198</td><td>LPTN [53]</td><td>182.36</td><td>24.599</td></tr><tr><td>SMAPGAN [15]</td><td>286.87</td><td>25.304</td><td>SMAPGAN [15]</td><td>246.88</td><td>27.005</td></tr><tr><td>CreativeGAN [16]</td><td>129.51</td><td>23.640</td><td>CreativeGAN [16]</td><td>100.54</td><td>24.799</td></tr><tr><td>LACM [47]</td><td>107.95</td><td>24.514</td><td>LACM [47]</td><td>78.44</td><td>25.970</td></tr><tr><td>Ours</td><td>105.48</td><td>26.870</td><td>Ours</td><td>69.768</td><td>27.428</td></tr></table>

# 4.2.2. Qualitative Evaluation

Under identical training sets, test sets, and training conditions, our proposed model was comprehensively compared against several mainstream map generation models. The experimental results unequivocally demonstrate the superior performance of our approach, particularly in the nuanced tasks of geographic feature recognition, occluded road reconstruction, and preserving complex road topologies. 

Specifically, the integration of a hybrid attention mechanism, comprising both channel and coordinate attention, facilitates the precise differentiation of multiple feature categories. Channel attention enhances the model’s sensitivity to distinct geographic features, while coordinate attention ensures accurate spatial localization, thereby guaranteeing the geometric integrity of elements such as roads and buildings and effectively mitigating issues like road distortion or blurred building contours prevalent in traditional methods (as illustrated in Figure 10). Furthermore, our model exhibits outstanding capabilities in reconstructing occluded roadways. Unlike conventional approaches that often yield discontinuities or erroneous connections when encountering obstructions, our model adeptly infers the trajectory of occluded roads. This is achieved through a U-net architecture incorporating skip connections for multi-scale feature fusion, complemented by a dynamic noise injection mechanism guided by discriminator feedback, which enables the generator to adaptively intensify learning on challenging samples, such as occluded regions, thereby maintaining the topological consistency of road networks (Figure 11). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/5af7cf0e3fd6cff19de21be8c0b85bfe909f2f4f1b9bbafd40ecc60f07304493.jpg)



Figure 10. Comparison of the results generated by this model versus other common image translation models in Google Maps generation experiments. From left to right are the real remote sensing image, real Map, and fake Maps generated by the Pix2pix, CycleGAN, and Pix2pixHD models and Ours.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/f137ed876e76b100feb80bc90755dfe7860aed655f6ad7c4d87465bbc41424a0.jpg)



Figure 11. Comparison of the generation details between this model and Pix2pix, CycleGAN, and Pix2pixHD shows that our model outperforms the others in reconstructing occluded roads and generating fine roads.


Moreover, in the generation of complex curved roads, our model demonstrates remarkable advantages. Whereas traditional methods frequently produce roads with jagged edges or unnatural curvatures, our model generates smooth and naturalistic winding paths. This proficiency is attributable to the iterative optimization inherent in the diffusion model and the long-range dependency modeling capabilities of coordinate attention. The progressive denoising process of the diffusion model circumvents local optima often encountered in single forward-pass GANs, while coordinate attention captures extensive spatial dependencies via horizontal and vertical global pooling, ensuring that road alignments adhere to geographic principles (Figures 10 and 12). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/1d362a88ab4402aad112d9fb76a872d22f3aef165ffd865d145a8b16fae063cd.jpg)



Figure 12. Comparison of model generation effects across four levels of the MLMG-US dataset, from top to bottom for levels 15 to 18. Level 16 is composed of four 256 × 256 images stitched together and then resized to the same size as Level 15, with other levels processed similarly.


In summary, by innovatively integrating diffusion denoising mechanisms with hybrid attention mechanisms, our model maintains GAN’s rapid generation advantage while significantly improving geometric accuracy, structural consistency, and semantic authenticity in map generation. This provides a reliable solution for high-precision real-time map production, demonstrating exceptional adaptability when handling complex geographic scenarios such as occlusions, curved roads, and multi-category features. 

# 4.3. Ablation Study

# 4.3.1. Effects of Diffusion and Attention

To verify that injecting the discriminator’s decision map as noise into the original remote sensing image helps improve generation quality, we conducted the following comparative ablation experiments. Under the same dataset, we compared the outputs of the baseline model, the model incorporating the designed attention mechanism without diffusion, the model incorporating diffusion without attention, and the final model integrating both diffusion and attention. The results are presented in Table 5 and Figure 13. 


Table 5. Quantitative comparison of ablation experiments for this model on the Maps dataset. From top to bottom, the rows sequentially represent: the baseline model without using diffusion or attention, the model using diffusion but without attention, the model using attention but without diffusion, and our final model using both diffusion and attention.


<table><tr><td>Method</td><td>SSIM↑</td><td>FID↓</td><td>PSNR↑</td></tr><tr><td>Baseline</td><td>0.631</td><td>216.557</td><td>24.535</td></tr><tr><td>Only-Diffusion</td><td>0.733</td><td>93.662</td><td>27.146</td></tr><tr><td>Only-Attention</td><td>0.759</td><td>56.944</td><td>27.741</td></tr><tr><td>Ours</td><td>0.817</td><td>39.832</td><td>28.700</td></tr></table>

The experimental results demonstrate that the diffusion model, when used independently (Only-Diffusion vs. Baseline), significantly improves the quality of map generation, with enhancements observed in SSIM, FID, and PSNR metrics. This improvement validates the effectiveness of the diffusion model in enhancing structural similarity and reducing the distributional gap between generated data and real maps. When combined with the attention mechanism (Ours vs. Only-Attention), the model’s performance further increases, surpassing that of the attention-only approach, indicating a synergistic effect that jointly optimizes map details and global consistency. 

The core advantages of the diffusion model lie in its progressive denoising generation mechanism, which ensures stable output by gradually removing noise, thereby avoiding common issues such as local distortions or mode collapse. Additionally, the innovative use of the discriminator’s decision map as noise input further constrains the alignment between generated content and real geographic features, leading to significant metric improvements. This characteristic makes the model particularly well-suited for map generation tasks, enabling the accurate reconstruction of occluded roads, the preservation of complex road network topologies, and the enhanced spatial fidelity of geographic elements. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/91e71fa454741da5c8eb6180fc8c69f2e514e35b6508297fc373609013313614.jpg)



Figure 13. Comparison of ablation experiment results generated by the model on the Google Maps dataset. From left to right, the columns sequentially represent: the real remote sensing image, the corresponding real electronic tile map, the generation result of the baseline model without using diffusion or attention, the generation result of the model using diffusion but without attention, the generation result of the model using attention but without diffusion, and the generation result of our final model using both diffusion and attention.


# 4.3.2. Analysis of Feature Enhancement Attention Effectiveness

To validate the superiority of our novel attention mechanism for generating maps from remote sensing images, we conducted additional experiments comparing various established attention mechanisms, including self-attention, channel-based attention, coordinatebased attention, spatial-based attention, and hybrid attention mechanisms. As shown in Figure 14, the results demonstrate that our attention mechanism outperforms existing methods in generating map tiles from remote sensing images. Notably, it generates intricate and winding roads, indicating enhanced effectiveness. 

The experimental results (Table 6) demonstrate that our attention mechanism achieves optimal performance due to its synergistic integration of channel attention and coordinate attention. The channel attention mechanism dynamically weighs different feature channels, significantly enhancing the model’s semantic recognition capability for multi-category geographic features while effectively avoiding the feature confusion commonly observed in traditional methods, as evidenced by the experimental results where channel attention alone achieves a substantially better FID score (49.105) compared to no attention mechanism (93.662). Meanwhile, the coordinate attention mechanism employs horizontal and vertical global pooling operations to precisely capture the spatial topological relationships of geographic elements (such as road continuity and building alignment patterns), with its SSIM score (0.783) outperforming pure spatial attention (0.767), demonstrating the superior effectiveness of coordinate encoding in spatial structure modeling. The deep integration of these two mechanisms creates a complementary advantage where channel attention ensures semantic correctness (e.g., accurate classification of vegetation versus buildings) while coordinate attention guarantees precise spatial alignment (e.g., repairing broken road segments obscured by tree canopies). This complementary interaction enables the model to surpass other attention combinations (such as CBAM) in both high-level semantic distribution (FID = 39.832) and visual structural similarity (SSIM = 0.817), while maintaining pixel-level accuracy (PSNR = 28.700). For instance, when generating complex urban scenes, this hybrid mechanism simultaneously optimizes building texture details (via channel attention) and street grid geometric regularity (via coordinate attention), ultimately producing maps that are both geographically constrained and visually realistic. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-12/3b137c0b-6e0d-4e67-ad39-c091deca79df/ad56ec619cffbe57ba2b89cfc8d7cf887f859722c45f2c46a238d6a8a46fa888.jpg)



Figure 14. Comparison of results generated by this model in Google Maps generation experiments using different attention mechanisms under the same network structure. From left to right are the real remote sensing image, ground truth, and generated maps using self-attention, channel attention, spatial attention, CBAM hybrid attention, and the attention model proposed by Ours.



Table 6. This model uses different attention mechanisms under the same network structure to generate maps in map generation experiments, with quantitative comparisons based on SSIM, FID, and PSNR evaluation criteria. From top to bottom, the attention mechanisms used are self-attention, channel attention, coordinate-based attention, spatial attention, CBAM hybrid attention, and our proposed attention mechanism. Bold and underlined entries denote the best and second-best results.


<table><tr><td>Attention</td><td>SSIM↑</td><td>FID↓</td><td>PSNR↑</td></tr><tr><td>Self-Attention</td><td>0.763</td><td>51.937</td><td>27.863</td></tr><tr><td>Channel</td><td>0.762</td><td>49.105</td><td>27.784</td></tr><tr><td>Coord</td><td>0.783</td><td>45.036</td><td>28.611</td></tr><tr><td>Spatial</td><td>0.767</td><td>52.381</td><td>27.846</td></tr><tr><td>CBAM</td><td>0.788</td><td>42.264</td><td>28.828</td></tr><tr><td>Ours</td><td>0.817</td><td>39.832</td><td>28.700</td></tr></table>

# 5. Conclusions

This study presents a diffusion-enhanced adversarial framework for generating structured tile maps from remote sensing imagery. Our model achieves precise spatial feature alignment through a novel cross-scale attention module by integrating progressive denoising mechanisms into a U-net backbone with adversarial training. Quantitative evaluations demonstrate superior performance over state-of-the-art methods in metrics and visual quality, particularly in rendering fine-grained boundaries and texture differentiation—critical requirements for high quality tile map generation. Compared to existing approaches, our solution exhibits enhanced structural fidelity with higher edge sharpness scores, effectively resolving ambiguities in complex geospatial patterns. Future work will integrate spectral normalization layers to improve color consistency, particularly in vegetation and rendering water bodies. 

Author Contributions: Conceptualization, C.S. and Z.C.; Data curation, L.Z.; Formal analysis, X.L.; Funding acquisition, X.L. and Z.C.; Investigation, Y.D.; Methodology, C.S. and X.F.; Project administration, C.S.; Resources, J.Z.; Software, X.F.; Supervision, X.L. and Y.D.; Validation, X.F.; Visualization, C.S. and L.Z.; Writing—original draft, C.S. and X.F.; Writing—review and editing, X.L. and Z.C. All authors have read and agreed to the published version of the manuscript. 

Funding: This study was funded by the National Natural Science Foundation of China under Grant 42471475; the Natural Science Foundation of Hubei Province, China (Grant No. 2025AFB107); the Opening Fund of Key Laboratory of Geological Survey and Evaluation of Ministry of Education (Grant No. GLAB 2024ZR06); the Fundamental Research Funds for the Central Universities; the Open Fund of Hubei Key Laboratory of Intelligent Vision Based Monitoring for Hydroelectric Engineering (No. 2024SDSJ10); and the Joint Open Fund of the Research Platforms of School of Computer Science, China University of Geosciences, Wuhan (No. PTLH2024-B-09). 

Data Availability Statement: The code and data for our key contributions will be open sourced in the code repository: https://github.com/Magician-MO/GDFM accessed on 2 July 2025. 

Acknowledgments: During the preparation of this manuscript, the author(s) used ChatGPT 4o and Gemini 2.5 Pro for the purposes of text polishing. The authors have reviewed and edited the output and take full responsibility for the content of this publication. 

Conflicts of Interest: Author Yuxuan Dong was employed by the company GeoScene Information Technology Co. Ltd. The remaining authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest. 

# References



1. Skidmore, A.K.; Bijker, W.; Schmidt, K.; Kumar, L. Use of Remote Sensing and GIS for Sustainable Land Management. ITC-J. 1997, 3, 302–315. 





2. Ezequiel, C.A.F.; Cua, M.; Libatique, N.C.; Tangonan, G.L.; Alampay, R.; Labuguen, R.T.; Favila, C.M.; Honrado, J.L.E.; Canos, V.; Devaney, C. UAV Aerial Imaging Applications for Post-Disaster Assessment, Environmental Management and Infrastructure Development. In Proceedings of the 2014 International Conference on Unmanned Aircraft Systems (ICUAS), Orlando, FL, USA, 20–30 May 2014; pp. 274–283. 





3. Tasar, O.; Happy, S.L.; Tarabalka, Y.; Alliez, P. ColorMapGAN: Unsupervised Domain Adaptation for Semantic Segmentation Using Color Mapping Generative Adversarial Networks. IEEE Trans. Geosci. Remote Sens. 2020, 58, 7178–7193. [CrossRef] 





4. Stefanakis, E. Web Mercator and Raster Tile Maps: Two Cornerstones of Online Map Service Providers. Geomatica 2017, 71, 100–109. [CrossRef] 





5. OGC 07-057r7; OpenGIS® Web Map Tile Service Implementation Standard. Open Geospatial Consortium Inc.: Arlington, TX, USA, 2010. 





6. Peterson, M.P. The Tile-Based Mapping Transition in Cartography. In Maps for the Future: Children, Education and Internet; Zentai, L., Reyes Nunez, J., Eds.; Springer: Berlin/Heidelberg, Germany, 2012; pp. 151–163, ISBN 978-3-642-19522-8. 





7. Haunold, P.; Kuhn, W. A Keystroke Level Analysis of Manual Map Digitizing. In Spatial Information Theory A Theoretical Basis for GIS; Frank, A.U., Campari, I., Eds.; Lecture Notes in Computer Science; Springer: Berlin/Heidelberg, Germany, 1993; Volume 716, pp. 406–420, ISBN 978-3-540-57207-7. 





8. Park, W.; Yu, K. Hybrid Line Simplification for Cartographic Generalization. Pattern Recognit. Lett. 2011, 32, 1267–1273. [CrossRef] 





9. Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; Bengio, Y. Generative Adversarial Nets. In Proceedings of the Advances in Neural Information Processing Systems 27; Ghahramani, Z., Welling, M., Cortes, C., Lawrence, N.D., Weinberger, K.Q., Eds.; Curran Associates, Inc.: Sydney, NSW, Australia, 2014; pp. 2672–2680. 





10. Isola, P.; Zhu, J.-Y.; Zhou, T.; Efros, A.A. Image-to-Image Translation with Conditional Adversarial Networks. In Proceedings of the 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 21–26 July 2017; pp. 5967–5976. 





11. Li, J.; Chen, Z.; Zhao, X.; Shao, L. MapGAN: An Intelligent Generation Model for Network Tile Maps. Sensors 2020, 20, 3119. [CrossRef] 





12. Liu, Y.; Wang, W.; Fang, F.; Zhou, L.; Sun, C.; Zheng, Y.; Chen, Z. CscGAN: Conditional Scale-Consistent Generation Network for Multi-Level Remote Sensing Image to Map Translation. Remote Sens. 2021, 13, 1936. [CrossRef] 





13. Ganguli, S.; Garzon, P.; Glaser, N. GeoGAN: A Conditional GAN with Reconstruction and Style Loss to Generate Standard Layer of Maps from Satellite Images. arXiv 2019, arXiv:1902.05611. 





14. Wolters, P.; Bastani, F.; Kembhavi, A. Zooming Out on Zooming In: Advancing Super-Resolution for Remote Sensing. arXiv 2023, arXiv:2311.18082. 





15. Chen, X.; Chen, S.; Xu, T.; Yin, B.; Peng, J.; Mei, X.; Li, H. SMAPGAN: Generative Adversarial Network-Based Semisupervised Styled Map Tile Generation Method. IEEE Trans. Geosci. Remote Sens. 2021, 59, 4388–4406. [CrossRef] 





16. Fu, Y.; Liang, S.; Chen, D.; Chen, Z. Translation of Aerial Image Into Digital Map via Discriminative Segmentation and Creative Generation. IEEE Trans. Geosci. Remote Sens. 2022, 60, 1–15. [CrossRef] 





17. Wang, T.-C.; Liu, M.-Y.; Zhu, J.-Y.; Tao, A.; Kautz, J.; Catanzaro, B. High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 18–23 June 2018; pp. 8798–8807. 





18. Pang, Y.; Lin, J.; Qin, T.; Chen, Z. Image-to-Image Translation: Methods and Applications. IEEE Trans. Multimed. 2022, 24, 3859–3881. [CrossRef] 





19. Solano-Carrillo, E.; Rodriguez, A.B.; Carrillo-Perez, B.; Steiniger, Y.; Stoppe, J. Look ATME: The Discriminator Mean Entropy Needs Attention. In Proceedings of the 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Vancouver, BC, Canada, 17–24 June 2023; pp. 787–796. 





20. Brock, A.; Donahue, J.; Simonyan, K. Large Scale GAN Training for High Fidelity Natural Image Synthesis. In Proceedings of the International Conference on Learning Representations, New Orleans, LA, USA, 6–9 May 2019. 





21. Sauer, A.; Schwarz, K.; Geiger, A. StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets. In Proceedings of the SIGGRAPH ’22: Special Interest Group on Computer Graphics and Interactive Techniques Conference, Vancouver, BC, Canada, 7–11 August 2022; Nandigjav, M., Mitra, N.J., Hertzmann, A., Eds.; ACM: New York, NY, USA, 2022; pp. 49:1–49:10. 





22. Lee, G.; Kim, H.; Kim, J.; Kim, S.; Ha, J.-W.; Choi, Y. Generator Knows What Discriminator Should Learn in Unconditional GANs. In Proceedings of the Computer Vision—ECCV 2022—17th European Conference, Tel Aviv, Israel, 23–27 October 2022; Proceedings, Part XVII. Avidan, S., Brostow, G.J., Cissé, M., Farinella, G.M., Hassner, T., Eds.; Springer: Berlin/Heidelberg, Germany, 2022; Volume 13677, pp. 406–422. 





23. Van Den Oord, A.; Vinyals, O. Neural Discrete Representation Learning. Adv. Neural Inf. Process. Syst. 2017, 30, 6309–6318. 





24. Ho, J.; Jain, A.; Abbeel, P. Denoising Diffusion Probabilistic Models. Adv. Neural Inf. Process. Syst. 2020, 33, 6840–6851. 





25. Dhariwal, P.; Nichol, A. Diffusion Models Beat Gans on Image Synthesis. Adv. Neural Inf. Process. Syst. 2021, 34, 8780–8794. 





26. Croitoru, F.-A.; Hondru, V.; Ionescu, R.T.; Shah, M. Diffusion Models in Vision: A Survey. IEEE Trans. Pattern Anal. Mach. Intell. 2023, 45, 10850–10869. [CrossRef] 





27. Wang, T.; Zhang, T.; Zhang, B.; Ouyang, H.; Chen, D.; Chen, Q.; Wen, F. Pretraining Is All You Need for Image-to-Image Translation. arXiv 2022, arXiv:2205.12952. 





28. Nichol, A.; Dhariwal, P. Improved Denoising Diffusion Probabilistic Models. arXiv 2021, arXiv:2102.09672. 





29. Song, J.; Meng, C.; Ermon, S. Denoising Diffusion Implicit Models. arXiv 2021, arXiv:2010.02502. 





30. Radford, A.; Metz, L.; Chintala, S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. In Proceedings of the 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, 2–4 May 2016; Conference Track Proceedings. Bengio, Y., LeCun, Y., Eds.; 2016. 





31. Arjovsky, M.; Chintala, S.; Bottou, L. Wasserstein Gan. arXiv 2017, arXiv:1701.07875. 





32. Zhu, J.-Y.; Park, T.; Isola, P.; Efros, A.A. Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. In Proceedings of the 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 22–29 October 2017; pp. 2242–2251. 





33. Karras, T.; Aila, T.; Laine, S.; Lehtinen, J. Progressive Growing of GANs for Improved Quality, Stability, and Variation. In Proceedings of the 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, 30 April–3 May 2018. Conference Track Proceedings; 2018. 





34. Xia, W.; Zhang, Y.; Yang, Y.; Xue, J.-H.; Zhou, B.; Yang, M.-H. GAN Inversion: A Survey. IEEE Trans. Pattern Anal. Mach. Intell. 2023, 45, 3121–3138. [CrossRef] 





35. Wang, Z.; She, Q.; Ward, T.E. Generative Adversarial Networks in Computer Vision: A Survey and Taxonomy. ACM Comput. Surv. 2021, 54, 1–38. [CrossRef] 





36. Gui, J.; Sun, Z.; Wen, Y.; Tao, D.; Ye, J. A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications. IEEE Trans. Knowl. Data Eng. 2023, 35, 3313–3332. [CrossRef] 





37. Karras, T.; Laine, S.; Aila, T. A Style-Based Generator Architecture for Generative Adversarial Networks. In Proceedings of the CVPR 2019, Long Beach, CA, USA, 15–20 June 2019. 





38. Karras, T.; Laine, S.; Aittala, M.; Hellsten, J.; Lehtinen, J.; Aila, T. Analyzing and Improving the Image Quality of StyleGAN. In Proceedings of the 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 14–19 June 2020; pp. 8107–8116. 





39. Li, Y.; Peng, X.; Wu, Z.; Yang, F.; He, X.; Li, Z. M3GAN: A Masking Strategy with a Mutable Filter for Multidimensional Anomaly Detection. Knowl.-Based Syst. 2023, 271, 110585. [CrossRef] 





40. Yang, J.; Shao, Y.; Li, C.-N. CNTS: Cooperative Network for Time Series. IEEE Access 2023, 11, 31941–31950. [CrossRef] 





41. Hu, J.; Shen, L.; Sun, G. Squeeze-and-Excitation Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 18–23 June 2018; pp. 7132–7141. 





42. Hou, Q.; Zhou, D.; Feng, J. Coordinate Attention for Efficient Mobile Network Design. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Nashville, TN, USA, 20–25 June 2021; pp. 13713–13722. 





43. Liu, C.; Chen, K.; Zhao, R.; Zou, Z.; Shi, Z. Text2Earth: Unlocking Text-Driven Remote Sensing Image Generation with a Global-Scale Dataset and a Foundation Model. IEEE Geosci. Remote Sens. Mag. 2025, 2–23. [CrossRef] 





44. Sebaq, A.; ElHelw, M. RSDiff: Remote Sensing Image Generation from Text Using Diffusion Model. Neural Comput. Appl. 2024, 36, 23103–23111. [CrossRef] 





45. Tian, J.; Wu, J.; Chen, H.; Ma, M. MapGen-Diff: An End-to-End Remote Sensing Image to Map Generator via Denoising Diffusion Bridge Model. Remote Sens. 2024, 16, 3716. [CrossRef] 





46. Woo, S.; Park, J.; Lee, J.-Y.; Kweon, I.S. CBAM: Convolutional Block Attention Module. In Proceedings of the European Conference on Computer Vision (ECCV), Munich, Germany, 8–14 September 2018; pp. 3–19. 





47. Fu, Y.; Fang, Z.; Chen, L.; Song, T.; Lin, D. Level-Aware Consistent Multilevel Map Translation from Satellite Imagery. IEEE Trans. Geosci. Remote Sens. 2023, 61, 1–14. [CrossRef] 





48. Wang, Z.; Bovik, A.C.; Sheikh, H.R.; Simoncelli, E.P. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Trans. Image Process. 2004, 13, 600–612. [CrossRef] 





49. Heusel, M.; Ramsauer, H.; Unterthiner, T.; Nessler, B.; Hochreiter, S. Gans Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Proceedings of the Advances in Neural Information Processing Systems, Long Beach, CA, USA, 4–9 September 2017; Volume 30. 





50. Park, T.; Liu, M.-Y.; Wang, T.-C.; Zhu, J.-Y. Semantic Image Synthesis with Spatially-Adaptive Normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Long Beach, CA, USA, 15–20 June 2019; pp. 2332–2341. 





51. Tang, H.; Xu, D.; Sebe, N.; Wang, Y.; Corso, J.J.; Yan, Y. Multi-Channel Attention Selection GAN with Cascaded Semantic Guidance for Cross-View Image Translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and pattern recognition, Long Beach, CA, USA, 15–20 June 2019; pp. 2417–2426. 





52. Jiang, L.; Zhang, C.; Huang, M.; Liu, C.; Shi, J.; Loy, C.C. TSIT: A Simple and Versatile Framework for Image-to-Image Translation. In Proceedings of the European Conference on Computer Vision (ECCV), Glasgow, UK, 23–28 August 2020; pp. 206–222. 





53. Liang, J.; Zeng, H.; Zhang, L. High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Nashville, TN, USA, 20–25 June 2021; pp. 9392–9400. 



Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content. 