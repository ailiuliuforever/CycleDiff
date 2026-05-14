# EarthMapper: Visual Autoregressive Models for Controllable Bidirectional Satellite–Map Translation

Zhe Dong, Yuzhe Sun, Tianzhu Liu, Member, IEEE, Wangmeng Zuo, Senior Member, IEEE, and Yanfeng Gu, Senior Member, IEEE, 

Abstract—Satellite imagery and maps, as two fundamental data modalities in remote sensing, offer direct observations of the Earth’s surface and human-interpretable geographic abstractions, respectively. The task of bidirectional translation between satellite images and maps (BSMT) holds significant potential for applications in urban planning and disaster response. However, this task presents two major challenges: first, the absence of precise pixel-wise alignment between the two modalities substantially complicates the translation process; second, it requires achieving both high-level abstraction of geographic features and high-quality visual synthesis, which further elevates the technical complexity. To address these limitations, we introduce EarthMapper, a novel autoregressive framework for controllable bidirectional satellite-map translation. EarthMapper employs geographic coordinate embeddings to anchor generation, ensuring regionspecific adaptability, and leverages multi-scale feature alignment within a geo-conditioned joint scale autoregression (GJSA) process to unify bidirectional translation in a single training cycle. A semantic infusion (SI) mechanism is introduced to enhance feature-level consistency, while a key point adaptive guidance (KPAG) mechanism is proposed to dynamically balance diversity and precision during inference. We further contribute CNSatMap, a large-scale dataset comprising 302,132 precisely aligned satellite-map pairs across 38 Chinese cities, enabling robust benchmarking. Extensive experiments on CNSatMap and the New York dataset demonstrate EarthMapper’s superior performance, achieving significant improvements in visual realism, semantic consistency, and structural fidelity over state-of-the-art methods. Additionally, EarthMapper excels in zero-shot tasks like in-painting, out-painting and coordinateconditional generation, underscoring its versatility. The source code for EarthMapper and the CNSatMap dataset will be publicly available at https://github.com/HIT-SIRS/EarthMapper. 

Index Terms—Bidirectional satellite-map translation (BSMT), remote sensing, controllable image generation (CIG), crossmodal. 

# I. INTRODUCTION

R Emote sensing technology has emerged as a pivotaltool for acquiring geospatial information, with satellite toolforacquiringgeospatialinformation,withsateite imagery and cartographic maps serving as two primary modalities. Satellite images capture raw, unprocessed representations of the Earth’s surface, while maps provide abstract, human-interpretable depictions of geographic features. In practical applications, as illustrated in Fig. 1, bidirectional translation between these modalities is often 

Z. Dong, Y. Sun, T. Liu and Y. Gu are with the School of Electronics and Information Engineering, Harbin Institute of Technology, Harbin 150001, China. (email: guyf@hit.edu.cn). 

W. Zuo is with the Faculty of Computing, Harbin Institute of Technology, Harbin, China, and also with the Peng Cheng Lab, Shenzhen, China. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/3a2c0d9443630e8f62253dd4f18d1907046a42e9040959fd998b9dc84c2d7500.jpg)



Fig. 1. Conceptual illustration of bidirectional satellite-map translation and their respective applications.


indispensable: transforming satellite imagery into maps facilitates rapid comprehension of key geographic information [1], whereas synthesizing realistic satellite images from edited maps enables rapid scenario simulations for post-disaster reconstruction, urban planning, and augmentation of satellite datasets with rare scenes [2]. Consequently, bidirectional satellite-map translation (BSMT) holds significant potential for advancing both remote sensing applications and automated cartography. 

Recent advancements in deep learning have spurred the development of powerful generative models, achieving remarkable success in image-to-image translation tasks. Generative adversarial networks (GANs) [3], [4] have been a cornerstone of this progress, leveraging adversarial training to refine synthetic outputs through a competitive interplay between generator and discriminator networks. Variants such as Pix2Pix [5] and CycleGAN [6] have demonstrated exceptional performance in style transfer and image enhancement. Meanwhile, diffusion models [7], [8] have introduced a probabilistic framework for highfidelity image synthesis, progressively refining details through iterative denoising. More recently, autoregressive (AR) models [9], [10] have gained prominence by explicitly modeling pixel-level dependencies, offering superior generation quality and diversity. Notably, VAR [11] pioneered multi-scale autoregressive prediction to mitigate context fragmentation and blurriness in traditional AR approaches. Subsequent works like ControlVar [12] and CAR [13] further incorporated conditional control mechanisms, proving highly effective in cross-domain mapping tasks. These advances inspire our exploration of AR-based frameworks for high-quality BSMT. Nevertheless, dedicated research on BSMT remains scarce, necessitating further methodological innovation. 

Despite the prowess of existing generative models, BSMT presents unique challenges beyond conventional image translation or style transfer. Satellite images and maps lack strict pixel-wise correspondence, as cartographic abstraction involves selective emphasis, simplification, and symbolization to enhance human interpretability. Merely performing pixel-level classification on satellite data fails to produce high-quality maps. Conversely, reconstructing realistic satellite imagery from maps demands not only textural synthesis but also semantically coherent and geographically plausible scene generation. Thus, BSMT requires models capable of both high-level abstraction and photorealistic reconstruction—a dual objective that existing methods struggle to fulfill. 

To address these challenges, we propose EarthMapper, a novel autoregressive framework for BSMT with geographic coordinate embedding. First, EarthMapper initiates generation from geographic coordinates, enabling region-specific adaptation to diverse terrains and urban layouts. Second, it enforces multi-scale feature alignment between conditional inputs and generated outputs within the designed geoconditioned joint scale autoregression (GJSA) process, permitting bidirectional translation via a single training cycle. Third, instead of direct pixel manipulation, EarthMapper modulates the autoregressive probability distribution at each step, balancing fidelity and diversity. Additionally, we introduce a semantic infusion (SI) mechanism during training, which constrains generation by minimizing featurespace discrepancies between synthetic and real images. For inference, we devise key point adaptive guidance (KPAG), comprising: (1) key point force (KPF), which anchors generation to salient geographic features in the conditional input to prevent excessive deviation; and (2) complexity guidance (CG), which dynamically adjusts the conditioning strength based on per-stage image complexity. 

To propel BSMT research, we curate CNSatMap, a largescale dataset of 302,132 precisely aligned satellite-map pairs spanning 38 major Chinese cities. Satellite images, sourced from Google Earth at 0.6-meter resolution, preserve intricate ground details, while vector maps from Tianditu service offer cartographically rigorous representations. Covering diverse climates, population densities, and urbanization levels, CNSatMap provides a robust benchmark for evaluating cross-modal translation algorithms. This dataset bridges a critical gap in BSMT research and promises broad utility in urban scene parsing, automated mapping, and geospatial feature extraction. 

In summary, the contributions of this work can be summarized in the following three aspects: 

(1) We construct CNSatMap, the first large-scale, highprecision dataset for BSMT, enabling rigorous exploration of cross-modal geographic translation and fostering advancements in urban analytics and automated cartography. 

(2) We propose EarthMapper, a AR-based generative framework that unifies GJSA process via multi-scale alignment, achieving superior generative capability and versatility. 

(3) The SI mechanism is designed to enforce featurelevel consistency between generated and real images, enhancing semantic fidelity in weakly aligned satellitemap pairs. 

(4) We introduce KPAG, which intelligently balances diversity and accuracy through key-point anchoring and dynamic complexity modulation, ensuring cartographic precision while preserving realism. 

The remainder of this paper is structured as follows: Section II surveys related work in controllable image generation, vision autoregressive models, and remote sensing image generation. Section III introduces CNSatMap dataset, detailing its construction, coverage, and quality controls. In Section IV, we describe the proposed EarthMapper in detail. Section V evaluates EarthMapper through comprehensive experiments, including comparisons with state-of-the-art methods, ablation studies, and zero-shot generalization across diverse tasks. Finally, Section VI summarizes findings and future directions. 

# II. RELATED WORK

# A. Controllable Image Generation

Controllable image generation (CIG) has emerged as a cornerstone in the fields of computer vision and generative modeling, offering transformative potential across diverse applications, including artistic creation, image editing, and automated content generation. Contemporary approaches to CIG can be systematically classified into three distinct paradigms based on the nature of the input conditions: labelbased control, visual control, and text-based control. 

Label-based control leverages structured annotations—such as semantic segmentation masks, class labels, or layout descriptors—to guide image synthesis with high precision. This paradigm excels in applications requiring strict spatial or categorical alignment. Early work, exemplified by conditional generative adversarial networks (CGANs) [4], demonstrated that conditioning generators on discrete class labels enhances generation quality. Recent innovations like ControlNet [14] advance this framework by integrating spatial cues (e.g., edge maps or human poses) into diffusion models [15] via trainable adapters, achieving fine-grained control over object placement and structure. In layout-to-image synthesis, GLIGEN [16] introduces gated self-attention layers to process bounding box information, enabling seamless compositional generation of multiple objects. These techniques often depend on paired training data—a persistent bottleneck—yet advances in multitask learning and dataset augmentation are alleviating this constraint. 

Visual control employs image-based inputs—such as sketches, edge maps, or reference images—to direct generative processes, effectively bridging abstract textual prompts and fine-grained pixel-level outcomes. Techniques like GAN inversion [17] map images into latent spaces for semantic editing, with advancements like InterFaceGAN [18] enabling real-time attribute manipulation. Cutting-edge brainguided generation [19] decodes functional magnetic resonance imaging (fMRI) or electroencephalography (EEG) signals to reconstruct visual concepts, highlighting cross-modal innovation. For artistic domains, methods such as Textual Inversion [20] and DreamBooth [21] fine-tune models with minimal reference images, preserving distinct visual identities. Challenges remain in interpreting ambiguous sketches or incomplete inputs, yet hybrid strategies integrating diffusion models with attention mechanisms are enhancing robustness. 

Text-based control has emerged as a cornerstone of CIG research, prized for its adeptness at encoding high-level semantics. Trailblazing models such as GLIDE [22] and DALL·E 2 [23] solidified text-conditioned diffusion as the state-of-the-art, harnessing large-scale language models like CLIP [24] to achieve robust cross-modal alignment. Stable diffusion [15] marked a leap in efficiency by operating within a compressed latent space, facilitating high-resolution synthesis on consumer-grade hardware. Innovations like classifier-free guidance [25] strike an elegant balance between diversity and fidelity through the joint training of conditional and unconditional diffusion models. Recent advances tackle multilingual generation and compositional reasoning, as exemplified by composable diffusion [26], though persistent challenges, such as text-image misalignment, remain. 

# B. Vision Autoregressive Models

Autoregressive models, renowned for their success in natural language processing (NLP) due to their ability to model long-range dependencies and generate coherent sequences [27], [28], have recently emerged as powerful tools in computer vision. These models have demonstrated significant potential in tasks such as image generation [9], [29], super-resolution [30], [31], image editing [32], [33], and image-to-image translation [34], driven by their sequential prediction framework. Depending on the representation strategy, autoregressive vision models can be broadly categorized into pixel-based, token-based, and scale-based approaches. 

Pixel-based autoregressive models, such as PixelRNN [35] and PixelCNN [10], treat images as 1D sequences of pixels, predicting each pixel conditioned on previous ones using architectures like LSTMs or CNNs. While effective for low-resolution image generation, these models struggle with high-resolution tasks due to quadratic computational complexity and inherent redundancy in pixel-level predictions. Although parallelization techniques [36] have been explored, the generated images often suffer from blurriness and suboptimal quality, highlighting the limitations of pixel-based approaches for scaling to higher resolutions. 

Token-based autoregressive models address these limitations by leveraging discrete token representations. Frameworks like VQ-VAE [37], VQ-VAE-2 [38], and VQGAN [39] compress images into sequences of discrete tokens using vector quantization (VQ), enabling efficient processing of high-resolution content. These models typically employ a twostage process: an encoder-decoder architecture learns discrete latent representations, followed by an autoregressive model that predicts token sequences for generation. By integrating Transformer-based decoders [40] and perceptual losses, tokenbased models achieve state-of-the-art performance in highresolution image generation while maintaining computational efficiency. 

Scale-based autoregressive models, such as VAR [11], introduce a hierarchical approach to image generation by processing visual content across multiple scales, from coarse to fine. Unlike token-based models that predict tokens sequentially, VAR generates entire token maps at each scale using residual quantization (RQ), as introduced in RQ-VAE [41]. This recursive quantization of feature residuals allows for compact representation of high-resolution images while preserving fine details. The multi-scale framework enables parallel token generation within each map, improving spatial locality and computational efficiency. 

# C. Remote Sensing Image Generation

Remote sensing image generation has become a critical task in Earth observation, addressing the growing need for synthetic images that accurately mimic those captured by satellites or unmanned aerial vehicles (UAVs). The scarcity and high cost of acquiring high-resolution annotated data, coupled with the variability introduced by factors such as acquisition time, weather conditions, and sensor types, make this task both challenging and essential. Recent advancements in generative models have shown significant promise in overcoming these challenges, enabling the synthesis of high-quality remote sensing images for applications ranging from environmental monitoring and urban planning to agricultural assessment. 

Among these advancements, HSIGene [42] stands out as a foundation model for hyperspectral image generation. By leveraging latent diffusion models (LDMs), HSIGene can produce precise hyperspectral images with detailed spectral information, which is critical for applications such as crop health monitoring and environmental analysis. Besides, RSDiff [43] and CRS-Diff [44] have introduced multi-stage diffusion processes to generate high-resolution satellite imagery from text prompts. RSDiff employs a two-stage framework, combining a low-resolution diffusion model (LRDM) with a super-resolution diffusion model (SRDM) to enhance spatial details, while CRS-Diff incorporates multi-scale feature fusion to improve control over conditional inputs such as text, metadata, and reference images. These models excel in handling multispectral and time-series data, offering superior geographic detail and resolution compared to traditional methods. Further expanding the scope of generative capabilities, MetaEarth [45] introduces a resolution-guided self-cascading framework, enabling the generation of unbounded, large-scale remote sensing images tailored to specific geographic and resolution requirements. This approach is particularly valuable for global-scale applications such as climate modeling and urban development planning. The field has also been significantly advanced by DiffusionSat [46], which supports diverse conditional generation tasks such as environmental monitoring and crop yield prediction, demonstrating the versatility and scalability of diffusion models in Earth observation. Despite these impressive advancements, challenges remain in achieving precise conditional control and ensuring the stability of generated images in complex scenarios. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/590c1f5055e098a9e4b1151af96f681cb363c9a3ff823ec2f6a600137302b6fa.jpg)



Fig. 2. Illustration of the geographical distribution of satellite-map pairs sampled from the proposed CNSatMap dataset.


# III. THE CNSATMAP DATASET

In this study, we introduce the CNSatMap dataset, a comprehensive resource designed for cross-modal translation tasks between remote sensing satellite imagery and cartographic maps. CNSatMap comprises 302,132 meticulously aligned satellite-map image pairs, making it the largest publicly available dataset of its kind. It is tailored to support geospatial analysis, cross-modal image translation, and urban scene understanding across diverse geographical contexts. The dataset was constructed through a rigorous pipeline to ensure high quality, precise alignment, and semantic richness, as detailed below. 

The satellite imagery in CNSatMap is sourced from Google Earth, captured at zoom level 19 with a spatial resolution of approximately 0.6 meters per pixel. This resolution preserves fine-grained spatial details essential for applications such as urban planning, infrastructure inspection, and land-use classification. For cartographic data, we exclusively utilize the Tianditu map service, a vector-based mapping platform maintained by China’s National Geomatics Center. This choice was motivated by the limitations of alternative services, such as Google Maps, which exhibit infrequent updates in Chinese regions—often lagging by months or years—and insufficient coverage of localized geographical features. In contrast, Tianditu provides up-to-date, high-precision cartographic data with exceptional fidelity in administrative boundaries, terrain representation, and urban infrastructure details. 

To ensure broad geographical representativeness, CNSatMap encompasses 38 major Chinese cities, including national capitals, provincial capitals, municipalities, and first-tier cities: Beijing, Changchun, Changsha, Chengdu, Chongqing, Dalian, Fuzhou, Guangzhou, Guiyang, Haikou, Hangzhou, Harbin, Hefei, Hohhot, Jinan, Kunming, Lanzhou, Lhasa, Nanchang, Nanjing, Ningbo, Qingdao, Shanghai, Shenyang, Shenzhen, Shijiazhuang, Suzhou, Taiyuan, Tianjin, Wuhan, Urumqi, Wuxi, Xiamen, Xi’an, Hong Kong, Xining, Yinchuan, and Zhengzhou. These cities span diverse climatic zones, population densities, and urbanization levels. Coastal metropolises like Shanghai and Shenzhen exemplify highly developed urban agglomerations, while inland cities such as Lhasa and Urumqi reflect unique topographical and sparsely populated characteristics. This distribution, illustrated in Fig. 2, underscores the dataset’s socioeconomic and geospatial diversity. 

The construction of CNSatMap involved a multi-stage workflow to ensure alignment accuracy and semantic coherence. Raw satellite and map imagery for the 38 cities were collected and georeferenced using the EPSG:3857 (Web Mercator) projection, the standard for web mapping services. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/4b2b500ef4bd35d274332531f9d787f8a0e785f5f5917e081f754a04ad952dda.jpg)



Fig. 3. Overview of our proposed EarthMapper framework, with the upper section dedicated to training and the lower section to inference. During training, paired satellite and map data are initially processed by a frozen image encoder and quantized into vector representations using the hierarchical residual quantization (HRQ) module. Concurrently, geographic coordinates are encoded as initial vectors and fed into the geo-conditioned joint scale autoregression (GJSA) model, which generates image content by progressively increasing the resolution. The resulting output is then passed to the semantic infusion (SI) module, where it is aligned with ground truth features to improve the realism of the generated image, before being decoded into an image using the VAE decoder. In the inference phase, for instance, when generating satellite images from maps, the map is quantized into vectors, combined with geographic coordinate vectors, and input into the model. Key points are computed using the key point force (KPF) method and incorporated into the generation process, while complexity guidance (CG) dynamically modulates the control intensity. Finally, the output is decoded to produce the satellite image.


Each image was partitioned into non-overlapping $2 5 6 \times 2 5 6 -$ pixel tiles, balancing computational efficiency with contextual preservation—a common input size for vision generation models in computer vision. Tile boundary coordinates were then transformed from EPSG:3857 to EPSG:4326 (WGS84 latitude-longitude), ensuring compatibility with geospatial information systems (GIS) through embedded metadata. A geometric alignment process was applied to correct potential misregistrations caused by discrepancies in data sources or georeferencing errors, ensuring pixel-level correspondence between satellite and map modalities. 

From an initial pool of over 600,000 tiles, three filtering criteria were enforced to refine the dataset. First, satellite tiles obscured by cloud cover or atmospheric artifacts were removed. Second, tiles with significant building tilt due to non-orthogonal imaging angles were excluded. Finally, map tiles with low structural diversity (standard deviation of color distribution < 10) were eliminated, targeting homogeneous regions such as water bodies, forests, or barren land. These filters yielded a refined dataset of 302,132 high-quality pairs, emphasizing urban centers and mixed-use landscapes. 

CNSatMap distinguishes itself through its unprecedented scale, geographical diversity, and meticulous construction. Its high-resolution imagery, expansive coverage, and stringent quality controls establish it as a robust benchmark for algorithms bridging the visual-semantic gap between satellite imagery and cartographic representations. Potential applications span urban scene parsing, automated cartography, and geospatial feature extraction. 

# IV. METHODOLOGY

# A. Preliminaries

Contemporary visual generation frameworks predominantly employ autoregressive modeling with next-token prediction objectives. In conventional AR paradigms, images undergo spatial compression through visual autoencoders to derive latent representations, which are subsequently quantized into discrete token sequences $\textbf { x } = ~ \left( x _ { 1 } , x _ { 2 } , \ldots , x _ { T } \right)$ . The AR model then sequentially predicts each token conditioned on its historical context: 

$$
p (\mathbf {x} | c) = \prod_ {t = 1} ^ {T} p (x _ {t} | x _ {<   t}, c) \tag {1}
$$

where c denotes optional conditioning signals. However, this token-wise generation paradigm manifests three fundamental limitations: (a) structural disintegration arising from the inherent conflict between unidirectional sequential modeling and the bidirectional spatial dependencies in visual data, (b) computational inefficiency stemming from obligatory raster-scan generation that prohibits parallel computation within spatial scales, and (c) constrained contextual reasoning capacity due to the absence of explicit hierarchical structural modeling in conventional AR frameworks. 

To overcome these fundamental limitations, recent advances in autoregressive visual modeling have introduced a paradigm shift from token-level to scale-level prediction. The proposed next-scale prediction framework establishes hierarchical autoregression through three pivotal components: First, a multi-scale visual encoder (VQ-VAE [37]) decomposes the input image I into K hierarchical feature maps $\left\{ f _ { k } \right\} _ { k = 1 } ^ { K }$ with progressively increasing spatial resolution: 

$$
f _ {k} = \mathcal {E} _ {k} (I) - \sum_ {m = 1} ^ {k - 1} \mathcal {U} (\mathcal {Q} (f _ {m})) \tag {2}
$$

where $\mathcal { E } _ { k } : \mathbb { R } ^ { H \times W \times 3 }  \mathbb { R } ^ { h _ { k } \times w _ { k } \times d }$ denotes the k-th scale encoder with spatial compression ratio $( H / h _ { k } , W / w _ { k } ) , \mathcal { U } ( \cdot )$ represents the spatial upsampling operator, and $\mathcal { Q } ( \cdot )$ performs vector quantization. This residual formulation preserves spatial coherence while enabling explicit hierarchical decomposition. 

Each continuous feature map $f _ { k } \in \mathbb { R } ^ { h _ { k } \times w _ { k } \times d }$ undergoes codebook projection to discrete token maps through nearestneighbor quantization: 

$$
r _ {k} ^ {(i, j)} = \underset {v \in \mathcal {V}} {\arg \min} | \mathcal {V} [ v ] - f _ {k} ^ {(i, j)} | _ {2} \tag {3}
$$

where $\boldsymbol { \mathcal { V } } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ represents the learned codebook and $r _ { k } \in$ $\nu ^ { h _ { k } \times w _ { k } }$ constitutes the discrete representation at scale k. 

The hierarchical autoregressive process then operates through scale-conditioned generation: 

$$
p (\mathbf {r} | c) = \prod_ {k = 1} ^ {K} p (r _ {k} | r _ {<   k}, c) \tag {4}
$$

where $\textbf { r } = \mathbf { \Psi } ( r _ { 1 } , \dots , r _ { K } )$ represents the complete multiscale representation. This formulation enables simultaneous prediction of all spatial positions within each scale while maintaining inter-scale dependencies, effectively resolving the limitations of conventional token-wise autoregression. 

# B. Overview of EarthMapper

EarthMapper introduces an advanced autoregressive framework tailored for controllable bidirectional translation between satellite imagery and cartographic maps, adeptly addressing the complexities of cross-modal geospatial synthesis. As depicted in Fig. 3, its core architecture hinges on the geo-conditioned joint scale autoregression (GJSA) mechanism, which orchestrates multi-scale feature alignment between input and output modalities within a unified training paradigm, facilitating seamless bidirectional translation. To enhance semantic fidelity, EarthMapper integrates a semantic infusion (SI) mechanism during training, minimizing featurespace disparities between synthetic and real images. During inference, a key point adaptive guidance (KPAG) mechanism dynamically modulates generation, balancing diversity and precision by anchoring outputs to salient geospatial features and adjusting conditioning strength based on image complexity. 

# C. Hierarchical Residual Quantization

For a given satellite image S and the corresponding map M , an encoder E first transforms these inputs into latent feature maps $E ( S ) , E ( M ) \in \mathbb { R } ^ { h \times w \times d }$ , where $h \times w$ denotes the spatial dimensions and d represents the feature dimensionality. To convert these continuous representations into discrete tokens suitable for autoregressive modeling, a baseline quantization step employs a learned codebook $\mathcal { Z } =$ $\{ z _ { k } \} _ { k = 1 } ^ { K } \ \stackrel { \textstyle \subset } { \subset } \ \mathbb { R } ^ { d }$ , where each $z _ { k }$ is a prototype vector in the latent space. The quantization operation assigns each spatial feature vector $\hat { z } _ { i j } \in \mathbb { R } ^ { d _ { - } }$ —extracted from $\pmb { { \cal E } } ( S )$ or ${ \pmb E } ( { \pmb M } ) - \mathrm { t o }$ its nearest codebook entry: 

$$
z _ {q} ^ {(i, j)} = \arg \min _ {z _ {k} \in \mathcal {Z}} \| \hat {z} _ {i, j} - z _ {k} \| _ {2} \tag {5}
$$

where $z _ { q } ^ { ( i , j ) }$ z q is the index of the selected codebook vector, and $\mathcal { Z } [ \tilde { z _ { q } ^ { ( i , j ) } } ] \quad \in \quad \mathbb { R } ^ { d }$ denotes the corresponding discrete representation. This process establishes a foundation for tokenbased generation but struggles to capture the multi-scale intricacies inherent in high-resolution remote sensing imagery without incurring significant computational costs through an expanded codebook. 

To address this limitation, inspired by [11], we introduce a hierarchical residual quantization (HRQ) strategy, designed to efficiently encode the rich, multi-scale structure of geospatial data in a coarse-to-fine manner. Unlike traditional VQ, which relies on a single quantization step, HRQ employs a recursive residual approach with a fixed-size codebook $\mathcal { C } \subset \mathbb { R } ^ { d }$ . For a given feature vector $z = \hat { z } _ { i , j } , \mathrm { H R Q }$ generates a sequence of quantization indices: 

$$
\mathrm{HRQ} (z; \mathcal {C}, D) = (k _ {1}, k _ {2}, \dots , k _ {D}) \tag {6}
$$

where $k _ { d } ~ =$ arg min $\mathsf { l } _ { z _ { i } \in \mathcal { C } } \left\| r _ { d - 1 } - z _ { i } \right\| _ { 2 }$ , with the residuals defined iteratively as $r _ { 0 } = z$ and $r _ { d } = \bar { r } _ { d - 1 } - \mathcal { C } \left[ k _ { d } \right]$ for $d =$ $1 , \ldots , D$ . Here, D denotes the quantization depth, controlling the granularity of the approximation. The reconstructed feature is approximated as $\begin{array} { l l l } { \hat { z } } & { \approx } & { \sum _ { d = 1 } ^ { D } { \mathcal C } \left[ k _ { d } \right] } \end{array}$ , enabling progressive refinement of residual errors. This approach compacts high-resolution representations into a hierarchical sequence of discrete indices, preserving fine-grained details and global structures without the computational overhead of large codebooks typical in conventional VQ-VAE frameworks. 

Leveraging this HRQ mechanism, EarthMapper employs a multi-scale quantization autoencoder to discretize S and M into sets of token maps $\{ s _ { 1 } , s _ { 2 } , \ldots , s _ { k } \}$ and $\{ m _ { 1 } , m _ { 2 } , \dots , m _ { k } \}$ , respectively. At each scale k, the encoder $\scriptstyle { E _ { k } }$ produces feature maps $\bar { \mathbf { \Gamma } } \bar { E _ { k } } ( S ) , E _ { k } ( M ) \ \in \ \mathbb { R } ^ { h _ { k } \times w _ { k } \times d }$ , with spatially varying resolutions $h _ { k } \ \times \ w _ { k }$ reflecting the hierarchical decomposition. These are then quantized via HRQ: 

$$
\mathbf {s} _ {k} ^ {(i, j)} = \operatorname{HRQ} \left(\boldsymbol {E} _ {k} (\mathbf {S}) ^ {(i, j)}; \mathcal {C}, D\right), \tag {7}
$$

$$
\mathbf {m} _ {k} ^ {(i, j)} = \mathrm{HRQ} \left(\boldsymbol {E} _ {k} (\mathbf {M}) ^ {(i, j)}; \mathcal {C}, D\right),
$$

yielding token maps $\begin{array} { l l l } { s _ { k } , m _ { k } } & { \in } & { \mathbb { Z } ^ { h _ { k } \times w _ { k } \times D } } \end{array}$ that encode multi-depth indices at each spatial position. This multi-scale tokenization captures the hierarchical nature of remote sensing data, aligning with the perceptual progression from coarse (e.g., regional topography) to fine (e.g., infrastructure details) scales. By integrating spatial coherence and resolution adaptability, HRQ ensures that EarthMapper meets the stringent fidelity requirements of BSMT task, enhancing both the realism and utility of the generated outputs for applications such as urban planning and environmental monitoring. 

# D. Geo-Conditioned Joint Scale Autoregression

In EarthMapper, we propose a pioneering geo-conditioned autoregressive framework to jointly model satellite imagery and corresponding maps, leveraging geographic coordinates as a spatially-informed control signal. Unlike prior works such as ControlVAR [12], which primarily explore class-level controls for general imagery, our approach introduces a domain-specific innovation by conditioning the AR process on geospatial context. This enables the generation of satellite image-map pairs that are inherently aligned with real-world locations, addressing a critical need in remote sensing applications. 

We first reformulate geographic coordinates $g \ = \ ( \phi , \lambda )$ as a continuous control signal through sinusoidal positional encoding: 

$$
\boldsymbol {c} _ {g} = \mathrm{MLP} (\text { Sin   Embed } (\phi) \oplus \text { SinEmbed } (\lambda)) \tag {8}
$$

where $\phi$ and λ denote latitude and longitude, respectively, and $\oplus$ represents concatenation. This encoding scheme maintains spatial continuity across the Earth’s surface, effectively accounting for the non-Euclidean properties inherent in geographic coordinates. 

After that, We redefine the conditional AR generation task to model the joint distribution $p ( S , M \mid c _ { g } )$ . Building on the multi-scale tokenization established by HRQ in the previous subsection, we derive discrete multi-scale feature maps for satellite imagery and maps, represented as: 

$$
\left\{\boldsymbol {s} _ {k} \right\} _ {k = 1} ^ {K} = \Phi_ {S} (\boldsymbol {S}), \quad \left\{\boldsymbol {m} _ {k} \right\} _ {k = 1} ^ {K} = \Phi_ {M} (\boldsymbol {M}) \tag {9}
$$

where both modalities are decomposed into K hierarchical scales using shared codebooks, this approach ensures geometric consistency between the latent spaces of satellite imagery and map representations across all scales. 

To preserve the autoregressive property while capturing cross-modal dependencies, we pair tokens at each scale into a joint representation ${ \pmb r } _ { k } = ( s _ { k } , m _ { k } )$ and define the generative process as: 

$$
p (\boldsymbol {S}, \boldsymbol {M} \mid c _ {g}) = \prod_ {k = 1} ^ {K} p \left(\boldsymbol {r} _ {k} \mid \boldsymbol {r} _ {<   k}, c _ {g}\right) \tag {10}
$$

where $\pmb { r } _ { < k } = \{ \pmb { r } _ { 1 } , \pmb { r } _ { 2 } , \dots , \pmb { r } _ { k - 1 } \}$ denotes the sequence of prior token pairs. This hierarchical joint modeling is particularly crucial for the BSMT task, where the geometric alignment between satellite features (e.g., building footprints) and their map representations (e.g., polygon symbols) must be preserved across scales. 

Following [11], EarthMapper employs a GPT-2-style Transformer architecture to implement the geo-conditioned AR process. Joint modeling optimizes the joint likelihood using a cross-entropy loss, supervising the prediction of $\mathbf { \nabla } _ { \mathbf { r } _ { t } }$ given $\scriptstyle { \pmb { r } } _ { < t }$ and $c _ { g } \mathrm { : }$ 

$$
\mathcal {L} _ {\mathrm{jot}} = \mathbb {E} _ {\boldsymbol {r} \sim p (\boldsymbol {r} | c _ {g})} \left[ - \sum_ {k = 1} ^ {K} \log p \left(\boldsymbol {r} _ {k} \mid \boldsymbol {r} _ {<   k}, c _ {g}\right) \right] \tag {11}
$$

where $\mathcal { L } _ { \mathrm { j o t } }$ presents the joint modeling loss. 

# E. Semantic Infusion for Geovisual Coherence

Although the geo-conditioned AR framework in EarthMapper adeptly generates structurally aligned satellite imagery and maps based on geographic coordinates, ensuring geovisual coherence—where outputs exhibit semantically meaningful and contextually consistent features—poses a challenge. To address this, we introduce a semantic infusion mechanism that integrates a pre-trained visual foundation model into the framework, leveraging its rich feature space to enrich the AR model’s latent representations with general geovisual knowledge, thereby enhancing the perceptual realism and utility of the generated satellite-map pairs for BSMT tasks. 

Let $\mathcal { G } ( \cdot )$ denote the Transformer-based AR model from the previous subsection, which predicts token pairs ${ \pmb r } _ { k } = ( s _ { k } , m _ { k } )$ given prior tokens $r _ { < k }$ and the geo-control signal $c _ { g } .$ . The pre-trained vision model’s encoder $\mathcal { E } _ { \mathrm { s e m } } ( \cdot )$ extracts semantic features from the input satellite imagery S and maps M . Then we introduce a lightweight alignment network A to bridge the dimensionality gap between the AR model’s embedding space and the feature space of a pre-trained visual foundation model: 

$$
\mathcal {A} \left(\mathcal {G} \left(\boldsymbol {r} _ {<   k}, c _ {g}\right)\right) \in \mathbb {R} ^ {h _ {k} \times w _ {k} \times d _ {\mathrm{sem}}}, \tag {12}
$$

where $h _ { k } \times w _ { k }$ corresponds to the spatial resolution at scale $k ,$ $d _ { \mathrm { s e m } }$ is the dimensionality of the semantic space. This aligned representation is then constrained to reflect the semantic context provided by $\mathcal { E } _ { \mathrm { s e m } } ( \cdot )$ . 

To guide the AR model toward geovisual coherence, we introduce a semantic infusion loss $\mathcal { L } _ { \mathrm { s e m } }$ that minimizes the discrepancy between the aligned AR embeddings and the pretrained foundation model’s semantic features. For a given scale t, we define: 

$$
\mathcal {L} _ {\mathrm{sem}} = \frac {1}{K} \sum_ {k = 1} ^ {K} \| \mathcal {A} (\mathcal {F} (\boldsymbol {r} _ {<   k}, c _ {g})) - \mathcal {E} _ {\mathrm{sem}} (\boldsymbol {r} _ {k}) \| _ {2} ^ {2} \tag {13}
$$

where $\mathcal { E } _ { \mathrm { s e m } } \left( \boldsymbol { r } _ { k } \right)$ approximates the semantic features of the token pair $\mathbf { \nabla } r _ { k }$ by mapping the decoded representations $\Phi _ { S } ^ { - 1 } \left( s _ { k } \right)$ and $\Phi _ { M } ^ { - 1 } \left( m _ { k } \right)$ —reconstructed from the HRQ tokenizer—back into the semantic space. This loss ensures that the AR predictions align with high-level geospatial semantics without overriding the joint modeling objective. 

The overall training objective combines the joint modeling loss $\mathcal { L } _ { \mathrm { j o t } }$ from the geo-conditioned AR with the semantic infusion loss $\mathcal { L } _ { \mathrm { s e m } } .$ weighted by a hyperparameter σ: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/431c5ab3bee053f8e0e4b43555b9412e41a53a3d6b1dc0c69036df740f1f3174.jpg)



Fig. 4. Schematic diagram of the inference section, with keypoint adaptive forcing on the left and complexity bootstrapping on the right, both of which are connected by the computation of complexity and together comprise the inference.


$$
\mathcal {L} = \mathcal {L} _ {\mathrm{jot}} + \sigma \cdot \mathcal {L} _ {\mathrm{sem}}, \tag {14}
$$

During training, the foundation model $\mathcal { E } _ { \mathrm { s e m } }$ remains frozen, while the parameters of G and A are optimized jointly. 

# F. Key Point Force

During the training phase of EarthMapper, the model is designed to learn to generate corresponding satellite image and map pairs [s, m] based on input geographic coordinates $g .$ In the inference phase, the task of generating satellite images from maps is taken as an example: the map acts as the conditional image, while the satellite image serves as the generation target (the reverse task works on a similar principle). Specifically, In the first autoregressive step, this geographic coordinate vector $c _ { g }$ is employed to predict [s, m]. However, geographic coordinates g alone provide only coarsegrained information and lack the precision needed to capture fine details within the image. To enhance the precision of the control, the map portion of the generated paired image is substituted with a vector-quantized real map: 

$$
Q _ {\mathbf {m}} = \mathrm{HRQ} ((\boldsymbol {E} (\mathbf {m})) \tag {15}
$$

where $Q _ { \mathbf { m } }$ denotes the map tokens after vector quantization, $E ( \mathbf { m } ) \in \mathbb { R } ^ { h \times w \times d }$ , where E denotes the visual encoder, $h \times w$ denotes the spatial dimensions and d represents the feature dimensionality. thereby imposing additional constraints on the model’s generation output. 

The architectural design of EarthMapper precludes direct information exchange between the generated paired images at each step, thus rendering the effectiveness of the conditional control contingent on the optimisation achieved during training. To address this limitation, we propose the key point force method, as shown on the left side of Fig. 4. Let C denote the conditional image and G the generated image. After performing vector quantization on $C ,$ we obtain the quantized indices $Q _ { C } = \{ q _ { C , 1 } , q _ { C , 2 } , . . . , q _ { C , N } \}$ , where each $q _ { C , i } \in \{ 0 , 1 , \ldots , K - 1 \}$ and K is the number of codebook vectors. We normalize these indices to the interval [0, 1] as follows: 

$$
\hat {q} _ {C, i} = \frac {q _ {C , i}}{K - 1}, \quad \forall i = 1, 2, \dots , N \tag {16}
$$

A threshold τ is established, and the set of key points is defined as: 

$$
\mathcal {K} = \{i \mid \hat {q} _ {C, i} > \tau \} \tag {17}
$$

During the autoregressive generation of G, for each position $i = 1$ to N, we sample the index $q _ { G , i }$ from the conditional distribution: 

$$
q _ {G, i} \sim p (q _ {G, i} | q _ {G, 1} ^ {\text { final }}, q _ {G, 2} ^ {\text { final }}, \dots , q _ {G, i - 1} ^ {\text { final }}, C) \tag {18}
$$

Then, the final index for position i is set as: 

$$
q _ {G, i} ^ {\text { final }} = \left\{ \begin{array}{l l} q _ {G, i} & \text { if   } i \notin \mathcal {K} \\ \max (0, \min (q _ {G, i} + q _ {C, i}, K - 1)) & \text { if   } i \in \mathcal {K} \end{array} \right. \tag {19}
$$

This integrates the information from the key points into the generated image at each stage of the autoregressive process. 

This approach involves the use of a small set of key points to restrict the generation process, thereby ensuring that conditional information has a pronounced effect on the generated image. Additionally, the implementation of a key point selection mechanism, in conjunction with index addition at the vector quantization level, ensures the provision of sufficient input information, while concurrently avoiding the excessive control that could potentially suppress image diversity. This design is intended to preserve a certain degree of variability in the generated images, thereby aligning them more closely with the characteristics of real images than merely replicating the style of the conditional image. 

# G. Complexity Guidance

In the existing conditional image generation inference processes, such as diffusion models, a commonly used method is to balance the condition weights and the model’s autonomous generation weights through a classifier-free guidance (CFG) to enhance the quality of image generation. In the context of autoregressive image generation models, the CFG method can be formulated as follows: 

$$
\begin{array}{l} p (x _ {i} | x _ {<   i}, c) = p (x _ {i} | x _ {<   i}, \emptyset) \\ + s \cdot (p (x _ {i} | x _ {<   i}, c) - p (x _ {i} | x _ {<   i}, \emptyset)) \tag {20} \\ \end{array}
$$

where $p ( x _ { i } | \boldsymbol x _ { < i } )$ denotes the probability distribution of the current pixel $x _ { i }$ conditioned on the previously generated pixels $x _ { < i }$ . The unconditional distribution $p ( x _ { i } | \boldsymbol x _ { < i } , \emptyset )$ represents the model’s autonomous generation capability without any conditional guidance, while the conditional distribution $p ( x _ { i } | \boldsymbol x _ { < i } , c )$ represents the model’s generation capability guided by the condition c. The CFG method balances these two distributions using a guidance strength parameter s. 

However, for EarthMapper’s multi-resolution autoregressive image generation structure, a single fixed CFG guidance strength cannot fully exploit the potential of the autoregressive generation model. To address this, a novel conditional guidance method is proposed, termed complexity guidance (CG), as shown on the right side of Fig. 4, which introduces a dynamic guidance strength $s ( x _ { i } , \phi )$ during the sampling process. 

Let $r _ { i }$ denote the resolution level at generation step $i ,$ and $C ( x _ { i } )$ represent the complexity of the generated image $x _ { i }$ . We introduce two dynamic parameters: $\alpha ( r _ { i } )$ and $\beta ( C ( x _ { i } ) )$ , which are functions of the resolution level and image complexity, respectively. The dynamic guidance strength $s ( x _ { i } , \phi )$ can be formulated as: 

$$
s (x _ {i}, \phi) = \gamma \cdot \alpha (r _ {i}) \cdot \beta (C (x _ {i})) \tag {21}
$$

where $\gamma$ is a hyperparameter to control the overall strength of the complexity-guided CFG. 

In the initial phases of image generation, characterized by low resolution, the model employs rudimentary coordinate vectors to predict, thereby yielding images of minimal complexity. At this stage, it is necessary to reduce the condition guidance strength to enhance the diversity of the generation results and to prevent the model from prematurely falling into erroneous generation paths. This can be achieved by setting $\alpha ( r _ { i } )$ to a low value when $r _ { i }$ is small. As the generation process advances and the resolution increases, $\alpha ( r _ { i } )$ should progressively increase to ensure the accuracy and precision of the generated images. 

Concurrently, the autoregressive image generation process entails a dynamic adjustment from the addition of details to the removal of redundant details. This process can be quantified through image complexity $C ( x _ { i } )$ . In the initial phase of generation, complexity is minimal, necessitating only a modest amount of condition guidance to facilitate the extensive addition of details. As the generation process progresses, the complexity increases, reaching a peak as a result of the incorporation of a substantial number of details. At this stage, it becomes essential to augment the guidance strength by setting $\beta ( C ( x _ { i } ) )$ to a high value. In the subsequent phase, the removal of redundant details leads to a modest decline in complexity, necessitating a corresponding reduction in $\beta ( C ( x _ { i } ) )$ to avert the retention of invalid details resulting from excessive constraint. 

The complexity $C ( x _ { i } )$ can be quantified using various measures, such as the entropy of the image or the number of distinct features present. One possible formulation is: 

$$
C (x _ {i}) = - \sum_ {j} p (x _ {i, j}) \log p (x _ {i, j}) \tag {22}
$$

where $x _ { i , j }$ represents the j-th pixel or feature of the generated image $x _ { i }$ , and $p ( x _ { i , j } )$ is the probability distribution over the pixels or features. 

By incorporating the dynamic guidance strength $s ( x _ { i } , \phi )$ into the CFG formulation, we obtain the complexity-guided conditional function (CG-CF) for autoregressive image generation: 

$$
\begin{array}{l} p _ {\theta} (x _ {i} | x _ {<   i}, c) = p _ {\theta} (x _ {i} | x _ {<   i}) \\ + s (x _ {i}, \phi) \cdot \nabla_ {p _ {\theta} (x _ {i} | x _ {<   i})} \log p _ {\varphi} (c | x _ {i}) \tag {23} \\ \end{array}
$$

where $p _ { \theta } ( x _ { i } | \boldsymbol x _ { < i } , c )$ represents the conditional probability distribution, $p _ { \theta } ( x _ { i } | \boldsymbol x _ { < i } )$ represents the unconditional probability distribution, $s ( x _ { i } , \phi )$ is the dynamic guidance strength , and $\nabla _ { p _ { \theta } ( x _ { i } | x _ { < i } ) }$ log $p _ { \varphi }$ (c|xi) represents the gradient of the conditional probability $p _ { \varphi } ( c | x _ { i } )$ with respect to the unconditional probability $p _ { \theta } ( x _ { i } | \boldsymbol x _ { < i } )$ . 

This complexity guidance method enables the model to adaptively balance between its autonomous generation capability and the conditional guidance based on the resolution level and complexity of the current generation context, thereby improving the realism and precision of the generated images in EarthMapper’s bidirectional translation tasks between satellite images and maps. 

# V. EXPERIMENTS

# A. Dataset and Evaluation Metrics

We evaluate our approach on two datasets: the New York dataset [5] and our constructed CNSatMap dataset. 

The New York dataset consists of paired aerial photographs and corresponding maps obtained from Google Maps, primarily covering New York City and its surrounding areas. It includes 1,096 training pairs, 1,098 validation pairs, and 1,098 test pairs, each at a resolution of 600×600 pixels. The dataset is geographically partitioned along latitudinal lines with a buffer zone to eliminate spatial overlap, establishing a robust benchmark for BSMT tasks. 

The CNSatMap dataset comprises 302,132 georeferenced image pairs for cross-modal translation between satellite imagery and cartographic maps. It integrates high-resolution satellite images (0.6 m/pixel) from Google Earth and vectorbased maps from Tianditu, spanning 38 major Chinese cities. Images are tiled into 256×256 pixels in the EPSG:3857 projection and undergo rigorous quality filtering to exclude cloud-covered, oblique, or homogeneous regions. The dataset is divided into 185,751 training, 58,190 validation, and 58,191 test pairs, serving as a large-scale and high-quality benchmark for urban scene understanding and geospatial image translation. 

For the map-to-satellite translation task, an open-domain generation problem, we employ Frechet inception distance ´ (FID) [47], kernel inception distance (KID) [48], Precision, and Recall [49] to evaluate visual realism, semantic plausibility, and distribution consistency of the generated images. For the satellite-to-map translation task, a structured reconstruction problem, we utilize mean square error (MSE), peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) [50] to assess pixel-level accuracy, structural preservation, and perceptual similarity, respectively. 


TABLE I PERFORMANCE COMPARISON ON NEW YORK DATASET. OPTIMAL VALUES ARE HIGHLIGHTED IN BOLD RED AND SUB-OPTIMAL VALUES IN BOLD BLUE.


<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="4">Map-to-Satellite Translation</td><td colspan="4">Satellite-to-Map Translation</td></tr><tr><td>FID ↓</td><td>KID ↓</td><td>Precision ↑</td><td>Recall ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>RMSE ↓</td><td>LPIPS ↓</td></tr><tr><td rowspan="3">GANs</td><td>CycleGAN</td><td>104.96</td><td>4.93</td><td>0.0580</td><td>0.3439</td><td>0.6344</td><td>23.55</td><td>0.0554</td><td>0.4345</td></tr><tr><td>Pix2Pix</td><td>86.36</td><td>4.10</td><td>0.0334</td><td>0.2986</td><td>0.5892</td><td>20.61</td><td>0.1460</td><td>0.4697</td></tr><tr><td>StegoGAN</td><td>121.96</td><td>6.39</td><td>0.0928</td><td>0.2456</td><td>0.6422</td><td>24.70</td><td>0.0530</td><td>0.4385</td></tr><tr><td rowspan="3">LDMs</td><td>BBDM</td><td>196.51</td><td>14.79</td><td>0.0190</td><td>0.1087</td><td>0.6303</td><td>23.71</td><td>0.0692</td><td>0.3211</td></tr><tr><td>ControlNet</td><td>133.83</td><td>7.18</td><td>0.6017</td><td>0.0229</td><td>0.6043</td><td>21.12</td><td>0.1058</td><td>0.4395</td></tr><tr><td>UniControl</td><td>101.05</td><td>4.27</td><td>0.3193</td><td>0.1100</td><td>0.3842</td><td>11.49</td><td>0.3001</td><td>0.4242</td></tr><tr><td rowspan="3">ARs</td><td>CAR</td><td>80.37</td><td>5.20</td><td>0.3749</td><td>0.1691</td><td>0.5831</td><td>23.57</td><td>0.0688</td><td>0.3602</td></tr><tr><td>ControlVAR</td><td>58.23</td><td>2.37</td><td>0.5008</td><td>0.2151</td><td>0.6465</td><td>24.40</td><td>0.0647</td><td>0.3156</td></tr><tr><td>EarthMapper</td><td>36.54</td><td>0.99</td><td>0.6182</td><td>0.4890</td><td>0.6534</td><td>25.04</td><td>0.0611</td><td>0.2819</td></tr></table>


Note: Arrows indicate desired direction of improvement (↓ lower is better, ↑ higher is better). 



TABLE II PERFORMANCE COMPARISON ON CNSATMAP DATASET. OPTIMAL VALUES ARE HIGHLIGHTED IN BOLD RED AND SUB-OPTIMAL VALUES IN BOLD BLUE.


<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="4">Map-to-Satellite Translation</td><td colspan="4">Satellite-to-Map Translation</td></tr><tr><td>FID ↓</td><td>KID ↓</td><td>Precision ↑</td><td>Recall ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>RMSE ↓</td><td>LPIPS ↓</td></tr><tr><td rowspan="3">GANs</td><td>CycleGAN</td><td>163.96</td><td>11.04</td><td>0.0946</td><td>0.3646</td><td>0.6521</td><td>22.05</td><td>0.0819</td><td>0.3716</td></tr><tr><td>Pix2Pix</td><td>150.85</td><td>12.50</td><td>0.0879</td><td>0.4334</td><td>0.6384</td><td>23.26</td><td>0.0701</td><td>0.3604</td></tr><tr><td>StegoGAN</td><td>73.55</td><td>4.89</td><td>0.1652</td><td>0.3722</td><td>0.6635</td><td>25.30</td><td>0.0577</td><td>0.3118</td></tr><tr><td rowspan="3">LDMs</td><td>BBDM</td><td>110.98</td><td>9.48</td><td>0.0584</td><td>0.4110</td><td>0.7184</td><td>26.36</td><td>0.0519</td><td>0.2954</td></tr><tr><td>ControlNet</td><td>87.08</td><td>6.79</td><td>0.4886</td><td>0.2110</td><td>0.5685</td><td>19.20</td><td>0.1265</td><td>0.3947</td></tr><tr><td>UniControl</td><td>235.70</td><td>20.42</td><td>0.0372</td><td>0.4284</td><td>0.4945</td><td>17.78</td><td>0.1951</td><td>0.5753</td></tr><tr><td rowspan="3">ARs</td><td>CAR</td><td>55.29</td><td>4.87</td><td>0.2216</td><td>0.4634</td><td>0.5584</td><td>22.15</td><td>0.0831</td><td>0.4631</td></tr><tr><td>ControlVAR</td><td>53.07</td><td>3.49</td><td>0.3134</td><td>0.3422</td><td>0.7114</td><td>25.29</td><td>0.0599</td><td>0.3542</td></tr><tr><td>EarthMapper</td><td>29.89</td><td>2.06</td><td>0.4294</td><td>0.3954</td><td>0.7300</td><td>26.88</td><td>0.0510</td><td>0.3103</td></tr></table>


Note: Arrows indicate desired direction of improvement (↓ lower is better, ↑ higher is better). 


# B. Experimental Setup

All images are resized to a resolution of 256×256 during both training and inference. We adopt a GPT-2-style transformer architecture with a depth of 24 layers, initialized with pretrained weights from the VAR model to accelerate convergence. For semantic feature fusion, the pretrained DINOv2 [51] model is incorporated as the visual backbone. The balancing hyperparameter σ is set to 0.5. Training is conducted on eight NVIDIA A800 GPUs, each with 80 GB of memory, using the AdamW [52] optimizer. The model is trained for 100 epochs with a batch size of 24, ensuring a trade-off between computational efficiency and gradient stability. During inference, we employ top-k and top-p sampling strategies with k=100 and p=0.55 to enhance the quality and diversity of the generated outputs. 

# C. Comparison with State-of-the-art Methods

To assess the robustness and generalizability of the proposed EarthMapper framework, we conducted a comparative evaluation on both the New York and CNSatMap datasets, with results presented in Tables I and II, respectively. These experiments encompass BSMT tasks, benchmarking EarthMapper against representative methods from GANs (CycleGAN [6], Pix2Pix [5], and StegoGAN [53]), LDMs (BBDM [54], ControlNet [14], UniControl [55]), and ARs (CAR [13], ControlVAR [12]). Performance is quantified using a suite of metrics—FID, KID, Precision, Recall, SSIM, PSNR, RMSE, and LPIPS—capturing generation quality, diversity, and structural fidelity. The optimal and suboptimal performances are distinctly highlighted in red and blue, respectively. 

In the map-to-satellite translation task, EarthMapper achieves an FID of 36.54 on New York and 29.89 on CNSatMap, significantly outperforming GAN-based methods like CycleGAN (104.96 and 163.96) and Pix2Pix (86.36 and 150.85). This substantial gap reflects EarthMapper’s ability to align the feature distributions of generated and real images closely, minimizing the Frechet distance and enhancing visual ´ realism—a critical advantage over GANs, which exhibit higher distributional mismatch. Similarly, its KID of 0.99 (New York) and 2.06 (CNSatMap) surpasses ControlVAR’s 2.37 and 3.49, indicating finer semantic consistency through reduced kernel-based discrepancies, unlike LDMs such as BBDM (14.79 and 9.48), which struggle with coarser feature alignment. 

EarthMapper’s Precision of 0.6182 on New York and 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/3f61b3cd7167a86e1ceb66279c7689e805bfef363f5dd1c9854d087fa15a5884.jpg)



Fig. 5. Qualitative comparison of bidirectional satellite-map translation results on the New York test set. The top five rows illustrate map-to-satellite translation, and the bottom five rows depict satellite-to-map translation.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/db1b80f84a4aeecc7b431207d79e96cf3d79525a42cb55e99f6e3810d99651cd.jpg)



Fig. 6. Qualitative comparison of bidirectional satellite-map translation results on the CNSatMap test set. The top five rows illustrate map-to-satellite translation, and the bottom five rows depict satellite-to-map translation.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/27a9db8d15c20ef9a1785db217b9f3808fd3927fb3694bbe64f090e05f8157b1.jpg)



Fig. 7. Illustration of generative diversity in EarthMapper. Given a single conditional input (left), our method produces multiple distinct outputs (right), showcasing varied yet semantically consistent satellite and map representations.


0.4294 on CNSatMap demonstrates high fidelity, with generated samples effectively residing within the real image manifold. Although ControlNet achieves a higher Precision of 0.4886 on CNSatMap, EarthMapper’s Recall of 0.4890 (New York) and 0.3954 (CNSatMap) exceeds ControlNet’s 0.2110, showcasing superior diversity and coverage of the real distribution. Compared to GANs like StegoGAN (Recall: 0.2456 on New York, 0.3722 on CNSatMap), EarthMapper balances quality and diversity more effectively, avoiding the overfitting tendencies seen in CycleGAN’s inflated Recall (0.3439 and 0.3646). This equilibrium underscores EarthMapper’s autoregressive strength in modeling complex, open-domain dependencies, delivering realistic and varied satellite imagery. 

For the satellite-to-map translation task, EarthMapper records an SSIM of 0.6534 on New York and 0.7300 on CNSatMap, outperforming ControlVAR (0.6465 and 0.7114) and BBDM (0.6303 and 0.7184). This indicates superior preservation of luminance, contrast, and structural details, critical for map reconstruction, where EarthMapper minimizes covariance discrepancies more effectively than GANs like CycleGAN (0.6344 and 0.6521). Its PSNR of 25.04 (New York) and 26.88 (CNSatMap) exceeds StegoGAN’s 24.70 and BBDM’s 26.36, reflecting lower pixel-level error and higher signal fidelity, a testament to its precise reconstruction capabilities over LDMs like UniControl (11.49 on New York). 

EarthMapper’s RMSE of 0.0611 (New York) and 0.0510 (CNSatMap) edges out ControlVAR (0.0647 and 0.0599), achieving pixel accuracy comparable to StegoGAN’s 0.0530 (New York) and BBDM’s 0.0519 (CNSatMap). This narrow margin highlights its robustness in minimizing pixel discrepancies, crucial for structured outputs. In terms of LPIPS, EarthMapper’s 0.2819 (New York) and 0.3103 (CNSatMap) closely rival BBDM’s 0.2954 (CNSatMap), indicating strong perceptual similarity despite nuanced competition from LDMs. Unlike ControlNet (0.4395 on New York, 0.3947 on CNSatMap), EarthMapper maintains consistency across datasets, avoiding the perceptual degradation seen in GANs like Pix2Pix (0.4697 and 0.3604). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/320b337092a9333316f26239d322aa0865a6b78c48daf6a624c802ddcf7a8866.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/b5de0e761d99ad2a6419670ebd61e52886586e2dd5106e2f3e59ae5dbc57bc99.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/bffd7f502c5022cf5d9ed1c3d2ed3fb038d1d44e83cd0de1632d606acb253dba.jpg)



(c)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/2946bff9f8d2f40209a1309f588b1d2282b60d72aef0517ef482eb380cc89bbc.jpg)



(d)



Fig. 8. T-SNE visualization comparing different methods on the New York test set. The blue points represent the feature distribution of the ground truth satellite images, while the red points indicate the feature distribution of the generated satellite images. (a) BBDM. (b) Pix2Pix. (c) CAR. (d) EarthMapper.


The visualization results in Fig.5 and Fig.6 also showcase EarthMapper’s ability to produce realistic satellite textures and precise map structures, highlighting its exceptional quality and fidelity in geospatial translation tasks. Besides, our EarthMapper demonstrates robust diversity in generated images, producing varied yet semantically consistent outputs from identical conditional inputs, as illustrated in Fig. 7. 

Complementing these findings, the t-SNE visualization in Fig. 8 further demonstrates EarthMapper’s superior feature alignment. Generated samples (red) form a tight, cohesive cluster closely aligned with real satellite imagery (blue). In comparison, BBDM exhibits significant dispersion, Pix2Pix shows partial overlap with fragmented clustering, and CAR, while improved, lacks the same level of coherence. This visualization emphasizes EarthMapper’s capability to accurately capture the real data manifold, ensuring enhanced fidelity and structural consistency, setting it apart from other methods. 

# D. Ablation Study

1) Effectiveness of EarthMapper’s Components: The ablation study, detailed in Table III, systematically dissects the contributions of EarthMapper’s components by building upon a baseline defined as the VAR with joint modeling. Introducing geo-conditioned joint scale autoregression enhances the baseline by embedding geospatial priors into the autoregressive process, enabling the model to explicitly account for spatial relationships and scale variations inherent in remote sensing data. This conditioning aligns token predictions with geographic structures, substantially improving feature distribution coherence in map-to-satellite translation and structural preservation in satellite-to-map translation tasks, as evidenced by the marked enhancement in visual realism and reconstruction fidelity. 


TABLE III ABLATION STUDY OF KEY COMPONENTS IN EARTHMAPPER ON THE CNSATMAP TEST SET.


<table><tr><td rowspan="2">ID</td><td rowspan="2">Method</td><td colspan="2">Map-to-Satellite</td><td colspan="2">Satellite-to-Map</td></tr><tr><td>FID↓</td><td>KID↓</td><td>SSIM↑</td><td>PSNR↑</td></tr><tr><td>1</td><td>Baseline</td><td>64.28</td><td>5.64</td><td>0.5329</td><td>20.87</td></tr><tr><td>2</td><td>+ Geo-Conditioned Joint Scale Autoregression</td><td>47.33 (-16.95)</td><td>4.51 (-1.13)</td><td>0.6381 (+0.1052)</td><td>23.46 (+2.59)</td></tr><tr><td>3</td><td>+ Semantic Infusion</td><td>34.69 (-12.64)</td><td>3.92 (-0.59)</td><td>0.6629 (+0.0248)</td><td>25.39 (+1.93)</td></tr><tr><td>4</td><td>+ Key Point Force</td><td>31.57(-3.12)</td><td>2.85 (-1.07)</td><td>0.7054 (+0.0425)</td><td>26.05 (+0.66)</td></tr><tr><td>5</td><td>+ Complexity Guidance</td><td>29.89 (-1.68)</td><td>2.06 (-0.79)</td><td>0.7300 (+0.0246)</td><td>26.88 (+0.83)</td></tr></table>


TABLE IV ABLATION ANALYSIS OF GUIDANCE SCALES FOR MAP-TO-SATELLITE TRANSLATION ON THE NEW YORK DATASET. THE BEST PERFORMANCE IS SHOWN IN BOLD.


<table><tr><td>Guidance Scale</td><td>FID ↓</td><td>KID ↓</td><td>Precision ↑</td><td>Recall ↑</td></tr><tr><td>[2, 2, 2]</td><td>39.60</td><td>1.24</td><td>0.4279</td><td>0.4907</td></tr><tr><td>[4, 4, 4]</td><td>39.92</td><td>1.20</td><td>0.4589</td><td>0.5099</td></tr><tr><td>[6, 6, 6]</td><td>37.86</td><td>1.16</td><td>0.5620</td><td>0.4775</td></tr><tr><td>[8, 8, 8]</td><td>36.54</td><td>0.99</td><td>0.6182</td><td>0.4890</td></tr><tr><td>[10, 10, 10]</td><td>39.26</td><td>1.08</td><td>0.5748</td><td>0.4538</td></tr><tr><td>[12, 12, 12]</td><td>42.39</td><td>1.35</td><td>0.5589</td><td>0.4215</td></tr></table>


TABLE V ABLATION ANALYSIS OF GUIDANCE SCALES FOR SATELLITE-TO-MAP TRANSLATION ON THE NEW YORK DATASET. THE BEST PERFORMANCE IS SHOWN IN BOLD.


<table><tr><td>Guidance Scale</td><td>SSIM ↑</td><td>PSNR ↑</td><td>RMSE ↓</td><td>LPIPS ↓</td></tr><tr><td>[1, 1, 1]</td><td>0.6363</td><td>24.48</td><td>0.0635</td><td>0.2969</td></tr><tr><td>[2, 2, 2]</td><td>0.6534</td><td>25.04</td><td>0.0611</td><td>0.2819</td></tr><tr><td>[4, 4, 4]</td><td>0.6477</td><td>24.46</td><td>0.0638</td><td>0.2960</td></tr><tr><td>[6, 6, 6]</td><td>0.6374</td><td>24.26</td><td>0.0639</td><td>0.2961</td></tr><tr><td>[8, 8, 8]</td><td>0.6263</td><td>24.17</td><td>0.0640</td><td>0.2978</td></tr><tr><td>[10, 10, 10]</td><td>0.6140</td><td>24.04</td><td>0.0644</td><td>0.2966</td></tr></table>

Further advancements are achieved through semantic infusion, key point force, and complexity guidance, each addressing distinct limitations of the baseline’s generic autoregressive approach. Semantic infusion integrates pretrained visual embeddings to enrich the model’s understanding of high-level semantics, fostering precise cross-modal alignment by bridging linguistic and visual representations. Key point force dynamically emphasizes critical spatial keypoints, adapting feature weighting to prioritize structurally significant regions, thus refining the model’s ability to reconstruct complex geospatial patterns. Finally, complexity guidance introduces a dynamic scaling mechanism that modulates conditional guidance based on image complexity, balancing generative diversity with reconstruction accuracy. This adaptive control mitigates overfitting to simplistic patterns, ensuring robust performance across diverse urban scenes. Together, these components transform the baseline’s generic framework into a specialized architecture adept at bidirectional geospatial translation, with each module addressing principled deficiencies to achieve superior alignment and fidelity. 

2) The Impact of Different Guidance Scales: To elucidate the role of the guidance scale in EarthMapper’s complexity guidance mechanism, we conducted an ablation study on the New York test set, evaluating its influence on BSMT task. The results, presented in Tables IV and Tables V, systematically explore a range of guidance scales. For map-to-satellite translation, a guidance scale of [8, 8, 8] achieves optimal performance, yielding an FID of 36.54, KID of 0.99, Precision of 0.6182, and Recall of 0.4890. This configuration excels in balancing generative fidelity and diversity, as evidenced by the minimal distributional discrepancies (FID and KID) and high fidelity within the real image manifold (Precision). Lower scales, such as [2, 2, 2], result in higher FID (39.60) and KID (1.24), indicating insufficient conditional control, while higher scales, like [12, 12, 12], degrade performance (FID: 42.39, KID: 1.35), suggesting overfitting to conditional constraints that stifles generative diversity. 

In contrast, for satellite-to-map translation, a guidance scale of [2, 2, 2] delivers the best performance, with an SSIM of 0.6534, PSNR of 25.04, RMSE of 0.0611, and LPIPS of 0.2819, reflecting superior structural preservation and perceptual similarity. Higher scales, such as [4, 4, 4] and beyond, progressively degrade performance (e.g., SSIM: 0.6140 at [10, 10, 10]), indicating excessive guidance that distorts fine-grained map structures. The divergence in optimal scales between tasks stems from their inherent objectives: map-to-satellite translation, an open-domain generation task, benefits from stronger guidance ([8, 8, 8]) to align complex, variable satellite textures with the real data manifold, requiring robust conditional steering to ensure realism. Conversely, for satellite-to-map translation, a structured reconstruction task, demands precise pixel-level accuracy and structural fidelity, where lighter guidance ([2, 2, 2]) prevents over-constraining the model, preserving intricate cartographic details. 

# E. Zero-shot Generalization

1) Cross-dataset Generalization: To evaluate the robustness and generalizability of the proposed EarthMapper framework across diverse geographical contexts, we conducted a cross-dataset generalization experiment, with results presented in Table VI. The experiment assesses BSMT performance by training EarthMapper on one dataset and testing it on another, specifically from CNSatMap to New York and vice versa. For the map-to-satellite translation, EarthMapper trained on CNSatMap and tested on New York achieves an FID of 164.91 and KID of 17.68, significantly outperforming the reverse configuration (New 


In-painting


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/9839b7591092f70fffad82d86ff1c1ac5883ca9a51b98014b9b59bf1b9e101be.jpg)



Out-painting


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/78ac0dd1eba35be58ebdebaa6b195ffbfe851ae32d1c41ff01bcf4471e13e17b.jpg)



Fig. 9. Zero-shot performance of EarthMapper in in-painting and out-painting tasks.



TABLE VI PERFORMANCE EVALUATION OF CROSS-DATASET GENERALIZATION FOR BIDIRECTIONAL SATELLITE-MAP TRANSLATION


<table><tr><td rowspan="2">Cross-dataset Generalization</td><td colspan="2">Map-to-Satellite</td><td colspan="2">Satellite-to-Map</td></tr><tr><td>FID ↓</td><td>KID ↓</td><td>SSIM ↑</td><td>PSNR ↑</td></tr><tr><td>CNSatMap→New York</td><td>164.91</td><td>17.68</td><td>0.5226</td><td>22.50</td></tr><tr><td>New York→CNSatMap</td><td>236.83</td><td>20.84</td><td>0.2711</td><td>8.18</td></tr></table>

York→CNSatMap), which yields an FID of 236.83 and KID of 20.84. Similarly, in satellite-to-map translation, the CNSatMap→New York setup delivers superior results with an SSIM of 0.5226 and PSNR of 22.50, compared to 0.2711 and 8.18 for New York→CNSatMap. These results underscore the CNSatMap dataset’s exceptional capacity to foster robust generalization. The dataset’s extensive scale, encompassing 302,132 high-quality satellite-map pairs across 38 diverse Chinese cities, captures a broad spectrum of urban and topographical variations, enabling models to learn rich, transferable representations. In contrast, the New York dataset, while valuable, is limited in geographical diversity, leading to overfitting and poor generalization when applied to the more varied CNSatMap test set. 

2) In-painting and Out-painting: To rigorously evaluate EarthMapper’s generalization to unseen geospatial distributions, we assess its zero-shot performance on inpainting (reconstructing occluded regions) and out-painting (extending beyond image boundaries)—tasks requiring robust contextual reasoning and spatial coherence. As illustrated in Fig. 9, EarthMapper reconstructs occluded regions, such as obscured buildings and roads, with high fidelity, seamlessly integrating synthesized textures with existing structures. Notably, EarthMapper avoids common artifacts, including blurring and misalignment, even under severe occlusion, demonstrating its ability to generalize beyond training distributions. For out-painting, EarthMapper generates realistic urban layouts and natural topographies without geometric distortions. Crucially, this performance is achieved without fine-tuning, highlighting EarthMapper’s inherent adaptability to diverse geospatial contexts. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-08/40c909b1-ee1b-40fe-a82a-af967ac28dde/6a163ca310277b1473858f0b643acb706ae9110ab41a3080ab5001ccf592d5c9.jpg)



Fig. 10. Zero-shot performance of EarthMapper in coordinate-conditional image generation task.


3) Coordinate-conditional Image Generation: To further evaluate the generalization capabilities of EarthMapper, we conduct a zero-shot coordinate-conditional image generation experiment. This task assesses the model’s ability to infer complex geospatial patterns without additional contextual inputs. As illustrated in Fig. 10, EarthMapper generates high-fidelity satellite-map image pairs that accurately capture the spatial and semantic characteristics of the provided coordinates, including urban layouts and natural features. The generated images demonstrate strong consistency between satellite and map representations, exhibiting coherent textures and structures. These results underscore EarthMapper’s robust generalization across diverse geospatial contexts, achieved without fine-tuning or auxiliary data. 

# VI. CONCLUSION

In this paper, we present EarthMapper, an innovative autoregressive framework that seamlessly integrates geographic coordinate embeddings with multi-scale feature alignment to achieve high-fidelity bidirectional satellitemap translation. This approach employs geo-conditioned joint scale autoregression (GJSA), enhanced by semantic infusion (SI) for feature consistency and key point adaptive guidance (KPAG) for balanced diversity and precision. We also construct CNSatMap, a dataset of 302,132 aligned satellite-map pairs across 38 Chinese cities, providing a robust benchmark for cross-modal research. Evaluations on CNSatMap and New York datasets show superior performance over state-of-the-art methods, with enhanced visual realism, semantic consistency, and structural fidelity. The framework’s versatility in zero-shot tasks, including in-painting, outpainting, and coordinate-conditional generation, highlights its strong generalization across diverse geospatial contexts. 

The significance of this work lies in its ability to bridge the visual-semantic gap between raw satellite data and human-interpretable maps, offering a scalable solution that balances precision and diversity. While EarthMapper sets a new standard in bidirectional translation, several avenues remain for future exploration. Integrating additional modalities, such as LiDAR or hyperspectral data, could further enrich translation quality by capturing complementary geospatial attributes. Extending EarthMapper to tasks like change detection or land cover classification may broaden its applicability in remote sensing. Moreover, optimizing its computational efficiency could enable real-time deployment, enhancing its practical utility. Collectively, these directions promise to amplify EarthMapper’s impact, paving the way for next-generation geospatial intelligence. 

# REFERENCES



[1] V. Ingale, R. Singh, and P. Patwal, “Image to image translation: Generating maps from satellite images,” arXiv preprint arXiv:2105.09253, 2021. 





[2] M. Espinosa and E. J. Crowley, “Generate your own scotland: Satellite image generation conditioned on maps,” arXiv preprint arXiv:2308.16648, 2023. 





[3] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” Communications of the ACM, vol. 63, no. 11, pp. 139–144, 2020. 





[4] M. Mirza and S. Osindero, “Conditional generative adversarial nets,” arXiv preprint arXiv:1411.1784, 2014. 





[5] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1125– 1134. 





[6] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2223–2232. 





[7] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020. 





[8] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021. 





[9] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever, “Generative pretraining from pixels,” in International conference on machine learning. PMLR, 2020, pp. 1691–1703. 





[10] A. Van den Oord, N. Kalchbrenner, L. Espeholt, O. Vinyals, A. Graves et al., “Conditional image generation with pixelcnn decoders,” Advances in neural information processing systems, vol. 29, 2016. 





[11] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang, “Visual autoregressive modeling: Scalable image generation via next-scale prediction,” Advances in neural information processing systems, vol. 37, pp. 84 839–84 865, 2024. 





[12] X. Li, K. Qiu, H. Chen, J. Kuen, Z. Lin, R. Singh, and B. Raj, “Controlvar: Exploring controllable visual autoregressive modeling,” arXiv preprint arXiv:2406.09750, 2024. 





[13] Z. Yao, J. Li, Y. Zhou, Y. Liu, X. Jiang, C. Wang, F. Zheng, Y. Zou, and L. Li, “Car: Controllable autoregressive modeling for visual generation,” arXiv preprint arXiv:2410.04671, 2024. 





[14] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 3836–3847. 





[15] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695. 





[16] Y. Li, H. Liu, Q. Wu, F. Mu, J. Yang, J. Gao, C. Li, and Y. J. Lee, “Gligen: Open-set grounded text-to-image generation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 22 511–22 521. 





[17] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4401– 4410. 





[18] Y. Shen, C. Yang, X. Tang, and B. Zhou, “Interfacegan: Interpreting the disentangled face representation learned by gans,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 4, pp. 2004–2018, 2020. 





[19] Y. Takagi and S. Nishimoto, “High-resolution image reconstruction with latent diffusion models from human brain activity,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 453–14 463. 





[20] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or, “An image is worth one word: Personalizing text-to-image generation using textual inversion,” arXiv preprint arXiv:2208.01618, 2022. 





[21] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 22 500–22 510. 





[22] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” arXiv preprint arXiv:2112.10741, 2021. 





[23] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, vol. 1, no. 2, p. 3, 2022. 





[24] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763. 





[25] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022. 





[26] Z. Tang, Z. Yang, C. Zhu, M. Zeng, and M. Bansal, “Any-to-any generation via composable diffusion,” Advances in Neural Information Processing Systems, vol. 36, pp. 16 083–16 099, 2023. 





[27] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019. 





[28] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020. 





[29] N. Parmar, A. Vaswani, J. Uszkoreit, L. Kaiser, N. Shazeer, A. Ku, and D. Tran, “Image transformer,” in International conference on machine learning. PMLR, 2018, pp. 4055–4064. 





[30] K. Li, Y. Zhu, J. Yang, and J. Jiang, “Video super-resolution using an adaptive superpixel-guided auto-regressive model,” Pattern Recognition, vol. 51, pp. 59–71, 2016. 





[31] B. Guo, X. Zhang, H. Wu, Y. Wang, Y. Zhang, and Y.-F. Wang, “Lar-sr: A local autoregressive model for image super-resolution,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 1909–1918. 





[32] K. Yao, P. Gao, X. Yang, J. Sun, R. Zhang, and K. Huang, “Outpainting by queries,” in European conference on computer vision. Springer, 2022, pp. 153–169. 





[33] K. Crowson, S. Biderman, D. Kornis, D. Stander, E. Hallahan, L. Castricato, and E. Raff, “Vqgan-clip: Open domain image generation and editing with natural language guidance,” in European conference on computer vision. Springer, 2022, pp. 88–105. 





[34] Z. Li, T. Cheng, S. Chen, P. Sun, H. Shen, L. Ran, X. Chen, W. Liu, and X. Wang, “Controlar: Controllable image generation with autoregressive models,” arXiv preprint arXiv:2410.02705, 2024. 





[35] A. Van Den Oord, N. Kalchbrenner, and K. Kavukcuoglu, “Pixel recurrent neural networks,” in International conference on machine learning. PMLR, 2016, pp. 1747–1756. 





[36] S. Reed, A. Oord, N. Kalchbrenner, S. G. Colmenarejo, Z. Wang, Y. Chen, D. Belov, and N. Freitas, “Parallel multiscale autoregressive density estimation,” in International conference on machine learning. PMLR, 2017, pp. 2912–2921. 





[37] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017. 





[38] A. Razavi, A. Van den Oord, and O. Vinyals, “Generating diverse high-fidelity images with vq-vae-2,” Advances in neural information processing systems, vol. 32, 2019. 





[39] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for highresolution image synthesis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12 873–12 883. 





[40] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017. 





[41] D. Lee, C. Kim, S. Kim, M. Cho, and W.-S. Han, “Autoregressive image generation using residual quantization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11 523–11 532. 





[42] L. Pang, X. Cao, D. Tang, S. Xu, X. Bai, F. Zhou, and D. Meng, “Hsigene: A foundation model for hyperspectral image generation,” arXiv preprint arXiv:2409.12470, 2024. 





[43] A. Sebaq and M. ElHelw, “Rsdiff: Remote sensing image generation from text using diffusion model,” Neural Computing and Applications, vol. 36, no. 36, pp. 23 103–23 111, 2024. 





[44] D. Tang, X. Cao, X. Hou, Z. Jiang, and D. Meng, “Crs-diff: Controllable generative remote sensing foundation model,” arXiv e-prints, pp. arXiv– 2403, 2024. 





[45] Z. Yu, C. Liu, L. Liu, Z. Shi, and Z. Zou, “Metaearth: A generative foundation model for global-scale remote sensing image generation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 





[46] S. Khanna, P. Liu, L. Zhou, C. Meng, R. Rombach, M. Burke, D. Lobell, and S. Ermon, “Diffusionsat: A generative foundation model for satellite imagery,” arXiv preprint arXiv:2312.03606, 2023. 





[47] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017. 





[48] M. Binkowski, D. J. Sutherland, M. Arbel, and A. Gretton, ´ “Demystifying mmd gans,” arXiv preprint arXiv:1801.01401, 2018. 





[49] T. Kynka¨anniemi, T. Karras, S. Laine, J. Lehtinen, and T. Aila, ¨ “Improved precision and recall metric for assessing generative models,” Advances in neural information processing systems, vol. 32, 2019. 





[50] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586–595. 





[51] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023. 





[52] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017. 





[53] S. Wu, Y. Chen, S. Mermet, L. Hurni, K. Schindler, N. Gonthier, and L. Landrieu, “Stegogan: Leveraging steganography for nonbijective image-to-image translation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7922–7931. 





[54] B. Li, K. Xue, B. Liu, and Y.-K. Lai, “Bbdm: Image-to-image translation with brownian bridge diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, 2023, pp. 1952–1961. 





[55] C. Qin, S. Zhang, N. Yu, Y. Feng, X. Yang, Y. Zhou, H. Wang, J. C. Niebles, C. Xiong, S. Savarese et al., “Unicontrol: A unified diffusion model for controllable visual generation in the wild,” arXiv preprint arXiv:2305.11147, 2023. 

