# MapGen-GAN: A Fast Translator for Remote Sensing Image to Map Via Unsupervised Adversarial Learning

Jieqiong Song , Jun Li, Hao Chen , and Jiangjiang Wu 

Abstract—Map is an essential medium for people to understand our changing planet. Recently, research on generating and updating maps through remote sensing images has been an important and challenging task in geographic information. Traditional methods for map generation are time-consuming and labor-intensive. Besides, most supervised learning methods for map generation lack labeled training samples. It is challenging to generate maps quickly and efficiently for emergency rescue operations such as earthquakes, fire disasters, or tsunami. In this article, we propose an unsupervised domain mapping model based on adversarial learning called MapGen-GAN. MapGen-GAN is a generative adversarial network (GAN) that can do end-to-end translation from remote sensing images to general map quickly, and trained with no human annotation data. In order to improve the fidelity and the geometry precision of generated maps, we employ circularityconsistency and geometrical-consistency constraints as a part of the loss function of the proposed model. And then, an improved residual block Unet is designed and adopted as the generator of MapGen-GAN to capture the geographic structure information of buildings, roads, and topography outlines under different resolutions in the map generation. By applying the proposed model to two distinct datasets, experiments demonstrate that our model can generate maps efficiently and quickly and outperform the state-of-the-art approaches. 

Index Terms—Adversarial learning, map generation, remote sensing images, unsupervised domain mapping. 

# I. INTRODUCTION

M AP is an essential medium for people to fully understandcultural activities and natural elements configuration of culturalactivitiesandnatural elementsconfigurationof different places. The generation and updating of maps from remote sensing images are vital in emergency rescue operations such as earthquakes, fire disasters, or tsunamis. Traditional map-making methods based on remote sensing images mainly include four steps: “Geographic feature extraction, updating of vector data, cartography, generation, and organization of maps” [1]. The first step of geographic feature extraction based on deep 

Manuscript received September 6, 2020; revised November 14, 2020; accepted December 25, 2020. Date of publication January 8, 2021; date of current version February 15, 2021. This work was supported in part by the Natural Science Foundation of Hunan Province under Grant 2020JJ4103, in part by the National NSF of China under Grant 41871284, Grant 61806211, Grant 4971362 and Grant U19A2058. (Corresponding author: Hao Chen.) 

The authors are with the College of Electronic Science and Technology, National University of Defense Technology, Changsha 430070, China (e-mail: sjq@nudt.edu.cn; junli@nudt.edu.cn; hchen@nudt.edu.cn; wujiangjiang08@nudt.edu.cn). 

Digital Object Identifier 10.1109/JSTARS.2021.3049905 

learning has become very popular in recent years. Several geographic information extraction methods are proposed. For example, the authors in [2] present new deep learning (DL) approach to support automatic detection of terrain features from remotely sensed images; the authors in[3] propose a novel end-to-end deep learning model for road extraction from remote sensing images. Although several steps are automated in producing traditional maps, the whole map-making procedure is still time-consuming and labor-intensive. After devastating natural forces attack the human-made infrastructure, existing maps often become useless. Since the defects of traditional map-making methods render nonreal time in disaster response scenarios, it is challenging to generate maps quickly and efficiently. Hence, we are trying to use DL methods to transform remote sensing images to maps end-to-end and make the process rapidly. 

Due to the advent of web-based service technologies, several online platforms such as Google Maps [4], Map World [5], and Baidu Maps [6] provide convenient access to maps for ordinary people. Different map service platforms have different styles. For example, Fig. 1 shows two styles of maps used in this article. Fig. 1(a) is a remote sensing image of New York city. Fig. 1(b) represents the corresponding Google Map [7] by artificial drawing. Fig. 1(c) and (d) are examples of Washington DC paired images from Map World [8]. 

Recent years have witnessed the prosperity of DL. DL methods can promote the performance of object extraction [9], [10] and classification [11], [12] from remote sensing images. However, most DL-based object extraction approaches need a large amount of high-quality annotated datasets like the research works in [13] and [14]. Annotation by humans is usually laborintensive and expensive. Most remote sensing images we obtain under the disaster response scenarios lack accurate labels and available paired maps, making it impractical to use DL-based object extraction methods for map generation in urgent scenarios. Hence, we adopt unsupervised domain mapping methods to generate maps using unpaired training samples and high-resolution remote sensing images without adding any labels annotated by humans. 

In the field of unsupervised domain mapping, we define two sample spaces,  and  as remote sensing images do-X Ymain and target maps domain, respectively. Several recent approaches [15]–[17] exploit cycle-consistency constraints, making a mapping $G _ { X Y } : X  Y$ and its inverse mapping $G _ { Y X }$ $Y  X$ GXY : X Ybe bijections. The goal requires $G _ { Y X } ( G _ { X Y } ( x ) ) \approx x .$ :, and $G _ { X Y } ( G _ { Y X } ( y ) ) \approx y$ . Distance constraints [18] make the GXY (GY X(y)) ymapping capable of distance preserving. Nevertheless, either circularity or distance consistency ignores the special properties that the simple geometrical transformation will not change the image’s semantic structure. Only using circularity or distance constraints cannot meet map-generation needs because the geometric distortion requirements are more stringent in the map translation task. The geometrical consistency assumption [19] focuses on the simple geometric transformation merely, which overlooks the absence of paired data and bijective translation to ensure consistency. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/e89ceda433dfe973480d9230f0795ac8ddaf5e46dd9d0c1be89f04eced833165.jpg)



Fig. 1. Samples of Google Map and Map World are used in this article. (a) is the rRemote sensing image of New York City, (b) is the corresponding man-made Google Map. (c) is the remote sensing image of Washington DC, and (d) is the corresponding man-made Map World.


Therefore, we seek a more restrictive bijection model based on GAN, aiming to learn the translation from remote sensing images to a certain type of map. The proposed model is called MapGen-GAN. To solve the absence of paired data and semantic distortion, we integrate circularity-consistency and geometricalconsistency constraints into MapGen-GAN. The cycle system helps the model use unpaired images, which alleviates the data collecting pressure. Adding geometrical-consistency constraints based on the predefined geometrical transformation function can improve geometry precision and reduce semantic distortion of generated maps. 

Map also belongs to one kind of image. It is more intuitive to construct the generator in the GAN with convolutional neural networks (CNN) [20]. For the map generator, in order to ensure that the map does not deform, it is necessary to translate at different resolutions. Unet [21] has different receptive fields, which is a reasonable choice. However, due to the low accuracy of the original Unet and detailed requirements for map translation, a novel map generator named basic residual block Unet (BRB-Unet) is proposed in our MapGen-GAN. BRB-Unet is an improved Unet to capture a wealth of geometrical structure information about buildings, roads, and terrain in the map translation task. We design BRB-Unet with several densely connected residual network blocks to increase the network’s depth and improve the accuracy of regional identification. In this article, we employ two available training datasets for map generation, one is New York City datasets from Google Map, and another is Washington DC datasets from Map World. Quantitative scores and qualitative comparisons of the above two datasets are considered distinct. Experiments demonstrate that MapGen-GAN produces more impressive and competitive translation than the other state-of-the-art methods. 

The technical contributions of this article are as follows. 

1) We propose a novel unsupervised domain mapping framework named MapGen-GAN to transform remote sensing images into maps. By employing circularity-consistency and geometrical-consistency constraints as a part of the loss function, MapGen-GAN considers both challenges for the absence of paired samples and semantic distortions in the translation of remote sensing images to maps. 

2) To further improve the proposed model’s performance, we devise a novel generator called BRB-Unet for MapGen-GAN. BRB-Unet consists of several densely connected residual blocks to increase network depth and provide an impressive enhancement to the low accuracy of regional identification suffered by standard Unet for map translation tasks. 

3) We conduct several experiments on two different datasets to validate the usability and effectiveness of the proposed method. Experiment results show that the proposed approach outperforms the state-of-the-art models on two distinct datasets. 

We organize the rest of the article as follows. In Section II, related works of map generation technique and evolution of GAN are introduced. The proposed map translation framework MapGen-GAN is detailed in Section III. Then we illustrate experiments and discussions in Section IV to demonstrate the effectiveness of our method. Finally, Section V concludes the article. 

# II. RELATED WORK

# A. Domain Mapping

Domain mapping in DL aims to learn the function from the source image domain to the target domain. We can use domain mapping methods to map a sample of remote sensing images domain to a similarity in a certain type of map domain. The most recent adversarial domain mapping has been studied in a supervised or unsupervised manner. For supervised domain mapping, model training requires paired training data. The pix2pix [7] method applies a conditional GAN to model the mapping function using CNNs for paired image-to-image translation. Similar ideas also have been adopted to many other tasks, such as generating photos from sketches [22]. Moreover, pix2pixH [23] is used for high-resolution photo-realistic imageto-image conversion. It can be used to synthesize portraits from face labeled images. 

However, most tasks in the real world lack available pairs of samples, due to the expensive data preparation. To overcome this limitation, the unpaired image-to-image translation task has been proposed in the absence of training pairs. Training with unpaired data, only unaligned examples in individual domains are provided, which makes the task more practical but more complicated. Specifically, the authors in[24] introduce coupling GAN to learn cross-domain representation by implementing weight sharing constraints. CycleGAN [15], Disco-GAN [16], and DualGAN [17] learn the mappings between two image domains instead of the paired images. Apart from these schemes, many other GAN variants are proposed to tackle the cross-domain problem by specific constraints and assumptions. For example, the authors in[25] leverage some shared content functionality between input and output, even though they may have different styles. The authors in[26] propose a model for learning object deformation from two sets of unpaired images. Tang et al. [27] devise G2GAN, a robust and scalable method that allows unpaired image-to-image conversion to perform on multiple domains. Subsequently, to solve the problem of schema collapse, a novel algorithm that implements a bijection mapping between the potential and the target space is invented by BicycleGAN [28]. However, the transformation from remote sensing images to maps requires high geometry precision and accuracy. The exiting methods overlook either the absence of labeled samples or geometric distortion during map translation. 

# B. GAN-Based Image Generation Technology

The rapid growth of DL has greatly improved image processing efficiency, such as image classification [29], semantic segmentation [30]. GAN [31]–[33] has been widely used to improve the accuracy of image processing in the field of DL. According to the current literature, GAN applications in image processing are mainly about high-resolution image generation. Representative works include: Image editing [34], image-toimage translation [7], [35]–[37], text to image translation [38]– [40], and single-image super resolution [41], [42]. The key component supporting GANs is adversarial constraints, making it difficult to distinguish between the generated images and the real images. 

# C. Geographic Information Extraction of Remote Sensing Images Based on DL

Generating and updating maps from remote sensing images is critical in disaster response scenarios. DL methods can promote the performance of object extraction to generate maps from remote sensing images. However, most DL-based object extraction approaches need a large amount of high-quality annotated datasets. According to a large number of existing labeled datasets, object extraction determines the geographical features of roads, buildings, and water systems. The representative research works are as follows: Zhang et al. [43] proposed a multiconditional generation of adversative network reconstruction, aiming at the problems of fuzzy boundary and incomplete extraction results of existing image segmentation results. Marmanis et al. [44] improve semantic segmentation quality by combining semantically informed edge detection, thus making class boundaries explicit in the model. Xu et al. [45] used a global and local attention model based on DenseNet [46]. 

In fact, most remote sensing images we obtain under disaster response scenarios lack accurate labels. The DL object extraction methods described above are usually inaccurate, timeconsuming, and labor-intensive, which cannot meet the map needs in emergency rescue operations. Our unsupervised domain mapping method is based on adversarial learning, which not only considers the absence of paired samples but also generates maps more quickly and expediently under specific scenarios. 

# III. PROPOSED METHOD

In this section, we present the detailed structure of the proposed method. 

# A. Overall Architecture

Our goal is to learn a mapping function from remote sensing images domain to a certain styled maps domain . X YLet and be remote sensing images and maps domain X  Y with training examples $\{ x _ { i } \} _ { i = 1 } ^ { N }$ , where $x _ { i } \in X$ , and - $\{ y _ { j } \} _ { j = 1 } ^ { M }$ where $y _ { j } \in Y$ xi i xi X yj j. As illustrated in Fig. 2, our model includes yj Yfour mappings: - $G _ { X Y } : X  Y , G _ { Y X } : Y  X , G _ { \widetilde { X Y } } : \widetilde { X } $ ${ \widetilde { Y } } , G _ { \widetilde { Y X } } : { \widetilde { Y } } \to { \widetilde { X } }$ Y : X, where $G _ { \widetilde { X Y } } , ~ G _ { \widetilde { Y X } }$ Y X GXY : Xrepresents generator $G _ { g c _ { - } X Y }$ and $G _ { g c _ { - } Y X \cdot } ~ \widetilde { X }$ Xand $\widetilde { Y }$ Y Xare two domains obtained-Ggc XYby applying - $f ( \cdot )$ c Y X X Yon  and  , where $f ( \cdot )$ is a predefined f ( ) X Y f ( )geometrical transformation function for images. In addition, we introduce four adversarial discriminators $D _ { X } , D _ { Y } , D _ { \widetilde { X } }$ . and-$D _ { \widetilde { Y } }$ , where - $D _ { X }$ DX DYaims to distinguish between images $\{ x \}$ and DY DXtranslated images $\{ G _ { Y X } ( y ^ { \prime } ) \} , D _ { Y }$ xaims to distinguish between GY X(y ) DYimages { } and translated images $\{ G _ { X Y } ( x ^ { \prime } ) \}$ , the same as $D _ { \widetilde { X } }$ and $D _ { \widetilde { \gamma } }$ . 

DYIn the article, we take advantage of circularity consistency presented in [15] and geometrical consistency of Gc-GAN [19]. $f ( \cdot )$ is defined as 90 degrees clockwise rotation that aims to f ( )illustrate geometrical consistency. First, to get accurate target domain maps, the framework should focus on circularity consistency. For each remote sensing image  and transformed image x, the translation circularity should bring  and  back to the xoriginal image. Hence, the constraints are: $G _ { Y X } ( G _ { X Y } ( x ) ) \approx x .$ , $G _ { \widetilde { Y X } } ( G _ { \widetilde { X Y } } ( f ( x ) ) ) \approx$ GY X(GXY (x)) x-. Second, for geometrical consistency, GY X(GXY (f(x))) xour framework enforces that $G _ { X Y } ( x )$ and $G _ { \widetilde { X Y } } ( \widetilde { x } )$ should GXY (x) GXY (x)remain the same geometric transformation between and . We have $f ( G _ { X Y } ( x ) ) \approx G _ { \widetilde { X Y } } ( \widetilde { x } )$ , where $f ( x ) = { \widetilde { x } } , \operatorname { a n d } G _ { X Y } ( x ) \approx$ $f ^ { - 1 } ( G _ { \widetilde { X Y } } ( \widetilde { x } ) )$ x)) G, where $\begin{array} { r } { \Vec { x } = f ^ { - 1 } ( \widetilde { x } ) } \end{array}$ . 

(GXY (x)) x = f (x)Our objective constains three types of terms: The first is adversarial losses, which is used to match the distribution of generators in the domain of the map; the second is circularity consistency losses that prevent the learned mappings $G _ { X Y }$ and $G _ { Y X }$ GXYfrom contradicting each other; the loss of geometrical GY Xconsistency can be viewed as a reconstruction loss that depends on a predefined geometrical transformation function $f ( \cdot )$ . The fformulations of our objective are described as follows. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/2f6db145e56aba746226d1a4931820c81ad5fbff3fcc2879e38e127ab628d6c8.jpg)



Fig. 2. Description of how MapGen-GAN works on domain mapping: x and y are random examples from domain X and $\boldsymbol { Y } ,$ respectively. - - $f ( \cdot )$ is a predefined geometrical transformation function for images, which satisfies $\tilde { \boldsymbol { x } } = \mathsf { \tilde { f } } ^ { - 1 } ( f ( \boldsymbol { x } ) ) = f ( f ^ { - 1 } ( \boldsymbol { x } ) ) . \mathsf { \tilde { G } } _ { X Y } , G _ { Y X } , G _ { g c }$ − and $G _ { g c _ { - } Y X }$ are generators which target the domain translation tasks from X to $Y , Y { \mathrm { ~ t o ~ } } X , { \widetilde { X { \mathrm { ~ t o ~ } } Y } } ,$ , and Y to X, where $\overrightharpoon { X }$ and $\widetilde { Y }$ are two domains obtained by applying $f ( \cdot )$ on X and Y . $G _ { g c _ { - } X Y } , G _ { g c _ { - } Y X }$ represents generator $G _ { \widetilde { X Y } } , G _ { \widetilde { Y X } } . D _ { X } , D _ { Y } , D _ { \widetilde { X } } ,$ , and $\mathcal { P } _ { \widetilde { Y } }$ are adversarial discriminator in domain $\boldsymbol { X } , \boldsymbol { Y } , \widetilde { \boldsymbol { X } }$ , and Y , respectively. The green gc XY gc Y X XY Y X X Y Xdotted lines denote the unsupervised constraints circularity-consistency $( G _ { Y X } ( G _ { X Y } ( x ) ) \approx x , G _ { \widetilde { Y X } } ( G _ { \widetilde { X Y } } ( f ( x ) ) ) \approx \widetilde { x ) }$ and the orange dotted lines denotegeometrical-consistency $( G _ { X Y } ( x ) \approx f ^ { - 1 } ( G _ { \widetilde { X Y } } ( x ) ) , G _ { \widetilde { X Y } } ( x ) = f ( G _ { X Y } ( x ) ) )$ .


# B. Formulation

1) Adversarial Loss: The principle of MapGen-GAN can be summarized as converting one kind of image to another. Taking $G _ { X Y }$ as an example, we aim to convert the samples in remote GXYsensing images domain  to the samples in maps domain $Y$ . Therefore, the objective is to learn the mapping from  to  and set this mapping on $G _ { X Y }$ , which corresponds to the generator in GXYthe GAN. For generated images, we also need discriminator $D _ { X }$ DXto determine whether it is a real image, to compose an adversarial generative network. Generator $G _ { X Y }$ aims to minimize object function against an adversary discriminator $D _ { X }$ , which tries to maximize the objective. We express the objective as 

For the mapping $G _ { X Y } : X  Y$ 

$$
\begin{array}{l} L _ {\mathrm{gan}} (G _ {X Y}, D _ {Y}, X, Y) = E _ {y \sim P _ {Y}} [ \log D _ {Y} (y) ] \\ + E _ {x \sim P _ {X}} [ \log (1 - D _ {Y} (G _ {X Y} (x))) ] \tag {1} \\ \end{array}
$$

The adversarial loss $L _ { \mathrm { g a n } } ( G _ { \widetilde { X Y } } , D _ { \widetilde { Y } } , \widetilde { X } , \widetilde { Y } )$ has the same form to $L _ { \mathrm { g a n } } ( G _ { X Y } , D _ { Y } , X , \mathbf { \bar { Y } } )$ (. 

L (GXY , DY , X,For the mapping $G _ { Y X } \colon Y \to X$ 

$$
\begin{array}{l} L _ {\text { gan }} (G _ {Y X}, D _ {X}, Y, X) = E _ {x \sim P _ {X}} [ \log D _ {X} (x) ] \\ + E _ {y \sim P _ {Y}} [ \log (1 - D _ {X} (G _ {Y X} (y))) ]. \tag {2} \\ \end{array}
$$

The adversarial loss $L _ { \mathrm { g a n } } ( G _ { \widetilde { Y X } } , D _ { \widetilde { Y } } , \widetilde { Y } , \widetilde { X } )$ has the same form to $L _ { \mathrm { g a n } } ( G _ { Y X } , D _ { X } , Y , X )$ (. 

L (GY X, DX, Y, X)Total Adversarial Loss : 

$$
\begin{array}{l} L _ {\text { gan } _ {\text { total }}} = L _ {\text { gan }} (G _ {X Y}, D _ {Y}, X, Y) \\ + L _ {\text { gan }} (G _ {\widetilde {X Y}}, D _ {\widetilde {Y}}, \widetilde {X}, \widetilde {Y} \\ + L _ {\text { gan }} (G _ {Y X}, D _ {X}, Y, X) \\ + L _ {\text { gan }} (G _ {\widetilde {Y X}}, D _ {\widetilde {X}}, \widetilde {Y}, \widetilde {X}). \tag {3} \\ \end{array}
$$

2) Circularity Consistency Loss: In practice, the adversarial loss is the same as the original GAN loss, but it is rather difficult to train the whole network by using adversarial loss alone. The reason is that the mapping $G _ { X Y } : X  Y$ can completely make all $\{ x _ { i } \}$ GXY : X Ythe same image in the domain  , which invalidates the xi Yloss. Therefore, the significance of circularity consistency loss is to assume another mapping $G _ { Y X } : Y  X$ , which can convert the map $\{ y _ { j } \}$ GY X : Y Xin the domain to the remote sensing images in yj Ydomain . The network studies two mappings $G _ { X Y } : X $ $Y$ and $G _ { Y X } : Y  X$ by making $G _ { X Y } ( G _ { Y X } ( y ) ) \approx y$ Xand $G _ { Y X } ( G _ { X Y } ( x ) ) \approx x$ X GXY (GY X(y)) y. In this way, we define the loss function of circularity consistency as 

$$
\begin{array}{l} L _ {c y c} (G _ {X Y}, G _ {Y X}, X, Y) \\ = E _ {x \sim P _ {X}} [ \| G _ {Y X} (G _ {X Y} (x)) - x \| _ {1} ] \\ + E _ {y \sim P _ {Y}} [ \| G _ {X Y} (G _ {Y X} (y)) - y \| _ {1} ]. \tag {4} \\ \end{array}
$$

The circularity loss of $L _ { \mathrm { c y c } } ( G _ { \widetilde { X Y } } , G _ { \widetilde { Y X } } , \widetilde { X } , \widetilde { Y } )$ are same as to $L _ { \mathrm { c y c } } ( G _ { X Y } , G _ { Y X } , X , Y )$ (GXY , GY X , X, Y ). After the availability of geometrical L (GXfunction $f ( \cdot )$ Y X , X, Y ), the circularity-consistency loss can be described as 

$$
\begin{array}{l} L _ {\text { cyc }} (G _ {\widetilde {X Y}}, G _ {\widetilde {Y X}}, \widetilde {X}, \widetilde {Y}) \\ = E _ {\widetilde {x} \sim P _ {\widetilde {X}}} [ \| G _ {\widetilde {Y X}} (G _ {\widetilde {X Y}} (x)) - \widetilde {x} \| _ {1} ] \\ + E _ {\widetilde {y} \sim P _ {\widetilde {Y}}} [ \| G _ {\widetilde {X Y}} (G _ {\widetilde {Y X}} (y)) - \widetilde {y} \| _ {1} ]. \tag {5} \\ \end{array}
$$

To sum up, the total circularity loss is 

$$
\begin{array}{l} L _ {c y c - t o t a l} = L _ {\mathrm{cyc}} (G _ {X Y}, G _ {Y X}, X, Y) \\ + L _ {\text { cyc }} (G _ {\widetilde {X Y}}, G _ {\widetilde {Y X}}, \widetilde {X}, \widetilde {Y}). \tag {6} \\ \end{array}
$$

3) Geometrical Consistency Loss: Given a predefined geometrical transformation function $f ( \cdot )$ and its inverse function $f ^ { - 1 } ( \cdot )$ , which satisfies $f ^ { - 1 } ( f ( x ) ) = f ( f ^ { - 1 } ( x ) ) = x . G _ { X Y }$ and $G _ { \widetilde { X Y } }$ ) f (f(x)) = f(f (x)) = x GXYare generators aiming to target the domain translation tasks XYfrom  to $Y$ and $\widetilde { X }$ to $\widetilde { Y }$ , where $\widetilde { X }$ and $\widetilde { Y }$ are two domains X Y Xacquired by applying $f ( \cdot )$ X Yon all images in remote sensing images $X$ and maps $Y$ f( ), respectively. According to the geometrical X Yconsistency followed by Gc-GAN [19], the generated maps $y ^ { \prime } =$ $G _ { X Y } ( x )$ and $\widetilde { y ^ { \prime } } = G _ { \widetilde { X Y } } ( \widetilde { x } )$ should meet the formula $\widetilde { y ^ { \prime } } = f ( y ^ { \prime } )$ and $y ^ { \prime } = f ^ { - 1 } ( \widetilde { y ^ { \prime } } )$ XY. Taking $f ( \cdot )$ and $f ^ { - 1 } ( \cdot )$ into consideration, the geometrical consistency loss of the MapGen-GAN can be described as 

For the mapping - $G _ { X Y } \colon X \to Y$ 

$$
\begin{array}{l} L _ {\text { geo }} (G _ {X Y}, G _ {\widetilde {X Y}}, X, Y) \\ = E _ {x \sim P _ {X}} [ G _ {X Y} - f ^ {- 1} (G _ {\widetilde {X Y}} (f (x))) \| _ {1} ] \\ + E _ {x \sim P _ {X}} [ \| G _ {\widetilde {X Y}} (f (x)) - f (G _ {X Y} (x)) \| _ {1} ]. \tag {7} \\ \end{array}
$$

For the mapping $G _ { Y X } \colon Y \to X$ 

The geometrical-consistency loss is-

$$
\begin{array}{l} L _ {\text { geo }} (G _ {Y X}, G _ {\widetilde {Y X}}, X, Y) \\ = E _ {y \sim P _ {Y}} [ G _ {Y X} - f ^ {- 1} (G _ {\widetilde {Y X}} (f (y))) \| _ {1} ] \\ + E _ {y \sim P _ {Y}} [ \| G _ {\widetilde {Y X}} (f (y)) - f (G _ {Y X} (y)) \| _ {1} ]. \tag {8} \\ \end{array}
$$

Total geometrical−Consistency Loss: 

The total loss of geometrical-consistency can be summarized as 

$$
\begin{array}{l} L _ {g e o \_ t o t a l} = L _ {\mathrm{geo}} (G _ {X Y}, G _ {\widetilde {X Y}}, X, Y) \\ + L _ {\text { geo }} (G _ {Y X}, G _ {\widetilde {Y X}}, X, Y). \tag {9} \\ \end{array}
$$

4) Full Objective: By combining standard adversarial loss with two-sided unsupervised circularity consistency loss and geometrical consistency constraint, the mapping $G _ { X Y }$ can be targeted. The full objective of MapGen-GAN is 

$$
L = L _ {g a n - t o t a l} + \sigma L _ {c y c - t o t a l} + \lambda L _ {g e o - t o t a l} \tag {10}
$$

where $\sigma$ and $\lambda$ control the relative importance of the loss σfunction of circularity-consistency and geometrical-consistency, respectively. We set $\sigma { = } 1 0$ and $\lambda { = } 2 0$ in (10) during the training procedure. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/9a8b9fbba9ecf46d0ec1b7f97027a3e8e2a20fdfaa09b1a3492df1e8d9b7b8ce.jpg)



Fig. 3. Architecture of BRB-Unet: The left half of the structure is encoding, and the right half represents decoding; Conv means convolution, $/ 2$ represents the stride for downsampling is 2, and *1 means the factor for upsampling is 1. Add batch norm after the convolution layer and then add ReLU.


# C. Generator

Our MapGen-GAN framework consists of four generators and four discriminators which do not share parameters. The generator is based on an improved U-net structure, which introduces a residual network and normalized processing. The combination of U-net and residual network helps improve the accuracy of regional identification. In the map translation task, it is essential to use low-level details while retaining high-level semantic information. It is well known that residual network is used to train a deeper network. A deeper network can improve performance, but it can hinder training and cause degradation problems. U-net is widely used because of its better recognition ability. The combination of U-net and residual blocks could facilitate geometrical structure information propagation without degradation and obtain better performance under reducing the number of parameters. Specific details will be discussed below. 

As presented in Fig. 3, we can perceive that there are two nonignorable characteristics in the structure of BRB-Unet: One is the U-shaped structure, the other is skip connection added in the corresponding layers. In such a network, we divide BRB-Unet into a contraction network and expansion network. The contraction network has five encoder blocks to obtain high-level semantic information. The input image size is $2 5 6 \times 2 5 6$ , after five times downsampling, the image size becomes $8 \times 8$ . Moreover, the decoder blocks naturally correspond to five encoder blocks to recover the resolution, and the final size of the output image returns to $2 5 6 ~ \times ~ 2 5 6 .$ To reduce the loss of spatial information caused by the downsampling process, the skip layer connection is introduced, making BRB-Unet capture a wealth of geographic structure information of buildings, roads, and terrain in the transformation accurately. Compared with other encoder-decoder structures, the generator network we use in the MapGen-GAN dramatically decreases the training parameters, increases the network depth, and reduces the training time. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/e72520bed303298376dcb52c671df17b26edf5ebebd9bdefc46e41f5503d4df2.jpg)



Fig. 4. Encoder block: 1–5.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/95be41e78fd0bf038ae19bfbd23055d0683cc24ec13ae802f4a464a131755220.jpg)



Fig. 5. Decoder block: 1–4


In generating maps from remote sensing images, the focus is to precisely extract well-defined features such as houses and roads from the geographic scene. As we all know, U-net is an encoder-decoder structure. The encoder of U-Net has four downsamplings to obtain the advanced semantic information of the map. Then, the decoder naturally corresponds to four upsampling to expand the spatial dimensions to generate a segmentation map with the same spatial resolution as the input image. The feature map of upsampling recovery contains more low-level semantic information and makes the result more precise. Layer connections are introduced to reduce the spatial information loss caused by the downsampling process. In BRB-Unet, we use Resnet18 [47] as the encoder’s backbone. Figs. 4–6 are convolutional modules of encoder-blocks and decoder-blocks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/1260acc440ba6e22c38b0da3adaa8925a0e036bd14dd1b47acf8bbd0078c9cc8.jpg)



Fig. 6. Decoder block: 5


As shown in Fig. 4, we combine two basic blocks as an encoder block with residual learning, and the output of each layer is normalized first and then activated by an activation function. In Fig. 5, the decoder block 1 to 4 contains one deconvolutional layer with a kernel size of $2 \times 2$ and stride of 2, and one convolutional layer with a kernel size of $3 \times 3$ and stride of 1. The last decoder consists of a upsample layer with a scale factor of 2.0, and two convolutional layers with a kernel size of $3 \times 3$ presented in Fig. 6. Each layer of the decoder gradually recovers the image’s position information by deconvolution and combines with the original features corresponding to the encoder. It is well known that the parameters of the network in DL represent space complexity. The parameters size of BRB-Unet is about 98 M, which is more lightweight than the generator in CycleGAN [15] and pix2pix [7]. 

# D. Discriminator

MapGen-GAN consists of four discriminators which do not share parameters. The discriminator takes an image as input and attempts to predict it as the original image or the generator’s output. It belongs to a convolutional network and extracts features from the image. By adding a convolution layer, the output of discriminator is the prediction probability value, which indicates whether the input is a real image. The closer the probability value is to 1, the more discriminator can determine that the input is a real image. 

In order to better discriminate the part of the image, we make the discriminator distinguish images at the patch level following $7 0 \times 7 0$ PatchGAN [7], [48]. The goal of PatchGAN is to classify whether $7 0 \times 7 0$ overlapping image patches are real or fake [49], [50]. This patch-level discriminator structure has fewer parameters than a full-image discriminator and can process images of any size in a fully convoluted manner. 

# IV. EXPERIMENTS

To explore the effectiveness of the proposed model, we evaluate MapGen-GAN on two distinct datasets. One is the public Google Map datasets of New York City [7], and the other is the Map World datasets of Washington DC that were preprocessed and singled out rigorously. In the meantime, we make both quantitative and qualitative comparisons with state-of-the-art methods. After the training, four metrics are used to assess each model’s performance on the testing sets. We also adopt ablation studies and transfer experiments to analyze the performance of our model further. In the end, we demonstrate the effectiveness and validity of our MapGen-GAN. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/5f1b80de1e505337a70b2712617fafe1e96652e929a876de7b2b5d6f89b94ea6.jpg)



Fig. 7. Example of a pair of imprecise maps in the New York City dataset.


# A. Dataset Description

In order to verify the effectiveness of the proposed unsupervised map translator, several experiments were carried out on two separate datasets. The details of datasets are described as follows. 

1) Dataset 1. Public Google Maps: We downloaded 2196 images in and around New York City from Google Maps [7], splitting them into training, testing, and validation sets with 1099, 550, and 547, respectively. The image size of New York City datasets is $2 5 6 \times 2 5 6$ . 

Through repetitious attempts and experiments, we recognize that the quality of datasets is directly related to the network’s training results. Public map datasets of New York City we acquired online are imprecise and irregular. As presented in Fig. 7, we seek out a pair of inaccurate images of the public map datasets. A portion of the area covered by vegetation on the remote sensing image is marked green on the Google Map; the other part is not marked. Such maps would lead to the entire network’s chaotic learning during training, which decreases the performance of transformation and the quality of generated maps. 

2) Dataset 2. Atlas of Washington DC: The Washington DC and New York datasets differ significantly in style and resolution. We handle a series of data preprocessing work on the datasets of Washington DC in the United States. The preparatory work includes deleting the incomplete images, finding out the shadowed images, and distinguishing whether the vegetation coverage is marked green. Urban areas datasets of Washington DC consist of 2058 images. We split them into training sets, testing, and validation sets with 1136, 460, and 462. We train remote sensing images → maps translator with an image size of 256 × 256 using the training set in an unsupervised manner (unpaired) by ignoring the pair information. 

# B. Evaluation Metrics

1) RMSE/PSNR: Mean square error (mse) [51], represents the mse of the current image  and the reference image  . X YPSNR is based on the error between corresponding pixels. As the most widely used performance quantification metric, PSNR is simple and easy to use. The larger the PSNR is, the higher the image quality is, but the range of values is not specified. Given an image  and a noisy image  of size, the function of PSNR Ican be described as 

$$
\begin{array}{l} \mathrm{PSNR} = 1 0 \cdot \log_ {1 0} \frac {\mathrm{MAX} _ {I} ^ {2}}{\text {MSE}} \tag {11} \\ = 2 0 \cdot \log_ {1 0} {\frac {\mathrm{MAX} _ {I}}{\sqrt {\mathrm{MSE}}}} \\ \end{array}
$$

where: 

$$
\mathrm{MSE} = \frac {1}{m n} \sum_ {i = 0} ^ {m - 1} \sum_ {j = 0} ^ {n - 1} [ I (i, j) - K (i, j) ] ^ {2} \tag {12}
$$

2) Structural Similarity (SSIM): The full name of SSIM is the structural similarity index [52], which is a measurement of similarity between two images. It has three major innovations: First, SSIM takes each pixel as the center of a small block (11,11). Second, SSIM compares three statistical features of image blocks: Luminance (mean), contrast (variance), and structure (covariance). Third, SSIM adopts the following formula: 

$$
q = \frac {2 a b + C}{a ^ {2} + b ^ {2} + C}. \tag {13}
$$

The similarity value range of $\ " a \ "$ and $\ddot { \cdot } b \vec { \cdot }$ is [0,1], and C is a constant to prevent the denominator from being 0. SSIM is a number between 0 and 1. If the two images are the same, SSIM 1. 

=3) ACC[Pixel Accurcy(%)]: Given a pixel i with the groundtruth RGB value $( r _ { i } , g _ { i } , b _ { i } )$ of a map and the predicted RGB value $( r _ { i } ^ { \prime } , g _ { i } ^ { \prime } , b _ { i } ^ { \prime } )$ i, gi, bi), if max $( \mid r _ { i } - r _ { i } ^ { \prime } \mid , \mid g _ { i } - g _ { i } ^ { \prime } \mid , \mid b _ { i } - b _ { i } ^ { \prime } \mid$ $) < \delta ( \delta = 5 $ (ri, gi, bi) ( ri ri , gi gi , bi biin this article) we consider this is an accurate ) < δ(δ = 5prediction [19]. 

# C. Baselines and Experiment Setting

We conduct the experiments by using Python 3.6 and the PyTorch framework. Due to the network’s complexity and large amounts of calculation for the loss function, we use NVIDIA Tesla V100 GPU and Nvidia’s CUDA 10.0 API model to train the proposed model. First, both generator and discriminator use Adam optimizer, the initial parameters with the same learning rate $2 \mathrm { e } { \cdot } 4 .$ , and the decay of the first-order momentum of the gradient is set as 0.5. Moreover, the learning rate fixes in the initial 100 epochs, then linearly decay to zero over the next 200 epochs. Second, when calculating the mapping $G _ { X Y } : X  Y$ , GXY : X Yit is appropriate to increase the adversarial loss’s coefficient weight because we want maps in the  domain. Third, we affil-Yiate identity loss [15] to improve training results. However, the addition of identity loss seems to make the training more difficult to converge. In the end, we replace batch normalization [53] with instance normalization [54] when the training results are unsatisfactory. 

It is impractical to compare maps generated by DL methods with manual map-making methods, which are inevitably subjective. We take manually generated maps as ground truth in our experiment. For remote sensing images converting to maps, we evaluate the performance of our MapGen-GAN and four baselines to make quantitative and qualitative comparisons by using two distinct datasets. 


TABLE I SCORES FOR DIFFERENT METHODS, EVALUATED ON GOOGLE MAPS OF NEW YORK CITY


<table><tr><td>Method</td><td>RMSE</td><td>PSNR</td><td>SSIM</td><td>ACC(%)</td></tr><tr><td colspan="5">Benchmark Performance</td></tr><tr><td>DiscoGAN</td><td>31.4334</td><td>23.6523</td><td>0.5767</td><td>34.3904</td></tr><tr><td>DualGAN</td><td>34.5225</td><td>21.4944</td><td>0.5274</td><td>28.0456</td></tr><tr><td>CycleGAN</td><td>30.7254</td><td>24.5589</td><td>0.6366</td><td>32.3634</td></tr><tr><td>Gc-GAN</td><td>30.3156</td><td>24.6457</td><td>0.6487</td><td>34.9453</td></tr><tr><td>MapGen-GAN</td><td>30.0567</td><td>25.1196</td><td>0.6887</td><td>38.5405</td></tr></table>


MapGen-GAN has the highest score for four metrics, shown in bold font. 


1) CycleGAN: The innovation of CycleGAN lies in transferring image content from the source domain to the target domain without paired training data [15]. The dataset 1 Google Maps are employed in CycleGAN. 

2) DualGAN: A dual-learning [17] mode GAN network structure is proposed for image-to-image translation. Google Maps are also used in DualGAN’s experiment. 

3) DiscoGAN: It processes cross-domain transformations in an unsupervised manner and achieves style transfer [16]. 

4) Gc-GAN: By developing a geometry consistency constraint [19], Gc-GAN aims to reduce the search space of possible solutions. It uses Google Maps as a part of training datasets. 

# D. Evaluation of the Performance of MapGen-GAN

To verify the effectiveness of map generation using MapGen-GAN, we established a comparison of the proposed method versus four baselines applying on two datasets. Four kinds of image quality evaluation metrics are calculated for each method. Table I shows the quantitative results training with dataset 1: Google Maps of New York City. MapGen-GAN has the best outstanding performance in four metrics than the other four baselines, and the runner up is Gc-GAN. In particular, MapGen-GAN yields an 8%–12% improvement over the baselines at most in pixel accuracy measurement with parameter   , and δ = 56%–31% increment in SSIM. The performance indicates that the Google Maps produced by our translator are more similar to the real image. 

For dataset 2 Map World of Washington DC, the evaluation is presented in Table II. As expected, the winning architecture for the map translation is our framework MapGen-GAN. In particular, MapGen-GAN achieves an enhancement of 13% against DualGAN and 5% against Gc-GAN on SSIM. Moreover, the improvement of MapGen-GAN on the ACC metric is also prominent. 

The qualitative results of two datasets are displayed in Figs. 8 and 9. We select a few samples of the reconstructed images from two datasets randomly. Obviously, our method can improve the training output in both datasets and generate more impressive translation empirically. 


TABLE II SCORES FOR DIFFERENT METHODS, EVALUATED ON DATASETS OF WASHINGTON DC


<table><tr><td>Method</td><td>RMSE</td><td>PSNR</td><td>SSIM</td><td>ACC(%)</td></tr><tr><td colspan="5">Benchmark Performance</td></tr><tr><td>DiscoGAN</td><td>19.2786</td><td>28.3624</td><td>0.7546</td><td>37.1105</td></tr><tr><td>DualGAN</td><td>17.7898</td><td>28.8578</td><td>0.7065</td><td>45.5231</td></tr><tr><td>CycleGAN</td><td>18.4752</td><td>28.9286</td><td>0.7846</td><td>40.6623</td></tr><tr><td>Gc-GAN</td><td>17.5387</td><td>28.4568</td><td>0.7763</td><td>43.5724</td></tr><tr><td>MapGen-GAN</td><td>16.3994</td><td>29.7351</td><td>0.7994</td><td>50.9489</td></tr></table>


Each bold entitle represents the highest score. 


As shown in Fig. 9, for remote sensing images with partial vegetation coverage, we can intuitively discriminate that MapGen-GAN can discern areas covered by vegetation exquisitely. Our method can also identify the more precise outline for buildings and make the city’s geographical layout more regular. In terms of the generation of roads, MapGen-GAN performs better than others due to the addition of geometrical and circularity consistency constraints, making it more sensitive to roads’ transformation. Moreover, MapGen-GAN is also good at training marginal contours of water systems such as lakes. 

In order to further prove the performance of our method, we conduct quantitative and qualitative comparisons with other deep networks VAE [55] and PixelCNN [56]: 

1) VAE: Variational Autoencoder is a deep learning generation model based on variational method. It is established in the standard function approximation unit. Besides, it can take advantage of stochastic gradient descent to optimize. The most characteristic of VAE is to imitate the learning prediction mechanism of automatic coding machine, encoding, and decoding between measurable functions. 

2) PixelCNN: It is also a generative model with tractable likelihood, so that it can be sampled easily. The core CNN computes the probability distribution on a pixel value and is constrained by the pixel values on the left and the upper sides. 

Table III shows the quantitative results for VAE, PixelCNN, and our proposed MapGen-GAN training with two datasets. There is no doubt that MapGen-GAN has the best outstanding performance in both New York City datasets and Washington DC datasets. From the qualitative results of two datasets displayed in Figs. 10 and 11, VAE and PixelCNN perform much worse than our method, especially in urban roads and buildings generation. It is probably because VAE is forced to fit data to mixed Gaussian or other distribution of the finite dimension in the optimization process. It will lead to two results: The inevitable loss of information in the mapping process, especially secondary information loss. The other is the poor encoding and recovery effect of information that does not conform to the present distribution. If such a distribution is forced to be projected onto the Gaussian distribution, it will inevitably lead to ambiguity. For PixelCNN, essentially, it serially generates images, pixel by pixel, and each pixel’s generation is only dependent on the above information. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/268ff588e46dd9214c436e7855ee7e3fc1cb14b9329e9ac98a7a4efcb304b85d.jpg)



Fig. 8. Results of different methods on New York City datasets.


<table><tr><td>Input</td><td>CycleGAN</td><td>DiscoGAN</td><td>DualGAN</td><td>Gc-GAN</td><td>MapGen-GAN</td><td>Ground Truth</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>


Fig. 9. Results of different methods on Washington DC datasets.



TABLE III QUANTITATIVE SCORES OF COMPARISON WITH VAE AND PIXELCNN ON TWO DATASETS BY FOUR EVALUATION METRICS


<table><tr><td rowspan="2">Method</td><td colspan="4">Dataset 1</td><td colspan="4">Dataset 2</td></tr><tr><td>RMSE</td><td>PSNR</td><td>SSIM</td><td>ACC(%)</td><td>RMSE</td><td>PSNR</td><td>SSIM</td><td>ACC(%)</td></tr><tr><td>VAE</td><td>37.4278</td><td>18.3150</td><td>0.3257</td><td>26.1235</td><td>24.3423</td><td>23.2206</td><td>0.6069</td><td>33.1642</td></tr><tr><td>PixelCNN</td><td>35.6443</td><td>20.3329</td><td>0.5158</td><td>30.8405</td><td>20.4302</td><td>25.0453</td><td>0.6980</td><td>38.7603</td></tr><tr><td>MapGen-GAN</td><td>30.0567</td><td>25.3854</td><td>0.6887</td><td>38.5430</td><td>16.3994</td><td>29.7351</td><td>0.7926</td><td>50.9422</td></tr></table>


Each bold entitle represents the highest score. 


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/dbc5659c20aaa28d58e4240527781a7f161f05e78eb34cfdfa80307687af4495.jpg)



Fig. 10. Results compared with VAE and PixelCNN on New York City datasets.


However, GAN generates images in a parallel manner. The generation of each pixel depends on the context information and uses the information below. That is why PixelCNN is much slower, and the image quality is worse than our method. MapGen-GAN avoids the direct measurement of distribution differences and lets the neural network learn the distance itself in a contrarious way. 

However, it is more difficult to match the distribution, which leads to mode collapse and loss of small distributions. We also added some skills to maintain the stability of GAN training. The convergence curve of MapGen-Gan is shown in Figs. 12 and 13. The loss of the discriminator is a rival to the generator. The goal is to confuse the discriminator about the output of the generator and the real label. So the ideal error curve for discriminator loss should vibrate around 0 to 1. The loss curves in Figs. 12 and 13 show that the error curve for discriminator loss Loss− ( D)ends up oscillating around 0. In other words, the discriminator has a 50% probability that the output is real, 50% probability that output is false. These two convergence curves show that the training of MapGen-GAN on the two datasets is successful, and there is no mode collapse. 

Next, we compare the time taken to convert remote sensing images into maps using traditional manual cartography and DL methods. Referring to the current manual map-making methods, generating maps from remote sensing images mostly relies on geographic information system (GIS). According to [57] and the investigation of practical manual drawing, it takes about one day to make a map of 0.5 square kilometers and 1:2000 scale manually. The Washington datasets used in this article are 18-level Map World with a scale of 1:2256, each sample representing about 0.14 square kilometers of an area. Using DL methods to generate a map of the Washington datasets only takes 0.5 s. From this, we can calculate that the generation of 10 square kilometers and 1:2000 scale of a map using traditional methods will take about 20 days. Whereas, by using DL methods, it takes only 39 seconds to generate 10 square kilometers and 1:2000 scale of a map from remote sensing image with an image size of $2 5 6 \times 2 5 6$ . Table IV is the time comparison between traditional manual map-making methods and several DL methods based on Washington DC datasets for generating 6 square kilometers and 1:2000 scale of a map. The results demonstrate that traditional map-making methods render nonreal time in disaster response scenarios that are not adopted in emergency rescue operations. By contrast, our model generates maps much faster with maps of the same size and scale. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/1942288f354b94e74ca251fd54c0a09c60972699ec9f4c1915263ebbc53c4503.jpg)



Fig. 11. Results compared with VAE and PixelCNN on Washington DC datasets.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/bc48cf3da9129d83c1653d30ba5e5364a1377f24d7f25578ed92b90cb6156b79.jpg)



Fig. 12. Convergence curve of MapGen-GAN during the training process on New York City datasets.


# E. Ablation Study

To verify the rationality of the proposed GAN, we perform ablation studies to discuss MapGen-GAN further. Specifically, we remove either circularity-consistency or geo-consistency- - constraints from the whole architecture to verify whether the combination of the two is sufficient. Moreover, we separately train generators $G _ { X Y }$ and $G _ { \widetilde { X } \widetilde { Y } }$ with shared parameters and GXY GXYnonshared parameters. Finally, we remove our improved generator BRB-Unet from the framework and replace it with the original U-net to demonstrate the effectiveness of BRB-Unet. Those several ablation approaches are described as follows. 

1) MapGen-GAN-noGeo: MapGen-GAN with circularityconsistency constraint only. 


TABLE IV TIME COMPARISON BETWEEN TRADITIONAL MANUAL MAP-MAKING METHODS, DL METHODS FOR GENERATING 10 SQUARE KILOMETERS, AND 1:2000 SCALE OF A MAP BASED ON WASHINGTON DC DATASETS


<table><tr><td>Method</td><td>Traditional</td><td>MapGen-GAN</td><td>CycleGAN</td><td>DualGAN</td><td>DiscoGAN</td><td>Gc-GAN</td></tr><tr><td>Time</td><td>20 days</td><td>39s</td><td>39s</td><td>39s</td><td>39s</td><td>39s</td></tr></table>


TABLE V SCORES OF ABLATION STUDY ON TWO DATASETS BY FOUR EVALUATION METRICS


<table><tr><td rowspan="2">Method</td><td colspan="4">Dataset 1</td><td colspan="4">Dataset 2</td></tr><tr><td>RMSE</td><td>PSNR</td><td>SSIM</td><td>ACC(%)</td><td>RMSE</td><td>PSNR</td><td>SSIM</td><td>ACC(%)</td></tr><tr><td>MapGen-GAN-noGeo</td><td>30.7367</td><td>24.4102</td><td>0.6387</td><td>33.0745</td><td>18.2706</td><td>27.6286</td><td>0.7796</td><td>44.0631</td></tr><tr><td>MapGen-GAN-share</td><td>32.7268</td><td>23.5203</td><td>0.5058</td><td>27.8405</td><td>20.8307</td><td>25.0453</td><td>0.7680</td><td>38.7603</td></tr><tr><td>MapGen-GAN-Unet</td><td>31.5476</td><td>23.9334</td><td>0.6276</td><td>32.7789</td><td>19.8657</td><td>26.8945</td><td>0.7731</td><td>42.7116</td></tr><tr><td>MapGen-GAN-noCycle</td><td>30.3275</td><td>24.8352</td><td>0.6428</td><td>35.0324</td><td>17.7328</td><td>29.1652</td><td>0.7804</td><td>45.5756</td></tr><tr><td>MapGen-GAN</td><td>30.0567</td><td>25.3854</td><td>0.6887</td><td>38.5430</td><td>16.3994</td><td>29.7351</td><td>0.7926</td><td>50.9422</td></tr></table>


The winner architecture is our MapGen-GAN in bold. 


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/a5230d3af0e53823a4eb827e8771de38f36e6f32e58899cf66d5c461707dc472.jpg)



Fig. 13. Convergence curve of MapGen-GAN during the training process on Washington DC datasets.


2) MapGen-GAN-noCycle: MapGen-GAN with geoconsistency constraint merely. 

3) MapGen-GAN-Share: $G _ { X Y }$ and $G _ { \widetilde { X } \widetilde { Y } }$ share parameters. 

GXY GXY4) MapGen-GAN-Unet: Training MapGen-GAN by removing generator network BRB-Unet and replacing it with U-net [21]. 

5) MapGen-GAN: It is the proposed model of this article. 

The quantitative scores are reported in Table V, and the qualitative results of two datasets are presented in Figs. 14 and 15. Clearly, the winner architecture is our MapGen-GAN, which can achieve higher scores and produce more precise maps on both datasets. The runner up is MapGen-GAN-noCycle. MapGen-GAN-noGeo and MapGen-GAN-noCycle perform more poorly than MapGen-GAN on all metrics. It means that the combination of geometrical-consistency and circularity-consistency constraints significantly improves the network’s learning capacity and produces more sensible translations. MapGen-GAN without shared parameters has a higher score than MapGen-- - GAN-share. The generator $G _ { X Y }$ applied in MapGen-GAN is GXYinsufficient to dispose both mappings $X  Y$ and $\widetilde X \to \widetilde Y$ . Conjointly modeling $G _ { X Y }$ and $G _ { \widetilde { X } \widetilde { Y } }$ X Y X Ywith the shared parameters XYlessens the performance due to domain diversities produced by geometrical transformations. When removing the proposed generator BRB-Unet and replacing it with the original U-net, the acquired scores are much lower than the framework employed BRB-Unet. Residual basic blocks used in BRB-Unet enlarge the receptive field on the different resolutions between encoder and decoder. More specifically, adding residual basic blocks dilates the network’s depth and systematically improves the accuracy of regional identification. 

# F. Transfer Experiments

We then conduct transfer experiments to discuss the performance of MapGen-GAN further when the training data are different from the testing data in the region of the earth, scale, and cartographic style. As is commonly known, infrastructure varies significantly from place to place. The task of training a map translator requires diverse training samples from different areas. However, we usually do not have access to map datasets from one location to another. It is necessary to use a trained map translator to generate maps of areas with inadequate datasets in a particular area. The New York City and Washington DC datasets have different levels and resolutions, making the well-trained generators have different styles and semantic features. We apply the generator trained by NYC datasets on WDC datasets to affirm whether this simple transfer learning method is applicable. 

The quantitative score is reported in Table VI, while qualitative results of two datasets are presented in Figs. 16 and 17. As shown in Table VI, we use four metrics to evaluate the performance of MapGen-GAN and other baselines. In both NYC datasets and WDC datasets, MapGen-GAN gets higher scores on three of four metrics than other methods, and the runner up is underlined. To further discuss our method’s performance, we employ six people to grade two kinds of map transfer experiment results subjectively. By sorting the results of four baselines and our MapGen-GAN, the best ranking corresponds to number one, and the worst ranking is corresponding to number five. Finally, we sum and average the numbers of each method marked by each people, while the lowest score is related to the best method. After statistics, the winner is our MapGen-GAN on both NYC datasets transferred by WDC-generator and WDC datasets transferred by NYC-generator experiments. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/0dd9432fe75216f87a4009843faad3ca0d9272f809f5cb9ac366dee1044d18ac.jpg)



Fig. 14. Results of Ablation Studies on New York City datasets.



TABLE VI TRANSFER EVALUATION SCORES ON TWO DATASETS BY FOUR EVALUATION METRICS


<table><tr><td rowspan="2">Method</td><td colspan="4">NYC transferred by WDC-generator</td><td colspan="4">WDC transferred by NYC-generator</td></tr><tr><td>RMSE</td><td>PSNR</td><td>SSIM</td><td>ACC(%)</td><td>RMSE</td><td>PSNR</td><td>SSIM</td><td>ACC(%)</td></tr><tr><td>CycleGAN</td><td>46.1875</td><td>20.0146</td><td>0.6105</td><td>2.4834</td><td>63.7952</td><td>14.7468</td><td>0.6128</td><td>1.5137</td></tr><tr><td>DiscoGAN</td><td>43.0042</td><td>22.2164</td><td>0.6085</td><td>2.5835</td><td>57.6740</td><td>17.8433</td><td>0.5713</td><td>2.5467</td></tr><tr><td>DualGAN</td><td>39.7066</td><td>20.8650</td><td>0.5931</td><td>2.5245</td><td>68.6099</td><td>17.7659</td><td>0.6564</td><td>4.0636</td></tr><tr><td>Gc-GAN</td><td>40.7048</td><td>21.4487</td><td>0.6225</td><td>2.8435</td><td>53.1131</td><td>22.0639</td><td>0.6340</td><td>3.8990</td></tr><tr><td>MapGen-GAN</td><td>38.0763</td><td>21.7925</td><td>0.6276</td><td>3.2326</td><td>55.8044</td><td>23.0479</td><td>0.6699</td><td>5.5646</td></tr></table>


NYC transferred by WDC-generator means that we use generator trained by WDC datasets to generate NYC Google Maps; WDC transferred by NYC-generator means that we use generator trained by NYC datasets to generate WDC Maps World. Each bold entitle represents the highest score. 


From the qualitative results, especially in Fig. 16, the NYCdatasets-trained MapGen-GAN can generate rough outlines of buildings and roads in WDC datasets, although the two datasets are quite diverse on both color style and plotting scale. The outputs of NYC datasets using the WDC-datasets-trained model 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/8dc05782b3d22ec97afb4497a19f0c3b4cfe47600e47068c58206d6d339df30e.jpg)



Fig. 15. Results of Ablation Studies on Washington DC datasets.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/f180cd79dc079624a44f6868b6c190e0d7a4abf48764548d86df311c3163fc3b.jpg)



Fig. 16. Transfer results of Washington DC datasets by using the generator trained with New York City datasets.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/6d7bee43030ae15f6505637cb07298d4e3f791092da08f8597819b8891d560f5.jpg)



Fig. 17. Transfer results of New York City datasets by using the generator trained with Washington DC dataset.


are not acceptable as expected, which possibly resulted from the light color, and the plotting scale of the Washington DC atlas is larger than NewYork City’s. Low-precision generators on high-precision maps may result in poor results. Improving the model’s versatility for map datasets of different levels and styles is practical and worthy researching in our future works. 

# V. CONCLUSION

In this article, we propose a novel unsupervised domain mapping framework named MapGen-GAN. MapGen-GAN can transform remote sensing images to maps directly and quickly under disaster emergency response scenarios. In MapGen-GAN, we integrate circularity and geometrical consistency constraints into the whole architecture to reduce the translation’s semantic distortions. Moreover, an improved map generator based on BRB-Unet is proposed to capture a wealth of geographic structure information in the conversion of maps. The experimental results show that our model produces more impressive and competitive translation than other state-of-the-art approaches on two distinct datasets. 

For future works, we plan to employ transfer learning techniques for handling maps at different levels and scales. Besides, we intend to perform more comprehensive checking on datasets while improving the translation accuracy, not only on urban datasets but also on the countryside or other general datasets. 

# ACKNOWLEDGMENT

The authors would also like to thank the anonymous referees for their valuable comments and helpful suggestions. 

# REFERENCES



[1] M. P. Peterson, Interactive and Animated Cartography. Englewood Cliffs, NJ, USA: Prentice Hall, 1995. 





[2] W. Li and C.-Y. Hsu, “Automated terrain feature identification from remote sensing imagery: A deep learning approach,” Int. J. Geographical Inf. Sci., vol. 34, no. 4, pp. 637–660, 2020. 





[3] X. Li, Y. Wang, L. Zhang, S. Liu, J. Mei, and Y. Li, “Topology-enhanced urban road extraction via a geographic feature-enhanced network,” IEEE Trans. Geosci. Remote Sens., vol. 58, no. 12, pp. 8819–8830, Dec. 2020. 





[4] Google Maps, Google, CA, USA. [Online]. Available: https://www. google.com/maps 





[5] Tianditu, China. [Onine]. Available: https://www.tianditu.gov.cn 





[6] Baidu, Beijing, China. [Online]. Available: https://map.baidu.com 





[7] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2017, pp. 1125–1134. 





[8] Z. Xin, T. Kunwang, Z. Yi, “Design of the earthquake emergency service platform based on map world,” Eng. Surveying Mapping, vol. 25, no. 3, pp. 46–50, 2016. 





[9] W. Wang, N. Yang, Y. Zhang, F. Wang, T. Cao, and P. Eklund, “A review of road extraction from remote sensing images,” (in English) J. Traffic Transp. Eng., vol. 3, no. 3, pp. 271–282, 2016. 





[10] Z. Wu, Y. Gao, L. Li, J. Xue, and Y. Li, “Semantic segmentation of high-resolution remote sensing images using fully convolutional network with adaptive threshold,” Connection Sci., vol. 31, no. 2, pp. 169–184, 2019. 





[11] R. Hang, Q. Liu, D. Hong, and P. Ghamisi, “Cascaded recurrent neural networks for hyperspectral image classification,” IEEE Trans. Geosci. Remote Sens., vol. 57, no. 8, pp. 5384–5394, Aug. 2019. 





[12] R. Hang, Z. Li, P. Ghamisi, D. Hong, G. Xia, and Q. Liu, “Classification of hyperspectral and LiDAR data using coupled CNNs,” IEEE Trans. Geosci. Remote Sens., vol. 58, no. 7, pp. 4939–4950, Jul. 2020. 





[13] P. Shamsolmoali, M. Zareapoor, R. Wang, H. Zhou, and J. Yang, “A novel deep structure u-Net for sea-land segmentation in remote sensing images,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 12, no. 9, pp. 3219–3232, Sep. 2019. 





[14] E. Maggiori, Y. Tarabalka, G. Charpiat, and P. Alliez, “Convolutional neural networks for large-scale remote-sensing image classification,” IEEE Trans. Geosci. Remote Sens., vol. 55, no. 2, pp. 645–657, Feb. 2017. 





[15] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proc. IEEE Int. Conf. Comput. Vis., 2017, pp. 2223–2232. 





[16] T. Kim, M. Cha, H. Kim, J. K. Lee, and J. Kim, “Learning to discover cross-domain relations with generative adversarial networks,” in Proc. Int. Conf. Mach. Learn., 2017, pp. 1857–1865. 





[17] Z. Yi, H. Zhang, P. Tan, and M. Gong, “Dualgan: Unsupervised dual learning for image-to-image translation,” in Proc. IEEE Int. Conf. Comput. Vis., 2017, pp. 2849–2857. 





[18] S. Benaim and L. Wolf, “One-shot unsupervised cross domain translation,” in Proc. Adv. Neural Inf. Process. Syst., 2018, pp. 2104–2114. 





[19] H. Fu, M. Gong, C. Wang, K. Batmanghelich, K. Zhang, and D. Tao, “Geometry-consistent generative adversarial networks for one-sided unsupervised domain mapping,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2019, pp. 2427–2436. 





[20] J. Gu et al., “Recent advances in convolutional neural networks,” Pattern Recognit., vol. 77, pp. 354–377, 2018. 





[21] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervention, 2015, pp. 234–241. 





[22] P. Sangkloy, J. Lu, C. Fang, F. Yu, and J. Hays, “Scribbler: Controlling deep image synthesis with sketch and color,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2017, pp. 5400–5409. 





[23] T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, A. Tao, J. Kautz, and B. Catanzaro, “High-resolution image synthesis and semantic manipulation with conditional GANs,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2018, pp. 8798–8807. 





[24] M. Qi, Y. Wang, A. Li, and J. Luo, “STC-GAN: Spatio-temporally coupled generative adversarial networks for predictive scene parsing,” IEEE Trans. Image Process., vol. 29, pp. 5420–5430, 2020. 





[25] Y. Taigman, A. Polyak, and L. Wolf, “Unsupervised cross-domain image generation,” Nov. 2016, arXiv:1611.02200. 





[26] S. Zhou, T. Xiao, Y. Yang, D. Feng, Q. He, and W. He, “GeneGAN: Learning object transfiguration and attribute subspace from unpaired data,” 2017, arXiv:1705.04932. 





[27] H. Tang, D. Xu, W. Wang, Y. Yan, and N. Sebe, “Dual generator generative adversarial networks for multi-domain image-to-image translation,” in Proc. Asian Conf. Comput. Vision, 2018, pp. 3–21. 





[28] X. Huang, M.-Y. Liu, S. Belongie, and J. Kautz, “Multimodal unsupervised image-to-image translation,” in Proc. Eur. Conf. Comput. Vision, 2018, pp. 172–189. 





[29] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” 2014, arXiv:1409.1556. 





[30] S. Ren, K. He, R. Girshick, and S. Jian, “Faster R-CNN: Towards real-time object detection with region proposal networks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 6, pp. 1137–1149, Jun. 2017. 





[31] I. Goodfellow et al., “Generative adversarial nets,” in Proc. Adv. Neural Inf. Process. Syst., 2014, pp. 2672–2680. 





[32] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training GANs,” in Proc. Adv. Neural Inf. Process. Syst., 2016, pp. 2234–2242. 





[33] T. DeVries, A. Romero, L. Pineda, G. W. Taylor, and M. Drozdzal, “On the evaluation of conditional GANs,” 2019, arXiv:1907.08175. 





[34] A. Brock, T. Lim, J. M. Ritchie, and N. Weston, “Neural photo editing with introspective adversarial networks,” 2016, arXiv:1609.07093. 





[35] F. Xiong, Q. Wang, and Q. Gao, “Consistent embedded GAN for imageto-image translation,” IEEE Access, vol. 7, pp. 126 651–126 661, 2019. 





[36] P. Welander, S. Karlsson, and A. Eklund, “Generative adversarial networks for image-to-image translation on multi-contrast MR images–A comparison of cycleGAN and unit,” 2018, arXiv:1806.07777. 





[37] Y. Choi et al., “StarGAN: Unified generative adversarial networks for multi-domain image-to-image translation,” in Proc. IEEE Conf. Comput. Vision Pattern Recognit., 2018, pp. 8789–8797. 





[38] M. Zhu, P. Pan, W. Chen, and Y. Yang, “DM-GAN: Dynamic memory generative adversarial networks for text-to-image synthesis,” in Proc. IEEE Conf. Comput. Vision Pattern Recognit., 2019, pp. 5802–5810. 





[39] X. Chen, L. Qing, X. He, X. Luo, and Y. Xu, “FTGAN: A fullytrained generative adversarial networks for text to face generation,” 2019, arXiv:1904.05729. 





[40] T. Xu et al., “AttnGAN: Fine-grained text to image generation with attentional generative adversarial networks,” in Proc. IEEE Conf. Comput. Vision Pattern Recognit., 2018, pp. 1316–1324. 





[41] S. Reed, Z. Akata, X. Yan, L. Logeswaran, B. Schiele, and H. Lee, “Generative adversarial text to image synthesis,” in Proc. Int. Conf. Mach. Learn., 2016, pp. 1060–1069. 





[42] H. M. Kasem, K.-W. Hung, and J. Jiang, “Spatial transformer generative adversarial network for robust image super-resolution,” IEEE Access, vol. 7, pp. 182 993–183009, 2019. 





[43] Y. Zhang, X. Li, and Q. Zhang, “Road topology refinement via a multiconditional generative adversarial network,” Sensors, vol. 19, no. 5, 2019, Art. no. 1162. 





[44] D. Marmanis, K. Schindler, J. D. Wegner, S. Galliani, M. Datcu, and U. Stilla, “Classification with an edge: Improving semantic image segmentation with boundary detection,” ISPRS J. Photogrammetry Remote Sens., vol. 135, pp. 158–172, 2018. 





[45] W. Li, C. He, J. Fang, J. Zheng, H. Fu, and L. Yu, “Semantic segmentationbased building footprint extraction using very high-resolution satellite images and multi-source gis data,” Remote Sens., vol. 11, no. 4, 2019, Art. no. 403. 





[46] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in Proc. IEEE Conf. Comput. Vision. Pattern Recognit., 2017, pp. 4700–4708. 





[47] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vision Pattern Recognit., 2016, pp. 770–778. 





[48] C. Li and M. Wand, “Precomputed real-time texture synthesis with Markovian generative adversarial networks,” in Proc. Eur. Conf. Comput. Vision, 2016, pp. 702–716. 





[49] J. Hoffman et al., “CyCADA: Cycle-consistent adversarial domain adaptation,” in Proc. Int. Conf. Mach. Learn, 2018, pp. 1989–1998. 





[50] C. Ledig et al.et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in Proc. IEEE Conf. Comput. Vision Pattern Recognit., 2017, pp. 4681–4690. 





[51] C. Willmott and K. Matsuura, “Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance,” Climate Res., vol. 30, no. 1, pp. 79–82, 2005. 





[52] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, Apr. 2004. 





[53] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in Proc. Int. Conf. Mach. Learn., 2015, pp. 448–456. 





[54] D. Ulyanov, A. Vedaldi, and V. Lempitsky, “Instance normalization: The missing ingredient for fast stylization,” 2016, arXiv:1607.08022. 





[55] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” 2013, arXiv:1312.6114. 





[56] A. Van den Oord, N. Kalchbrenner, L. Espeholt, O. Vinyals, A. Graves et al. “Conditional image generation with pixelCNN decoders,” in Proc. Adv. Neural Inf. Process. Syst., 2016, pp. 4790–4798. 





[57] C. B. Jones, Geographical Information Systems and Computer Cartography. Evanston, IL, USA: Routledge, 2014. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/57638c0842ebff470ea311e6492407d53f12cf3c9b5ee53573e41b88f990bb56.jpg)


Jieqiong Song received the B.S. degree in information security, and the M.S. degree in software engineering from Central South University, Changsha, China, in 2011 and 2016, respectively. She is currently working toward the Ph.D. degree with the College of Electronic Science and Technology, National University of Defense Technology, Changsha, China. 

Her research interests include machine learning, remote sensing image processing, and photogrammetry. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/3bf86aa8cf6798620913175418be34c604fc9a917e294ed51b32f6f8a424d88b.jpg)


Jun Li received the M.S and Ph.D. degrees in information and communication engineering from the National University of Defense Technology, Changsha, China. 

He is currently a Professor with the College of Electronic Science and Technology, National University of Defense Technology. His research interests include management and analysis of big data and spatial information system. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/5e4e9d0c6ffe9dc59b2623bcea0ed63408f705e082ad9eb093490e2b960ef3e4.jpg)


Hao Chen received the Ph.D. degree in information and communication engineering from the National University of Defense Technology, Changsha, China, in 2010. 

He is currently a Professor with the College of Electronic Science and Technology, National University of Defense Technology. His research interests include data mining, machine learning, and evolutionary computation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-14/afb552b0-af27-4310-9d19-a632ac5fdd85/8b06a043c1b0c0330cfebb9ed57c424607183099c037bae835552a22c2f62b27.jpg)


Jiangjiang Wu received the M.S and Ph.D. degrees in computer science and technology from the National University of Defense Technology, Changsha, China. 

He is currently an Associate Professor with the College of Electronic Science and Technology, National University of Defense Technology. His research interests include analysis and storage management of big data. 