# Art.AI

**Team members:** Maika Nishiguchi, Seho Kwak, and Rachel Yang

![](cs152.jpg)

## Project Description

The goal of our project is to train a neural network to apply the style of one image to another image. The network will be trained on a large dataset of style images and content images, and it will learn to transfer the style of the style image to the content image. The resulting model will be capable of transforming a simple photo into a masterpiece painting, for example. To add on to this, we want to train a Generative Adversarial Network (GAN) to take the style of one image and generate new images in the same style. The model will be capable of generating new images that are similar in style to the images in the training set.

## Project Goals

1. Collect a large dataset of style images and content images
2. Train a neural network to transfer the style of a style image to a content image
3. Train a GAN on the collected dataset 
4. Evaluate the quality of the generated images using standard metrics
5. Compare the results to other state-of-the-art models for style transfer
6. Explore the potential of using the model for applications such as creating custom art pieces or enhancing the visual appeal of existing photos or videos

## Project Scope
The goal of this project is to train a neural network to create and transfer styles of images

## Introduction Outline
#### Five to Ten Sentence Outline:
Paragraph 1. Introductory paragraph: 
- The task of generating new images in the style of a specific artist or artwork is challenging as it requires an understanding of the unique characteristics and techniques used by the artist. Our solution to this challenge is to train a neural network on a large dataset of images in the desired style.

Paragraph 2. Background paragraph: 
- Prior work in this field has focused on using transfer learning from pre-trained models, however, these models are not always able to capture the fine-grained details of a specific style. Our approach will involve training a neural network from scratch to overcome this limitation.

Paragraph 3. Transition paragraph: 
- By training a neural network from scratch, we can ensure that it has a deeper understanding of the desired style and is able to generate more accurate results.

Paragraph 4. Details paragraph: 
- The project will involve collecting a large dataset of images in the desired style, preprocessing the data, and training a neural network using a generative adversarial network (GAN) architecture. We will evaluate the quality of the generated images using standard metrics such as inception score and Fréchet Inception Distance (FID).

Paragraph 5. Assessment paragraph: 
- Our expected results are high-quality generated images in the style of the desired artist or artwork. These results support the conclusion that neural networks can be trained to generate new images in a specific style and have potential applications in areas such as digital art and advertising. The structure of the paper will include an introduction, background, methodology, results, and conclusion sections.

#### Ethical Sweep:
- Should we even be doing this? This project is focused on creating new artwork using artificial intelligence which may raise discussions on AI having a negative impact towards artists. 
- What might be the accuracy of a simple non-ML alternative? A simple non-ML alternative, such as manual painting or drawing, may not be able to generate as many high-quality images in a specific style as a neural network.
- What processes will we use to handle appeals/mistakes? If any mistakes are made during the project, they will be addressed and corrected as needed.
- How diverse is our team? Our team consists of three individuals with diverse backgrounds and experiences.

## Related Works 
1. "Neural Style Transfer: A Review" by Yongcheng Jing, Yezhou Yang. Zunlei Feng,  Jingwen Ye, Yizhou Yu, and Mingli Song
    - The review covers both traditional style transfer techniques, including texture synthesis, and deep learning-based approaches. It also discusses various optimization methods and loss functions used in neural style transfer, as well as techniques for improving the efficiency of the process. The paper concludes by examining the potential applications of neural style transfer and the challenges that must be overcome to make it a viable tool in various domains, including fashion, film, and art.
    - link: https://www.researchgate.net/publication/333702745_Neural_Style_Transfer_A_Review
    
2. "Deep Generative Adversarial Networks for Image-to-Image Translation: A Review" by Aziz Alotaibi
    - This article provides a comprehensive overview of image-to-image translation based on GAN algorithms and its variants. It also discusses and analyzes current state-of-the-art image-to-image translation techniques that are based on multimodal and multidomain representations. Finally, open issues and future research directions utilizing reinforcement learning and three-dimensional (3D) modal translation are summarized and discussed.
  - link: https://doi.org/10.3390/sym12101705
   
3. "GAN computers generate arts? A survey on visual arts, music, and literary text generation using generative adversarial network" by Sakib Shahriar
    - This survey takes a comprehensive look at the recent works using GANs for generating visual arts, music, and literary text. A performance comparison and description of the various GAN architecture are also presented. Finally, some of the key challenges in GAN-based art generation are highlighted along with recommendations for future work.
    - link: https://doi.org/10.1016/j.displa.2022.102237
    
4. "A Method for Style Transfer from Artistic Images Based on Depth Extraction Generative Adversarial Network" by Xinying Han, Yang Wu, and Rui Wan
    - The researchers propose a multi-feature extractor to extract color features, texture features, depth features, and shape masks from style images with U-net, multi-factor extractor, fast Fourier transform, and MiDas depth estimation network. At the same time, a self-encoder structure is used as the content extraction network core to generate a network that shares style parameters with the feature extraction network and finally realizes the generation of artwork images in three-dimensional artistic styles. The experimental analysis shows that compared with other advanced methods, DE-GAN-generated images have higher subjective image quality, and the generated style pictures are more consistent with the aesthetic characteristics of real works of art. The quantitative data analysis shows that images generated using the DE-GAN method have better performance in terms of structural features, image distortion, image clarity, and texture details. 
    - link: https://doi.org/10.3390/app13020867
    
5. "Neural Style Transfer: A Paradigm Shift for Image-based Artistic Rendering?" by Amir Semmo, Tobias Isenberg, and Jürgen Döllner
    - The authors discuss the potential benefits of neural style transfer, including the ability to create high-quality, artistically styled images quickly and easily, as well as the potential for automating certain aspects of the artistic process. They also explore the limitations and challenges of neural style transfer, such as the difficulty of controlling the output and the potential for over-reliance on pre-existing styles. The paper concludes by proposing future research directions, including the need for a better understanding of the relationship between style and content in neural style transfer, and the development of new techniques for style transfer that address some of its current limitations.
    - link: https://www.researchgate.net/publication/317105787_Neural_Style_Transfer_A_Paradigm_Shift_for_Image-based_Artistic_Rendering

