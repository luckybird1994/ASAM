## ASAM: Boosting Segment Anything Model with Adversarial Tuning, CVPR2024

<font size=7><div align='center'>ASAM: Boosting Segment Anything Model with Adversarial Tuning</div></font>

<div align=center><img width="70%" src=imgs/framework.png/></div>

## Abstract

In the evolving landscape of computer vision, foundation models have emerged as pivotal tools, exhibiting exceptional adaptability to a myriad of tasks. Among these, the Segment Anything Model (SAM) by Meta AI has distinguished itself in image segmentation. However, SAM, like its counterparts, encounters limitations in specific niche applications, prompting a quest for enhancement strategies that do not compromise its inherent capabilities. This paper introduces ASAM, a novel methodology that amplifies SAM's performance through adversarial tuning. We harness the potential of natural adversarial examples, inspired by their successful implementation in natural language processing (NLP). By utilizing a stable diffusion model, we augment a subset (1\%) of the SA-1B dataset, generating adversarial instances that are more representative of natural variations rather than conventional imperceptible perturbations. Our approach maintains the photorealism of adversarial examples and ensures alignment with original mask annotations, thereby preserving the integrity of the segmentation task. The fine-tuned ASAM demonstrates significant improvements across a diverse range of segmentation tasks without necessitating additional data or architectural modifications. The results of our extensive evaluations confirm that ASAM establishes new benchmarks in segmentation tasks, thereby contributing to the advancement of foundational models in computer vision.

## Performance
<div align=center><img width="100%" src=imgs/performance.png/></div>

## News
- [x] [2024.02.27] Paper is accepted by CVPR2024 and GitHub repo is created.




