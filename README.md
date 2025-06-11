# Cost of DP for Deep Learning-Based Trajectory Generation

[![Unittests](../../actions/workflows/unittests.yml/badge.svg)](../../actions/workflows/unittests.yml)
[![Conda Environment Builds](../../actions/workflows/conda.yml/badge.svg)](../../actions/workflows/conda.yml)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)


Artifacts for paper _"What is the Cost of Differential Privacy for Deep Learning-Based Trajectory Generation?"_.

## Table of Contents

<!-- TOC -->
* [Cost of DP for Deep Learning-Based Trajectory Generation](#cost-of-dp-for-deep-learning-based-trajectory-generation)
  * [Table of Contents](#table-of-contents)
  * [Citation](#citation)
  * [Abstract](#abstract)
  * [Requirements](#requirements)
    * [Hardware](#hardware)
    * [Software](#software)
    * [Storage](#storage)
  * [Setup](#setup)
    * [Pull Repository Including Submodules (for Data)](#pull-repository-including-submodules-for-data)
    * [Install Dependencies](#install-dependencies)
    * [Data](#data)
    * [Parameters](#parameters)
  * [Usage](#usage)
    * [Train Models](#train-models)
      * [Using Configuartions](#using-configuartions)
      * [Using Command Line Arguments](#using-command-line-arguments)
    * [Evaluate Models](#evaluate-models)
    * [View / Visualise Results](#view--visualise-results)
  * [Contact](#contact)
  * [Acknowledgements](#acknowledgements)
  * [References](#references)
  * [License](#license)
<!-- TOC -->

## Citation

If you use the code in this repository, please cite the following paper:

```bibtex
@article{BFN+25,
  title         = {What is the Cost of Differential Privacy for Deep Learning-Based Trajectory Generation?},
  author        = {Buchholz, Erik and Fernandes, Natasha and Nguyen, David D. and Abuadbba, Alsharif and Wang, Shuo and Nepal, Surya and Kanhere, Salil S.},
  year          = {2025},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  eprint        = {XXXX.XXXXX},  % Todo
  url           = {https://arxiv.org/abs/XXXX.XXXXX},  % Todo
  doi           = {XXXX.XXXXX/arXiv.XXXX.XXXXX},  % Todo
  month         = jun,
  note          = {21 pages}
}
```

## Abstract

> While location trajectories offer valuable insights, they also reveal sensitive personal information.
> Differential Privacy (DP) offers formal protection, but achieving a favourable utility-privacy trade-off remains challenging.
> Recent works explore deep learning-based generative models to produce synthetic trajectories.
> However, current models lack formal privacy guarantees and rely on conditional information derived from real data during generation.
> This work investigates the utility cost of enforcing DP in such models, addressing three research questions across two datasets and eleven utility metrics.
> (1) We evaluate how DP-SGD, the standard DP training method for deep learning, affects the utility of state-of-the-art generative models.
> (2) Since DP-SGD is limited to unconditional models, we propose a novel DP mechanism for conditional generation that provides formal guarantees and assess its impact on utility. 
> (3) We analyse how model types -- Diffusion, VAE, and GAN -- affect the utility-privacy trade-off.
> Our results show that DP-SGD significantly impacts performance, although some utility remains if the datasets is sufficiently large.
> The proposed DP mechanism improves training stability, particularly when combined with DP-SGD, for unstable models such as GANs and on smaller datasets.
> Diffusion models yield the best utility without guarantees, but with DP-SGD, GANs perform best, indicating that the best non-private model is not necessarily optimal when targeting formal guarantees.
> In conclusion, DP trajectory generation remains a challenging task, and formal guarantees are currently only feasible with large datasets and in constrained use cases.

## Requirements

### Hardware

The final measurements were conducted on a high performance computing (HPC) cluster.
However, the code runs on a standard GPU server and many initial mesurements were conducted on such a server.
Especially the evaluations without DP-SGD are feasible on a standard GPU server.
A GPU is required for training the models - the code assumes existence of a GPU.
The server used for initial measurements was equipped with the following hardware:

- **OS:** Ubuntu 24.04.1 LTS
- **CPU:** AMD EPYC 7763 (64) @ 2.450GHz
- **GPU:** 2x NVIDIA RTX A6000
- **Memory:** 503GiB

On this server, the most computational expensive model (Conditional DiffTraj with DP-SGD) took around _80
hours_ to train for 100'000 iterations.
The non DP-SGD models complete training in less than _20 hours_ for 100'000 iterations, and in less than _4
hours_ for 20'000 iterations.

### Software

- Ubuntu 20.04 or later
- Python 3.12
- Conda or venv
- Required packages (see `requirements.txt` or `environment.yml`)

### Storage

- **Data:** Less than 10GB
- **Code:** Less than 1GB
- **Parameters:** 4.59GB per full run (refer [Parameters](#parameters))

## Setup

### Pull Repository Including Submodules (for Data)

```shell
git clone --recurse-submodules git@github.com:erik-buchholz/CostOfTrajectoryPrivacy.git
```

### Install Dependencies

We provide scripts for both `venv` and `conda` environments.
However, we recommend using `conda` because it defines the Python version to be used.
When using `venv`, make sure to use Python 3.12.

**Conda:** (Will install Python 3.12 and miniconda if not already installed - in that case, sudo might be required)
```shell
./setup_conda.sh
``` 

**Venv:** (Will install Python 3.12 and miniconda if not already installed - in that case, sudo might be required)
```shell
./setup_venv.sh
```

### Data

The data is included through a submodule.
Due to the size of the GeoLife and Porto datasets, they were split up into smaller parts.
To reassemble the datasets, run the following command:
```shell
python data/processed/reconstruct_datasets.py
```
After that, all data required for our experiments should be available.

If you would like to preprocess the raw datasets yourself, please refer to the [README](data/README.md).

### Parameters

The model parameters for all 5 runs used in the paper have been shared on Figshare.
To use them, please unzip the archive and copy the parameters into `parameters/`.

- [Run 01](https://dx.doi.org/10.6084/m9.figshare.28169489)
- [Run 02](https://dx.doi.org/10.6084/m9.figshare.28169525)
- [Run 03](https://dx.doi.org/10.6084/m9.figshare.28169555)
- [Run 04](https://dx.doi.org/10.6084/m9.figshare.29275703)
- [Run 05](https://dx.doi.org/10.6084/m9.figshare.29275709)


## Usage

Make sure that you have completed the [Setup](#setup) steps before running the code.
You need to activate the environment you created in the setup steps before running the code: `conda activate ptg` or
`source venv/bin/activate`.

### Train Models

You can either train models via configuration or by specificying parameters via command line arguments.

#### Using Configuartions

Define a configration as a base class of one the configuation in `evaluation/config/*` and add them to `ALL_CONFIGS` in
`evaluation/config/__init__.py`.
This dictionary contains all configurations that are available from the command line interface.

You can train the models by running the following command:

```shell
python3 -m privtrajgen.evaluation.train --config <config_name> --gpu <gpu_id> 
```

#### Using Command Line Arguments

Each of the three models, DiffTraj, UNetVAE, and UNetGAN, has their own training script in `privtrajgen/run/`.
You can train the models by running the following command:

```shell
python3 -m privtrajgen.run.train_<model_name> --gpu <gpu_id> --<parameter> <value>
```

Use the `--help` flag to see all available parameters for each model.

### Evaluate Models

The evaluation script will load a trained model, generate trajectories, and evaluate the generated trajectories with our metrics.
The results will be saved into `results/`.
Evaluation is only possibel for cases defined via configurations.

You either have to train a model you intend to evaluate, or you can download our parameters (refer to [Parameters](#parameters)).

You can evaluate the models by running the following command:

```shell
python3 -m privtrajgen.evaluation.evaluate --gpu <gpu_id> --config <config_name>
```

### View / Visualise Results

You can recreate the paper figures by using the following notebooks:

- [notebooks/plots.ipynb](notebooks/plots.ipynb): Generate the bar plots from the paper
- [notebooks/tables.ipynb](notebooks/tables.ipynb): Recreate the tables from the paper
- [notebooks/example_figures.ipynb](notebooks/example_figures.ipynb): Generate example figures from the paper

## Contact

**Author:** [Erik Buchholz](https://www.erikbuchholz.de) ([e.buchholz@unsw.edu.au](mailto:e.buchholz@unsw.edu.au))

**Supervision:**

- [Prof. Salil Kanhere](https://salilkanhere.net/)
- [Dr. Surya Nepal](https://people.csiro.au/N/S/Surya-Nepal)

**Involved Researchers:**

- [Dr. Natasha Fernandes](https://researchers.mq.edu.au/en/persons/natasha-fernandes)
- [Dr. David D. Nguyen](https://people.csiro.au/w/s/shuo-wang)
- [Dr. Sharif Abuadbba](https://people.csiro.au/A/S/sharif-abuadbba)

## Acknowledgements

The authors would like to thank the University of New South Wales,
the Commonwealth of Australia, and the Cybersecurity Cooperative Research Centre Limited, whose activities are partially
funded by the Australian Government’s Cooperative Research Centres Programme, for their support.

## References

We would like to thank the following authors for sharing their datasets and/or code that we used in our research:

1. Y. Zhu, Y. Ye, S. Zhang, X. Zhao, and J. J. Q. Yu,
   "DiffTraj: generating GPS trajectory with diffusion probabilistic model,"
   in Advances in Neural Information Processing Systems, in 1, vol. 23. New Orleans,
   USA: Curran Associates, Inc., 2023, p. 21. doi: https://doi.org/10.5555/3666122.3668965
2. E. Buchholz, A. Abuadbba, S. Wang, S. Nepal, and S. S. Kanhere,
   "SoK: can trajectory generation combine privacy and utility?," PoPETS, vol. 2024, no. 3, pp. 75--93, Jul. 2024,
   doi: https://doi.org/10.56553/popets-2024-0068
3. D. Whittenbury, "DS-2: sampling and visualising the von mises-fisher distribution in p dimensions,"
   dlwhittenbury.com. Accessed: Nov. 18, 2024.
   Available: https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html
4. R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer,
   "High-resolution image synthesis with latent diffusion models,"
   in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA: IEEE, Jun. 2022,
   pp. 10674–10685. doi: https://doi.org/10.1109/CVPR52688.2022.01042
5. Rao, J., Gao, S.*, Kang, Y. and Huang, Q. (2020).
   "LSTM-TrajGAN: A Deep Learning Approach to Trajectory Privacy Protection."
   In the Proceedings of the 11th International Conference on Geographic Information Science (GIScience 2021),
   12:1--12:17, doi: https://doi.org/10.4230/LIPIcs.GIScience.2021.I.12
6. Dingqi Yang, Daqing Zhang, V. W. Zheng, and Zhiyong Yu,
   "Modeling user activity preference by leveraging user spatial temporal characteristics in LBSNs,"
   IEEE Trans. Syst., Man, Cybern., Syst., vol. 45, no. 1, pp. 129–142, Jan. 2015,
   doi: https://doi.org/10.1109/TSMC.2014.2327053
7. Y. Zheng, L. Zhang, X. Xie, and W.-Y. Ma,
   "Mining interesting locations and travel sequences from GPS trajectories," in Proceedings of the 18th international
   conference on World wide web, in WWW ’09. New York, NY, USA: Association for Computing Machinery, Apr. 2009,
   pp. 791–800. doi: https://doi.org/10.1145/1526709.1526816
8. L. Moreira-Matias, M. Ferreira, J. Mendes-Moreira, L. L., and J. J.,
   "Porto taxi - taxi service trajectory - prediction challenge, ECML PKDD 2015."
   UCI Machine Learning Repository, 2015. doi: https://doi.org/10.24432/C55W25

## License

CSIRO Open Source Software Licence Agreement (variation of the BSD / MIT License)

Copyright (c) 2025, Commonwealth Scientific and Industrial Research Organisation (CSIRO) ABN 41 687 119 230.

All rights reserved. CSIRO is willing to grant you a licence to these datasets on the following terms,
except where otherwise indicated for third party material.
Redistribution and use of this software in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of CSIRO nor the names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission of CSIRO.

EXCEPT AS EXPRESSLY STATED IN THIS AGREEMENT AND TO THE FULL EXTENT PERMITTED BY APPLICABLE LAW,
THE SOFTWARE IS PROVIDED "AS-IS". CSIRO MAKES NO REPRESENTATIONS, WARRANTIES OR CONDITIONS OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY REPRESENTATIONS, WARRANTIES
OR CONDITIONS REGARDING THE CONTENTS OR ACCURACY OF THE SOFTWARE, OR OF TITLE, MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, THE ABSENCE OF LATENT OR OTHER DEFECTS,
OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT DISCOVERABLE.
TO THE FULL EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL CSIRO BE LIABLE ON ANY LEGAL THEORY(INCLUDING,
WITHOUT LIMITATION, IN AN ACTION FOR BREACH OF CONTRACT, NEGLIGENCE OR OTHERWISE) FOR ANY CLAIM, LOSS,
DAMAGES OR OTHER LIABILITY HOWSOEVER INCURRED. WITHOUT LIMITING THE SCOPE OF THE PREVIOUS SENTENCE
THE EXCLUSION OF LIABILITY SHALL INCLUDE: LOSS OF PRODUCTION OR OPERATION TIME, LOSS,
DAMAGE OR CORRUPTION OF DATA OR RECORDS; OR LOSS OF ANTICIPATED SAVINGS, OPPORTUNITY, REVENUE, PROFIT OR GOODWILL,
OR OTHER ECONOMIC LOSS; OR ANY SPECIAL, INCIDENTAL, INDIRECT, CONSEQUENTIAL, PUNITIVE OR EXEMPLARY DAMAGES,
ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT, ACCESS OF THE SOFTWARE OR ANY OTHER DEALINGS WITH THE SOFTWARE,
EVEN IF CSIRO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH CLAIM, LOSS, DAMAGES OR OTHER LIABILITY.
APPLICABLE LEGISLATION SUCH AS THE AUSTRALIAN CONSUMER LAW MAY APPLY REPRESENTATIONS, WARRANTIES,
OR CONDITIONS, OR IMPOSES OBLIGATIONS OR LIABILITY ON CSIRO THAT CANNOT BE EXCLUDED, RESTRICTED
OR MODIFIED TO THE FULL EXTENT SET OUT IN THE EXPRESS TERMS OF THIS CLAUSE ABOVE "CONSUMER GUARANTEES".
TO THE EXTENT THAT SUCH CONSUMER GUARANTEES CONTINUE TO APPLY, THEN TO THE FULL EXTENT PERMITTED BY THE APPLICABLE
LEGISLATION, THE LIABILITY OF CSIRO UNDER THE RELEVANT CONSUMER GUARANTEE IS LIMITED (WHERE PERMITTEDAT CSIRO'S
OPTION) TO ONE OF FOLLOWING REMEDIES OR SUBSTANTIALLY EQUIVALENT REMEDIES:
(a)               THE REPLACEMENT OF THE SOFTWARE, THE SUPPLY OF EQUIVALENT SOFTWARE,
OR SUPPLYING RELEVANT SERVICES AGAIN;
(b)               THE REPAIR OF THE SOFTWARE;
(c)               THE PAYMENT OF THE COST OF REPLACING THE SOFTWARE, OF ACQUIRING EQUIVALENT SOFTWARE,
HAVING THE RELEVANT SERVICES SUPPLIED AGAIN, OR HAVING THE SOFTWARE REPAIRED.
IN THIS CLAUSE, CSIRO INCLUDES ANY THIRD PARTY AUTHOR OR OWNER OF ANY PART OF THE SOFTWARE
OR MATERIAL DISTRIBUTED WITH IT. CSIRO MAY ENFORCE ANY RIGHTS ON BEHALF OF THE RELEVANT THIRD PARTY.

Third Party Components

The following third party components are distributed with the Software.
You agree to comply with the licence terms for these components as part of accessing the Software.
Other third party software may also be identified in separate files distributed with the Software.

___________________________________________________________________
DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model

Location: `models/diffTraj`

Reference:
Y. Zhu, Y. Ye, S. Zhang, X. Zhao, and J. J. Q. Yu,
"DiffTraj: generating GPS trajectory with diffusion probabilistic model,"
in Advances in Neural Information Processing Systems, in 1, vol. 23. New Orleans,
USA: Curran Associates, Inc., 2023, p. 21. doi: https://doi.org/10.5555/3666122.3668965

This software is licensed under the MIT License: https://github.com/Yasoz/DiffTraj/blob/main/LICENSE
___________________________________________________________________
SoK: Can Trajectory Generation Combine Privacy and Utility?

Location: Base framework

Reference:
E. Buchholz, A. Abuadbba, S. Wang, S. Nepal, and S. S. Kanhere,
"SoK: can trajectory generation combine privacy and utility?," PoPETS, vol. 2024, no. 3, pp. 75--93, Jul. 2024,
doi: https://doi.org/10.56553/popets-2024-0068

This software is licensed under the MIT License: https://github.com/erik-buchholz/SoK-TrajGen/blob/main/LICENCE
___________________________________________________________________
VMF Distribution Sampling

Location `privacy/vmf.py`

Reference:
D. Whittenbury, "DS-2: sampling and visualising the von mises-fisher distribution in p dimensions,"
dlwhittenbury.com. Accessed: Nov. 18, 2024.
Available:

- https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html
- https://github.com/dlwhittenbury/von-Mises-Fisher-Sampling

___________________________________________________________________
Latent Diffusion Models (Used for UNet Architecture)

Location: `models/unet.py` and `models/unet_layers.py`

Reference:
R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer,
"High-resolution image synthesis with latent diffusion models,"
in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA: IEEE, Jun. 2022,
pp. 10674–10685. doi: https://doi.org/10.1109/CVPR52688.2022.01042

This software is licensed under the MIT License: https://github.com/CompVis/latent-diffusion/blob/main/LICENSE
___________________________________________________________________
PrivTrace

Reference:
H. Wang et al., “PrivTrace: differentially private trajectory synthesis by adaptive markov models,”
presented at the 32nd USENIX Security Symposium (USENIX Security 23), Anaheim, CA, USA: USENIX Association, 2023,
pp. 1649–1666. Accessed: May 02, 2025.
[Online]. Available: https://www.usenix.org/conference/usenixsecurity23/presentation/wang-haiming
___________________________________________________________________
