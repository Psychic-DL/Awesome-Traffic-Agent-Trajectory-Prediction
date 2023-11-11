# Awesome-Traffic-Agent-Trajectory-Prediction
![Version](https://img.shields.io/badge/Version-1.0-ff69b4.svg) ![LastUpdated](https://img.shields.io/badge/LastUpdated-2023.10-lightgrey.svg) ![Topic](https://img.shields.io/badge/Topic-trajectory--prediction-yellow.svg?logo=github) ![Awesome](https://awesome.re/badge.svg)

![image](https://github.com/Psychic-DL/Awesome-Traffic-Agent-Trajectory-Prediction/blob/main/image/8189c63cf7894232e1573be4c217653.png)
![image](https://github.com/Psychic-DL/Awesome-Traffic-Agent-Trajectory-Prediction/blob/main/image/ef4e2adbba8af28f8850b5fa2eab76f.png)
![image](https://github.com/Psychic-DL/Awesome-Traffic-Agent-Trajectory-Prediction/blob/main/image/4e785d2f0c1a1601d1dc25073463af2.png)
![image](https://github.com/Psychic-DL/Awesome-Traffic-Agent-Trajectory-Prediction/blob/main/image/59c497e2e43acb2c0ff6a33244b19a6.png)
![image](https://github.com/Psychic-DL/Awesome-Traffic-Agent-Trajectory-Prediction/blob/main/image/5a250b453aca04ab3402b4d6279b215.png)
![image](https://github.com/Psychic-DL/Awesome-Traffic-Agent-Trajectory-Prediction/blob/main/image/172b1087122e79c8a744c7acc9bea62.png)

This is a list of the latest research materials (datasets, papers and codes) related to traffic agent trajectory prediction. (Continuous update)

**Maintainers: Chaoneng Li (Xidian University)**

**Emails: xdchaonengli@163.com**

Please feel free to pull request to add new resources or send emails to us for questions, discussions and collaborations. We would like to connect more students, teachers, and bigwigs in the field of multi-agent trajectory prediction, and if you would like to do the same, you can add me on WeChat (CN15691969157). Let's create the Trajectory Prediction Community Group together!

******

# Table of Contents

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->
- [**Traditional Methods**](#traditional-methods)
- [**2018 and Before Conference and Journal Papers**](#2018-and-before-conference-and-journal-papers)
	- [Conference Papers](#conference-papers)
	- [Journal Papers](#journal-papers)
	- [Others](#others)
- [**2019 Conference and Journal Papers**](#2019-conference-and-journal-papers)
	- [Conference Papers](#conference-papers-2019)
	- [Journal Papers](#journal-papers-2019)
	- [Others](#others-2019)
- [**2020 Conference and Journal Papers**](#2020-conference-and-journal-papers)
	- [Conference Papers](#conference-papers-2020)
	- [Journal Papers](#journal-papers-2020)
	- [Others](#others-2020)
- [**2021 Conference and Journal Papers**](#2021-conference-and-journal-papers)
	- [Conference Papers](#conference-papers-2021)
	- [Journal Papers](#journal-papers-2021)
	- [Others](#others-2021)
- [**2022 Conference and Journal Papers**](#2022-conference-and-journal-papers)
	- [Conference Papers](#conference-papers-2022)
	- [Journal Papers](#journal-papers-2022)
	- [Others](#others-2022)
- [**2023 Conference and Journal Papers**](#2023-conference-and-journal-papers)
	- [Conference Papers](#conference-papers-2023)
	- [Journal Papers](#journal-papers-2023)
	- [Others](#others-2023)
- [**Related Review Papers**](#related-review-papers)
- [**Datasets**](#datasets)
	- [Vehicles Publicly Available Datasets](#vehicles-publicly-available-datasets)
	- [Pedestrians Publicly Available Datasets](#pedestrians-publicly-available-datasets)
	- [Others Agents Datasets](#others-agents-datasets)
		- [Aircraft](#aircraft)
		- [Ship](#ship)
		- [Hurricane and Animal](#hurricane-and-animal)
	<!-- /TOC -->

******

# Traditional Methods
* Social force model for pedestrian dynamics, Physical review E 1995. [[paper](https://arxiv.org/pdf/cond-mat/9805244.pdf?ref=https://githubhelp.com)]
* Simulating dynamical features of escape panic, Nature 2000. [[paper](https://arxiv.org/pdf/cond-mat/0009448.pdf)] [[code](https://github.com/obisargoni/repastInterSim)]
* Congested traffic states in empirical observations and microscopic simulations, Physical review E 2000. [[paper](https://arxiv.org/pdf/cond-mat/0002177.pdf)]
* A methodology for automated trajectory prediction analysis, AIAA Guidance, Navigation, and Control Conference and Exhibit 2004. [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.2942&rep=rep1&type=pdf)]
* Continuum crowds, ACM Transactions on Graphics (TOG 2006). [[paper](https://www.khoury.neu.edu/home/scooper/index_files/pub/treuille2006continuum.pdf)]
* New Algorithms for Aircraft Intent Inference and Trajectory Prediction, Journal of guidance, control, and dynamics 2007. [[paper](https://sci-hub.hkvisa.net/10.2514/1.26750)]
* Reciprocal Velocity Obstacles for Real-Time Multi-Agent Navigation, ICRA 2008. [[paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.9395&rep=rep1&type=pdf)]
* You’ll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking, ICCV 2009. [[paper](http://vision.cse.psu.edu/courses/Tracking/vlpr12/PellegriniNeverWalkAlone.pdf)]
* Real time trajectory prediction for collision risk estimation between vehicles, International Conference on Intelligent Computer Communication and Processing 2009. [[paper](https://hal.inria.fr/inria-00438624/document)]
* People Tracking with Human Motion Predictions from Social Forces, ICRA 2010. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5509779)]
* Unfreezing the robot: Navigation in dense, interacting crowds, IROS 2010. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5654369)]
* Who are you with and where are you going?, CVPR 2011. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5995468)]
* Social force model with explicit collision prediction, Europhysics Letters 2011. [[paper](https://iopscience.iop.org/article/10.1209/0295-5075/93/68005/pdf)]
* A Machine Learning Approach to Trajectory Prediction, AIAA Guidance, Navigation, and Control (GNC) Conference 2013. [[paper](https://sci-hub.hkvisa.net/10.2514/6.2013-4782)]
* Cyclist Social Force Model at Unsignalized Intersections With Heterogeneous Traffic, IEEE Transactions on Industrial Informatics 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7536132)]
* Walking Ahead: The Headed Social Force Model, PLoS ONE 2017. [[paper](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0169734&type=printable)]
* AutoRVO: Local Navigation with Dynamic Constraints in Dense Heterogeneous Traffic, arXiv preprint arXiv:1804.02915, 2018. [[paper](https://arxiv.org/pdf/1804.02915.pdf)]
* Social force models for pedestrian traffic – state of the art, Transport reviews 2018. [[paper](https://www.researchgate.net/profile/Xu-Chen-67/publication/320872442_Social_force_models_for_pedestrian_traffic_-_state_of_the_art/links/5bce680b4585152b144eac39/Social-force-models-for-pedestrian-traffic-state-of-the-art.pdf)]


# 2018 and Before Conference and Journal Papers
## Conference Papers
* Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks, CVPR 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gupta_Social_GAN_Socially_CVPR_2018_paper.pdf)] [[code](https://github.com/agrimgupta92/sgan)]
* Encoding Crowd Interaction with Deep Neural Network for Pedestrian Trajectory Prediction, CVPR 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Encoding_Crowd_Interaction_CVPR_2018_paper.pdf)] [[code](https://github.com/svip-lab/CIDNN)]
* Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net, CVPR 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf)]
* MX-LSTM: Mixing Tracklets and Vislets to Jointly Forecast Trajectories and Head Poses, CVPR 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hasan_MX-LSTM_Mixing_Tracklets_CVPR_2018_paper.pdf)]
* Long-Term On-Board Prediction of People in Traffic Scenes under Uncertainty, CVPR 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bhattacharyya_Long-Term_On-Board_Prediction_CVPR_2018_paper.pdf)] [[code](https://github.com/apratimbhattacharyya18/onboard_long_term_prediction)]
* R2P2: A ReparameteRized Pushforward Policy for Diverse, Precise Generative Path Forecasting, ECCV 2018. [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Nicholas_Rhinehart_R2P2_A_ReparameteRized_ECCV_2018_paper.pdf)]
* Where Will They Go? Predicting Fine-Grained Adversarial Multi-Agent Motion using Conditional Variational Autoencoders, ECCV 2018. [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Panna_Felsen_Where_Will_They_ECCV_2018_paper.pdf)]
* Generating Comfortable, Safe and Comprehensible Trajectories for Automated Vehicles in Mixed Traffic, International Conference on Intelligent Transportation Systems (ITSC 2018). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8569658)]
* Set-Based Prediction of Pedestrians in Urban Environments Considering Formalized Traffic Rules, ITSC 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8569434)]
* Intention-aware Long Horizon Trajectory Prediction of Surrounding Vehicles using Dual LSTM Networks, ITSC 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8569595)]
* Social Attention: Modeling Attention in Human Crowds, ICRA 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460504)]
* A Data-driven Model for Interaction-Aware Pedestrian Motion Prediction in Object Cluttered Environments, ICRA 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461157)]
* Multimodal Probabilistic Model-Based Planning for Human-Robot Interaction, ICRA 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460766)] [[code](https://github.com/StanfordASL/TrafficWeavingCVAE)]
* GD-GAN: Generative Adversarial Networks for Trajectory Prediction and Group Detection in Crowds, ACCV 2018. [[paper](https://arxiv.org/pdf/1812.07667.pdf)]
* Multi-Modal Trajectory Prediction of Surrounding Vehicles with Maneuver based LSTMs, IEEE Intelligent Vehicles Symposium (IV 2018). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8500493)]
* Sequence-to-Sequence Prediction of Vehicle Trajectory via LSTM Encoder-Decoder Architecture, IV 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8500658)]
* Predicting Trajectories of Vehicles Using Large-Scale Motion Priors, IV 2018. [[paper](http://mssuraj.com/publications/2018_IV_0596.pdf)]
* Road Infrastructure Indicators for Trajectory Prediction, IV 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8500678)]
* Mixed Traffic Trajectory Prediction Using LSTM–Based Models in Shared Space, Annual International Conference on Geographic Information Science 2018. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-319-78208-9_16.pdf)]
* SS-LSTM: A Hierarchical LSTM Model for Pedestrian Trajectory Prediction, WACV 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354239)] [[code](https://github.com/xuehaouwa/SS-LSTM)]
* “Seeing is Believing”: Pedestrian Trajectory Forecasting Using Visual Frustum of Attention, WACV 2018. [[paper](http://irtizahasan.com/WACV_2018_Seeing_is_believing.pdf)]
* Tracking by Prediction: A Deep Generative Model for Mutli-person Localisation and Tracking, WACV 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354232)]
* Context-Aware Trajectory Prediction, International Conference on Pattern Recognition (ICPR 2018). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8545447)]
* Transferable Pedestrian Motion Prediction Models at Intersections, IROS 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593783)]
* Generative Modeling of Multimodal Multi-Human Behavior, IROS 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594393)] [[code](https://github.com/StanfordASL/NHumanModeling)]
* Building Prior Knowledge: A Markov Based Pedestrian Prediction Model Using Urban Environmental Data, International Conference on Control, Automation, Robotics and Vision (ICARCV 2018). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8581368)]
* Cyclist Trajectory Prediction Using Bidirectional Recurrent Neural Networks, Australasian Joint Conference on Artificial Intelligence 2018. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-03991-2_28.pdf)]
* Attention Is All You Need, NIPS 2017. [[paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)]
* Bi-Prediction: Pedestrian Trajectory Prediction Based on Bidirectional LSTM Classification, International Conference on Digital Image Computing: Techniques and Applications (DICTA 2017). [[paper](https://www.researchgate.net/profile/Du-Huynh-2/publication/322001876_Bi-Prediction_Pedestrian_Trajectory_Prediction_Based_on_Bidirectional_LSTM_Classification/links/5c03cef4a6fdcc1b8d5029bb/Bi-Prediction-Pedestrian-Trajectory-Prediction-Based-on-Bidirectional-LSTM-Classification.pdf)]
* Probabilistic Vehicle Trajectory Prediction over Occupancy Grid Map via Recurrent Neural Network, ITSC 2017. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8317943)]
* Natural Vision Based Method for Predicting Pedestrian Behaviour in Urban Environments, ITSC 2017. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8317848)]
* How good is my prediction? Finding a similarity measure for trajectory prediction evaluation, ITSC 2017. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8317825)]
* An LSTM network for highway trajectory prediction, ITSC 2017. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8317913)]
* DESIRE: Distant Future Prediction in Dynamic Scenes with Interacting Agents, CVPR 2017. [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lee_DESIRE_Distant_Future_CVPR_2017_paper.pdf)] [[code](https://github.com/tdavchev/DESIRE)]
* Forecasting Interactive Dynamics of Pedestrians with Fictitious Play, CVPR 2017. [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ma_Forecasting_Interactive_Dynamics_CVPR_2017_paper.pdf)]
* Forecast the Plausible Paths in Crowd Scenes, IJCAI 2017. [[paper](https://www.ijcai.org/proceedings/2017/0386.pdf)]
* What will Happen Next? Forecasting Player Moves in Sports Videos, ICCV 2017. [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Felsen_What_Will_Happen_ICCV_2017_paper.pdf)]
* Using road topology to improve cyclist path prediction, IV 2017. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7995734)]
* Short-term 4D Trajectory Prediction Using Machine Learning Methods, Proc. SID 2017. [[paper](https://www.sesarju.eu/sites/default/files/documents/sid/2017/SIDs_2017_paper_11.pdf)]
* Generating Long-term Trajectories Using Deep Hierarchical Networks, NIPS 2016. [[paper](https://proceedings.neurips.cc/paper/2016/file/fe8c15fed5f808006ce95eddb7366e35-Paper.pdf)]
* Learning Social Etiquette: Human Trajectory Understanding In Crowded Scenes, ECCV 2016. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-319-46484-8_33.pdf)]
* Knowledge Transfer for Scene-Specific Motion Prediction, ECCV 2016. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-319-46448-0_42.pdf)]
* Structural-RNN: Deep Learning on Spatio-Temporal Graphs, CVPR 2016. [[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Jain_Structural-RNN_Deep_Learning_CVPR_2016_paper.pdf)] [[code](https://github.com/asheshjain399/RNNexp)]
* Visual Path Prediction in Complex Scenes with Crowded Moving Objects, CVPR 2016. [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yoo_Visual_Path_Prediction_CVPR_2016_paper.pdf)]
* Social LSTM: Human Trajectory Prediction in Crowded Spaces, CVPR 2016. [[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Alahi_Social_LSTM_Human_CVPR_2016_paper.pdf)] [[code](https://github.com/quancore/social-lstm)]
* Comparison and Evaluation of Pedestrian Motion Models for Vehicle Safety Systems, ITSC 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7795912)]
* Intent-aware long-term prediction of pedestrian motion, ICRA 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7487409)]
* Novel planning-based algorithms for human motion prediction, ICRA 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7487505)]
* GLMP-realtime pedestrian path prediction using global and local movement patterns, ICRA 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7487768)]
* Augmented Dictionary Learning for Motion Prediction, ICRA 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7487407&tag=1)]
* Predicting Future Agent Motions for Dynamic Environments, International Conference on Machine Learning and Applications (ICMLA 2016). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7838128)]
* Trajectory prediction of cyclists using a physical model and an artificial neural network, IV 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7535484)]
* STF-RNN: Space Time Features-based Recurrent Neural Network for predicting people next location, IEEE Symposium Series on Computational Intelligence (SSCI 2016). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7849919)]
* Trajectory analysis and prediction for improved pedestrian safety: Integrated framework and evaluations, IV 2015. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7225707)]
* Bayesian intention inference for trajectory prediction with an unknown goal destination, IROS 2015. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7354203)]
* Unsupervised robot learning to predict person motion, ICRA 2015. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7139254)]
* A Controlled Interactive Multiple Model Filter for Combined Pedestrian Intention Recognition and Path Prediction, ITSC 2015. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7313129)]
* Socially-aware Large-scale Crowd Forecasting, CVPR 2014. [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Alahi_Socially-aware_Large-scale_Crowd_2014_CVPR_paper.pdf)]
* Patch to the Future: Unsupervised Visual Prediction, CVPR 2014. [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Walker_Patch_to_the_2014_CVPR_paper.pdf)]
* Online maneuver recognition and multimodal trajectory prediction for intersection assistance using non-parametric regression, IV 2014. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6856480)]
* Pedestrian Path Prediction using Body Language Traits, IV 2014. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6856498)]
* Behavior estimation for a complete framework for human motion prediction in crowded environments, ICRA 2014. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907734)]
* Learning to predict trajectories of cooperatively navigating agents, ICRA 2014. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907442)]
* Pedestrian's Trajectory Forecast in Public Traffic with Artificial Neural Networks, International Conference on Pattern Recognition (ICPR 2014). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6977417)]
* Context-Based Pedestrian Path Prediction, ECCV 2014. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-319-10599-4_40.pdf)]
* Bayesian, Maneuver-Based, Long-Term Trajectory Prediction and Criticality Assessment for Driver Assistance Systems, ITSC 2014. [[paper](https://www.researchgate.net/profile/Matthias-Schreier/publication/266954831_Bayesian_Maneuver-Based_Long-Term_Trajectory_Prediction_and_Criticality_Assessment_for_Driver_Assistance_Systems/links/543fb6250cf2be1758cf3c39/Bayesian-Maneuver-Based-Long-Term-Trajectory-Prediction-and-Criticality-Assessment-for-Driver-Assistance-Systems.pdf)]
* Trajectory generator for autonomous vehicles in urban environments, ICRA 2013. [[paper](https://hal.inria.fr/file/index/docid/789760/filename/ICRA_Perez_et_al_2360.pdf)]
* Vehicle trajectory prediction based on motion model and maneuver recognition, IROS 2013. [[paper](https://hal.archives-ouvertes.fr/hal-00881100/document)]
* Predictive maneuver evaluation for enhancement of Car-to-X mobility data, IV 2012. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6232217)]
* Probabilistic trajectory prediction with Gaussian mixture models, IV 2012. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6232277)]
* Exploiting map information for driver intention estimation at road intersections, IV 2011. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5940452)]
* Trajectory Prediction: Learning to Map Situations to Robot Trajectories, ICML 2009. [[paper](https://dl.acm.org/doi/pdf/10.1145/1553374.1553433)]
* Monte Carlo based Threat Assessment: Analysis and Improvements, IV 2007. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4290120)]
* Gaussian Processes in Machine Learning, Summer school on machine learning 2003. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-540-28650-9_4.pdf)]

## Journal Papers
* Soft + Hardwired Attention: An LSTM Framework for Human Trajectory Prediction and Abnormal Event Detection, Neural networks 2018. [[paper](https://arxiv.org/pdf/1702.05552.pdf?ref=https://githubhelp.com)]
* Long-term path prediction in urban scenarios using circular distributions, Image and Vision Computing 2018. [[paper](https://reader.elsevier.com/reader/sd/pii/S0262885617301853?token=DAD7B9F10835E05341405E75C5AB9F8F114FE99410544AD2BB4EFAA23BFC99D63EA8811C4A8C4F679593A61D0D3E35B6&originRegion=eu-west-1&originCreation=20220509082210)]
* An Efficient Algorithm for Optimal Trajectory Generation for Heterogeneous Multi-Agent Systems in Non-Convex Environments, RAL 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8260912)]
* Network-Wide Vehicle Trajectory Prediction in Urban Traffic Networks using Deep Learning, Transportation Research Record 2018. [[paper](https://www.researchgate.net/profile/Seongjin-Choi-2/publication/327524033_Network-Wide_Vehicle_Trajectory_Prediction_in_Urban_Traffic_Networks_using_Deep_Learning/links/5e3a123e458515072d8015d2/Network-Wide-Vehicle-Trajectory-Prediction-in-Urban-Traffic-Networks-using-Deep-Learning.pdf)]
* Intent Prediction of Pedestrians via Motion Trajectories Using Stacked Recurrent Neural Networks, IEEE Transactions on Intelligent Vehicles 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8481390)]
* How Would Surround Vehicles Move? A Unified Framework for Maneuver Classification and Motion Prediction, IEEE Transactions on Intelligent Vehicles 2018. [[paper](https://ieeexplore.ieee.org/abstract/document/8286935)]
* Pedestrian Path, Pose, and Intention Prediction Through Gaussian Process Dynamical Models and Pedestrian Activity Recognition, TITS 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8370119)]
* Dictionary-based Fidelity Measure for Virtual Traffic, TVCG 2018. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8481568)]
* Realistic Data-Driven Traffic Flow Animation Using Texture Synthesis, TVCG 2017. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7814314)]
* Vehicle Trajectory Prediction by Integrating Physics- and Maneuver-Based Approaches Using Interactive Multiple Models, IEEE Transactions on Industrial Electronics 2017. [[paper](https://www.researchgate.net/profile/Jianqiang-Wang/publication/321738692_Vehicle_Trajectory_Prediction_by_Integrating_Physics-_and_Maneuver-Based_Approaches_Using_Interactive_Multiple_Models/links/5fcde8c445851568d1469e52/Vehicle-Trajectory-Prediction-by-Integrating-Physics-and-Maneuver-Based-Approaches-Using-Interactive-Multiple-Models.pdf)]
* Real-Time Certified Probabilistic Pedestrian Forecasting, RAL 2017. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7959047)]
* Deep Learning Driven Visual Path Prediction from a Single Image, TIP 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7576681)]
* Age and Group-driven PedestrianBehaviour: from Observations to Simulations, Collective Dynamics 2016. [[paper](https://collective-dynamics.eu/index.php/cod/article/view/A3/5)]
* An Integrated Approach to Maneuver-Based Trajectory Prediction and Criticality Assessment in Arbitrary Road Environments, TITS 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7412746)]
* Trajectory Data and Flow Characteristics of Mixed Traffic, Transportation Research Record 2015. [[paper](https://www.researchgate.net/profile/Gowri-Asaithambi/publication/284708700_Trajectory_Data_and_Flow_Characteristics_of_Mixed_Traffic/links/5710718008ae68dc79097605/Trajectory-Data-and-Flow-Characteristics-of-Mixed-Traffic.pdf)]
* Predicting and recognizing human interactions in public spaces, Journal of Real-Time Image Processing 2015. [[paper](https://fabiopoiesi.github.io/files/papers/journals/2014_JRTIP_PredictingRecognizingInteractionsPublic_Poiesi_Cavallaro.pdf)]
* Learning Collective Crowd Behaviors with Dynamic Pedestrian-Agents, International Journal of Computer Vision 2015. [[paper](https://dspace.mit.edu/bitstream/handle/1721.1/103360/11263_2014_735_ReferencePDF.pdf?sequence=1&isAllowed=y)]
* Real-Time Predictive Modeling and Robust Avoidance of Pedestrians with Uncertain, Changing Intentions, Algorithmic Foundations of Robotics XI 2015. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-319-16595-0_10.pdf)]
* BRVO: Predicting pedestrian trajectories using velocity-space reasoning, International Journal of Robotics Research 2015. [[paper](https://www.cs.cityu.edu.hk/~rynson/papers/ijrr15.pdf)]
* Learning intentions for improved human motion prediction, Robotics and Autonomous Systems 2014. [[paper](https://www.techunited.nl/media/images/Kwalificatie%20materiaal%202014/Elfring_2014.pdf)]
* A Self-Adaptive Parameter Selection Trajectory Prediction Approach via Hidden Markov Models, TITS 2014. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6918501)]
* TraPlan: An Effective Three-in-One Trajectory-Prediction Model in Transportation Networks, TITS 2014. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6899589)]
* Will the Pedestrian Cross? A Study on Pedestrian Path Prediction, TITS 2013. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6632960)]
* Mobile Agent Trajectory Prediction Using Bayesian Nonparametric Reachability Trees, Infotech@ Aerospace 2011. [[paper](https://dspace.mit.edu/bitstream/handle/1721.1/114899/Aoude_Infotech11.pdf?sequence=1&isAllowed=y)]
* Gaussian Process Dynamical Models for Human Motion, TPAMI 2008. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4359316)]
* A new approach to linear filtering and prediction problems, Journal of Basic Engineering 1960. [[paper](http://160.78.24.2/Public/Kalman/Kalman1960.pdf)]

## Others
* An Evaluation of Trajectory Prediction Approaches and Notes on the TrajNet Benchmark. arXiv preprint arXiv:1805.07663, 2018. [[paper](https://arxiv.org/pdf/1805.07663.pdf)] [[paper](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Becker_RED_A_simple_but_effective_Baseline_Predictor_for_the_TrajNet_ECCVW_2018_paper.pdf)]
* Scene-LSTM: A Model for Human Trajectory Prediction, arXiv preprint arXiv:1808.04018, 2018. [[paper](https://arxiv.org/ftp/arxiv/papers/1808/1808.04018.pdf)]
* Convolutional Social Pooling for Vehicle Trajectory Prediction, CVPR Workshops 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w29/Deo_Convolutional_Social_Pooling_CVPR_2018_paper.pdf)] [[code](https://github.com/nachiket92/conv-social-pooling)]
* Convolutional Neural Network for Trajectory Prediction, ECCV Workshops 2018. [[paper](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Nikhil_Convolutional_Neural_Network_for_Trajectory_Prediction_ECCVW_2018_paper.pdf)]
* Group LSTM: Group Trajectory Prediction in Crowded Scenarios, ECCV Workshops 2018. [[paper](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Bisagno_Group_LSTM_Group_Trajectory_Prediction_in_Crowded_Scenarios_ECCVW_2018_paper.pdf)]
* Are they going to cross? a benchmark dataset and baseline for pedestrian crosswalk behavior, ICCV Workshops 2017. [[paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w3/Rasouli_Are_They_Going_ICCV_2017_paper.pdf)] [[website](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)]
* Human Trajectory Prediction using Spatially aware Deep Attention Models, arXiv preprint arXiv:1705.09436, 2017. [[paper](https://arxiv.org/pdf/1705.09436.pdf)]
* Modeling Spatial-Temporal Dynamics of Human Movements for Predicting Future Trajectories, AAAI Workshops 2015. [[paper](https://www.diva-portal.org/smash/get/diva2:808848/FULLTEXT01.pdf)]

# 2019 Conference and Journal Papers
## Conference Papers 2019
* MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction, Conference on Robot Learning (CoRL 2019). [[paper](https://arxiv.org/pdf/1910.05449.pdf)]
* Generating Multi-Agent Trajectories using Programmatic Weak Supervision, ICLR 2019. [[paper](https://arxiv.org/pdf/1803.07612.pdf)] [[code](https://github.com/ezhan94/multiagent-programmatic-supervision)]
* Stochastic Prediction of Multi-Agent Interactions from Partial Observations, ICLR 2019. [[paper](https://arxiv.org/pdf/1902.09641.pdf)]
* TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents, AAAI 2019. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4569/4447)] [[code](https://github.com/huang-xx/TrafficPredict)]
* Data-Driven Crowd Simulation with Generative Adversarial Networks, International Conference on Computer Animation and Social Agents (CASA 2019). [[paper](https://dl.acm.org/doi/pdf/10.1145/3328756.3328769)] [[code](https://github.com/amiryanj/crowdGAN)]
* RobustTP: End-to-End Trajectory Prediction for Heterogeneous Road-Agents in Dense Traffic with Noisy Sensor Inputs, ACM Computer Science in Cars Symposium (CSCS 2019). [[paper](https://dl.acm.org/doi/pdf/10.1145/3359999.3360495)] [[code](https://github.com/rohanchandra30/TrackNPred)]
* Which Way Are You Going? Imitative Decision Learning for Path Forecasting in Dynamic Scenes, CVPR 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Which_Way_Are_You_Going_Imitative_Decision_Learning_for_Path_CVPR_2019_paper.pdf)]
* Multi-Agent Tensor Fusion for Contextual Trajectory Prediction, CVPR 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Multi-Agent_Tensor_Fusion_for_Contextual_Trajectory_Prediction_CVPR_2019_paper.pdf)] [[code](https://github.com/programmingLearner/MATF-architecture-details)]
* Peeking into the Future: Predicting Future Person Activities and Locations in Videos, CVPR 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Peeking_Into_the_Future_Predicting_Future_Person_Activities_and_Locations_CVPR_2019_paper.pdf)] [[code](https://github.com/google/next-prediction)] [[website](https://next.cs.cmu.edu/)]
* SoPhie: An Attentive GAN for Predicting Paths Compliant to Social and Physical Constraints, CVPR 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sadeghian_SoPhie_An_Attentive_GAN_for_Predicting_Paths_Compliant_to_Social_CVPR_2019_paper.pdf)] [[code](https://github.com/coolsunxu/sophie)]
* SR-LSTM: State Refinement for LSTM towards Pedestrian Trajectory Prediction, CVPR 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_SR-LSTM_State_Refinement_for_LSTM_Towards_Pedestrian_Trajectory_Prediction_CVPR_2019_paper.pdf)] [[code](https://github.com/zhangpur/SR-LSTM)]
* TraPHic: Trajectory Prediction in Dense and Heterogeneous Traffic Using Weighted Interactions, CVPR 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chandra_TraPHic_Trajectory_Prediction_in_Dense_and_Heterogeneous_Traffic_Using_Weighted_CVPR_2019_paper.pdf)] [[code](https://github.com/BenMSK/trajectory_prediction_TraPHic)]
* Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction, CVPR 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Makansi_Overcoming_Limitations_of_Mixture_Density_Networks_A_Sampling_and_Fitting_CVPR_2019_paper.pdf)] [[code](https://github.com/lmb-freiburg/Multimodal-Future-Prediction)]
* Argoverse: 3D Tracking and Forecasting with Rich Maps, CVPR 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.pdf)] [[code](https://github.com/argoai/argoverse-api)]
* Diverse Generation for Multi-agent Sports Games, CVPR 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yeh_Diverse_Generation_for_Multi-Agent_Sports_Games_CVPR_2019_paper.pdf)]
* Looking to Relations for Future Trajectory Forecast, ICCV 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Looking_to_Relations_for_Future_Trajectory_Forecast_ICCV_2019_paper.pdf)]
* Analyzing the Variety Loss in the Context of Probabilistic Trajectory Prediction, ICCV 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thiede_Analyzing_the_Variety_Loss_in_the_Context_of_Probabilistic_Trajectory_ICCV_2019_paper.pdf)]
* The Trajectron: Probabilistic Multi-Agent Trajectory Modeling With Dynamic Spatiotemporal Graphs, ICCV 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ivanovic_The_Trajectron_Probabilistic_Multi-Agent_Trajectory_Modeling_With_Dynamic_Spatiotemporal_Graphs_ICCV_2019_paper.pdf)] [[code](https://github.com/StanfordASL/Trajectron)]
* Joint Prediction for Kinematic Trajectories in Vehicle-Pedestrian-Mixed Scenes, ICCV 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bi_Joint_Prediction_for_Kinematic_Trajectories_in_Vehicle-Pedestrian-Mixed_Scenes_ICCV_2019_paper.pdf)]
* STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction, ICCV 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_STGAT_Modeling_Spatial-Temporal_Interactions_for_Human_Trajectory_Prediction_ICCV_2019_paper.pdf)] [[code](https://github.com/huang-xx/STGAT)]
* PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction, ICCV 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf)] [[code](https://github.com/aras62/PIEPredict)]
* A Multi-Vehicle Trajectories Generator to Simulate Vehicle-to-Vehicle Encountering Scenarios, ICRA 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8793776)]
* Multimodal Trajectory Predictions for Autonomous Driving using Deep Convolutional Networks, ICRA 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8793868)] [[code](https://github.com/daeheepark/PathPredictNusc)]
* Force-based Heterogeneous Traffic Simulation for Autonomous Vehicle Testing, ICRA 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8794430)]
* Interaction-aware Multi-agent Tracking and Probabilistic Behavior Prediction via Adversarial Learning, ICRA 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8793661)]
* StarNet: Pedestrian Trajectory Prediction using Deep Neural Network in Star Topology, IROS 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967811)]
* Deep Predictive Autonomous Driving Using Multi-Agent Joint Trajectory Prediction and Traffic Rules, IROS 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967708)]
* Conditional Generative Neural System for Probabilistic Trajectory Prediction, IROS 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967822)]
* Jointly Learnable Behavior and Trajectory Planning for Self-Driving Vehicles, IROS 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967615)]
* INFER: INtermediate representations for FuturE pRediction, IROS 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8968553)] [[code](https://github.com/talsperre/INFER)] [[website](https://talsperre.github.io/INFER/)]
* Stochastic Sampling Simulation for Pedestrian Trajectory Prediction, IROS 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967857)]
* Long-term Prediction of Motion Trajectories Using Path Homology Clusters, IROS 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8968125)]
* Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks, NIPS 2019. [[paper](https://proceedings.neurips.cc/paper/2019/file/d09bf41544a3365a46c9077ebb5e35c3-Paper.pdf)]
* Multiple Futures Prediction, NIPS 2019. [[paper](https://proceedings.neurips.cc/paper/2019/file/86a1fa88adb5c33bd7a68ac2f9f3f96b-Paper.pdf)] [[code](https://github.com/apple/ml-multiple-futures-prediction)]
* Trajectory Prediction by Coupling Scene-LSTM with Human Movement LSTM, International Symposium on Visual Computing (ISVC 2019). [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-33720-9_19.pdf)]
* Pedestrian Trajectory Prediction Using a Social Pyramid, Pacific Rim International Conference on Artificial Intelligence (PRICAI 2019). [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-29911-8_34.pdf)]
* Situation-Aware Pedestrian Trajectory Prediction with Spatio-Temporal Attention Model, Computer Vision Winter Workshop (CVWW 2019). [[paper](https://arxiv.org/pdf/1902.05437.pdf)]
* Location-Velocity Attention for Pedestrian Trajectory Prediction, WACV 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8659060)]
* Coordination and trajectory prediction for vehicle interactions via bayesian generative modeling, IV 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8813821)]
* Wasserstein Generative Learning with Kinematic Constraints for Probabilistic Interactive Driving Behavior Prediction, IV 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8813783)]
* AGen: Adaptable Generative Prediction Networks for Autonomous Driving, IV 2019. [[paper](http://www.cs.cmu.edu/~cliu6/files/iv19-1.pdf)]
* Vehicle Trajectory Prediction at Intersections using Interaction based Generative Adversarial Networks, ITSC 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8916927), [paper](https://www.researchgate.net/profile/Debaditya-Roy-2/publication/337629029_Vehicle_Trajectory_Prediction_at_Intersections_using_Interaction_based_Generative_Adversarial_Networks/links/5de5e6224585159aa45cc76c/Vehicle-Trajectory-Prediction-at-Intersections-using-Interaction-based-Generative-Adversarial-Networks.pdf)]
* GRIP: Graph-based Interaction-aware Trajectory Prediction, Intelligent Transportation Systems Conference (ITSC 2019). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8917228)] [[code](https://github.com/xincoder/GRIP)]
* GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction for Autonomous Driving, arXiv preprint arXiv:1907.07792, 2019. [[paper](https://arxiv.org/pdf/1907.07792.pdf)] [[code](https://github.com/xincoder/GRIP)]
* Pose Based Trajectory Forecast of Vulnerable Road Users, IEEE Symposium Series on Computational Intelligence (SSCI 2019). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9003023)]
* Path Predictions using Object Attributes and Semantic Environment, International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2019). [[paper](https://pdfs.semanticscholar.org/1d36/88ae8738335f6452147de3c2f33bcfbd81b3.pdf)]
* Probabilistic Path Planning using Obstacle Trajectory Prediction, CoDS-COMAD 2019. [[paper](https://dl.acm.org/doi/pdf/10.1145/3297001.3297006)]
* Human Trajectory Prediction using Adversarial Loss, Proceedings of the 19th Swiss Transport Research Conference 2019. [[paper](https://www.strc.ch/2019/Kothari_Alahi.pdf)] [[code](https://github.com/vita-epfl/AdversarialLoss-SGAN)]

## Journal Papers 2019
* A Scalable Framework for Trajectory Prediction, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8658195)]
* Contextual Recurrent Predictive Model for Long-Term Intent Prediction of Vulnerable Road Users, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766889&tag=1)]
* Interactive Trajectory Prediction of Surrounding Road Users for Autonomous Driving Using Structural-LSTM Network, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8848853)]
* A Deep Learning-Based Framework for Intersectional Traffic Simulation and Editing, TVCG. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8600335)]
* Heter-Sim: Heterogeneous Multi-Agent Systems Simulation by Interactive Data-Driven Optimization, TVCG. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8865441)]
* AADS: Augmented Autonomous Driving Simulation using Data-driven Algorithms, SCIENCE ROBOTICS. [[paper](https://arxiv.org/ftp/arxiv/papers/1901/1901.07849.pdf)]
* Learning Generative Socially Aware Models of Pedestrian Motion, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8760356)]
* Pedestrian Trajectory Prediction in Extremely Crowded Scenarios, Sensors. [[paper](https://www.mdpi.com/1424-8220/19/5/1223/pdf)]
* Human trajectory prediction in crowded scene using social-affinity Long Short-Term Memory, PR. [[paper](https://www.sciencedirect.com/science/article/pii/S0031320319301712)]

## Others 2019
* Joint Interaction and Trajectory Prediction for Autonomous Driving using Graph Neural Networks, arXiv preprint arXiv:1912.07882, 2019. [[paper](https://arxiv.org/pdf/1912.07882.pdf)]
* Learning to Infer Relations for Future Trajectory Forecast, CVPR Workshops 2019. [[paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Choi_Learning_to_Infer_Relations_for_Future_Trajectory_Forecast_CVPRW_2019_paper.pdf)]
* Social Ways: Learning Multi-Modal Distributions of Pedestrian Trajectories, CVPR Workshops 2019. [[paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Precognition/Amirian_Social_Ways_Learning_Multi-Modal_Distributions_of_Pedestrian_Trajectories_With_GANs_CVPRW_2019_paper.pdf)] [[code](https://github.com/crowdbotp/socialways)]
* Social and Scene-Aware Trajectory Prediction in Crowded Spaces, ICCV Workshops 2019. [[paper](https://arxiv.org/pdf/1909.08840.pdf)] [[code](https://github.com/Oghma/sns-lstm/)]
* Probabilistic Trajectory Prediction for Autonomous Vehicles with Attentive Recurrent Neural Process, arXiv preprint arXiv:1910.08102, 2019. [[paper](https://arxiv.org/pdf/1910.08102.pdf)]
* Stochastic Trajectory Prediction with Social Graph Network, arXiv preprint arXiv:1907.10233, 2019. [[paper](https://arxiv.org/pdf/1907.10233.pdf)]

# 2020 Conference and Journal Papers
## Conference Papers 2020
* Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction, ECCV 2020. [[paper](https://arxiv.org/pdf/2005.08514.pdf)] [[code](https://github.com/Majiker/STAR)]
* AutoTrajectory: Label-Free Trajectory Extraction and Prediction from Videos Using Dynamic Points, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58601-0_38.pdf)]
* PiP: Planning-Informed Trajectory Prediction for Autonomous Driving, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58589-1_36.pdf)] [[code](https://github.com/Haoran-SONG/PiP-Planning-informed-Prediction)]
* SMART: Simultaneous Multi-Agent Recurrent Trajectory Prediction, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58583-9_28.pdf)]
* Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58523-5_40.pdf)] [[code](https://github.com/StanfordASL/Trajectron-plus-plus)]
* SimAug: Learning Robust Representations from Simulation for Trajectory Prediction, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58601-0_17.pdf)] [[code](https://next.cs.cmu.edu/simaug/)]
* Diverse and Admissible Trajectory Forecasting Through Multimodal Context Understanding, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58621-8_17.pdf)] [[code](https://github.com/kami93/CMU-DATF)]
* It Is Not the Journey But the Destination: Endpoint Conditioned Trajectory Prediction, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58536-5_45.pdf)] [[code](https://github.com/HarshayuGirase/Human-Path-Prediction)]
* How Can I See My Future? FvTraj: Using First-Person View for Pedestrian Trajectory Prediction, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58571-6_34.pdf)]
* Dynamic and Static Context-Aware LSTM for Multi-agent Motion Prediction, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58589-1_33.pdf)]
* Learning Lane Graph Representations for Motion Forecasting, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58536-5_32.pdf)] [[code](https://github.com/uber-research/LaneGCN)]
* Implicit Latent Variable Model for Scene-Consistent Motion Forecasting, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58592-1_37.pdf)]
* Testing the Safety of Self-driving Vehicles by Simulating Perception and Prediction, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58574-7_19.pdf)]
* Perceive, Predict, and Plan: Safe Motion Planning Through Interpretable Semantic Representations, ECCV 2020. [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-58592-1_25.pdf)]
* Transformer Networks for Trajectory Forecasting, International Conference on Pattern Recognition (ICPR 2020). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9412190)] [[code](https://github.com/FGiuliari/Trajectory-Transformer)]
* DAG-Net: Double Attentive Graph Neural Network for Trajectory Forecasting, ICPR 2020. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9412114)] [[code](https://github.com/alexmonti19/dagnet)]
* TNT: Target-driveN Trajectory Prediction, Conference on Robot Learning (CoRL 2020). [[paper](https://arxiv.org/pdf/2008.08294.pdf)] [[code](https://github.com/Henry1iu/TNT-Trajectory-Predition)]
* Social-VRNN: One-Shot Multi-modal Trajectory Prediction for Interacting Pedestrians, CoRL 2020. [[paper](https://autonomousrobots.nl/docs/20-Brito-CoRL.pdf)] [[code](https://github.com/tud-amr/social_vrnn)]
* Kernel Trajectory Maps for Multi-Modal Probabilistic Motion Prediction, CoRL 2020. [[paper](http://proceedings.mlr.press/v100/zhi20a/zhi20a.pdf)] [[code](https://github.com/wzhi/KernelTrajectoryMaps)]
* MATS: An Interpretable Trajectory Forecasting Representation for Planning and Control, CoRL 2020. [[paper](https://arxiv.org/pdf/2009.07517)] [[code](https://github.com/StanfordASL/MATS)]
* An Attention-Based Interaction-Aware Spatio-Temporal Graph Neural Network for Trajectory Prediction, International Conference on Neural Information Processing (ICONIP 2020). [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-63823-8_5.pdf)]
* OpenTraj: Assessing Prediction Complexity in Human Trajectories Datasets, ACCV 2020. [[paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Amirian_OpenTraj_Assessing_Prediction_Complexity_in_Human_Trajectories_Datasets_ACCV_2020_paper.pdf)] [[code](https://github.com/crowdbotp/OpenTraj)]
* Goal-GAN: Multimodal Trajectory Prediction Based on Goal Position Estimation, ACCV 2020. [[paper](https://arxiv.org/pdf/2010.01114.pdf)] [[code](https://github.com/dendorferpatrick/GoalGAN)]
* Semantic Synthesis of Pedestrian Locomotion, ACCV 2020. [[paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Priisalu_Semantic_Synthesis_of_Pedestrian_Locomotion_ACCV_2020_paper.pdf)] [[code](https://github.com/MariaPriisalu/spl)]
* EvolveGraph: Multi-Agent Trajectory Prediction with Dynamic Relational Reasoning, NIPS 2020. [[paper](https://proceedings.neurips.cc/paper/2020/file/e4d8163c7a068b65a64c89bd745ec360-Paper.pdf)] [[website](https://jiachenli94.github.io/publications/Evolvegraph/)]
* Multi-agent Trajectory Prediction with Fuzzy Query Attention, NIPS 2020. [[paper](https://proceedings.neurips.cc/paper/2020/file/fe87435d12ef7642af67d9bc82a8b3cd-Paper.pdf)] [[code](https://github.com/nitinkamra1992/FQA)]
* Spatio-Temporal Graph Structure Learning for Traffic Forecasting, AAAI 2020. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/5470/5326)]
* GMAN: A Graph Multi-Attention Network for Traffic Prediction, AAAI 2020. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/5477/5333)] [[code](https://github.com/zhengchuanpan/GMAN)]
* CF-LSTM: Cascaded Feature-Based Long Short-Term Networks for Predicting Pedestrian Trajectory, AAAI 2020. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6943/6797)]
* OMuLeT: Online Multi-Lead Time Location Prediction for Hurricane Trajectory Forecasting, AAAI 2020. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/5444/5300)]
* Multimodal Interaction-Aware Trajectory Prediction in Crowded Space, AAAI 2020. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6874/6728)]
* STINet: Spatio-Temporal-Interactive Network for Pedestrian Detection and Trajectory Prediction, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_STINet_Spatio-Temporal-Interactive_Network_for_Pedestrian_Detection_and_Trajectory_Prediction_CVPR_2020_paper.pdf)]
* CoverNet: Multimodal Behavior Prediction using Trajectory Sets, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Phan-Minh_CoverNet_Multimodal_Behavior_Prediction_Using_Trajectory_Sets_CVPR_2020_paper.pdf)]
* TPNet: Trajectory Proposal Network for Motion Prediction, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_TPNet_Trajectory_Proposal_Network_for_Motion_Prediction_CVPR_2020_paper.pdf)]
* Reciprocal Learning Networks for Human Trajectory Prediction, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Reciprocal_Learning_Networks_for_Human_Trajectory_Prediction_CVPR_2020_paper.pdf)]
* MANTRA: Memory Augmented Networks for Multiple Trajectory Prediction, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Marchetti_MANTRA_Memory_Augmented_Networks_for_Multiple_Trajectory_Prediction_CVPR_2020_paper.pdf)]
* Recursive Social Behavior Graph for Trajectory Prediction, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Recursive_Social_Behavior_Graph_for_Trajectory_Prediction_CVPR_2020_paper.pdf)]
* The Garden of Forking Paths: Towards Multi-Future Trajectory Prediction, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liang_The_Garden_of_Forking_Paths_Towards_Multi-Future_Trajectory_Prediction_CVPR_2020_paper.pdf)] [[code](https://next.cs.cmu.edu/multiverse/)]
* Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.pdf)] [[code](https://github.com/abduallahmohamed/Social-STGCNN)]
* VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_VectorNet_Encoding_HD_Maps_and_Agent_Dynamics_From_Vectorized_Representation_CVPR_2020_paper.pdf)] [[code](https://github.com/DQSSSSS/VectorNet)]
* Imitative Non-Autoregressive Modeling for Trajectory Forecasting and Imputation, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qi_Imitative_Non-Autoregressive_Modeling_for_Trajectory_Forecasting_and_Imputation_CVPR_2020_paper.pdf)]
* Collaborative Motion Prediction via Neural Motion Message Passing, CVPR 2020. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Collaborative_Motion_Prediction_via_Neural_Motion_Message_Passing_CVPR_2020_paper.pdf)] [[code](https://github.com/PhyllisH/NMMP)]
* UST: Unifying Spatio-Temporal Context for Trajectory Prediction in Autonomous Driving, IROS 2020. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9340943)]
* Interaction-Aware Trajectory Prediction of Connected Vehicles using CNN-LSTM Networks, Annual Conference of the IEEE Industrial Electronics Society (IECON 2020). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9255162)]
* GISNet:Graph-Based Information Sharing Network For Vehicle Trajectory Prediction, International Joint Conference on Neural Networks (IJCNN 2020). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9206770)]
* Disentangling Human Dynamics for Pedestrian Locomotion Forecasting with Noisy Supervision, WACV 2020. [[paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Mangalam_Disentangling_Human_Dynamics_for_Pedestrian_Locomotion_Forecasting_with_Noisy_Supervision_WACV_2020_paper.pdf)] [[website](https://karttikeya.github.io/publication/plf/)]
* Deep Imitative Models for Flexible Inference, Planning, and Control, ICLR 2020. [[paper](https://openreview.net/pdf?id=Skl4mRNYDr)] [[code](https://github.com/nrhine1/deep_imitative_models)] [[website](https://sites.google.com/view/imitative-models)]
* Diverse Trajectory Forecasting with Determinantal Point Processes, ICLR 2020. [[paper](https://arxiv.org/pdf/1907.04967.pdf)] [[code](https://github.com/Gruntrexpewrus/TrajectoryFor-and-DPP)]
* Trajectory Prediction in Heterogeneous Environment via Attended Ecology Embedding, ACM International Conference on Multimedia 2020. [[paper](http://basiclab.lab.nycu.edu.tw/assets/AEE-GAN_MM2020.pdf)] [[code](https://github.com/Ego2Eco/AEE-GAN)]
* Multiple Trajectory Prediction with Deep Temporal and Spatial Convolutional Neural Networks, IROS 2020. [[paper](http://ras.papercept.net/images/temp/IROS/files/1081.pdf)]
* Probabilistic Multi-modal Trajectory Prediction with Lane Attention for Autonomous Vehicles, IROS 2020. [[paper](https://ieeexplore.ieee.org/abstract/document/9341034/)]
* Lane-Attention: Predicting Vehicles’ Moving Trajectories by Learning Their Attention Over Lanes, IROS 2020. [[paper](https://arxiv.org/pdf/1909.13377.pdf)]
* Interaction-aware Kalman Neural Networks for Trajectory Prediction, IEEE Intelligent Vehicles Symposium (IV 2020). [[paper](https://arxiv.org/pdf/1902.10928.pdf)]
* Multi-Head Attention for Multi-Modal Joint Vehicle Motion Forecasting, ICRA 2020. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9197340)]

## Journal Papers 2020
* TrajVAE: A Variational AutoEncoder model for trajectory generation, Neurocomputing. [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220312017)]
* Social-Aware Pedestrian Trajectory Prediction via States Refinement LSTM, TPAMI. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9261113)]
* Forecasting Trajectory and Behavior of Road-Agents Using Spectral Clustering in Graph-LSTMs, IEEE Robotics and Automation Letters. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9126166)]
* Attention Based Vehicle Trajectory Prediction, IEEE Transactions on Intelligent Vehicles. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9084255)]
* AC-VRNN: Attentive Conditional-VRNN for multi-future trajectory prediction, Computer Vision and Image Understanding. [[paper](https://reader.elsevier.com/reader/sd/pii/S1077314221000898?token=F06466B50D3AE170EC14D460C1AFE91DFE5D61047357252C808857A2BBD4FE4CF2FF3076AD391F842F155CAD2B102C5F&originRegion=eu-west-1&originCreation=20220421024623)]
* PoPPL: Pedestrian Trajectory Prediction by LSTM With Automatic Route Class Clustering, TNNLS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9031707)]
* Real Time Trajectory Prediction Using Deep Conditional Generative Models, IEEE Robotics and Automation Letters. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8957482)]
* Scene Compliant Trajectory Forecast with Agent-Centric Spatio-Temporal Grids, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9000540)]
* What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction, RAL. [[paper](https://arxiv.org/pdf/1903.07933.pdf)] [[code](https://github.com/cschoeller/constant_velocity_pedestrian_motion)]
* Multimodal Deep Generative Models for Trajectory Prediction: A Conditional Variational Autoencoder Approach, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286482)]
* Deep Context Maps: Agent Trajectory Prediction using Location-specific Latent Maps, RAL. [[paper](http://ras.papercept.net/images/temp/IROS/files/2532.pdf)]
* Learning Structured Representations of Spatial and Interactive Dynamics for Trajectory Prediction in Crowded Scenes, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9309332)] [[code](https://github.com/tdavchev/structured-trajectory-prediction), [code](https://github.com/tdavchev/Stochastic-Futures-Prediction)]
* Probabilistic Crowd GAN: Multimodal Pedestrian Trajectory Prediction Using a Graph Vehicle-Pedestrian Attention Network, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9123560)]
* Multimodal Interaction-aware Motion Prediction for Autonomous Street Crossing, International Journal of Robotics Research. [[paper](https://arxiv.org/pdf/1808.06887)]
* Pedestrian Trajectory Prediction Based on Deep Convolutional LSTM Network, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9043898)] [[code](https://github.com/ParadiseCK/DeepConvLstmNet)]
* Multi-Vehicle Collaborative Learning for Trajectory Prediction With Spatio-Temporal Tensor Fusion, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9151374)]
* Multiple Trajectory Prediction of Moving Agents with Memory Augmented Networks, TPAMI. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9138768)]
* Regularizing Neural Networks for Future Trajectory Prediction via Inverse Reinforcement Learning Framework, IET Computer Vision. [[paper](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/iet-cvi.2019.0546)] [[code](https://github.com/d1024choi/traj-pred-irl)]
* Motion trajectory prediction based on a CNN-LSTM sequential model, Science China Information Sciences. [[paper](https://link.springer.com/content/pdf/10.1007/s11432-019-2761-y.pdf)]

## Others 2020
* Scene Gated Social Graph: Pedestrian Trajectory Prediction Based on Dynamic Social Graphs and Scene Constraints, arXiv preprint arXiv:2010.05507, 2020. [[paper](https://arxiv.org/pdf/2010.05507.pdf)]
* Robust Trajectory Forecasting for Multiple Intelligent Agents in Dynamic Scene, arXiv preprint arXiv:2005.13133, 2020. [[paper](https://arxiv.org/pdf/2005.13133.pdf)]
* Map-Adaptive Goal-Based Trajectory Prediction, arXiv preprint arXiv:2009.04450, 2020. [[paper](https://arxiv.org/pdf/2009.04450.pdf)]
* A Spatial-Temporal Attentive Network with Spatial Continuity for Trajectory Prediction, arXiv preprint arXiv:2003.06107, 2020. [[paper](https://arxiv.org/pdf/2003.06107v1.pdf)]
* Trajformer: Trajectory Prediction with Local Self-Attentive Contexts for Autonomous Driving, arXiv preprint arXiv:2011.14910, 2020. [[paper](https://arxiv.org/pdf/2011.14910.pdf)] [[code](https://github.com/Manojbhat09/Trajformer)]
* TPPO: A Novel Trajectory Predictor with Pseudo Oracle, arXiv preprint arXiv:2002.01852, 2020. [[paper](https://arxiv.org/pdf/2002.01852.pdf)]
* Vehicle Trajectory Prediction by Transfer Learning of Semi-Supervised Models, arXiv preprint arXiv:2007.06781, 2020. [[paper](https://arxiv.org/pdf/2007.06781.pdf)]
* Social-WaGDAT: Interaction-aware Trajectory Prediction via Wasserstein Graph Double-Attention Network, arXiv preprint arXiv:2002.06241, 2020. [[paper](https://arxiv.org/pdf/2002.06241.pdf)]
* Trajectory Forecasts in Unknown Environments Conditioned on Grid-Based Plans, arXiv preprint arXiv:2001.00735, 2020. [[paper](https://arxiv.org/pdf/2001.00735.pdf)] [[code](https://github.com/nachiket92/P2T)]
* Multi-modal Trajectory Prediction for Autonomous Driving with Semantic Map and Dynamic Graph Attention Network, NIPS Workshops 2020. [[paper](https://arxiv.org/pdf/2103.16273.pdf)]
* Scene Gated Social Graph: Pedestrian Trajectory Prediction Based on Dynamic Social Graphs and Scene Constraints, arXiv preprint arXiv:2010.05507, 2020. [[paper](https://arxiv.org/pdf/2010.05507v1.pdf)]
* PathGAN: Local Path Planning with Attentive Generative Adversarial Networks, arXiv preprint arXiv:2007.03877, 2020. [[paper](https://arxiv.org/pdf/2007.03877.pdf)] [[code](https://github.com/d1024choi/pathgan_pytorch)]

# 2021 Conference and Journal Papers
## Conference Papers 2021
* Collaborative Uncertainty in Multi-Agent Trajectory Forecasting, NIPS 2021. [[paper](https://proceedings.neurips.cc/paper/2021/file/31ca0ca71184bbdb3de7b20a51e88e90-Paper.pdf)]
* GRIN: Generative Relation and Intention Network for Multi-agent Trajectory Prediction, NIPS 2021. [[paper](https://proceedings.neurips.cc/paper/2021/file/e3670ce0c315396e4836d7024abcf3dd-Paper.pdf)] [[code](https://github.com/longyuanli/GRIN_NeurIPS21)]
* LibCity: An Open Library for Traffic Prediction, SIGSPATIAL 2021. [[paper](https://dl.acm.org/doi/pdf/10.1145/3474717.3483923)] [[code](https://github.com/LibCity/Bigscity-LibCity)]
* Predicting Vehicles Trajectories in Urban Scenarios with Transformer Networks and Augmented Information, IEEE Intelligent Vehicles Symposium (IV 2021). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9575242)]
* Social-STAGE: Spatio-Temporal Multi-Modal Future Trajectory Forecast, ICRA 2021. [[paper](https://arxiv.org/pdf/2011.04853.pdf)]
* AVGCN: Trajectory Prediction using Graph Convolutional Networks Guided by Human Attention, ICRA 2021. [[paper](https://arxiv.org/pdf/2101.05682.pdf)]
* Exploring Dynamic Context for Multi-path Trajectory Prediction, ICRA 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9562034)] [[code](https://github.com/wtliao/DCENet)]
* Pedestrian Trajectory Prediction using Context-Augmented Transformer Networks, ICRA 2021. [[paper](https://www.researchgate.net/publication/346614349_Pedestrian_Trajectory_Prediction_using_Context-Augmented_Transformer_Networks)] [[code](https://github.com/KhaledSaleh/Context-Transformer-PedTraj)]
* Spectral Temporal Graph Neural Network for Trajectory Prediction, ICRA 2021. [[paper](https://arxiv.org/pdf/2106.02930.pdf)]
* Congestion-aware Multi-agent Trajectory Prediction for Collision Avoidance, ICRA 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9560994)] [[code](https://github.com/xuxie1031/CollisionFreeMultiAgentTrajectoryPrediciton)]
* Anticipatory Navigation in Crowds by Probabilistic Prediction of Pedestrian Future Movements, ICRA 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561022)]
* AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_AgentFormer_Agent-Aware_Transformers_for_Socio-Temporal_Multi-Agent_Forecasting_ICCV_2021_paper.pdf)] [[code](https://github.com/Khrylx/AgentFormer)] [[website](https://ye-yuan.com/agentformer/)]
* Likelihood-Based Diverse Sampling for Trajectory Forecasting, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jason_Likelihood-Based_Diverse_Sampling_for_Trajectory_Forecasting_ICCV_2021_paper.pdf)] [[code](https://github.com/JasonMa2016/LDS)]
* MG-GAN: A Multi-Generator Model Preventing Out-of-Distribution Samples in Pedestrian Trajectory Prediction, ICCV 2021. [[paper](https://arxiv.org/pdf/2108.09274.pdf)] [[code](https://github.com/selflein/MG-GAN)]
* Spatial-Temporal Consistency Network for Low-Latency Trajectory Forecasting, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Spatial-Temporal_Consistency_Network_for_Low-Latency_Trajectory_Forecasting_ICCV_2021_paper.pdf)]
* Three Steps to Multimodal Trajectory Prediction: Modality Clustering, Classification and Synthesis, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Three_Steps_to_Multimodal_Trajectory_Prediction_Modality_Clustering_Classification_and_ICCV_2021_paper.pdf)]
* From Goals, Waypoints & Paths To Long Term Human Trajectory Forecasting, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mangalam_From_Goals_Waypoints__Paths_to_Long_Term_Human_Trajectory_ICCV_2021_paper.pdf)] [[code](https://karttikeya.github.io/publication/ynet/)]
* Where are you heading? Dynamic Trajectory Prediction with Expert Goal Examples, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Where_Are_You_Heading_Dynamic_Trajectory_Prediction_With_Expert_Goal_ICCV_2021_paper.pdf)] [[code](https://github.com/JoeHEZHAO/expert_traj)]
* DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_DenseTNT_End-to-End_Trajectory_Prediction_From_Dense_Goal_Sets_ICCV_2021_paper.pdf)]
* Safety-aware Motion Prediction with Unseen Vehicles for Autonomous Driving, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ren_Safety-Aware_Motion_Prediction_With_Unseen_Vehicles_for_Autonomous_Driving_ICCV_2021_paper.pdf)] [[code](https://github.com/xrenaa/Safety-Aware-Motion-Prediction)]
* LOKI: Long Term and Key Intentions for Trajectory Prediction, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Girase_LOKI_Long_Term_and_Key_Intentions_for_Trajectory_Prediction_ICCV_2021_paper.pdf)] [[dataset](https://usa.honda-ri.com/loki)]
* Human Trajectory Prediction via Counterfactual Analysis, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Human_Trajectory_Prediction_via_Counterfactual_Analysis_ICCV_2021_paper.pdf)] [[code](https://github.com/CHENGY12/CausalHTP)]
* Personalized Trajectory Prediction via Distribution Discrimination, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Personalized_Trajectory_Prediction_via_Distribution_Discrimination_ICCV_2021_paper.pdf)] [[code](https://github.com/CHENGY12/DisDis)]
* Unlimited Neighborhood Interaction for Heterogeneous Trajectory Prediction, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Unlimited_Neighborhood_Interaction_for_Heterogeneous_Trajectory_Prediction_ICCV_2021_paper.pdf)] [[code](https://github.com/zhengfang1997/Unlimited-Neighborhood-Interaction-for-Heterogeneous-Trajectory-Prediction)]
* Social NCE: Contrastive Learning of Socially-aware Motion Representations, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Social_NCE_Contrastive_Learning_of_Socially-Aware_Motion_Representations_ICCV_2021_paper.pdf)] [[code](https://github.com/vita-epfl/social-nce)]
* RAIN: Reinforced Hybrid Attention Inference Network for Motion Forecasting, ICCV 2021. [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_RAIN_Reinforced_Hybrid_Attention_Inference_Network_for_Motion_Forecasting_ICCV_2021_paper.pdf)]
* Temporal Pyramid Network for Pedestrian Trajectory Prediction with Multi-Supervision, AAAI 2021. [[paper](https://arxiv.org/pdf/2012.01884.pdf)]
* SCAN: A Spatial Context Attentive Network for Joint Multi-Agent Intent Prediction, AAAI 2021. [[paper](https://arxiv.org/pdf/2102.00109.pdf)]
* Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction, AAAI 2021. [[paper](https://www.aaai.org/AAAI21Papers/AAAI-1677.BaeI.pdf)] [[code](https://github.com/InhwanBae/DMRGCN)]
* MotionRNN: A Flexible Model for Video Prediction with Spacetime-Varying Motions, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_MotionRNN_A_Flexible_Model_for_Video_Prediction_With_Spacetime-Varying_Motions_CVPR_2021_paper.pdf)]
* Multimodal Motion Prediction with Stacked Transformers, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Multimodal_Motion_Prediction_With_Stacked_Transformers_CVPR_2021_paper.pdf)] [[code](https://github.com/decisionforce/mmTransformer)] [[website](https://decisionforce.github.io/mmTransformer/?utm_source=catalyzex.com)]
* SGCN: Sparse Graph Convolution Network for Pedestrian Trajectory Prediction, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_SGCN_Sparse_Graph_Convolution_Network_for_Pedestrian_Trajectory_Prediction_CVPR_2021_paper.pdf)] [[code](https://github.com/shuaishiliu/SGCN)]
* LaPred: Lane-Aware Prediction of Multi-Modal Future Trajectories of Dynamic Agents, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_LaPred_Lane-Aware_Prediction_of_Multi-Modal_Future_Trajectories_of_Dynamic_Agents_CVPR_2021_paper.pdf)]
* Divide-and-Conquer for Lane-Aware Diverse Trajectory Prediction, CVPR 2021. [[paper](https://arxiv.org/pdf/2104.08277.pdf)]
* Euro-PVI: Pedestrian Vehicle Interactions in Dense Urban Centers, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Bhattacharyya_Euro-PVI_Pedestrian_Vehicle_Interactions_in_Dense_Urban_Centers_CVPR_2021_paper.pdf)] [[dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/euro-pvi-dataset)]
* Trajectory Prediction with Latent Belief Energy-Based Model, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Pang_Trajectory_Prediction_With_Latent_Belief_Energy-Based_Model_CVPR_2021_paper.pdf)] [[code](https://github.com/bpucla/lbebm)]
* Shared Cross-Modal Trajectory Prediction for Autonomous Driving, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Shared_Cross-Modal_Trajectory_Prediction_for_Autonomous_Driving_CVPR_2021_paper.pdf)]
* Pedestrian and Ego-vehicle Trajectory Prediction from Monocular Camera, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.pdf)] [[code](https://gitlab.com/lukeN86/pedFutureTracking)]
* Interpretable Social Anchors for Human Trajectory Forecasting in Crowds, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Kothari_Interpretable_Social_Anchors_for_Human_Trajectory_Forecasting_in_Crowds_CVPR_2021_paper.pdf)]
* Introvert: Human Trajectory Prediction via Conditional 3D Attention, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shafiee_Introvert_Human_Trajectory_Prediction_via_Conditional_3D_Attention_CVPR_2021_paper.pdf)]
* MP3: A Unified Model to Map, Perceive, Predict and Plan, CVPR 2021. [[paper](https://arxiv.org/pdf/2101.06806.pdf)]
* TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors, CVPR 2021. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Suo_TrafficSim_Learning_To_Simulate_Realistic_Multi-Agent_Behaviors_CVPR_2021_paper.pdf)]
* Multimodal Transformer Network for Pedestrian Trajectory Prediction, IJCAI 2021. [[paper](https://www.ijcai.org/proceedings/2021/0174.pdf)] [[code](https://github.com/ericyinyzy/MTN_trajectory)]
* Decoder Fusion RNN: Context and Interaction Aware Decoders for Trajectory Prediction, IROS 2021. [[paper](https://arxiv.org/pdf/2108.05814.pdf)]
* Joint Intention and Trajectory Prediction Based on Transformer, IROS 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9636241)]
* Maneuver-based Trajectory Prediction for Self-driving Cars Using Spatio-temporal Convolutional Networks, IROS 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9636875)]
* Multiple Contextual Cues Integrated Trajectory Prediction for Autonomous Driving, IROS 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9476975)]
* MultiXNet: Multiclass Multistage Multimodal Motion Prediction, IEEE Intelligent Vehicles Symposium (IV 2021). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9575718)]
* Trajectory Prediction for Autonomous Driving based on Multi-Head Attention with Joint Agent-Map Representation, IEEE Intelligent Vehicles Symposium (IV 2021). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9576054)]
* Social-IWSTCNN: A Social Interaction-Weighted Spatio-Temporal Convolutional Neural Network for Pedestrian Trajectory Prediction in Urban Traffic Scenarios, IV 2021. [[paper](https://ieeexplore.ieee.org/abstract/document/9575958)]
* Generating Scenarios with Diverse Pedestrian Behaviors for Autonomous Vehicle Testing, Conference on Robot Learning (CoRL 2021). [[paper](https://openreview.net/pdf?id=HTfApPeT4DZ)] [[code](https://github.com/MariaPriisalu/spl)]
* Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals, CoRL 2021. [[paper](https://proceedings.mlr.press/v164/deo22a.html)] [[code](https://github.com/nachiket92/PGP)]
* Learning to Predict Vehicle Trajectories with Model-based Planning, CoRL 2021. [[paper](https://arxiv.org/pdf/2103.04027.pdf)]
* Pose Based Trajectory Forecast of Vulnerable Road Users Using Recurrent Neural Networks, International Conference on Pattern Recognition (ICPR 2021). [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-68763-2_5.pdf)]
* GraphTCN: Spatio-Temporal Interaction Modeling for Human Trajectory Prediction, WACV 2021. [[paper](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_GraphTCN_Spatio-Temporal_Interaction_Modeling_for_Human_Trajectory_Prediction_WACV_2021_paper.pdf)]
* Goal-driven Long-Term Trajectory Prediction, WACV 2021. [[paper](https://openaccess.thecvf.com/content/WACV2021/papers/Tran_Goal-Driven_Long-Term_Trajectory_Prediction_WACV_2021_paper.pdf)]
* Multimodal Trajectory Predictions for Autonomous Driving without a Detailed Prior Map, WACV 2021. [[paper](https://openaccess.thecvf.com/content/WACV2021/papers/Kawasaki_Multimodal_Trajectory_Predictions_for_Autonomous_Driving_Without_a_Detailed_Prior_WACV_2021_paper.pdf)]
* Self-Growing Spatial Graph Network for Context-Aware Pedestrian Trajectory Prediction, IEEE International Conference on Image Processing (ICIP 2021). [[paper](https://arxiv.org/pdf/2012.06320v2.pdf)] [[code](https://github.com/serenetech90/AOL_ovsc)]
* S2TNet: Spatio-Temporal Transformer Networks for Trajectory Prediction in Autonomous Driving, Asian Conference on Machine Learning 2021. [[paper](https://arxiv.org/pdf/2206.10902.pdf)] [[code](https://github.com/chenghuang66/s2tnet)]
* Trajectory Prediction using Equivariant Continuous Convolution, ICLR 2021. [[paper](https://arxiv.org/pdf/2010.11344.pdf)] [[code](https://github.com/Rose-STL-Lab/ECCO)]
* TridentNet: A Conditional Generative Model for Dynamic Trajectory Generation, International Conference on Intelligent Autonomous Systems 2021. [[paper](https://link.springer.com/chapter/10.1007/978-3-030-95892-3_31#Abs1)]
* HOME: Heatmap Output for future Motion Estimation, ITSC 2021. [[paper](https://arxiv.org/pdf/2105.10968.pdf)]
* Graph and Recurrent Neural Network-based Vehicle Trajectory Prediction For Highway Driving, ITSC 2021. [[paper](https://ieeexplore.ieee.org/abstract/document/9564929)]
* SCSG Attention: A Self-Centered Star Graph with Attention for Pedestrian Trajectory Prediction, International Conference on Database Systems for Advanced Applications (DASFAA 2021). [[paper](https://link.springer.com/content/pdf/10.1007/978-3-030-73194-6_29.pdf)]
* Leveraging Trajectory Prediction for Pedestrian Video Anomaly Detection, IEEE Symposium Series on Computational Intelligence (SSCI 2021). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9660004)] [[code](https://github.com/akanuasiegbu/Leveraging-Trajectory-Prediction-for-Pedestrian-Video-Anomaly-Detection)]

## Journal Papers 2021
* Are socially-aware trajectory prediction models really socially-aware?, Transportation Research: Part C. [[paper](https://arxiv.org/pdf/2108.10879.pdf), [paper](https://iccv21-adv-workshop.github.io/short_paper/s-attack-arow2021.pdf)] [[code](https://s-attack.github.io/)]
* Injecting knowledge in data-driven vehicle trajectory predictors, Transportation Research: Part C. [[paper](https://reader.elsevier.com/reader/sd/pii/S0968090X21000425?token=F03D20769BFB255F56662C10348A81F3D07A42C6B4AB9BA19E3F7B2A5F1DA7D99B96B783616BDA86C12866AFCF4C5671&originRegion=eu-west-1&originCreation=20220506090622)] [[code](https://github.com/vita-epfl/RRB)]
* Decoding pedestrian and automated vehicle interactions using immersive virtual reality and interpretable deep learning, Transportation Research: Part C. [[paper](https://www.sciencedirect.com/science/article/pii/S0968090X2030855X)]
* Human Trajectory Forecasting in Crowds: A Deep Learning Perspective,  IEEE Transactions on Intelligent Transportation Systems. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9408398)] [[code](https://github.com/vita-epfl/trajnetplusplusbaselines)]
* NetTraj: A Network-Based Vehicle Trajectory Prediction Model With Directional Representation and Spatiotemporal Attention Mechanisms, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9629362)]
* Spatio-Temporal Graph Dual-Attention Network for Multi-Agent Prediction and Tracking, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9491972)]
* A Hierarchical Framework for Interactive Behaviour Prediction of Heterogeneous Traffic Participants Based on Graph Neural Network, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9468360&tag=1)]
* TrajGAIL: Generating urban vehicle trajectories using generative adversarial imitation learning, Transportation Research Part C. [[paper](https://reader.elsevier.com/reader/sd/pii/S0968090X21001121?token=3DEACAF2AD919E99B3331E74F747B61A0EAC2741E79B6F99F4F806155EB394F163D74F2F83806358BBD65911E107EF01&originRegion=us-east-1&originCreation=20220416040814)] [[code](https://github.com/benchoi93/TrajGAIL)]
* Vehicle Trajectory Prediction Using Generative Adversarial Network With Temporal Logic Syntax Tree Features, IEEE ROBOTICS AND AUTOMATION LETTERS. [[paper](https://www.gilitschenski.org/igor/publications/202104-ral-logic_gan/ral21-logic_gan.pdf)]
* Vehicle Trajectory Prediction Using LSTMs with Spatial-Temporal Attention Mechanisms, IEEE Intelligent Transportation Systems Magazine. [[paper](http://urdata.net/files/2020_VTP.pdf)] [[code](https://github.com/leilin-research/VTP)]
* Long Short-Term Memory-Based Human-Driven Vehicle Longitudinal Trajectory Prediction in a Connected and Autonomous Vehicle Environment, Transportation Research Record. [[paper](http://sage.cnpereading.com/paragraph/download/?doi=10.1177/0361198121993471)]
* Temporal Pyramid Network with Spatial-Temporal Attention for Pedestrian Trajectory Prediction, IEEE Transactions on Network Science and Engineering. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9373939)]
* An efficient Spatial–Temporal model based on gated linear units for trajectory prediction, Neurocomputing. [[paper](https://reader.elsevier.com/reader/sd/pii/S0925231221018907?token=C894F657732BB6078B77AEC9BD3858338C1A7F1254CCC0BBC34ADA1421A95CF9A4F68BDCA8812457DE27FB37EEB8F198&originRegion=us-east-1&originCreation=20220420144432)]
* SRAI-LSTM: A Social Relation Attention-based Interaction-aware LSTM for human trajectory prediction, Neurocomputing. [[paper](https://reader.elsevier.com/reader/sd/pii/S0925231221018014?token=BB22DAAC41E3BF453C326A9D72A0CC900C2DFFD0D8AE07B7DEED51C7F2250B9CB40CC89B6812CA20DBFA6A7EDD32AAD6&originRegion=us-east-1&originCreation=20220512100647)]
* AST-GNN: An attention-based spatio-temporal graph neural network for Interaction-aware pedestrian trajectory prediction, Neurocomputing. [[paper](https://www.sciencedirect.com/science/article/pii/S092523122100388X)]
* Multi-PPTP: Multiple Probabilistic Pedestrian Trajectory Prediction in the Complex Junction Scene, IEEE Transactions on Intelligent Transportation Systems. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9619864)]
* A Novel Graph-Based Trajectory Predictor With Pseudo-Oracle, TNNLS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9447207)]
* Large Scale GPS Trajectory Generation Using Map Based on Two Stage GAN, Journal of Data Science. [[paper](https://www.jds-online.com/files/JDS202001-08.pdf)] [[code](https://github.com/XingruiWang/Two-Stage-Gan-in-trajectory-generation)]
* Pose and Semantic Map Based Probabilistic Forecast of Vulnerable Road Users’ Trajectories, IEEE Transactions on Intelligent Vehicles. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9707640)]
* STI-GAN: Multimodal Pedestrian Trajectory Prediction Using Spatiotemporal Interactions and a Generative Adversarial Network, IEEE Access. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9387292)]
* Holistic LSTM for Pedestrian Trajectory Prediction, TIP. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9361440)]
* Pedestrian trajectory prediction with convolutional neural networks, PR. [[paper](https://www.sciencedirect.com/science/article/pii/S0031320321004325)]
* LSTM based trajectory prediction model for cyclist utilizing multiple interactions with environment, PR. [[paper](https://www.sciencedirect.com/science/article/pii/S0031320320306038)]
* Human trajectory prediction and generation using LSTM models and GANs, PR. [[paper](https://www.sciencedirect.com/science/article/pii/S003132032100323X)]
* Vehicle trajectory prediction and generation using LSTM models and GANs, Plos one. [[paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0253868)]
* BiTraP: Bi-Directional Pedestrian Trajectory Prediction With Multi-Modal Goal Estimation, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9345445)] [[code](https://github.com/umautobots/bidireaction-trajectory-prediction)]
* A Kinematic Model for Trajectory Prediction in General Highway Scenarios, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9472993)] [[code](https://github.com/umautobots/kinematic_highway)]
* Trajectory Prediction in Autonomous Driving With a Lane Heading Auxiliary Loss, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9387075)]
* Vehicle Trajectory Prediction Using Generative Adversarial Network With Temporal Logic Syntax Tree Features, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9366373)]
* Tra2Tra: Trajectory-to-Trajectory Prediction With a Global Social Spatial-Temporal Attentive Neural Network, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9347678)]
* Social graph convolutional LSTM for pedestrian trajectory prediction, IET Intelligent Transport Systems. [[paper](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/itr2.12033)]
* HSTA: A Hierarchical Spatio-Temporal Attention Model for Trajectory Prediction, IEEE Transactions on Vehicular Technology (TVT). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9548801)]
* Environment-Attention Network for Vehicle Trajectory Prediction, TVT. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9534487)]
* Where Are They Going? Predicting Human Behaviors in Crowded Scenes, ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM). [[paper](https://dl.acm.org/doi/pdf/10.1145/3449359)]
* Multi-Agent Trajectory Prediction with Spatio-Temporal Sequence Fusion, IEEE Transactions on Multimedia (TMM). [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9580659)]

## Others 2021
* Trajectory Prediction using Generative Adversarial Network in Multi-Class Scenarios, arXiv preprint arXiv:2110.11401, 2021. [[paper](https://arxiv.org/pdf/2110.11401.pdf)]
* Spatial-Channel Transformer Network for Trajectory Prediction on the Traffic Scenes, arXiv preprint arXiv:2101.11472, 2021. [[paper](https://arxiv.org/ftp/arxiv/papers/2101/2101.11472.pdf)]
* Physically Feasible Vehicle Trajectory Prediction, arXiv preprint arXiv:2104.14679, 2021. [[paper](https://arxiv.org/pdf/2104.14679.pdf)]
* MSN: Multi-Style Network for Trajectory Prediction, arXiv preprint arXiv:2107.00932, 2021. [[paper](https://arxiv.org/pdf/2107.00932.pdf)] [[code](https://github.com/NorthOcean/MSN)]
* Rethinking Trajectory Forecasting Evaluation, arXiv preprint arXiv:2107.10297, 2021. [[paper](https://arxiv.org/pdf/2107.10297)]
* Pedestrian Trajectory Prediction via Spatial Interaction Transformer Network, IEEE Intelligent Vehicles Symposium Workshops (IV Workshops 2021). [[paper](https://arxiv.org/pdf/2112.06624)]
* Deep Social Force, arXiv preprint arXiv:2109.12081, 2021. [[paper](https://arxiv.org/pdf/2109.12081)] [[code](https://github.com/svenkreiss/socialforce)]

# 2022 Conference and Journal Papers
## Conference Papers 2022
* Social Interpretable Tree for Pedestrian Trajectory Prediction, AAAI 2022. [[paper](https://arxiv.org/pdf/2205.13296.pdf)] [[code](https://github.com/lssiair/SIT)]
* Complementary Attention Gated Network for Pedestrian Trajectory Prediction, AAAI 2022. [[paper](https://www.aaai.org/AAAI22Papers/AAAI-1963.DuanJ.pdf)] [[code](https://github.com/jinghaiD/CAGN)]
* Scene Transformer: A unified architecture for predicting future trajectories of multiple agents, ICLR 2022. [[paper](https://openreview.net/pdf?id=Wm3EA5OlHsG)]
* You Mostly Walk Alone: Analyzing Feature Attribution in Trajectory Prediction, ICLR 2022. [[paper](https://arxiv.org/pdf/2110.05304.pdf)]
* Latent Variable Sequential Set Transformers For Joint Multi-Agent Motion Prediction, ICLR 2022. [[paper](https://openreview.net/pdf?id=Dup_dDqkZC5)] [[code](https://fgolemo.github.io/autobots/)]
* THOMAS: Trajectory Heatmap Output with learned Multi-Agent Sampling, ICLR 2022. [[paper](https://arxiv.org/pdf/2110.06607)]
* Remember Intentions: Retrospective-Memory-based Trajectory Prediction, CVPR 2022. [[paper](https://arxiv.org/pdf/2203.11474.pdf)] [[code](https://github.com/MediaBrain-SJTU/MemoNet)]
* STCrowd: A Multimodal Dataset for Pedestrian Perception in Crowded Scenes, CVPR 2022. [[paper](https://arxiv.org/pdf/2204.01026.pdf)] [[code](https://github.com/4DVLab/STCrowd.git)]
* Vehicle trajectory prediction works, but not everywhere, CVPR 2022. [[paper](https://arxiv.org/pdf/2112.03909.pdf)] [[code](https://s-attack.github.io/)]
* Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion, CVPR 2022. [[paper](https://arxiv.org/pdf/2203.13777.pdf)] [[code](https://github.com/gutianpei/MID)]
* Non-Probability Sampling Network for Stochastic Human Trajectory Prediction, CVPR 2022. [[paper](https://arxiv.org/pdf/2203.13471.pdf)] [[code](https://github.com/inhwanbae/NPSN)]
* On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles, CVPR 2022. [[paper](https://arxiv.org/pdf/2201.05057.pdf)] [[code](https://github.com/zqzqz/AdvTrajectoryPrediction)]
* Adaptive Trajectory Prediction via Transferable GNN, CVPR 2022. [[paper](https://arxiv.org/pdf/2203.05046.pdf)]
* Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective, CVPR 2022. [[paper](https://arxiv.org/pdf/2111.14820.pdf)] [[code](https://github.com/vita-epfl/causalmotion), [code](https://github.com/sherwinbahmani/ynet_adaptive)]
* How many Observations are Enough? Knowledge Distillation for Trajectory Forecasting, CVPR 2022. [[paper](https://arxiv.org/pdf/2203.04781.pdf)]
* Learning from All Vehicles, CVPR 2022. [[paper](https://arxiv.org/pdf/2203.11934.pdf)] [[code](https://github.com/dotchen/LAV)]
* Forecasting from LiDAR via Future Object Detection, CVPR 2022. [[paper](https://arxiv.org/pdf/2203.16297.pdf)] [[code](https://github.com/neeharperi/FutureDet)]
* End-to-End Trajectory Distribution Prediction Based on Occupancy Grid Maps, CVPR 2022. [[paper](https://arxiv.org/pdf/2203.16910.pdf)] [[code](https://github.com/Kguo-cs/TDOR)]
* M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction, CVPR 2022. [[paper](https://arxiv.org/pdf/2202.11884.pdf)] [[code](https://tsinghua-mars-lab.github.io/M2I/)]
* GroupNet: Multiscale Hypergraph Neural Networks for Trajectory Prediction with Relational Reasoning, CVPR 2022. [[paper](https://arxiv.org/pdf/2204.08770.pdf)] [[code](https://github.com/MediaBrain-SJTU/GroupNet)]
* Whose Track Is It Anyway? Improving Robustness to Tracking Errors with Affinity-Based Prediction, CVPR 2022. [[paper](https://xinshuoweng.com/papers/Affinipred/camera_ready.pdf)]
* ScePT: Scene-consistent, Policy-based Trajectory Predictions for Planning, CVPR 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_ScePT_Scene-Consistent_Policy-Based_Trajectory_Predictions_for_Planning_CVPR_2022_paper.pdf)] [[code](https://github.com/NVlabs/ScePT)]
* Graph-based Spatial Transformer with Memory Replay for Multi-future Pedestrian Trajectory Prediction, CVPR 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Graph-Based_Spatial_Transformer_With_Memory_Replay_for_Multi-Future_Pedestrian_Trajectory_CVPR_2022_paper.pdf)] [[code](https://github.com/Jacobieee/ST-MR)]
* MUSE-VAE: Multi-Scale VAE for Environment-Aware Long Term Trajectory Prediction, CVPR 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_MUSE-VAE_Multi-Scale_VAE_for_Environment-Aware_Long_Term_Trajectory_Prediction_CVPR_2022_paper.pdf)]
* LTP: Lane-based Trajectory Prediction for Autonomous Driving, CVPR 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_LTP_Lane-Based_Trajectory_Prediction_for_Autonomous_Driving_CVPR_2022_paper.pdf)]
* ATPFL: Automatic Trajectory Prediction Model Design under Federated Learning Framework, CVPR 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ATPFL_Automatic_Trajectory_Prediction_Model_Design_Under_Federated_Learning_Framework_CVPR_2022_paper.pdf)]
* Human Trajectory Prediction with Momentary Observation, CVPR 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Human_Trajectory_Prediction_With_Momentary_Observation_CVPR_2022_paper.pdf)]
* HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction, CVPR 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf)] [[code](https://github.com/ZikangZhou/HiVT)]
* Path-Aware Graph Attention for HD Maps in Motion Prediction, ICRA 2022. [[paper](https://arxiv.org/pdf/2202.13772.pdf)]
* Trajectory Prediction with Linguistic Representations, ICRA 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9811928)]
* Leveraging Smooth Attention Prior for Multi-Agent Trajectory Prediction, ICRA 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9811718)] [[website](https://sites.google.com/view/smoothness-attention)]
* KEMP: Keyframe-Based Hierarchical End-to-End Deep Model for Long-Term Trajectory Prediction, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9812337)]
* Domain Generalization for Vision-based Driving Trajectory Generation, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9812070)] [[website](https://sites.google.com/view/dg-traj-gen)]
* A Deep Concept Graph Network for Interaction-Aware Trajectory Prediction, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9811567)]
* Conditioned Human Trajectory Prediction using Iterative Attention Blocks, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9812404)]
* StopNet: Scalable Trajectory and Occupancy Prediction for Urban Autonomous Driving, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9811830)]
* Meta-path Analysis on Spatio-Temporal Graphs for Pedestrian Trajectory Prediction, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9811632)] [[website](https://sites.google.com/illinois.edu/mesrnn/home)]
* Propagating State Uncertainty Through Trajectory Forecasting, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9811776)] [[code](https://github.com/StanfordASL/PSU-TF)]
* HYPER: Learned Hybrid Trajectory Prediction via Factored Inference and Adaptive Sampling, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9812254)]
* Grouptron: Dynamic Multi-Scale Graph Convolutional Networks for Group-Aware Dense Crowd Trajectory Forecasting, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9811585)]
* Crossmodal Transformer Based Generative Framework for Pedestrian Trajectory Prediction, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9812226)]
* Trajectory Prediction for Autonomous Driving with Topometric Map, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9811712)] [[code](https://github.com/Jiaolong/trajectory-prediction)]
* CRAT-Pred: Vehicle Trajectory Prediction with Crystal Graph Convolutional Neural Networks and Multi-Head Self-Attention, ICRA 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9811637)] [[code](https://github.com/schmidt-ju/crat-pred)]
* MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9812107)]
* Multi-modal Motion Prediction with Transformer-based Neural Network for Autonomous Driving, ICRA 2022. [[paper](https://ieeexplore.ieee.org/document/9812060/)]
* GOHOME: Graph-Oriented Heatmap Output for future Motion Estimation, ICRA 2022. [[paper](https://arxiv.org/pdf/2109.01827.pdf)]
* TridentNetV2: Lightweight Graphical Global Plan Representations for Dynamic Trajectory Generation, ICRA 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9811591)]
* Heterogeneous-Agent Trajectory Forecasting Incorporating Class Uncertainty, IROS 2022. [[paper](https://arxiv.org/pdf/2104.12446.pdf)] [[code](https://github.com/TRI-ML/HAICU)] [[trajdata](https://github.com/nvr-avg/trajdata)]
* Trajectory Prediction with Graph-based Dual-scale Context Fusion, IROS 2022. [[paper](https://arxiv.org/pdf/2111.01592.pdf)] [[code](https://github.com/HKUST-Aerial-Robotics/DSP)]
* Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction, ECCV 2022. [[paper](https://arxiv.org/pdf/2207.09953.pdf)] [[code](https://github.com/InhwanBae/GPGraph)]
* Social-Implicit: Rethinking Trajectory Prediction Evaluation and The Effectiveness of Implicit Maximum Likelihood Estimation, ECCV 2022. [[paper](https://arxiv.org/pdf/2203.03057.pdf)] [[code](https://github.com/abduallahmohamed/Social-Implicit)] [[website](https://www.abduallahmohamed.com/social-implicit-amdamv-adefde-demo)] 
* Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting, ECCV 2022. [[paper](https://arxiv.org/pdf/2207.04624.pdf)] [[code](https://github.com/d1024choi/HLSTrajForecast)]
* SocialVAE: Human Trajectory Prediction using Timewise Latents, ECCV 2022. [[paper](https://arxiv.org/pdf/2203.08207.pdf)] [[code](https://github.com/xupei0610/SocialVAE)]
* View Vertically: A Hierarchical Network for Trajectory Prediction via Fourier Spectrums, ECCV 2022. [[paper](https://arxiv.org/pdf/2110.07288.pdf)] [[code](https://github.com/cocoon2wong/Vertical)]
* Entry-Flipped Transformer for Inference and Prediction of Participant Behavior, ECCV 2022. [[paper](https://arxiv.org/pdf/2207.06235.pdf)]
* D2-TPred: Discontinuous Dependency for Trajectory Prediction under Traffic Lights, ECCV 2022. [[paper](https://arxiv.org/pdf/2207.10398.pdf)] [[code](https://github.com/VTP-TL/D2-TPred)]
* Human Trajectory Prediction via Neural Social Physics, ECCV 2022. [[paper](https://arxiv.org/pdf/2207.10435.pdf)] [[code](https://github.com/realcrane/Human-Trajectory-Prediction-via-Neural-Social-Physics)]
* Social-SSL: Self-Supervised Cross-Sequence Representation Learning Based on Transformers for Multi-Agent Trajectory Prediction, ECCV 2022. [[paper](https://basiclab.lab.nycu.edu.tw/assets/Social-SSL.pdf)] [[code](https://github.com/Sigta678/Social-SSL)]
* Aware of the History: Trajectory Forecasting with the Local Behavior Data, ECCV 2022. [[paper](https://arxiv.org/pdf/2207.09646.pdf)] [[code](https://github.com/Kay1794/Aware-of-the-history)]
* Action-based Contrastive Learning for Trajectory Prediction, ECCV 2022. [[paper](https://arxiv.org/pdf/2207.08664.pdf)]
* AdvDO: Realistic Adversarial Attacks for Trajectory Prediction, ECCV 2022. [[paper](https://arxiv.org/pdf/2209.08744.pdf)]
* ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning, ECCV 2022. [[paper](https://arxiv.org/pdf/2207.07601.pdf)] [[code](https://github.com/OpenPerceptionX/ST-P3)]
* Social ODE: Multi-Agent Trajectory Forecasting with Neural Ordinary Differential Equations, ECCV 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820211.pdf)]
* Forecasting Human Trajectory from Scene History, NIPS 2022. [[paper](https://arxiv.org/pdf/2210.08732.pdf)] [[code](https://github.com/MaKaRuiNah/SHENet)]
* Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline, NIPS 2022. [[paper](https://arxiv.org/pdf/2206.08129)] [[code](https://github.com/OpenPerceptionX/TCP)]
* Motion Transformer with Global Intention Localization and Local Movement Refinement, NIPS 2022. [[paper](https://arxiv.org/pdf/2209.13508.pdf)] [[website](https://vas.mpi-inf.mpg.de/motion-transformer-with-global-intention-localization-and-local-movement-refinement/)]
* Interaction Modeling with Multiplex Attention, NIPS 2022. [[paper](https://arxiv.org/pdf/2208.10660.pdf)] [[code](https://github.com/fanyun-sun/IMMA)]
* Deep Interactive Motion Prediction and Planning: Playing Games with Motion Prediction Models, Conference on Learning for Dynamics and Control (L4DC). [[paper](https://arxiv.org/pdf/2204.02392.pdf)] [[website](https://sites.google.com/view/deep-interactive-predict-plan)]
* Robust Trajectory Prediction against Adversarial Attacks, CoRL 2022. [[paper](https://arxiv.org/pdf/2208.00094.pdf)] [[code](https://robustav.github.io/RobustTraj/)]
* Planning with Diffusion for Flexible Behavior Synthesis, ICML 2022. [[paper](https://arxiv.org/abs/2205.09991)] [[website](https://diffusion-planning.github.io/)]
* Synchronous Bi-Directional Pedestrian Trajectory Prediction with Error Compensation, ACCV 2022. [[paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Xie_Synchronous_Bi-Directional_Pedestrian_Trajectory_Prediction_with_Error_Compensation_ACCV_2022_paper.pdf)]
* Model-Based Imitation Learning for Urban Driving, NIPS 2022. [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/827cb489449ea216e4a257c47e407d18-Paper-Conference.pdf)] [[code](https://github.com/wayveai/mile)]

## Journal Papers 2022
* AI-TP: Attention-based Interaction-aware Trajectory Prediction for Autonomous Driving, IEEE Transactions on Intelligent Vehicles. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9723649)] [[code](https://github.com/KP-Zhang/AI-TP)]
* MDST-DGCN: A Multilevel Dynamic Spatiotemporal Directed Graph Convolutional Network for Pedestrian Trajectory Prediction, Computational Intelligence and Neuroscience. [[paper](https://downloads.hindawi.com/journals/cin/2022/4192367.pdf)]
* Graph-Based Spatial-Temporal Convolutional Network for Vehicle Trajectory Prediction in Autonomous Driving, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9737058)]
* Multi-Agent Trajectory Prediction with Heterogeneous Edge-Enhanced Graph Attention Network, TITS. [[paper](https://dspace.lib.cranfield.ac.uk/bitstream/handle/1826/17541/Multi-agent_trajectory_prediction-2022.pdf?sequence=1&isAllowed=y)]
* Fully Convolutional Encoder-Decoder With an Attention Mechanism for Practical Pedestrian Trajectory Prediction, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9768201)]
* STGM: Vehicle Trajectory Prediction Based on Generative Model for Spatial-Temporal Features, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9743363)]
* Trajectory Prediction for Autonomous Driving Using Spatial-Temporal Graph Attention Transformer, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9768029)]
* Intention-Aware Vehicle Trajectory Prediction Based on Spatial-Temporal Dynamic Attention Network for Internet of Vehicles, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9767719)] [[code](https://xbchen82.github.io/resource/)]
* Trajectory Forecasting Based on Prior-Aware Directed Graph Convolutional Neural Network, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9686621&tag=1)]
* DeepTrack: Lightweight Deep Learning for Vehicle Trajectory Prediction in Highways, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9770480)]
* Interactive Trajectory Prediction Using a Driving Risk Map-Integrated Deep Learning Method for Surrounding Vehicles on Highways, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9745461&tag=1)]
* Vehicle Trajectory Prediction in Connected Environments via Heterogeneous Context-Aware Graph Convolutional Networks, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9781338)]
* Trajectory Prediction Neural Network and Model Interpretation Based on Temporal Pattern Attention, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9945660)]
* Learning Sparse Interaction Graphs of Partially Detected Pedestrians for Trajectory Prediction, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9664278)] [[code](https://github.com/tedhuang96/gst)]
* GAMMA: A General Agent Motion Prediction Model for Autonomous Driving, RAL. [[paper](https://arxiv.org/pdf/1906.01566.pdf)] [[code](https://github.com/AdaCompNUS/gamma)]
* Stepwise Goal-Driven Networks for Trajectory Prediction, RAL. [[paper](https://arxiv.org/pdf/2103.14107v3.pdf)] [[code](https://github.com/ChuhuaW/SGNet.pytorch)]
* GA-STT: Human Trajectory Prediction with Group Aware Spatial-Temporal Transformer, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9779572)]
* Long-term 4D trajectory prediction using generative adversarial networks, Transportation Research Part C: Emerging Technologies. [[paper](https://www.sciencedirect.com/science/article/pii/S0968090X22000031)]
* A context-aware pedestrian trajectory prediction framework for automated vehicles, Transportation Research Part C: Emerging Technologies. [[paper](https://www.sciencedirect.com/science/article/pii/S0968090X21004423)]
* Explainable multimodal trajectory prediction using attention models, Transportation Research Part C: Emerging Technologies. [[paper](https://www.sciencedirect.com/science/article/pii/S0968090X22002509)]
* CSCNet: Contextual semantic consistency network for trajectory prediction in crowded spaces, PR. [[paper](https://www.sciencedirect.com/science/article/pii/S0031320322000334)]
* CSR: Cascade Conditional Variational AutoEncoder with Social-aware Regression for Pedestrian Trajectory Prediction, PR. [[paper](https://www.sciencedirect.com/science/article/pii/S0031320322005106)]
* Step Attention: Sequential Pedestrian Trajectory Prediction, IEEE Sensors Journal. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9732437)]
* Vehicle Trajectory Prediction Method Coupled With Ego Vehicle Motion Trend Under Dual Attention Mechanism, IEEE Transactions on Instrumentation and Measurement. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9749176)]
* Spatio-temporal Interaction Aware and Trajectory Distribution Aware Graph Convolution Network for Pedestrian Multimodal Trajectory Prediction, IEEE Transactions on Instrumentation and Measurement. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9997233)]
* Deep encoder–decoder-NN: A deep learning-based autonomous vehicle trajectory prediction and correction model, Physica A: Statistical Mechanics and its Applications. [[paper](https://www.sciencedirect.com/science/article/pii/S0378437122000139)]
* PTPGC: Pedestrian trajectory prediction by graph attention network with ConvLSTM, Robotics and Autonomous Systems. [[paper](https://www.sciencedirect.com/science/article/pii/S0921889021002165)]
* GCHGAT: pedestrian trajectory prediction using group constrained hierarchical graph attention networks, Applied Intelligence. [[paper](https://link.springer.com/article/10.1007/s10489-021-02997-w)]
* Vehicles Trajectory Prediction Using Recurrent VAE Network, IEEE Access. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9740177)] [[code](https://github.com/midemig/traj_pred_vae)]
* SEEM: A Sequence Entropy Energy-Based Model for Pedestrian Trajectory All-Then-One Prediction, TPAMI. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9699076)]
* PTP-STGCN: Pedestrian Trajectory Prediction Based on a Spatio-temporal Graph Convolutional Neural Network, Applied Intelligence. [[paper](https://link.springer.com/article/10.1007/s10489-022-03524-1)]
* Trajectory distributions: A new description of movement for trajectory prediction, Computational Visual Media. [[paper](https://link.springer.com/content/pdf/10.1007/s41095-021-0236-6.pdf)]
* Trajectory prediction for autonomous driving based on multiscale spatial-temporal graph, IET Intelligent Transport Systems. [[paper](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/itr2.12265)]
* Continual learning-based trajectory prediction with memory augmented networks, Knowledge-Based Systems. [[paper](https://www.sciencedirect.com/science/article/pii/S0950705122011157)]
* Atten-GAN: Pedestrian Trajectory Prediction with GAN Based on Attention Mechanism, Cognitive Computation. [[paper](https://link.springer.com/article/10.1007/s12559-022-10029-z#Abs1)]
* EvoSTGAT: Evolving spatiotemporal graph attention networks for pedestrian trajectory prediction, Neurocomputing. [[paper](https://www.sciencedirect.com/science/article/pii/S0925231222003460?ref=pdf_download&fr=RR-2&rr=7da0ead45e800fcc)]

## Others 2022
* Raising context awareness in motion forecasting, CVPR Workshops 2022. [[paper](https://arxiv.org/pdf/2109.08048.pdf)] [[code](https://github.com/valeoai/CAB)]
* Goal-driven Self-Attentive Recurrent Networks for Trajectory Prediction, CVPR Workshops 2022. [[paper](https://arxiv.org/pdf/2204.11561.pdf)] [[code](https://github.com/luigifilippochiara/Goal-SAR)]
* Importance Is in Your Attention: Agent Importance Prediction for Autonomous Driving, CVPR Workshops 2022. [[paper](https://arxiv.org/pdf/2204.09121.pdf)]
* MPA: MultiPath++ Based Architecture for Motion Prediction, CVPR Workshops 2022. [[paper](https://arxiv.org/pdf/2206.10041.pdf)] [[code](https://github.com/stepankonev/waymo-motion-prediction-challenge-2022-multipath-plus-plus)]
* TPAD: Identifying Effective Trajectory Predictions Under the Guidance of Trajectory Anomaly Detection Model, arXiv:2201.02941, 2022. [[paper](https://arxiv.org/pdf/2201.02941v1.pdf)]
* Wayformer: Motion Forecasting via Simple & Efficient Attention Networks, arXiv preprint arXiv:2207.05844, 2022. [[paper](https://arxiv.org/pdf/2207.05844.pdf)]
* PreTR: Spatio-Temporal Non-Autoregressive Trajectory Prediction Transformer, arXiv preprint arXiv:2203.09293, 2022. [[paper](https://arxiv.org/pdf/2203.09293.pdf)]
* LatentFormer: Multi-Agent Transformer-Based Interaction Modeling and Trajectory Prediction, arXiv preprint arXiv:2203.01880, 2022. [[paper](https://arxiv.org/pdf/2203.01880.pdf)]
* Diverse Multiple Trajectory Prediction Using a Two-stage Prediction Network Trained with Lane Loss, arXiv preprint arXiv:2206.08641, 2022. [[paper](https://arxiv.org/pdf/2206.08641.pdf)]
* Semi-supervised Semantics-guided Adversarial Training for Trajectory Prediction, arXiv preprint arXiv:2205.14230, 2022. [[paper](https://arxiv.org/pdf/2205.14230.pdf)]
* Heterogeneous Trajectory Forecasting via Risk and Scene Graph Learning, arXiv preprint arXiv:2211.00848, 2022. [[paper](https://arxiv.org/pdf/2211.00848.pdf)]
* GATraj: A Graph- and Attention-based Multi-Agent Trajectory Prediction Model, arXiv preprint arXiv:2209.07857, 2022. [[paper](https://arxiv.org/pdf/2209.07857.pdf)] [[code](https://github.com/mengmengliu1998/GATraj)]
* Dynamic-Group-Aware Networks for Multi-Agent Trajectory Prediction with Relational Reasoning, arXiv preprint arXiv:2206.13114, 2022. [[paper](https://arxiv.org/pdf/2206.13114.pdf)]
* Collaborative Uncertainty Benefits Multi-Agent Multi-Modal Trajectory Forecasting, arXiv preprint arXiv:2207.05195, 2022. [[paper](https://arxiv.org/abs/2207.05195)] [[code](https://github.com/MediaBrain-SJTU/Collaborative-Uncertainty)]
* Guided Conditional Diffusion for Controllable Traffic Simulation, arXiv preprint arXiv:2210.17366, 2022. [[paper](https://arxiv.org/pdf/2210.17366.pdf)] [[website](https://aiasd.github.io/ctg.github.io/)]
* PhysDiff: Physics-Guided Human Motion Diffusion Model, arXiv preprint arXiv:2212.02500, 2022. [[paper](http://xxx.itp.ac.cn/pdf/2212.02500.pdf)]
* Trajectory Forecasting on Temporal Graphs, arXiv preprint arXiv:2207.00255, 2022. [[paper](https://arxiv.org/pdf/2207.00255.pdf)] [[website](https://kuis-ai.github.io/ftgn/)]

# 2023 Conference and Journal Papers
## Conference Papers 2023
* Human Joint Kinematics Diffusion-Refinement for Stochastic Motion Prediction, AAAI 2023. [[paper](https://arxiv.org/pdf/2210.05976.pdf)]
* Multi-stream Representation Learning for Pedestrian Trajectory Prediction, AAAI 2023. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25389)]
* Continuous Trajectory Generation Based on Two-Stage GAN, AAAI 2023. [[paper](https://arxiv.org/pdf/2301.07103.pdf)] [[code](https://github.com/WenMellors/TS-TrajGen)]
* A Set of Control Points Conditioned Pedestrian Trajectory Prediction, AAAI 2023. [[paper](https://assets.underline.io/lecture/67747/paper/82988b653861eb7a0d5cdc91c4b26f8c.pdf)] [[code](https://github.com/InhwanBae/GraphTERN)]
* Leveraging Future Relationship Reasoning for Vehicle Trajectory Prediction, ICLR 2023. [[paper](https://openreview.net/forum?id=CGBCTp2M6lA)]
* IPCC-TP: Utilizing Incremental Pearson Correlation Coefficient for Joint Multi-Agent Trajectory Prediction, CVPR 2023. [[paper](https://arxiv.org/pdf/2303.00575.pdf)]
* FEND: A Future Enhanced Distribution-Aware Contrastive Learning Framework for Long-tail Trajectory Prediction, CVPR 2023. [[paper](https://arxiv.org/pdf/2303.16574.pdf)]
* Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion, CVPR 2023. [[paper](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf)] [[website](https://nv-tlabs.github.io/trace-pace/)]
* FJMP: Factorized Joint Multi-Agent Motion Prediction over Learned Directed Acyclic Interaction Graphs, CVPR 2023. [[paper](https://arxiv.org/pdf/2211.16197.pdf)] [[website](https://rluke22.github.io/FJMP/)]
* Leapfrog Diffusion Model for Stochastic Trajectory Prediction, CVPR 2023. [[paper](https://arxiv.org/pdf/2303.10895.pdf)] [[code](https://github.com/MediaBrain-SJTU/LED)]
* ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries, CVPR 2023. [[paper](http://xxx.itp.ac.cn/pdf/2208.01582.pdf)] [[website](https://tsinghua-mars-lab.github.io/ViP3D/)]
* EqMotion: Equivariant Multi-Agent Motion Prediction with Invariant Interaction Reasoning, CVPR 2023. [[paper](https://arxiv.org/pdf/2303.10876.pdf)] [[code](https://github.com/MediaBrain-SJTU/EqMotion)]
* Uncovering the Missing Pattern: Unified Framework Towards Trajectory Imputation and Prediction, CVPR 2023. [[paper](http://xxx.itp.ac.cn/pdf/2303.16005.pdf)]
* Unsupervised Sampling Promoting for Stochastic Human Trajectory Prediction, CVPR 2023. [[paper](https://chengy12.github.io/files/Bosampler.pdf)] [[code](https://github.com/viewsetting/Unsupervised_sampling_promoting)]
* Stimulus Verification is a Universal and Effective Sampler in Multi-modal Human Trajectory Prediction, CVPR 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Stimulus_Verification_Is_a_Universal_and_Effective_Sampler_in_Multi-Modal_CVPR_2023_paper.pdf)]
* Query-Centric Trajectory Prediction, CVPR 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf)] [[code](https://github.com/ZikangZhou/QCNet)] [[QCNeXt](https://arxiv.org/pdf/2306.10508.pdf)]
* Weakly Supervised Class-agnostic Motion Prediction for Autonomous Driving, CVPR 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Weakly_Supervised_Class-Agnostic_Motion_Prediction_for_Autonomous_Driving_CVPR_2023_paper.pdf)]
* Decompose More and Aggregate Better: Two Closer Looks at Frequency Representation Learning for Human Motion Prediction, CVPR 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_Decompose_More_and_Aggregate_Better_Two_Closer_Looks_at_Frequency_CVPR_2023_paper.pdf)]
* MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion, CVPR 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_MotionDiffuser_Controllable_Multi-Agent_Motion_Prediction_Using_Diffusion_CVPR_2023_paper.pdf)]
* Planning-oriented Autonomous Driving, CVPR 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.pdf)] [[code](https://github.com/OpenDriveLab/UniAD)]
* TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios, ICRA 2023. [[paper](https://arxiv.org/pdf/2210.06609.pdf)] [[code](https://github.com/metadriverse/trafficgen)]
* GANet: Goal Area Network for Motion Forecasting, ICRA 2023. [[paper](https://arxiv.org/pdf/2209.09723.pdf)] [[code](https://github.com/kingwmk/GANet)]
* TOFG: A Unified and Fine-Grained Environment Representation in Autonomous Driving, ICRA 2023. [[paper](https://arxiv.org/pdf/2305.20068.pdf)]
* SSL-Lanes: Self-Supervised Learning for Motion Forecasting in Autonomous Driving, CoRL 2023. [[paper](https://arxiv.org/pdf/2206.14116.pdf)] [[code](https://github.com/AutoVision-cloud/SSL-Lanes)]
* PowerBEV: A Powerful Yet Lightweight Framework for Instance Prediction in Bird’s-Eye View, IJCAI 2023. [[paper](https://arxiv.org/pdf/2306.10761.pdf)] [[code](https://github.com/EdwardLeeLPZ/PowerBEV)]
* HumanMAC: Masked Motion Completion for Human Motion Prediction, ICCV 2023. [[paper](https://arxiv.org/abs/2302.03665)] [[code](https://github.com/LinghaoChan/HumanMAC)]
* BeLFusion: Latent Diffusion for Behavior-Driven Human Motion Prediction, ICCV 2023. [[paper](https://arxiv.org/abs/2211.14304)] [[code](https://github.com/BarqueroGerman/BeLFusion)]
* EigenTrajectory: Low-Rank Descriptors for Multi-Modal Trajectory Forecasting, ICCV 2023. [[paper](https://arxiv.org/abs/2307.09306)] [[code](https://github.com/InhwanBae/EigenTrajectory)]
* ADAPT: Efficient Multi-Agent Trajectory Prediction with Adaptation, ICCV 2023. [[paper](https://arxiv.org/pdf/2307.14187.pdf)] [[code](https://kuis-ai.github.io/adapt/)]
* LimSim: A Long-term Interactive Multi-scenario Traffic Simulator, ITSC 2023. [[paper](https://arxiv.org/pdf/2307.06648.pdf)] [[code](https://github.com/PJLab-ADG/LimSim)]
* V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting, CVPR 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_V2X-Seq_A_Large-Scale_Sequential_Dataset_for_Vehicle-Infrastructure_Cooperative_Perception_and_CVPR_2023_paper.pdf)] [[code](https://github.com/AIR-THU/DAIR-V2X-Seq)]
* INT2: Interactive Trajectory Prediction at Intersections, ICCV 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_INT2_Interactive_Trajectory_Prediction_at_Intersections_ICCV_2023_paper.pdf)] [[code](https://github.com/AIR-DISCOVER/INT2)]
* Trajectory Unified Transformer for Pedestrian Trajectory Prediction, ICCV 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf)]
* Sparse Instance Conditioned Multimodal Trajectory Prediction, ICCV 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Dong_Sparse_Instance_Conditioned_Multimodal_Trajectory_Prediction_ICCV_2023_paper.pdf)]
* MotionLM: Multi-Agent Motion Forecasting as Language Modeling, ICCV 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Seff_MotionLM_Multi-Agent_Motion_Forecasting_as_Language_Modeling_ICCV_2023_paper.pdf)]
* ADAPT: Action-aware Driving Caption Transformer, ICRA 2023. [[paper](https://browse.arxiv.org/pdf/2302.00673.pdf)] [[code](https://github.com/jxbbb/ADAPT)]
* Scenario Diffusion: Controllable Driving Scenario Generation With Diffusion, NIPS 2023. [[paper](https://openreview.net/pdf?id=99MHSB98yZ)]

## Journal Papers 2023
* MVHGN: Multi-View Adaptive Hierarchical Spatial Graph Convolution Network Based Trajectory Prediction for Heterogeneous Traffic-Agents, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10056303)]
* Adaptive and Simultaneous Trajectory Prediction for Heterogeneous Agents via Transferable Hierarchical Transformer Network, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10149109)]
* SSAGCN: Social Soft Attention Graph Convolution Network for Pedestrian Trajectory Prediction, TNNLS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10063206)] [[code](https://github.com/WW-Tong/ssagcn_for_path_prediction)]
* Disentangling Crowd Interactions for Pedestrians Trajectory Prediction, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10083225)]
* VNAGT: Variational Non-Autoregressive Graph Transformer Network for Multi-Agent Trajectory Prediction, IEEE Transactions on Vehicular Technology. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10121688)]
* Spatial-Temporal-Spectral LSTM: A Transferable Model for Pedestrian Trajectory Prediction, TIV. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10149368)]
* Holistic Transformer: A Joint Neural Network for Trajectory Prediction and Decision-Making of Autonomous Vehicles, PR. [[paper](https://www.sciencedirect.com/science/article/pii/S0031320323002935)]
* Tri-HGNN: Learning triple policies fused hierarchical graph neural networks for pedestrian trajectory prediction, PR. [[paper](https://www.sciencedirect.com/science/article/pii/S0031320323004703)]
* Multimodal Vehicular Trajectory Prediction With Inverse Reinforcement Learning and Risk Aversion at Urban Unsignalized Intersections, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10164651)]
* Trajectory prediction for autonomous driving based on multiscale spatial‐temporal graph, IET Intelligent Transport Systems. [[paper](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/itr2.12265)]
* Social Self-Attention Generative Adversarial Networks for Human Trajectory Prediction, IEEE Transactions on Artificial Intelligence. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10197467)]
* CSIR: Cascaded Sliding CVAEs With Iterative Socially-Aware Rethinking for Trajectory Prediction, TITS. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10215313)]
* Multimodal Manoeuvre and Trajectory Prediction for Automated Driving on Highways Using Transformer Networks, RAL. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10207845)]
* A physics-informed Transformer model for vehicle trajectory prediction on highways, Transportation Research Part C: Emerging Technologies. [[paper](https://www.sciencedirect.com/science/article/pii/S0968090X23002619)] [[code](https://github.com/Gengmaosi/PIT-IDM)]
* MacFormer: Map-Agent Coupled Transformer for Real-time and Robust Trajectory Prediction, RAL. [[paper](https://arxiv.org/pdf/2308.10280.pdf)]
* MRGTraj: A Novel Non-Autoregressive Approach for Human Trajectory Prediction, TCSVT. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10226250)] [[code](https://github.com/wisionpeng/MRGTraj)]
* Planning-inspired Hierarchical Trajectory Prediction via Lateral-Longitudinal Decomposition for Autonomous Driving, TIV. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10226224)]
* A multi-modal vehicle trajectory prediction framework via conditional diffusion model: A coarse-to-fine approach, KBS. [[paper](https://www.sciencedirect.com/science/article/pii/S0950705123007402)]
* Modality Exploration, Retrieval and Adaptation for Trajectory Prediction, TPAMI. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10254381)]
* MFAN: Mixing Feature Attention Network for Trajectory Prediction, PR. [[paper](https://www.sciencedirect.com/science/article/pii/S0031320323006957#abs0001)]
* IE-GAN: a data-driven crowd simulation method via generative adversarial networks, Multimedia Tools and Applications. [[paper](https://link.springer.com/article/10.1007/s11042-023-17346-x)]

## Others 2023
* Traj-MAE: Masked Autoencoders for Trajectory Prediction, arXiv preprint arXiv:2303.06697, 2023. [[paper](https://arxiv.org/pdf/2303.06697.pdf)]
* Uncertainty-Aware Pedestrian Trajectory Prediction via Distributional Diffusion, arXiv preprint arXiv:2303.08367, 2023. [[paper](https://arxiv.org/pdf/2303.08367.pdf)]
* DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model, arXiv preprint arXiv:2304.11582, 2023. [[paper](https://arxiv.org/pdf/2304.11582.pdf)] [[code](https://github.com/Yasoz/DiffTraj)]
* Multiverse Transformer: 1st Place Solution for Waymo Open Sim Agents Challenge 2023, CVPR 2023 Workshop on Autonomous Driving. [[paper](https://arxiv.org/pdf/2306.11868.pdf)] [[website](https://multiverse-transformer.github.io/sim-agents/)]
* Joint-Multipath++ for Simulation Agents: 2nd Place Solution for Waymo Open Sim Agents Challenge 2023, CVPR 2023 Workshop on Autonomous Driving. [[paper](https://storage.googleapis.com/waymo-uploads/files/research/2023%20Technical%20Reports/SA_hm_jointMP.pdf)] [[code](https://github.com/wangwenxi-handsome/Joint-Multipathpp)]
* MTR++: Multi-Agent Motion Prediction with Symmetric Scene Modeling and Guided Intention Querying, 1st Place Solution for Waymo Open Motion Prediction Challenge 2023, CVPR 2023 Workshop on Autonomous Driving. [[paper](https://arxiv.org/pdf/2306.17770.pdf)] [[code](https://github.com/sshaoshuai/MTR)]
* GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving, arXiv preprint arXiv:2303.05760, 2023. [[paper](https://arxiv.org/pdf/2303.05760.pdf)] [[code](https://github.com/MCZhi/GameFormer)] [[website](https://mczhi.github.io/GameFormer/)]
* GameFormer Planner: A Learning-enabled Interactive Prediction and Planning Framework for Autonomous Vehicles, the nuPlan Planning Challenge at the CVPR 2023 End-to-End Autonomous Driving Workshop. [[paper](https://opendrivelab.com/e2ead/AD23Challenge/Track_4_AID.pdf)] [[code](https://github.com/MCZhi/GameFormer-Planner/)]
* trajdata: A Unified Interface to Multiple Human Trajectory Datasets, arXiv preprint arXiv:2307.13924, 2023. [[paper](https://arxiv.org/pdf/2307.13924.pdf)] [[code](https://github.com/NVlabs/trajdata)]
* Graph-Based Interaction-Aware Multimodal 2D Vehicle Trajectory Prediction using Diffusion Graph Convolutional Networks, arXiv preprint arXiv:2309.01981, 2023. [[paper](https://arxiv.org/pdf/2309.01981.pdf)]
* EquiDiff: A Conditional Equivariant Diffusion Model For Trajectory Prediction, arXiv preprint arXiv:2308.06564, 2023. [[paper](https://arxiv.org/pdf/2308.06564.pdf)]
* DICE: Diverse Diffusion Model with Scoring for Trajectory Prediction, arXiv preprint arXiv:2310.14570, 2023. [[paper](https://arxiv.org/pdf/2310.14570.pdf)]
* Pedestrian Trajectory Prediction Using Dynamics-based Deep Learning, arXiv preprint arXiv:2309.09021, 2023. [[paper](https://browse.arxiv.org/pdf/2309.09021.pdf)] [[code](https://github.com/sydney-machine-learning/pedestrianpathprediction)]
* DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving, arXiv preprint arXiv:2309.09777, 2023. [[paper](https://arxiv.org/pdf/2309.09777.pdf)] [[website](https://drivedreamer.github.io/)]
* Language Prompt for Autonomous Driving, arXiv preprint arXiv:2309.04379, 2023. [[paper](https://arxiv.org/pdf/2309.04379.pdf)] [[code](https://github.com/wudongming97/Prompt4Driving)]
* GAIA-1: A Generative World Model for Autonomous Driving, arXiv preprint arXiv:2309.17080, 2023. [[paper](https://browse.arxiv.org/pdf/2309.17080.pdf)] [[website](https://wayve.ai/thinking/scaling-gaia-1/)]
* LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving, arXiv preprint arXiv:2310.03026, 2023. [[paper](https://arxiv.org/pdf/2310.03026.pdf)] [[website](https://sites.google.com/view/llm-mpc)]
* DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model, arXiv preprint arXiv:2310.01412, 2023. [[paper](https://browse.arxiv.org/pdf/2310.01412.pdf)] [[website](https://tonyxuqaq.github.io/projects/DriveGPT4/)]
* Drive Like a Human: Rethinking Autonomous Driving with Large Language Models, arXiv preprint arXiv:2307.07162, 2023. [[paper](https://arxiv.org/pdf/2307.07162.pdf)] [[code](https://github.com/PJLab-ADG/DriveLikeAHuman)]
* DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models, arXiv preprint arXiv:2309.16292, 2023. [[paper](https://browse.arxiv.org/pdf/2309.16292.pdf)] [[website](https://pjlab-adg.github.io/DiLu/)]
* DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model, arXiv preprint arXiv:2310.07771, 2023. [[paper](https://arxiv.org/pdf/2310.07771.pdf)] [[website](https://drivingdiffusion.github.io/)]
* Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving, arXiv preprint arXiv:2310.01957, 2023. [[paper](https://browse.arxiv.org/pdf/2310.01957.pdf)] [[code](https://github.com/wayveai/Driving-with-LLMs)]
* WEDGE: A Multi-Weather Autonomous Driving Dataset Built From Generative Vision-Language Models, CVPR Workshops 2023. [[paper](https://arxiv.org/pdf/2305.07528.pdf)] [[website](https://infernolia.github.io/WEDGE)]
* BEVGPT: Generative Pre-trained Large Model for Autonomous Driving Prediction, Decision-Making, and Planning, arXiv preprint arXiv:2310.10357, 2023. [[paper](https://arxiv.org/pdf/2310.10357.pdf)]
* Diffusion World Models, ICLR 2024 Conference Submission, 2023. [[paper](https://openreview.net/pdf?id=bAXmvOLtjA)]
* Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research, arXiv preprint arXiv:2310.08710, 2023. [[paper](https://arxiv.org/pdf/2310.08710.pdf)] [[code](https://github.com/waymo-research/waymax)] [[website](https://waymo.com/intl/zh-cn/research/waymax/)]
* MagicDrive: Street View Generation with Diverse 3D Geometry Control, arXiv preprint arXiv:2310.02601, 2023. [[paper](https://arxiv.org/pdf/2310.02601.pdf)] [[website](https://gaoruiyuan.com/magicdrive/)]
* GPT-Driver: Learning to Drive with GPT, arXiv preprint arXiv:2310.01415, 2023. [[paper](https://arxiv.org/pdf/2310.01415.pdf)] [[code](https://github.com/PointsCoder/GPT-Driver)]
* Can you text what is happening? Integrating pre-trained language encoders into trajectory prediction models for autonomous driving, arXiv preprint arXiv:2309.05282, 2023. [[paper](https://arxiv.org/pdf/2309.05282.pdf)]
* HiLM-D: Towards High-Resolution Understanding in Multimodal Large Language Models for Autonomous Driving, arXiv preprint arXiv:2309.05186, 2023. [[paper](https://arxiv.org/pdf/2309.05186.pdf)]

# Related Review Papers
* Pedestrian and vehicle behaviour prediction in autonomous vehicle system — A review, Expert Systems With Applications 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S0957417423024855)]
* Data-driven Traffic Simulation: A Comprehensive Review, arXiv preprint arXiv:2310.15975, 2023. [[paper](https://arxiv.org/ftp/arxiv/papers/2310/2310.15975.pdf)]
* Pedestrian Trajectory Prediction in Pedestrian-Vehicle Mixed Environments: A Systematic Review, TITS 2023. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10181234)]
* Machine Learning for Autonomous Vehicle’s Trajectory Prediction: A comprehensive survey, Challenges, and Future Research Directions, arXiv preprint arXiv:2307.07527, 2023. [[paper](https://arxiv.org/pdf/2307.07527.pdf)]
* Incorporating Driving Knowledge in Deep Learning Based Vehicle Trajectory Prediction: A Survey, IEEE Transactions on Intelligent Vehicles 2023. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10100881)]
* A Survey of Vehicle Trajectory Prediction Based on Deep Learning Models, International Conference on Sustainable Expert Systems: ICSES 2022. [[paper](https://link.springer.com/chapter/10.1007/978-981-19-7874-6_48)]
* A Survey on Trajectory-Prediction Methods for Autonomous Driving, IEEE Transactions on Intelligent Vehicles 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9756903)]
* Generative Adversarial Networks for Spatio-temporal Data: A Survey, ACM Transactions on Intelligent Systems and Technology 2022. [[paper](https://dl.acm.org/doi/pdf/10.1145/3474838)]
* Scenario Understanding and Motion Prediction for Autonomous Vehicles – Review and Comparison, TITS 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9733973)]
* Deep Reinforcement Learning for Autonomous Driving: A Survey, TITS 2022. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9351818)]
* Social Interactions for Autonomous Driving: A Review and Perspective, arXiv preprint arXiv:2208.07541, 2022. [[paper](https://arxiv.org/pdf/2208.07541.pdf)]
* Behavioral Intention Prediction in Driving Scenes: A Survey, arXiv preprint arXiv:2211.00385, 2022. [[paper](https://arxiv.org/pdf/2211.00385.pdf)]
* Multi-modal Fusion Technology based on Vehicle Information: A Survey, arXiv preprint arXiv:2211.06080, 2022. [[paper](https://arxiv.org/pdf/2211.06080.pdf)]
* Pedestrian Behavior Prediction for Automated Driving: Requirements, Metrics, and Relevant Features, TITS 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9660784)]
* A Review of Deep Learning-Based Methods for Pedestrian Trajectory Prediction, Sensors 2021. [[paper](https://www.mdpi.com/1424-8220/21/22/7543/pdf)]
* A Survey on Deep-Learning Approaches for Vehicle Trajectory Prediction in Autonomous Driving, IEEE International Conference on Robotics and Biomimetics (ROBIO 2021). [[paper](https://arxiv.org/pdf/2110.10436.pdf)] [[code](https://github.com/Henry1iu/TNT-Trajectory-Predition)]
* Review of Pedestrian Trajectory Prediction Methods: Comparing Deep Learning and Knowledge-based Approaches, arXiv preprint arXiv:2111.06740, 2021. [[paper](https://arxiv.org/pdf/2111.06740.pdf)]
* A Survey on Trajectory Data Management, Analytics, and Learning, ACM Computing Surveys (CSUR 2021). [[paper](https://dl.acm.org/doi/pdf/10.1145/3440207)]
* A Survey on Motion Prediction of Pedestrians and Vehicles for Autonomous Driving, IEEE Access 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9559998)]
* Autonomous Driving with Deep Learning: A Survey of State-of-Art Technologies, arXiv preprint arXiv:2006.06091, 2020. [[paper](https://arxiv.org/ftp/arxiv/papers/2006/2006.06091.pdf)]
* A Survey on Visual Traffic Simulation: Models, Evaluations, and Applications in Autonomous Driving, Computer Graphics Forum 2020. [[paper](https://onlinelibrary.wiley.com/doi/epdf/10.1111/cgf.13803?saml_referrer)]
* A Survey of Deep Learning Techniques for Autonomous Driving, Journal of Field Robotics 2020. [[paper](https://onlinelibrary.wiley.com/doi/epdf/10.1002/rob.21918?saml_referrer)]
* Human Motion Trajectory Prediction: A Survey, International Journal of Robotics Research 2020. [[paper](http://sage.cnpereading.com/paragraph/download/?doi=10.1177/0278364920917446)]
* Vehicle Trajectory Similarity: Models, Methods, and Applications, ACM Computing Surveys (CSUR 2020). [[paper](https://dl.acm.org/doi/pdf/10.1145/3406096)]
* Deep Learning-Based Vehicle Behavior Prediction for Autonomous Driving Applications: A Review, TITS 2020. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9158529)]
* Survey of Deep Reinforcement Learning for Motion Planning of Autonomous Vehicles, TITS 2020. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9210154)]
* Overview of Tools Supporting Planning for Automated Driving, ITSC 2020. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9294512)]
* Autonomous Vehicles that Interact with Pedestrians: A Survey of Theory and Practice, TITS 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8667866)]
* A Survey on Path Prediction Techniques for Vulnerable Road Users: From Traditional to Deep-Learning Approaches, ITSC 2019. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8917053)]
* Spatio-Temporal Data Mining: A Survey of Problems and Methods, ACM Computing Surveys 2018. [[paper](https://dl.acm.org/doi/pdf/10.1145/3161602)]
* Survey on Vision-Based Path Prediction, International Conference on Distributed, Ambient, and Pervasive Interactions (DAPI 2018). [[paper](https://link.springer.com/content/pdf/10.1007/978-3-319-91131-1_4.pdf)]
* Moving Objects Analytics: Survey on Future Location & Trajectory Prediction Methods, arXiv preprint arXiv:1807.04639, 2018. [[paper](https://arxiv.org/ftp/arxiv/papers/1807/1807.04639.pdf)]
* A Survey on Trajectory Data Mining: Techniques and Applications, IEEE Access 2016. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7452339)]
* Trajectory Data Mining: An Overview, ACM Transactions on Intelligent Systems and Technology 2015. [[paper](http://urban-computing.com/pdf/TrajectoryDataMining-tist-yuzheng.pdf)]
* A survey on motion prediction and risk assessment for intelligent vehicles, ROBOMECH Journal 2014. [[paper](https://robomechjournal.springeropen.com/track/pdf/10.1186/s40648-014-0001-z.pdf)]

# Datasets
## Vehicles Publicly Available Datasets
* [Porto](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data), [website](https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+-+Prediction+Challenge,+ECML+PKDD+2015)
* [NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
* [NYC](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
* [T-drive](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)
* [Greek Trucks](http://www.chorochronos.org/)
* [highD](https://www.highd-dataset.com/)
* [inD](https://www.ind-dataset.com/)
* [rounD](https://www.round-dataset.com/)
* [uniD](https://www.unid-dataset.com/)
* [exiD](https://www.exid-dataset.com/)
* [Mirror-Traffic](http://www.scenarios.cn/html/dataset.html)
* [Argoverse Website](https://www.argoverse.org/), [Argoverse 1](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.pdf), [Argoverse 2](https://arxiv.org/pdf/2301.00493.pdf)
* [ApolloScape](http://apolloscape.auto/trajectory.html)
* [INTERACTION](https://interaction-dataset.com/)
* [Waymo Open Dataset](https://waymo.com/open/)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/)
* [nuScenes](https://www.nuscenes.org/)
* [TRAF](https://gamma.umd.edu/researchdirections/autonomousdriving/trafdataset)
* [Lyft Level 5](https://level-5.global/)
* [METEOR](https://gamma.umd.edu/researchdirections/autonomousdriving/meteor/)
* [DiDi GAIA](https://outreach.didichuxing.com/research/opendata/), [D²-City](https://www.scidb.cn/en/detail?dataSetId=804399692560465920&dataSetType=personal), [paper](https://arxiv.org/pdf/1904.01975)
* [Shanghai & Hangzhou](https://dl.acm.org/doi/abs/10.1145/2700478)
* [Beijing](https://dl.acm.org/doi/10.1145/2525314.2525343)
* [VMT](https://ieeexplore.ieee.org/document/6482546)
* [TRAFFIC](https://ieeexplore.ieee.org/document/7565640), [website](https://min.sjtu.edu.cn/lwydemo/Trajectory%20analysis.htm)
* [CROSS](https://cvrr-nas.ucsd.edu/publications/2011/Morris_PAMI2011.pdf), [website](http://cvrr.ucsd.edu/bmorris/datasets/)
* [Ubiquitous Traffic Eyes (UTE)](http://seutraffic.com/#/home)
## Pedestrians Publicly Available Datasets
* [GeoLife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)
* [UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data)
* [ETH](https://icu.ee.ethz.ch/research/datsets.html), [paper](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/pellegrini09iccv.pdf)
* [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/)
* [TrajNet](http://trajnet.stanford.edu/)
* [Oxford Town Center](https://exposing.ai/oxford_town_centre/)
* [New York Grand Central Station](https://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html), [paper](https://ieeexplore.ieee.org/abstract/document/5995459), [paper](https://people.csail.mit.edu/bzhou/project/cvpr2012/zhoucvpr2012.pdf), [paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Yi_Understanding_Pedestrian_Behaviors_2015_CVPR_paper.pdf)
* [PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)
* [JAAD](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
* [DS4C-PPP](https://www.kaggle.com/datasets/kimjihoo/coronavirusdataset)
* [BDBC COVID-19](https://github.com/BDBC-KG-NLP/COVID-19-tracker)
## Others Agents Datasets
### Aircraft
* [LocaRDS](https://atmdata.github.io/)
* [ZUMAVD](https://rpg.ifi.uzh.ch/zurichmavdataset.html)
### Ship
* [Ushant](https://figshare.com/articles/dataset/Ushant_AIS_dataset/8966273)
* [Cargo](https://link.springer.com/article/10.1007/s10707-020-00421-y)
### Hurricane and Animal
* [HURDAT2](https://www.nhc.noaa.gov/data/)
* [Movebank](https://www.movebank.org/cms/movebank-main)
