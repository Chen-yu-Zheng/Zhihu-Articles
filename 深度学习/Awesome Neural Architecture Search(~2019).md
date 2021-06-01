#! https://zhuanlan.zhihu.com/p/372915108
# Awesome Neural Architecture Search（~2019）

先收集一波论文，暑假来填坑。争取暑假能得到一些有质地的进步，希望下半年出一篇自己的论文（不管它多垃圾了），把大创完美推进吧。参考Neural Architecture Search: A Survey，JMLR,2019。

## 图像分类

* Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures for scalable image recognition. In Conference on Computer Vision and Pattern Recognition, 2018.
* Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Aging Evolution for Image Classifier Architecture Search. In AAAI, 2019.

## 搜索空间

### 加入跳跃和并行结构（覆盖GoogleNet,ResNet,DenseNet)

* Andrew Brock, Theodore Lim, James M. Ritchie, and Nick Weston. SMASH: one-shot model architecture search through hypernetworks. In NIPS Workshop on Meta-Learning, 2017.
* Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Simple And Efficient Architecture Search for Convolutional Neural Networks. In NIPS Workshop on Meta-Learning, 2017.
* Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures for scalable image recognition. In Conference on Computer Vision and Pattern Recognition, 2018.
* Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Efficient multi-objective neural architecture search via lamarckian evolution. In International Conference on Learning Representations, 2019.
* Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Aging Evolution for Image Classifier Architecture Search. In AAAI, 2019.
* Han Cai, Jiacheng Yang, Weinan Zhang, Song Han, and Yong Yu. Path-Level Network Transformation for Efficient Architecture Search. In International Conference on Machine Learning, June 2018b.

### 重复cell block

* Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures for scalable image recognition. In Conference on Computer Vision and Pattern Recognition, 2018.（normal block and reduced block）
* Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Aging Evolution for Image Classifier Architecture Search. In AAAI, 2019.
* Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In International Conference on Learning Representations, 2019b.
* Zhao Zhong, Zichen Yang, Boyang Deng, Junjie Yan, Wei Wu, Jing Shao, and ChengLin Liu. Blockqnn: Efficient block-wise neural network architecture generation. arXiv preprint, 2018b.

### 微观-宏观架构搜索

* Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures for scalable image recognition. In Conference on Computer Vision and Pattern Recognition, 2018.（线性堆叠）
* Han Cai, Jiacheng Yang, Weinan Zhang, Song Han, and Yong Yu. Path-Level Network Transformation for Efficient Architecture Search. In International Conference on Machine Learning, June 2018b.（在现有框架下置换cell）
* Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, and Koray Kavukcuoglu. Hierarchical Representations for Efficient Architecture Search. In International Conference on Learning Representations, 2018b.（层次化搜索）

## 搜索策略

### 贝叶斯优化

贝叶斯优化是超参数优化领域最热门的算法之一。传统的贝叶斯优化基于高斯过程且针对低维的连续优化问题。应用在NAS中做了一系列改进。

* James Bergstra, Dan Yamins, and David D. Cox. Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. In ICML, 2013.（SOTA)
* T. Domhan, J. T. Springenberg, and F. Hutter. Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves. In Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), 2015. (SOTA performance for CIFAR-10 without data augmentation) 
* H. Mendoza, A. Klein, M. Feurer, J. Springenberg, and F. Hutter. Towards AutomaticallyTuned Neural Networks. In International Conference on Machine Learning, AutoML Workshop, June 2016. (first automaticallytuned neural networks to win on competition data sets against human experts)
* Kevin Swersky, David Duvenaud, Jasper Snoek, Frank Hutter, and Michael Osborne. Raiders of the lost architecture: Kernels for bayesian optimization in conditional parameter spaces. In NIPS Workshop on Bayesian Optimization in Theory and Practice, 2013.（GP）
* Kirthevasan Kandasamy, Willie Neiswanger, Jeff Schneider, Barnabas Poczos, and Eric P Xing. Neural architecture search with bayesian optimisation and optimal transport. In Advances in Neural Information Processing Systems 31. 2018.（GP)
* James S. Bergstra, R´emi Bardenet, Yoshua Bengio, and Bal´azs K´egl. Algorithms for hyperparameter optimization. In J. Shawe-Taylor, R. S. Zemel, P. L. Bartlett, F. Pereira, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 24, pages 2546–2554, 2011.（Tree）
* F. Hutter, H. Hoos, and K. Leyton-Brown. Sequential model-based optimization for general algorithm configuration. In LION, pages 507–523, 2011.（Tree，随机森林）
* James Bergstra, Dan Yamins, and David D. Cox. Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. In ICML, 2013.
* T. Domhan, J. T. Springenberg, and F. Hutter. Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves. In Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), 2015.
* H. Mendoza, A. Klein, M. Feurer, J. Springenberg, and F. Hutter. Towards AutomaticallyTuned Neural Networks. In International Conference on Machine Learning, AutoML Workshop, June 2016.
* Arber Zela, Aaron Klein, Stefan Falkner, and Frank Hutter. Towards automated deep learning: Efficient joint neural architecture and hyperparameter search. In ICML 2018 Workshop on AutoML (AutoML 2018), 2018.
* R. Negrinho and G. Gordon. DeepArchitect: Automatically Designing and Training Deep Architectures. arXiv:1704.08792, 2017.（Tree，MTCS)
* Martin Wistuba. Finding Competitive Network Architectures Within a Day Using UCT. In arXiv:1712.07420, December 2017.（Tree，MTCS)
* Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Simple And Efficient Architecture Search for Convolutional Neural Networks. In NIPS Workshop on Meta-Learning, 2017.（爬山法）

### 强化学习（策略和优化的进步）

* Bowen Baker, Otkrist Gupta, Nikhil Naik, and Ramesh Raskar. Designing neural network architectures using reinforcement learning. In International Conference on Learning Representations, 2017a（Q-learning）
* Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. In International Conference on Learning Representations, 2017.（800GPUs，一个月，recurrent neural network (RNN) policy，REINFORCE policy gradient algorithm）
* Zhao Zhong, Junjie Yan, Wei Wu, Jing Shao, and Cheng-Lin Liu. Practical block-wise neural network architecture generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2423–2432, 2018a.
* Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures for scalable image recognition. In Conference on Computer Vision and Pattern Recognition, 2018.（Proximal Policy Optimization）
* Han Cai, Tianyao Chen, Weinan Zhang, Yong Yu, and Jun Wang. Efficient architecture search by network transformation. In Association for the Advancement of Artificial Intelligence, 2018a.（决策过程）

### 演化计算（算了，不想搞）

### 转离散为连续，应用梯度

* Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In International Conference on Learning Representations, 2019b.（optimizing a weighting α of possible operations）
* Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. SNAS: stochastic neural architecture search. In International Conference on Learning Representations, 2019.（optimize a parametrized distribution over the possible operations）
* Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In International Conference on Learning Representations, 2019.（optimize a parametrized distribution over the possible operations）
* Richard Shin, Charles Packer, and Dawn Song. Differentiable neural network architecture search. In International Conference on Learning Representations Workshop, 2018.（optimizing layer hyperparameters）
* Karim Ahmed and Lorenzo Torresani. Maskconnect: Connectivity learning by gradient descent. In European Conference on Computer Vision (ECCV), 2018.（connectivity patterns）

### 一些比较实验

* Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V. Le. Aging Evolution for Image Classifier Architecture Search. In AAAI, 2019. （强化学习，随机搜索，演化计算）
* Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, and Koray Kavukcuoglu. Hierarchical Representations for Efficient Architecture Search. In International Conference on Learning Representations, 2018b.（随即搜索，演化计算）
* Aaron Klein, Eric Christiansen, Kevin Murphy, and Frank Hutter. Towards reproducible neural architecture and hyperparameter search. In ICML 2018 Workshop on Reproducibility in ML (RML 2018), 2018.（贝叶斯优化与演化算法）

## 评估策略

如果每个模型都在训练集上训练，然后在验证集上得到分数，计算量将会异常恐怖（17强化学习，800GPUs，一个月），所以更换评估策略是非常必要的。

### lower fidelities

减少计算消耗，但可能存在估计与真实情况相差较大的情况

* Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures for scalable image recognition. In Conference on Computer Vision and Pattern Recognition, 2018.（训练短时间）
* Arber Zela, Aaron Klein, Stefan Falkner, and Frank Hutter. Towards automated deep learning: Efficient joint neural architecture and hyperparameter search. In ICML 2018 Workshop on AutoML (AutoML 2018), 2018.（训练短时间）
* Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter. Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets. In Aarti Singh and Jerry Zhu, editors, Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pages 528–536, Fort Lauderdale, FL, USA, 20–22 Apr 2017b. PMLR.（在subset上训练）
* Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of imagenet as an alternative to the CIFAR datasets. CoRR, abs/1707.08819, 2017.（使用低分辨率的图像）
* Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hyperband: bandit-based configuration evaluation for hyperparameter optimization. In International Conference on Learning Representations, 2017.（要加强保真度，防止区别过大）
* Stefan Falkner, Aaron Klein, and Frank Hutter. BOHB: Robust and efficient hyperparameter optimization at scale. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 1436–1445, Stockholmsmssan, Stockholm Sweden, 10–15 Jul 2018. PMLR.（要加强保真度，防止区别过大）

### learning curve extrapolation 

预测神经体系结构性能的主要挑战是，为了加快搜索过程，需要基于相对较少的评估来在相对较大的搜索空间中进行良好的预测。

* T. Domhan, J. T. Springenberg, and F. Hutter. Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves. In Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), 2015.（通过预测学习曲线，终止那些比较差的网络）
* A. Klein, S. Falkner, J. T. Springenberg, and F. Hutter. Learning curve prediction with Bayesian neural networks. In International Conference on Learning Representations, 2017a.（选择有希望的网络）
* Bowen Baker, Otkrist Gupta, Ramesh Raskar, and Nikhil Naik. Accelerating Neural Architecture Search using Performance Prediction. In NIPS Workshop on Meta-Learning, 2017b.（选择有希望的网络）
* Aditya Rawal and Risto Miikkulainen. From Nodes to Networks: Evolving Recurrent Neural Networks. In arXiv:1803.04439, March 2018.（选择有希望的网络）
* Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li FeiFei, Alan Yuille, Jonathan Huang, and Kevin Murphy. Progressive Neural Architecture Search. In European Conference on Computer Vision, 2018a.（PNAS，训练相似的小模型预测大模型）

### Weight Inheritance/ Network Morphisms

在原有的网络基础上（继承参数、结构等）继续训练，不需要从头开始，大大减小时间。但没有限制的持续增加模型大小也会导致过拟合，需要一定限制。

* Tao Wei, Changhu Wang, Yong Rui, and Chang Wen Chen. Network morphism. In International Conference on Machine Learning, 2016.（用先前训练的模型参数来初始化新模型）
* Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Simple And Efficient Architecture Search for Convolutional Neural Networks. In NIPS Workshop on Meta-Learning, 2017.
* Han Cai, Tianyao Chen, Weinan Zhang, Yong Yu, and Jun Wang. Efficient architecture search by network transformation. In Association for the Advancement of Artificial Intelligence, 2018a.
* Haifeng Jin, Qingquan Song, and Xia Hu. Auto-keras: Efficient neural architecture search with network morphism, 2018.
* Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. Efficient multi-objective neural architecture search via lamarckian evolution. In International Conference on Learning Representations, 2019.（限制网络的不断加大）

### One-Shot Models/ Weight Sharing

认为所有模型都来自于一个超网络（one-shot model）的子图。只需要训练这个超网络，且不中断训练，将其子图取出做评估，并优化这个取子图的方式。The one-shot model typically incurs a large bias as it underestimates the actual performance of the best architectures severely; nevertheless, it allows ranking architectures, which would be sufficient if the estimated performance correlates strongly with the actual performance. **However, it is currently not clear if this is actually the case** (Bender et al., 2018; Sciuto et al., 2019).

* Shreyas Saxena and Jakob Verbeek. Convolutional neural fabrics. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems 29, pages 4053–4061. Curran Associates, Inc., 2016.
* Andrew Brock, Theodore Lim, James M. Ritchie, and Nick Weston. SMASH: one-shot model architecture search through hypernetworks. In NIPS Workshop on Meta-Learning, 2017.
* Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, and Jeff Dean. Efficient neural architecture search via parameter sharing. In International Conference on Machine Learning, 2018.（RNN取网络，one-shot模型采用近似梯度训练）
* Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In International Conference on Learning Representations, 2019b.（在one-shot模型的每一条边上混合运算，构造连续梯度，优化各个运算的权重）
* Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, and Quoc Le. Understanding and simplifying one-shot architecture search. In International Conference on Machine Learning, 2018.（only train the one-shot model once and show that this is sufficient when deactivating parts of this model stochastically during training using path dropout. fixed distribution,amazing）
* Han Cai, Ligeng Zhu, and Song Han. ProxylessNAS: Direct neural architecture search on target task and hardware. In International Conference on Learning Representations, 2019.（防止把one-shot模型一直全放在GPU中，每次加载一条边，加载哪一条边的概率分布可学习）
* Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. SNAS: stochastic neural architecture search. In International Conference on Learning Representations, 2019.（优化各个运算的分布，离散分布和reparametrization，使其可微，然后使用梯度下降）
* Christian Sciuto, Kaicheng Yu, Martin Jaggi, Claudiu Musat, and Mathieu Salzmann. Evaluating the search phase of neural architecture search. arXiv preprint, 2019.（分析one-shot取样引入的偏差）
* Andrew Brock, Theodore Lim, James M. Ritchie, and Nick Weston. SMASH: one-shot model architecture search through hypernetworks. In NIPS Workshop on Meta-Learning, 2017.（meta-learning）
* Chris Zhang, Mengye Ren, and Raquel Urtasun. Graph hypernetworks for neural architecture search. In International Conference on Learning Representations, 2019.（meta-learning）

## Future Direction

### 领域扩展

* **语义分割**
  * Liang-Chieh Chen, Maxwell Collins, Yukun Zhu, George Papandreou, Barret Zoph, Florian Schroff, Hartwig Adam, and Jon Shlens. Searching for efficient multi-scale architectures for dense image prediction. In S. Bengio, H.Wallach, H. Larochelle, K. Grauman, N. CesaBianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 8713–8724. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/ 8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction.pdf.
  * Vladimir Nekrasov, Hao Chen, Chunhua Shen, and Ian D. Reid. Fast neural architecture search of compact semantic segmentation models via auxiliary cells. arXiv preprint, 2018.（语义分割）
  * Chenxi Liu, Liang-Chieh Chen, Florian Schroff, Hartwig Adam, Wei Hua, Alan Yuille, and Li Fei-Fei. Auto-deeplab: Hierarchical neural architecture search for semantic image segmentation. arXiv preprint, 2019a.
* **迁移学习**
  * Catherine Wong, Neil Houlsby, Yifeng Lu, and Andrea Gesmundo. Transfer learning with neural automl. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 8366–8375. Curran Associates, Inc., 2018.
* **RNN/NLP**
  * Klaus Greff, Rupesh Kumar Srivastava, Jan Koutnk, Bas R. Steunebrink, and Jrgen Schmidhuber. Lstm: A search space odyssey. IEEE transactions on neural networks and learning systems, 28, 2015.
  * Rafal Jozefowicz, Wojciech Zaremba, and Ilya Sutskever. An empirical exploration of recurrent network architectures. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2342–2350, Lille, France, 07–09 Jul 2015. PMLR.
  * Aditya Rawal and Risto Miikkulainen. From Nodes to Networks: Evolving Recurrent Neural Networks. In arXiv:1803.04439, March 2018.

### 多任务

* Jason Liang, Elliot Meyerson, and Risto Miikkulainen. Evolutionary Architecture Search For Deep Multitask Networks. In arXiv:1803.03745, March 2018.
* Elliot Meyerson and Risto Miikkulainen. Pseudo-task Augmentation: From Deep Multitask Learning to Intratask Sharing and Back. In arXiv:1803.03745, March 2018.

### 多目标-网络压缩

* Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations, 2016.
* Yu Cheng, Duo Wang, Pan Zhou, and Tao Zhang. Model compression and acceleration for deep neural networks: The principles, progress, and challenges. IEEE Signal Process. Mag., 35(1):126–136, 2018.
* Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 1135–1143. Curran Associates, Inc., 2015. URL http://papers.nips.cc/ paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf.
* Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. 2017 IEEE International Conference on Computer Vision (ICCV), pages 2755–2763, 2017.
* Ariel Gordon, Elad Eban, Ofir Nachum, Bo Chen, Hao Wu, Tien-Ju Yang, and Edward Choi. Morphnet: Fast and simple resource-constrained structure learning of deep networks. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.
* Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. Rethinking the value of network pruning. In International Conference on Learning Representations, 2019c.
* Shengcao Cao, Xiaofang Wang, and Kris M. Kitani. Learnable embedding space for efficient neural architecture compression. In International Conference on Learning Representations, 2019.
* Shreyas Saxena and Jakob Verbeek. Convolutional neural fabrics. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems 29, pages 4053–4061. Curran Associates, Inc., 2016.
* Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In International Conference on Learning Representations, 2019b.
* Sirui Xie, Hehui Zheng, Chunxiao Liu, and Liang Lin. SNAS: stochastic neural architecture search. In International Conference on Learning Representations, 2019.

### 搜索空间

cell block搜索方式多适用于图像分类领域（本身已经技术成熟），但在其它领域还没有很好地得到应用（难迁移，难编码），比如语义分割和目标检测。Moreover, common search spaces are also based on predefined building blocks, such as different kinds of convolutions and pooling, but do not allow identifying novel building blocks on this level; going beyond this limitation might substantially increase the power of NAS.

* Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, and Koray Kavukcuoglu. Hierarchical Representations for Efficient Architecture Search. In International Conference on Learning Representations, 2018b.
* Chenxi Liu, Liang-Chieh Chen, Florian Schroff, Hartwig Adam, Wei Hua, Alan Yuille, and Li Fei-Fei. Auto-deeplab: Hierarchical neural architecture search for semantic image segmentation. arXiv preprint, 2019a.

### RL拓展

* Prajit Ramachandran and Quoc V. Le. Dynamic Network Architectures. In AutoML 2018 (ICML workshop), 2018.
* Ekin D. Cubuk, Barret Zoph, Samuel S. Schoenholz, and Quoc V. Le. Intriguing Properties of Adversarial Examples. In arXiv:1711.02846, November 2017.

### benchmark

tricks太多了，要形成一个统一的公平的比较标准

* Chris Ying, Aaron Klein, Esteban Real, Eric Christiansen, Kevin Murphy, and Frank Hutter. Nas-bench-101: Towards reproducible neural architecture search. arXiv preprint, 2019.
* Aaron Klein, Eric Christiansen, Kevin Murphy, and Frank Hutter. Towards reproducible neural architecture and hyperparameter search. In ICML 2018 Workshop on Reproducibility in ML (RML 2018), 2018.

### NAS得到的结构可解释性

蓝海，冲冲冲











