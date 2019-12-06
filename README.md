# SketchANetClustering

An interactive exploration of visual concepts in the Sketch-A-Net CNN (trained on the TU Berlin sketch dataset). Concepts were derived using K-Means++ clustering.

[Interactive Explorer](https://ulberge.github.io/SketchANetClustering/)

## Method
For each layer:
⋅⋅*Randomly select activations and their theoretical receptive field
⋅⋅*Run K-Means++ at various K
⋅⋅⋅*Save cluster data, top 100 matches and their avg
⋅⋅*Select clusters with semantic and visual coherence

## Sources
J. Wang, Z. Zhang, C. Xie, V. Premachandran, and A. Yuille, “Unsupervised learning of object semantic parts from internal states of cnns by population encoding,” arXiv preprint arXiv:1511.06855, 2015.

M. Eitz, J. Hays, and M. Alexa, “How do humans sketch objects?,” ACM Trans. Graph., vol. 31, no. 4, pp. 44–1, 2012.

Q. Yu, Y. Yang, F. Liu, Y.-Z. Song, T. Xiang, and T. M. Hospedales, “Sketch-a-Net: A Deep Neural Network that Beats Humans,” Int J Comput Vis, vol. 122, no. 3, pp. 411–425, May 2017.

