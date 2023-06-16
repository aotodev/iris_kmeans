K-Means Clustering
=====
[![CMake](https://github.com/aotodev/iris_kmeans/actions/workflows/cmake.yml/badge.svg)](https://github.com/aotodev/iris_kmeans/actions/workflows/cmake.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://git.stabletec.com/utilities/vksbc/blob/master/LICENSE)

Classifying the famous iris dataset using k-means clustering in C++ with SSE SIMD instructions

***

## Motivation
The IRIS dataset is probably one of the most popular datasets for testing clustering/unsupervised machine learning algorithms.<br />
By having 4 features (namely sepal length, sepal width, petal length, petal width), it seemed like the perfect dataset to build a model using Intel x86 SSE instructions as each observation would be 4 floats, matching perfectly to the instrinsic type __m128. <br />
The actual SSE code is implemented in the vec4 class, so that no intrinsic calls need to be made in the actual k-means implementation.<br />
<br />
The model does not classify all of the data points 100% correctly, but it does perform resonably well. A possible future improvement could be to change the centroids initialization algorithm, with technics such as k-means++.
***
