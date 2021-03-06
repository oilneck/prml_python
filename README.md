# PRML
These are codes implementing some algorithms introduced in  "Pattern Recognition and Machine Learning" (Author: [C.M.Bishop](https://www.microsoft.com/en-us/research/people/cmbishop/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fcmbishop%2Fprml%2F)). Python language used for these implementation.
## Required packages
- python 3
- numpy
- pandas
- scipy
- matplotlib
## Installation
1. Download the file to a local folder (e.g. ~/prml_python/) by executing:  
```console
git clone https://github.com/oilneck/prml_python.git
```

2. Run Python and change your directory (~/prml_python/), then run the **`init.py`** script.  

3. Run some demonstration files in Chap1~Chap11 folder.
## Execution example
|<img src="https://user-images.githubusercontent.com/60212785/74105720-b1070080-4ba3-11ea-90b3-e56cb9769cf4.png" width="250px" alt="section1">|<img src="https://user-images.githubusercontent.com/60212785/74105786-41dddc00-4ba4-11ea-9c01-ebb161f89986.png" width="210px">|<img src="https://user-images.githubusercontent.com/60212785/74105498-b6fbe200-4ba1-11ea-9b99-879ecac3d67c.png" width="250px">|
|:---:|:---:|:---:|
|chap.1 : bayesian fitting|chap.4 : logistic regression|chap.5 : neural network|

## Jupyter Notebook files
> The contents of Pattern Recognition and Machine Learning
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter01.ipynb">chap.1 : Introduction</a>
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter02.ipynb">chap.2 : Probability distributions </a>
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter03.ipynb">chap.3 : Linear Models for Regression </a>
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter04.ipynb">chap.4 : Linear Models for Classification </a>
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter05.ipynb">chap.5 : Neural Networks </a>
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter06.ipynb">chap.6 : Kernel Methods </a>
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter07.ipynb">chap.7 : Sparse Kernel Machines </a>
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter08.ipynb">chap.8 : Graphical Models
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter09.ipynb">chap.9 : Mixture Models and EM </a>
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter10.ipynb">chap.10 : Approximate Inference </a>
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/chapter11.ipynb">chap.11 : Sampling Methods </a>
 > Deep learning and Convolutional neural network for image recognition
 - <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/test_CNN_keras.ipynb"> Image recognition </a>【 Required libraries : keras, TensorFlow, OpenCV 】
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/test_Deep_learning.ipynb"><div style="text-align: justify"> Deep learning test</dim> </a> 【 Required libraries : numpy, sklearn (←to fetch data) 】
- <a href="https://nbviewer.jupyter.org/github/oilneck/prml_python/blob/master/Notebook/simple_CNN_model.ipynb"> Sequential models </a> 【 Required libraries : numpy 】


### NOTICE
All sources in [~/prml_python/prml] are the module file. If you want to change certain parameters (ex. iteration number, activation function in each layer for Neural Network), check the files in that directory.
## External links
<table class="table table-hover"></td>
<tbody>
<tr>
    <th>Wiki</th>
  <td align="center" valign="top"><a href="https://github.com/oilneck/prml_python/wiki/Pattern-Recognition-and-Machine-Learning">Wiki for prml algorithm</a>
    </td>
</tr>
<tr>
  <th>Text</tx>
  <td align="center" valign="top"><a href="https://wixlabs-pdf-dev.appspot.com/assets/pdfjs/web/viewer.html?file=%2Fpdfproxy%3Finstance%3DPhAPrWQZ4rfZkxO607vJvgQBVJ-6erZwrBa0iW6P2iU.eyJpbnN0YW5jZUlkIjoiZWU1M2FhOTctZWJjMS00NjIwLTk5NDQtYWU3MjVmMjA0ZjM3IiwiYXBwRGVmSWQiOiIxM2VlMTBhMy1lY2I5LTdlZmYtNDI5OC1kMmY5ZjM0YWNmMGQiLCJtZXRhU2l0ZUlkIjoiM2EwMmU4ZTUtYzAzMS00ZTIxLWE3ZjItOTUyYzZmYzk2NTQ0Iiwic2lnbkRhdGUiOiIyMDIwLTAzLTI4VDE5OjAxOjQ5LjkxNVoiLCJkZW1vTW9kZSI6ZmFsc2UsImFpZCI6IjI2Yjc4MDE0LTllYTktNGNlMi04MTllLTkyODM5MzMxN2IxYyIsImJpVG9rZW4iOiJkNDUxNDI3Mi0yYmYwLTA4MDEtM2ViNi0zYjVlMzBlOTJhNzMiLCJzaXRlT3duZXJJZCI6IjVkNTdjNjUwLTA1YTktNDFiNS1iMmFiLTEyNTkxMGE5Zjk4ZCJ9%26compId%3Dcomp-k8bzb1s7%26url%3Dhttps%3A%2F%2Fdocs.wixstatic.com%2Fugd%2F5d57c6_f680a0fce2ee45b28726639a77613f7f.pdf#page=1&links=true&originalFileName=RegularizationMethod_NN&locale=ja&allowDownload=false&allowPrinting=true">Regularization_of_NN.pdf</a>
         </td>
  </tr>
  <tr>
  <th>Slide</tx>
  <td align="center" valign="top"><a href="https://wixlabs-pdf-dev.appspot.com/assets/pdfjs/web/viewer.html?file=%2Fpdfproxy%3Finstance%3DnAtQJVl6b_8f6vNklZQi1dpTKDA1Z0NHZX_EcLmcdtY.eyJpbnN0YW5jZUlkIjoiZWU1M2FhOTctZWJjMS00NjIwLTk5NDQtYWU3MjVmMjA0ZjM3IiwiYXBwRGVmSWQiOiIxM2VlMTBhMy1lY2I5LTdlZmYtNDI5OC1kMmY5ZjM0YWNmMGQiLCJtZXRhU2l0ZUlkIjoiM2EwMmU4ZTUtYzAzMS00ZTIxLWE3ZjItOTUyYzZmYzk2NTQ0Iiwic2lnbkRhdGUiOiIyMDIwLTAzLTIxVDE5OjI5OjA2Ljc1NVoiLCJkZW1vTW9kZSI6ZmFsc2UsImFpZCI6IjI2Yjc4MDE0LTllYTktNGNlMi04MTllLTkyODM5MzMxN2IxYyIsImJpVG9rZW4iOiJkNDUxNDI3Mi0yYmYwLTA4MDEtM2ViNi0zYjVlMzBlOTJhNzMiLCJzaXRlT3duZXJJZCI6IjVkNTdjNjUwLTA1YTktNDFiNS1iMmFiLTEyNTkxMGE5Zjk4ZCJ9%26compId%3Dcomp-k8208f4w%26url%3Dhttps%3A%2F%2Fdocs.wixstatic.com%2Fugd%2F5d57c6_39e405e3617f4724a1869d4a9713e97b.pdf#page=1&links=true&originalFileName=image_recognition_nn&locale=en&allowDownload=true&allowPrinting=true">CNN.pdf</a>
  </td>
  </tr>
</tbody>
</table>
