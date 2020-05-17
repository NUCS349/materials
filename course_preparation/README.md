# NUCS 349 - Prequisities

Machine Learning is an exciting field of study, and it is used in many different applications from medicine and industry through research. However, Machine Learning is also a somewhat complicated subject, especially when you study it the first time.

CS-349, as taught at Northwestern, will introduce you to basics of Machine Learning through a series of lectures accompanied by a weekly Python assignment where you have to implement the algorithm taught in the lecture videos.

Most Machine Learning algorithms make heavy use of Linear Algebra; hence fundamental understanding of Linear Algebra is required. In the course description, we have listed several prerequisites such as 
1.  COMP_SCI 214 OR COMP_SCI 325 and OR Graduate Standing in Computer Science or Computer Engineering and equivalent programming experience.
2.  GEN_ENG 205 OR equivalent experience in Linear Algebra
3.  COMP_SCI 110 OR equivalent experience in Python OR good command of MATLAB/Julia/R

We understand that many of you have not taken those courses yet, or it has been a while since those courses were taken. While we believe that understanding those concepts is essential for a successful learning experience, we are also convinced that those prerequisites can be self-taught by the students in a short time before the start of on-the-fly during the course. For this reason, we provide a list of materials that we encourage you to study before or during the beginning of the course, to be well prepared for the quarter. 

# Python Programming
## Introduction to Python
1. Nice Documentation that shows you the essentials: https://www.w3schools.com/python/default.asp
2. Interactive tutorial in browser:  https://www.learnpython.org/
## Introduction to Jupyter
Jupyter Notebooks have emerged to be an essential tool for many data scientists and researchers. We recommend using Jupyter to test out small snippets of codes, etc., but theoretically, you could do most of your homework purely withing Jupyter. 
1. https://realpython.com/jupyter-notebook-introduction/
2. Video Tutorial: https://www.youtube.com/playlist?list=PL1m-6MPBNAZfF-El7BzqaOrCrTBRgH1Nk
# Linear Algebra
Linear Algebra is essential for understanding Machine Learning algorithms, and without understanding Linear Algebra, you will have a tough time in this course. Please read through the following resources to get a refresher on Linear Algebra if you feel you need it:
1. https://machinelearningmastery.com/linear-algebra-machine-learning/
2. https://towardsdatascience.com/linear-algebra-for-deep-learning-f21d7e7d7f23
3. The following is a great book chapter on Linear Algebra from the famous "Deep Learning Book". The first chapters until 2.5 are prerequisites for this course, but the other sections will be discussed in the lecture (e.g. Norms, PCA and SVD) and this book is a great resource if you want to read up on those: https://www.deeplearningbook.org/contents/linear_algebra.html
4. This is a beautiful lecture on Linear Algebra (quite lengthy), but if you enjoy watching videos this is something to watch for you guys: https://www.youtube.com/playlist?list=PL2jykFOD1AWazz20_QRfESiJ2rthDF9-Z
5. This is short video series on Linear Algebra, specially tailored for ML applications: https://www.youtube.com/playlist?list=PLupD_xFct8mEYg5q8itrFDuDaWKDtjSj_


## Introduction to Numpy and Vector Math
Numpy will be there core package that you use in the NUCS349. We will implement most of our algorithms using numpy functionality only. 
1. https://www.analyticsvidhya.com/blog/2017/05/comprehensive-guide-to-linear-algebra/
2. https://www.geeksforgeeks.org/numpy-linear-algebra/

## Python Slides from Machine Learning Refined (by Aggelos Konstantinos Katsaggelos, Jeremy Watt, and Reza Borhani)
Machine Learning Refined is a great book with many online resources found here : https://github.com/jermwatt/machine_learning_refined
I recommend having a look at the following 4 resources to get a good primer in Linear Algebra that cover almost everything that's needed for this course:
1. Vector and Vector Operations: https://jermwatt.github.io/machine_learning_refined/notes/16_Linear_algebra/16_2_Vectors.html
2. Matrix and Matrix Operations: https://jermwatt.github.io/machine_learning_refined/notes/16_Linear_algebra/16_3_Matrices.html
3. Eigenvalues and Eigenvectors: https://jermwatt.github.io/machine_learning_refined/notes/16_Linear_algebra/16_4_Eigen.html
4. Vector and Vectornorms: https://jermwatt.github.io/machine_learning_refined/notes/16_Linear_algebra/16_5_Norms.html

## Vectorizing For-Loops with Numpy
Many of the algorithms implemented in this course will run slowly if implemented using only "for-loops." I understand that coming from a Java/C++ background, avoiding for-loops might be a bit counter-intuitive. Still, it's an essential part of data-science prototyping with Python (and even more critical in Matlab). I recommend you to use Matrix algebra WHENEVER possible because it speeds up your code tremendously! Please find some resources here:
1. https://realpython.com/numpy-array-programming/
2. https://towardsdatascience.com/data-science-with-python-turn-your-conditional-loops-to-numpy-vectors-9484ff9c622e
3. https://hackernoon.com/speeding-up-your-code-2-vectorizing-the-loops-with-numpy-e380e939bed3

# A refresher on Statistics
Most of the concepts of statistics will be introduced in the lecture; however, basic knowledge of statistics will be required. I recommend watching the following videos (about 90 min in total) on probability before the course:
1. Introduction to Probability, Basic Overview - Sample Space, & Tree Diagrams: https://www.youtube.com/watch?v=SkidyDQuupA
2. Probability - Independent and Dependent Events: https://www.youtube.com/watch?v=lWAdPyvm400
3. Conditional Probability With Venn Diagrams & Contingency Tables: https://www.youtube.com/watch?v=sqDVrXq_eh0
4. Bayes' Theorem of Probability With Tree Diagrams & Venn Diagrams: https://www.youtube.com/watch?v=OByl4RJxnKA
5. Compound Probability of Independent Events - Coins & 52 Playing Cards: https://www.youtube.com/watch?v=EHU6pVSczb4
6. Multiplication & Addition Rule - Probability - Mutually Exclusive & Independent Events: https://www.youtube.com/watch?v=94AmzeR9n2w

# Trees and Graphs (Important for Homework 1)
The first lecture and homework will be on Decision Trees. In order to understand decision trees you need to understand what trees or graphs are. We will quickl go over this in the lectures, but it cannot hurt to have a refresher on this before the course starts. You don't have to learn those concepts by heart, but just reading through them really helps:
1. https://medium.com/basecs/a-gentle-introduction-to-graph-theory-77969829ead8
2. https://www.analyticsvidhya.com/blog/2018/09/introduction-graph-theory-applications-python/
3. https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/
4. Everything you need to know about tree data structures: https://www.freecodecamp.org/news/all-you-need-to-know-about-tree-data-structures-bceacb85490c/
