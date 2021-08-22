# CudaMandelbrot
Some CUDA for making interactive ASCII art of the Mandelbrot set.
```
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    0
+                                                           ......oo....                                +    1
+                                                         ....o..##*.....                               +    2
+                                                       ......########o....                             +    3
+                                                .............*#######........                          +    4
+                                              ..o.o.....#..o.oo#####o..o.oo.........                   +    5
+                                            ......###o*o###################....oo.o..                  +    6
+                                          ........###########################o###o...                  +    7
+                                      .......o#oo###############################.....                  +    8
+                       ..o....................o*#################################o.....                +    9
+                       ..........o..........o#######################################o..                +   10
+                     .......o##o###oo*.o....o######################################...                 +   11
+                   ........oo###########*..o#######################################**.                 +   12
+               ...........o###############o#######################################o#..                 +   13
+      ...............o*#oo########################################################...                  +   14
+##############################################################################*......                  +   15
+      ...............o*#oo########################################################...                  +   16
+               ...........o###############o#######################################o#..                 +   17
+                   ........oo###########*..o#######################################**.                 +   18
+                     .......o##o###oo*.o....o######################################...                 +   19
+                       ..........o..........o#######################################o..                +   20
+                       ..o....................o*#################################o.....                +   21
+                                      .......o#oo###############################.....                  +   22
+                                          ........###########################o###o...                  +   23
+                                            ......###o*o###################....oo.o..                  +   24
+                                              ..o.o.....#..o.oo#####o..o.oo.........                   +   25
+                                                .............*#######........                          +   26
+                                                       ......########o....                             +   27
+                                                         ....o..##*.....                               +   28
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   29
```
Once your CUDA drivers are setup, it should be a matter of running `nvcc mandelbrot.cu -o main` followed by `./main`.
The `w`, `a`, `s`, `d` keys can be used to navigate and the `=` and `-` keys can be used to zoom in and out.

