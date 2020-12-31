# PASCAL Annotation Version 1.00 Image Extraction Tool

This repository contains a tool for parsing files in the PASCAL Annotation
Version 1.00 format and the extraction of annotated image regions into separate
image files.


## Motivation

This tool was created for evaluating an R-HOG implementation based on integral
histograms by recreating the experiments performed by the authors of the
Histogram of Oriented Gradients (HOG) feature descriptor on the [INRIA person
dataset](#inria-person-dataset) (see below).


## Requirements

* C++14 compiler
* CMake 3.5
* Boost 1.70
* OpenCV 4.0
* TBB 4.2

## Usage

First, compile using

```bash
$ mkdir build
$ cmake . -B build
$ cmake --build build
```

Then run:

```bash
$ pav1iet Train.lst -o 'train-%04i.png'
```

In order to use the tool to extract annotations from the INRIA person dataset,
you need to run `prepare_INRIA_person_dataset.sh` script from the `examples`
directory. The script downloads the dataset, removes broken files and moves the
listing files to correct location.


### INRIA Person Dataset

The INRIA person dataset contains a collection of upright standing pedestrian
images. The dataset was used to train a human detector using Histogram of
Oriented Gradients (HOG) features and a linear support vector machine by Dalal &
Triggs for their 2005 CVPR paper.

As of May 2020, the [INRIA person dataset](http://lear.inrialpes.fr/data)
cannot be accessed on its official website anymore. However,
[mirrors](ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar) do exist.

Independently from which source the dataset is obtained, the original images
used for training by Dalal & Triggs seems to have been lost because the
postprocessed (i.e., cropped) images in the dataset archives---at least the ones
I came across---are corrupt.

Inspecting the image archive [uploaded in
2005](https://web.archive.org/web/20050701030429/http://pascal.inrialpes.fr/data/human/)
by the authors reveals the same problem.

Despite this situation, the full resolution images and the corresponding
annotations are still available even if the dataset is now two annotations short
(1237 vs. 1239 bounding boxes mentioned in the CVPR paper).
