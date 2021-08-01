#!/bin/bash
#
# pav1iet - PASCAL Annotation Version 1.00 Extractor Tool
# Copyright (C) 2020-2021 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# This file is part of pav1iet.
#
# pav1iet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pav1iet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pav1iet.  If not, see <http://www.gnu.org/licenses/>.
#

# These script downloads the INRIA person dataset that was introduced by Dalal &
# Triggs to train a linear support vector machine using Histogram of Oriented
# Gradients feature descriptors extracted from these images for the purpose of
# detecting pedestrians.
#
# The original images used for trainings seems to have been lost becaused the
# postprocessed (i.e., cropped) images are corrupt (at least from sources I am
# aware of). However, the full resolution images with their annotations are
# still available, even if the dataset is now two annotations short (1237 vs.
# 1239 bounding boxes).

name=INRIAPerson.tar
URL=${URL:-"ftp://ftp.inrialpes.fr/pub/lear/douze/data/$name"}

{ [[ -s $name ]] && echo $name already downloaded; } || wget -nv $URL

dir=${name%.*}

{ [[ -d $dir ]] && echo $name already extracted; } || tar xvf $name && \
    chmod -R u+rwX $dir

# Remove directories that are not called Train or Test. These directories
# contain broken .png files and therefore can be discarded.
readarray -t broken_dirs < <(ls -d $dir/*/ | grep -vP '(Test|Train)/$')

[[ ${#broken_dirs[@]} -eq 0 ]] || \
    { rm -r ${broken_dirs[@]} && echo removed directories ${broken_dirs[@]}} which contain broken files; }

# The listings are not placed in the correct directory. Move the files one level
# up and give the files unique names.
[[ -s $dir/Train-pos.lst ]] || mv $dir/Train/annotations.lst $dir/Train-pos.lst
[[ -s $dir/Test-pos.lst ]] || mv $dir/Test/annotations.lst $dir/Test-pos.lst
[[ -s $dir/Train-neg.lst ]] || mv $dir/Train/neg.lst $dir/Train-neg.lst
[[ -s $dir/Test-neg.lst ]] || mv $dir/Test/neg.lst $dir/Test-neg.lst
