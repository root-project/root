#!/bin/bash

# author Eckhard von Toerne (Bonn U.)
# this script implements changes to tex files for migration to pdflatex
# please not that the latex default image extension is eps
# and the pdflatex default extension is pdf
# after migration use pdflatex TMVAUsersGuide.tex to compile

for i in `ls *.tex`; do
    mv $i tobechanged 
    cat tobechanged | sed "s/\.eps//g" > $i
    echo updated $i
    diff tobechanged $i
    rm tobechanged
done
