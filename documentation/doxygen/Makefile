
.PHONY: filter folders mathjax images doxygen

DOXYGEN_OUTPUT_DIRECTORY ?= $(HOME)/rootdoc
export DOXYGEN_OUTPUT_DIRECTORY
export DOXYGEN_SOURCE_DIRECTORY := ../..
export DOXYGEN_ROOT_VERSION := $(shell root -l -b -q -e 'printf("%s\n", gROOT->GetVersion());')
export DOXYGEN_EXAMPLE_PATH := $(DOXYGEN_OUTPUT_DIRECTORY)/macros
export DOXYGEN_IMAGE_PATH := $(DOXYGEN_OUTPUT_DIRECTORY)/html
export DOXYGEN_STRIP_PATH := $(shell cd $(PWD)/../..; pwd)
export DOXYGEN_INCLUDE_PATH := $(shell find $(DOXYGEN_STRIP_PATH) -type d -name dictpch -prune -o -type d -name inc)

all: filter folders mathjax images doxygen

filter:
	`root-config --cxx` -o filter filter.cxx -std=c++11

folders:
	if [ ! -d $(DOXYGEN_OUTPUT_DIRECTORY) ]; then mkdir $(DOXYGEN_OUTPUT_DIRECTORY); fi
	if [ ! -d $(DOXYGEN_EXAMPLE_PATH) ]; then mkdir $(DOXYGEN_EXAMPLE_PATH); fi
	if [ ! -d $(DOXYGEN_MATHJAX_PATH) ]; then mkdir $(DOXYGEN_MATHJAX_PATH); fi
	if [ ! -d $(DOXYGEN_IMAGE_PATH) ]; then mkdir $(DOXYGEN_IMAGE_PATH); fi

mathjax:
	tar xfz mathjax.tar.gz -C $(DOXYGEN_OUTPUT_DIRECTORY)/html

images:
	cp images/* $(DOXYGEN_IMAGE_PATH)

doxygen: filter
	./makehtmlfooter.sh > htmlfooter.html
	doxygen
	rm -rf files c1* *.ps *.png *.svg *.pdf *.root *.xpm *.out *.dat

clean:
	rm -r $(DOXYGEN_OUTPUT_DIRECTORY) filter htmlfooter.html

