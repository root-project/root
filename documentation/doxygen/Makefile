
.PHONY: filter doxygen clean

DOXYGEN_OUTPUT_DIRECTORY ?= $(HOME)/rootdoc
export DOXYGEN_OUTPUT_DIRECTORY
export DOXYGEN_ROOT_VERSION := $(shell root -l -b -q -e 'printf("%s\n", gROOT->GetVersion());')
export DOXYGEN_EXAMPLE_PATH := $(DOXYGEN_OUTPUT_DIRECTORY)/macros
export DOXYGEN_IMAGE_PATH := $(DOXYGEN_OUTPUT_DIRECTORY)/html
export DOXYGEN_STRIP_PATH := $(shell cd $(PWD)/../..; pwd)
export DOXYGEN_INCLUDE_PATH := $(shell find $(DOXYGEN_STRIP_PATH) -type d -name inc)



all: filter doxygen

filter:
	`root-config --cxx` -o filter filter.cxx `root-config --libs --glibs --cflags`

doxygen:
	if [ ! -d $(DOXYGEN_OUTPUT_DIRECTORY) ]; then mkdir $(DOXYGEN_OUTPUT_DIRECTORY); fi
	if [ ! -d $(DOXYGEN_EXAMPLE_PATH) ]; then mkdir $(DOXYGEN_EXAMPLE_PATH); fi
	if [ ! -d $(DOXYGEN_IMAGE_PATH) ]; then mkdir $(DOXYGEN_IMAGE_PATH); fi
	./makehtmlfooter.sh > htmlfooter.html
	cp images/* $(DOXYGEN_IMAGE_PATH)
	doxygen
	rm c1*
	rm stdout.dat
	rm htmlfooter.html

clean:
	rm -r $(DOXYGEN_OUTPUT_DIRECTORY)
	rm filter

