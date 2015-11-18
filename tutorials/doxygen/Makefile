
.PHONY: filter doxygen clean

TUTORIALS_OUTPUT_DIRECTORY ?= $(HOME)/root_tutorials/
export TUTORIALS_OUTPUT_DIRECTORY
export TUTORIALS_IMAGE_PATH := $(TUTORIALS_OUTPUT_DIRECTORY)/html

all: filter doxygen

filter:
	`root-config --cxx` -o filter filter.cxx -std=c++11

doxygen: filter
	if [ ! -d $(TUTORIALS_OUTPUT_DIRECTORY) ]; then mkdir $(TUTORIALS_OUTPUT_DIRECTORY); fi
	if [ ! -d $(TUTORIALS_IMAGE_PATH) ]; then mkdir $(TUTORIALS_IMAGE_PATH); fi
	cp images/* $(TUTORIALS_IMAGE_PATH)
	doxygen
	rm -rf files c1* *.ps *.png *.svg *.pdf *.gif *.root

clean:
	rm -r $(TUTORIALS_OUTPUT_DIRECTORY)
	rm filter

