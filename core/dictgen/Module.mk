# Module.mk for utilities for libMeta and rootcint
# Copyright (c) 1995-2016 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2016-12-14

MODNAME        := dictgen
MODDIR         := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
# MODDIRI        := $(MODDIR)/inc

DICTGENDIR   := $(MODDIR)
DICTGENDIRS  := $(DICTGENDIR)/src
# DICTGENDIRI  := $(DICTGENDIR)/inc
DICTGENDIRR  := $(DICTGENDIR)/res

##### $(DICTGENO) #####
DICTGENH     := $(DICTGENDIRS)/TCling.h
DICTGENS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

DICTGENCXXFLAGS = $(filter-out -fno-exceptions,$(CLINGCXXFLAGS))
ifneq ($(CXX:g++=),$(CXX))
DICTGENCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif

DICTGENO     := $(call stripsrc,$(DICTGENS:.cxx=.o))

# DICTGENL     := $(MODDIRI)/LinkDef.h

DICTGENDEP   := $(DICTGENO:.o=.d)

# used in the main Makefile
# ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(DICTGENH))

# include all dependency files
INCLUDEFILES += $(DICTGENDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

# include/%.h:    $(DICTGENDIRI)/%.h
# 		cp $< $@

all-$(MODNAME): $(DICTGENO)

clean-$(MODNAME):
		@rm -f $(DICTGENO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(DICTGENDEP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(DICTGENO): CXXFLAGS += $(DICTGENCXXFLAGS) -I$(DICTGENDIRR) -I$(CLINGUTILSDIRR) -I$(FOUNDATIONDIRR)
$(DICTGENO): $(LLVMDEP)
