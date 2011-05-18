# Module.mk for rootd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := rootd
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOTDDIR     := $(MODDIR)
ROOTDDIRS    := $(ROOTDDIR)/src
ROOTDDIRI    := $(ROOTDDIR)/inc

##### rootd #####
ROOTDH       := $(wildcard $(MODDIRI)/*.h)
ROOTDS       := $(wildcard $(MODDIRS)/*.cxx)
ROOTDO       := $(call stripsrc,$(ROOTDS:.cxx=.o))
ROOTDDEP     := $(ROOTDO:.o=.d)
ROOTD        := bin/rootd

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOTDH))
ALLEXECS     += $(ROOTD)

# include all dependency files
INCLUDEFILES += $(ROOTDDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(ROOTDDIRI)/%.h
		cp $< $@

$(ROOTD):       $(ROOTDO) $(RSAO) $(SNPRINTFO) $(GLBPATCHO) $(RPDUTILO) $(STRLCPYO)
		$(LD) $(LDFLAGS) -o $@ $(ROOTDO) $(RPDUTILO) $(GLBPATCHO) \
		   $(RSAO) $(SNPRINTFO) $(CRYPTLIBS) $(AUTHLIBS) $(STRLCPYO) $(SYSLIBS)

all-$(MODNAME): $(ROOTD)

clean-$(MODNAME):
		@rm -f $(ROOTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTDDEP) $(ROOTD)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(ROOTDO):  CXXFLAGS += $(AUTHFLAGS)
