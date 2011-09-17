# Module.mk for pq2 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G. Ganis, 2010

MODNAME      := pq2
MODDIR       := $(ROOT_SRCDIR)/proof/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PQ2DIR       := $(MODDIR)
PQ2DIRS      := $(PQ2DIR)/src
PQ2DIRI      := $(PQ2DIR)/inc

##### pq2 #####
PQ2H         := $(wildcard $(MODDIRI)/*.h)
PQ2S         := $(wildcard $(MODDIRS)/*.cxx)
PQ2O         := $(call stripsrc,$(PQ2S:.cxx=.o))
PQ2DEP       := $(PQ2O:.o=.d)
PQ2          := bin/pq2

##### Libraries needed #######
PQ2LIBS      := -lProof -lMatrix -lHist -lTree \
                -lRIO -lNet -lThread $(BOOTLIBS) 
PQ2LIBSDEP    = $(ORDER_) $(CORELIB) $(CINTLIB) $(IOLIB) $(NETLIB) $(HISTLIB) \
                $(TREELIB) $(MATRIXLIB) $(MATHCORELIB) $(PROOFLIB) $(THREADLIB)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PQ2H))
ALLEXECS     += $(PQ2)

# include all dependency files
INCLUDEFILES += $(PQ2DEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PQ2DIRI)/%.h
		cp $< $@

$(PQ2):       $(PQ2O) $(PQ2LIBSDEP)
		$(LD) $(LDFLAGS) -o $@ $(PQ2O)  $(RPATH) $(PQ2LIBS) $(SYSLIBS)

all-$(MODNAME): $(PQ2)

clean-$(MODNAME):
		@rm -f $(PQ2O)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PQ2DEP) $(PQ2)

distclean::     distclean-$(MODNAME)
