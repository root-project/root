# Module.mk for hbook module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 18/2/2002

MODNAME      := minicern
MODDIR       := $(ROOT_SRCDIR)/misc/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MINICERNDIR  := $(MODDIR)
MINICERNDIRS := $(MINICERNDIR)/src
MINICERNDIRI := $(MINICERNDIR)/inc

##### libminicern.a #####
MINICERNS1   := $(wildcard $(MODDIRS)/*.c)
MINICERNS2   := $(wildcard $(MODDIRS)/*.f)
MINICERNO1   := $(call stripsrc,$(MINICERNS1:.c=.o))
MINICERNO2   := $(call stripsrc,$(MINICERNS2:.f=.o))
MINICERNO    := $(MINICERNO1) $(MINICERNO2)

MINICERNDEP  := $(MINICERNS1:.c=.d)

ifneq ($(PLATFORM),win32)
MINICERNLIB  := $(LPATH)/libminicern.a
else
MINICERNLIB  := $(LPATH)/libminicern.lib
LNKFLAGS      = -nologo -ignore:4049,4075,4217,4221
ifeq (yes,$(WINRTDEBUG))
  LNKFLAGS   += -nodefaultlib:msvcrt.lib
endif
endif

# used in the main Makefile
ALLLIBS      += $(MINICERNLIB)

# include all dependency files
INCLUDEFILES += $(MINICERNDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MINICERNDIRI)/%.h
		cp $< $@

ifneq ($(PLATFORM),win32)
$(MINICERNLIB): $(MINICERNO) $(ORDER_) $(MINICERNLIBDEP)
		$(AR) cru $@ $(MINICERNO)
		@(if [ $(PLATFORM) = "macosx" ]; then \
		  ranlib $@; \
		fi)
else
$(MINICERNLIB): $(MINICERNO) $(ORDER_) $(MINICERNLIBDEP)
		$(LD) -LIB $(LNKFLAGS) -o $@ "$(MINICERNO1) $(MINICERNO2)"
endif

all-$(MODNAME): $(MINICERNLIB)

clean-$(MODNAME):
		@rm -f $(MINICERNO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MINICERNDEP) $(MINICERNLIB)

distclean::     distclean-$(MODNAME)
