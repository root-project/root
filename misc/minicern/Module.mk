# Module.mk for hbook module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 18/2/2002

MODNAME      := minicern
MODDIR       := misc/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MINICERNDIR  := $(MODDIR)
MINICERNDIRS := $(MINICERNDIR)/src
MINICERNDIRI := $(MINICERNDIR)/inc

##### libminicern.a #####
MINICERNS1   := $(wildcard $(MODDIRS)/*.c)
MINICERNS2   := $(wildcard $(MODDIRS)/*.f)
MINICERNO1   := $(MINICERNS1:.c=.o)
MINICERNO2   := $(MINICERNS2:.f=.o)
MINICERNO    := $(MINICERNO1) $(MINICERNO2)

MINICERNDEP  := $(MINICERNS1:.c=.d)

MINICERNLIB  := $(LPATH)/libminicern.$(SOEXT)

# used in the main Makefile
ALLLIBS      += $(MINICERNLIB)

# include all dependency files
INCLUDEFILES += $(MINICERNDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MINICERNDIRI)/%.h
		cp $< $@

$(MINICERNLIB): $(MINICERNO) $(ORDER_) $(MINICERNLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libmincern.$(SOEXT) $@ "$(MINICERNO1)" \
		   "$(MINICERNO2) $(MINICERNLIBEXTRA) $(F77LIBS)"

all-$(MODNAME): $(MINICERNLIB)

clean-$(MODNAME):
		@rm -f $(MINICERNO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MINICERNDEP) $(MINICERNLIB)

distclean::     distclean-$(MODNAME)
