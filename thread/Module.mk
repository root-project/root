# Module.mk for thread module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := thread
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

THREADDIR    := $(MODDIR)
THREADDIRS   := $(THREADDIR)/src
THREADDIRI   := $(THREADDIR)/inc

##### libThread #####
THREADL      := $(MODDIRI)/LinkDef.h
THREADDS     := $(MODDIRS)/G__Thread.cxx
THREADDO     := $(THREADDS:.cxx=.o)
THREADDH     := $(THREADDS:.cxx=.h)

THREADH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
THREADS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
THREADO      := $(THREADS:.cxx=.o)

THREADDEP    := $(THREADO:.o=.d) $(THREADDO:.o=.d)

THREADLIB    := $(LPATH)/libThread.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(THREADH))
ALLLIBS     += $(THREADLIB)
CXXFLAGS    += -D_REENTRANT
CFLAGS      += -D_REENTRANT

# include all dependency files
INCLUDEFILES += $(THREADDEP)

##### local rules #####
include/%.h:    $(THREADDIRI)/%.h
		cp $< $@

$(THREADLIB):   $(THREADO) $(THREADDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libThread.$(SOEXT) $@ "$(THREADO) $(THREADDO)" \
		   "$(THREADLIBEXTRA) $(THREAD)"

$(THREADDS):    $(THREADH) $(THREADL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		@$(ROOTCINTTMP) -f $@ -c $(THREADH) $(THREADL)

$(THREADDO):    $(THREADDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-thread:     $(THREADLIB)

clean-thread:
		@rm -f $(THREADO) $(THREADDO)

clean::         clean-thread

distclean-thread: clean-thread
		@rm -f $(THREADDEP) $(THREADDS) $(THREADDH) $(THREADLIB)

distclean::     distclean-thread
