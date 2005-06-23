# Module.mk for reflex module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := reflex
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

REFLEXDIR    := $(MODDIR)
REFLEXDIRS   := $(REFLEXDIR)/src
REFLEXDIRI   := $(REFLEXDIR)/inc

##### libReflex #####
REFLEXAH     := $(wildcard $(MODDIRI)/Reflex/*.h)
REFLEXBH     := $(wildcard $(MODDIRI)/Reflex/Builder/*.h)
REFLEXH      := $(REFLEXAH) $(REFLEXBH)
REFLEXS      := $(wildcard $(MODDIRS)/*.cxx)
REFLEXO      := $(REFLEXS:.cxx=.o)

REFLEXDEP    := $(REFLEXO:.o=.d)

REFLEXLIBNAME := libReflex.$(SOEXT)
REFLEXLIB    := $(LPATH)/$(REFLEXLIBNAME)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Reflex/%.h,include/Reflex/%.h,$(REFLEXH))
ALLLIBS      += $(REFLEXLIB)

# include all dependency files
INCLUDEFILES += $(REFLEXDEP)

##### local rules #####
include/Reflex/%.h: $(REFLEXDIRI)/Reflex/%.h 
		@ ( if [ ! -d "include/Reflex" ] ;         \
		    then mkdir include/Reflex; fi ) ;      \
		  ( if [ ! -d "include/Reflex/Builder" ] ; \
		    then mkdir include/Reflex/Builder; fi ) 
		cp $< $@

$(REFLEXLIB):   $(REFLEXO) $(MAINLIBS)
		$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"     \
		"$(SOFLAGS)" $(REFLEXLIBNAME) $@ "$(REFLEXO)" \
		"$(REFLEXLIBDIR) $(REFLEXCLILIB)"

all-reflex:     $(REFLEXLIB)
		echo $(REFLEXO)

map-reflex:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(REFLEXLIB) \
		-d $(REFLEXLIBDEP) -c $(REFLEXL)

map::           map-reflex

clean-reflex:
		@rm -f $(REFLEXO)

clean::         clean-reflex

distclean-reflex: clean-reflex
		@rm -f $(REFLEXDEP) $(REFLEXLIB)

distclean::     distclean-reflex

##### extra rules ######
$(REFLEXO): %.o: %.cxx
		$(CXX) $(OPT) $(CXXFLAGS) $(REFLEXINCDIR:%=-I%) -o $@ -c $<

