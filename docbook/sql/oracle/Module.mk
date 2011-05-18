# Module.mk for oracle module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Yan Liu, 11/17/2004

MODNAME       := oracle
MODDIR        := $(ROOT_SRCDIR)/sql/$(MODNAME)
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

ORACLEDIR     := $(MODDIR)
ORACLEDIRS    := $(ORACLEDIR)/src
ORACLEDIRI    := $(ORACLEDIR)/inc

##### libOracle #####
ORACLEL       := $(MODDIRI)/LinkDef.h
ORACLEDS      := $(call stripsrc,$(MODDIRS)/G__Oracle.cxx)
ORACLEDO      := $(ORACLEDS:.cxx=.o)
ORACLEDH      := $(ORACLEDS:.cxx=.h)

ORACLEH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
ORACLES       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ORACLEO       := $(call stripsrc,$(ORACLES:.cxx=.o))

ORACLEDEP     := $(ORACLEO:.o=.d) $(ORACLEDO:.o=.d)

ORACLELIB     := $(LPATH)/libOracle.$(SOEXT)
ORACLEMAP     := $(ORACLELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ORACLEH))
ALLLIBS      += $(ORACLELIB)
ALLMAPS      += $(ORACLEMAP)

# include all dependency files
INCLUDEFILES += $(ORACLEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(ORACLEDIRI)/%.h
		cp $< $@

$(ORACLELIB):   $(ORACLEO) $(ORACLEDO) $(ORDER_) $(MAINLIBS) $(ORACLELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libOracle.$(SOEXT) $@ "$(ORACLEO) $(ORACLEDO)" \
		   "$(ORACLELIBEXTRA) $(ORACLELIBDIR) $(ORACLECLILIB)"

$(ORACLEDS):    $(ORACLEH) $(ORACLEL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ORACLEINCDIR:%=-I%) $(ORACLEH) $(ORACLEL)

$(ORACLEMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(ORACLEL)
		$(RLIBMAP) -o $@ -l $(ORACLELIB) \
		   -d $(ORACLELIBDEPM) -c $(ORACLEL)

all-$(MODNAME): $(ORACLELIB) $(ORACLEMAP)

clean-$(MODNAME):
		@rm -f $(ORACLEO) $(ORACLEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ORACLEDEP) $(ORACLEDS) $(ORACLEDH) $(ORACLELIB) $(ORACLEMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(ORACLEO) $(ORACLEDO): CXXFLAGS += -I$(ORACLEINCDIR)
