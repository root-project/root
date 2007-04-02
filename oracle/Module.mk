# Module.mk for oracle module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Yan Liu, 11/17/2004

MODDIR        := oracle
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

ORACLEDIR     := $(MODDIR)
ORACLEDIRS    := $(ORACLEDIR)/src
ORACLEDIRI    := $(ORACLEDIR)/inc

##### libOracle #####
ORACLEL       := $(MODDIRI)/LinkDef.h
ORACLEDS      := $(MODDIRS)/G__Oracle.cxx
ORACLEDO      := $(ORACLEDS:.cxx=.o)
ORACLEDH      := $(ORACLEDS:.cxx=.h)

ORACLEH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
ORACLES       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ORACLEO       := $(ORACLES:.cxx=.o)

ORACLEDEP     := $(ORACLEO:.o=.d) $(ORACLEDO:.o=.d)

ORACLELIB     := $(LPATH)/libOracle.$(SOEXT)
ORCALEMAP     := $(ORACLELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ORACLEH))
ALLLIBS      += $(ORACLELIB)
ALLMAPS      += $(ORACLEMAP)

# include all dependency files
INCLUDEFILES += $(ORACLEDEP)

##### local rules #####
include/%.h:    $(ORACLEDIRI)/%.h
		cp $< $@

$(ORACLELIB):   $(ORACLEO) $(ORACLEDO) $(ORDER_) $(MAINLIBS) $(ORACLELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libOracle.$(SOEXT) $@ "$(ORACLEO) $(ORACLEDO)" \
		   "$(ORACLELIBEXTRA) $(ORACLELIBDIR) $(ORACLECLILIB)"

$(ORACLEDS):    $(ORACLEH) $(ORACLEL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ORACLEH) $(ORACLEL)

$(ORACLEMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(ORACLEL)
		$(RLIBMAP) -o $(ORACLEMAP) -l $(ORACLELIB) \
		   -d $(ORACLELIBDEPM) -c $(ORACLEL)

all-oracle:     $(ORACLELIB) $(ORACLEMAP)

clean-oracle:
		@rm -f $(ORACLEO) $(ORACLEDO)

clean::         clean-oracle

distclean-oracle: clean-oracle
		@rm -f $(ORACLEDEP) $(ORACLEDS) $(ORACLEDH) $(ORACLELIB) $(ORACLEMAP)

distclean::     distclean-oracle

##### extra rules ######
$(ORACLEO) $(ORACLEDO): CXXFLAGS += -I$(ORACLEINCDIR)
