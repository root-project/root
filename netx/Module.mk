# Module.mk for netx module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G. Ganis, 8/7/2004

MODDIR       := netx
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

NETXDIR      := $(MODDIR)
NETXDIRS     := $(NETXDIR)/src
NETXDIRI     := $(NETXDIR)/inc

##### libNetx #####
NETXL        := $(MODDIRI)/LinkDef.h
NETXDS       := $(MODDIRS)/G__Netx.cxx
NETXDO       := $(NETXDS:.cxx=.o)
NETXDH       := $(NETXDS:.cxx=.h)

NETXH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
NETXH1       := $(filter-out $(MODDIRI)/TXProtocol.h,$(NETXH))
NETXS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
NETXO        := $(NETXS:.cxx=.o)

NETXDEP      := $(NETXO:.o=.d) $(NETXDO:.o=.d)

NETXLIB      := $(LPATH)/libNetx.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETXH))
ALLLIBS      += $(NETXLIB)

# include all dependency files
INCLUDEFILES += $(NETXDEP)

# Xrootd includes
NETXINCEXTRA := $(XROOTDDIRI:%=-I%)

##### local rules #####
include/%.h:    $(NETXDIRI)/%.h
		cp $< $@

$(NETXLIB):     $(NETXO) $(NETXDO) $(MAINLIBS) $(NETXLIBDEP) $(XROOTDETAG)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libNetx.$(SOEXT) $@ "$(NETXO) $(NETXDO)" \
		"$(NETXLIBEXTRA)"

$(NETXDS):      $(NETXH1) $(NETXL) $(ROOTCINTTMP) $(XROOTDETAG) 
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(NETXINCEXTRA) $(NETXH1) $(NETXL)

$(NETXDO):      $(NETXDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. $(NETXINCEXTRA) -o $@ -c $<

all-netx:       $(NETXLIB)

map-netx:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(NETXLIB) \
		   -d $(NETXLIBDEP) -c $(NETXL)

map::           map-netx

clean-netx:
		@rm -f $(NETXO) $(NETXDO)

clean::         clean-netx

distclean-netx: clean-netx
		@rm -f $(NETXDEP) $(NETXDS) $(NETXDH) $(NETXLIB)

distclean::     distclean-netx

##### extra rules ######
$(NETXO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(NETXINCEXTRA) -o $@ -c $<
