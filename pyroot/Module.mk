# Module.mk for pyroot module
#
# Authors: Pere Mato, Wim Lavrijsen, 22/4/2004

MODDIR       := pyroot
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PYROOTDIR    := $(MODDIR)
PYROOTDIRS   := $(PYROOTDIR)/src
PYROOTDIRI   := $(PYROOTDIR)/inc

PYROOTL      := $(MODDIRI)/LinkDef.h
PYROOTDS     := $(MODDIRS)/G__PyROOT.cxx
PYROOTDO     := $(PYROOTDS:.cxx=.o)
PYROOTDH     := $(PYROOTDS:.cxx=.h)

PYROOTH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PYROOTS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PYROOTO      := $(PYROOTS:.cxx=.o)

PYROOTDEP    := $(PYROOTO:.o=.d) $(PYROOTDO:.o=.d)

PYROOTLIB    := $(LPATH)/PyROOT.$(SOEXT)

ROOTPY       := $(MODDIR)/ROOT.py

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PYROOTH))
ALLLIBS     += $(PYROOTLIB)

# include all dependency files
INCLUDEFILES += $(PYROOTDEP)

##### local rules #####
include/%.h:    $(PYROOTDIRI)/%.h
		cp $< $@

$(PYROOTLIB):   $(PYROOTO) $(PYROOTDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		"$(SOFLAGS)" PyROOT.$(SOEXT) $@ \
		"$(PYROOTO) $(PYROOTDO)" "$(PYTHONLIBDIR) $(PYTHONLIB)" \
                "$(PYTHONLIBFLAGS)"
ifeq ($(PLATFORM),win32)
		@cp $(ROOTPY) bin
else
		@cp $(ROOTPY) lib
endif

$(PYROOTDS):    $(PYROOTH) $(PYROOTL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PYROOTH) $(PYROOTL)

$(PYROOTDO):    $(PYROOTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-pyroot:     $(PYROOTLIB)

clean-pyroot:
		@rm -f $(PYROOTO) $(PYROOTDO)

clean::         clean-pyroot

distclean-pyroot: clean-pyroot
		@rm -f $(PYROOTDEP) $(PYROOTDS) $(PYROOTDH) $(PYROOTLIB)

distclean::     distclean-pyroot

##### extra rules ######
$(PYROOTO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(PYTHONINCDIR) -o $@ -c $<
