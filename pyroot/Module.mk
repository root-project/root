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

PYROOTLIB    := $(LPATH)/libPyROOT.$(SOEXT)

ROOTPYS      := $(MODDIR)/ROOT.py
ifeq ($(PLATFORM),win32)
ROOTPY       := bin/ROOT.py
else
ROOTPY       := $(LPATH)/ROOT.py
endif

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PYROOTH))
ALLLIBS     += $(PYROOTLIB)

# include all dependency files
INCLUDEFILES += $(PYROOTDEP)

##### local rules #####
include/%.h:    $(PYROOTDIRI)/%.h
		cp $< $@

$(ROOTPY):      $(ROOTPYS)
		cp $< $@

$(PYROOTLIB):   $(PYROOTO) $(PYROOTDO) $(MAINLIBS) $(ROOTPY)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		"$(SOFLAGS)" libPyROOT.$(SOEXT) $@ \
		"$(PYROOTO) $(PYROOTDO)" "$(PYTHONLIBDIR) $(PYTHONLIB)" \
                "$(PYTHONLIBFLAGS)"

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
