# Module.mk for qtroot module
# Copyright (c) 2003 Valeri Fine
#
# Author: Valeri Fine, 20/5/2003

MODDIR       := qtroot
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

QTROOTDIR       := $(MODDIR)
QTROOTDIRS      := $(QTROOTDIR)/src
QTROOTDIRI      := $(QTROOTDIR)/inc

##### libQtRoot #####
QTROOTL        := $(MODDIRI)/LinkDef.h
QTROOTDS       := $(MODDIRS)/G__QtRoot.cxx
QTROOTDO       := $(QTROOTDS:.cxx=.o)
QTROOTDH       := $(QTROOTDS:.cxx=.h)

QTROOTH1       := TQtRootGuiFactory.h 

QTROOTH        := $(patsubst %,$(QTROOTDIRI)/%,$(QTROOTH1))

QTROOTS        := $(filter-out $(QTROOTDIRS)/G__%,$(wildcard $(QTROOTDIRS)/*.cxx))
QTROOTO        := $(QTROOTS:.cxx=.o)

QTROOTDEP      := $(QTROOTO:.o=.d) $(QTROOTDO:.o=.d)

QTROOTLIB      := $(LPATH)/libQtRoot.$(SOEXT)

QTROOTCXXFLAGS := -DQT_DLL -DQT_THREAD_SUPPORT -I. -I$(QTDIR)/include

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(QTROOTH))
ALLLIBS     += $(QTROOTLIB)

# include all dependency files
INCLUDEFILES += $(QTROOTDEP)

##### local rules #####
include/%.h:    $(QTROOTDIRI)/%.h
		cp $< $@

$(QTROOTLIB):      $(QTROOTO) $(QTROOTDO) $(MAINLIBS) $(QTROOTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGui.$(SOEXT) $@ "$(QTROOTO) $(QTROOTDO)" \
		   "$(QTROOTLIBEXTRA)"

$(QTROOTDS):      $(QTROOTH) $(QTROOTL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(QTROOTH1) $(QTROOTL)

$(QTROOTDO):      $(QTROOTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) $(QTROOTCXXFLAGS) -I. -o $@ -c $<

all-qtroot:        $(QTROOTLIB)

clean-qtroot:
		@rm -f $(QTROOTO) $(QTROOTDO)

clean::         clean-qtroot

distclean-qtroot:  clean-qtroot
		@rm -f $(QTROOTDEP) $(QTROOTDS) $(QTROOTDH) $(QTROOTLIB)

distclean::     distclean-qtroot


##### extra rules ######
$(sort $(QTROOTO)): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(QTROOTCXXFLAGS) -o $@ -c $<

