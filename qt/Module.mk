# Module.mk for qt module
# Copyright (c) 2001 Valeri Fine
#
# Author: Valeri Fine, 21/10/2001

MODDIR        := qt
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

GQTDIR        := $(MODDIR)
GQTDIRS       := $(GQTDIR)/src
GQTDIRI       := $(GQTDIR)/inc

##### libGQt #####
GQTL          := $(MODDIRI)/LinkDef.h
GQTDS         := $(MODDIRS)/G__GQt.cxx
GQTDO         := $(GQTDS:.cxx=.o)
GQTDH         := $(GQTDS:.cxx=.h)

GQTH1          := $(GQTDIRI)/TGQt.h $(GQTDIRI)/TQtThread.h $(GQTDIRI)/TQtApplication.h \
                  $(GQTDIRI)/TQtBrush.h $(GQTDIRI)/TQMimeTypes.h $(GQTDIRI)/TQtClientFilter.h\
                  $(GQTDIRI)/TQtClientWidget.h $(GQTDIRI)/TQtWidget.h $(GQTDIRI)/TQtMarker.h \
                  $(GQTDIRI)/TQtTimer.h

GQTH          := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GQTS          := $(filter-out $(MODDIRS)/moc_%,\
                 $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))
GQTO          := $(GQTS:.cxx=.o)

GQTMOC        := $(subst $(MODDIRI)/,$(MODDIRS)/moc_,$(patsubst %.h,%.cxx,$(GQTH)))
GQTMOCO       := $(GQTMOC:.cxx=.o)

GQTDEP        := $(GQTO:.o=.d) $(GQTDO:.o=.d)

GQTCXXFLAGS   := -DQT_DLL -DQT_THREAD_SUPPORT -I. $(QTINCDIR:%=-I%)

GQTLIB        := $(LPATH)/libGQt.$(SOEXT)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GQTH))
ALLLIBS       += $(GQTLIB)

# include all dependency files
INCLUDEFILES  += $(GQTDEP)

##### local rules #####
include/%.h:    $(GQTDIRI)/%.h
		cp $< $@

$(GQTLIB):      $(GQTO) $(GQTDO) $(GQTMOCO) $(MAINLIBS) $(GQTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGQt.$(SOEXT) $@ \
		   "$(GQTO) $(GQTMOCO) $(GQTDO)" \
		   "$(GQTLIBEXTRA) $(QTLIBDIR) $(QTLIB)"

$(GQTDS):       $(GQTH1) $(GQTL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GQTH1) $(GQTL)

$(GQTDO):       $(GQTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) $(GQTCXXFLAGS) -o $@ -c $<

all-qt:         $(GQTLIB)

map-qt:         $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GQTLIB) \
		   -d $(GQTLIBDEP) -c $(GQTL)

map::           map-qt

clean-qt:
		@rm -f $(GQTO) $(GQTDO) $(GQTMOCO)

clean::         clean-qt

distclean-qt:   clean-qt
		@rm -f $(GQTDEP) $(GQTDS) $(GQTDH) $(GQTMOC) $(GQTLIB)

distclean::     distclean-qt

##### extra rules ######
$(sort $(GQTMOCO) $(GQTO)): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(GQTCXXFLAGS) -o $@ -c $<

$(GQTMOC): $(GQTDIRS)/moc_%.cxx: $(GQTDIRI)/%.h
	$(QTMOCEXE) $< -o $@
