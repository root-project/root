# $Id: Module.mk,v 1.21 2007/01/29 07:38:36 brun Exp $
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

GQTH1         := $(GQTDIRI)/TGQt.h  $(GQTDIRI)/TQtTimer.h              \
                 $(GQTDIRI)/TQtApplication.h $(GQTDIRI)/TQtBrush.h     \
                 $(GQTDIRI)/TQMimeTypes.h $(GQTDIRI)/TQtClientFilter.h \
                 $(GQTDIRI)/TQtClientWidget.h $(GQTDIRI)/TQtWidget.h   \
                 $(GQTDIRI)/TQtMarker.h $(GQTDIRI)/TQtTimer.h \
                 $(GQTDIRI)/TQtRootSlot.h

GQTH          := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GQTS          := $(filter-out $(MODDIRS)/moc_%,\
                 $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))
GQTO          := $(GQTS:.cxx=.o)

GQTMOCH       := $(MODDIRI)/TQtWidget.h       $(MODDIRI)/TQtEmitter.h \
                 $(MODDIRI)/TQtClientFilter.h $(MODDIRI)/TQtClientGuard.h \
                 $(MODDIRI)/TQtClientWidget.h  $(MODDIRI)/TQtTimer.h \
                 $(MODDIRI)/TQtRootSlot.h

GQTMOC        := $(subst $(MODDIRI)/,$(MODDIRS)/moc_,$(patsubst %.h,%.cxx,$(GQTMOCH)))
GQTMOCO       := $(GQTMOC:.cxx=.o)

GQTDEP        := $(GQTO:.o=.d) $(GQTDO:.o=.d)

GQTCXXFLAGS   := -DQT_DLL  -DQT_NO_DEBUG -DQT_SHARED -DQT_THREAD_SUPPORT -I$(QTDIR)/mkspecs/default -I. $(QTINCDIR:%=-I%)

GQTLIB        := $(LPATH)/libGQt.$(SOEXT)
GQTMAP        := $(GQTLIB:.$(SOEXT)=.rootmap)

# Qt project header files

QCUSTOMWIDGETS += $(GQTDIRI)/TQtWidget.cw
QMAKERULES     += $(GQTDIRI)/rootcint.pri $(GQTDIRI)/rootcintrule.pri \
                  $(GQTDIRI)/rootlibs.pri

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GQTH))
ALLHDRS       += $(patsubst $(MODDIRI)/%.cw,include/%.cw,$(QCUSTOMWIDGETS))
ALLHDRS       += $(patsubst $(MODDIRI)/%.pri,include/%.pri,$(QMAKERULES))
ALLLIBS       += $(GQTLIB)
ALLMAPS       += $(GQTMAP)

# include all dependency files
INCLUDEFILES  += $(GQTDEP)

##### local rules #####
include/%.h:    $(GQTDIRI)/%.h
		cp $< $@

include/%.cw:   $(GQTDIRI)/%.cw
		cp $< $@

include/%.pri:  $(GQTDIRI)/%.pri
		cp $< $@

$(GQTLIB):      $(GQTO) $(GQTDO) $(GQTMOCO) $(ORDER_) $(MAINLIBS) $(GQTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGQt.$(SOEXT) $@ \
		   "$(GQTO) $(GQTMOCO) $(GQTDO)" \
		   "$(GQTLIBEXTRA) $(QTLIBDIR) $(QTLIB)"

$(GQTDS):       $(GQTH1) $(GQTL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GQTH1) $(GQTL)

$(GQTMAP):      $(RLIBMAP) $(MAKEFILEDEP) $(GQTL)
		$(RLIBMAP) -o $(GQTMAP) -l $(GQTLIB) \
		   -d $(GQTLIBDEPM) -c $(GQTL)

all-qt:         $(GQTLIB) $(GQTMAP)

clean-qt:
		@rm  -f $(GQTO) $(GQTDO) $(GQTMOCO) $(GQTMOC)

clean::         clean-qt

distclean-qt: clean-qt
		@rm -f $(GQTDEP) $(GQTDS) $(GQTDH) $(GQTMOC) $(GQTLIB) $(GQTMAP)

distclean::     distclean-qt

##### extra rules ######
$(sort $(GQTMOCO) $(GQTO)): CXXFLAGS += $(GQTCXXFLAGS)
$(GQTDO): CXXFLAGS += $(GQTCXXFLAGS)

$(GQTMOC) : $(GQTDIRS)/moc_%.cxx: $(GQTDIRI)/%.h
	$(QTMOCEXE) $< -o $@

##### cintdlls ######

# qtcint is not longer part of the cintdlls
# cintdlls: $(CINTDIRDLLS)/qtcint.dll

qtcint: lib/qtcint.dll

lib/qtcint.dll: $(CINTTMP) $(ROOTCINTTMPEXE) cint/lib/qt/qtcint.h \
                cint/lib/qt/qtclasses.h cint/lib/qt/qtglobals.h \
                cint/lib/qt/qtfunctions.h
	$(MAKECINTDLL) $(PLATFORM) C++ qtcint qt \
	  " -p $(GQTCXXFLAGS) qtcint.h " \
           "$(CINTTMP)" "$(ROOTCINTTMP)" \
	   "$(MAKELIB)" "$(CXX)" "$(CC)" "$(LD)" "$(OPT)" \
           "$(CINTCXXFLAGS) $(GQTCXXFLAGS)" "$(CINTCFLAGS)" \
           "$(LDFLAGS)  $(QTLIBDIR) $(QTLIB)" "$(SOFLAGS)" \
           "$(SOEXT)" "$(COMPILER)" "$(CXXOUT)"
