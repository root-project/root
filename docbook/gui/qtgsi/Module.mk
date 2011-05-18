# Module.mk for qtgsi module
# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Author: Bertrand Bellenot, 22/02/2006

MODNAME      := qtgsi
MODDIR       := $(ROOT_SRCDIR)/gui/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

QTGSIDIR     := $(MODDIR)
QTGSIDIRS    := $(QTGSIDIR)/src
QTGSIDIRI    := $(QTGSIDIR)/inc
QTGSIDIRT    := $(call stripsrc,$(QTGSIDIR)/test)

##### libQtGSI #####
QTGSIL        := $(MODDIRI)/LinkDef.h
QTGSIDS       := $(call stripsrc,$(MODDIRS)/G__QtGSI.cxx)
QTGSIDO       := $(QTGSIDS:.cxx=.o)
QTGSIDH       := $(QTGSIDS:.cxx=.h)

QTGSIH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
QTGSIS        := $(filter-out $(MODDIRS)/moc_%,\
                 $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))
QTGSIO        := $(call stripsrc,$(QTGSIS:.cxx=.o))

QTGSIMOCH     := $(MODDIRI)/TQCanvasMenu.h $(MODDIRI)/TQRootApplication.h \
                 $(MODDIRI)/TQRootCanvas.h $(MODDIRI)/TQRootDialog.h

QTGSIMOC      := $(call stripsrc,$(subst $(MODDIRI)/,$(MODDIRS)/moc_,$(patsubst %.h,%.cxx,$(QTGSIMOCH))))
QTGSIMOCO     := $(QTGSIMOC:.cxx=.o)

QTGSIDEP      := $(QTGSIO:.o=.d) $(QTGSIDO:.o=.d) $(QTGSIMOCO:.o=.d)

QTGSICXXFLAGS := -DQT3_SUPPORT -DQT_DLL -DQT_THREAD_SUPPORT -I. $(QTINCDIR:%=-I%)

QTGSILIB      := $(LPATH)/libQtGSI.$(SOEXT)
QTGSIMAP      := $(QTGSILIB:.$(SOEXT)=.rootmap)

ifeq ($(PLATFORM),win32)
QTTESTOPTS    := -f Makefile.win
else
QTTESTOPTS    :=
endif
QTTESTPATH    := $(PATH):$(abspath ./bin)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(QTGSIH))
ALLLIBS       += $(QTGSILIB)
ALLMAPS       += $(QTGSIMAP)

# include all dependency files
INCLUDEFILES  += $(QTGSIDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME)

include/%.h:    $(QTGSIDIRI)/%.h
		cp $< $@

$(QTGSILIB):    $(QTGSIO) $(QTGSIDO) $(QTGSIMOCO) $(ORDER_) $(MAINLIBS) $(QTGSILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libQtGSI.$(SOEXT) $@ \
		   "$(QTGSIO) $(QTGSIDO) $(QTGSIMOCO)" \
		   "$(QTGSILIBEXTRA) $(QTLIBDIR) $(QTLIB)"

$(QTGSIDS):     $(QTGSIH) $(QTGSIL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DQTVERS=$(QTVERS) $(QTGSIH) $(QTGSIL)

$(QTGSIMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(QTGSIL)
		$(RLIBMAP) -o $@ -l $(QTGSILIB) \
		   -d $(QTGSILIBDEPM) -c $(QTGSIL)

all-$(MODNAME): $(QTGSILIB) $(QTGSIMAP)

test-$(MODNAME): all-$(MODNAME)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(INSTALL) $(QTGSIDIR)/test $(QTGSIDIRT)
endif
		cd $(QTGSIDIRT) && $(MAKE) $(QTTESTOPTS)

clean-$(MODNAME):
		@rm -f $(QTGSIO) $(QTGSIDO) $(QTGSIMOCO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(QTGSIDEP) $(QTGSIMOC) $(QTGSILIB) $(QTGSIMAP)
		@rm -f $(QTGSIDS) $(QTGSIDH)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(QTGSIDIRT)
endif

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(sort $(QTGSIMOCO) $(QTGSIO)): CXXFLAGS := $(filter-out -Wshadow,$(CXXFLAGS))
$(QTGSIDO): CXXFLAGS := $(filter-out -Wshadow,$(CXXFLAGS))

$(sort $(QTGSIMOCO) $(QTGSIO)): CXXFLAGS += $(QTGSICXXFLAGS)
$(QTGSIDO): CXXFLAGS += $(QTGSICXXFLAGS)

$(QTGSIMOC): $(call stripsrc,$(QTGSIDIRS)/moc_%.cxx): $(QTGSIDIRI)/%.h
	$(MAKEDIR)
	$(QTMOCEXE) $< -o $@
