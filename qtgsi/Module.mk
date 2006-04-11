# Module.mk for qtgsi module
# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Author: Bertrand Bellenot, 22/02/2006

MODDIR       := qtgsi
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

QTGSIDIR     := $(MODDIR)
QTGSIDIRS    := $(QTGSIDIR)/src
QTGSIDIRI    := $(QTGSIDIR)/inc

##### libQtGSI #####
QTGSIH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
QTGSIS        := $(filter-out $(MODDIRS)/moc_%,\
                 $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))
QTGSIO        := $(QTGSIS:.cxx=.o)

QTGSIMOC      := $(subst $(MODDIRI)/,$(MODDIRS)/moc_,$(patsubst %.h,%.cxx,$(QTGSIH)))
QTGSIMOCO     := $(QTGSIMOC:.cxx=.o)

QTGSIDEP      := $(QTGSIO:.o=.d) $(QTGSIMOCO:.o=.d)

QTGSICXXFLAGS := -DQT_DLL -DQT_THREAD_SUPPORT -I. $(QTINCDIR:%=-I%)

QTGSILIB      := $(LPATH)/libQtGSI.$(SOEXT)

ifeq ($(PLATFORM),win32)
ifeq (yes,$(WINRTDEBUG))
QTGSINMCXXFLAGS := "$(BLDCXXFLAGS)" DEBUG=1
else
QTGSINMCXXFLAGS := "$(BLDCXXFLAGS)"
endif
QTTESTOPTS    := -f Makefile.win INMCXXFLAGS:=$(QTGSINMCXXFLAGS)
QTTESTOPTS    += GSIQTLIBI:="$(QTLIBDIR)$(QTLIB)"
else
QTTESTOPTS    := GSIQTLIBI:="$(QTLIBDIR) $(QTLIB)"
endif

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(QTGSIH))
ALLLIBS       += $(QTGSILIB)

# include all dependency files
INCLUDEFILES  += $(QTGSIDEP)

##### local rules #####
include/%.h:    $(QTGSIDIRI)/%.h
		cp $< $@

$(QTGSILIB):    $(QTGSIO) $(QTGSIMOCO) $(ORDER_) $(MAINLIBS) $(QTGSILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libQtGSI.$(SOEXT) $@ "$(QTGSIO) $(QTGSIMOCO)" \
		   "$(QTGSILIBEXTRA) $(QTLIBDIR) $(QTLIB)"

all-qtgsi:      $(QTGSILIB)

test-qtgsi: 	$(QTGSILIB)
		cd $(QTGSIDIR)/test; make $(QTTESTOPTS)

clean-qtgsi:
		@rm -f $(QTGSIO) $(QTGSIMOCO)
		-@cd $(QTGSIDIR)/test; make clean

clean::         clean-qtgsi

distclean-qtgsi:   clean-qtgsi
		@rm -f $(QTGSIDEP) $(QTGSIMOC) $(QTGSILIB)

distclean::     distclean-qtgsi

##### extra rules ######
$(sort $(QTGSIMOCO) $(QTGSIO)): CXXFLAGS += $(QTGSICXXFLAGS)

$(QTGSIMOC): $(QTGSIDIRS)/moc_%.cxx: $(QTGSIDIRI)/%.h
	$(QTMOCEXE) $< -o $@
