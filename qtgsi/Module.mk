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
QTGSIL        := $(MODDIRI)/LinkDef.h
QTGSIDS       := $(MODDIRS)/G__QtGSI.cxx
QTGSIDO       := $(QTGSIDS:.cxx=.o)
QTGSIDH       := $(QTGSIDS:.cxx=.h)

QTGSIH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
QTGSIS        := $(filter-out $(MODDIRS)/moc_%,\
                 $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))
QTGSIO        := $(QTGSIS:.cxx=.o)

QTGSIMOC      := $(subst $(MODDIRI)/,$(MODDIRS)/moc_,$(patsubst %.h,%.cxx,$(QTGSIH)))
QTGSIMOCO     := $(QTGSIMOC:.cxx=.o)

QTGSIDEP      := $(QTGSIO:.o=.d) $(QTGSIDO:.o=.d) $(QTGSIMOCO:.o=.d)

QTGSICXXFLAGS := -DQT_DLL -DQT_THREAD_SUPPORT -I. $(QTINCDIR:%=-I%)

QTGSILIB      := $(LPATH)/libQtGSI.$(SOEXT)

ifeq ($(PLATFORM),win32)
QTTESTOPTS    := -f Makefile.win
else
QTTESTOPTS    :=
endif

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(QTGSIH))
ALLLIBS       += $(QTGSILIB)

# include all dependency files
INCLUDEFILES  += $(QTGSIDEP)

##### local rules #####
include/%.h:    $(QTGSIDIRI)/%.h
		cp $< $@

$(QTGSILIB):    $(QTGSIO) $(QTGSIDO) $(QTGSIMOCO) $(ORDER_) $(MAINLIBS) $(QTGSILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libQtGSI.$(SOEXT) $@ \
		   "$(QTGSIO) $(QTGSIDO) $(QTGSIMOCO)" \
		   "$(QTGSILIBEXTRA) $(QTLIBDIR) $(QTLIB)"

$(QTGSIDS):     $(QTGSIH) $(QTGSIL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(QTGSIH) $(QTGSIL)

all-qtgsi:      $(QTGSILIB)

test-qtgsi: 	$(QTGSILIB)
		cd $(QTGSIDIR)/test; make $(QTTESTOPTS)

map-qtgsi:      $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(QTGSILIB) \
		   -d $(QTGSILIBDEP) -c $(QTGSIL)

map::           map-qtgsi

clean-qtgsi:
		@rm -f $(QTGSIO) $(QTGSIMOCO)
		-@cd $(QTGSIDIR)/test; make clean

clean::         clean-qtgsi

distclean-qtgsi: clean-qtgsi
		@rm -f $(QTGSIDEP) $(QTGSIMOC) $(QTGSILIB)
		@rm -f $(QTGSIDS) $(QTGSIDH)

distclean::     distclean-qtgsi

##### extra rules ######
$(sort $(QTGSIMOCO) $(QTGSIO)): CXXFLAGS += $(QTGSICXXFLAGS)

$(QTGSIDO): CXXFLAGS += $(QTGSICXXFLAGS)

$(QTGSIMOC): $(QTGSIDIRS)/moc_%.cxx: $(QTGSIDIRI)/%.h
	$(QTMOCEXE) $< -o $@
