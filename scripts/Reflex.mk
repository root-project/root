# Generic chain of files for running Reflex tests.
# Naming scheme: MyReflex_test() in MyReflex_test.C tests
# the dictionary MyReflex_rflx.cxx which is generated from
# MyReflex.h with MyReflex_selection.xml.

ifeq ($(PLATFORM),win32)
GENREFLEX = genreflex.bat
GENREFLEXCXXFLAGS = --gccxmlopt='-include "$(ROOTSYS)/include/w32pragma.h"'
else
GENREFLEX = genreflex
endif

# We only run these tests if libCintex exists and gccxml is found
ifeq ($(findstring cintex,$(shell root-config --features)),)
HAVEGCCXML:="nocintex"
endif
HAVEGCCXML?=$(shell which gccxml 2>/dev/null)

# The dictionary:
%_rflx.cpp: %_selection.xml %.h
	$(CMDECHO) $(GENREFLEX) $*.h -s $*_selection.xml $(GENREFLEXFLAGS) -I"$(ROOTSYS)/include" $(GENREFLEXCXXFLAGS) > $@.log

%_cint.cpp: %.h %LinkDef.h
	$(CMDECHO) rootcint -f $@ -c $^

lib%_dictrflx.$(DllSuf): %_rflx.o
	$(BuildFromObj)

lib%_dictcint.$(DllSuf): %_cint.o
	$(BuildFromObj)

ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)


endif
