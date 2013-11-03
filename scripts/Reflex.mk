# Generic chain of files for running Reflex tests.
# Naming scheme: MyReflex_test() in MyReflex_test.C tests
# the dictionary MyReflex_rflx.cxx which is generated from
# MyReflex.h with MyReflex_selection.xml.

ifeq ($(PLATFORM),win32)
GENREFLEX = genreflex.bat
GENREFLEXCXXFLAGS = --gccxmlopt='-include "$(ROOTSYS)/include/w32pragma.h"'
ifneq ($(findstring -MDd,$(shell root-config --cflags)),)
GENREFLEXCXXFLAGS += -D_DEBUG
endif
else
GENREFLEX = genreflex
endif

# The dictionary:
%_rflx.cpp: %_selection.xml %.h
	$(CMDECHO) $(GENREFLEX) $*.h -s $*_selection.xml $(GENREFLEXFLAGS) -I"$(ROOTSYS)/include" $(GENREFLEXCXXFLAGS) > $@.log || (cat $@.log && exit 1)

%_cint.cpp: %.h %LinkDef.h
	$(CMDECHO) rootcint -f $@ -c $^

lib%_dictrflx.$(DllSuf): %_rflx.$(ObjSuf)
	$(BuildFromObj)

lib%_dictcint.$(DllSuf): %_cint.$(ObjSuf)
	$(BuildFromObj)

