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

.PRECIOUS: %_rflx.cpp %_cling.cpp %.o

# The dictionary:
%_rflx.cpp: %_selection.xml %.h $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(GENREFLEX) $*.h -s $*_selection.xml $(GENREFLEXFLAGS) -I"$(ROOTSYS)/include" $(GENREFLEXCXXFLAGS) > $@.log 2>&1 || (cat $@.log && exit 1)

%_cling.cpp: %.h %LinkDef.h $(ROOTCINT) $(ROOTV)
	$(CMDECHO) rootcint -f $@ $*.h $*LinkDef.h  > $@.log 2>&1 || (cat $@.log && exit 1)

%_cling.cpp: %.h %_linkdef.h $(ROOTCINT) $(ROOTV)
	$(CMDECHO) rootcint -f $@ $*.h $*_linkdef.h  > $@.log 2>&1 || (cat $@.log && exit 1)

lib%_dictrflx.$(DllSuf): %_rflx.$(ObjSuf) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(BuildFromObj)

lib%_dictcint.$(DllSuf): %_cling.$(ObjSuf) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(BuildFromObj)

