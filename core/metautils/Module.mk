# Module.mk for utilities for libMeta and rootcint
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Philippe Canal 9/1/2004

MODNAME        := metautils
MODDIR         := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

METAUTILSDIR   := $(MODDIR)
METAUTILSDIRS  := $(METAUTILSDIR)/src
METAUTILSDIRI  := $(METAUTILSDIR)/inc

##### $(METAUTILSO) #####
METAUTILSH     := $(filter-out $(MODDIRI)/TMetaUtils.%,\
  $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h)))
METAUTILSS     := $(filter-out $(MODDIRS)/TMetaUtils.%,\
  $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))

ifeq ($(BUILDCLING),yes)
METAUTILSTH     += $(MODDIRI)/TMetaUtils.h
METAUTILSTS     += $(MODDIRS)/TMetaUtils.cxx
METAUTILSCXXFLAGS = $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS)))
ifneq ($(CXX:g++=),$(CXX))
METAUTILSCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif
endif

METAUTILSO     := $(call stripsrc,$(METAUTILSS:.cxx=.o))
METAUTILSTO    := $(call stripsrc,$(METAUTILSTS:.cxx=.o))

METAUTILSL     := $(MODDIRI)/LinkDef.h
METAUTILSDS    := $(call stripsrc,$(MODDIRS)/G__MetaUtils.cxx)
METAUTILSDO    := $(METAUTILSDS:.cxx=.o)
METAUTILSDH    := $(METAUTILSDS:.cxx=.h)

METAUTILSDEP   := $(METAUTILSO:.o=.d) $(METAUTILSDO:.o=.d) $(METAUTILSTO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAUTILSH) $(METAUTILSTH))

# include all dependency files
INCLUDEFILES += $(METAUTILSDEP)

#### STL dictionary (replacement for cintdlls)

STLDICTS =

ifeq ($(BUILDCLING),yes)

STLDICTS_NAME = vector list deque map map2 set \
                  multimap multimap2 multiset \
                  stack queue complex
ifneq ($(PLATFORM),win32)
# FIX THEM!
  CINTSTLDLLNAMES += valarray
endif

STLDICTS += lib/libvectorDict.$(SOEXT)
STLDICTS += lib/liblistDict.$(SOEXT)
STLDICTS += lib/libdequeDict.$(SOEXT)
STLDICTS += lib/libmapDict.$(SOEXT)
STLDICTS += lib/libmap2Dict.$(SOEXT)
STLDICTS += lib/libsetDict.$(SOEXT)
STLDICTS += lib/libmultimapDict.$(SOEXT)
STLDICTS += lib/libmultimap2Dict.$(SOEXT)
STLDICTS += lib/libmultisetDict.$(SOEXT)
ifneq ($(PLATFORM),win32)
STLDICTS += lib/libvalarrayDict.$(SOEXT)
endif
STLDICTS += lib/libcomplexDict.$(SOEXT)

STLDICTS_SRC := $(patsubst lib/lib%Dict.$(SOEXT),$(METAUTILSDIRS)/G__std__%.cxx,$(STLDICTS))
STLDICTS_OBJ := $(patsubst %.cxx,%.o,$(STLDICTS_SRC))
STLDICTS_DEP := $(patsubst %.cxx,%.d,$(STLDICTS_SRC))

$(METAUTILSDIRS)/G__std__%.cxx: $(METAUTILSDIRS)/%Linkdef.h $(ROOTCINTTMPDEP)
	$(ROOTCINTTMP) -f $@ -c $(subst multi,,${*:2=}) \
	   $(ROOT_SRCDIR)/core/metautils/src/$*Linkdef.h

$(STLDICTS): lib/lib%Dict.$(SOEXT): $(METAUTILSDIRS)/G__std__%.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ "$(filter-out $(MAINLIBS),$^)" ""

lib/lib%Dict.rootmap: $(RLIBMAP) $(MAKEFILEDEP) $(METAUTILSDIRS)/%Linkdef.h
	$(RLIBMAP) -o $@ -l lib%Dict.$(SOEXT) -c $(METAUTILSDIRS)/$*Linkdef.h

METAUTILSDEP   += $(STLDICTS_DEP)

STLDICTSMAPS = $(STLDICTS:.$(SOEXT)=.rootmap)

.PRECIOUS: $(STLDICTS_SRC)

# used in the main Makefile
ALLLIBS    += $(STLDICTS)
ALLMAPS    += $(STLDICTSMAPS)
   
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METAUTILSDIRI)/%.h
		cp $< $@

$(METAUTILSDS): $(METAUTILSH) $(METAUTILSL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(METAUTILSH) $(METAUTILSL)

all-$(MODNAME): $(METAUTILSO) $(METAUTILSDO) $(STLDICTS)

clean-$(MODNAME):
		@rm -f $(METAUTILSO) $(METAUTILSDO) $(STLDICTS_OBJ) $(STLDICTS_DEP)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METAUTILSDEP) $(METAUTILSDS) $(METAUTILSDH) $(STLDICTS_OBJ) $(STLDICTS_DEP) $(STLDICTS_SRC)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(METAUTILSTO): CXXFLAGS += $(METAUTILSCXXFLAGS)
$(METAUTILSO): CXXFLAGS += $(METAUTILSCXXFLAGS)
ifeq ($(BUILDCLING),yes)
$(METAUTILSO): $(LLVMDEP)
$(METAUTILSTO): $(LLVMDEP)
endif
