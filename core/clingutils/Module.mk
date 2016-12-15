# Module.mk for utilities for libMeta and rootcint
# Copyright (c) 1995-2016 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2016-12-14

MODNAME        := clingutils
MODDIR         := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

CLINGUTILSDIR   := $(MODDIR)
CLINGUTILSDIRS  := $(CLINGUTILSDIR)/src
CLINGUTILSDIRI  := $(CLINGUTILSDIR)/inc
CLINGUTILSDIRR  := $(CLINGUTILSDIR)/res

##### $(CLINGUTILSO) #####
CLINGUTILSS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CLINGUTILSH     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))

CLINGUTILSCXXFLAGS = $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS)))
ifneq ($(CXX:g++=),$(CXX))
CLINGUTILSCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif

CLINGUTILSO     := $(call stripsrc,$(CLINGUTILSS:.cxx=.o))

CLINGUTILSDEP   := $(CLINGUTILSO:.o=.d)

# used in the main Makefile
CLINGUTILS_REL := $(patsubst $(MODDIRI)/%,include/%,$(CLINGUTILSH))
ALLHDRS     += $(CLINGUTILS_REL)

# include all dependency files
INCLUDEFILES += $(CLINGUTILSDEP)

#### STL dictionary (replacement for cintdlls)

STLDICTS =
STLDICTS += lib/libvectorDict.$(SOEXT)
STLDICTS += lib/liblistDict.$(SOEXT)
STLDICTS += lib/libforward_listDict.$(SOEXT)
STLDICTS += lib/libdequeDict.$(SOEXT)
STLDICTS += lib/libmapDict.$(SOEXT)
STLDICTS += lib/libmap2Dict.$(SOEXT)
STLDICTS += lib/libsetDict.$(SOEXT)
STLDICTS += lib/libunordered_setDict.$(SOEXT)
STLDICTS += lib/libunordered_multisetDict.$(SOEXT)
STLDICTS += lib/libunordered_mapDict.$(SOEXT)
STLDICTS += lib/libunordered_multimapDict.$(SOEXT)
STLDICTS += lib/libmultimapDict.$(SOEXT)
STLDICTS += lib/libmultimap2Dict.$(SOEXT)
STLDICTS += lib/libmultisetDict.$(SOEXT)
STLDICTS += lib/libcomplexDict.$(SOEXT)
ifneq ($(PLATFORM),win32)
STLDICTS += lib/libvalarrayDict.$(SOEXT)
endif

STLDICTS_SRC := $(call stripsrc,$(patsubst lib/lib%Dict.$(SOEXT),$(CLINGUTILSDIRS)/G__std__%.cxx,$(STLDICTS)))
STLDICTS_HDR := $(call stripsrc,$(patsubst lib/lib%Dict.$(SOEXT),$(CLINGUTILSDIRS)/G__std__%.h,$(STLDICTS)))
STLDICTS_OBJ := $(patsubst %.cxx,%.o,$(STLDICTS_SRC))
STLDICTS_DEP := $(patsubst %.cxx,%.d,$(STLDICTS_SRC))

$(call stripsrc,$(CLINGUTILSDIRS)/G__std__%.cxx): $(CLINGUTILSDIRS)/%Linkdef.h $(ROOTCLINGSTAGE1DEP)
	$(ROOTCLINGSTAGE1) -f $@ -s lib/lib$*Dict.pcm -m lib/libCore.pcm $(call rootmapModule, lib/lib$*Dict.$(SOEXT)) -c $(subst multi,,${*:2=}) $(ROOT_SRCDIR)/core/clingutils/src/$*Linkdef.h

$(STLDICTS): lib/lib%Dict.$(SOEXT): $(call stripsrc,$(CLINGUTILSDIRS)/G__std__%.o) $(ORDER_) $(MAINLIBS)
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ "$(filter-out $(MAINLIBS),$^)" ""

lib/lib%Dict.rootmap: $(CLINGUTILSDIRS)/%Linkdef.h $(ROOTCLINGSTAGE1DEP)
	$(ROOTCLINGSTAGE1) -r $(CLINGUTILSDIRS)/G__std__%.cxx -s lib/lib$*Dict.pcm -m lib/libCore.pcm $(call rootmapModule, lib/lib$*Dict.$(SOEXT))\
	   -c $(subst multi,,${*:2=}) \
	   $(ROOT_SRCDIR)/core/clingutils/src/$*Linkdef.h

CLINGUTILSDEP   += $(STLDICTS_DEP)

STLDICTSMAPS = $(STLDICTS:.$(SOEXT)=.rootmap)

.PRECIOUS: $(STLDICTS_SRC)

# used in the main Makefile
ALLLIBS    += $(STLDICTS)
ALLMAPS    += $(STLDICTSMAPS)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(CLINGUTILSDIRI)/%.h
		cp $< $@

all-$(MODNAME): $(CLINGUTILSO) $(STLDICTS)

clean-$(MODNAME):
		@rm -f $(CLINGUTILSO) $(STLDICTS_OBJ) \
		   $(STLDICTS_DEP)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CLINGUTILSDEP) \
		   $(STLDICTS_OBJ) $(STLDICTS_DEP) $(STLDICTS_SRC) \
		   $(STLDICTS_HDR) $(STLDICTSMAPS)

distclean::     distclean-$(MODNAME)

 ##### extra rules ######
 $(CLINGUTILSO): CXXFLAGS += $(CLINGUTILSCXXFLAGS) -I$(DICTGENDIRR) -I$(CLINGUTILSDIRR) -I$(FOUNDATIONDIRR)
 $(CLINGUTILSO): $(LLVMDEP)
