# cintdlls.mk for cintdlls target
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2006-09-14

.PHONY: cintdlls distclean-cintdll distclean-cint7dll clean-cintdll distclean-cint7dll

# no: iterator pair
# already in libCore (core/base/inc/Linkdef2.h): string 
CINTSTLDLLNAMES = vector vectorbool list deque map map2 set \
                  multimap multimap2 multiset \
                  stack queue exception stdexcept climits complex
CINTINCDLLNAMES = stdfunc stdcxxfunc
ifneq ($(PLATFORM),win32)
# FIX THEM!
  CINTSTLDLLNAMES += valarray
  CINTINCDLLNAMES += posix ipc
endif
# ".dll", not ".$(SOEXT)"!
CINTDLLS = $(addsuffix .dll,$(addprefix $(CINTDLLDIRSTL)/,$(CINTSTLDLLNAMES)) \
                            $(addprefix $(CINTDLLDIRDLLS)/,$(CINTINCDLLNAMES)))

CINTDLLNAMES = $(CINTSTLDLLNAMES) $(CINTINCDLLNAMES)

ifeq ($(subst cint7,,$(CINTDLLDIRL)),$(CINTDLLDIRL))
BUILDINGCINT :=5
CINT7VERSIONNO:=
else
ifneq ($(BUILDBOTHCINT),)
BUILDINGCINT :=7
CINT7VERSIONNO:=7
else
BUILDINGCINT :=5
CINT7VERSIONNO:=
endif
endif


.PRECIOUS: \
	$(addsuffix .cc ,$(addprefix core/metautils/src/stlLoader$(CINT7VERSIONNO)_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .o  ,$(addprefix core/metautils/src/stlLoader$(CINT7VERSIONNO)_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .cxx,$(addprefix $(CINTDLLDIRDLLSTL)/G__cpp_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .o  ,$(addprefix $(CINTDLLDIRDLLSTL)/G__cpp_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .cxx,$(addprefix $(CINTDLLDIRDLLSTL)/rootcint_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .o  ,$(addprefix $(CINTDLLDIRDLLSTL)/rootcint_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .cxx,$(addprefix $(CINTDLLDIRL)/G__cpp_,$(CINTINCDLLNAMES))) \
	$(addsuffix .o  ,$(addprefix $(CINTDLLDIRL)/G__cpp_,$(CINTINCDLLNAMES))) \
	$(addsuffix .c  ,$(addprefix $(CINTDLLDIRL)/G__c_,$(CINTINCDLLNAMES))) \
	$(addsuffix .o  ,$(addprefix $(CINTDLLDIRL)/G__c_,$(CINTINCDLLNAMES)))

# these need dictionaries
CINTDICTDLLS =
ifneq ($(findstring vector,$(CINTDLLS)),)
CINTDICTDLLS += lib/libvectorDict.$(SOEXT)
$(CINTDLLDIRSTL)/vector.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_vector.o
endif
ifneq ($(findstring list,$(CINTDLLS)),)
CINTDICTDLLS += lib/liblistDict.$(SOEXT)
$(CINTDLLDIRSTL)/list.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_list.o
endif
ifneq ($(findstring deque,$(CINTDLLS)),)
CINTDICTDLLS += lib/libdequeDict.$(SOEXT)
$(CINTDLLDIRSTL)/deque.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_deque.o
endif
ifneq ($(findstring map,$(CINTDLLS)),)
CINTDICTDLLS += lib/libmapDict.$(SOEXT)
CINTDICTDLLS += lib/libmap2Dict.$(SOEXT)
$(CINTDLLDIRSTL)/map.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_map.o
$(CINTDLLDIRSTL)/map2.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_map2.o
endif
ifneq ($(findstring set,$(CINTDLLS)),)
CINTDICTDLLS += lib/libsetDict.$(SOEXT)
$(CINTDLLDIRSTL)/set.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_set.o
endif
ifneq ($(findstring multimap,$(CINTDLLS)),)
CINTDICTDLLS += lib/libmultimapDict.$(SOEXT)
CINTDICTDLLS += lib/libmultimap2Dict.$(SOEXT)
$(CINTDLLDIRSTL)/multimap.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_multimap.o
$(CINTDLLDIRSTL)/multimap2.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_multimap2.o
endif
ifneq ($(findstring multiset,$(CINTDLLS)),)
CINTDICTDLLS += lib/libmultisetDict.$(SOEXT)
$(CINTDLLDIRSTL)/multiset.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_multiset.o
endif
ifneq ($(findstring valarray,$(CINTDLLS)),)
CINTDICTDLLS += lib/libvalarrayDict.$(SOEXT)
$(CINTDLLDIRSTL)/valarray.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_valarray.o
endif

CINTDICTMAPS = $(CINTDICTDLLS:.$(SOEXT)=.rootmap)

ALLCINTDLLS = $(CINTDLLS) $(CINTDICTDLLS)

# used in the main Makefile
ALLLIBS    += $(ALLCINTDLLS)
ALLMAPS    += $(CINTDICTMAPS)

INCLUDEFILES += $(addsuffix .d,$(addprefix core/metautils/src/stlLoader$(CINT7VERSIONNO)_,$(CINTSTLDLLNAMES)))\
   $(CINTDLLDIRL)/posix/mktypes.d $(CINTDLLDIRL)/posix/exten.d

cintdlls: $(ALLCINTDLLS)

CINTCPPDEP := $(CINTDLLDICTVER) $(ORDER_) $(CINTDLLCINTTMP) $(CINTDLLIOSENUM)

$(CINTDLLDIRDLLSTL)/G__cpp_string.cxx:	$(CINTDLLDIRL)/dll_stl/str.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_vector.cxx:	$(CINTDLLDIRL)/dll_stl/vec.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_vectorbool.cxx: $(CINTDLLDIRL)/dll_stl/vecbool.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_list.cxx:	$(CINTDLLDIRL)/dll_stl/lst.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_deque.cxx:	$(CINTDLLDIRL)/dll_stl/dqu.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_map.cxx:	$(CINTDLLDIRL)/dll_stl/mp.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_map2.cxx:	$(CINTDLLDIRL)/dll_stl/mp.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_set.cxx:	$(CINTDLLDIRL)/dll_stl/st.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_multimap.cxx:	$(CINTDLLDIRL)/dll_stl/multmp.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_multimap2.cxx:	$(CINTDLLDIRL)/dll_stl/multmp.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_multiset.cxx:	$(CINTDLLDIRL)/dll_stl/multst.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_stack.cxx:	$(CINTDLLDIRL)/dll_stl/stk.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_queue.cxx:	$(CINTDLLDIRL)/dll_stl/que.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_valarray.cxx:	$(CINTDLLDIRL)/dll_stl/vary.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_exception.cxx:	$(CINTDLLDIRL)/dll_stl/cinteh.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_stdexcept.cxx:	$(CINTDLLDIRL)/dll_stl/se.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_climits.cxx:	$(CINTDLLDIRL)/dll_stl/clim.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_complex.cxx:	$(CINTDLLDIRL)/dll_stl/cmplx.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_iterator.cxx:	$(CINTDLLDIRL)/dll_stl/iter.h $(CINTCPPDEP)
$(CINTDLLDIRDLLSTL)/G__cpp_pair.cxx:	$(CINTDLLDIRL)/dll_stl/pr.h $(CINTCPPDEP)

$(CINTDLLDIRL)/G__cpp_stdcxxfunc.cxx: 	$(CINTDLLDIRL)/stdstrct/stdcxxfunc.h $(CINTCPPDEP)
$(CINTDLLDIRL)/G__c_stdfunc.c:		$(CINTDLLDIRL)/stdstrct/stdfunc.h $(CINTCPPDEP)
$(CINTDLLDIRL)/G__c_posix.c:		$(CINTDLLDIRL)/posix/exten.h $(CINTCPPDEP)
$(CINTDLLDIRL)/G__c_ipc.c:			$(CINTDLLDIRL)/ipc/ipcif.h $(CINTCPPDEP)

##### all cintdlls end on .dll
ifneq ($(SOEXT),dll)
CINTDLLSOEXTCMD = mv $(@:.dll=.$(SOEXT)) $@
ifeq ($(PLATFORM),macosx)
ifeq ($(subst $(MACOSX_MINOR),,456789),456789)
# MACOSX_MINOR < 4
  CINTDLLSOEXTCMD += ;mv $(@:.dll=.so) $@
else
  CINTDLLSOEXTCMD += ;rm -f $(@:.dll=.so)
endif
endif # macosx
endif # need to mv to .dll
##### all cintdlls end on .dll - END

# Filter out the explicit link flag
ifneq ($(subst build/unix/makelib.sh,,$(MAKELIB)),$(MAKELIB))
  $(CINTDLLS): MAKELIB := $(subst -x,,$(MAKELIB))
endif

$(ALLCINTDLLS): $(ORDER_) $(MAINLIBS)

$(CINTDLLDIRSTL)/%.dll: $(CINTDLLDIRDLLSTL)/G__cpp_%.o 
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $(@:.dll=.$(SOEXT))) $(@:.dll=.$(SOEXT)) $(filter-out $(MAINLIBS),$^)
	$(CINTDLLSOEXTCMD)
ifneq ($(subst win,,$(ARCH)),$(ARCH))
	@rm -f $(@:.dll=.lib) $(@:.dll=.exp) # remove import libs
endif

$(CINTDLLDIRDLLS)/stdcxxfunc.dll: $(CINTDLLDIRL)/G__cpp_stdcxxfunc.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $(@:.dll=.$(SOEXT))) $(@:.dll=.$(SOEXT)) $(filter-out $(MAINLIBS),$^)
	$(CINTDLLSOEXTCMD)
ifneq ($(subst win,,$(ARCH)),$(ARCH))
	@rm -f $(@:.dll=.lib) $(@:.dll=.exp) # remove import libs
endif

$(CINTDLLDIRDLLS)/%.dll: $(CINTDLLDIRL)/G__c_%.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $(@:.dll=.$(SOEXT))) $(@:.dll=.$(SOEXT)) $(filter-out $(MAINLIBS),$^)
	$(CINTDLLSOEXTCMD)

core/metautils/src/stlLoader$(CINT7VERSIONNO)_%.cc: core/metautils/src/stlLoader.cc
	cp -f $< $@

core/metautils/src/stlLoader$(CINT7VERSIONNO)_%.o: core/metautils/src/stlLoader$(CINT7VERSIONNO)_%.cc
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- $(CINTDLLCXXFLAGS) -D__cplusplus -- $<
	$(CXX) $(OPT) $(CINTDLLCXXFLAGS) $(INCDIRS) -DWHAT=\"$*\" $(CXXOUT)$@ -c $<

$(CINTDLLDIRDLLSTL)/G__cpp_%.cxx:
	$(patsubst %lib/dll_stl/,%,$(dir $@))/main/cint_tmp \
           -w1 -z$(notdir $*) -n$@ $(subst $*,,$(patsubst %map2,-DG__MAP2,$*)) \
	   -D__MAKECINT__ -DG__MAKECINT \
           $(addprefix $(patsubst %lib/dll_stl/,-I%,$(dir $@)),lib/dll_stl lib) \
	   -c-1 -A -Z0 $(filter-out $(IOSENUM),$(filter %.h,$^))

$(CINTDLLDIRL)/G__cpp_%.cxx:
	$(patsubst %lib/,%,$(dir $@))/main/cint_tmp \
	   -w1 -z$(notdir $*) -n$@ $(subst $*,,$(patsubst %map2,-DG__MAP2,$*)) \
	   -D__MAKECINT__ -DG__MAKECINT -I$(dir $@) \
	   -c-1 -A -Z0 $(filter-out $(IOSENUM),$(filter %.h,$^))

$(CINTDLLDIRL)/G__c_%.c:
	$(patsubst %lib/,%,$(dir $@))/main/cint_tmp \
	   -K -w1 -z$(notdir $*) -n$@ -D__MAKECINT__ -DG__MAKECINT \
	   $(MACOSX_UNIX03) -c-2 -Z0 $(filter-out $(IOSENUM),$(filter %.h,$^))

ifeq ($(subst $(MACOSX_MINOR),,1234),1234)
# MACOSX_MINOR > 4
$(CINTDLLDIRL)/G__c_posix.c: MACOSX_UNIX03 = -D__DARWIN_UNIX03
endif

$(CINTDLLDIRL)/G__c_%.o: CFLAGS := $(filter-out -Iinclude,$(CINTDLLCFLAGS)) -I. -DG__SYSTYPES_H

##### posix special treatment
$(CINTDLLDIRL)/posix/exten.o: $(CINTDLLDIRL)/posix/exten.c
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- $(CINTCFLAGS) -- $<
	$(CC) $(OPT) $(CINTCFLAGS) -I. $(CXXOUT)$@ -c $<

$(CINTDLLDIRL)/G__c_posix.c: $(CINTDLLDIRDLLS)/sys/types.h $(CINTDLLDIRL)/posix/exten.h $(CINTDLLDIRL)/posix/posix.h

$(CINTDLLDIRDLLS)/posix.dll: $(CINTDLLDIRL)/G__c_posix.o $(CINTDLLDIRL)/posix/exten.o

$(CINTDLLDIRDLLS)/sys/types.h: $(CINTDLLDIRL)/posix/mktypes$(EXEEXT)
	(cd $(dir $<) && \
	./$(notdir $<))
	cp -f $(@:/sys/types.h=/systypes.h) $@

$(CINTDLLDIRL)/posix/mktypes$(EXEEXT): $(CINTDLLDIRL)/posix/mktypes.c
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- $(CINTCFLAGS) -- $<
	$(CC) $(OPT) $(CXXOUT)$@ $<
##### posix special treatment - END

##### ipc special treatment
$(CINTDLLDIRDLLS)/sys/ipc.dll: $(CINTDLLDIRL)/G__c_ipc.o

##### ipc special treatment - END

##### dictionaries
ifeq ($(BUILDBOTHCINT),)
$(CINTDLLDIRDLLSTL)/rootcint_%.cxx: core/metautils/src/%Linkdef.h $(CINTDLLROOTCINTTMPDEP)
	core/utils/src/rootcint_tmp -f $@ -c \
	   $(subst multi,,${*:2=}) \
	   core/metautils/src/$*Linkdef.h
else
$(CINTDLLDIRDLLSTL)/rootcint_%.cxx: core/metautils/src/%Linkdef.h $(CINTDLLROOTCINTTMPDEP)
	core/utils/src/root$(patsubst cint/%/lib/dll_stl/,%,$(dir $@))_tmp -f $@ -c \
	   $(subst multi,,${*:2=}) \
	   core/metautils/src/$*Linkdef.h
endif

$(patsubst lib/lib%Dict.$(SOEXT),$(CINTDLLDIRDLLSTL)/rootcint_%.o,$(CINTDICTDLLS)): CINTCXXFLAGS += -I.
$(patsubst lib/lib%Dict.$(SOEXT),$(CINTDLLDIRDLLSTL)/rootcint_%.cxx,$(CINTDICTDLLS)): $(CINTDLLROOTCINTTMPDEP)

ifeq ($(BUILDINGCINT),5)
$(CINTDICTMAPS): lib/lib%Dict.rootmap: bin/rlibmap$(EXEEXT) $(MAKEFILEDEP)
	$(RLIBMAP) -o $@ -l \
		    $*.dll -c core/metautils/src/$*Linkdef.h

$(CINTDICTDLLS): lib/lib%Dict.$(SOEXT): $(CINTDLLDIRDLLSTL)/rootcint_%.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ $(filter-out $(MAINLIBS),$^)
endif

##### dictionaries - END

##### clean

# remove only .o, .dll, .$(SOEXT)
ifeq ($(BUILDINGCINT),5)
CLEANCINTDLLSTARGET := cintdll
else
CLEANCINTDLLSTARGET := cint7dll
endif

clean-$(CLEANCINTDLLSTARGET):
	@(for cintdll in $(CINTDLLNAMES); do \
	  rm -f $(patsubst clean-%dll,cint/%,$@)/lib/dll_stl/rootcint_$${cintdll}.o \
	  $(patsubst clean-%dll,cint/%,$@)/lib/dll_stl/G__cpp_$${cintdll}.o \
	  $(patsubst clean-%dll,cint/%,$@)/lib/G__c_$${cintdll}.o \
	  $(patsubst clean-%dll,cint/%,$@)/lib/G__cpp_$${cintdll}.o \
	  core/metautils/src/stlLoader$(CINT7VERSIONNO)_$${cintdll}.o; done)
	@rm -f $(ALLCINTDLLS) \
	  $(patsubst clean-%dll,cint/%,$@)/lib//posix/exten.o \
	  $(patsubst clean-%dll,cint/%,$@)/include/posix.* \
	  $(patsubst clean-%dll,cint/%,$@)/include/ipc.*

clean:: clean-$(CLEANCINTDLLSTARGET)

# remove generated code, too.
distclean-$(CLEANCINTDLLSTARGET): clean-$(CLEANCINTDLLSTARGET)
	@(for cintdll in $(CINTDLLNAMES); do \
	  rm -f $(patsubst distclean-%dll,cint/%,$@)/lib/dll_stl/rootcint_$${cintdll}.* \
	  $(patsubst distclean-%dll,cint/%,$@)/lib/dll_stl/G__cpp_$${cintdll}.* \
	  $(patsubst distclean-%dll,cint/%,$@)/lib/G__c_$${cintdll}.* \
	  $(patsubst distclean-%dll,cint/%,$@)/lib/G__cpp_$${cintdll}.* \
	  core/metautils/src/stlLoader$(CINT7VERSIONNO)_$${cintdll}.*; done)
	@rm -f $(ALLCINTDLLS) $(CINTDICTMAPS) \
	  $(patsubst distclean-%dll,cint/%,$@)/lib/posix/mktypes$(EXEEXT)
ifeq ($(PLATFORM),macosx)
	@rm -f  $(patsubst distclean-%dll,cint/%,$@)/stl/*.so
	@rm -rf $(patsubst distclean-%dll,cint/%,$@)/lib/posix/mktypes.dSYM
endif

distclean:: distclean-$(CLEANCINTDLLSTARGET)

##### clean - END
