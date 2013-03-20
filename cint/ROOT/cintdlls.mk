# cintdlls.mk for cintdlls target
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2006-09-14

MODNAME      := cintdlls

.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

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

CINTDLLS_SOURCE_FILES = $(addsuffix .cc ,$(addprefix core/metautils/src/stlLoader_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .cxx,$(addprefix $(CINTDLLDIRDLLSTL)/G__cpp_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .cxx,$(addprefix $(CINTDLLDIRDLLSTL)/rootcint_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .cxx,$(addprefix $(CINTDLLDIRL)/G__cpp_,$(CINTINCDLLNAMES))) \
	$(addsuffix .c  ,$(addprefix $(CINTDLLDIRL)/G__c_,$(CINTINCDLLNAMES)))

.PRECIOUS: $(CINTDLLS_SOURCE_FILES) \
	$(addsuffix .o  ,$(addprefix core/metautils/src/stlLoader_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .o  ,$(addprefix $(CINTDLLDIRDLLSTL)/G__cpp_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .o  ,$(addprefix $(CINTDLLDIRDLLSTL)/rootcint_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .o  ,$(addprefix $(CINTDLLDIRL)/G__cpp_,$(CINTINCDLLNAMES))) \
	$(addsuffix .o  ,$(addprefix $(CINTDLLDIRL)/G__c_,$(CINTINCDLLNAMES)))

# these need dictionaries
CINTDICTDLLS =
ifneq ($(findstring vector,$(CINTDLLS)),)
CINTDICTDLLS += lib/libvectorDict.$(SOEXT)
$(CINTDLLDIRSTL)/vector.dll: core/metautils/src/stlLoader_vector.o
endif
ifneq ($(findstring list,$(CINTDLLS)),)
CINTDICTDLLS += lib/liblistDict.$(SOEXT)
$(CINTDLLDIRSTL)/list.dll: core/metautils/src/stlLoader_list.o
endif
ifneq ($(findstring deque,$(CINTDLLS)),)
CINTDICTDLLS += lib/libdequeDict.$(SOEXT)
$(CINTDLLDIRSTL)/deque.dll: core/metautils/src/stlLoader_deque.o
endif
ifneq ($(findstring map,$(CINTDLLS)),)
CINTDICTDLLS += lib/libmapDict.$(SOEXT)
CINTDICTDLLS += lib/libmap2Dict.$(SOEXT)
$(CINTDLLDIRSTL)/map.dll: core/metautils/src/stlLoader_map.o
$(CINTDLLDIRSTL)/map2.dll: core/metautils/src/stlLoader_map2.o
endif
ifneq ($(findstring set,$(CINTDLLS)),)
CINTDICTDLLS += lib/libsetDict.$(SOEXT)
$(CINTDLLDIRSTL)/set.dll: core/metautils/src/stlLoader_set.o
endif
ifneq ($(findstring multimap,$(CINTDLLS)),)
CINTDICTDLLS += lib/libmultimapDict.$(SOEXT)
CINTDICTDLLS += lib/libmultimap2Dict.$(SOEXT)
$(CINTDLLDIRSTL)/multimap.dll: core/metautils/src/stlLoader_multimap.o
$(CINTDLLDIRSTL)/multimap2.dll: core/metautils/src/stlLoader_multimap2.o
endif
ifneq ($(findstring multiset,$(CINTDLLS)),)
CINTDICTDLLS += lib/libmultisetDict.$(SOEXT)
$(CINTDLLDIRSTL)/multiset.dll: core/metautils/src/stlLoader_multiset.o
endif
ifneq ($(findstring valarray,$(CINTDLLS)),)
CINTDICTDLLS += lib/libvalarrayDict.$(SOEXT)
$(CINTDLLDIRSTL)/valarray.dll: core/metautils/src/stlLoader_valarray.o
endif
ifneq ($(findstring complex,$(CINTDLLS)),)
CINTDICTDLLS += lib/libcomplexDict.$(SOEXT)
$(CINTDLLDIRSTL)/complex.dll: core/metautils/src/stlLoader$(CINT7VERSIONNO)_complex.o
endif

CINTDICTMAPS = $(CINTDICTDLLS:.$(SOEXT)=.rootmap)

ALLCINTDLLS = $(CINTDLLS) $(CINTDICTDLLS)

# used in the main Makefile
ALLLIBS    += $(ALLCINTDLLS)
ALLMAPS    += $(CINTDICTMAPS)

CINTDLLS_DEPENDENCY_FILES = $(addsuffix .d,$(addprefix core/metautils/src/stlLoader_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .d,$(addprefix $(CINTDLLDIRL)/G__cpp_,$(CINTINCDLLNAMES))) \
	$(addsuffix .d,$(addprefix $(CINTDLLDIRDLLSTL)/G__cpp_,$(CINTSTLDLLNAMES))) \
	$(addsuffix .d,$(addprefix $(CINTDLLDIRDLLSTL)/rootcint_,$(CINTSTLDLLNAMES))) \
	$(CINTDLLDIRL)/posix/mktypes.d $(CINTDLLDIRL)/posix/exten.d

cintdlls_cleanup_dependency_files_trigger := $(shell grep ORDER_ $(wildcard $(CINTDLLS_DEPENDENCY_FILES)) /dev/null > /dev/null && ( rm -f `find . -name \*.d -exec grep -c ORDER_ {} /dev/null \; 2>&1 | grep -v ':0' | cut -d: -f1 | sed -e 's/\.d/\.o/' `  1>&2 ) )

INCLUDEFILES += $(CINTDLLS_DEPENDENCY_FILES)

all-$(MODNAME): $(ALLCINTDLLS) $(CINTDICTMAPS)

CINTCPPDEP := $(CINTDLLDICTVER) $(ORDER_) $(CINTDLLCINTTMP) $(CINTDLLIOSENUM)

ifeq ($(EXPLICITLINK),yes)
ifeq ($(PLATFORM),win32)
CINTDLLLIBLINK := lib/libCint.lib
else
CINTDLLLIBLINK := -Llib -lCint
endif
endif

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
  # On macosx one should change the install_name as well.
  # FIXME: not tested on 10.4, should be the same also there?
  CINTDLLSOEXTCMD += ;install_name_tool -id `otool -D $@ | tail -1 | sed -e's|:$$||;s|[.]so$$|.dll|'` $@
  CINTDLLSOEXTCMD += ;rm -f $(@:.dll=.so)
endif
endif # macosx
endif # need to mv to .dll
##### all cintdlls end on .dll - END

# Filter out the explicit link flag
ifneq ($(subst $(ROOT_SRCDIR)/build/unix/makelib.sh,,$(MAKELIB)),)
  SPACE:= # a space.
  SPACE+= # Actually create the space by appending 'nothing'
  $(CINTDLLS): MAKELIB := $(subst $(SPACE)-x,,$(MAKELIB))
endif

$(ALLCINTDLLS): $(ORDER_) $(MAINLIBS)

$(CINTDLLDIRSTL)/%.dll: $(CINTDLLDIRDLLSTL)/G__cpp_%.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $(@:.dll=.$(SOEXT))) $(@:.dll=.$(SOEXT)) "$(filter-out $(MAINLIBS),$^)" "$(CINTDLLLIBLINK)"
	$(CINTDLLSOEXTCMD)
ifneq ($(subst win,,$(ARCH)),$(ARCH))
	@rm -f $(@:.dll=.lib) $(@:.dll=.exp) # remove import libs
endif

$(CINTDLLDIRDLLS)/stdcxxfunc.dll: $(CINTDLLDIRL)/G__cpp_stdcxxfunc.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $(@:.dll=.$(SOEXT))) $(@:.dll=.$(SOEXT)) "$(filter-out $(MAINLIBS),$^)" "$(CINTDLLLIBLINK)"
	$(CINTDLLSOEXTCMD)
ifneq ($(subst win,,$(ARCH)),$(ARCH))
	@rm -f $(@:.dll=.lib) $(@:.dll=.exp) # remove import libs
endif

$(CINTDLLDIRDLLS)/%.dll: $(CINTDLLDIRL)/G__c_%.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $(@:.dll=.$(SOEXT))) $(@:.dll=.$(SOEXT)) "$(filter-out $(MAINLIBS),$^)" "$(CINTDLLLIBLINK)"
	$(CINTDLLSOEXTCMD)

core/metautils/src/stlLoader_%.cc: $(ROOT_SRCDIR)/core/metautils/src/stlLoader.cc
	cp -f $< $@

core/metautils/src/stlLoader_%.o: core/metautils/src/stlLoader_%.cc
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- $(CINTDLLCXXFLAGS) -D__cplusplus -- $<
	$(CXX) $(OPT) $(CINTDLLCXXFLAGS) $(INCDIRS) -DWHAT=\"$*\" $(CXXOUT)$@ -c $<

$(CINTDLLDIRDLLSTL)/G__cpp_%.cxx:
	$(CINTDLLCINTTMP) \
           -w1 -z$(notdir $*) -n$@ $(subst $*,,$(patsubst %map2,-DG__MAP2,$*)) \
	   -D__MAKECINT__ -DG__MAKECINT \
           $(addprefix $(patsubst %lib/dll_stl/,-I%,$(dir $@)),lib/dll_stl lib) \
	   -c-1 -A -Z0 $(filter-out $(CINTDLLDIRDLLSTL)/G__cpp_%, $(filter-out $(IOSENUM),$(filter $(CINTDLLDIRDLLSTL)/%,$(filter %.h,$^))))
	touch $@

$(CINTDLLDIRL)/G__cpp_%.cxx:
	$(CINTDLLCINTTMP) \
	   -w1 -z$(notdir $*) -n$@ $(subst $*,,$(patsubst %map2,-DG__MAP2,$*)) \
	   -D__MAKECINT__ -DG__MAKECINT -I$(dir $@) \
	   -c-1 -A -Z0 $(filter-out $(CINTDLLDIRL)/G__cpp%, $(filter-out $(IOSENUM),$(filter $(CINTDLLDIRL)/%,$(filter %.h,$^))))
	touch $@

$(CINTDLLDIRL)/G__c_%.c:
	$(CINTDLLCINTTMP) \
	   -K -w1 -z$(notdir $*) -n$@ -D__MAKECINT__ -DG__MAKECINT \
	   $(MACOSX_UNIX03) -c-2 -Z0 $(filter-out $(CINTDLLDIRL)/G__c_%, $(filter-out $(IOSENUM),$(filter $(CINTDLLDIRL)/%,$(filter %.h,$^))))
	touch $@

$(CINTDLLDIRDLLSTL)/G__cpp_complex.cxx: $(CINTDLLDIRL)/dll_stl/cmplx.h $(CINTCPPDEP)
	$(CINTDLLCINTTMP) \
           -w1 -z$(notdir $*) -n$@ $(subst $*,,$(patsubst %map2,-DG__MAP2,$*)) \
	   -D__MAKECINT__ -DG__MAKECINT \
           $(addprefix $(patsubst %lib/dll_stl/,-I%,$(dir $@)),lib/dll_stl lib) \
	   -V -c-1 -A -Z0 $(CINTDLLDIRL)/dll_stl/cmplx.h
	touch $@

ifeq ($(subst $(MACOSX_MINOR),,1234),1234)
# MACOSX_MINOR > 4
$(CINTDLLDIRL)/G__c_posix.c: MACOSX_UNIX03 = -D__DARWIN_UNIX03
endif

ifneq (,$(GCC_MAJOR))
# gcc
ifeq (,$(findstring $(GCC_MAJOR),1 2 3))
ifneq ($(GCC_MAJOR),4)
# GCC 5 and up
CINTDLLCDEPR := -Wno-deprecated-declarations
else
ifeq (,$(findstring $(GCC_MINOR),0 1 2 3 4 5 6))
# GCC 4.7 and up
CINTDLLCDEPR := -Wno-deprecated-declarations
endif
endif
endif
endif

$(CINTDLLDIRL)/G__c_%.o: CFLAGS := $(filter-out -Iinclude,$(CINTDLLCFLAGS)) -I. -DG__SYSTYPES_H $(CINTDLLCDEPR)

##### posix special treatment
$(CINTDLLDIRL)/posix/exten.o: $(CINTDLLDIRL)/posix/exten.c
	$(MAKEDEP) -R -f$(patsubst %.o,%.d,$@) -Y -w 1000 -- $(CINTCFLAGS) -- $<
	$(CC) $(OPT) $(CINTCFLAGS) -I. $(CXXOUT)$@ -c $<

$(CINTDLLDIRL)/G__c_posix.c: $(CINTDLLDIRDLLS)/sys/types.h $(CINTDLLDIRL)/posix/exten.h $(CINTDLLDIRL)/posix/posix.h
$(CINTDLLDIRL)/G__c_ipc.c:	  $(CINTDLLDIRDLLS)/sys/types.h $(CINTDLLDIRL)/ipc/ipcif.h
$(CINTDLLDIRDLLS)/posix.dll: $(CINTDLLDIRL)/G__c_posix.o $(CINTDLLDIRL)/posix/exten.o

ifneq ($(HOST),)
$(CINTDLLDIRDLLS)/sys/types.h: $(BUILDTOOLSDIR)/$(CINTDLLDIRL)/posix/mktypes$(EXEEXT)
	(cd $(CINTDLLDIRL)/posix && \
	$(ROOT_OBJDIR)/$<)
	cp -f $(@:/sys/types.h=/systypes.h) $@
else
$(CINTDLLDIRDLLS)/sys/types.h: $(CINTDLLDIRL)/posix/mktypes$(EXEEXT)
	(cd $(dir $<) && \
	./$(notdir $<))
	cp -f $(@:/sys/types.h=/systypes.h) $@
endif

$(CINTDLLDIRL)/posix/mktypes.c: $(ORDER_) $(CINTINCLUDES)

$(CINTDLLDIRL)/posix/mktypes$(EXEEXT): $(CINTDLLDIRL)/posix/mktypes.c
	$(MAKEDEP) -R -f$(patsubst %.c,%.d,$<) -Y -w 1000 -- $(CINTCFLAGS) -- $<
	$(CC) $(OPT) $(CXXOUT)$@ $<
##### posix special treatment - END

##### ipc special treatment
$(CINTDLLDIRDLLS)/sys/ipc.dll: $(CINTDLLDIRL)/G__c_ipc.o

##### ipc special treatment - END

##### dictionaries
$(CINTDLLDIRDLLSTL)/rootcint_%.cxx: $(ROOT_SRCDIR)/core/metautils/src/%Linkdef.h $(CINTDLLROOTCINTTMPDEP)
	$(CINTDLLROOTCINTTMP) -f $@ -c $(subst multi,,${*:2=}) \
	   $(ROOT_SRCDIR)/core/metautils/src/$*Linkdef.h

$(patsubst lib/lib%Dict.$(SOEXT),$(CINTDLLDIRDLLSTL)/rootcint_%.o,$(CINTDICTDLLS)): CINTCXXFLAGS += -I.
$(patsubst lib/lib%Dict.$(SOEXT),$(CINTDLLDIRDLLSTL)/rootcint_%.cxx,$(CINTDICTDLLS)): $(CINTDLLROOTCINTTMPDEP)

lib/libvectorDict.rootmap: $(RLIBMAP) $(MAKEFILEDEP) $(ROOT_SRCDIR)/core/metautils/src/vectorLinkdef.h
	$(RLIBMAP) -o $@ -l vector.dll -d vectorbool.dll -c $(ROOT_SRCDIR)/core/metautils/src/vectorLinkdef.h

$(filter-out lib/libvectorDict.rootmap,$(CINTDICTMAPS)): lib/lib%Dict.rootmap: $(RLIBMAP) $(MAKEFILEDEP) $(ROOT_SRCDIR)/core/metautils/src/%Linkdef.h
	$(RLIBMAP) -o $@ -l $*.dll -c $(ROOT_SRCDIR)/core/metautils/src/$*Linkdef.h

$(CINTDICTDLLS): lib/lib%Dict.$(SOEXT): $(CINTDLLDIRDLLSTL)/rootcint_%.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ "$(filter-out $(MAINLIBS),$^)" "$(CINTDLLLIBLINK)"

##### dictionaries - END

##### clean

# remove only .o, .dll, .$(SOEXT)

clean-$(MODNAME):
	@(for cintdll in $(CINTDLLNAMES); do \
	  rm -f $(CINTDLLDIRDLLSTL)/rootcint_$${cintdll}.o \
	  $(CINTDLLDIRDLLSTL)/G__cpp_$${cintdll}.o \
	  $(CINTDLLDIRL)/G__c_$${cintdll}.o \
	  $(CINTDLLDIRL)/G__cpp_$${cintdll}.o \
	  core/metautils/src/stlLoader_$${cintdll}.o; done)
	@(for cintdll in $(CINTDLLNAMES); do \
	  rm -f $(CINTDLLDIRDLLSTL)/rootcint_$${cintdll}.d \
	  $(CINTDLLDIRDLLSTL)/G__cpp_$${cintdll}.d \
	  $(CINTDLLDIRL)/G__c_$${cintdll}.d \
	  $(CINTDLLDIRL)/G__cpp_$${cintdll}.d \
	  core/metautils/src/stlLoader_$${cintdll}.d; done)
	@rm -f $(CINTDLLDIRL)/posix/exten.o \
	  $(CINTDLLDIRDLLS)/posix.* \
	  $(CINTDLLDIRDLLS)/ipc.*

clean:: clean-$(MODNAME)

# remove generated code, too.
distclean-$(MODNAME): clean-$(MODNAME)
	@(for cintdll in $(CINTDLLNAMES); do \
	  rm -f $(CINTDLLDIRDLLSTL)/rootcint_$${cintdll}.* \
	  $(CINTDLLDIRDLLSTL)/G__cpp_$${cintdll}.* \
	  $(CINTDLLDIRL)/G__c_$${cintdll}.* \
	  $(CINTDLLDIRL)/G__cpp_$${cintdll}.* \
	  core/metautils/src/stlLoader_$${cintdll}.*; done)
	@rm -f $(ALLCINTDLLS) $(CINTDICTMAPS) \
	  $(CINTDLLDIRL)/posix/mktypes$(EXEEXT)
ifeq ($(PLATFORM),macosx)
	@rm -f  $(CINTDLLDIRSTL)/*.so
	@rm -rf $(CINTDLLDIRL)/posix/mktypes.dSYM
endif

distclean:: distclean-$(MODNAME)

##### clean - END
