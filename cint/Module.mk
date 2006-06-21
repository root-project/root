# Module.mk for cint module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := cint
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CINTDIR      := $(MODDIR)
CINTDIRS     := $(CINTDIR)/src
CINTDIRI     := $(CINTDIR)/inc
CINTDIRM     := $(CINTDIR)/main
CINTDIRT     := $(CINTDIR)/tool
CINTDIRL     := $(CINTDIR)/lib
CINTDIRDLLS  := $(CINTDIR)/include
CINTDIRSTL   := $(CINTDIR)/stl
CINTDIRDLLSTL:= $(CINTDIRL)/dll_stl

##### libCint #####
CINTH        := $(wildcard $(MODDIRI)/*.h)
CINTS1       := $(wildcard $(MODDIRS)/*.c)
CINTS2       := $(wildcard $(MODDIRS)/*.cxx)

CINTS1       += $(CINTDIRM)/G__setup.c

CINTALLO     := $(CINTS1:.c=.o) $(CINTS2:.cxx=.o)
CINTALLDEP   := $(CINTALLO:.o=.d)

CINTCONF     := include/configcint.h
CINTCONFMK   := $(MODDIR)/configcint.mk

CINTS1       := $(filter-out $(MODDIRS)/dlfcn.%,$(CINTS1))

CINTS2       := $(filter-out $(MODDIRS)/v6_sunos.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/v6_macos.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/v6_winnt.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/v6_newsos.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/v6_loadfile_tmp.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/allstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/kccstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/sunstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/sun5strm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/gcc3strm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif3.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/accstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/iccstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/fakestrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/vcstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/vc7strm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/bcstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/vcstrmold.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/alphastrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/Apiifold.%,$(CINTS2))

# strip off possible leading path from compiler command name
CXXCMD       := $(shell echo $(CXX) | sed s/".*\/"//)

ifeq ($(CXXCMD),KCC)
CINTS2       += $(MODDIRS)/kccstrm.cxx
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/longif3.cxx
else
ifeq ($(PLATFORM),linux)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),hurd)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),fbsd)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),obsd)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),hpux)
ifeq ($(ARCH),hpuxia64acc)
CINTS2       += $(MODDIRS)/accstrm.cxx
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/longif3.cxx
else
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),solaris)
ifeq ($(SUNCC5),true)
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/longif3.cxx
ifeq ($(findstring $(CXXFLAGS),-library=iostream,no%Cstd),)
CINTS2       += $(MODDIRS)/sunstrm.cxx
#CINTS2       += $(MODDIRS)/sun5strm.cxx
else
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
else
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),aix3)
CINTS1       += $(MODDIRS)/dlfcn.c
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix5)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),sgi)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),alpha)
CINTS2       += $(MODDIRS)/alphastrm.cxx
endif
ifeq ($(PLATFORM),alphagcc)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),sunos)
CINTS1       += $(MODDIRS)/sunos.c
endif
ifeq ($(PLATFORM),macos)
CINTS2       += $(MODDIRS)/v6_macos.cxx
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),macosx)
CINTS2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),lynxos)
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),win32)
CINTS2       += $(MODDIRS)/v6_winnt.cxx
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/longif3.cxx
ifeq ($(VC_MAJOR),13)
 ifeq ($(VC_MINOR),10)
  CINTS2       += $(MODDIRS)/vc7strm.cxx
 else
  CINTS2       += $(MODDIRS)/iccstrm.cxx
 endif
else
 ifeq ($(VC_MAJOR),14)
  CINTS2       += $(MODDIRS)/vc7strm.cxx
 else
  CINTS2       += $(MODDIRS)/iccstrm.cxx
 endif
endif
endif
ifeq ($(PLATFORM),vms)
CINTS2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(CXXCMD),icc)
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
ifeq ($(ICC_MAJOR),8)
ifneq ($(ICC_MINOR),0)
CINTS2       += $(MODDIRS)/gcc3strm.cxx
else
CINTS2       += $(MODDIRS)/iccstrm.cxx
endif
else
ifeq ($(ICC_MAJOR),9)
CINTS2       += $(MODDIRS)/gcc3strm.cxx
else
CINTS2       += $(MODDIRS)/iccstrm.cxx
endif
endif
CINTS2       += $(MODDIRS)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),3)
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/gcc3strm.cxx
CINTS2       += $(MODDIRS)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),4)
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/gcc3strm.cxx
CINTS2       += $(MODDIRS)/longif3.cxx
endif
ifeq ($(CXXCMD),xlC)
ifeq ($(PLATFORM),macosx)
CINTS2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINTS2))
CINTS2       := $(filter-out $(MODDIRS)/longif.%,$(CINTS2))
CINTS2       += $(MODDIRS)/gcc3strm.cxx
CINTS2       += $(MODDIRS)/longif3.cxx
endif
endif

CINTS        := $(CINTS1) $(CINTS2)
CINTO        := $(CINTS1:.c=.o) $(CINTS2:.cxx=.o)
CINTTMPO     := $(subst v6_loadfile.o,v6_loadfile_tmp.o,$(CINTO))
CINTTMPINC   := -I$(MODDIR)/include -I$(MODDIR)/stl -I$(MODDIR)/lib
CINTDEP      := $(CINTO:.o=.d)
CINTDEP      += $(MODDIRS)/v6_loadfile_tmp.d
CINTALLDEP   += $(MODDIRS)/v6_loadfile_tmp.d

CINTLIB      := $(LPATH)/libCint.$(SOEXT)

##### cint #####
CINTEXES     := $(CINTDIRM)/cppmain.cxx
CINTEXEO     := $(CINTEXES:.cxx=.o)
CINTEXEDEP   := $(CINTEXEO:.o=.d)
CINTTMP      := $(CINTDIRM)/cint_tmp$(EXEEXT)
CINT         := bin/cint$(EXEEXT)

##### makecint #####
MAKECINTS    := $(CINTDIRT)/makecint.cxx
MAKECINTO    := $(MAKECINTS:.cxx=.o)
MAKECINT     := bin/makecint$(EXEEXT)

##### iosenum.h #####
IOSENUM      := $(MODDIR)/include/iosenum.h
IOSENUMC     := $(MODDIR)/iosenum/iosenum.cxx
ifeq ($(GCC_MAJOR),4)
IOSENUMA     := $(MODDIR)/iosenum/iosenum.$(ARCH)3
else
ifeq ($(GCC_MAJOR),3)
IOSENUMA     := $(MODDIR)/iosenum/iosenum.$(ARCH)3
else
IOSENUMA     := $(MODDIR)/iosenum/iosenum.$(ARCH)
endif
endif

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CINTH)) $(CINTCONF)

# include all dependency files
INCLUDEFILES += $(CINTDEP) $(CINTEXEDEP)

CINTCXXFLAGS += -DG__HAVE_CONFIG -DG__NOMAKEINFO -DG__CINTBODY

##### local rules #####
include/%.h:    $(CINTDIRI)/%.h
		cp $< $@

$(CINTLIB):     $(CINTO)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libCint.$(SOEXT) $@ "$^" "$(CINTLIBEXTRA)"

$(CINT):        $(CINTEXEO) $(CINTLIB)
		$(LD) $(LDFLAGS) -o $@ $(CINTEXEO) \
		   $(RPATH) $(CINTLIBS) $(CILIBS)

$(CINTTMP):     $(CINTEXEO) $(CINTTMPO)
		$(LD) $(LDFLAGS) -o $@ $(CINTEXEO) \
		   $(CINTTMPO) $(CILIBS)

$(MAKECINT):    $(MAKECINTO)
		$(LD) $(LDFLAGS) -o $@ $(MAKECINTO)

$(IOSENUM):     $(IOSENUMA)
		cp $< $@

$(IOSENUMA):    $(CINTTMP)
		@(if [ ! -r $@ ]; then \
			echo "Making $@..."; \
			$(CINTTMP) $(CINTTMPINC) $(IOSENUMC) > /dev/null; \
			mv iosenum.h $@; \
		else \
			touch $@; \
		fi)

all-cint:       $(CINTLIB) $(CINT) $(CINTTMP) $(MAKECINT) $(IOSENUM)

clean-cint:
		@rm -f $(CINTTMPO) $(CINTALLO) $(CINTEXEO) $(MAKECINTO)

clean::         clean-cint

distclean-cint: clean-cint
		@rm -f $(CINTALLDEP) $(CINTLIB) $(IOSENUM) $(CINTEXEDEP) \
		   $(CINT) $(CINTTMP) $(MAKECINT) $(CINTDIRM)/*.exp \
		   $(CINTDIRM)/*.lib $(CINTDIRS)/v6_loadfile_tmp.cxx

distclean::     distclean-cint

##### extra rules ######
$(CINTDIRS)/libstrm.o:  CINTCXXFLAGS += -I$(CINTDIRL)/stream
$(CINTDIRS)/sunstrm.o:  CINTCXXFLAGS += -I$(CINTDIRL)/snstream
$(CINTDIRS)/sun5strm.o: CINTCXXFLAGS += -I$(CINTDIRL)/snstream
$(CINTDIRS)/vcstrm.o:   CINTCXXFLAGS += -I$(CINTDIRL)/vcstream
$(CINTDIRS)/%strm.o:    CINTCXXFLAGS += -I$(CINTDIRL)/$(notdir $(basename $@))

$(MAKECINTO) $(CINTALLO): $(CINTCONF)

$(CINTDIRS)/v6_stdstrct.o:     CINTCXXFLAGS += -I$(CINTDIRL)/stdstrct
$(CINTDIRS)/v6_loadfile_tmp.o: CINTCXXFLAGS += -UHAVE_CONFIG -DROOTBUILD

$(CINTDIRS)/v6_loadfile_tmp.cxx: $(CINTDIRS)/v6_loadfile.cxx
	cp -f $< $@

##### cintdlls ######
# no: string valarray, limits iterator pair vectorbool
CINTDLLNAMES = vector list deque map map2 set multimap multimap2 multiset \
           stack queue exception complex stdfunc stdcxxfunc
ifneq ($(PLATFORM),win32)
# FIX THEM!
  CINTDLLNAMES += posix ipc
endif
# ".dll", not ".$(SOEXT)"!
CINTDLLS = $(subst $(CINTDIRSTL)/ipc.dll,$(CINTDIRDLLS)/sys/ipc.$(SOEXT),\
  $(subst $(CINTDIRSTL)/posix.dll,$(CINTDIRDLLS)/posix.$(SOEXT),\
  $(addprefix $(CINTDIRSTL)/,$(addsuffix .dll,$(CINTDLLNAMES)))))

# these need dictionaries
ifneq ($(findstring vector,$(CINTDLLS)),)
CINTDICTDLLS += lib/libvectorDict.$(SOEXT)
$(CINTDIRSTL)/vector.dll: metautils/src/stlLoader_vector.o
endif
ifneq ($(findstring list,$(CINTDLLS)),)
CINTDICTDLLS += lib/liblistDict.$(SOEXT)
$(CINTDIRSTL)/list.dll: metautils/src/stlLoader_list.o
endif
ifneq ($(findstring deque,$(CINTDLLS)),)
CINTDICTDLLS += lib/libdequeDict.$(SOEXT)
$(CINTDIRSTL)/deque.dll: metautils/src/stlLoader_deque.o
endif
ifneq ($(findstring map,$(CINTDLLS)),)
CINTDICTDLLS += lib/libmapDict.$(SOEXT)
CINTDICTDLLS += lib/libmap2Dict.$(SOEXT)
$(CINTDIRSTL)/map.dll: metautils/src/stlLoader_map.o
$(CINTDIRSTL)/map.dll: metautils/src/stlLoader_map2.o
endif
ifneq ($(findstring set,$(CINTDLLS)),)
CINTDICTDLLS += lib/libsetDict.$(SOEXT)
$(CINTDIRSTL)/set.dll: metautils/src/stlLoader_set.o
endif
ifneq ($(findstring multimap,$(CINTDLLS)),)
CINTDICTDLLS += lib/libmultimapDict.$(SOEXT)
CINTDICTDLLS += lib/libmultimap2Dict.$(SOEXT)
$(CINTDIRSTL)/multimap.dll: metautils/src/stlLoader_multimap.o
$(CINTDIRSTL)/multimap.dll: metautils/src/stlLoader_multimap2.o
endif
ifneq ($(findstring multiset,$(CINTDLLS)),)
CINTDICTDLLS += lib/libmultisetDict.$(SOEXT)
$(CINTDIRSTL)/multiset.dll: metautils/src/stlLoader_multiset.o
endif
ifneq ($(findstring valarray,$(CINTDLLS)),)
CINTDICTDLLS += lib/libvalarrayDict.$(SOEXT)
$(CINTDIRSTL)/valarray.dll: metautils/src/stlLoader_valarray.o
endif


ALLCINTDLLS = $(CINTDLLS) $(CINTDICTDLLS)

cintdlls: $(ALLCINTDLLS)

$(CINTDIRDLLSTL)/G__cpp_vector.cxx:	$(CINTDIRL)/dll_stl/vec.h
$(CINTDIRDLLSTL)/G__cpp_list.cxx:	$(CINTDIRL)/dll_stl/lst.h
$(CINTDIRDLLSTL)/G__cpp_deque.cxx:	$(CINTDIRL)/dll_stl/dqu.h
$(CINTDIRDLLSTL)/G__cpp_map.cxx:	$(CINTDIRL)/dll_stl/mp.h
$(CINTDIRDLLSTL)/G__cpp_map2.cxx:	$(CINTDIRL)/dll_stl/mp.h
$(CINTDIRDLLSTL)/G__cpp_set.cxx:	$(CINTDIRL)/dll_stl/st.h
$(CINTDIRDLLSTL)/G__cpp_multimap.cxx:	$(CINTDIRL)/dll_stl/multmp.h
$(CINTDIRDLLSTL)/G__cpp_multimap2.cxx:	$(CINTDIRL)/dll_stl/multmp.h
$(CINTDIRDLLSTL)/G__cpp_multiset.cxx:	$(CINTDIRL)/dll_stl/multst.h
$(CINTDIRDLLSTL)/G__cpp_stack.cxx:	$(CINTDIRL)/dll_stl/stk.h
$(CINTDIRDLLSTL)/G__cpp_queue.cxx:	$(CINTDIRL)/dll_stl/que.h
$(CINTDIRDLLSTL)/G__cpp_exception.cxx:	$(CINTDIRL)/dll_stl/cinteh.h
$(CINTDIRDLLSTL)/G__cpp_complex.cxx:	$(CINTDIRL)/dll_stl/cmplx.h
$(CINTDIRDLLSTL)/G__cpp_limits.cxx:	$(CINTDIRL)/dll_stl/clim.h
$(CINTDIRDLLSTL)/G__cpp_iterator.cxx:	$(CINTDIRL)/dll_stl/iter.h
$(CINTDIRDLLSTL)/G__cpp_pair.cxx:	$(CINTDIRL)/dll_stl/pr.h
$(CINTDIRDLLSTL)/G__cpp_string.cxx:	$(CINTDIRL)/dll_stl/str.h
$(CINTDIRDLLSTL)/G__cpp_valarray.cxx:	$(CINTDIRL)/dll_stl/vary.h
$(CINTDIRDLLSTL)/G__cpp_vectorbool.cxx: $(CINTDIRL)/dll_stl/vecbool.h
$(CINTDIRDLLSTL)/G__cpp_stdcxxfunc.cxx: $(CINTDIRL)/stdstrct/stdcxxfunc.h
#$(CINTDIRDLLSTL)/G__c_stdfunc.cxx:	$(CINTDIRL)/stdstrct/stdfunc.h
$(CINTDIRDLLSTL)/G__c_posix.c:		$(CINTDIRL)/posix/exten.h
$(CINTDIRDLLSTL)/G__c_ipc.c:		$(CINTDIRL)/ipc/ipcif.h


FAVOR_SYSINC := -I-
ifeq ($(PLATFORM),sgi)
   FAVOR_SYSINC=
endif
CINTDLLINCDIRS := -I. -I$(CINTDIRDLLSTL) $(FAVOR_SYSINC)
ifeq ($(GCC_MAJOR),4)
  CINTDLLINCDIRS := -iquote. -iquote$(CINTDIRDLLSTL)
endif
$(CINTDLLS): CINTCXXFLAGS += $(CINTDLLINCDIRS)

##### all cintdlls end on .dll
ifneq ($(SOEXT),dll)
CINTDLLSOEXTCMD = mv $(@:.dll=.$(SOEXT)) $@
ifeq ($(PLATFORM),maxosx)
ifeq ($(subst $(MACOSX_MINOR),,456789),456789)
# MACOSX_MINOR < 4
  CINTDLLSOEXTCMD += ;mv $(@:.dll=.so) $@;rm -f $(@:.dll=.so)
endif
endif # macosx
endif # need to mv to .dll
##### all cintdlls end on .dll - END

# Filter out the explicit link flag
ifneq ($(subst build/unix/makelib.sh,,$(MAKELIB)),$(MAKELIB))
  $(CINTDLLS): MAKELIB := $(subst -x,,$(MAKELIB))
endif

$(CINTDIRSTL)/%.dll: $(CINTDIRDLLSTL)/G__cpp_%.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $(@:.dll=.$(SOEXT)) $^
	$(CINTDLLSOEXTCMD)

.PRECIOUS: $(addprefix metautils/src/stlLoader_,$(addsuffix .o,$(CINTDLLNAMES)))\
	   $(addprefix $(CINTDIRDLLSTL)/G__cpp_,$(addsuffix .o,$(CINTDLLNAMES)))

metautils/src/stlLoader_%.o: metautils/src/stlLoader.cc
	$(CXX) $(OPT) $(CINTCXXFLAGS) $(INCDIRS) -DWHAT=\"$*\" $(CXXOUT)$@ -c $<

#$(CINTDIRDLLSTL)/G__cpp_%.o: $(CINTDIRDLLSTL)/G__cpp_%.cxx
#	$(CXX) $(OPT) $(CINTCXXFLAGS) $(CXXOUT)$@ -c $<

$(CINTDIRDLLSTL)/G__cpp_%.cxx: $(ORDER_) $(CINTTMP)
	$(CINTTMP) -w1 -z$(notdir $*) -n$@ $(subst $*,,$(patsubst %map2,-DG__MAP2,$*)) \
            -D__MAKECINT__ -DG__MAKECINT -I$(CINTDIRDLLSTL) -I$(CINTDIRL) \
            -c-1 -A -Z0 $(filter %.h,$^)

$(CINTDIRL)/G__c_%.c: $(ORDER_) $(CINTTMP)
	$(CINTTMP) -K -w1 -z$(notdir $*) -n$@ -D__MAKECINT__ -DG__MAKECINT \
        -c-2 -Z0 $(filter %.h,$^) 

$(CINTDIRL)/G__c_%.o: CINTCFLAGS += -I. -DG__SYSTYPES_H
#$(CINTDIRL)/G__c_%.o: $(CINTDIRL)/G__c_%.c
#	$(CC) $(OPT) $(CINTCFLAGS) -I. $(CXXOUT)$@ -c $<

##### posix special treatment
$(CINTDIRL)/posix/exten.o: $(CINTDIRL)/posix/exten.c
	$(CC) $(OPT) $(CINTCFLAGS) -I. $(CXXOUT)$@ -c $<

$(CINTDIRL)/G__c_posix.c: $(CINTDIRDLLS)/sys/types.h cint/lib/posix/exten.h

$(CINTDIRDLLS)/posix.$(SOEXT): $(CINTDIRL)/G__c_posix.o metautils/src/stlLoader_posix.o \
	                      $(CINTDIRL)/posix/exten.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ $^

$(CINTDIRDLLS)/sys/types.h: $(CINTDIRL)/posix/mktypes$(EXEEXT)
	(cd $(dir $<) && \
	./$(notdir $<))
	cp -f $(CINTDIRDLLS)/systypes.h $@

$(CINTDIRL)/posix/mktypes$(EXEEXT): $(CINTDIRL)/posix/mktypes.c
	$(CC) $(OPT) $(CXXOUT)$@ $<
##### posix special treatment - END

##### ipc special treatment
$(CINTDIRDLLS)/sys/ipc.$(SOEXT): $(CINTDIRL)/G__c_ipc.o metautils/src/stlLoader_ipc.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ $^
##### ipc special treatment - END

##### dictionaries
.PRECIOUS: $(CINTDIRDLLSTL)/rootcint_%.cxx
$(CINTDIRDLLSTL)/rootcint_%.cxx: $(ORDER_) $(ROOTCINTTMPEXE)
	$(ROOTCINTTMP) -f $@ -c \
	$(subst $*,,$(patsubst %map2,-DG__MAP2,$*)) $(subst multi,,${*:2=}) \
	metautils/src/${*:2=}Linkdef.h

$(patsubst lib/lib%Dict.$(SOEXT),$(CINTDIRDLLSTL)/rootcint_%.o,$(CINTDICTDLLS)): CINTCXXFLAGS += -I.

$(CINTDICTDLLS): lib/lib%Dict.$(SOEXT): $(CINTDIRDLLSTL)/rootcint_%.o
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ $^
ifeq ($(PLATFORM),maxosx)
ifeq ($(subst $(MACOSX_MINOR),,456789),456789)
	rm -f $@
else
	mv -f $@ $(basename $@).so
endif
endif
##### dictionaries - END

##### configcint.h
ifeq ($(CPPPREP),)
# cannot use "CPPPREP?=", as someone might set "CPPPREP="
  CPPPREP = $(CXX) -E -C
endif
include $(CINTCONFMK)
##### configcint.h - END

##### clean
.PHONY: distclean-cintdlls clean-cintdlls
clean:: clean-cintdlls
distclean:: distclean-cintdlls

# remove only .o, .dll, .$(SOEXT)
clean-cintdlls:
	for cintdll in $(CINTDLLNAMES); do \
	  rm -f $(CINTDIRDLLSTL)/rootcint_$${cintdll}.o \
	  $(CINTDIRDLLSTL)/G__cpp_$${cintdll}.o \
	  metautils/src/stlLoader_$${cintdll}.o; done
	rm -f $(ALLCINTDLLS) \
	  $(CINTDIRL)/posix/exten.o $(CINTDIRSTL)/posix.dll \
	  $(CINTDIRL)/posix/G__c_posix.o

# remove generated code, too.
distclean-cintdlls: clean-cintdlls
	for cintdll in $(CINTDLLNAMES); do \
	  rm -f $(CINTDIRDLLSTL)/rootcint_$${cintdll}.* \
	  $(CINTDIRDLLSTL)/G__cpp_$${cintdll}.* \
	  metautils/src/stlLoader_$${cintdll}.*; done
	rm -f $(ALLCINTDLLS) \
	  $(CINTDIRL)/posix/G__c_posix.* $(CINTDIRL)/posix/mktypes$(EXEEXT)
##### clean - END
