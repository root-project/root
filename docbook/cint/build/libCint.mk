############################################################################
############################################################################
# libCint sub-makefile
############################################################################
############################################################################
#
# Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
#
# For the licensing terms see the file COPYING
#
############################################################################


############################################################################
# VARIABLES
############################################################################

CINTTMP       = bin/cint_tmp$(G__CFG_EXEEXT)
CINTLIBIMPORT = lib/libCint$(G__CFG_IMPLIBEXT)

CXXAPIO    = $(addsuffix $(G__CFG_OBJEXT),$(addprefix $(G__CFG_COREVERSION)/src/,\
	      Api Class BaseCls Type DataMbr Method MethodAr \
              CallFunc Typedf Token Shadow))

CXXAPIH    = $(addsuffix .h,$(addprefix $(G__CFG_COREVERSION)/inc/,\
	      Api Class BaseCls Type DataMbr Method MethodAr \
              CallFunc Typedf Token Shadow))

ifeq ($(G__CFG_COREVERSION),cint7)
CXXAPIO   += $(addsuffix $(G__CFG_OBJEXT),$(addprefix $(G__CFG_COREVERSION)/src/,\
	      Dict))
CXXAPIH   += $(addsuffix .h,$(addprefix $(G__CFG_COREVERSION)/src/,\
              Dict))
endif

RFLXO      = $(addsuffix $(G__CFG_OBJEXT),$(addprefix $(G__CFG_COREVERSION)/src/,\
              rflx_gendict rflx_gensrc rflx_tools))

BCO        = $(addsuffix $(G__CFG_OBJEXT),$(addprefix $(G__CFG_COREVERSION)/src/,\
              bc_autoobj bc_cfunc bc_inst bc_item bc_parse \
              bc_reader bc_type bc_exec bc_vtbl bc_debug \
              bc_assign))

CONFIGO      = cint/src/config/snprintf$(G__CFG_OBJEXT) cint/src/config/strlcpy$(G__CFG_OBJEXT) cint/src/config/strlcat$(G__CFG_OBJEXT)

STUBSCXX     = $(addprefix $(G__CFG_COREVERSION)/src/,symbols.cxx)

COREO        = $(filter-out $(CXXAPIO),\
	      $(filter-out $(RFLXO),\
	      $(subst .cxx,$(G__CFG_OBJEXT),\
	      $(filter-out $(G__CFG_COREVERSION)/src/dmy%,\
	      $(filter-out $(G__CFG_COREVERSION)/src/bc_%,\
	      $(filter-out $(G__CFG_COREVERSION)/src/stdstrct.cxx,\
	      $(filter-out $(STUBSCXX), \
	      $(filter-out $(PRAGMATMPCXX),\
	      $(filter-out $(LOADFILETMPCXX),\
	      $(wildcard $(G__CFG_COREVERSION)/src/*.cxx))))))))))

STREAMO    = $(G__CFG_COREVERSION)/src/dict/$(G__CFG_STREAMDIR)$(G__CFG_OBJEXT)

STDSTRCTO  = $(G__CFG_COREVERSION)/src/dict/stdstrct$(G__CFG_OBJEXT)

LIBOBJECTS = $(CXXAPIO) $(APIDICTO) $(BCO) $(STREAMO) $(RFLXO) $(COREO) $(CONFIGO) $(G__CFG_COREVERSION)/src/g__cfunc$(G__CFG_OBJEXT) \
	     $(STDSTRCTO)
ifneq ($(G__CFG_PLATFORMO),)
LIBOBJECTS+= $(G__CFG_COREVERSION)/src/config/$(G__CFG_PLATFORMO)$(G__CFG_OBJEXT)
endif
CINTTMPOBJ = $(filter-out $(LOADFILEO) $(PRAGMAO) $(APIDICTO),$(LIBOBJECTS)) $(PRAGMATMPO) $(LOADFILETMPO)

STREAMCXX  = $(G__CFG_COREVERSION)/src/dict/$(subst stream,libstrm,$(G__CFG_STREAMDIR)).cxx

PRAGMACXX  = $(G__CFG_COREVERSION)/src/pragma.cxx
PRAGMATMPCXX= $(G__CFG_COREVERSION)/src/pragma_tmp.cxx
LOADFILECXX= $(G__CFG_COREVERSION)/src/loadfile.cxx
LOADFILETMPCXX= $(G__CFG_COREVERSION)/src/loadfile_tmp.cxx
APIDICTCXX = $(G__CFG_COREVERSION)/src/dict/Apiif.cxx
PRAGMAO    = $(PRAGMACXX:.cxx=$(G__CFG_OBJEXT))
PRAGMATMPO = $(PRAGMATMPCXX:.cxx=$(G__CFG_OBJEXT))
LOADFILEO  = $(LOADFILECXX:.cxx=$(G__CFG_OBJEXT))
LOADFILETMPO= $(LOADFILETMPCXX:.cxx=$(G__CFG_OBJEXT))
APIDICTO   = $(APIDICTCXX:.cxx=$(G__CFG_OBJEXT))

APIDICTHDRS= $(filter-out $(G__CFG_COREVERSION)/inc/Shadow.h,$(CXXAPIH))

ALLDEPO    += $(LIBOBJECTS) $(PRAGMATMPO) $(LOADFILETMPO)

############################################################################
# TARGETS
############################################################################

# Cint core as static library
static: $(CINTLIBSTATIC)

$(CINTLIBSTATIC): $(LIBOBJECTS) $(SETUPO)
	$(G__CFG_AR)$(shell $(G__CFG_MANGLEPATHS) $@) \
	  $^ $(G__CFG_READLINELIB) $(G__CFG_CURSESLIB)

# Cint core as shared library
shared: $(CINTLIBSHARED)

CINTLIBIMPORTINSODIR:=$(subst $(dir $(CINTLIBIMPORT)),$(dir $(CINTLIBSHARED)),$(CINTLIBIMPORT))
$(CINTLIBSHARED): $(LIBOBJECTS) $(SETUPO) $(REFLEXLIBDEP)
	$(G__CFG_LD) $(subst @so@,$(dir $@)/libCint,$(G__CFG_SOFLAGS)) \
	  $(G__CFG_SOOUT)$@ $(LIBOBJECTS) $(SETUPO) \
	  $(G__CFG_READLINELIB4SHLIB) $(G__CFG_CURSESLIB4SHLIB) $(G__CFG_DEFAULTLIBS) $(REFLEXLINK)
ifneq ($(G__CFG_MAKEIMPLIB),)
	$(subst @imp@,$(CINTLIBIMPORT),\
	  $(subst @so@,${PWD}/$@,$(G__CFG_MAKEIMPLIB)))
endif
ifneq ($(CINTLIBIMPORT),$(CINTLIBIMPORTINSODIR))
# Windows automatically creates the import lib next to the DLL; move it to lib/
	[ -f $(CINTLIBIMPORTINSODIR) ] \
	  && mv -f $(CINTLIBIMPORTINSODIR) $(CINTLIBIMPORT) \
	  || true
endif

############################################################################
# iostream library
############################################################################
$(STREAMO): CXXFLAGS += $(G__CFG_INCP)$(G__CFG_COREVERSION)/lib/$(G__CFG_STREAMDIR) 

############################################################################
# lconv, div_t, ldiv_t, tm struct
############################################################################
$(G__CFG_COREVERSION)/src/dict/stdstrct$(G__CFG_OBJEXT): CXXFLAGS += $(G__CFG_INCP)$(G__CFG_COREVERSION)/lib/stdstrct

##############################################################
# Generate standard header files
##############################################################
$(MAKEINCL): $(MAKEINCL:$(G__CFG_EXEEXT)=).c
ifeq ($(G__CFG_ARCH),$(subst msvc,,$(G__CFG_ARCH)))
	$(G__CFG_CC) $< $(G__CFG_COUT)$@ $(CFLAGS)
else
	cd $(dir $@) && $(G__CFG_CC) $(notdir $<) $(CFLAGS)
endif

$(G__CFG_COREVERSION)/include/stdio.h : $(MAKEINCL)
	(cd $(dir $(MAKEINCL)) && ./$(notdir $(MAKEINCL)))

##############################################################
# Generate ios enum value specific to the platform
##############################################################
$(IOSENUMH): $(ORDER_) $(CINTTMP) $(G__CFG_COREVERSION)/include/stdio.h $(MAKECINT) $(G__CFG_COREVERSION)/include/iosenum.cxx
	@(if test -r $(IOSENUMH); \
	then \
		touch $(IOSENUMH); \
	else \
		(echo Generating $(IOSENUMH). This might take a while...; \
		(set -x; cd $(G__CFG_COREVERSION)/include;$(G__CFG_RM) stdfunc$(G__CFG_SOEXT) ) ; unset VS_UNICODE_OUTPUT ; \
		cd $(G__CFG_COREVERSION) && \
		  (PATH=../lib:$${PATH} LD_LIBRARY_PATH=$${LD_LIBRARY_PATH}:../lib DYLD_LIBRARY_PATH=../lib:.:$$DYLD_LIBRARY_PATH ../$(CINTTMP) $(G__CFG_INCP)inc include/iosenum.cxx ) ); \
	fi)

############################################################################
# Cint Dictionary lib
############################################################################

$(PRAGMATMPCXX): $(PRAGMACXX)
	cp -f $< $@

$(LOADFILETMPCXX): $(LOADFILECXX)
	cp -f $< $@

$(PRAGMATMPO) $(LOADFILETMPO): CXXFLAGS += -DG__BUILDING_CINTTMP

$(APIDICTCXX): $(APIDICTHDRS) $(ORDER_) $(CINTTMP) $(IOSENUMH)
	cd $(G__CFG_COREVERSION)/src/dict && PATH=../../../lib:$$PATH LD_LIBRARY_PATH=../../../lib:$$LD_LIBRARY_PATH DYLD_LIBRARY_PATH=../lib:.:$$DYLD_LIBRARY_PATH\
	  ../../../$(CINTTMP) -n$(notdir $@) -NG__API -Z0 -D__MAKECINT__ \
	  -c-1 -I../../inc -I../../../reflex/inc -I.. Api.h


$(CINTTMP): $(SETUPO) $(MAINO) $(G__CFG_READLINELIB) $(CINTTMPOBJ) $(REFLEXLIBDEP)
	$(G__CFG_LD) $(G__CFG_LDFLAGS) $(G__CFG_LDOUT)$@ \
	  $(SETUPO) $(MAINO) $(CINTTMPOBJ) $(REFLEXLINK) \
	  $(G__CFG_READLINELIB) $(G__CFG_CURSESLIB) $(G__CFG_DEFAULTLIBS)

clean::
	-$(G__CFG_RM) $(LIBOBJECTS) $(LIBOBJECTS:$(G__CFG_OBJEXT)=.d) \
	  $(PRAGMATMPO) $(PRAGMATMPO:$(G__CFG_OBJEXT)=.d) $(PRAGMATMPCXX) \
	  $(LOADFILETMPO) $(LOADFILETMPO:$(G__CFG_OBJEXT)=.d) $(LOADFILETMPCXX) \
	  $(CINTTMP) $(CINTLIBSTATIC) $(CINTLIBSHARED) \
	  *.exp *.manifest

