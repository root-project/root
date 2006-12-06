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

CINTTMP       = cint_tmp$(G__CFG_EXEEXT)
CINTLIBIMPORT = libcint$(G__CFG_IMPLIBEXT)

CXXAPIO    = $(addsuffix $(G__CFG_OBJEXT),$(addprefix src/,\
	      Api Dict Class BaseCls Type DataMbr Method MethodAr \
              CallFunc Typedf Token Shadow))

CXXAPIH    = $(addsuffix .h,$(addprefix inc/,\
	      Api Class BaseCls Type DataMbr Method MethodAr \
              CallFunc Typedf Token Shadow)) \
             $(addsuffix .h,$(addprefix src/,\
              Dict))

BCO        = $(addsuffix $(G__CFG_OBJEXT),$(addprefix src/,\
              bc_autoobj bc_cfunc bc_inst bc_item bc_parse \
              bc_reader bc_type bc_exec bc_vtbl bc_eh bc_debug \
              bc_assign))
V6O        = $(subst .cxx,$(G__CFG_OBJEXT),\
	      $(filter-out src/v6_dmy%,\
	      $(filter-out src/v6_stdstrct.cxx,\
	      $(filter-out src/v6_macos.cxx,\
	      $(filter-out src/v6_winnt.cxx,\
	      $(filter-out $(PRAGMATMPCXX),\
	      $(filter-out $(LOADFILETMPCXX),\
	      $(wildcard src/v6_*.cxx))))))))
RFLXO      = $(addsuffix $(G__CFG_OBJEXT),$(addprefix src/,\
              rflx_gendict rflx_gensrc rflx_tools))
STREAMO    = src/$(G__CFG_STREAMDIR)$(G__CFG_OBJEXT)
STDSTRCTO  = src/v6_stdstrct$(G__CFG_OBJEXT)

LIBOBJECTS = $(CXXAPIO) $(APIDICTO) $(BCO) $(STREAMO) $(RFLXO) $(V6O) src/g__cfunc$(G__CFG_OBJEXT) \
	     $(STDSTRCTO) src/longif3$(G__CFG_OBJEXT)
ifneq ($(G__CFG_PLATFORMO),)
LIBOBJECTS+= src/$(G__CFG_PLATFORMO)$(G__CFG_OBJEXT)
endif
CINTTMPOBJ = $(filter-out $(LOADFILEO) $(PRAGMAO) $(APIDICTO),$(LIBOBJECTS)) $(PRAGMATMPO) $(LOADFILETMPO)

STREAMCXX  = src/$(subst stream,libstrm,$(G__CFG_STREAMDIR)).cxx

PRAGMACXX  = src/v6_pragma.cxx
PRAGMATMPCXX= src/v6_pragma_tmp.cxx
LOADFILECXX= src/v6_loadfile.cxx
LOADFILETMPCXX= src/v6_loadfile_tmp.cxx
APIDICTCXX = src/Apiif.cxx
PRAGMAO    = $(PRAGMACXX:.cxx=$(G__CFG_OBJEXT))
PRAGMATMPO = $(PRAGMATMPCXX:.cxx=$(G__CFG_OBJEXT))
LOADFILEO  = $(LOADFILECXX:.cxx=$(G__CFG_OBJEXT))
LOADFILETMPO= $(LOADFILETMPCXX:.cxx=$(G__CFG_OBJEXT))
APIDICTO   = $(APIDICTCXX:.cxx=$(G__CFG_OBJEXT))

APIDICTHDRS= $(filter-out inc/Shadow.h,$(CXXAPIH))

ALLDEPO    += $(LIBOBJECTS) $(PRAGMATMPO) $(LOADFILETMPO)

############################################################################
# TARGETS
############################################################################

# Cint core as static library
$(CINTLIBSTATIC): $(LIBOBJECTS) $(SETUPO)
	$(G__CFG_AR)$(shell $(G__CFG_MANGLEPATHS) $@) \
	  $^ $(G__CFG_READLINELIB) $(G__CFG_CURSESLIB)

# Cint core as shared library
$(CINTLIBSHARED): $(LIBOBJECTS) $(SETUPO) $(REFLEXLIBDEP)
	$(G__CFG_LD) $(subst @so@,libcint,$(G__CFG_SOFLAGS)) \
	  $(G__CFG_SOOUT)$@ $(LIBOBJECTS) $(SETUPO) \
	  $(G__CFG_EXP_READLINELIB) $(G__CFG_EXP_CURSESLIB) $(G__CFG_DEFAULTLIBS) $(REFLEXLINK)
ifneq ($(G__CFG_MAKEIMPLIB),)
	$(subst @imp@,$(@:$(G__CFG_SOEXT)=$(G__CFG_IMPLIBEXT)),\
	  $(subst @so@,$@,$(G__CFG_MAKEIMPLIB)))
endif

############################################################################
# iostream library
############################################################################
$(STREAMO): CXXFLAGS += $(G__CFG_INCP)lib/$(G__CFG_STREAMDIR) 

############################################################################
# lconv, div_t, ldiv_t, tm struct
############################################################################
src/v6_stdstrct$(G__CFG_OBJEXT): CXXFLAGS += $(G__CFG_INCP)lib/stdstrct

##############################################################
# Generate standard header files
##############################################################
include/stdio.h : $(MAKEINCL)
	(cd $(dir $(MAKEINCL)) && ./$(notdir $(MAKEINCL)))

##############################################################
# Generate ios enum value specific to the platform
##############################################################
$(IOSENUMH): $(ORDER_) $(CINTTMP) include/stdio.h $(MAKECINT) include/iosenum.cxx
	@(if test -r $(IOSENUMH); \
	then \
		touch $(IOSENUMH); \
	else \
		(echo Generating $(IOSENUMH). This might take a while...; \
		cd include;$(G__CFG_RM) stdfunc$(G__CFG_SOEXT); cd ..\
		unset VS_UNICODE_OUTPUT; \
		LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:. ./$(CINTTMP) $(G__CFG_INCP)inc include/iosenum.cxx); \
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
	cd src && LD_LIBRARY_PATH=..:$$LD_LIBRARY_PATH \
	  ../$(CINTTMP) -n$(notdir $@) -NG__API -Z0 -D__MAKECINT__ \
	  -c-1 -I$(shell $(G__CFG_MANGLEPATHS) ../inc) \
	  -I$(shell $(G__CFG_MANGLEPATHS) $(G__CFG_REFLEXINCDIR)) Api.h


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

