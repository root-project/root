##############################################################
# $CINTSYSDIR/src/Makefile.base
#
# Author                 Masaharu Goto 
# Copyright(c) 1995~2003 Masaharu Goto (MXJ02154@niftyserver.jp)
#
# Description:
#   This file is the cint library makefile. 
#
##############################################################

include ../Makefile.conf

APIO        = Api Class BaseCls Type DataMbr Method MethodAr \
              CallFunc Typedf 

API2O       = Apiif Token Shadow

BCO         = bc_autoobj bc_cfunc bc_inst bc_item bc_parse \
              bc_reader bc_type bc_exec bc_vtbl bc_eh bc_debug \
              bc_assign

RFLXO       = rflx_gendict rflx_gensrc rflx_tools

STREAMO     = $(G__CFG_STREAMDIR)

STDSTRCTO   = v6_stdstrct

IPATH    = $(G__CFG_INCP)../inc

APIH        = $(addprefix(../inc/,$(addsuffix .h,$(filter-out Shadow,$(APIO))))
API2H       = $(addsuffix .h,$(filter-out Shadow,$(API2O)))
BCH         = $(addsuffix .h,$(BCO)) bc_exec_asm.h
RFLXH       = $(addsuffix .h,$(RFLXO))

# for win dllexport
G__CFG_CXXFLAGS += -DG__CINTBODY

CINTLIB  = libcint$(G__CFG_LIBEXT)
############################################################################
# common variable definition 
############################################################################

SECURITYH =     security.h
DLLREVH =       dllrev.h

HEADERS =       common.h global.h fproto.h memtest.h sunos.h ../inc/G__ci.h 

OBJECTS = 	$(addsuffix $(G__CFG_OBJEXT), \
		$(APIO) \
		$(API2O) \
		$(BCO) \
		$(STREAMO) \
		$(RFLXO) \
		g__cfunc \
		v6_auxu v6_cast v6_debug v6_decl v6_disp v6_dump \
		v6_end v6_error v6_expr v6_fread v6_func v6_gcoll \
		v6_global1 v6_global2 v6_ifunc v6_inherit v6_init \
		v6_input v6_intrpt v6_loadfile v6_macro v6_malloc \
		v6_memtest v6_new v6_newlink v6_oldlink v6_opr \
		v6_parse v6_pause v6_pcode v6_pragma v6_quote \
		v6_scrupto v6_shl v6_sizeof v6_struct v6_stub \
		v6_tmplt v6_typedef v6_val2a v6_value v6_var \
		$(STDSTRCTO) longif3 \
		$(G__CFG_PLATFORMO))

# Cint core as shared library on Linux
../libcint$(G__CFG_SOEXT) : $(OBJECTS) ../main/G__setup$(G__CFG_OBJEXT)
	$(G__CFG_LD) \
	  $(subst @so@,libcint,$(G__CFG_SOFLAGS)) \
	  $(G__CFG_SOOUT)$(notdir $@) \
	  ../main/G__setup$(G__CFG_OBJEXT) $(OBJECTS) \
	  $(G__CFG_EXP_READLINELIB) $(G__CFG_EXP_CURSESLIB) $(G__CFG_DEFAULTLIBS)
ifneq ($(G__CFG_MAKEIMPLIB),)
	$(subst @imp@,../$(notdir $(@:$(G__CFG_SOEXT)=$(G__CFG_IMPLIBEXT))),\
	  $(subst @so@,$(PWD)/$(notdir $@),$(G__CFG_MAKEIMPLIB)))
endif
	mv $(notdir $@) $@
	test -f libcint$(G__CFG_IMPLIBEXT) \
	  && cp libcint$(G__CFG_IMPLIBEXT) .. \
	  || true

# Cint core as static library
$(CINTLIB) : $(OBJECTS)
	-[ "$(G__CFG_AR)" != ""  -a -r $(CINTLIB) ] && rm $(CINTLIB)
	-[ "$(G__CFG_AR)" ] && $(G__CFG_AR) $(G__CFG_ARFLAGS) $@ $(OBJECTS);

############################################################################
# iostream library
############################################################################
# Common for many old C++ compilers
$(G__CFG_STREAMDIR)$(G__CFG_OBJEXT) : $(subst stream,libstrm,$(G__CFG_STREAMDIR)).cxx $(subst stream,libstrm,$(G__CFG_STREAMDIR)).h $(HEADERS) ../lib/$(G__CFG_STREAMDIR)/iostrm.h
	$(G__CFG_CXX) -I../lib/$(G__CFG_STREAMDIR) $(IPATH) $(G__CFG_CXXFLAGS) $(G__CFG_CXXMACROS) $(G__CFG_COMP) $(G__CFG_STREAMDIR:stream=libstrm).cxx

############################################################################
# lconv, div_t, ldiv_t, tm struct
############################################################################
v6_stdstrct$(G__CFG_OBJEXT) : v6_stdstrct.cxx stdstrct.h $(HEADERS)
	$(G__CFG_CXX) $(G__CFG_INCP)../lib/stdstrct $(IPATH) $(G__CFG_CXXFLAGS) $(G__CFG_CXXMACROS) $(G__CFG_COMP) v6_stdstrct.cxx

############################################################################
# To install cint core as shared library
############################################################################

../main/G__setup$(G__CFG_OBJEXT) : ../main/G__setup.c
	$(G__CFG_CC) $(IPATH) $(G__CFG_CFLAGS) $(G__CFG_CMACROS) $(G__CFG_COMP) $< $(G__CFG_COUT)$@

############################################################################
# Common sources
############################################################################

g__cfunc$(G__CFG_OBJEXT) : g__cfunc.c $(HEADERS) $(SECURITYH)
	$(G__CFG_CC) $(IPATH) $(G__CFG_CFLAGS) $(G__CFG_CMACROS) $(G__CFG_COMP) g__cfunc.c

v6_newlink$(G__CFG_OBJEXT) : $(DLLREVH)
v6_shl$(G__CFG_OBJEXT) : $(DLLREVH)
v6_loadfile$(G__CFG_OBJEXT) : ../inc/configcint.h

%$(G__CFG_OBJEXT) : %.cxx $(HEADERS) $(SECURITYH)
	$(G__CFG_CXX) $(IPATH) $(G__CFG_CXXFLAGS) $(G__CFG_CXXMACROS) $(G__CFG_COMP) $<

$(addsuffix $(G__CFG_OBJEXT),$(APIO)) : $(APIH)
$(addsuffix $(G__CFG_OBJEXT),$(BCO))  : $(BCH)
$(addsuffix $(G__CFG_OBJEXT),$(RFLXO)): $(RFLXH)

############################################################################
# clean
############################################################################
clean :
	$(G__CFG_RM) $(OBJECTS) G__setup$(G__CFG_OBJEXT) $(CINTLIB) \
		Api$(G__CFG_LIBEXT) Class$(G__CFG_LIBEXT)\
		../libcint$(G__CFG_SOEXT) core

