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

REFLEXIPATH  = $(G__CFG_INCP)$(G__CFG_REFLEXINCDIR)
REFLEXSRCDIR = $(G__CFG_REFLEXINCDIR:/inc=/src)
REFLEXLIB_OBJ= $(subst .cxx,$(G__CFG_OBJEXT),$(wildcard $(REFLEXSRCDIR)/*.cxx))
REFLEXLIB    = libReflex_static$(G__CFG_LIBEXT)
REFLEXSO     = libReflex$(G__CFG_SOEXT)
REFLEXIMPLIB = libReflex$(G__CFG_IMPLIBEXT)

ifeq ($(LINKSTATIC),yes)
  REFLEXLINK   = $(G__CFG_LIBP). $(subst @imp@,Reflex_static,$(G__CFG_LIBL))
  REFLEXLIBDEP = $(REFLEXLIB)
else
  REFLEXLINK   = $(G__CFG_LIBP). $(subst @imp@,Reflex,$(G__CFG_LIBL))
  REFLEXLIBDEP = $(REFLEXSO)
endif


##############################################################
# TARGETS
##############################################################

ifeq ($(G__CFG_EXTRACTSYMBOLS),)
REFLEXLIB_DEF=$(REFLEXSO:$(G__CFG_SOEXT)=.def)
$(REFLEXLIB_DEF): $(REFLEXLIB_OBJ)
	@[ ! -d `dirname $@` ] && mkdir -p `dirname $@` || true
	@echo 'LIBRARY  "LIBREFLEX"' > $@
	@echo 'VERSION  1.0' >> $@
	@echo 'HEAPSIZE 1048576,4096' >> $@
	@echo 'EXPORTS' >> $@
	$(subst @obj@,$(REFLEXLIB_OBJ),$(G__CFG_EXTRACTSYMBOLS))>>$@
endif

ALLDEPO += $(REFLEXLIB_OBJ)

reflex: $(REFLEXLIB)
$(REFLEXLIB): $(REFLEXLIB_OBJ)
	echo $^
	@[ ! -d `dirname $(REFLEXLIB)` ] && mkdir -p `dirname $(REFLEXLIB)` || true
	$(G__CFG_AR)$(shell $(G__CFG_MANGLEPATHS) $(REFLEXLIB)) \
	    $(shell $(G__CFG_MANGLEPATHS) $(REFLEXLIB_OBJ))

$(REFLEXSO): $(REFLEXLIB_OBJ) $(REFLEXLIB_DEF)
	@[ ! -d `dirname $(REFLEXLIB)` ] && mkdir -p `dirname $(REFLEXLIB)` || true
	$(G__CFG_LD) $(subst @so@,$(shell $(G__CFG_MANGLEPATHS) $(@:$(G__CFG_SOEXT)=)),$(G__CFG_SOFLAGS)) \
	    $(G__CFG_LDOUT)$@ $(REFLEXLIB_OBJ)
ifneq ($(G__CFG_MAKEIMPLIB),)
	$(subst @imp@,$(@:$(G__CFG_SOEXT)=$(G__CFG_IMPLIBEXT)),\
	  $(subst @so@,$@,$(G__CFG_MAKEIMPLIB)))
endif

$(REFLEXSRCDIR)/%$(G__CFG_OBJEXT): $(REFLEXSRCDIR)/%.cxx
	$(RMKDEPEND) -R -f$(REFLEXSRCDIR)/$*.d -Y -w 1000 -- $(CXXFLAGS) -DREFLEX_DLL -D__cplusplus -- $<
	$(G__CFG_CXX) $(CXXFLAGS) -DREFLEX_DLL \
          $(G__CFG_COMP) $< $(G__CFG_COUT)$@

clean:: clean-reflex

clean-reflex:
	-$(G__CFG_RM) $(REFLEXSO) $(REFLEXSO:$(G__CFG_SOEXT)=$(G__CFG_IMPLIBEXT)) \
	  $(REFLEXLIB) $(REFLEXLIB_OBJ) $(REFLEXLIB_OBJ:$(G__CFG_OBJEXT)=.d) $(REFLEXLIB_DEF) 
