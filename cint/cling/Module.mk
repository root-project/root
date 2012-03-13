# Module.mk for cling module
# Copyright (c) 2011 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2011-10-18

MODNAME      := cling
MODDIR       := $(ROOT_SRCDIR)/cint/$(MODNAME)

CLINGDIR     := $(MODDIR)

##### libCling #####
CLINGS       := $(wildcard $(MODDIR)/lib/Interpreter/*.cpp) \
                $(wildcard $(MODDIR)/lib/MetaProcessor/*.cpp)
CLINGO       := $(call stripsrc,$(CLINGS:.cpp=.o))

CLINGDEP     := $(CLINGO:.o=.d)

CLINGLIB     := $(LPATH)/libCling.$(SOEXT)

CLINGETC     := $(addprefix etc/cling/Interpreter/,RuntimeUniverse.h ValuePrinter.h ValuePrinterInfo.h)

# used in the main Makefile
ALLLIBS      += $(CLINGLIB)
ALLHDRS      += $(CLINGETC)

# include all dependency files
INCLUDEFILES += $(CLINGDEP)

ifneq ($(LLVMCONFIG),)
# include dir for picking up RuntimeUniverse.h etc - need to
# 1) copy relevant headers to include/
# 2) rely on TCling to addIncludePath instead of using CLING_..._INCL below
CLINGCXXFLAGS := $(shell $(LLVMCONFIG) --cxxflags) -I$(MODDIR)/include \
	'-DR__LLVMDIR="$(shell cd $(shell $(LLVMCONFIG) --libdir)/..; pwd)"'
CLINGLLVMLIBS:= -L$(shell $(LLVMCONFIG) --libdir) \
	$(addprefix -lclang,\
		Frontend Serialization Driver CodeGen Parse Sema Analysis Rewrite AST Lex Basic Edit) \
	$(patsubst -lLLVM%Disassembler,,\
	$(patsubst -lLLVM%AsmParser,,\
	$(filter-out -lLLVMipa,\
	$(shell $(LLVMCONFIG) --libs linker jit executionengine debuginfo \
	  archive bitreader all-targets codegen selectiondag asmprinter \
	  mcparser scalaropts instcombine transformutils analysis target)))) \
	$(shell $(LLVMCONFIG) --ldflags)
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) 

$(CLINGLIB):   $(CLINGO)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"      \
		"$(SOFLAGS)" libCling.$(SOEXT) $@ "$(CLINGO)" \
		"$(CLINGLLVMLIBS) $(CLINGLIBEXTRA)"

all-$(MODNAME): $(CLINGLIB)

clean-$(MODNAME):
		@rm -f $(CLINGO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CLINGDEP) $(CLINGLIB) $(CLINGETC)

distclean::     distclean-$(MODNAME)

etc/cling/%.h: $(MODDIR)/include/cling/%.h
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	@cp $< $@
etc/cling/%.h: $(call stripsrc,$(MODDIR)/%.o)/include/cling/%.h
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	@cp $< $@

$(MODDIR)/%.o: $(MODDIR)/%.cpp
	$(MAKEDEP) -R -f$(@:.o=.d) -Y -w 1000 -- $(CXXFLAGS) $(CLINGCXXFLAGS) -D__cplusplus -- $<
	$(CXX) $(OPT) $(CLINGCXXFLAGS) $(CXXOUT)$@ -c $<

$(call stripsrc,$(MODDIR)/%.o): $(MODDIR)/%.cpp
	+@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(MAKEDEP) -R -f$(@:.o=.d) -Y -w 1000 -- $(CXXFLAGS) $(CLINGCXXFLAGS)  -D__cplusplus -- $<
	$(CXX) $(OPT) $(CLINGCXXFLAGS) $(CXXOUT)$@ -c $<

##### extra rules ######
