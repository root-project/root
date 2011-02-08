# Module.mk for the bench module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: 

MODNAME      := proofbench
MODDIR       := proof/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

BENCHDIR    := $(MODDIR)
BENCHDIRS   := $(BENCHDIR)/src
BENCHDIRI   := $(BENCHDIR)/inc

##### libProofBench #####
BENCHL      := $(MODDIRI)/LinkDef.h
BENCHDS     := $(MODDIRS)/G__Bench.cxx
BENCHDO     := $(BENCHDS:.cxx=.o)
BENCHDH     := $(BENCHDS:.cxx=.h)

BENCHH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
BENCHH      := $(filter-out $(MODDIRI)/TSel%,$(BENCHH))

BENCHS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
BENCHS      := $(filter-out $(MODDIRS)/TSel%,$(BENCHS))

BENCHO      := $(BENCHS:.cxx=.o)
BENCHDEP    := $(BENCHO:.o=.d) $(BENCHDO:.o=.d)
BENCHLIB    := $(LPATH)/libProofBench.$(SOEXT)
BENCHMAP    := $(BENCHLIB:.$(SOEXT)=.rootmap)

##### ProofBenchDataSel PAR file #####
PBDPARDIR   := $(BENCHDIRS)/ProofBenchDataSel
PBDPARINF   := $(PBDPARDIR)/PROOF-INF
PBDPARH     := test/Event.h $(MODDIRI)/TProofBenchTypes.h
PBDPARS     := test/Event.cxx
PBDPARH     += $(wildcard $(MODDIRI)/TSel*.h)
PBDPARS     += $(wildcard $(MODDIRS)/TSel*.cxx)
PBDPARH     := $(filter-out $(MODDIRI)/TSelHist%, $(PBDPARH))
PBDPARS     := $(filter-out $(MODDIRS)/TSelHist%, $(PBDPARS))

PBDPAR      := $(MODDIRS)/ProofBenchDataSel.par

##### ProofBenchCPUSel PAR file #####
PBCPARDIR   := $(BENCHDIRS)/ProofBenchCPUSel
PBCPARINF   := $(PBCPARDIR)/PROOF-INF
PBCPARH     := $(MODDIRI)/TProofBenchTypes.h $(MODDIRI)/TSelHist.h
PBCPARS     := $(MODDIRS)/TSelHist.cxx

PBCPAR      := $(MODDIRS)/ProofBenchCPUSel.par

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(BENCHH))
ALLLIBS      += $(BENCHLIB) $(PBDPAR) $(PBCPAR)
ALLMAPS      += $(BENCHMAP)

# include all dependency files
INCLUDEFILES += $(BENCHDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(BENCHDIRI)/%.h
		cp $< $@

$(BENCHLIB):   $(BENCHO) $(BENCHDO) $(ORDER_) $(MAINLIBS) \
                $(BENCHLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProofBench.$(SOEXT) $@ \
		   "$(BENCHO) $(BENCHDO)"

$(BENCHDS):    $(BENCHH) $(BENCHL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(BENCHH) $(BENCHL)

$(BENCHMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(BENCHL)
		$(RLIBMAP) -o $(BENCHMAP) -l $(BENCHLIB) \
		   -d $(PROOFBENCHLIBDEPM) -c $(BENCHL)

$(PBDPAR):   $(PBDPARH) $(PBDPARS)
		@echo "Generating PAR file $@..."
		@(if test -d $(PBDPARDIR); then\
		   rm -fr $(PBDPARDIR); \
		fi;\
		mkdir -p $(PBDPARINF); \
		for f in $(PBDPARH) $(PBDPARS); do \
		   cp -rp $$f $(PBDPARDIR); \
		done; \
		echo "#include \"TClass.h\"" > $(PBDPARINF)/SETUP.C ; \
		echo "#include \"TROOT.h\"" > $(PBDPARINF)/SETUP.C ; \
		echo "Int_t SETUP() {" >> $(PBDPARINF)/SETUP.C ; \
		echo "   if (!TClass::GetClass(\"TPBReadType\")) {" >> $(PBDPARINF)/SETUP.C ; \
		echo "      gROOT->ProcessLine(\".L TProofBenchTypes.h+\");" >> $(PBDPARINF)/SETUP.C ; \
		echo "   }" >> $(PBDPARINF)/SETUP.C ; \
		for f in $(PBDPARS); do \
		   b=`basename $$f`; \
		   echo "   gROOT->ProcessLine(\".L $$b+\");" >> $(PBDPARINF)/SETUP.C ; \
		done; \
		echo "   return 0;" >> $(PBDPARINF)/SETUP.C ; \
		echo "}" >> $(PBDPARINF)/SETUP.C ; \
		builddir=$(PWD); \
		cd $(BENCHDIRS); \
		par=`basename $(PBDPAR)`;\
		pardir=`basename $(PBDPARDIR)`;\
		tar czvf $$par $$pardir; \
		cd $$builddir; \
		rm -fr $(PBDPARDIR))

$(PBCPAR):   $(PBCPARH) $(PBCPARS)
		@echo "Generating PAR file $@..."
		@(if test -d $(PBCPARDIR); then\
		   rm -fr $(PBCPARDIR); \
		fi;\
		mkdir -p $(PBCPARINF); \
		for f in $(PBCPARH) $(PBCPARS); do \
		   cp -rp $$f $(PBCPARDIR); \
		done; \
		echo "#include \"TClass.h\"" > $(PBCPARINF)/SETUP.C ; \
		echo "#include \"TROOT.h\"" > $(PBCPARINF)/SETUP.C ; \
		echo "Int_t SETUP() {" >> $(PBCPARINF)/SETUP.C ; \
		echo "   if (!TClass::GetClass(\"TPBReadType\")) {" >> $(PBCPARINF)/SETUP.C ; \
		echo "      gROOT->ProcessLine(\".L TProofBenchTypes.h+\");" >> $(PBCPARINF)/SETUP.C ; \
		echo "   }" >> $(PBCPARINF)/SETUP.C ; \
		for f in $(PBCPARS); do \
		   b=`basename $$f`; \
		   echo "   gROOT->ProcessLine(\".L $$b+\");" >> $(PBCPARINF)/SETUP.C ; \
		done; \
		echo "   return 0;" >> $(PBCPARINF)/SETUP.C ; \
		echo "}" >> $(PBCPARINF)/SETUP.C ; \
		builddir=$(PWD); \
		cd $(BENCHDIRS); \
		par=`basename $(PBCPAR)`;\
		pardir=`basename $(PBCPARDIR)`;\
		tar czvf $$par $$pardir; \
		cd $$builddir; \
		rm -fr $(PBCPARDIR))

all-$(MODNAME): $(BENCHLIB) $(BENCHMAP) $(PBDPAR) $(PBCPAR)

clean-$(MODNAME):
		@rm -f $(BENCHO) $(BENCHDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(BENCHDEP) $(BENCHDS) $(BENCHDH) $(BENCHLIB) $(BENCHMAP) $(PBDPAR) $(PBCPAR); \
		if test -d $(PBDPARDIR); then\
		   rm -fr $(PBDPARDIR); \
		fi; \
		if test -d $(PBCPARDIR); then\
		   rm -fr $(PBCPARDIR); \
		fi

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(BENCHO) $(BENCHDO): CXXFLAGS += -I.
