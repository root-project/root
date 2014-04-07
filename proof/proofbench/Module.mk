# Module.mk for the proofbench module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 17/2/2011

MODNAME      := proofbench
MODDIR       := $(ROOT_SRCDIR)/proof/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc
PBPARDIR     := etc/proof/proofbench

PROOFBENCHDIR  := $(MODDIR)
PROOFBENCHDIRS := $(PROOFBENCHDIR)/src
PROOFBENCHDIRI := $(PROOFBENCHDIR)/inc

##### libProofBench #####
PROOFBENCHL  := $(MODDIRI)/LinkDef.h
PROOFBENCHDS := $(call stripsrc,$(MODDIRS)/G__ProofBench.cxx)
PROOFBENCHDO := $(PROOFBENCHDS:.cxx=.o)
PROOFBENCHDH := $(PROOFBENCHDS:.cxx=.h)

PROOFBENCHH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PROOFBENCHH  := $(filter-out $(MODDIRI)/TSel%,$(PROOFBENCHH))
PROOFBENCHS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PROOFBENCHS  := $(filter-out $(MODDIRS)/TSel%,$(PROOFBENCHS))
PROOFBENCHO  := $(call stripsrc,$(PROOFBENCHS:.cxx=.o))

PROOFBENCHDEP := $(PROOFBENCHO:.o=.d) $(PROOFBENCHDO:.o=.d)

PROOFBENCHLIB := $(LPATH)/libProofBench.$(SOEXT)
PROOFBENCHMAP := $(PROOFBENCHLIB:.$(SOEXT)=.rootmap)

##### ProofBenchDataSel PAR file #####
PBDPARDIR   := $(call stripsrc,$(PROOFBENCHDIRS)/ProofBenchDataSel)
PBDPARINF   := $(PBDPARDIR)/PROOF-INF
PBDPARH     := $(ROOT_SRCDIR)/test/Event.h $(MODDIRI)/TProofBenchTypes.h
PBDPARS     := $(ROOT_SRCDIR)/test/Event.cxx
PBDPARH     += $(wildcard $(MODDIRI)/TSel*.h)
PBDPARS     += $(wildcard $(MODDIRS)/TSel*.cxx)
PBDPARH     := $(filter-out $(MODDIRI)/TSelHist%, $(PBDPARH))
PBDPARS     := $(filter-out $(MODDIRS)/TSelHist%, $(PBDPARS))

PBDPAR      := $(PBPARDIR)/ProofBenchDataSel.par

##### ProofBenchCPUSel PAR file #####
PBCPARDIR   := $(call stripsrc,$(PROOFBENCHDIRS)/ProofBenchCPUSel)
PBCPARINF   := $(PBCPARDIR)/PROOF-INF
PBCPARH     := $(MODDIRI)/TProofBenchTypes.h $(MODDIRI)/TSelHist.h
PBCPARS     := $(MODDIRS)/TSelHist.cxx

PBCPAR      := $(PBPARDIR)/ProofBenchCPUSel.par

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFBENCHH))
ALLLIBS      += $(PROOFBENCHLIB) $(PBDPAR) $(PBCPAR)
ALLMAPS      += $(PROOFBENCHMAP)

# include all dependency files
INCLUDEFILES += $(PROOFBENCHDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PROOFBENCHDIRI)/%.h
		cp $< $@

$(PROOFBENCHLIB): $(PROOFBENCHO) $(PROOFBENCHDO) $(ORDER_) $(MAINLIBS) \
                  $(PROOFBENCHLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProofBench.$(SOEXT) $@ \
		   "$(PROOFBENCHO) $(PROOFBENCHDO)" \
         "$(PROOFBENCHLIBEXTRA)"

$(call pcmrule,PROOFBENCH)
	$(noop)

$(PROOFBENCHDS): $(PROOFBENCHH) $(PROOFBENCHL) $(ROOTCLINGEXE) $(call pcmdep,PROOFBENCH)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,PROOFBENCH) -c $(PROOFBENCHH) $(PROOFBENCHL)

$(PROOFBENCHMAP): $(PROOFBENCHH) $(PROOFBENCHL) $(ROOTCLINGEXE) $(call pcmdep,PROOFBENCH)
		   $(MAKEDIR)
		   @echo "Generating rootmap $@..."
		   $(ROOTCLINGSTAGE2) -r $(PROOFBENCHDS) $(call dictModule,PROOFBENCH) -c $(PROOFBENCHH) $(PROOFBENCHL)

$(PBDPAR):   $(PBDPARH) $(PBDPARS)
		$(MAKEDIR)
		@echo "Generating PAR file $@..."
		@(if test -d $(PBDPARDIR); then \
		   rm -fr $(PBDPARDIR); \
		fi; \
		mkdir -p $(PBDPARINF); \
		for f in $(PBDPARH) $(PBDPARS); do \
		   $(INSTALL) $$f $(PBDPARDIR); \
		done; \
		echo "#include \"TClass.h\"" > $(PBDPARINF)/SETUP.C ; \
		echo "#include \"TROOT.h\"" >> $(PBDPARINF)/SETUP.C ; \
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
		builddir=`pwd`; \
		cd $(call stripsrc,$(PROOFBENCHDIRS)); \
		par=`basename $(PBDPAR)`; \
		pardir=`basename $(PBDPARDIR)`; \
		tar cf - $$pardir | gzip > $$par || exit 1; \
		mv $$par $$builddir/$(PBPARDIR) || exit 1; \
		cd $$builddir; \
		rm -fr $(PBDPARDIR))

$(PBCPAR):   $(PBCPARH) $(PBCPARS)
		$(MAKEDIR)
		@echo "Generating PAR file $@..."
		@(if test -d $(PBCPARDIR); then \
		   rm -fr $(PBCPARDIR); \
		fi; \
		mkdir -p $(PBCPARINF); \
		for f in $(PBCPARH) $(PBCPARS); do \
		   $(INSTALL) $$f $(PBCPARDIR); \
		done; \
		echo "#include \"TClass.h\"" > $(PBCPARINF)/SETUP.C ; \
		echo "#include \"TROOT.h\"" >> $(PBCPARINF)/SETUP.C ; \
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
		builddir=`pwd`; \
		cd $(call stripsrc,$(PROOFBENCHDIRS)); \
		par=`basename $(PBCPAR)`; \
		pardir=`basename $(PBCPARDIR)`; \
		tar cf - $$pardir | gzip > $$par || exit 1; \
		mv $$par $$builddir/$(PBPARDIR) || exit 1; \
		cd $$builddir; \
		rm -fr $(PBCPARDIR))

all-$(MODNAME): $(PROOFBENCHLIB) $(PBDPAR) $(PBCPAR)

clean-$(MODNAME):
		@rm -f $(PROOFBENCHO) $(PROOFBENCHDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PROOFBENCHDEP) $(PROOFBENCHDS) $(PROOFBENCHDH) \
		   $(PROOFBENCHLIB) $(PROOFBENCHMAP) $(PBDPAR) $(PBCPAR); \
		if test -d $(PBDPARDIR); then \
		   rm -fr $(PBDPARDIR); \
		fi; \
		if test -d $(PBCPARDIR); then \
		   rm -fr $(PBCPARDIR); \
		fi

distclean::     distclean-$(MODNAME)

##### extra rules ######

####$(PROOFBENCHO) $(PROOFBENCHDO): CXXFLAGS += -I.
