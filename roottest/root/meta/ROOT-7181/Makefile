ALL_CASES = case1 case2 case3 case4
ALL_ROOTMAP_CASES = $(addprefix rmap-,$(ALL_CASES))

CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* $(ALL_LIBRARIES) \
  obj/* lib/* 
TEST_TARGETS += execLoadLibs $(ALL_CASES) $(ALL_ROOTMAP_CASES)

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

obj/jet_dict.cxx: inc/jet_selection.xml inc/Jet_v1.h inc/DataVector.h
	$(CMDECHO) mkdir -p obj lib && genreflex inc/Jet_v1.h --output obj/jet_dict.cxx -l lib/libjet.so -s inc/jet_selection.xml --rootmap lib/libjet.rootmap -Iinc 

obj/sjet_dict.cxx: inc/sjet_selection.xml inc/SJet_v1.h inc/DataVector.h
	$(CMDECHO) mkdir -p obj lib && genreflex inc/SJet_v1.h --output obj/sjet_dict.cxx -l lib/libsjet.so -s inc/sjet_selection.xml --rootmap lib/libsjet.rootmap -Iinc 

obj/btag_dict.cxx: inc/btag_selection.xml inc/BTagVertex_v1.h inc/DataVector.h
	$(CMDECHO) mkdir -p obj lib && genreflex inc/BTagVertex_v1.h --output obj/btag_dict.cxx -l lib/libbtag.so -s inc/btag_selection.xml --rootmap lib/libbtag.rootmap -Iinc 


lib/libjet.$(DllSuf): obj/jet_dict.$(ObjSuf)
	$(BuildFromObj)

lib/libsjet.$(DllSuf): obj/sjet_dict.$(ObjSuf)
	$(BuildFromObj)

lib/libbtag.$(DllSuf): obj/btag_dict.$(ObjSuf)
	$(BuildFromObj)

$(ALL_CASES) : %: %.success
$(ALL_ROOTMAP_CASES) : %: %.success

ALL_CASES_LOG = $(addsuffix .log,$(ALL_CASES))
ALL_ROOTMAP_CASES_LOG = $(addsuffix .log,$(ALL_ROOTMAP_CASES))

$(addsuffix .success,$(ALL_CASES) $(ALL_ROOTMAP_CASES)) : %.success: %.log %.ref
	$(SuccessTestDiff) && touch $@


$(ALL_CASES_LOG): %.log: scripts/%.script lib/libjet.$(DllSuf) lib/libbtag.$(DllSuf) lib/libsjet.$(DllSuf) $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -l -b < $< > $@ 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=$@ --test=$*

$(ALL_ROOTMAP_CASES_LOG): rmap-%.log: scripts/%.script lib/libjet.$(DllSuf) lib/libbtag.$(DllSuf) lib/libsjet.$(DllSuf) $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -l -b scripts/loadrootmap.C < $< > $@ 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=$@ --test=$*



