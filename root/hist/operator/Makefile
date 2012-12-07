TEST_TARGETS += add_test copy_test Projections_test sparse_test

CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog add copy Projections sparse

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


add_test : add

add :
	$(CMDECHO) $(CALLROOTEXE) -b -q -l run.C > run.log 2>&1 && touch $@


copy_test : copy

copy : copy.clog
	$(TestDiff) && touch $@


Projections_test : Projections

Projections : Projections.log Projections.ref
	$(TestDiff) && touch $@


sparse_test : sparse

sparse : sparse.log sparse.ref
ifeq ($(ClingWorkAroundJITandInline),)
	$(TestDiff) && touch $@
endif

ifneq ($(ClingWorkAroundMissingAutoLoading),)
Projections.log : CALLROOTEXE += -e "gSystem->Load(\"libHist\");" 
sparse.log : CALLROOTEXE += -e "gSystem->Load(\"libHist\");" 
endif
