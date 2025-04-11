ALL_CASES = case1 case2 case3
# case4
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


obj/edm_dict.cxx: inc/Wrapper.h inc/edm_selection.xml
	$(CMDECHO) mkdir -p obj lib && genreflex inc/Wrapper.h --output obj/edm_dict.cxx -l lib/libEdm.so -Iinc -s inc/edm_selection.xml --rootmap lib/libEdm.rootmap --rootmap-lib lib/libEdm.so > edm_dict.log

lib/libEdm.so: obj/edm_dict.$(ObjSuf)
	$(BuildFromObj)

# SiStripCluster
obj/strip_dict.cxx: inc/SiStripCluster_classes.h inc/strip_selection.xml \
			inc/DetSetNew.h inc/DetSetVector.h inc/DetSetVectorNew.h \
			inc/SiStripCluster.h
	$(CMDECHO) mkdir -p obj lib && genreflex inc/SiStripCluster_classes.h --output obj/strip_dict.cxx -l lib/libStrip.so -Iinc -s inc/strip_selection.xml --rootmap lib/libStrip.rootmap > strip_dict.log

lib/libStrip.so: obj/strip_dict.$(ObjSuf) lib/libEdm.so
	$(BuildFromObj)


# ClusterTotals
obj/cluster_dict.cxx: inc/ClusterTotals.h inc/cluster_selection.xml inc/DetSetNew.h \
			inc/DetSetVector.h inc/DetSetVectorNew.h inc/SiStripCluster.h
	$(CMDECHO) mkdir -p obj lib && genreflex inc/ClusterTotals.h --output obj/cluster_dict.cxx -l lib/libCluster.so -Iinc -s cluster_selection.xml --rootmap lib/libCluster.rootmap > cluster_dict.log

lib/libCluster.so: obj/cluster_dict.$(ObjSuf) lib/libStrip.so
	$(BuildFromObj)


$(ALL_CASES) : %: %.success
$(ALL_ROOTMAP_CASES) : %: %.success

ALL_CASES_LOG = $(addsuffix .log,$(ALL_CASES))
ALL_ROOTMAP_CASES_LOG = $(addsuffix .log,$(ALL_ROOTMAP_CASES))

$(addsuffix .success,$(ALL_CASES) $(ALL_ROOTMAP_CASES)) : %.success: %.log %.ref
	$(SuccessTestDiff) && touch $@


$(ALL_CASES_LOG): %.log: scripts/%.script lib/libEdm.so lib/libStrip.so lib/libCluster.so $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) LD_LIBRARY_PATH=${CURDIR}/lib:${LD_LIBRARY_PATH} $(CALLROOTEXE) -l -b < $< > $@ 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=$@ --test=$*

$(ALL_ROOTMAP_CASES_LOG): rmap-%.log: scripts/%.script lib/libEdm.so lib/libStrip.so lib/libCluster.so  $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) LD_LIBRARY_PATH=${CURDIR}/lib:${LD_LIBRARY_PATH} $(CALLROOTEXE) -l -b scripts/loadrootmap.C < $< > $@ 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=$@ --test=$*



