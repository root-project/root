TEST_TARGETS = multipleInheritTest
CLEAN_TARGETS = na_cxx.$(DllSuf) hout.root

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


multipleInheritTest: na_cxx.$(DllSuf) hout.root
	$(CMDECHO) root.exe -q -l -b re.C > run.C 

na_cxx.$(DllSuf) : na.cxx na.h
	$(CMDECHO) echo '{gSystem->Exit(!gSystem->CompileMacro("na.cxx","kf"));}' > build.C && \
	root.exe -q -l -b build.C && \
	rm build.C > build.log

hout.root: wr.C na_cxx.$(DllSuf)
	$(CMDECHO) root.exe -q -l -b wr.C>& /dev/null





