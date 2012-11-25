# %ROOTTEST-duration%:0
TEST_TARGETS += withspace.test

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


all: tests
test: tests
clean : withspace.clean

ifneq ($(TIME),)
   TESTTIMEACTION_WS = else if [ -f $(TESTTIMINGFILE) ]; then printf " %8s\n" "[`cat $(TESTTIMINGFILE)`ms]" && root.exe -q -b -l -n '$(ROOTTEST_HOME)/scripts/recordtiming.cc+("$(ROOTTEST_HOME)",$(ROOTTEST_RUNID),$(ROOTTEST_TESTID),"$(PWD)/with space","$(TESTTIMINGFILE)")' > /dev/null && rm -f $(TESTTIMINGFILE); fi
endif

withspace.test: utils
	@(echo Running test in $(CALLDIR)/withspace)
	@(cd "with space"; $(TESTTIMEPRE) $(MAKE) "CURRENTDIR=withspace" --no-print-directory $(TESTGOAL) $(TESTTIMEPOST); \
     result=$$?; \
     if [ $$result -ne 0 ] ; then \
         len=`echo Tests in $(CALLDIR)/withspace | wc -c `;end=`expr 68 - $$len`;printf 'Test in withspace %*.*s ' $$end $$end $(DOTS); \
	      printf 'FAIL\n' ; \
         false ; \
			$(TESTTIMEACTION_WS) \
     fi )

withspace.clean: 
	@(cd "with space"; $(MAKE) --no-print-directory clean)
