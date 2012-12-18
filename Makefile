.DEFAULT_GOAL := clingtest

CLEAN_TARGETS += $(ALL_LIBRARIES)

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(PWD)/
endif

# Temporary addition for cling developement
CLING_SUBDIRS := $(shell cat cling.tests)
CLING_TESTS := $(patsubst %,%.test,$(CLING_SUBDIRS) )

ifeq ($(MAKECMDGOALS),clingtest)
MAKE += FIXCLING=yes
else
ifeq ($(.DEFAULT_GOAL),clingtest)
MAKE += FIXCLING=yes
endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

# Temporary addition for cling developement
CLING_SUBDIR := $(shell cat cling.tests)
CLING_TESTS := $(patsubst %,%.test,$(CLING_SUBDIR) )

#%.test:
#	$(CMDECHO) cd $* && make test

root/aclic/withspace.test: utils
	@(echo Running test in $(CALLDIR)/withspace)
	@(cd "root/aclic/with space"; $(TESTTIMEPRE) $(MAKE) "CURRENTDIR=withspace" --no-print-directory $(TESTGOAL) $(TESTTIMEPOST); \
     result=$$?; \
     if [ $$result -ne 0 ] ; then \
         len=`echo Tests in $(CALLDIR)/withspace | wc -c `;end=`expr 68 - $$len`;printf 'Test in withspace %*.*s ' $$end $$end $(DOTS); \
              printf 'FAIL\n' ; \
         false ; \
                        $(TESTTIMEACTION_WS) \
     fi )

%.test: $(EVENTDIR)/$(SUCCESS_FILE) utils
	@(echo Running test in $(CALLDIR)/$*)
	@(cd $(CALLDIR)/$* && $(TESTTIMEPRE) $(MAKE) CURRENTDIR=$* --no-print-directory $(TESTGOAL) $(TESTTIMEPOST) ; \
     result=$$?; \
     if [ $$result -ne 0 ] ; then \
         len=`echo Tests in $(CALLDIR)/$* | wc -c `;end=`expr 68 - $$len`;printf 'Test in %s %*.*s ' $(CALLDIR)/$* $$end $$end $(DOTS); \
              printf 'FAIL\n' ; \
         false ; \
     $(TESTTIMEACTION) \
     fi ) 

cling-$(SUCCESS_FILE): $(CLING_TESTS)
	@touch cling-$(SUCCESS_FILE)

clingtest: cling-$(SUCCESS_FILE)
	@len=`echo Cling tests in $(CALLDIR) | wc -c `;end=`expr 68 - $$len`; \
           printf 'Cling tests in %s %*.*s ' $(CALLDIR) $$end $$end $(DOTS) ;
ifeq ($(RUNNINGWITHTIMING),)
	@if [ -f cling-$(SUCCESS_FILE) ] ; then printf 'OK\n' ; else printf 'FAIL\n' ; fi
else
	@if [ -f cling-$(SUCCESS_FILE) ] ; then printf 'OK' ; else printf 'FAIL\n' ; fi
endif

