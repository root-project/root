CLEAN_TARGETS += $(ALL_LIBRARIES)

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(PWD)/
endif

# Temporary addition for cling developement
CLING_SUBDIRS := $(shell cat cling.tests)
CLING_TESTS := $(patsubst %,%.test,$(CLING_SUBDIRS) )

include $(ROOTTEST_HOME)/scripts/Rules.mk

# Temporary addition for cling developement
CLING_SUBDIR := $(shell cat cling.tests)
CLING_TESTS := $(patsubst %,%.test,$(CLING_SUBDIR) )

#%.test:
#	$(CMDECHO) cd $* && make test

%.test: $(EVENTDIR)/$(SUCCESS_FILE) utils
	@(echo Running test in $(CALLDIR)/$*)
	@(cd $(CALLDIR)/$*; $(TESTTIMEPRE) $(MAKE) CURRENTDIR=$* --no-print-directory $(TESTGOAL) $(TESTTIMEPOST) ; \
     result=$$?; \
     if [ $$result -ne 0 ] ; then \
         len=`echo Tests in $(CALLDIR)/$* | wc -c `;end=`expr 68 - $$len`;printf 'Test in %s %*.*s ' $(CALLDIR)/$* $$end $$end $(DOTS); \
              printf 'FAIL\n' ; \
         false ; \
     $(TESTTIMEACTION) \
     fi ) 

clingtest: $(CLING_TESTS)
