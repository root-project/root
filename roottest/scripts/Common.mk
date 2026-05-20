# Definitions and targets which are commong to Rules.mk and Disable.mk

ifeq ($(CURRENTDIR),)
  export CURRENTDIR := $(shell basename `pwd`)
else
  export CURRENTDIR
endif
ifeq ($(CALLDIR),)
	export CALLDIR:=.
else
	export CALLDIR:=$(CALLDIR)/$(CURRENTDIR)
endif
#debug2:=$(shell echo CALLDIR=$(CALLDIR) CURRENTDIR=$(CURRENTDIR) PWD=$(PWD) 1>&2 ) 

DOTS="................................................................................"
SUCCESS_FILE = .success.log

# Force the removal of the sucess file ANY time the make is run
REMOVE_SUCCESS := $(shell rm  -f $(SUCCESS_FILE) *.summary)

$(SUCCESS_FILE): check $(TEST_TARGETS)
	@touch $(SUCCESS_FILE)

tests: $(SUCCESS_FILE) 
	@len=`echo Tests in $(CALLDIR) | wc -c `;end=`expr 68 - $$len`; \
           printf 'Tests in %s %*.*s ' $(CALLDIR) $$end $$end $(DOTS) ;
ifeq ($(RUNNINGWITHTIMING),)
	@if [ -f $(SUCCESS_FILE) ] ; then printf 'OK\n' ; else printf 'FAIL\n' ; fi
else
	@if [ -f $(SUCCESS_FILE) ] ; then printf 'OK' ; else printf 'FAIL\n' ; fi
endif
