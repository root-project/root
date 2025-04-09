ifeq (,$(strip $(ROOTTEST_HOME)))
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

TEST_TARGETS += vararyobj varary51508 vararynoclassdefobj vararypolypobj
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog vararypolyp.root varyingArray.root varyingArrayNoClassDef.root


include $(ROOTTEST_HOME)/scripts/Rules.mk

.PHONY: vararyobj vararyruns varary51508 vararynoclassdefobj vararynoclassdefruns vararypolypobj vararypolypruns

vararyobj: vararyruns
	$(CMDECHO) rm -f tmp1.log tmp2.log ; \
	tail -3 varyingArrayWrite.clog > tmp1.log ; \
	tail -3 varyingArrayRead.clog > tmp2.log ; \
	diff -b tmp1.log tmp2.log ; \
	if [ "$$?" != "0" ] ; \
	then \
	  exit 255 ; \
	fi ; \
	: rm -f *.log *.clog *.d *.so *.root; \
	rm tmp1.log tmp2.log ; \
	:


# We do it this way so that if make -j is used the Read test
# does not run until the Write test has finished.
vararyruns: varyingArrayWrite.clog
	$(CMDECHO) $(MAKE) --no-print-directory varyingArrayRead.clog

varary51508: varyingArrayRead_51508.clog
	$(CMDECHO) diff -bu varyingArrayRead_51508.good varyingArrayRead_51508.clog

vararynoclassdefobj: vararynoclassdefruns
	$(CMDECHO) rm -rf tmp3.log tmp4.log ; \
	tail -3 varyingArrayNoClassDefWrite.clog > tmp3.log ; \
	tail -3 varyingArrayNoClassDefRead.clog > tmp4.log ; \
	diff tmp3.log tmp4.log ; \
	if [ "$$?" != "0" ] ; \
	then \
	  exit 255 ; \
	fi ; \
	: rm -f *.log *.clog *.d *.so *.root; \
	rm tmp3.log tmp4.log ; \
	:


# We do it this way so that if make -j is used the Read test
# does not run until the Write test has finished.
vararynoclassdefruns: varyingArrayNoClassDefWrite.clog
	$(CMDECHO) $(MAKE) --no-print-directory varyingArrayNoClassDefRead.clog

vararypolypobj: vararypolypruns
	$(CMDECHO)diff -bu writevararypolyp.good writevararypolyp.clog; \
	status=$$?; \
	if test $$status -ne 0; \
	then \
	  exit $$status; \
	fi; \
	diff -bu readvararypolyp.good readvararypolyp.clog


# We do it this way so that if make -j is used the Read test
# does not run until the Write test has finished.
vararypolypruns: writevararypolyp.clog
	$(CMDECHO) $(MAKE) --no-print-directory readvararypolyp.clog

