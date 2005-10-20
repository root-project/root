ifeq (,$(strip $(ROOTTEST_HOME)))
export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

TEST_TARGETS += vararyobj vararynoclassdefobj
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *.root

include $(ROOTTEST_HOME)/scripts/Rules.mk

.PHONY: vararyobj vararyruns vararynoclassdefobj vararynoclassdefruns

vararyobj: vararyruns
	$(CMDECHO) rm -f tmp1.log tmp2.log ; \
	tail +3 varyingArrayWrite.clog > tmp1.log ; \
	tail +3 varyingArrayRead.clog > tmp2.log ; \
	diff tmp1.log tmp2.log ; \
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

vararynoclassdefobj: vararynoclassdefruns
	$(CMDECHO) rm -rf tmp3.log tmp4.log ; \
	tail +3 varyingArrayNoClassDefWrite.clog > tmp3.log ; \
	tail +3 varyingArrayNoClassDefRead.clog > tmp4.log ; \
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

