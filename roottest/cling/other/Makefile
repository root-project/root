# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.clog *.log runStaticDataMemberDict.* *.timestamp

#Set the list of target to make while testing
TEST_TARGETS += fileClose snprintfselect staticConstTest StaticDataMemberDict StaticDataMember \
    savannah54662 assertBaseMembers missingXFunc

# adjust the location of Rules.mk as needed.
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

fileClose : fileClose.log
	$(TestDiff)

fileClose.log : fileClose_C.$(DllSuf)

snprintfselect : snprintfselect.log
	$(TestDiff)

staticConstTest : staticConstTest.log
	$(CMDECHO) diff -b staticConstTest.log staticConstTest.ref

staticConstTest.log : staticConst.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q staticConst.C >  staticConstTest.log 

StaticDataMemberDict : runStaticDataMemberDict.C

runStaticDataMemberDict.C : runStaticDataMember.h runStaticDataMember_linkdef.h
	$(CMDECHO)rootcint -f runStaticDataMemberDict.C runStaticDataMember.h runStaticDataMember_linkdef.h

ifeq (,$(strip $(FAIL)))
StaticDataMember :
	$(WarnFailTest)

else
StaticDataMember : StaticDataMember.timestamp

StaticDataMember.timestamp : StaticDataMember.clog StaticDataMember.log
	$(CMDECHO)egrep -v 'Processing runStaticDataMember\.C' StaticDataMember.clog > tmp$$$$.clog ; \
	mv tmp$$$$.clog StaticDataMember.clog ; \
	egrep -v 'Processing runStaticDataMember\.C' StaticDataMember.log > tmp$$$$.log ; \
	mv tmp$$$$.log StaticDataMember.log ; \
	touch $@ ; \
	diff -b -u StaticDataMember.clog StaticDataMember.log

endif

savannah54662.log: testCompile_C.$(DllSuf)

savannah54662: savannah54662.log
	$(TestDiff)

assertBaseMembers:
	$(WarnFailTest)

missingXFunc.log: runmissingXFunc.C
	$(CMDECHO) ( ! $(CALLROOTEXE) -l -b -q '$^(12)' > $@.tmp 2>&1) || handleError.sh --cmd='! $(CALLROOTEXE) -l -b -q $^' --result=$$? --log=$@.tmp
	$(CMDECHO) sed -e 's,input_line_[^:]*,input_line_,' -e 's,^.*/runmissingXFunc.C:4:6:,,' $@.tmp > $@
	$(CMDECHO) rm -f $@.tmp

missingXFunc: missingXFunc.log
	$(TestDiff)
