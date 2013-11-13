# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.clog *.log runStaticDataMemberDict.* *.timestamp

#Set the list of target to make while testing
TEST_TARGETS += fileClose sprintfselect staticConstTest StaticDataMemberDict StaticDataMember \
    savannah54662 assertBaseMembers

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

fileClose : fileClose.log
	$(TestDiff)

fileClose.log : fileClose_C.$(DllSuf)

sprintfselect : sprintfselect.log
	$(TestDiff)

staticConstTest : staticConstTest.log
	$(CMDECHO) diff -b staticConstTest.log staticConstTest.ref

staticConstTest.log : staticConst.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q staticConst.C >  staticConstTest.log 

StaticDataMemberDict : runStaticDataMemberDict.C

runStaticDataMemberDict.C : runStaticDataMember.h runStaticDataMember_linkdef.h
	$(CMDECHO)rootcint -f runStaticDataMemberDict.C -c runStaticDataMember.h runStaticDataMember_linkdef.h

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
