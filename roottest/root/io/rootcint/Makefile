# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log  MidadCint* cltest.root

#Set the list of target to make while testing
TEST_TARGETS += sigbug virtualbase templateKeyword mytest rootcintIntTest constMemberTest cltest nestedtemplate weirdpairTest

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

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

testDict.cxx: test.h testLinkDef.h
	$(CMDECHO) rootcint -f testDict.cxx test.h testLinkDef.h

rootcintIntTest: testDict.$(ObjSuf)

constMemberDict.cxx: constMember.h linkdef.h $(ROOTCINT)
	$(CMDECHO) rootcint -f constMemberDict.cxx constMember.h linkdef.h 

constMemberTest: constMemberDict.$(ObjSuf)

templateKeyword.log: templateKeyword_C.$(DllSuf)

templateKeyword: templateKeyword.log
	$(TestDiff)

virtualbase.log: virtualbase_h.$(DllSuf)

virtualbase: virtualbase.log
	$(TestDiff)

sigbug.log:
	$(CMDECHO) (echo start && rootcint -f MidadCint.cc -p -Isigbug -DUNIX -DLINUX -D__UNIX__ -D__LINUX__ -DDATAREP_LITTLE_IEEE -DDATAREP_LITTLE_ENDIAN -DDEFECT_NO_IOSTREAM_NAMESPACES -DDEFECT_NO_JZEXT -DDEFECT_NO_INTHEX -DDEFECT_NO_INTHOLLERITH -DDEFECT_NO_READONLY -DDEFECT_NO_DIRECT_FIXED -DDEFECT_NO_STRUCTURE -DUSE_ROOT -I${ROOTSYS}/include Mint.h LinkDef.h) | grep -v _cint.cxx > sigbug.log 2>&1


sigbug: sigbug.log
	$(TestDiff)

out.cxx: foo.hh foo_linkdef.h
	$(CMDECHO) rootcint -f out.cxx foo.hh foo_linkdef.h

weirdpair_dict.cxx: weirdpair.h weirdpair_LinkDef.h
	$(CMDECHO) rootcling -f weirdpair_dict.cxx  weirdpair.h weirdpair_LinkDef.h > weirdpair_dict.rootcling.log 2>&1

weirdpairTest: weirdpair_dict.$(ObjSuf)

namespacetemp: out.$(ObjSuf)

cltest_C.$(DllSuf): cltestClass.h cltestClass.cxx cltestLinkdef.h

cltest.log: cltest_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l cltest.C+ > cltest.log 2>&1 

ifeq ($(FAIL)$(PLATFORM),win32)
cltest: 
	$(WarnFailTest)
else
cltest: cltest.log
	$(TestDiff)
endif

ifeq ($(FAIL),)
nestedtemplate:
	$(WarnFailTest)
else
nestedtemplate: nestedtemplate.clog
	$(TestDiff)
endif
