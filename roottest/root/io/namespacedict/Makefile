ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


tests: dicto

CLEAN_TARGETS += *.build.log TeleDictTmp* MyClassDictTmp*

MyClassDict.C: Header.h Linkdef.h $(ROOTCINT)
	$(CMDECHO) rootcint -f MyClassDict.C Header.h Linkdef.h

TeleDict.C: Tele.h TeleLinkdef.h $(ROOTCINT)
	$(CMDECHO) test ! `(rootcint -f TeleDict.C Tele.h TeleLinkdef.h >TeleDict.log 2>&1 )`  && touch TeleDict.C 

dicto : MyClassDict.o TeleDict.o
	$(CMDECHO) diff -b -w dicto.ref TeleDict.log 
