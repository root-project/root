ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


tests: dicto

CLEAN_TARGETS += *.build.log TeleDictTmp* MyClassDictTmp*

MyClassDict.C: Header.h Linkdef.h $(ROOTCINT)
	$(CMDECHO) rootcint -f MyClassDict.C -c Header.h Linkdef.h

TeleDict.C: Tele.h TeleLinkdef.h $(ROOTCINT)
	$(CMDECHO) test ! `(rootcint -f TeleDict.C -c Tele.h TeleLinkdef.h >TeleDict.log 2>&1 )`  && touch TeleDict.C 

dicto : MyClassDict.o TeleDict.o
	$(CMDECHO) diff -b -w dicto.ref TeleDict.log 
