TEST_TARGETS = FileInfoTest
CLEAN_TARGETS = dict.* dictold.*

include ../../Rules.mk

FileInfoTest: dict.o dictold.o

dict.cxx: TFileInfo.h LinkDef.h $(ROOTSYS)/bin/rootcint
	rootcint -f dict.cxx -c TFileInfo.h LinkDef.h

dictold.cxx: TFileInfo.h oldLinkDef.h $(ROOTSYS)/bin/rootcint
	rootcint -f dictold.cxx -c TFileInfo.h oldLinkDef.h



