TEST_TARGETS += GlobalAclicTest
CLEAN_TARGETS += *Dict* output *.o *.so *.dll *.lib *.exp *.def *.d *.pdb

include ../../Rules.mk

GlobalAclicTest: id_cxx.so idDict.cxx

id_cxx.so: id.cxx buildId.C
	root.exe -l -b -q buildId.C

idDict.cxx: id.cxx
	rootcint -f idDict.cxx -c id.cxx LinkDef.h
