TEST_TARGETS += GlobalAclicTest
CLEAN_TARGETS += *Dict* output *.o *.so

include ../../Rules.mk

GlobalAclicTest: id_cxx.so idDict.cxx

id_cxx.so: id.cxx
	@echo 'gSystem->Exit(!gSystem->CompileMacro("id.cxx","k"));' | root.exe -l -b 

idDict.cxx: id.cxx
	rootcint -f idDict.cxx -c id.cxx LinkDef.h
