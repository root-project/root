TEST_TARGETS += GlobalAclicTest
CLEAN_TARGETS += *Dict* output *.o *.so

include ../../Rules.mk

GlobalAclicTest: idDict.cxx

id_cxx.so: id.cxx
	@echo 'gSystem->Exit(!gROOT->ProcessLine(".L id.cxx++"));'  | root.exe -l -b 

idDict.cxx: id.cxx
	rootcint -f idDict.cxx -c id.cxx 