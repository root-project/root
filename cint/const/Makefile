TEST_TARGETS += constTest constCharTest
CLEAN_TARGET += *.so output

include ../../root/Rules.mk


constTest: const.C
	$(CMDECHO) echo "gROOT->ProcessLine(\".L const.C\");run()" | root.exe -b -l  2>&1 | grep -v 'Info in <TUnixSystem::ACLiC>' > output  2>&1
	$(CMDECHO) echo "gROOT->ProcessLine(\".L const.C++\");run()" | root.exe -b -l  2>&1 | grep -v 'Info in <TUnixSystem::ACLiC>' >> output 2>&1 
	$(CMDECHO) diff good.output output

constCharTest: 
	$(CMDECHO) root.exe -b -q -l ../../root/build.C\(\"constissue.h\"\) > constCharTest.log
