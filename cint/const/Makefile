TEST_TARGETS += constTest
CLEAN_TARGET += *.so output

include ../../root/Rules.mk


constTest: const.C
	echo "gROOT->ProcessLine(\".L const.C\");run()" | root.exe -b -l > output  2>&1
	echo "gROOT->ProcessLine(\".L const.C++\");run()" | root.exe -b -l >> output 2>&1 
	diff good.output output

