SHELL = /bin/bash




test: operator1
	true;

operator1:
	@rm  -f operator1.cint.out operator1.comp.out
	@(echo '.x smalltest.C' | root.exe -l -b ) > operator1.cint.out 2>&1
	@(echo '.x smalltest.C++' | root.exe -l -b ) > operator1.comp.out 2>&1
	@cat operator1.comp.out | grep -v 'not declared' |grep -v 'Info in <ACLiC>:' | diff - operator1.cint.out


