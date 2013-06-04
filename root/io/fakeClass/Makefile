# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) little.root empty.root testbase.root t01.out *.log

#Set the list of target to make while testing
TEST_TARGETS += forwardDeclAndFake littleTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


forwardDeclAndFake: t01.out
		  $(CMDECHO)diff -bw t01.out t01.good

t01.out: t01.C
		  $(CMDECHO)$(CALLROOTEXE) -q -b -l t01.C > t01.out 2>&1 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

little.root: produceLittleFile.C little_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q produceLittleFile.C > produceLittleFile.log

printLittle.log : printLittle.C little_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q printLittle.C > printLittle.log 2>&1 

checkLittle.log : checkLittle.C little_v2_C.$(DllSuf) little.root
	$(CMDECHO) $(CALLROOTEXE) -b -l -q checkLittle.C > checkLittle.log 2>&1 

checkLittle0.log : checkLittle.C little_v2_C.$(DllSuf) little.root
	$(CMDECHO) $(CALLROOTEXE) -b -l -q checkLittle0.C > checkLittle0.log 2>&1 

checkLittle1.log : checkLittle1.C little_v2_C.$(DllSuf) little.root
	$(CMDECHO) $(CALLROOTEXE) -b -l -q checkLittle1.C > checkLittle1.log 2>&1 

littleTest: printLittle.log checkLittle0.log checkLittle.log checkLittle1.log 
ifeq ($(FAIL),)
	@echo "The library loading order problem is not solved yet"
	$(WarnFailTest)	
else
	$(CMDECHO) diff -b -u checkLittle0.ref checkLittle0.log && \
	diff -b -u checkLittle.ref checkLittle.log && \
	diff -b -u checkLittle1.ref checkLittle1.log 
endif

