#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += libs check

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


FooDict.cxx : Foo.h
	$(CMDECHO)rootcint -f FooDict.cxx -c Foo.h+

Foo.$(ObjSuf): Foo.h

libs : Foo.$(DllSuf)

Foo.$(DllSuf): Foo.$(ObjSuf) FooDict.$(ObjSuf)
	$(BuildFromObjs) 

reader: reader.$(ObjSuf)
	$(CMDECHO)g++ -o $@ reader.o `root-config --libs` \
		-Wl,-rpath,`root-config --libdir`

writer: writer.$(ObjSuf) Foo.$(DllSuf) 
	$(CMDECHO)g++ -o $@ writer.o Foo.$(DllSuf) `root-config --libs` \
		-Wl,-rpath,`root-config --libdir` -Wl,-rpath,.

dist:
	$(CMDECHO)mkdir -p treeproblem 
	$(CMDECHO)cp Foo.{cxx,h} tree.{C,h} reader.C writer.C Makefile README \
		treeproblem
	$(CMDECHO)tar -czvf treeproblem.tar.gz treeproblem 
	$(CMDECHO)rm -rf treeproblem 

check: Foo.$(DllSuf) reader.C writer.C 
	$(CMDECHO)root -b -l -q writer.C 
	$(CMDECHO)root -b -l -q reader.C

#clean:
#	$(CMDECHO)rm -f *.root *~ *.o *Dict.cxx *Dict.h *.$(DllSuf) reader writer
