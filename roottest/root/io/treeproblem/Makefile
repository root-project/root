#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log file.root

#Set the list of target to make while testing
TEST_TARGETS += libs check

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


FooDict.cxx : Foo.h
	$(CMDECHO)rootcint -f FooDict.cxx Foo.h+

reader:reader.C
	$(CMDECHO)$(CXX) -c -g `root-config --cflags` $<  
	$(CMDECHO)$(CXX) -o $@ reader.o `root-config --libs` \
                -Wl,-rpath,`root-config --libdir`

Foo.$(ObjSuf): Foo.h

libs : Foo.$(DllSuf)

Foo.$(DllSuf): Foo.$(ObjSuf) FooDict.$(ObjSuf)
	$(BuildFromObjs) 

writer: writer.$(ObjSuf) Foo.$(DllSuf) 
	$(CMDECHO)$(CXX) -o $@ writer.o Foo.$(DllSuf) `root-config --libs` \
		-Wl,-rpath,`root-config --libdir` -Wl,-rpath,.

dist:
	$(CMDECHO)mkdir -p treeproblem 
	$(CMDECHO)cp Foo.{cxx,h} tree.{C,h} reader.C writer.C Makefile README \
		treeproblem
	$(CMDECHO)tar -czvf treeproblem.tar.gz treeproblem 
	$(CMDECHO)rm -rf treeproblem 

reader.log: Foo.$(DllSuf) reader.C writer.log
	$(CMDECHO)$(CALLROOTEXE) -b -l -q reader.C > reader.log 2>&1

ifneq ($(ClingWorkAroundMissingDynamicScope),)
reader.log writer.log: CALLROOTEXE += -e 'gSystem->Load("Foo");'
endif

ifneq ($(ClingWorkAroundMissingAutoLoading),)
reader.log writer.log: CALLROOTEXE += -e 'gSystem->Load("libTree");'
endif

writer.log: Foo.$(DllSuf) writer.C 
	$(CMDECHO)$(CALLROOTEXE) -b -l -q writer.C > writer.log 2>&1

check: writer.log reader.log
	$(CMDECHO) diff -buw reader.ref reader.log && diff -ubw writer.ref writer.log

#clean:
#	$(CMDECHO)rm -f *.root *~ *.o *Dict.cxx *Dict.h *.$(DllSuf) reader writer
