#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += Foo.so  reader writer

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

Foo.so:Foo.h Foo.cxx
	$(CMDECHO)rootcint -f FooDict.cxx -c Foo.h+
	$(CMDECHO)g++ -Wall -c -g `root-config --cflags` Foo.cxx 
	$(CMDECHO)g++ -Wall -c -g `root-config --cflags` FooDict.cxx 
	$(CMDECHO)g++ -shared -Wl,-soname,Foo.so -o Foo.so Foo.o FooDict.o

reader:reader.C
	$(CMDECHO)g++ -Wall -c -g `root-config --cflags` $<  
	$(CMDECHO)g++ -o $@ reader.o `root-config --libs` \
		-Wl,-rpath,`root-config --libdir`

writer:writer.C Foo.so 
	$(CMDECHO)g++ -Wall -c -g `root-config --cflags` $< 
	$(CMDECHO)g++ -o $@ writer.o Foo.so `root-config --libs` \
		-Wl,-rpath,`root-config --libdir` -Wl,-rpath,.

dist:
	$(CMDECHO)mkdir -p treeproblem 
	$(CMDECHO)cp Foo.{cxx,h} tree.{C,h} reader.C writer.C Makefile README \
		treeproblem
	$(CMDECHO)tar -czvf treeproblem.tar.gz treeproblem 
	$(CMDECHO)rm -rf treeproblem 

check:Foo.so reader writer 
	$(CMDECHO)root -b -l -q writer.C 
	$(CMDECHO)root -b -l -q reader.C

#clean:
#	$(CMDECHO)rm -f *.root *~ *.o *Dict.cxx *Dict.h *.so reader writer
