arch := $(shell grep arch `which root-config` | head -1 | cut -d'=' -f2-)
CXXFLAGS := $(shell root-config --cflags)
LIBS := $(shell root-config --libs)

ifeq ($(findstring kcc,$(arch)),kcc)
CC = KCC --no_exceptions
LD = KCC --no_exceptions
else
ifeq ($(findstring sgicc,$(arch)),sgicc)
CC = CC -n32 -I/usr/include/CC.sgi 
LD = CC -n32 -I/usr/include/CC.sgi 

else

ifeq ($(findstring solarisCC5,$(arch)),solarisCC5)

CC = CC 
LD = CC 

else

CC = g++
LD = g++

endif
endif
endif


test2:
	echo $(arch)

all:	driver libIotest.so
	./driver

test:   driver
	rm driver.output
	./driver 2>&1 | grep -v -- '->[^0][^ ]' > driver.output 
	diff good.output driver.output

clean:
	rm -rf *.o *.so ti_files ii_files 
	rm -rf mydict.* driver

driver: main.cc mydict.cc RootCaloHit.h  RootData.h  RootPCellID.h $(ROOTSYS)/include/Rtypes.h
	$(CC) -g -o driver $(CXXFLAGS) main.cc mydict.cc $(LIBS) 

infodriver: tmain.cc mydict.cc RootCaloHit.h  RootData.h  RootPCellID.h $(ROOTSYS)/include/Rtypes.h
	$(CC) -g -o infodriver $(CXXFLAGS) tmain.cc mydict.cc $(LIBS) 

mydict.cc:  LinkDef.h RootData.h RootCaloHit.h  RootPCellID.h $(ROOTSYS)/bin/rootcint
	rootcint -f mydict.cc -c RootData.h RootCaloHit.h LinkDef.h

.cc.o:
	g++ -v

libIotest.so: main.cc mydict.cc RootCaloHit.h  RootData.h  RootPCellID.h $(ROOTSYS)/include/Rtypes.h
	$(CC) -g -shared -o libIotest.so -DSHARED $(CXXFLAGS) main.cc mydict.cc $(LIBS) 
