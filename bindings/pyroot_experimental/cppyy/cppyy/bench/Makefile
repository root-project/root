dicts = functioncallsDict.so
all : $(dicts)

ifeq ($(ROOTSYS),)
  clingconfig := cling-config
else
  clingconfig := root-config
endif

cppflags=$(shell $(clingconfig) --cflags) -O3 -fPIC

PLATFORM := $(shell uname -s)
ifeq ($(PLATFORM),Darwin)
  cppflags+=-dynamiclib -single_module -arch x86_64 -undefined dynamic_lookup
endif

%Dict.so: %_rflx.cpp %.cxx
	$(CXX) $(cppflags) -shared -o $@ $^

%_rflx.cpp: %.h %.xml
	genreflex $< --selection=$*.xml --rootmap=$*Dict.rootmap --rootmap-lib=$*Dict.so

.PHONY: test clean

test:
	pytest test_*.py

clean:
	-rm -f $(dicts) $(subst .so,.rootmap,$(dicts)) $(subst Dict.so,_rflx_rdict.pcm,$(dicts)) $(subst Dict.so,_rflx.cpp,$(dicts)) $(wildcard *.pyc)
