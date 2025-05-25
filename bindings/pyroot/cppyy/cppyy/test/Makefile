dicts = advancedcppDict.so \
        advancedcpp2Dict.so \
        conversionsDict.so \
        cpp11featuresDict.so \
        crossinheritanceDict.so \
        datatypesDict.so \
        doc_helperDict.so \
        example01Dict.so \
        fragileDict.so \
        operatorsDict.so \
        overloadsDict.so \
        pythonizablesDict.so \
        std_streamsDict.so \
        stltypesDict.so \
        templatesDict.so

all : $(dicts)

cppflags=$(shell root-config --cflags) -O3 -fPIC -I$(shell python -c 'import sysconfig as sc; print(sc.get_config_var("INCLUDEPY"))') -Wno-register
PLATFORM := $(shell uname -s)
ifeq ($(PLATFORM),Darwin)
  cppflags+=-dynamiclib -undefined dynamic_lookup -Wno-delete-non-virtual-dtor
endif

# For these tests, the rootcling invocation looks like the following:
# rootcling advancedcpp_rflx.cpp --rmf advancedcppDict.rootmap --rml advancedCppDict.so advancedcpp.h advancedcpp.xml
%_rflx.cpp: %.h %.xml
	rootcling $@ --rmf $*Dict.rootmap --rml $*Dict.so $^

# Compiling the test dictionaries to shared libs:
# g++ $(cppflags) -shared -o advancedcppDict.so advancedcpp_rflx.cpp advancedcpp.cxx
%Dict.so: %_rflx.cpp %.cxx
	$(CXX) $(cppflags) -shared -o $@ $^

.PHONY: test clean

test:
	pytest test_*.py

clean:
	-rm -f $(dicts) $(subst .so,.rootmap,$(dicts)) $(subst Dict.so,_rflx_rdict.pcm,$(dicts)) $(subst Dict.so,_rflx.cpp,$(dicts)) $(wildcard *.pyc)
