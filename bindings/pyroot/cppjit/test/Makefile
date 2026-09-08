# the C++ sides of the tests live under cpp/ and build into shared
# libraries next to their sources (load_reflection_info resolves the
# header as a sibling of the loaded Dict library)
soext = .so
ifeq ($(OS),Windows_NT)
  soext = .dll
endif

dictnames = advancedcpp \
            advancedcpp2 \
            conversions \
            cpp11features \
            crossinheritance \
            datatypes \
            doc_helper \
            example01 \
            fragile \
            operators \
            overloads \
            pythonizables \
            std_streams \
            stltypes \
            templates

dicts = $(addprefix cpp/,$(addsuffix Dict$(soext),$(dictnames)))

all : $(dicts)

PYTHON ?= python3
cppflags= -std=c++17 -O3 -fPIC -I$(shell $(PYTHON) -c 'import sysconfig as sc; print(sc.get_config_var("INCLUDEPY"))') -Wno-register

PLATFORM := $(shell uname -s)
ifeq ($(PLATFORM),Darwin)
  cppflags+=-dynamiclib -single_module -undefined dynamic_lookup -Wno-delete-non-virtual-dtor
endif

# a worker can load the library while another rebuilds it, so publish it whole
cpp/%Dict$(soext): cpp/%.cxx cpp/%.h
	$(CXX) $(cppflags) -shared -o $@.$$$$.tmp $< && mv -f $@.$$$$.tmp $@

# convenience: `make datatypesDict$(soext)` builds cpp/datatypesDict$(soext)
%Dict$(soext): cpp/%Dict$(soext) ;

.PHONY: test clean

test:
	pytest test_*.py

clean:
	-rm -f $(dicts) cpp/*.tmp cpp/*.lock
