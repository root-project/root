
#===[ usefull variables ]=====================================================#

GCCXMLPATH?=$(shell which gccxml 2>/dev/null)
#GCCXMLPATH = /home/ljanyst/apps/gccxml/cvs01/install/bin

ifneq ($(findstring gccxml,$(notdir $(GCCXMLPATH))),gccxml)
   DICTS := $(filter-out %REFLEX.so,$(DICTS) )
endif

CLEAN_TARGETS += *\.o *\.x core* *~ a\.out \.*sw* *\.so Makefile\.* _* *_dictREFLEX.cxx *_dictCINT* *\.d *\.root *\.log
LOGS = $(patsubst %.x,%.log, $(PROGS))
LOCAL_TEST =  $(patsubst %.x, %.test, $(PROGS))
TEST_TARGETS += $(LOCAL_TEST)

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

CPPFLAGS   += -I../common -I.
LDFLAGS    = $(ROOTLIBS)

REFV       = lib%_dictREFLEX.so
CINTV      = lib%_dictCINT.so

#=============================================================================#

#===[ general rules ]=========================================================#

.PHONY: dict all $(LOCAL_TEST)
all: $(PROGS)

dict: $(DICTS)

$(PROGS): | $(DICTS)

%.o: %.cxx
	@echo -e "[i] compiling: $@"
	@$(CXX) -MD $(CXXFLAGS) $(CPPFLAGS) -c $<

-include $(OBJS:.o=.d)

#clean:
#	@echo -e "[i] deleting useless files"
#	@rm -rf *\.o *\.x core* *~ a\.out \.*sw* *\.so Makefile\.* _* *_dictREFLEX.cxx *_dictCINT*
#	@rm -rf *\.d *\.root *\.log

deps-clean:
	@echo -e "[i] Cleaning dependency files"
	@rm -rf *\.d

ifeq ($(findstring gccxml,$(notdir $(GCCXMLPATH))),gccxml)
lib%_dictREFLEX.so:
	@echo -e "[i] generating dictionary: $@"
	@$(CXX) -M -x c++ $(patsubst $(REFV),%.h,$@) | sed -e 's/$(patsubst $(REFV),%.o,$@):/$@: $(patsubst $(REFV),selection_%.xml,$@)/' > $(patsubst %.so,%.d,$@)
	@genreflex $(patsubst $(REFV),%.h,$@)  --gccxmlpath=$(dir $(GCCXMLPATH)) --gccxmlopt='$(GCCXMLOPTS)' -o $(patsubst $(REFV),%_dictREFLEX.cxx,$@) --selection=$(patsubst $(REFV),selection_%.xml,$@)
	@echo -e "[i] building dictionary"
	@$(CXX) -shared $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -L. -o $@ $(patsubst $(REFV),%_dictREFLEX.cxx,$@)
endif

lib%_dictCINT.so:
	@echo -e "[i] generating dictionaly: $@"
	@$(CXX) -M -x c++ $(patsubst $(CINTV),%.h,$@) | sed -e 's/$(patsubst $(CINTV),%.o,$@):/$@: $(patsubst $(CINTV),%LinkDef.h,$@)/' > $(patsubst %.so,%.d,$@)
	@rootcint -f $(patsubst $(CINTV),%_dictCINT.cxx,$@) $(patsubst $(CINTV),%.h,$@) $(patsubst $(CINTV),%LinkDef.h,$@)
	@echo -e "[i] building dictionary"
	@$(CXX) -shared $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $(patsubst $(CINTV),%_dictCINT.cxx,$@)

-include $(DICTS:.so=.d)

%.x: 
	@echo -e "[i] linking: $@"
	$(CMDECHO)$(CXX) -o $@ $(LDFLAGS) $($(patsubst %.x,%_LIBS,$@)) $^

$(LOGS): %.log: %.x
	@echo -e "[i] running test: $^"
	$(CMDECHO)./$^ > $@ 2>&1

$(LOCAL_TEST): %.test: %.log
	$(TestDiff)

#=============================================================================#
