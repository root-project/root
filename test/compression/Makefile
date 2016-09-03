CPPFLAGS=$(shell root-config --cflags) -march=native -g -Wextra -Wall -Wshadow
LDFLAGS=$(shell root-config --libs) -lstdc++ -m64 -g -march=native
CPP=c++
CXX=c++
CC=c++

parent.h: org.root
	root -l -b -q generate.C

parent.C: org.root
	root -l -b -q generate.C

read: read.o parent.o

readtime: readtime.o parent.o

read.o: parent.h
 
# http://stackoverflow.com/questions/1490949/how-to-write-loop-in-makefile
LAST_LEVEL := 9
LEVELS := $(shell seq 1 ${LAST_LEVEL})

ALGS := 1 2 4 5 6 7

CRJOBS   := $(foreach ALG,$(ALGS),$(addsuffix .out, $(addprefix callgrind-read.${ALG}.,${LEVELS})))
CWJOBS   := $(foreach ALG,$(ALGS),$(addsuffix .out, $(addprefix callgrind-write.${ALG}.,${LEVELS})))
MWJOBS   := $(foreach ALG,$(ALGS),$(addsuffix .out, $(addprefix massif-write.${ALG}.,${LEVELS})))
MRJOBS   := $(foreach ALG,$(ALGS),$(addsuffix .out, $(addprefix massif-read.${ALG}.,${LEVELS})))
RWJOBS   := $(foreach ALG,$(ALGS),$(addsuffix .log, $(addprefix realtime-write.${ALG}.,${LEVELS})))
RRJOBS   := $(foreach ALG,$(ALGS),$(addsuffix .log, $(addprefix realtime-read.${ALG}.,${LEVELS})))
SIZEJOBS := $(foreach ALG,$(ALGS),$(addsuffix .root,$(addprefix size.${ALG}.,${LEVELS})))

all: callgrind-write massif-write realtime-write size callgrind-read realtime-read massif-read
parallel-tests: callgrind-write massif-write size callgrind-read massif-read
sequential-tests: realtime-read realtime-write
callgrind-write: ${CWJOBS}
callgrind-read: ${CRJOBS}
massif-write: ${MWJOBS}
massif-read: ${MRJOBS}
realtime-write: ${RWJOBS}
realtime-read: ${RRJOBS}
size: ${SIZEJOBS}
exe: writetime read readtime todevnull convert

define RWJOBTEMPLATE
realtime-write.$(1).$(2).log: writetime
	./writetime $(1) $(2) > $$@
endef
define CRJOBTEMPLATE
callgrind-read.$(1).$(2).out: read size.$(1).$(2).root
	valgrind --callgrind-out-file=$$@ --tool=callgrind -v --dump-instr=yes --trace-jump=yes --smc-check=all-non-file ./read $(1) $(2)
endef
define RRJOBTEMPLATE
realtime-read.$(1).$(2).log: readtime size.$(1).$(2).root
	./readtime $(1) $(2) > $$@
endef
define CWJOBTEMPLATE
callgrind-write.$(1).$(2).out: todevnull
	valgrind --callgrind-out-file=$$@ --tool=callgrind -v --dump-instr=yes --trace-jump=yes --smc-check=all-non-file ./todevnull $(1) $(2)
endef
define MRJOBTEMPLATE
massif-read.$(1).$(2).out: read size.$(1).$(2).root
	valgrind --massif-out-file=$$@ --tool=massif --peak-inaccuracy=0.5 ./read $(1) $(2)
endef
define MWJOBTEMPLATE
massif-write.$(1).$(2).out: todevnull
	valgrind --massif-out-file=$$@ --tool=massif --peak-inaccuracy=0.5 ./todevnull $(1) $(2)
endef
define SIZETEMPLATE
size.$(1).$(2).root: convert
	./convert $(1) $(2)
endef
$(foreach ALG,$(ALGS),$(foreach LVL,${LEVELS},$(eval $(call CWJOBTEMPLATE,$(ALG),$(LVL)))))
$(foreach ALG,$(ALGS),$(foreach LVL,${LEVELS},$(eval $(call CRJOBTEMPLATE,$(ALG),$(LVL)))))
$(foreach ALG,$(ALGS),$(foreach LVL,${LEVELS},$(eval $(call MWJOBTEMPLATE,$(ALG),$(LVL)))))
$(foreach ALG,$(ALGS),$(foreach LVL,${LEVELS},$(eval $(call MRJOBTEMPLATE,$(ALG),$(LVL)))))
$(foreach ALG,$(ALGS),$(foreach LVL,${LEVELS},$(eval $(call RWJOBTEMPLATE,$(ALG),$(LVL)))))
$(foreach ALG,$(ALGS),$(foreach LVL,${LEVELS},$(eval $(call RRJOBTEMPLATE,$(ALG),$(LVL)))))
$(foreach ALG,$(ALGS),$(foreach LVL,${LEVELS},$(eval $(call SIZETEMPLATE,$(ALG),$(LVL)))))


.PHONY: callgrind-write massif-write realtime-write size callgrind-read realtime-read massif-read

clean:
	rm -f convert todevnull readtime writetime read

.DELETE_ON_ERROR:
