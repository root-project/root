all: tests
test: tests

SUBDIRS =  

TEST_TARGETS = $(SUBDIRS:%=%.test) 

tests: $(TEST_TARGETS) ArraySyntax
	@echo "All test succeeded in `pwd`"

$(TEST_TARGETS): %.test:
	@(cd $*; gmake test)

#ArraySyntax: Event.root
#	@echo built

ArraySyntax: main $(ROOTSYS)/lib/libTreePlayer.so
	./main

main: main.C fooDict.C barDict.C foo.C bar.C
	g++ -o $@ $^ -I. `root-config --cflags --libs`

%Dict.C: %.h
	rootcint -f $@ -c $^

clean:
	rm -f main *Dict* Event.root *~
