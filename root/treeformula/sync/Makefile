CLEAN_TARGETS += main *Dict* Event.root *~ Event$(ExeSuf) createfile.out readfile.out libEvent.$(DllSuf) Event.h $(ALL_LIBRARIES)
TEST_TARGETS += Simple

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


$(ROOT_LOC)/test/Event$(ExeSuf):
	(cd $(ROOT_LOC)/test; $(MAKE) Event$(ExeSuf))

libEvent.$(DllSuf): $(ROOT_LOC)/test/libEvent.$(DllSuf)
	$(CMDECHO) cp $(ROOT_LOC)/test/Event$(ExeSuf) $(ROOT_LOC)/test/libEvent.* $(ROOT_LOC)/test/Event.h .

Event$(ExeSuf): $(ROOT_LOC)/test/Event$(ExeSuf)
	$(CMDECHO)cp $(ROOT_LOC)/test/Event$(ExeSuf) $(ROOT_LOC)/test/libEvent.* $(ROOT_LOC)/test/Event.h .

Event.root: ./Event$(ExeSuf) ./libEvent.$(DllSuf)
	$(CMDECHO) ./Event 20 1 9 1 > createfile.out

Simple: Event.root
ifeq ($(FAIL),)
	$(WarnFailTest)
	$(CMDECHO) root.exe -l -b -q 'Run.C(1)' > readfile.out
else
	$(CMDECHO) root.exe -l -b -q 'Run.C(0)' > readfile.out
endif
