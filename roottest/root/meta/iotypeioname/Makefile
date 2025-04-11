CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* $(ALL_LIBRARIES) *rootmap newFile.xml exampleTrimmed.xml newFileTrimmed.xml
TEST_TARGETS += execionameiotype testNewFileDiff

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

GENREFLEXCXXFLAGS += --rootmap autoload.rootmap --rootmap-lib libclasses_dictrflx.so


execionameiotype.log: libclasses_dictrflx.$(DllSuf)

newFileTrimmed: execionameiotype.log
	$(CMDECHO)cat newFile.xml |grep -v created| grep -v XmlKey |grep -v "TObject fUniqueID" | sed 's,size="[^"]*",size="masked",g' >$@.xml 2>&1

exampleTrimmed: execionameiotype.log
	$(CMDECHO) cat example.xml |grep -v created| grep -v XmlKey |grep -v "TObject fUniqueID" | sed 's,size="[^"]*",size="masked",g' >$@.xml 2>&1

# Here we use true since diff returns 1 if the files are different. Here we expect them to be different!
newDiffFile.log: newFileTrimmed exampleTrimmed
	$(CMDECHO) diff newFileTrimmed.xml exampleTrimmed.xml >$@ ; true

testNewFileDiff: newDiffFile.success

newDiffFile.success: newDiffFile.log
	$(SuccessTestDiff) && touch $@
