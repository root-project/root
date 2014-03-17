CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* $(ALL_LIBRARIES) *rootmap newFile.xml exampleTrimmed.xml newFileTrimmed.xml
TEST_TARGETS += execionameiotype testNewFileDiff

ifeq ($(strip $(ROOTTEST_HOME)),)
        export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

GENREFLEXCXXFLAGS += --rootmap autoload.rootmap --rootmap-lib libclasses_dictrflx.so


execionameiotype.log: libclasses_dictrflx.$(DllSuf)

newFileTrimmed: execionameiotype.log
	$(CMDECHO)cat newFile.xml |grep -v created| grep -v XmlKey |grep -v "TObject fUniqueID" >$@.xml 2>&1

exampleTrimmed: execionameiotype.log
	$(CMDECHO) cat example.xml |grep -v created| grep -v XmlKey |grep -v "TObject fUniqueID" >$@.xml 2>&1

newDiffFile.log: newFileTrimmed exampleTrimmed
	$(CMDECHO) diff newFileTrimmed.xml exampleTrimmed.xml >$@ ; :
	

testNewFileDiff: newDiffFile.success

newDiffFile.success: newDiffFile.log
	$(SuccessTestDiff) && touch $@
