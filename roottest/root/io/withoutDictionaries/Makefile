CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *pcm *o *rflx.* dummy* *root

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


execRowWiseNoProxyWarning.log: libRowWiseNoProxyWarning_dictrflx.$(DllSuf)
execIOWithoutDictionaries.log: libIOWithoutDictionaries_dictrflx.$(DllSuf)
execVectorDMWriteWithoutDictionary.log: libVectorDMWriteWithoutDictionary_dictrflx.$(DllSuf)

libRowWiseNoProxyWarning_dictrflx.$(DllSuf): \
   GENREFLEXFLAGS += --rootmap=libRowWiseNoProxyWarning_dictrflx.rootmap \
      --rootmap-lib=libRowWiseNoProxyWarning_dictrflx.$(DllSuf)


libIOWithoutDictionaries_dictrflx.$(DllSuf): \
   GENREFLEXFLAGS += --rootmap=libIOWithoutDictionaries_dictrflx.rootmap \
       --rootmap-lib=libIOWithoutDictionaries_dictrflx.$(DllSuf)

libVectorDMWriteWithoutDictionary_dictrflx.$(DllSuf): \
   GENREFLEXFLAGS += --rootmap=libVectorDMWriteWithoutDictionary_dictrflx.rootmap \
       --rootmap-lib=libVectorDMWriteWithoutDictionary_dictrflx.$(DllSuf)
