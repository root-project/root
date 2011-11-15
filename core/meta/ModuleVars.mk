ifeq ($(BUILDCLING),yes)
#MAKECLINGPCH := $(ROOT_SRCDIR)/build/unix/makeclingpch.sh
#ROOTCINTTMP  := $(MAKECLINGPCH) "$(LLVMCONFIG:llvm-config=clang++) $(CXXFLAGS:-fPIC=)" $(call stripsrc,$(UTILSDIRS)/rootcint_tmp$(EXEEXT))
ifeq ($(ROOT_REVERT_TO_ROOTCINT),)
ROOTCINTTMP := $(ROOTCLINGTMP)
else
ROOTCINTTMP := $(ROOTCINTTMP)
endif
endif
