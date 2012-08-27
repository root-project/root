# Module.mk for llvm module
# Copyright (c) 2012 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 6/8/2012

MODNAME      := llvm
MODDIR       := $(ROOT_SRCDIR)/interpreter/$(MODNAME)
LLVMDIRO     := $(call stripsrc,$(MODDIR)/obj)
LLVMDIRI     := $(call stripsrc,$(MODDIR)/inst)

ifneq ($(BUILTINLLVM), yes)

LLVMDEP      :=

.PHONY:         distclean-$(MODNAME)
distclean-$(MODNAME):
		@rm -rf $(LLVMDIRO) $(LLVMDIRI)
distclean::     distclean-$(MODNAME)

else

MODDIRS      := $(MODDIR)/src

LLVMDIR      := $(MODDIR)
LLVMDIRS     := $(MODDIRS)

##### libllvm #####
LLVMLIB      := $(LLVMDIRI)/lib/libclang.a
LLVMCONFIG   := $(LLVMDIRI)/bin/llvm-config
LLVMDEP      := $(LLVMLIB)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

$(LLVMLIB): $(LLVMDIRO)
		@(echo "*** Building $@..."; \
		cd $(LLVMDIRO); \
		$(MAKE) ENABLE_OPTIMIZED=1; \
		$(MAKE) ENABLE_OPTIMIZED=1 install)

$(LLVMDIRO): $(LLVMDIRS)
		$(MAKEDIR)
		@(cd $(dir $@); \
		mkdir $(notdir $@); \
		echo "*** Configuring LLVM in $@..."; \
		cd $(notdir $@); \
		LLVMCC="$(CC)"; \
		LLVMCXX="$(CXX)"; \
		if [ $(ARCH) = "aix5" ]; then \
			LLVM_CFLAGS="-DBOOL=int"; \
		fi; \
		if [ $(ARCH) = "linux" ]; then \
			LLVM_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664gcc" ]; then \
			LLVM_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxicc" ]; then \
			LLVM_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664icc" ]; then \
			LLVM_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "macosx" ]; then \
			LLVM_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "macosx64" ]; then \
			LLVM_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "iossim" ]; then \
			LLVM_CFLAGS="-arch i386 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LLVM_HOST="--host=i686-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "ios" ]; then \
			LLVM_CFLAGS="-arch armv7 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LLVM_HOST="--host=arm-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "solaris64CC5" ]; then \
			LLVM_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxppc64gcc" ]; then \
			LLVM_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "hpuxia64acc" ]; then \
			LLVMCC="cc"; \
			LLVMCXX="aCC"; \
			LLVM_CFLAGS="+DD64 -Ae"; \
		fi; \
		GNUMAKE=$(MAKE) $(LLVMDIRS)/configure $$LLVM_HOST \
		--prefix=$(ROOT_OBJDIR)/$(LLVMDIRI) \
		--enable-targets=host CC=$$LLVMCC CXX=$LLVMCXX CFLAGS="$$LLVM_CFLAGS" CXXFLAGS="$$LLVM_CFLAGS")

all-$(MODNAME): $(LLVMLIB)

clean-llvm:
		-@(if [ -d $(LLVMDIRO) ]; then \
			cd $(LLVMDIRO); \
			$(MAKE) clean; \
		fi)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -rf $(LLVMDIRO) $(LLVMDIRI)

distclean::     distclean-$(MODNAME)

endif
