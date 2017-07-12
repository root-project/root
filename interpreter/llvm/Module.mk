# Module.mk for llvm module
# Copyright (c) 2012 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 6/8/2012

MODNAME      := llvm
MODDIR       := $(ROOT_SRCDIR)/interpreter/$(MODNAME)
LLVMDIRO     := $(call stripsrc,$(MODDIR)/obj)
LLVMDIRI     := $(call stripsrc,$(MODDIR)/inst)

ifneq ($(BUILTINLLVM),yes)

LLVMDEP      :=

.PHONY:         distclean-$(MODNAME)

distclean-$(MODNAME):
		@rm -rf $(LLVMDIRO) $(LLVMDIRI)

# Don't clean LLVM when doing standard "make distclean" as it is mostly not
# needed. To explicitly clean LLVM do either: "make maintainer-clean" or
# "make distclean-llvm".
#distclean::     distclean-$(MODNAME)
maintainer-clean:: distclean-$(MODNAME)

else

MODDIRS      := $(MODDIR)/src

LLVMDIR      := $(MODDIR)
LLVMDIRS     := $(MODDIRS)

##### libllvm #####
LLVMLIB      := $(LLVMDIRI)/lib/libclangSema.a
LLVMDEPO     := $(LLVMDIRO)/CMakeCache.txt
LLVMDEPS     := $(LLVMDIRS)/CMakeLists.txt
ifeq ($(strip $(LLVMCONFIG)),)
LLVMCONFIG   := interpreter/llvm/inst/bin/llvm-config
endif
LLVMGOODS    := $(MODDIRS)/../../cling/LastKnownGoodLLVMSVNRevision.txt
LLVMGOODO    := $(LLVMDIRO)/$(notdir $(LLVMGOODS))
define llvm_version_part
$(shell grep 'set.LLVM_VERSION_$1' $(LLVMDIRS)/CMakeLists.txt | sed 's,.*LLVM_VERSION_$1 ,,' | sed 's,.$$,,')
endef
LLVMVERSION  := $(call llvm_version_part,MAJOR).$(call llvm_version_part,MINOR).$(call llvm_version_part,PATCH)
LLVMRES      := etc/cling/lib/clang/$(LLVMVERSION)/include/stddef.h
LLVMRESEXTRA := $(addprefix etc/cling/lib/clang/$(LLVMVERSION)/include/, assert.h stdlib.h unistd.h)
LLVMSYSEXTRA := $(wildcard $(addprefix /usr/include/, wchar.h bits/stat.h bits/time.h))
LLVMSYSEXTRA := $(patsubst /usr/include/%,etc/cling/lib/clang/$(LLVMVERSION)/include/%,$(LLVMSYSEXTRA))
LLVMDEP      := $(LLVMLIB) $(LLVMRES) $(LLVMRESEXTRA) $(LLVMSYSEXTRA)

ROOT_NOCLANG := "ROOT_NOCLANG=yes"
ifeq ($(LLVMDEV),)
LLVMOPTFLAGS := -DCMAKE_BUILD_TYPE=Release
else
ifeq (,$(findstring debug,$(ROOTBUILD)))
LLVMOPTFLAGS := -DCMAKE_BUILD_TYPE=RelWithDebInfo
else
LLVMOPTFLAGS := -DCMAKE_BUILD_TYPE=Debug
endif
endif

FORCELLVM := 0
ifeq ($(findstring $(MAKECMDGOALS),clean distclean maintainer-clean dist distsrc),)
ifeq ($(findstring clean-,$(MAKECMDGOALS)),)
ifeq ($(shell which git 2>&1 | sed -ne "s@.*/git@git@p"),git)
FORCELLVM := $(shell bash $(ROOT_SRCDIR)/build/unix/gitinfollvm.sh $(ROOT_SRCDIR))
ENDLLVMBUILD := && cd ../../.. && ( $(ROOT_SRCDIR)/build/unix/gitinfollvm.sh $(ROOT_SRCDIR) > /dev/null )
endif
endif
endif

ifneq ($(FORCELLVM),0)
FORCELLVMTARGET := FORCELLVMTARGET
FORCENEXTLLVM := $(shell rm -f $(LLVMDIRO)/llvmrev.txt)
endif

##### local rules #####
.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) FORCELLVMTARGET

# clang resource directory gets copied to lib/clang/
# clang version extraction as in tools/clang/lib/Headers/Makefile
ifeq ($(LLVMCONFIG),interpreter/llvm/inst/bin/llvm-config)
$(LLVMCONFIG): $(LLVMLIB)
endif

$(LLVMRES): $(LLVMLIB)
		@mkdir -p $(dir $(LLVMRES))
		@cp -r $(LLVMDIRI)/lib/clang/$(LLVMVERSION)/include/* $(dir $(LLVMRES))

$(LLVMRESEXTRA): $(dir $(firstword $(LLVMRESEXTRA)))%: $(MODDIR)/ROOT/%
		@mkdir -p $(dir $@)
		@cp $< $@
$(LLVMSYSEXTRA): $(dir $(firstword $(LLVMSYSEXTRA)))%: /usr/include/%
		@mkdir -p $(dir $@)
		@cp $< $@

$(LLVMLIB): $(LLVMDEPO) $(FORCELLVMTARGET)
		@(echo "*** Building $@..."; \
		cd $(LLVMDIRO) && \
		$(MAKE) && \
		rm -rf ../inst/lib/clang && \
		$(MAKE) install \
                $(ENDLLVMBUILD) )

$(LLVMGOODO): $(LLVMGOODS) $(LLVMLIB)
		@cp $(LLVMGOODS) $(LLVMGOODO)

ifeq ($(CXX14),yes)
LLVM_CXX_VERSION=-DLLVM_ENABLE_CXX1Y=ON
endif

$(LLVMDEPO): $(LLVMDEPS)
		$(MAKEDIR)
		@(LLVMCC="$(CC)" && \
		LLVMCXX="$(CXX)" && \
		if [ $(ARCH) = "aix5" ]; then \
			LLVM_CFLAGS="-DBOOL=int"; \
		fi; \
		if [ $(ARCH) = "linux" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=ON"; \
			LLVM_CFLAGS="-m32 -fvisibility=hidden"; \
		fi; \
		if [ $(ARCH) = "linuxx8664gcc" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=OFF"; \
			LLVM_CFLAGS="-fvisibility=hidden"; \
		fi; \
		if [ $(ARCH) = "linuxicc" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=ON"; \
			LLVM_CFLAGS="-fvisibility=hidden"; \
		fi; \
		if [ $(ARCH) = "linuxx8664icc" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=OFF"; \
			LLVM_CFLAGS="-fvisibility=hidden"; \
		fi; \
		if [ $(ARCH) = "macosx" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=ON"; \
			LLVM_CFLAGS="-Wno-unused-private-field -fvisibility=hidden"; \
		fi; \
		if [ $(ARCH) = "macosx64" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=OFF"; \
			LLVM_CFLAGS="-Wno-unused-private-field -fvisibility=hidden"; \
		fi; \
		if [ $(ARCH) = "macosx64" -a x$(GCC_MAJOR) != "x" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=OFF"; \
			LLVM_CFLAGS="$$LLVM_CFLAGS -fno-omit-frame-pointer -fvisibility=hidden"; \
		fi; \
		if [ $(ARCH) = "iossim" ]; then \
			LLVM_CFLAGS="-arch i386 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LLVM_HOST="-DLLVM_HOST_TRIPLE=i386-apple-darwin"; \
		fi; \
		if [ $(ARCH) = "ios" ]; then \
			LLVM_CFLAGS="-arch armv7 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LLVM_HOST="-DLLVM_HOST_TRIPLE=armv7-apple-darwin"; \
			LLVM_TARGET="-DLLVM_TARGET_ARCH=armv7-apple-darwin"; \
		fi; \
		if [ $(ARCH) = "solaris64CC5" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=OFF"; \
		fi; \
		if [ $(ARCH) = "linuxppc64gcc" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=OFF"; \
		fi; \
		if [ $(ARCH) = "linuxppcgcc" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=ON"; \
		fi; \
		if [ $(ARCH) = "hpuxia64acc" ]; then \
			LLVM_32BITS="-DLLVM_BUILD_32_BITS=OFF"; \
			LLVMCC="cc"; \
			LLVMCXX="aCC"; \
			LLVM_CFLAGS="+DD64 -Ae"; \
		fi; \
		if [ $(LIBCXX) = "yes" ]; then \
			LLVMLIBCXX="-DLLVM_ENABLE_LIBCXX=ON"; \
		fi; \
		if [ $(GCCTOOLCHAIN) ]; then \
			LLVM_CFLAGS="$(LLVM_CFLAGS) --gcc-toolchain=$(GCCTOOLCHAIN) "; \
		fi; \
		if [ $(CXXMODULES) = "yes" ]; then \
			LLVM_CXXMODULES=" -DLLVM_ENABLE_MODULES=ON "; \
		fi; \
		echo "*** Configuring LLVM in $(dir $@) ..."; \
		mkdir -p $(dir $@) && \
		cd $(dir $@)  && \
		unset LDFLAGS && \
		cmake $(LLVM_CXX_VERSION) \
		$$LLVM_HOST \
		$$LLVM_TARGET \
		-DCMAKE_INSTALL_PREFIX=$(ROOT_OBJDIR)/$(LLVMDIRI) \
		-DLLVM_BUILD_DOCS=OFF \
		-DLLVM_BUILD_TESTS=OFF \
		-DLLVM_ENABLE_WARNINGS=OFF \
		-DLLVM_INCLUDE_TESTS=OFF \
		-DLLVM_INCLUDE_EXAMPLES=OFF \
		-DLLVM_FORCE_USE_OLD_TOOLCHAIN=ON \
		-DLLVM_ENABLE_THREADS=OFF \
		-DCLANG_ENABLE_STATIC_ANALYZER=OFF \
		-DCLANG_ENABLE_ARCMT=OFF \
		-DCLANG_ENABLE_COMPILER=OFF \
		-DCLANG_ENABLE_FORMAT=OFF \
		-DCLANG_TOOL_C_INDEX_TEST_BUILD=OFF \
		-DCLANG_TOOL_LIBCLANG_BUILD=OFF \
		-DCLANG_TOOL_DRIVER_BUILD=OFF \
		-DCLANG_INCLUDE_TESTS=OFF \
		-DCLANG_TOOL_CLANG_OFFLOAD_BUNDLER_BUILD=OFF \
      -DCLANG_BUILD_TOOLS=OFF \
      -DLLVM_TOOL_LLVM_AR_BUILD=OFF \
      -DLLVM_INCLUDE_TOOLS=ON \
		$$LLVMLIBCXX \
		$(LLVMOPTFLAGS) \
		$$LLVM_32BITS \
		-DLLVM_TARGETS_TO_BUILD=host \
		$$LLVM_GCC_TOOLCHAIN \
		$$LLVM_CXXMODULES \
		-DCMAKE_C_COMPILER=$$LLVMCC \
                -DCMAKE_CXX_COMPILER=$$LLVMCXX \
		-DCMAKE_CXX_FLAGS="$$LLVM_CFLAGS" \
		-DCMAKE_C_FLAGS="$$LLVM_CFLAGS" \
		$(LLVMDIRS) )

all-$(MODNAME): $(LLVMLIB)

clean-$(MODNAME):
		-@(if [ -d $(LLVMDIRO) ]; then \
			cd $(LLVMDIRO); \
			$(MAKE) clean; \
		fi)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -rf $(LLVMDIRO) $(LLVMDIRI)

# Don't clean LLVM when doing standard "make distclean" as it is mostly not
# needed. To explicitly clean LLVM do either: "make maintainer-clean" or
# "make distclean-llvm".
#distclean::     distclean-$(MODNAME)
maintainer-clean:: distclean-$(MODNAME)

endif
