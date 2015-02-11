##===- unittests/Makefile ----------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

ifeq ($(DISABLE_UNITTESTS_FOR_ROOT),DISABLE_UNITTESTS_FOR_ROOT)

# If CLANG_LEVEL is not set, then we are the top-level Makefile. Otherwise, we
# are being included from a subdirectory makefile.

ifndef CLANG_LEVEL

IS_UNITTEST_LEVEL := 1
CLANG_LEVEL := ..
PARALLEL_DIRS = CodeGen Basic Lex Driver Format ASTMatchers AST Tooling \
	        Sema

include $(CLANG_LEVEL)/../..//Makefile.config

ifeq ($(ENABLE_CLANG_ARCMT),1)
PARALLEL_DIRS += Frontend libclang
endif

# If ENABLE_CLANG_ARCMT=0 we don't build libclang, thus we don't have this
# target available.
ifeq ($(ENABLE_CLANG_ARCMT),0)
PARALLEL_DIRS := $(filter-out libclang, $(PARALLEL_DIRS))
endif

endif  # CLANG_LEVEL

include $(CLANG_LEVEL)/Makefile

ifndef IS_UNITTEST_LEVEL

MAKEFILE_UNITTEST_NO_INCLUDE_COMMON := 1
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest

endif  # IS_UNITTEST_LEVEL

else # DISABLE_UNITTESTS_FOR_ROOT
all:
clean:
install:
endif # DISABLE_UNITTESTS_FOR_ROOT
