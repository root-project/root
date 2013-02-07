##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ..

include $(CLANG_LEVEL)/../../Makefile.config

ifneq ($(ROOT_BUILDCLANG),)
DIRS := driver libclang c-index-test arcmt-test c-arcmt-test diagtool \
        clang-check
endif

# Recurse into the extra repository of tools if present.
OPTIONAL_DIRS := extra

include $(CLANG_LEVEL)/Makefile
