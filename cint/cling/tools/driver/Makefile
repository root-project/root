##===- tools/driver/Makefile -------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../../..

TOOLNAME = cling
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include \
-I$(PROJ_SRC_DIR)/../../../clang/include -I$(PROJ_SRC_DIR)/../../../clang/lib \
-I$(PROJ_OBJ_DIR)/../../../clang/include
CXXFLAGS = -fno-rtti

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(LEVEL)/Makefile.common

LIBS += $(LibDir)/libcling$(SHLIBEXT)
