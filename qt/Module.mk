# $Id: Module.mk,v 1.10 2004/06/14 03:31:57 fine Exp $
# Module.mk for qt module
# Copyright (c) 2001 Valeri Fine
#
# Author: Valeri Fine, 21/10/2001

ifeq ($(ARCH),win32old)
include qt/Module.mk.win32
else
include qt/Module.mk.unix
endif
