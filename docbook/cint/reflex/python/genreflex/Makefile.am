
AUTOMAKE_OPTIONS = foreign

EXTRA_DIST =       \
  gccdemangler.py  \
  gencapa.py       \
  gendict.py       \
  genheader.py     \
  genreflex.py     \
  genreflex-rootcint.py     \
  genrootmap.py             \
  selclass.py

pkgpython_PYTHON = \
  gccdemangler.py  \
  gencapa.py       \
  gendict.py       \
  genheader.py     \
  genreflex.py     \
  genreflex-rootcint.py     \
  genrootmap.py             \
  selclass.py

if PLATFORM_IS_WINDOWS 
SCRIPT_TXT_L1 = "@echo off"
SCRIPT_TXT_L2 = "python $(pkgpyexecdir)/genreflex.py %*"
SCRIPT_NAME   = "$(bindir)/genreflex.bat"
else 
SCRIPT_TXT_L1 = "\#!/bin/csh -f"
SCRIPT_TXT_L2 = "python $(pkgpyexecdir)/genreflex.py $$""*"
SCRIPT_NAME   = "$(bindir)/genreflex"
endif

install-exec-am: 
	if !(test -d $(bindir)); then mkdir $(bindir); fi
	if (test -f $(SCRIPT_NAME)); then rm -f $(SCRIPT_NAME); fi
	echo $(SCRIPT_TXT_L1) >  $(SCRIPT_NAME)
	echo $(SCRIPT_TXT_L2) >> $(SCRIPT_NAME)
	chmod a+x $(SCRIPT_NAME)