# Module.mk for python module
# Copyright (c) 2015 Rene Brun and Fons Rademakers
#
# Authors: Fons Rademakers, 22/10/2015

MODNAME      := python
MODDIR       := $(ROOT_SRCDIR)/main/$(MODNAME)

CMDTOOLSDIR  := $(MODDIR)

##### bin/root*.py command line utilities #####
CMDTOOLSPYS  := $(wildcard $(MODDIR)/root*.py)
CMDTOOLSPY   := $(subst $(MODDIR),bin,$(CMDTOOLSPYS))
CMDTOOLSPY   := $(subst .py,,$(CMDTOOLSPY))

CMDUTILPYS   := $(wildcard $(MODDIR)/cmdLine*.py)
CMDUTILPY    := $(subst $(MODDIR),$(LPATH),$(CMDUTILPYS))

# compilation defined in bindings/pyroot/Module.mk
CMDUTILPYC   := $(CMDUTILPY:.py=.pyc)
CMDUTILPYO   := $(CMDUTILPY:.py=.pyo)

# used in the main Makefile
ALLLIBS      += $(CMDUTILPYC) $(CMDUTILPYO)
ALLEXECS     += $(CMDTOOLSPY)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

bin/%:          $(CMDTOOLSDIR)/%.py
		cp $< $@

$(LPATH)/%.py:  $(CMDTOOLSDIR)/%.py
		cp $< $@

#%.pyc: %.py;    python -c 'import py_compile; py_compile.compile( "$<" )'
#%.pyo: %.py;    python -O -c 'import py_compile; py_compile.compile( "$<" )'

$(CMDUTILPYC) $(CMDUTILPYO): $(CMDUTILPY)

all-$(MODNAME): $(CMDTOOLSPY) $(CMDUTILPYC) $(CMDUTILPYO)

clean-$(MODNAME):

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CMDTOOLSPY) $(CMDUTILPY) $(CMDUTILPYC) $(CMDUTILPYO)

distclean::     distclean-$(MODNAME)

##### extra rules ######
