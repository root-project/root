# Module.mk for gl module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := gl
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GLDIR        := $(MODDIR)
GLDIRS       := $(GLDIR)/src
GLDIRI       := $(GLDIR)/inc

##### libRGL #####
GLH          := $(wildcard $(MODDIRI)/*.h)
GLS          := TGLKernel.cxx
ifeq ($(ARCH),win32)
GLS          += TWin32GLKernel.cxx TWin32GLViewerImp.cxx
else
GLS          += TRootGLKernel.cxx TRootGLViewer.cxx
endif
GLS          := $(patsubst %,$(MODDIRS)/%,$(GLS))

GLO          := $(GLS:.cxx=.o)

GLDEP        := $(GLO:.o=.d)

GLLIB        := $(LPATH)/libRGL.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GLH))
ALLLIBS     += $(GLLIB)

# include all dependency files
INCLUDEFILES += $(GLDEP)

##### local rules #####
include/%.h:    $(GLDIRI)/%.h
		cp $< $@

ifeq ($(ARCH),win32)
$(GLLIB):       $(GLO) $(MAINLIBS) $(G3DLIB) $(WIN32LIB)
else
$(GLLIB):       $(GLO) $(MAINLIBS)
endif
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRGL.$(SOEXT) $@ "$(GLO)" "$(GLLIBEXTRA)"

all-gl:         $(GLLIB)

clean-gl:
		@rm -f $(GLO)

clean::         clean-gl

distclean-gl:   clean-gl
		@rm -f $(GLDEP) $(GLLIB)

distclean::     distclean-gl

##### extra rules ######
$(GLO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(OPENGLINCDIR) -o $@ -c $<
