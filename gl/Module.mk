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
ifeq ($(ARCH),win32gdk)
GLS          += TRootGLKernel.cxx TRootWin32GLViewer.cxx
else
GLS          += TRootGLKernel.cxx TRootGLViewer.cxx
ifneq ($(IVROOT),)
GLS          += TRootOIViewer.cxx
IVLIBDIR     := -L$(IVROOT)/usr/lib
IVLIB        := -lInventor -lInventorXt -lXm -lXt -lXext -lX11
IVINCDIR     := $(IVROOT)/usr/include
endif
endif
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

ifneq ($(IVROOT),)
$(GLLIB):       $(GLO) $(MAINLIBS) $(GLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRGL.$(SOEXT) $@ "$(GLO)" \
		   "$(GLLIBEXTRA) $(IVLIBDIR) $(IVLIB)"
else
$(GLLIB):       $(GLO) $(MAINLIBS) $(GLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRGL.$(SOEXT) $@ "$(GLO)" "$(GLLIBEXTRA)"
endif

all-gl:         $(GLLIB)

clean-gl:
		@rm -f $(GLO)

clean::         clean-gl

distclean-gl:   clean-gl
		@rm -f $(GLDEP) $(GLLIB)

distclean::     distclean-gl

##### extra rules ######
ifneq ($(IVROOT),)
$(GLO): %.o: %.cxx
	$(CXX) $(OPT) -DR__OPENINVENTOR $(CXXFLAGS) -I$(OPENGLINCDIR) \
	   -I$(IVINCDIR) -o $@ -c $<
else
ifeq ($(ARCH),win32gdk)
$(GLO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(OPENGLINCDIR) -I$(WIN32GDKDIR)/gdk/inc \
	   -I$(WIN32GDKDIR)/gdk/inc/gdk -I$(WIN32GDKDIR)/gdk/inc/glib \
	   -o $@ -c $<
else
$(GLO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(OPENGLINCDIR) -o $@ -c $<
endif
endif

