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
GLL          := $(MODDIRI)/LinkDef.h
GLDS         := $(MODDIRS)/G__GL.cxx
GLDO         := $(GLDS:.cxx=.o)
GLDH         := $(GLDS:.cxx=.h)
GLH          := $(wildcard $(MODDIRI)/*.h)
GLH1         := $(MODDIRI)/TViewerOpenGL.h

GLS          := TGLKernel.cxx TArcBall.cxx

ifeq ($(ARCH),win32old)
GLS          += TWin32GLKernel.cxx TWin32GLViewerImp.cxx
else

ifneq ($(OPENGLLIB),)
GLLIBS       := $(OPENGLLIBDIR) $(OPENGLULIB) $(OPENGLLIB) \
                $(X11LIBDIR) -lX11 -lXext -lXmu -lXi -lm
endif
ifneq ($(OPENIVLIB),)
GLS          += TRootOIViewer.cxx
IVFLAGS      := -DR__OPENINVENTOR -I$(OPENIVINCDIR)
IVLIBS       := $(OPENIVLIBDIR) $(OPENIVLIB) \
                $(X11LIBDIR) -lXm -lXt -lXext -lX11 -lm
endif
endif

ifeq ($(ARCH),win32)
GLLIBS       += opengl32.lib glu32.lib
endif

GLS          += TViewerOpenGL.cxx
GLS          := $(patsubst %,$(MODDIRS)/%,$(GLS))

GLO          := $(GLS:.cxx=.o)

GLDEP        := $(GLO:.o=.d)

GLLIB        := $(LPATH)/libRGL.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GLH))
ALLLIBS      += $(GLLIB)

# include all dependency files
INCLUDEFILES += $(GLDEP)

##### local rules #####
include/%.h:    $(GLDIRI)/%.h
		cp $< $@

$(GLLIB):       $(GLO) $(GLDO) $(MAINLIBS) $(GLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRGL.$(SOEXT) $@ "$(GLO) $(GLDO)" \
		   "$(GLLIBEXTRA) $(GLLIBS) $(IVLIBS)"

$(GLDS):	$(GLH1) $(GLL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GLH1) $(GLL)

ifeq ($(ARCH),win32)
$(GLDO):        $(GLDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -I$(WIN32GDKDIR)/gdk/src \
		   -I$(GDKDIRI) -I$(GLIBDIRI) -o $@ -c $<
else
$(GLDO):        $(GLDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<
endif

all-gl:         $(GLLIB)

map-gl:         $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GLLIB) -d $(GLLIBDEP) -c $(GLL)

map::           map-gl

clean-gl:
		@rm -f $(GLO) $(GLDO)

clean::         clean-gl

distclean-gl:   clean-gl
		@rm -f $(GLDEP) $(GLLIB) $(GLDS) $(GLDH)

distclean::     distclean-gl

##### extra rules ######
ifeq ($(ARCH),win32)
$(GLO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(OPENGLINCDIR) -I$(WIN32GDKDIR)/gdk/src \
	   -I$(GDKDIRI) -I$(GLIBDIRI) -o $@ -c $<
else
$(GLO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(OPENGLINCDIR) $(IVFLAGS) -o $@ -c $<
endif

