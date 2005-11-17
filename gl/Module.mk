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

GLH          := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GLS          := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GLS1         := $(wildcard $(MODDIRS)/*.c)

# Excluded from win32 builds
ifeq ($(ARCH),win32)
GLS          := $(filter-out $(MODDIRS)/TX11GL.cxx, $(GLS))
GLH          := $(filter-out $(MODDIRI)/TX11GL.h, $(GLH))
endif

# Excluded from rootcint
GLH1         := $(MODDIRI)/gl2ps.h $(MODDIRI)/CsgOps.h $(MODDIRI)/TGLKernel.h \
                $(MODDIRI)/TGLIncludes.h $(MODDIRI)/TRootGLU.h \
                $(MODDIRI)/TRootGLX.h
# Used by rootcint
GLH2         := $(filter-out $(GLH1), $(GLH))

ifneq ($(OPENGLLIB),)
GLLIBS       := $(OPENGLLIBDIR) $(OPENGLULIB) $(OPENGLLIB) \
                $(X11LIBDIR) -lX11 -lXext -lXmu -lXi -lm
endif
ifeq ($(ARCH),win32)
GLLIBS       := opengl32.lib glu32.lib $(LPATH)/libGraf.lib $(LPATH)/libHist.lib
endif

GLO          := $(GLS:.cxx=.o)
GLO1         := $(GLS1:.c=.o)

GLDEP        := $(GLO:.o=.d) $(GLDO:.o=.d) $(GLO1:.o=.d)

GLLIB        := $(LPATH)/libRGL.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GLH))
ALLLIBS      += $(GLLIB)

# include all dependency files
INCLUDEFILES += $(GLDEP)

##### local rules #####
include/%.h:    $(GLDIRI)/%.h
		cp $< $@

$(GLLIB):       $(GLO) $(GLO1) $(GLDO) $(MAINLIBS) $(GLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRGL.$(SOEXT) $@ "$(GLO) $(GLO1) $(GLDO)" \
		   "$(GLLIBEXTRA) $(GLLIBS)"

$(GLDS):	$(GLH2) $(GLL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GLH2) $(GLL)

ifeq ($(ARCH),win32)
$(GLDO):        $(GLDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -I$(WIN32GDKDIR)/gdk/src \
		   $(GDKDIRI:%=-I%) $(GLIBDIRI:%=-I%) -o $@ -c $<
else
$(GLDO):        $(GLDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. $(OPENGLINCDIR:%=-I%) -o $@ -c $<
endif

all-gl:         $(GLLIB)

map-gl:         $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GLLIB) -d $(GLLIBDEP) -c $(GLL)

map::           map-gl

clean-gl:
		@rm -f $(GLO) $(GLO1) $(GLDO)

clean::         clean-gl

distclean-gl:   clean-gl
		@rm -f $(GLDEP) $(GLLIB) $(GLDS) $(GLDH)

distclean::     distclean-gl

##### extra rules ######
ifeq ($(ARCH),win32)
$(GLO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(OPENGLINCDIR:%=-I%) -I$(WIN32GDKDIR)/gdk/src \
	   $(GDKDIRI:%=-I%) $(GLIBDIRI:%=-I%) -o $@ -c $<
else
$(GLO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(OPENGLINCDIR:%=-I%) -o $@ -c $<
endif

$(GLDIRS)/gl2ps.o: $(GLDIRS)/gl2ps.c
	$(CC) $(OPT) $(CFLAGS) $(OPENGLINCDIR:%=-I%) -o $@ -c $<
