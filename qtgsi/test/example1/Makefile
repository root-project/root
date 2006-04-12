#############################################################################
# Qt ROOT  example Makefile
# Author: D.Bertini <D.Bertini@gsi.de>
# Date:06/29/01
#############################################################################

MAKEDEP = rmkdepend

QT_CXXFLAGS_OPT = -DQT_BUILTIN_GIF_READER  -I$(QTDIR)/src/3rdparty/zlib -I$(QTDIR)/src/3rdparty/libpng
QT_CFLAGS_OPT   = -I$(QTDIR)/src/3rdparty/zlib -I$(QTDIR)/src/3rdparty/libpng
QT_LIBS_OPT	=  -lpthread 
QT_PNG_OBJ      = $(PNG_OBJECTS)
QT_ZLIB_OBJ	= $(ZLIB_OBJECTS)

# Compiling
SYSCONF_CXX		= g++
SYSCONF_CC		= gcc

#debuggeng
QDL			=0

# Compiling with support libraries
SYSCONF_CXXFLAGS_X11	= -I/usr/X11R6/include
SYSCONF_CXXFLAGS_QT	= -I$(QTDIR)/include
SYSCONF_CXXFLAGS_OPENGL	= -I/usr/X11R6/include

# Compiline YACC output
SYSCONF_CXXFLAGS_YACC     = -Wno-unused -Wno-parentheses -Wno-deprecated
# Linking with support libraries
# X11
SYSCONF_LFLAGS_X11	= -L/usr/X11R6/lib
SYSCONF_LIBS_X11	= -lXext -lX11 -lm
# Qt, Qt+OpenGL
SYSCONF_LFLAGS_QT	= -L$(QTDIR)/lib
SYSCONF_LIBS_QT		= $(QTLINKLIB)
SYSCONF_LIBS_QT_OPENGL	= -lqgl
# OpenGL
SYSCONF_LFLAGS_OPENGL	= -L/usr/X11R6/lib
SYSCONF_LIBS_OPENGL	= -lGLU -lGL -lXmu
# Yacc
SYSCONF_LIBS_YACC	= 

# Linking applications
SYSCONF_LINK		= g++
SYSCONF_LIBS		=

# Link flags shared objects
SYSCONF_LFLAGS_SHOBJ	= -shared

# Meta-object compiler
SYSCONF_MOC		= $(QTDIR)/bin/moc

# Linking shared libraries
#   - Build the $(TARGET) library, eg. lib$(TARGET).so.2.0.1
#   - Place target in $(DESTDIR) - which has a trailing /
#   - Usually needs to incorporate $(VER_MAJ), $(VER_MIN) and $(VER_PATCH)
#

SYSCONF_LINK_SHLIB	= g++
SYSCONF_LINK_TARGET_SHARED	= lib$(TARGET).so.$(VER_MAJ).$(VER_MIN).$(VER_PATCH)

SYSCONF_LINK_LIB_SHARED	=  $(SYSCONF_LINK_SHLIB) -shared -Wl,-soname,lib$(TARGET).so.$(VER_MAJ) \
				     $(LFLAGS) -o $(SYSCONF_LINK_TARGET_SHARED) \
				     $(OBJECTS) $(OBJMOC) $(LIBS) && \
				 mv $(SYSCONF_LINK_TARGET_SHARED) $(DESTDIR); \
				 cd $(DESTDIR) && \
				 rm -f lib$(TARGET).so lib$(TARGET).so.$(VER_MAJ) lib$(TARGET).so.$(VER_MAJ).$(VER_MIN); \
				 ln -s $(SYSCONF_LINK_TARGET_SHARED) lib$(TARGET).so; \
				 ln -s $(SYSCONF_LINK_TARGET_SHARED) lib$(TARGET).so.$(VER_MAJ); \
				 ln -s $(SYSCONF_LINK_TARGET_SHARED) lib$(TARGET).so.$(VER_MAJ).$(VER_MIN)

# Linking static libraries
#   - Build the $(TARGET) library, eg. lib$(TARGET).a
#   - Place target in $(DESTDIR) - which has a trailing /
#
SYSCONF_AR		= ar cqs
SYSCONF_LINK_TARGET_STATIC = lib$(TARGET).a
SYSCONF_LINK_LIB_STATIC	= rm -f $(DESTDIR)$(SYSCONF_LINK_TARGET_STATIC) ; \
				 $(SYSCONF_AR) $(DESTDIR)$(SYSCONF_LINK_TARGET_STATIC) $(OBJECTS) $(OBJMOC)
# Compiling application source
SYSCONF_CXXFLAGS	= -pipe -g -Wno-deprecated
SYSCONF_CFLAGS		= -pipe -g  -Wno-deprecated
# Compiling library source
SYSCONF_CXXFLAGS_LIB	= -fPIC
SYSCONF_CFLAGS_LIB	= -fPIC
# Compiling shared-object source
SYSCONF_CXXFLAGS_SHOBJ	= -fPIC
SYSCONF_CFLAGS_SHOBJ	= -fPIC
# Default link type (static linking is still be used where required)
SYSCONF_LINK_LIB	= $(SYSCONF_LINK_LIB_SHARED)
SYSCONF_LINK_TARGET	= $(SYSCONF_LINK_TARGET_SHARED)
#############################################################################

####### Compiler, tools and options

CXX	=	$(SYSCONF_CXX)
CXXFLAGS=	$(SYSCONF_CXXFLAGS_QT) $(SYSCONF_CXXFLAGS)  -DDEBUG_LEVEL=$(QDL) $(shell root-config --cflags)
CC	=	$(SYSCONF_CC)
CFLAGS	=	$(SYSCONF_CFLAGS)
INCPATH =	-I$(ROOTSYS)/include
LFLAGS	=	$(SYSCONF_LFLAGS_QT) $(SYSCONF_LFLAGS_X11)
LIBS	=	$(SYSCONF_LIBS_QT) $(SYSCONF_LIBS_X11) $(SYSCONF_LIBS) $(shell root-config --prefix=$(ROOTSYS) --glibs) -lQtGSI
MOC	=	$(SYSCONF_MOC)

####### Target

DESTDIR = $(PWD)
DESTDIRBIN = $(PWD)
VER_MAJ = 2
VER_MIN = 0
VER_PATCH = 0
TARGET	= qtroot
TARGET1 =

####### Files

HEADERS =	qtroot.h \
		guitest.h

SOURCES =	main.cpp \
		qtroot.cpp \
		guitest.cpp 

OBJECTS =	main.o \
		qtroot.o \
		guitest.o



SRCMOC	=	moc_qtroot.cpp 
OBJMOC	=	moc_qtroot.o  

####### Implicit rules

.SUFFIXES: .cpp .cxx .cc .C .c

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o $@ $<

####### Build rules

all: $(DESTDIR)$(TARGET) 

lib: $(DESTDIR)$(TARGET) 



$(DESTDIR)$(TARGET): $(OBJECTS) $(OBJMOC) 
	$(SYSCONF_LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJMOC) $(LIBS)
	@mkdir -p $(DESTDIRBIN)
	@mkdir -p $(DESTDIR)


$(DESTDIR)$(TARGET1):$(OBJECTS) $(OBJMOC)
	$(SYSCONF_LINK_LIB_SHARED)

moc: $(SRCMOC)

tmake:
	tmake qtroot.pro

clean:
	-@rm -f $(OBJECTS) $(OBJMOC) $(SRCMOC)
	-@rm -f *~ core qtroot

#depend:
#	touch Make-depend
#	$(MAKEDEP) -fMake-depend -Y -- $(CXXFLAGS) -- $(SOURCES) > /dev/null 2>&1
####### Compile

main.o: main.cpp \
		qtroot.h

qtroot.o: qtroot.cpp \
		qtroot.h

guitest.o: guitest.cpp \
           guitest.h

moc_qtroot.o:       moc_qtroot.cpp \
		    qtroot.h

moc_qtroot.cpp: qtroot.h
	$(MOC) qtroot.h -o moc_qtroot.cpp
