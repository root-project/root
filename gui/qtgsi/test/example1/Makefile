#############################################################################
# Qt ROOT  example Makefile
# Author: D.Bertini <D.Bertini@gsi.de>
# Date:06/29/01
#############################################################################

include ../../../../config/Makefile.config
include ../../../../etc/Makefile.arch

MAKEDEP = rmkdepend

#debuggeng
QDL			= 0

# Linking with support libraries
# X11
SYSCONF_LIBS_X11	= -lXft -lXext -lX11 -lm

# Meta-object compiler

# Linking shared libraries
#   - Build the $(TARGET) library, eg. lib$(TARGET).so.2.0.1
#   - Place target in $(DESTDIR) - which has a trailing /
#   - Usually needs to incorporate $(VER_MAJ), $(VER_MIN) and $(VER_PATCH)
#

SYSCONF_LINK_TARGET_SHARED	= lib$(TARGET).so.$(VER_MAJ).$(VER_MIN).$(VER_PATCH)

SYSCONF_LINK_LIB_SHARED	=  $(LD) -shared -Wl,-soname,lib$(TARGET).so.$(VER_MAJ) \
				     $(LFLAGS) -o $(SYSCONF_LINK_TARGET_SHARED) \
				     $(OBJECTS) $(OBJMOC) $(LIBS) && \
				 mv $(SYSCONF_LINK_TARGET_SHARED) $(DESTDIR); \
				 cd $(DESTDIR) && \
				 rm -f lib$(TARGET).so lib$(TARGET).so.$(VER_MAJ) lib$(TARGET).so.$(VER_MAJ).$(VER_MIN); \
				 ln -s $(SYSCONF_LINK_TARGET_SHARED) lib$(TARGET).so; \
				 ln -s $(SYSCONF_LINK_TARGET_SHARED) lib$(TARGET).so.$(VER_MAJ); \
				 ln -s $(SYSCONF_LINK_TARGET_SHARED) lib$(TARGET).so.$(VER_MAJ).$(VER_MIN)


#############################################################################

####### Compiler, tools and options
ROOTCONF	= ../../../../bin/root-config
#QTCXXFLAGS=	-I$(QTINCDIR) $(CXXFLAGS) -DQT3_SUPPORT -DDEBUG_LEVEL=$(QDL) $(shell root-config --cflags)
QTCXXFLAGS=	-I. $(QTINCDIR:%=-I%) $(CXXFLAGS) -DQT3_SUPPORT -DQT_QT3SUPPORT_LIB -DDEBUG_LEVEL=$(QDL) $(shell $(ROOTCONF) --cflags)
INCPATH =	-I../../../../include
LFLAGS	=	$(LDFLAGS) $(QTLIBDIR) $(X11LIBDIR)
LIBS	=	$(QTLIB) $(SYSCONF_LIBS_X11) $(shell $(ROOTCONF) --glibs) -lQtGSI

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
	$(CXX) -c $(QTCXXFLAGS) $(INCPATH) -o $@ $<

.cxx.o:
	$(CXX) -c $(QTCXXFLAGS) $(INCPATH) -o $@ $<

.cc.o:
	$(CXX) -c $(QTCXXFLAGS) $(INCPATH) -o $@ $<

.C.o:
	$(CXX) -c $(QTCXXFLAGS) $(INCPATH) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o $@ $<

####### Build rules

all: $(DESTDIR)$(TARGET)

lib: $(DESTDIR)$(TARGET)


$(DESTDIR)$(TARGET): $(OBJECTS) $(OBJMOC)
	$(LD) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJMOC) $(LIBS)
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
	$(QTMOCEXE) qtroot.h -o moc_qtroot.cpp
