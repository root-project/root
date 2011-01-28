#              $Id$

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

SOURCES  = \
        XrdPosix.cc  \
        XrdPosixLinkage.cc  \
        XrdPosixXrootd.cc

SOURCET = \
        XrdPosixPreload.cc \
        XrdPosixPreload32.cc

OBJECTS  = \
        $(OBJDIR)/XrdPosix.o  \
        $(OBJDIR)/XrdPosixLinkage.o  \
        $(OBJDIR)/XrdPosixXrootd.o

OBJECTT = \
        $(OBJDIR)/XrdPosixPreload.o  \
        $(OBJDIR)/XrdPosixPreload32.o

OBJECTU = \
        $(LIBDIR)/libXrdClient.a

MORELIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys

LIBDEP   = $(LIBDIR)/libXrdClient.a \
           $(LIBDIR)/libXrdNet.a \
           $(LIBDIR)/libXrdNetUtil.a \
           $(LIBDIR)/libXrdOuc.a \
           $(LIBDIR)/libXrdSys.a

LIBARCH  = $(LIBDIR)/libXrdPosix.a

LIBRARY  = $(LIBDIR)/libXrdPosix.$(TYPESHLIB)

LIBPLDR  = $(LIBDIR)/libXrdPosixPreload.$(TYPESHLIB)

TARGETS  = $(LIBARCH) $(LIBRARY) $(LIBPLDR)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdClient% ../XrdClient
vpath XrdOuc%    ../XrdOuc
vpath XrdSys%    ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdPosix done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBARCH): $(OBJECTP) $(OBJECTS) $(OBJECTT) $(LIBDEP)
	@echo Creating archive $(LIBARCH) 
	$(ECHO)rm -f $(LIBARCH)
	$(ECHO)ar -rc $(LIBARCH) $(OBJECTS)
	$(ECHO)ranlib $(LIBARCH)

$(LIBPLDR): $(OBJECTP) $(OBJECTS) $(OBJECTT) $(LIBDEP)
	@echo Creating shared library $(LIBPLDR)
	$(ECHO)$(CC) $(CFLAGS) $(OBJECTS) $(OBJECTT) $(OBJECTU) $(LDSO) $(MORELIBS) $(LIBS) -o $(LIBPLDR)

$(LIBRARY): $(OBJECTP) $(OBJECTS) $(OBJECTT) $(LIBDEP)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(CFLAGS) $(OBJECTS) $(OBJECTU) $(LDSO) $(MORELIBS) $(LIBS) -o $(LIBRARY)
ifneq ($(subst $(MACOSX_MINOR),,1234),1234)
	$(ECHO)if [ "x$(TYPE)" = "xDarwin" ]; then \
	   $(CC) $(CFLAGS) $(OBJECTS) $(TYPELDDY) $(MORELIBS) $(LIBS) -o $(LIBDIR)/libXrdPosix.$(TYPEDYLIB); \
	fi
endif
# MacOS X does not support linking against shared libraries so include
# all of the objects in the preload library (yech).

$(OBJDIR)/XrdPosix.o:  XrdPosix.cc XrdPosix.hh XrdPosixXrootd.hh  \
                       XrdPosixExtern.hh XrdPosixLinkage.hh \
                       XrdOucTokenizer.hh XrdSysHeaders.hh
	@echo Compiling XrdPosix.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdPosix.o XrdPosix.cc

$(OBJDIR)/XrdPosixLinkage.o:  XrdPosixLinkage.cc   XrdPosixLinkage.hh \
                              XrdPosixOsDep.hh     XrdSysPlatform.hh
	@echo Compiling XrdPosixLinkage.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdPosixLinkage.o XrdPosixLinkage.cc

$(OBJDIR)/XrdPosixPreload.o:  XrdPosixPreload.cc   XrdPosixLinkage.hh \
                              XrdPosixExtern.hh    XrdPosixOsDep.hh \
                              XrdPosixXrootd.hh
	@echo Compiling XrdPosixPreload.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdPosixPreload.o XrdPosixPreload.cc

$(OBJDIR)/XrdPosixPreload32.o:  XrdPosixPreload32.cc   XrdPosixLinkage.hh \
                                XrdPosixExtern.hh      XrdPosixXrootd.hh \
                                XrdPosixOsDep.hh \
                                XrdSysHeaders.hh       XrdSysPlatform.hh
	@echo Compiling XrdPosixPreload32.cc
	$(ECHO)$(CC) -c $(CFLAGS32) $(CFUSENEWHDRS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdPosixPreload32.o XrdPosixPreload32.cc

$(OBJDIR)/XrdPosixXrootd.o:  XrdPosixXrootd.cc   XrdPosixXrootd.hh \
                             XrdClient.hh        XrdPosixExtern.hh \
                             XrdSysHeaders.hh    XrdPosixCallBack.hh \
                             XrdPosixOsDep.hh
	@echo Compiling XrdPosixXrootd.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdPosixXrootd.o XrdPosixXrootd.cc
