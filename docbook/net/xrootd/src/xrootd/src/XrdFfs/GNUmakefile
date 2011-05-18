#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

SOURCES = \
        XrdFfsDent.cc     \
        XrdFfsFsinfo.cc   \
        XrdFfsMisc.cc     \
        XrdFfsPosix.cc    \
        XrdFfsQueue.cc    \
        XrdFfsWcache.cc   \
        XrdFfsXrootdfs.cc

OBJLIB  = \
        $(OBJDIR)/XrdFfsDent.o     \
        $(OBJDIR)/XrdFfsFsinfo.o   \
        $(OBJDIR)/XrdFfsMisc.o     \
        $(OBJDIR)/XrdFfsPosix.o    \
        $(OBJDIR)/XrdFfsQueue.o    \
        $(OBJDIR)/XrdFfsWcache.o

OBJBIN  = \
        $(OBJDIR)/XrdFfsXrootdfs.o

OBJECTS = $(OBJLIB) $(OBJBIN)

LIBRARY = $(LIBDIR)/libXrdFfs.a

BINARY  = $(BINDIR)/xrootdfsd

BINDEPS = \
          $(LIBDIR)/libXrdFfs.a \
          $(LIBDIR)/libXrdPosix.a \
          $(LIBDIR)/libXrdClient.a \
          $(LIBDIR)/libXrdSec.a \
          $(LIBDIR)/libXrdSecsss.a \
          $(LIBDIR)/libXrdNet.a \
          $(LIBDIR)/libXrdNetUtil.a \
          $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdSys.a 


BINLIBS = -L$(LIBDIR) $(LIBFUSE) 

TARGETS = $(LIBRARY) $(BINARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdFfs%    ../XrdFfs
vpath XrdOuc%    ../XrdOuc
vpath XrdPosix%  ../XrdPosix
vpath XrdSecsss% ../XrdSecsss
vpath XrdSec%    ../XrdSec
vpath XrdClient% ../XrdClient

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdFfs done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBRARY): $(OBJLIB)
	@echo Creating archive $(LIBRARY) 
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)ar -rc $(LIBRARY) $(OBJLIB)
	$(ECHO)ranlib $(LIBRARY)

$(BINARY): $(OBJECTS) $(OBJBIN) $(BINDEPS)
	@echo Creating executable $(BINARY)
	$(ECHO)$(LD) $(LDOP) $(OBJBIN) $(BINDEPS) $(BINLIBS) $(LIBS) $(TYPELIBMT) -o $(BINARY)

$(OBJDIR)/XrdFfsDent.o: XrdFfsDent.cc  XrdFfsDent.hh
	@echo Compiling XrdFfsDent.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdFfsDent.cc -o $(OBJDIR)/XrdFfsDent.o

$(OBJDIR)/XrdFfsFsinfo.o: XrdFfsFsinfo.cc  XrdFfsFsinfo.hh
	@echo Compiling XrdFfsFsinfo.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdFfsFsinfo.cc -o $(OBJDIR)/XrdFfsFsinfo.o

$(OBJDIR)/XrdFfsMisc.o: XrdFfsMisc.cc   XrdFfsMisc.hh       XrdFfsDent.hh \
                        XrdFfsFsinfo.hh XrdFfsPosix.hh      XrdFfsQueue.hh \
                        XrdClient.hh    XrdClientAdmin.hh   XrdClientConst.hh \
                        XrdClientEnv.hh XrdClientUrlInfo.hh \
                        XrdOucString.hh \
                        XrdSecEntity.hh XrdSecsssID.hh
	@echo Compiling XrdFfsMisc.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdFfsMisc.cc -o $(OBJDIR)/XrdFfsMisc.o

$(OBJDIR)/XrdFfsPosix.o: XrdFfsPosix.cc    XrdFfsPosix.hh  \
                         XrdFfsMisc.hh     XrdFfsDent.hh   XrdFfsQueue.hh \
                         XrdPosixXrootd.hh
	@echo Compiling XrdFfsPosix.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdFfsPosix.cc -o $(OBJDIR)/XrdFfsPosix.o

$(OBJDIR)/XrdFfsQueue.o: XrdFfsQueue.cc  XrdFfsQueue.hh
	@echo Compiling XrdFfsQueue.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdFfsQueue.cc -o $(OBJDIR)/XrdFfsQueue.o

$(OBJDIR)/XrdFfsWcache.o: XrdFfsWcache.cc  XrdFfsWcache.hh \
                          XrdFfsPosix.hh
	@echo Compiling XrdFfsWcache.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdFfsWcache.cc -o $(OBJDIR)/XrdFfsWcache.o

$(OBJDIR)/XrdFfsXrootdfs.o: XrdFfsXrootdfs.cc \
                            XrdFfsDent.hh      XrdFfsFsinfo.hh \
                            XrdFfsMisc.hh      XrdFfsPosix.hh \
                            XrdFfsQueue.hh     XrdFfsWcache.hh \
                            XrdPosixXrootd.hh
	@echo Compiling XrdFfsXrootdfs.cc
	$(ECHO)$(CC) -c  $(CFLAGS) $(INCLUDE) $(CFHASFUSE) XrdFfsXrootdfs.cc -o $(OBJDIR)/XrdFfsXrootdfs.o

ifneq ($(findstring g++, $(CC)),)
$(OBJDIR)/XrdFfsPosix.o: CFLAGS += -fno-strict-aliasing
endif
