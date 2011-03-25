#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

SOURCES  = \
        XrdPss.cc  \
        XrdPssAio.cc  \
        XrdPssConfig.cc

OBJECTS  = \
        $(OBJDIR)/XrdPss.o  \
        $(OBJDIR)/XrdPssAio.o  \
        $(OBJDIR)/XrdPssConfig.o

OBJECTT  = \
        $(OBJDIR)/XrdPosixLinkage.o \
        $(OBJDIR)/XrdPosixXrootd.o

OBJECTU  = $(LIBDIR)/libXrdFfs.a    $(LIBDIR)/libXrdClient.a \
           $(LIBDIR)/libXrdSecsss.a $(LIBDIR)/libXrdSec.a

MORELIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys

LIBDEP   = $(LIBDIR)/libXrdClient.a \
           $(LIBDIR)/libXrdFfs.a \
           $(LIBDIR)/libXrdNet.a \
           $(LIBDIR)/libXrdNetUtil.a \
           $(LIBDIR)/libXrdOuc.a \
           $(LIBDIR)/libXrdSec.a \
           $(LIBDIR)/libXrdSecsss.a \
           $(LIBDIR)/libXrdSys.a

LIBRARY  = $(LIBDIR)/libXrdPss.$(TYPESHLIB)

TARGETS  = $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdClient% ../XrdClient
vpath XrdFfs%    ../XrdFfs
vpath XrdNet%    ../XrdNet
vpath XrdOss%    ../XrdOss
vpath XrdOuc%    ../XrdOuc
vpath XrdPosix%  ../XrdPosix
vpath XrdPss%    ../XrdPss
vpath XrdSec%    ../XrdSec
vpath XrdSfs%    ../XrdSfs
vpath XrdSys%    ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdPss done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS) $(OBJECTT) $(LIBDEP)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(CFLAGS) $(OBJECTT) $(OBJECTS) $(OBJECTU) $(LDSO) $(MORELIBS) $(LIBS) -o $(LIBRARY)

$(OBJDIR)/XrdPss.o:  XrdPss.cc         XrdPss.hh \
                     XrdFfsPosix.hh    XrdPosixXrootd.hh \
                     XrdOss.hh         XrdOssError.hh \
                     XrdOucEnv.hh      XrdOucName2Name.hh \
                     XrdSecEntity.hh \
                     XrdSysError.hh    XrdSysHeaders.hh \
                     XrdSysPlatform.hh \
                     ../XrdVersion.hh
	@echo Compiling XrdPss.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdPss.o XrdPss.cc

$(OBJDIR)/XrdPssAio.o:  XrdPssAio.cc   XrdPss.hh \
                        XrdSfsAio.hh
	@echo Compiling XrdPssAio.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdPssAio.o XrdPssAio.cc

$(OBJDIR)/XrdPssConfig.o:  XrdPssConfig.cc   XrdPss.hh         XrdFfsDent.hh \
                           XrdFfsMisc.hh     XrdFfsWcache.hh   XrdFfsQueue.hh \
                           XrdOuca2x.hh      XrdOucEnv.hh      XrdSysError.hh \
                           XrdOucStream.hh   XrdOucTList.hh    XrdOucUtils.hh \
                           XrdNetDNS.hh      XrdSysPlatform.hh XrdSysHeaders.hh \
                           XrdPosixXrootd.hh XrdOucName2Name.hh
	@echo Compiling XrdPssConfig.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdPssConfig.o XrdPssConfig.cc
