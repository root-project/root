#              $Id: GNUmakefile,v 1.31 2010/07/30 03:19:08 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

MORELIBS = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys

BINLIBS  = -L$(LIBDIR) -lXrdSec -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys

LIBDEPS = $(LIBDIR)/libXrdNet.a $(LIBDIR)/libXrdNetUtil.a \
          $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdSecClient.cc        \
        XrdSecPManager.cc      \
        XrdSecProtocolhost.cc  \
        XrdSecServer.cc        \
        XrdSecTLayer.cc        \
        XrdSectestClient.cc    \
        XrdSectestServer.cc    \
  
OBJSLIB = \
        $(OBJDIR)/XrdSecClient.o        \
        $(OBJDIR)/XrdSecPManager.o      \
        $(OBJDIR)/XrdSecProtocolhost.o  \
        $(OBJDIR)/XrdSecServer.o        \
        $(OBJDIR)/XrdSecTLayer.o

OBJECTA = \
        $(OBJDIR)/XrdSectestClient.o

OBJECTB = \
        $(OBJDIR)/XrdSectestServer.o

OBJECTS = $(OBJSLIB) $(OBJECTA) $(OBJECTB)

LIBRARY = $(LIBDIR)/libXrdSec.$(TYPESHLIB)
LIBARCH = $(LIBDIR)/libXrdSec.a

TSTBINA = $(BINDIR)/testclient
TSTBINB = $(BINDIR)/testserver

TARGETS = $(LIBARCH) $(LIBRARY) $(TSTBINA) $(TSTBINB)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdNet% ../XrdNet
vpath XrdOuc% ../XrdOuc
vpath XrdSys% ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdSec done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBARCH): $(OBJSLIB)
	@echo Creating archive $(LIBARCH) 
	$(ECHO)rm -f $(LIBARCH)
	$(ECHO)ar -rc $(LIBARCH) $(OBJSLIB)
	$(ECHO)ranlib $(LIBARCH)

$(LIBRARY): $(OBJSLIB) $(LIBDEPS)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(OBJSLIB) $(LDSO) $(MORELIBS) $(LIBS) -o $(LIBRARY);

$(TSTBINA): $(OBJECTA) $(LIBDEPS)
	@echo Creating executable $(TSTBINA) 
	$(ECHO)$(LD) $(LDOP) $(OBJECTA) $(LIBS) $(BINLIBS) -o $(TSTBINA)

$(TSTBINB): $(OBJECTB) $(LIBDEPS)
	@echo Creating executable $(TSTBINB) 
	$(ECHO)$(LD) $(LDOP) $(OBJECTB) $(LIBS) $(BINLIBS) -o $(TSTBINB)

$(OBJDIR)/XrdSecClient.o: XrdSysPthread.hh XrdSecPManager.hh XrdSecInterface.hh \
                          XrdOucErrInfo.hh XrdSecClient.cc   XrdSecEntity.hh
	@echo Compiling XrdSecClient.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecClient.o XrdSecClient.cc

$(OBJDIR)/XrdSecPManager.o: XrdSecInterface.hh XrdSecPManager.hh \
                            XrdSecProtocolhost.hh \
                            XrdOucErrInfo.hh   XrdSysPthread.hh \
                            XrdSecPManager.cc
	@echo Compiling XrdSecPManager.cc 
	$(ECHO)$(CC) -c $(CFLAGS32) $(CFUSENEWHDRS) $(INCLUDE) -o $(OBJDIR)/XrdSecPManager.o XrdSecPManager.cc

$(OBJDIR)/XrdSecProtocolhost.o: XrdSecProtocolhost.cc XrdSecProtocolhost.hh \
                                XrdSecInterface.hh    XrdSecEntity.hh
	@echo Compiling XrdSecProtocolhost.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecProtocolhost.o XrdSecProtocolhost.cc
$(OBJDIR)/XrdSecServer.o: XrdSecServer.cc       XrdSecServer.hh \
                          XrdNetDNS.hh          XrdSysHeaders.hh \
                          XrdOucErrInfo.hh      XrdSysError.hh \
                          XrdSysLogger.hh       XrdOucStream.hh \
                          XrdSecInterface.hh    XrdSecPManager.hh \
                          XrdSecTrace.hh        XrdOucEnv.hh
	@echo Compiling XrdSecServer.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecServer.o XrdSecServer.cc

$(OBJDIR)/XrdSecTLayer.o: XrdSecTLayer.cc       XrdSecTLayer.hh \
                          XrdSysPthread.hh      XrdSysHeaders.hh \
                          XrdOucErrInfo.hh \
                          XrdSecInterface.hh
	@echo Compiling XrdSecTLayer.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecTLayer.o XrdSecTLayer.cc

$(OBJDIR)/XrdSectestClient.o:   XrdSectestClient.cc XrdSecInterface.hh \
                                XrdSysHeaders.hh    XrdSecEntity.hh
	@echo Compiling XrdSectestClient.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSectestClient.o XrdSectestClient.cc

$(OBJDIR)/XrdSectestServer.o: XrdSecInterface.hh XrdSecEntity.hh XrdSysLogger.hh \
                              XrdSysHeaders.hh   XrdSectestServer.cc
	@echo Compiling XrdSectestServer.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) -o $(OBJDIR)/XrdSectestServer.o XrdSectestServer.cc
