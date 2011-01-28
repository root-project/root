# $Id: GNUmakefile,v 1.10 2009/05/26 12:48:32 ganis Exp $
#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
include ../GNUmake.env

BINLIBS  = $(CRYPTLIB) $(LIBAFS) -L$(LIBDIR) -lXrdCrypto -lXrdSut -lXrdOuc -lXrdSys

LIBDEPS = $(LIBDIR)/libXrdCrypto.a $(LIBDIR)/libXrdSut.a \
          $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
SOURCES = \
        XrdSecProtocolpwd.cc \
        XrdSecpwdSrvAdmin.cc

HEADERS = \
        XrdSecProtocolpwd.hh \

OBJECTS = \
        $(OBJDIR)/XrdSecProtocolpwd.o 

OBJECTA = \
        $(OBJDIR)/XrdSecpwdSrvAdmin.o

HEADERA = \
        XrdSutAux.hh \
        XrdSutPFEntry.hh \
        XrdSutPFile.hh \
        XrdSutRndm.hh \
        XrdSutString.hh

LIBRARY = $(LIBDIR)/libXrdSecpwd.$(TYPESHLIB)

SRVADMIN = $(BINDIR)/xrdpwdadmin

TARGETS = $(SRVADMIN) $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XProtocol%    ../XProtocol
vpath XPty%         ../XProtocol
vpath XrdOuc% ../XrdOuc
vpath XrdSec% ../XrdSec
vpath XrdSut% ../XrdSut
vpath XrdSys% ../XrdSys
vpath XrdSecpwd% ../XrdSecpwd
vpath XrdCrypto% ../XrdCrypto

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdSecpwd done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBRARY): $(OBJECTS) $(LIBDEPS)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(OBJECTS) $(LDSO) $(BINLIBS) $(LIBS) $(RESOLVLIB) -o $(LIBRARY)

$(SRVADMIN): $(OBJECTA) $(LIBDEPS)
	@echo Creating executable $(SRVADMIN) 
	$(ECHO)$(LD) $(LDOP) $(OBJECTA) $(BINLIBS) $(LIBS)  $(RESOLVLIB) -o $(SRVADMIN)

$(OBJDIR)/XrdSecpwdSrvAdmin.o: XrdSecpwdSrvAdmin.cc \
		XrdOucString.hh XrdSutAux.hh XProtocol.hh XPtypes.hh \
		XrdSutPFEntry.hh XrdSutPFile.hh XrdOucHash.hh \
		XrdOucHash.icc XrdSutRndm.hh XrdCryptoCipher.hh \
		XrdSutBucket.hh XrdCryptoBasic.hh \
		XrdCryptoFactory.hh XrdCryptoAux.hh
	@echo Compiling XrdSecpwdSrvAdmin.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecpwdSrvAdmin.o XrdSecpwdSrvAdmin.cc

$(OBJDIR)/XrdSecProtocolpwd.o: XrdSecProtocolpwd.cc XrdSysLogger.hh \
		XrdSysPthread.hh  XrdSysError.hh XrdOucStream.hh XrdSysHeaders.hh \
		XrdSutCache.hh XPtypes.hh XrdSutPFEntry.hh \
		XProtocol.hh XrdOucHash.hh XrdOucHash.icc XrdOucString.hh \
		XrdSecProtocolpwd.hh XrdOucErrInfo.hh XrdSysPlatform.hh \
		XrdOucTokenizer.hh XrdSecInterface.hh XrdSecEntity.hh XrdSecTrace.hh \
		XrdOucTrace.hh XrdSutPFile.hh XrdSutBuffer.hh \
		XrdSutBuckList.hh XrdSutBucket.hh XrdSutAux.hh \
		XrdSutRndm.hh XrdCryptoAux.hh XrdCryptoCipher.hh XrdCryptoBasic.hh \
		XrdCryptoFactory.hh XrdSecpwdPlatform.hh
	@echo Compiling XrdSecProtocolpwd.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(PWDEXTRACFLAGS) $(INCLUDE) $(INCAFS) \
		-o $(OBJDIR)/XrdSecProtocolpwd.o XrdSecProtocolpwd.cc
