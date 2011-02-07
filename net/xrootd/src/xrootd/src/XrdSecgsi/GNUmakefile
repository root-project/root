# $Id$
#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
include ../GNUmake.env

LIBBASE = $(LIBDIR)/libXrdCrypto.a  $(LIBDIR)/libXrdSut.a \
          $(LIBDIR)/libXrdNet.a     $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdNetUtil.a $(LIBDIR)/libXrdSys.a

ifneq ($(TYPE),Darwin)

ifneq ($(LIBSSL),)
LIBDEPS = $(LIBDIR)/libXrdCryptossl.$(TYPESHLIB) $(LIBBASE)

BINLIBS = -L$(LIBDIR) -lXrdCryptossl $(LIBBASE)

LIBLIBS = -L$(LIBDIR) -lXrdCryptossl $(LIBBASE)
else
LIBDEPS = $(LIBBASE)

BINLIBS = $(LIBBASE)

LIBLIBS = $(LIBBASE)
endif

else

ifneq ($(LIBSSL),)

LIBUTIL = $(LIBDIR)/libXrdCryptossl.a $(LIBBASE)

BINLIBS = $(LIBSSL) $(LIBUTIL)

LIBLIBS = $(LIBSSL) $(LIBUTIL)

else

BINLIBS = $(LIBBASE)

LIBLIBS = $(LIBBASE)

endif

LIBDEPS = $(LIBBASE)

endif

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
SOURCES = \
        XrdSecProtocolgsi.cc \
        XrdSecgsiProxy.cc \
        XrdSecgsitest.cc \
        XrdSecgsiGMAPFunLDAP.cc

HEADERS = \
        XrdSecProtocolgsi.hh

OBJECTS = \
        $(OBJDIR)/XrdSecProtocolgsi.o

OBJECTA = \
        $(OBJDIR)/XrdSecgsitest.o

OBJECTB = \
        $(OBJDIR)/XrdSecgsiProxy.o

OBJECTC = \
        $(OBJDIR)/XrdSecgsiGMAPFunLDAP.o

GSIEXTRACFLAGS = $(INCSSL) -DPERL5 $(CRYPTOSSLEXTRACFLAGS)

LIBRARY = $(LIBDIR)/libXrdSecgsi.$(TYPESHLIB)

LIBLDAP = $(LIBDIR)/libXrdSecgsiGMAPLDAP.$(TYPESHLIB)

SECGSITEST = $(BINDIR)/xrdsecgsitest

SECGSIPROXY = $(BINDIR)/xrdgsiproxy

TARGETS = $(LIBRARY) $(LIBLDAP) $(SECGSIPROXY) $(SECGSITEST)

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
	@echo Make XrdSecgsi done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBRARY): $(OBJECTS) $(LIBDEPS)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(LD) $(OBJECTS) $(LDSO) $(LIBLIBS) $(LIBS) -o $(LIBRARY)

$(LIBLDAP): $(OBJECTC)
	@echo Creating shared library $(LIBLDAP) 
	$(ECHO)$(LD) $(OBJECTC) $(LDSO) $(LIBS) -o $(LIBLDAP)

$(SECGSITEST): $(OBJECTA) $(LIBDEPS)
	@echo Creating executable $(SECGSITEST) 
	$(ECHO)$(LD) $(LDOP) $(OBJECTA) $(BINLIBS) $(LIBS) -o $(SECGSITEST)

$(SECGSIPROXY): $(OBJECTB) $(LIBDEPS)
	@echo Creating executable $(SECGSIPROXY) 
	$(ECHO)$(LD) $(LDOP) $(OBJECTB) $(BINLIBS) $(LIBS) -o $(SECGSIPROXY)

$(OBJDIR)/XrdSecgsitest.o: XrdSecgsitest.cc XrdOucString.hh \
		XrdSysError.hh XrdSutAux.hh XProtocol.hh XPtypes.hh XrdCryptoAux.hh \
		XrdCryptoFactory.hh XrdCryptoX509.hh XrdSutBucket.hh XrdCryptoRSA.hh \
		XrdCryptoX509Req.hh XrdCryptoX509Chain.hh XrdCryptoX509Crl.hh \
		XrdCryptosslgsiX509Chain.hh XrdCryptosslgsiAux.hh XrdSecgsiTrace.hh \
		XrdOucTrace.hh
	@echo Compiling XrdSecgsitest.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(GSIEXTRACFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecgsitest.o XrdSecgsitest.cc

$(OBJDIR)/XrdSecgsiProxy.o: XrdSecgsiProxy.cc XrdOucString.hh \
		XrdSysError.hh XrdSutAux.hh XProtocol.hh XPtypes.hh XrdCryptoAux.hh \
		XrdCryptoFactory.hh XrdCryptoX509.hh XrdSutBucket.hh XrdCryptoRSA.hh \
		XrdCryptoX509Req.hh XrdCryptoX509Chain.hh XrdCryptoX509Crl.hh \
		XrdCryptosslgsiX509Chain.hh XrdCryptosslgsiAux.hh XrdSecgsiTrace.hh \
		XrdOucTrace.hh
	@echo Compiling XrdSecgsiProxy.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(GSIEXTRACFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecgsiProxy.o XrdSecgsiProxy.cc

$(OBJDIR)/XrdSecProtocolgsi.o: XrdSecProtocolgsi.cc XrdSysLogger.hh \
		XrdSysError.hh XrdOucStream.hh XrdSutCache.hh XPtypes.hh XrdSutPFEntry.hh \
		XProtocol.hh XrdOucHash.hh XrdOucHash.icc XrdOucString.hh \
		XrdCryptoMsgDigest.hh XrdCryptoBasic.hh XrdSutBucket.hh XrdSutAux.hh \
		XrdCryptosslgsiAux.hh XrdCryptosslgsiX509Chain.hh \
		XrdCryptoX509Chain.hh XrdCryptoX509.hh XrdCryptoRSA.hh XrdCryptoAux.hh \
		XrdCryptoX509Crl.hh XrdCryptoX509Req.hh XrdSecProtocolgsi.hh \
		XrdOucErrInfo.hh XrdSysPlatform.hh  XrdOucTokenizer.hh \
		XrdSecInterface.hh XrdSecEntity.hh XrdSecgsiTrace.hh \
		XrdOucTrace.hh XrdSutPFile.hh XrdSutBuffer.hh XrdSutBuckList.hh \
		XrdSutRndm.hh XrdCryptoCipher.hh XrdCryptoFactory.hh
	@echo Compiling XrdSecProtocolgsi.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(GSIEXTRACFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecProtocolgsi.o XrdSecProtocolgsi.cc

$(OBJDIR)/XrdSecgsiGMAPFunLDAP.o: XrdSecgsiGMAPFunLDAP.cc
	@echo Compiling XrdSecgsiGMAPFunLDAP.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecgsiGMAPFunLDAP.o XrdSecgsiGMAPFunLDAP.cc

