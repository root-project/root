# $Id$
#-----------------------------------------------------------------------------#
#                            E n v i r o n m e n t                            #
#-----------------------------------------------------------------------------#
include ../GNUmake.env

BINLIBS = $(CRYPTLIB) -L$(LIBDIR) -lXrdCrypto -lXrdSut -lXrdOuc -lXrdSys

LIBUTIL = $(LIBDIR)/libXrdSut.a $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
SOURCES = \
        PC1.cc      \
        PC3.cc      \
        XrdCryptoAux.cc      \
        XrdCryptoBasic.cc      \
        XrdCryptoCipher.cc      \
	XrdCryptoFactory.cc \
        XrdCryptoLite.cc \
        XrdCryptoLite_bf32.cc \
        XrdCryptoMsgDigest.cc \
        XrdCryptoRSA.cc \
        XrdCryptoX509.cc \
        XrdCryptoX509Chain.cc \
        XrdCryptoX509Crl.cc \
        XrdCryptoX509Req.cc \
        XrdCryptolocalCipher.cc  \
	XrdCryptolocalFactory.cc \
        XrdCryptosslAux.cc         \
        XrdCryptosslCipher.cc         \
        XrdCryptosslMsgDigest.cc      \
        XrdCryptosslRSA.cc      \
        XrdCryptosslX509.cc      \
        XrdCryptosslX509Crl.cc      \
        XrdCryptosslX509Req.cc      \
        XrdCryptosslFactory.cc      \
        XrdCryptosslgsiAux.cc \
        XrdCryptosslgsiX509Chain.cc \
	XrdCryptotest.cc

OBJECTS = \
        $(OBJDIR)/PC1.o      \
        $(OBJDIR)/PC3.o      \
        $(OBJDIR)/XrdCryptoAux.o      \
        $(OBJDIR)/XrdCryptoBasic.o      \
        $(OBJDIR)/XrdCryptoCipher.o      \
	$(OBJDIR)/XrdCryptoFactory.o \
        $(OBJDIR)/XrdCryptoMsgDigest.o  \
        $(OBJDIR)/XrdCryptoRSA.o \
        $(OBJDIR)/XrdCryptoX509.o  \
        $(OBJDIR)/XrdCryptoX509Chain.o  \
        $(OBJDIR)/XrdCryptoX509Crl.o  \
        $(OBJDIR)/XrdCryptoX509Req.o  \
	$(OBJDIR)/XrdCryptolocalCipher.o \
	$(OBJDIR)/XrdCryptolocalFactory.o \

OBJCSSL = \
        $(OBJDIR)/XrdCryptosslAux.o         \
        $(OBJDIR)/XrdCryptosslCipher.o         \
        $(OBJDIR)/XrdCryptosslFactory.o      \
        $(OBJDIR)/XrdCryptosslMsgDigest.o      \
        $(OBJDIR)/XrdCryptosslRSA.o      \
        $(OBJDIR)/XrdCryptosslX509.o      \
        $(OBJDIR)/XrdCryptosslX509Crl.o      \
        $(OBJDIR)/XrdCryptosslX509Req.o    \
        $(OBJDIR)/XrdCryptosslgsiAux.o      \
        $(OBJDIR)/XrdCryptosslgsiX509Chain.o

OBJECTA = \
        $(OBJDIR)/XrdCryptotest.o

OBJECTL = \
        $(OBJDIR)/XrdCryptoLite.o \
        $(OBJDIR)/XrdCryptoLite_bf32.o \

LIBARCH = $(LIBDIR)/libXrdCrypto.a
LIBARCL = $(LIBDIR)/libXrdCryptoLite.a
LIBRARY = $(LIBDIR)/libXrdCrypto.$(TYPESHLIB)
LIBCSSL = $(LIBDIR)/libXrdCryptossl.$(TYPESHLIB)

ifeq ($(TYPE),Darwin)
LIBASSL = $(LIBDIR)/libXrdCryptossl.a
endif

CRYPTOTEST = $(BINDIR)/xrdcryptotest

CRYPTOSSLEXTRACFLAGS = -DOPENSSL_NO_KRB5 -DPERL5 $(SSLEXTRACFLAGS)

ifneq ($(LIBSSL),)
TARGETS = $(LIBARCH) $(LIBARCL) $(LIBRARY) $(LIBASSL) $(LIBCSSL)
else
TARGETS = $(LIBARCH) $(LIBARCL) $(LIBRARY)
endif

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                           -
#------------------------------------------------------------------------------#

vpath XProtocol%    ../XProtocol
vpath XPty%         ../XProtocol
vpath XrdOuc%       ../XrdOuc
vpath XrdSec%       ../XrdSec
vpath XrdSut%       ../XrdSut
vpath XrdSys%       ../XrdSys
vpath PSCH%         ../XrdCrypto

#------------------------------------------------------------------------------#
#                     The C compiler: temporary solution                       -
#------------------------------------------------------------------------------#

CCMP      = gcc
CCMPFLAGS = -pipe -Wall -W -fPIC

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make crypto done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBARCH): $(OBJECTS)
	@echo Creating archive $(LIBARCH) 
	$(ECHO)rm -f $(LIBARCH)
	$(ECHO)ar -rc $(LIBARCH) $(OBJECTS)
	$(ECHO)ranlib $(LIBARCH)

$(LIBARCL): $(OBJECTL)
	@echo Creating archive $(LIBARCL) 
	$(ECHO)rm -f $(LIBARCL)
	$(ECHO)ar -rc $(LIBARCL) $(OBJECTL)
	$(ECHO)ranlib $(LIBARCL)

$(LIBRARY): $(OBJECTS) $(LIBUTIL)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(OBJECTS) $(LDSO) $(LIBUTIL) -o $(LIBRARY)

$(LIBASSL): $(OBJCSSL)
	@echo Creating archive $(LIBASSL) 
	$(ECHO)rm -f $(LIBASSL)
	$(ECHO)ar -rc $(LIBASSL) $(OBJCSSL)
	$(ECHO)ranlib $(LIBASSL)

$(LIBCSSL): $(OBJCSSL) $(LIBARCH) $(LIBUTIL)
	@echo Creating shared library $(LIBCSSL) 
	$(ECHO)$(LD) $(OBJCSSL) $(LDSO) $(LIBSSL) $(LIBARCH) $(LIBUTIL) -o $(LIBCSSL)

$(CRYPTOTEST): $(OBJECTA) $(LIBARCH) $(LIBDEPS)
	@echo Creating executable $(CRYPTOTEST) 
	$(ECHO)$(LD) $(OBJECTA) $(LIBARCH) $(BINLIBS) $(LIBS) -o $(CRYPTOTEST)

$(OBJDIR)/XrdCryptotest.o: XrdCryptotest.cc XrdSutAux.hh XProtocol.hh XrdSutBucket.hh \
			   XrdCryptoAux.hh XrdCryptoFactory.hh XrdCryptoCipher.hh \
			   XrdCryptoMsgDigest.hh XrdCryptoRSA.hh XrdCryptoBasic.hh \
			   XrdOucString.hh
	@echo Compiling XrdCryptotest.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptotest.o XrdCryptotest.cc

$(OBJDIR)/PC1.o: PC1.cc PC1.hh
	@echo Compiling PC1.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/PC1.o PC1.cc

$(OBJDIR)/PC3.o: PC3.cc PC3.hh
	@echo Compiling PC3.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/PC3.o PC3.cc

$(OBJDIR)/XrdCryptoAux.o: XrdCryptoAux.cc XrdSysLogger.hh \
			  XrdSysPthread.hh XrdSysError.hh \
			  XrdCryptoAux.hh XProtocol.hh XPtypes.hh \
			  XrdCryptoTrace.hh XrdOucTrace.hh
	@echo Compiling XrdCryptoAux.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoAux.o XrdCryptoAux.cc

$(OBJDIR)/XrdCryptoBasic.o: XrdCryptoBasic.cc XProtocol.hh \
			XPtypes.hh XrdCryptoAux.hh \
			XrdCryptoBasic.hh XrdSutBucket.hh
	@echo Compiling XrdCryptoBasic.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoBasic.o XrdCryptoBasic.cc

$(OBJDIR)/XrdCryptoCipher.o: XrdCryptoCipher.cc XrdCryptoAux.hh \
			XProtocol.hh XPtypes.hh \
			XrdCryptoCipher.hh XrdSutBucket.hh \
			XrdSutAux.hh XrdCryptoBasic.hh
	@echo Compiling XrdCryptoCipher.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoCipher.o XrdCryptoCipher.cc

$(OBJDIR)/XrdCryptoFactory.o: XrdCryptoFactory.cc \
			XrdCryptoAux.hh XProtocol.hh \
			XPtypes.hh XrdCryptoTrace.hh \
			XrdOucTrace.hh XrdSysError.hh \
			XrdCryptoFactory.hh XrdCryptolocalFactory.hh
	@echo Compiling XrdCryptoFactory.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoFactory.o XrdCryptoFactory.cc

$(OBJDIR)/XrdCryptoLite.o: XrdCryptoLite.cc    XrdCryptoLite.hh
	@echo Compiling XrdCryptoLite.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoLite.o XrdCryptoLite.cc

$(OBJDIR)/XrdCryptoLite_bf32.o: XrdCryptoLite_bf32.cc    XrdCryptoLite.hh  \
                                XrdOucCRC.hh
	@echo Compiling XrdCryptoLite_bf32.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(SSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptoLite_bf32.o XrdCryptoLite_bf32.cc

$(OBJDIR)/XrdCryptoMsgDigest.o: XrdCryptoMsgDigest.cc \
			XrdCryptoAux.hh XProtocol.hh \
			XPtypes.hh XrdCryptoMsgDigest.hh \
			XrdCryptoBasic.hh XrdSutBucket.hh XrdSutAux.hh
	@echo Compiling XrdCryptoMsgDigest.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoMsgDigest.o XrdCryptoMsgDigest.cc

$(OBJDIR)/XrdCryptoRSA.o: XrdCryptoRSA.cc XrdCryptoRSA.hh \
			XrdSutBucket.hh XrdSutAux.hh \
			XProtocol.hh XPtypes.hh XrdOucString.hh \
                        XrdCryptoAux.hh
	@echo Compiling XrdCryptoRSA.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoRSA.o XrdCryptoRSA.cc

$(OBJDIR)/XrdCryptoX509.o: XrdCryptoX509.cc XrdCryptoX509.hh \
			XrdSutBucket.hh XrdSutAux.hh \
			XProtocol.hh XPtypes.hh \
			XrdCryptoRSA.hh XrdOucString.hh \
			XrdCryptoAux.hh XrdCryptoTrace.hh \
			XrdOucTrace.hh XrdSysError.hh
	@echo Compiling XrdCryptoX509.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoX509.o XrdCryptoX509.cc

$(OBJDIR)/XrdCryptoX509Chain.o: XrdCryptoX509Chain.cc \
			XrdCryptoX509Chain.hh XrdSutBucket.hh \
			XrdSutAux.hh XProtocol.hh \
			XPtypes.hh XrdCryptoX509.hh \
			XrdCryptoRSA.hh XrdOucString.hh \
			XrdCryptoAux.hh XrdCryptoX509Crl.hh \
			XrdCryptoTrace.hh XrdOucTrace.hh XrdSysError.hh
	@echo Compiling XrdCryptoX509Chain.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoX509Chain.o XrdCryptoX509Chain.cc

$(OBJDIR)/XrdCryptoX509Req.o: XrdCryptoX509Req.cc \
			XrdCryptoX509Req.hh XrdSutBucket.hh \
			XrdSutAux.hh XProtocol.hh \
			XPtypes.hh XrdCryptoRSA.hh \
			XrdOucString.hh XrdCryptoAux.hh \
			XrdCryptoTrace.hh XrdOucTrace.hh \
			XrdSysError.hh
	@echo Compiling XrdCryptoX509Req.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoX509Req.o XrdCryptoX509Req.cc

$(OBJDIR)/XrdCryptoX509Crl.o: XrdCryptoX509Crl.cc XrdCryptoX509Crl.hh \
			XrdCryptoX509.hh XrdSutBucket.hh \
			XrdSutAux.hh XProtocol.hh \
			XPtypes.hh XrdCryptoRSA.hh XrdOucString.hh \
			XrdCryptoAux.hh
	@echo Compiling XrdCryptoX509Crl.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptoX509Crl.o XrdCryptoX509Crl.cc

$(OBJDIR)/XrdCryptolocalCipher.o: XrdCryptolocalCipher.cc \
			XrdSutRndm.hh XrdSutAux.hh XProtocol.hh \
			XPtypes.hh XrdCryptolocalCipher.hh \
			XrdCryptoCipher.hh XrdSutBucket.hh \
			XrdCryptoBasic.hh PC1.hh PC3.hh
	@echo Compiling XrdCryptolocalCipher.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptolocalCipher.o XrdCryptolocalCipher.cc

$(OBJDIR)/XrdCryptolocalFactory.o: XrdCryptolocalFactory.cc \
			PC1.hh XrdCryptolocalCipher.hh \
			XrdCryptoCipher.hh XrdSutBucket.hh \
			XrdSutAux.hh XProtocol.hh \
			XPtypes.hh XrdCryptoBasic.hh \
			XrdCryptolocalFactory.hh XrdCryptoFactory.hh \
			XrdCryptoAux.hh  \
			XrdCryptoRSA.hh XrdOucString.hh \
			XrdCryptoTrace.hh \
			XrdOucTrace.hh XrdSysError.hh
	@echo Compiling XrdCryptolocalFactory.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptolocalFactory.o XrdCryptolocalFactory.cc

$(OBJDIR)/XrdCryptosslAux.o: XrdCryptosslAux.cc XrdCryptoX509Chain.hh \
			XrdSutBucket.hh XrdSutAux.hh \
			XProtocol.hh XPtypes.hh \
			XrdCryptoX509.hh XrdCryptoRSA.hh \
			XrdOucString.hh XrdCryptoX509Crl.hh \
			XrdCryptosslAux.hh XrdCryptosslRSA.hh \
			XrdCryptosslX509.hh XrdCryptosslTrace.hh \
			XrdOucTrace.hh XrdSysError.hh
	@echo Compiling XrdCryptosslAux.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CRYPTOSSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptosslAux.o XrdCryptosslAux.cc

$(OBJDIR)/XrdCryptosslCipher.o: XrdCryptosslCipher.cc XrdSutRndm.hh \
			XrdSutAux.hh XProtocol.hh XPtypes.hh \
			XrdCryptosslTrace.hh XrdOucTrace.hh \
			XrdSysError.hh XrdCryptoAux.hh \
			XrdCryptosslCipher.hh XrdCryptoCipher.hh \
			XrdSutBucket.hh XrdCryptoBasic.hh
	@echo Compiling XrdCryptosslCipher.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CRYPTOSSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptosslCipher.o XrdCryptosslCipher.cc

$(OBJDIR)/XrdCryptosslFactory.o: XrdCryptosslFactory.cc \
			XrdCryptosslFactory.hh XrdCryptoFactory.hh \
			XrdCryptoAux.hh XProtocol.hh \
			XPtypes.hh XrdCryptosslAux.hh \
			XrdCryptoX509Chain.hh XrdSutBucket.hh \
			XrdSutAux.hh XrdCryptoX509.hh \
			XrdCryptoRSA.hh XrdOucString.hh \
			XrdCryptoX509Crl.hh XrdCryptosslCipher.hh \
			XrdCryptoCipher.hh XrdCryptoBasic.hh \
			XrdCryptosslMsgDigest.hh XrdCryptoMsgDigest.hh \
			XrdCryptosslRSA.hh XrdCryptosslX509.hh \
			XrdCryptosslX509Crl.hh XrdSutCache.hh \
			XrdSutPFEntry.hh XrdOucHash.hh XrdOucHash.icc \
			XrdSysLogger.hh XrdSysPthread.hh \
			XrdSysError.hh XrdSutRndm.hh \
			XrdCryptosslTrace.hh XrdOucTrace.hh
	@echo Compiling XrdCryptosslFactory.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CRYPTOSSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptosslFactory.o XrdCryptosslFactory.cc


$(OBJDIR)/XrdCryptosslMsgDigest.o: XrdCryptosslMsgDigest.cc \
			XrdCryptoAux.hh XProtocol.hh \
			XPtypes.hh XrdCryptosslTrace.hh \
			XrdOucTrace.hh XrdSysError.hh \
			XrdCryptosslMsgDigest.hh XrdCryptoMsgDigest.hh \
			XrdCryptoBasic.hh XrdSutBucket.hh XrdSutAux.hh
	@echo Compiling XrdCryptosslMsgDigest.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CRYPTOSSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptosslMsgDigest.o XrdCryptosslMsgDigest.cc

$(OBJDIR)/XrdCryptosslRSA.o: XrdCryptosslRSA.cc XrdSutRndm.hh \
			XrdSutAux.hh XProtocol.hh XPtypes.hh \
			XrdCryptosslTrace.hh XrdOucTrace.hh \
			XrdSysError.hh XrdCryptoAux.hh \
			XrdCryptosslRSA.hh XrdCryptoRSA.hh \
			XrdSutBucket.hh XrdOucString.hh
	@echo Compiling XrdCryptosslRSA.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CRYPTOSSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptosslRSA.o XrdCryptosslRSA.cc

$(OBJDIR)/XrdCryptosslX509.o: XrdCryptosslX509.cc \
			XrdCryptosslRSA.hh XrdCryptoRSA.hh \
			XrdSutBucket.hh XrdSutAux.hh \
			XProtocol.hh XPtypes.hh XrdOucString.hh \
			XrdCryptoAux.hh XrdCryptosslX509.hh \
			XrdCryptoX509.hh XrdCryptosslAux.hh \
			XrdCryptoX509Chain.hh XrdCryptoX509Crl.hh \
			XrdCryptosslTrace.hh XrdOucTrace.hh \
			XrdSysError.hh
	@echo Compiling XrdCryptosslX509.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CRYPTOSSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptosslX509.o XrdCryptosslX509.cc

$(OBJDIR)/XrdCryptosslX509Req.o: XrdCryptosslX509Req.cc \
			XrdCryptosslRSA.hh XrdCryptoRSA.hh \
			XrdSutBucket.hh XrdSutAux.hh \
			XProtocol.hh XPtypes.hh XrdOucString.hh \
			XrdCryptoAux.hh XrdCryptosslX509Req.hh \
			XrdCryptoX509Req.hh XrdCryptosslAux.hh \
			XrdCryptoX509Chain.hh XrdCryptoX509.hh \
			XrdCryptoX509Crl.hh XrdCryptosslTrace.hh \
			XrdOucTrace.hh XrdSysError.hh 
	@echo Compiling XrdCryptosslX509Req.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CRYPTOSSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptosslX509Req.o XrdCryptosslX509Req.cc

$(OBJDIR)/XrdCryptosslX509Crl.o: XrdCryptosslX509Crl.cc \
			XrdCryptosslRSA.hh XrdCryptoRSA.hh \
			XrdSutBucket.hh XrdSutAux.hh \
			XProtocol.hh XPtypes.hh XrdOucString.hh \
			XrdCryptoAux.hh XrdCryptosslX509Crl.hh \
			XrdSutCache.hh XrdSutPFEntry.hh \
			XrdOucHash.hh XrdOucHash.icc \
			XrdCryptoX509Crl.hh XrdCryptoX509.hh \
			XrdCryptosslAux.hh XrdCryptoX509Chain.hh \
			XrdCryptosslTrace.hh XrdOucTrace.hh \
			XrdSysError.hh
	@echo Compiling XrdCryptosslX509Crl.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CRYPTOSSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptosslX509Crl.o XrdCryptosslX509Crl.cc


$(OBJDIR)/XrdCryptosslgsiAux.o: XrdCryptosslgsiAux.cc XrdSutRndm.hh XrdSutAux.hh \
			XProtocol.hh XPtypes.hh XrdCryptosslgsiAux.hh \
			XrdCryptosslgsiX509Chain.hh XrdCryptoX509Chain.hh XrdSutBucket.hh \
			XrdCryptoX509.hh XrdCryptoRSA.hh XrdOucString.hh \
			XrdCryptoAux.hh XrdCryptoX509Crl.hh XrdCryptoX509Req.hh \
			XrdCryptoTrace.hh XrdOucTrace.hh XrdSysError.hh \
			XrdCryptosslAux.hh XrdCryptosslRSA.hh XrdCryptosslX509.hh \
			XrdCryptosslX509Req.hh
	@echo Compiling XrdCryptosslgsiAux.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(CRYPTOSSLEXTRACFLAGS) $(INCLUDE) $(INCSSL) -o $(OBJDIR)/XrdCryptosslgsiAux.o XrdCryptosslgsiAux.cc

$(OBJDIR)/XrdCryptosslgsiX509Chain.o: XrdCryptosslgsiX509Chain.cc \
			XrdCryptosslgsiAux.hh XrdCryptosslgsiX509Chain.hh \
			XrdCryptoX509Chain.hh XrdSutBucket.hh \
			XrdSutAux.hh XProtocol.hh XPtypes.hh \
			XrdCryptoX509.hh XrdCryptoRSA.hh XrdOucString.hh \
			XrdCryptoAux.hh XrdCryptoX509Crl.hh XrdCryptoX509Req.hh \
			XrdCryptoTrace.hh XrdOucTrace.hh XrdSysError.hh
	@echo Compiling XrdCryptosslgsiX509Chain.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCryptosslgsiX509Chain.o XrdCryptosslgsiX509Chain.cc
