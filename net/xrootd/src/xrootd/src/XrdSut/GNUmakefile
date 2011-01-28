# $Id: GNUmakefile,v 1.6 2010/07/30 03:19:16 abh Exp $
#-----------------------------------------------------------------------------#
#                            E n v i r o n m e n t                            #
#-----------------------------------------------------------------------------#
include ../GNUmake.env

BINLIBS = -L$(LIBDIR) -lXrdOuc -lXrdSut -lXrdSys

LIBDEPS =  $(LIBDIR)/libXrdSut.a $(LIBDIR)/libXrdOuc.a $(LIBDIR)/libXrdSys.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
SOURCES = \
        XrdSutAux.cc      \
        XrdSutCache.cc \
        XrdSutBucket.cc \
	XrdSutBuckList.cc \
	XrdSutBuffer.cc \
	XrdSutPFile.cc \
	XrdSutPFEntry.cc \
	XrdSutRndm.cc

OBJECTS = \
        $(OBJDIR)/XrdSutAux.o      \
        $(OBJDIR)/XrdSutCache.o      \
        $(OBJDIR)/XrdSutBucket.o \
	$(OBJDIR)/XrdSutBuckList.o \
	$(OBJDIR)/XrdSutBuffer.o \
	$(OBJDIR)/XrdSutPFile.o \
	$(OBJDIR)/XrdSutPFEntry.o \
	$(OBJDIR)/XrdSutRndm.o

LIBRARY = $(LIBDIR)/libXrdSut.$(TYPESHLIB)
LIBARCH = $(LIBDIR)/libXrdSut.a

TARGETS = $(LIBARCH) $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                           -
#------------------------------------------------------------------------------#

vpath XProtocol%    ../XProtocol
vpath XPty%         ../XProtocol
vpath XrdCrypto%    ../XrdCrypto
vpath XrdOuc%       ../XrdOuc
vpath XrdSec%       ../XrdSec
vpath XrdSut%       ../XrdSut
vpath XrdSys%       ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make sut "(sec utils)" done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#
$(LIBARCH): $(OBJECTS)
	@echo Creating archive $(LIBARCH) 
	$(ECHO)rm -f $(LIBARCH)
	$(ECHO)ar -rc $(LIBARCH) $(OBJECTS)
	$(ECHO)ranlib $(LIBARCH)

$(LIBRARY): $(OBJECTS) $(LIBDIR)/libXrdOuc.a
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(OBJECTS) $(LIBDIR)/libXrdOuc.a $(LDSO) -o $(LIBRARY)

$(OBJDIR)/XrdSutPFAdmin.o: XrdSutPFAdmin.cc XrdOucString.hh \
		XrdSutAux.hh XProtocol.hh XPtypes.hh \
		XrdSutPFEntry.hh XrdSutPFile.hh XrdOucHash.hh \
		XrdOucHash.icc XrdSutRndm.hh
	@echo Compiling XrdSutPFAdmin.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSutPFAdmin.o XrdSutPFAdmin.cc

$(OBJDIR)/XrdSutAux.o: XrdSutAux.cc XrdSysLogger.hh \
		XrdSysPthread.hh XrdSysError.hh XrdOucString.hh \
		XProtocol.hh XPtypes.hh XrdSutTrace.hh \
		XrdOucTrace.hh
	@echo Compiling XrdSutAux.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSutAux.o XrdSutAux.cc

$(OBJDIR)/XrdSutCache.o: XrdSutCache.cc XrdSutCache.hh \
		XPtypes.hh XrdSutPFEntry.hh XProtocol.hh \
		XrdOucHash.hh XrdSutPFile.hh XrdSutTrace.hh \
		XrdOucTrace.hh XrdSysError.hh XrdSutAux.hh
	@echo Compiling XrdSutCache.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSutCache.o XrdSutCache.cc

$(OBJDIR)/XrdSutBucket.o: XrdSutBucket.cc XrdOucString.hh \
		XrdSutBucket.hh XrdSutAux.hh XProtocol.hh \
		XPtypes.hh XrdSutTrace.hh XrdOucTrace.hh \
		XrdSysError.hh
	@echo Compiling XrdSutBucket.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSutBucket.o XrdSutBucket.cc

$(OBJDIR)/XrdSutBuckList.o: XrdSutBuckList.cc XrdSutBuckList.hh \
		XrdSutBucket.hh XrdSutAux.hh \
		XProtocol.hh XPtypes.hh
	@echo Compiling XrdSutBuckList.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSutBuckList.o XrdSutBuckList.cc

$(OBJDIR)/XrdSutBuffer.o: XrdSutBuffer.cc /usr/include/stdio.h \
		XrdSecInterface.hh XrdSecEntity.hh \
		XrdOucString.hh XrdSutBuffer.hh \
		XrdSutBuckList.hh XrdSutBucket.hh XrdSutAux.hh \
		XProtocol.hh XPtypes.hh XrdSutTrace.hh \
		XrdOucTrace.hh XrdSysError.hh
	@echo Compiling XrdSutBuffer.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSutBuffer.o XrdSutBuffer.cc

$(OBJDIR)/XrdSutPFile.o: XrdSutPFile.cc XrdSutAux.hh \
		XProtocol.hh XPtypes.hh XrdSutPFEntry.hh \
		XrdSutPFile.hh XrdOucHash.hh XrdOucHash.icc \
		XrdOucString.hh XrdSutTrace.hh XrdOucTrace.hh \
		XrdSysError.hh
	@echo Compiling XrdSutPFile.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSutPFile.o XrdSutPFile.cc

$(OBJDIR)/XrdSutPFEntry.o: XrdSutPFEntry.cc XrdSutAux.hh \
		XProtocol.hh XPtypes.hh XrdSutPFEntry.hh
	@echo Compiling XrdSutPFEntry.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSutPFEntry.o XrdSutPFEntry.cc

$(OBJDIR)/XrdSutRndm.o: XrdSutRndm.cc XrdOucString.hh \
		XrdSutRndm.hh XrdSutAux.hh XProtocol.hh \
		XPtypes.hh XrdSutTrace.hh XrdOucTrace.hh \
		XrdSysError.hh
	@echo Compiling XrdSutRndm.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSutRndm.o XrdSutRndm.cc
