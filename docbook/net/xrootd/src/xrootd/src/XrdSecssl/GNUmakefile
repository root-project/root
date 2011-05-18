#              $Id$

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

MORELIBS = -L$(LIBDIR) -lXrdSec        $(LIBDIR)/libsslGridSite.a \
             $(LIBDIR)/libXrdNet.a     $(LIBDIR)/libXrdOuc.a \
             $(LIBDIR)/libXrdNetUtil.a $(LIBDIR)/libXrdSys.a \
             $(LIBSSL) $(LIBXML)

LIBDEPS = $(LIBDIR)/libsslGridSite.a  $(LIBDIR)/libXrdNet.a \
          $(LIBDIR)/libXrdNetUtil.a   $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdSys.a

NCXXFLAGS = $(INCSSL) $(SSLEXTRACFLAGS) -DOPENSSL_NO_KRB5 $(subst deprecated,deprecated-declarations,$(CFLAGS)) 
NCFLAGS = $(INCXML) $(INCSSL) $(SSLEXTRACFLAGS) -DOPENSSL_NO_KRB5 -U_FORTIFY_SOURCE $(subst deprecated,deprecated-declarations,$(CFLAGS)) 
 
#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

CCOMPILE = cc

SOURCES = \
        XrdSecProtocolssl.cc \
	XrdSecProtocolsslProc.cc \
        XrdSecProtocolsslProc.hh \
        XrdSecProtocolsslTest.cc \
        libsslGridSite/grst_asn1.c \
        libsslGridSite/grst_err.c \
        libsslGridSite/grst_gacl.c \
        libsslGridSite/grst_http.c \
        libsslGridSite/grst_verifycallback.c \
        libsslGridSite/grst_x509.c \
        libsslGridSite/grst_xacml.c \

OBJECTA = \
        $(OBJDIR)/XrdSecProtocolsslTest.o

OBJGRST = \
        $(OBJDIR)/grst_asn1.o \
        $(OBJDIR)/grst_err.o \
        $(OBJDIR)/grst_gacl.o \
        $(OBJDIR)/grst_http.o \
        $(OBJDIR)/grst_verifycallback.o \
        $(OBJDIR)/grst_x509.o \
        $(OBJDIR)/grst_xacml.o\

OBJECTS = $(OBJGRST) $(OBJECTA) \
        $(OBJDIR)/XrdSecProtocolssl.o \
        $(OBJDIR)/XrdSecProtocolsslProc.o


LIBGRST = $(LIBDIR)/libsslGridSite.a
LIBRARY = $(LIBDIR)/libXrdSecssl.$(TYPESHLIB)

SECSSLTEST = $(BINDIR)/xrdsecssltest

TARGETS = $(LIBGRST) $(LIBRARY) $(SECSSLTEST)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdOuc% ../XrdOuc
vpath XrdSec% ../XrdSec
vpath XrdSys% ../XrdSys
vpath XrdNet% ../XrdNet

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdSecssl done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(LIBRARY): $(OBJECTS) $(LIBDEPS)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(OBJECTS) $(LDSO) $(MORELIBS) $(LIBS) -o $(LIBRARY);

$(LIBGRST): $(OBJGRST)
	@echo Creating archive $(LIBGRST)
	$(ECHO)rm -f $(LIBGRST)
	$(ECHO)ar -rc $(LIBGRST) $(OBJGRST)
	$(ECHO)ranlib $(LIBGRST)

$(SECSSLTEST): $(OBJECTA) $(LIBDEPS)
	@echo Creating executable $(SECSSLTEST)
	$(ECHO)$(LD) $(LDOP) $(OBJECTA) -L$(LIBDIR) -lXrdSecssl $(MORELIBS) $(LIBS) -o $(SECSSLTEST)

$(OBJDIR)/XrdSecProtocolsslProc.o:  XrdSecProtocolsslProc.cc 	XrdSecProtocolsslProc.hh \
			        XrdOucString.hh 	XrdOucHash.hh

	@echo Compiling XrdSecProtocolsslProc.cc
	$(ECHO)$(CC) -c $(NCXXFLAGS)  $(INCLUDE) -o $(OBJDIR)/XrdSecProtocolsslProc.o XrdSecProtocolsslProc.cc

$(OBJDIR)/XrdSecProtocolssl.o:  XrdSecProtocolssl.cc 	XrdSecProtocolssl.hh 	XrdSecInterface.hh \
                                XrdSecProtocolsslProc.cc XrdSecProtocolsslProc.hh \
                                XrdSysHeaders.hh 	XrdNetDNS.hh 		XrdOucErrInfo.hh XrdOucHash.hh \
				XrdOucString.hh 	XrdOucTrace.hh 		XrdOucTokenizer.hh XrdSysPthread.hh \
				XrdSysLogger.hh 	XrdSecInterface.hh 	XrdSecTLayer.hh \
                                XrdSecEntity.hh       	XrdOucErrInfo.hh	XrdSecTLayer.hh \
				libsslGridSite/grst_asn1.c	libsslGridSite/grst_err.c \
				libsslGridSite/grst_gacl.c	libsslGridSite/grst_http.c \
				libsslGridSite/grst_verifycallback.c libsslGridSite/grst_x509.c \
				libsslGridSite/grst_xacml.c libsslGridSite/gridsite.h
	@echo Compiling XrdSecProtocolssl.cc 
	$(ECHO)$(CC) -c $(NCXXFLAGS)  $(INCLUDE) -o $(OBJDIR)/XrdSecProtocolssl.o XrdSecProtocolssl.cc

$(OBJDIR)/XrdSecProtocolsslTest.o:  XrdSecProtocolsslTest.cc 	XrdSecProtocolssl.hh 	XrdSecInterface.hh \
                                XrdSysHeaders.hh 	XrdNetDNS.hh 		XrdOucErrInfo.hh XrdOucHash.hh \
				XrdOucString.hh 	XrdOucTrace.hh 		XrdOucTokenizer.hh XrdSysPthread.hh \
				XrdSysLogger.hh 	XrdSecInterface.hh 	XrdSecTLayer.hh \
                                XrdSecEntity.hh       	XrdOucErrInfo.hh	XrdSecTLayer.hh \
				libsslGridSite/grst_asn1.c	libsslGridSite/grst_err.c \
				libsslGridSite/grst_gacl.c	libsslGridSite/grst_http.c \
				libsslGridSite/grst_verifycallback.c libsslGridSite/grst_x509.c \
				libsslGridSite/grst_xacml.c libsslGridSite/gridsite.h
	@echo Compiling XrdSecProtocolsslTest.cc
	$(ECHO)$(CC) -c $(NCXXFLAGS)  $(INCLUDE) -o $(OBJDIR)/XrdSecProtocolsslTest.o XrdSecProtocolsslTest.cc

$(OBJDIR)/grst_asn1.o:  	libsslGridSite/grst_asn1.c libsslGridSite/gridsite.h
	@echo Compiling grst_asn1.c 
	$(ECHO)$(CCOMPILE)  -c $(NCFLAGS)  $(INCLUDE) -o $(OBJDIR)/grst_asn1.o libsslGridSite/grst_asn1.c

$(OBJDIR)/grst_err.o:  		libsslGridSite/grst_err.c libsslGridSite/gridsite.h
	@echo Compiling grst_err.c
	$(ECHO)$(CCOMPILE)  -c $(NCFLAGS)  $(INCLUDE) -o $(OBJDIR)/grst_err.o libsslGridSite/grst_err.c

$(OBJDIR)/grst_gacl.o:  	libsslGridSite/grst_gacl.c libsslGridSite/gridsite.h
	@echo Compiling grst_gacl.c
	$(ECHO)$(CCOMPILE)  -c $(NCFLAGS)  $(INCLUDE) -o $(OBJDIR)/grst_gacl.o libsslGridSite/grst_gacl.c

$(OBJDIR)/grst_http.o:  	libsslGridSite/grst_http.c libsslGridSite/gridsite.h
	@echo Compiling grst_http.c
	$(ECHO)$(CCOMPILE)  -c $(NCFLAGS)  $(INCLUDE) -o $(OBJDIR)/grst_http.o libsslGridSite/grst_http.c

$(OBJDIR)/grst_verifycallback.o:  libsslGridSite/grst_verifycallback.c libsslGridSite/gridsite.h
	@echo Compiling grst_verifycallback.c
	$(ECHO)$(CCOMPILE)  -c $(NCFLAGS)  $(INCLUDE) -o $(OBJDIR)/grst_verifycallback.o libsslGridSite/grst_verifycallback.c

$(OBJDIR)/grst_x509.o:  	libsslGridSite/grst_x509.c libsslGridSite/gridsite.h
	@echo Compiling grst_x509.c
	$(ECHO)$(CCOMPILE)  -c $(NCFLAGS)  $(INCLUDE) -o $(OBJDIR)/grst_x509.o libsslGridSite/grst_x509.c

$(OBJDIR)/grst_xacml.o:  	libsslGridSite/grst_xacml.c libsslGridSite/gridsite.h
	@echo Compiling grst_xacml.c
	$(ECHO)$(CCOMPILE)  -c $(NCFLAGS)  $(INCLUDE) -o $(OBJDIR)/grst_xacml.o libsslGridSite/grst_xacml.c
