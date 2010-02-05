#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author: ganis $
#######################################################################

SUBDIRS = libsslGridSite

bin_PROGRAMS = xrdsecssltest

lib_LTLIBRARIES = libXrdSecssl.la

INCLUDES = -I$(top_srcdir)/src -I${XROOTD_INCDIR}/  -IlibsslGridSite ${OPENSSL_INCDIR}
libXrdSecssl_la_SOURCES = XrdSecProtocolssl.cc XrdSecProtocolssl.hh XrdSecProtocolsslTrace.hh
libXrdSecssl_la_LIBADD = 
libXrdSecssl_la_LDFLAGS = -L${XROOTD_LIBDIR}/ -lXrdSec -lXrdSys -lXrdOuc -lXrdNet ${SSLLINKLIB} libsslGridSite/libsslGridSite.la ${OPENSSL_LIBDIR} -lcrypto -lssl ${XML2_LIBDIR} -lxml2 -module  
libXrdSecssl_la_CXXFLAGS = -Wall -Werror -Wno-deprecated-declarations ${CXXFLAGS}
libXrdSecssl_la_LIBTOOLFLAGS = --tag=disable-static

xrdsecssltest_SOURCES = XrdSecProtocolsslTest.cc
xrdsecssltest_LDADD = libXrdSecssl.la



EXTRA_DIST = bootstrap.sh getAutotools.sh README acinclude.m4 configure.ac GNUmakefile.classic

rpm:    dist
	cp $(DIST_ARCHIVES) /usr/src/redhat/SOURCES/
	rpmbuild -bb xrootd-secssl.spec

srpm:	dist
	rpmbuild -bs xrootd-secssl.spec

test:  xrdsecssltest
	if [ -d "/etc/grid-security/certificates" ] && [ -e "/etc/grid-security/hostcert.pem" ] && [ -e "/etc/grid-security/hostkey.pem" ] &&  [ -e "/tmp/x509up_u$$UID" ] ; then\
		( echo "Starting Server ..." && ./xrdsecssltest server >& /tmp/xrdsecssltest-server.log & ) && echo "Starting Client ..." && sleep 1 && ./xrdsecssltest client 2>  /tmp/xrdsecssltest-client.log ; /usr/bin/killall -9 xrdsecssltest lt-xrdsecssltest >& /dev/null || echo Server stopped! ; \
	else \
		echo "Error: you need /etc/grid-security/certificates /etc/grid-security/hostcert.pem /etc/grid-security /tmp/x509up_u$$UID";\
	fi

