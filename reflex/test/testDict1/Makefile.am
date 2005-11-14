
AUTOMAKE_OPTIONS = foreign

INCLUDES = -I$(top_srcdir)/inc

EXTRA_DIST = selection.xml

check_LTLIBRARIES = libtest_Reflex1Rflx.la

libtest_Reflex1Rflx_la_SOURCES = Reflex_rflx.cpp

libtest_Reflex1Rflx_la_LDFLAGS = -rpath /tmp/ -module

libtest_Reflex1Rflx_la_LIBADD = $(top_srcdir)/src/libReflex.la

MOSTLYCLEANFILES = Reflex_rflx.cpp

Reflex_rflx.cpp:
	$(PYTHON) $(top_srcdir)/python/genreflex/genreflex.py $(top_srcdir)/inc/Reflex/Reflex.h -s$(top_srcdir)/test/testDict1/selection.xml --gccxmlpath=$(GCCXMLROOT)/bin -I$(top_srcdir)/inc/

