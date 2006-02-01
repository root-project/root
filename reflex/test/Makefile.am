
AUTOMAKE_OPTIONS = foreign

SUBDIRS = testDict1 testDict2

INCLUDES = -I$(top_srcdir)/inc -I$(top_srcdir) -I$(CPPUNITROOT)/include

TESTS = test_Reflex_generate test_Reflex_simple1 test_Reflex_simple2 test_Reflex_unit test_ReflexBuilder_unit

TESTS_ENVIRONMENT = DYLD_LIBRARY_PATH="$(srcdir)/testDict1/.libs:$(srcdir)/testDict2/.libs"

check_PROGRAMS = test_Reflex_generate test_Reflex_simple1 test_Reflex_simple2 test_Reflex_unit test_ReflexBuilder_unit

test_Reflex_generate_SOURCES = test_Reflex_generate.cxx
test_Reflex_generate_LDFLAGS = -export-dynamic 
test_Reflex_generate_LDADD = $(LIBLTDL) "-dlopen" $(top_srcdir)/test/testDict2/libtest_Class2DictRflx.la $(top_srcdir)/src/libReflex.la
test_Reflex_generate_DEPENDENCIES = $(LIBLTDL) $(top_srcdir)/test/testDict2/libtest_Class2DictRflx.la $(top_srcdir)/src/libReflex.la

test_Reflex_simple1_SOURCES = test_Reflex_simple1.cxx
test_Reflex_simple1_LDFLAGS = -export-dynamic
test_Reflex_simple1_LDADD = $(LIBLTDL) "-dlopen" $(top_srcdir)/test/testDict1/libtest_ReflexRflx.la $(top_srcdir)/src/libReflex.la $(CPPUNITROOT)/lib/libcppunit.la
test_Reflex_simple1_DEPENDENCIES = $(LIBLTDL) $(top_srcdir)/test/testDict1/libtest_ReflexRflx.la $(top_srcdir)/src/libReflex.la $(CPPUNITROOT)/lib/libcppunit.la

test_Reflex_simple2_SOURCES = test_Reflex_simple2.cxx
test_Reflex_simple2_LDFLAGS = -export-dynamic
test_Reflex_simple2_LDADD = $(LIBLTDL) "-dlopen" $(top_srcdir)/test/testDict2/libtest_Class2DictRflx.la $(top_srcdir)/src/libReflex.la $(CPPUNITROOT)/lib/libcppunit.la
test_Reflex_simple2_DEPENDENCIES = $(LIBLTDL) $(top_srcdir)/test/testDict2/libtest_Class2DictRflx.la $(top_srcdir)/src/libReflex.la $(CPPUNITROOT)/lib/libcppunit.la

test_Reflex_unit_SOURCES = test_Reflex_unit.cxx
test_Reflex_unit_LDADD = $(top_srcdir)/src/libReflex.la $(CPPUNITROOT)/lib/libcppunit.la
test_Reflex_unit_DEPENDENCIES = $(top_srcdir)/src/libReflex.la $(CPPUNITROOT)/lib/libcppunit.la

test_ReflexBuilder_unit_SOURCES = test_ReflexBuilder_unit.cxx
test_ReflexBuilder_unit_LDADD = $(top_srcdir)/src/libReflex.la $(CPPUNITROOT)/lib/libcppunit.la
test_ReflexBuilder_unit_DEPENDENCIES = $(top_srcdir)/src/libReflex.la $(CPPUNITROOT)/lib/libcppunit.la
