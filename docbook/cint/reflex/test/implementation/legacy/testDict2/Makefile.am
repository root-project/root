
AUTOMAKE_OPTIONS = foreign

INCLUDES = -I$(top_srcdir)/inc

EXTRA_DIST =          \
  Class2Dict.h        \
  ClassA.h            \
  ClassB.h            \
  ClassC.h            \
  ClassD.h            \
  ClassE.h            \
  ClassF.h            \
  ClassG.h            \
  ClassH.h            \
  ClassI.h            \
  ClassJ.h            \
  ClassK.h            \
  ClassL.h            \
  ClassM.h            \
  ClassT.h            \
  ClassSelections.h   \
  MemberPointer.h     \
  Functions.h         \
  InheritanceTree.txt \
  TemplatedClasses.h  \
  selection.xml 

check_LTLIBRARIES = libtest_Class2DictRflx.la

libtest_Class2DictRflx_la_SOURCES = Class2Dict_rflx.cpp

libtest_Class2DictRflx_la_LDFLAGS = -rpath /tmp/ -module

libtest_Class2DictRflx_la_LIBADD = $(top_srcdir)/src/libReflex.la

MOSTLYCLEANFILES = Class2Dict_rflx.cpp

Class2Dict_rflx.cpp: $(top_srcdir)/test/testDict2/selection.xml $(top_srcdir)/test/testDict2/Class2Dict.h
	$(PYTHON) $(top_srcdir)/python/genreflex/genreflex.py ./Class2Dict.h -s./selection.xml --gccxmlpath=$(GCCXMLROOT)/bin -I$(top_srcdir)/inc/
