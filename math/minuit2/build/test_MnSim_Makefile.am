## Process this file with automake to produce Makefile.in .
#
# Author: Paul_Kunz@slac.stanford.edu
#

# The following is set; otherwise, it would follow GNU conventions.
# AUTOMAKE_OPTIONS = foreign

## Subsitution from pfkeb_cxx_lib_path
CXX_LIB_PATH = @CXX_LIB_PATH@

check_PROGRAMS =	\
	test_Minuit2_DemoGaussSim   \
	test_Minuit2_DemoFumili   \
	test_Minuit2_DemoMinimizer   \
	test_Minuit2_PaulTest    \
	test_Minuit2_PaulTest2  \
	test_Minuit2_PaulTest3  \
	test_Minuit2_PaulTest4  \
	test_Minuit2_ReneTest  \
	test_Minuit2_Parallel

test_Minuit2_DemoGaussSim_SOURCES =	\
	GaussFunction.h \
	GaussRandomGen.h \
	FlatRandomGen.h \
	GaussFcn.h	\
	GaussFcn.cxx	\
	GaussDataGen.h	\
	GaussDataGen.cxx	\
	DemoGaussSim.cxx	


test_Minuit2_DemoFumili_SOURCES =	\
	GaussianModelFunction.h \
	GaussDataGen.h	\
	GaussDataGen.cxx	\
	DemoFumili.cxx			

test_Minuit2_DemoMinimizer_SOURCES = demoMinimizer.cxx

test_Minuit2_PaulTest_SOURCES =	PaulTest.cxx 

test_Minuit2_PaulTest2_SOURCES =	PaulTest2.cxx 

test_Minuit2_PaulTest3_SOURCES =	        \
		      GaussFcn2.h       \
                      GaussFcn2.cxx     \
                      PaulTest3.cxx

test_Minuit2_PaulTest4_SOURCES =	PaulTest4.cxx 

test_Minuit2_ReneTest_SOURCES =	ReneTest.cxx 

test_Minuit2_Parallel_SOURCES =	\
	GaussRandomGen.h \
	ParallelTest.cxx 


INCLUDES =					\
	-I$(top_srcdir)/inc

test_Minuit2_DemoGaussSim_LDADD =				\
	$(top_builddir)/src/libMinuit2.la

test_Minuit2_DemoGaussSim_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit2_DemoFumili_LDADD =				\
	$(top_builddir)/src/libMinuit2.la

test_Minuit2_DemoFumili_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit2_DemoMinimizer_LDADD =				\
	$(top_builddir)/src/libMinuit2.la

test_Minuit2_DemoMinimizer_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit2_PaulTest_LDADD =	 \
	GaussFcn.o			\
	$(top_builddir)/src/libMinuit2.la

test_Minuit2_PaulTest_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit2_PaulTest2_LDADD =				\
	GaussFcn.o			\
	$(top_builddir)/src/libMinuit2.la

test_Minuit2_PaulTest2_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit2_PaulTest3_LDADD =				\
	$(top_builddir)/src/libMinuit2.la

test_Minuit2_PaulTest3_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit2_PaulTest4_LDADD =				\
	$(top_builddir)/src/libMinuit2.la

test_Minuit2_PaulTest4_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit2_ReneTest_LDADD =				\
	$(top_builddir)/src/libMinuit2.la

test_Minuit2_ReneTest_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit2_Parallel_LDFLAGS =				\
	-L$(CXX_LIB_PATH)			\
	-R$(CXX_LIB_PATH)

test_Minuit2_Parallel_LDADD =				\
	$(top_builddir)/src/libMinuit2.la



AllSOURCES =				        \
	$(test_Minuit2_DemoGaussSim_SOURCES)     \
	$(test_Minuit2_DemoFumili_SOURCES)       \
	$(test_Minuit2_DemoMinimizer_SOURCES)       \
	$(test_Minuit2_PaulTest_SOURCES)		\
	$(test_Minuit2_PaulTest2_SOURCES)        \
	$(test_Minuit2_PaulTest3_SOURCES)	\
	$(test_Minuit2_PaulTest4_SOURCES)	\
	$(test_Minuit2_ReneTest_SOURCES)	\
	$(test_Minuit2_Parallel_SOURCES)		

EXTRA_DIST = paul.txt paul2.txt paul3.txt paul4.txt

# For platforms where automake dependency generation doesn't work.
depend: $(AllSOURCES)
	$(top_srcdir)/config/makedepend $(DEFS) $(INCLUDES) $(CPPFLAGS) $?

clean-local:
	-rm -rf *.C

