dnl @synopsis PFK_CXX_LIB_PATH
dnl
dnl Sets the output variable CXXLIB_PATH to the path of the Standard C++ 
dnl library used by the compiler.   Basically if $CXX is found in say
dnl `/usr/local/bin' then the assumtion is that its library is found in 
dnl `/usr/local/lib'.
dnl
dnl @author Paul_Kunz@slac.stanford.edu
dnl
AC_DEFUN(PFK_CXX_LIB_PATH,
[ AC_PATH_PROG(pfk_cxx_lib_path, $CXX, $CXX )
  AC_MSG_CHECKING(standard C++ library path)
  CXX_LIB_PATH=`dirname $pfk_cxx_lib_path | sed -e "s/bin/lib/"`
  AC_MSG_RESULT($CXX_LIB_PATH )
  AC_SUBST(CXX_LIB_PATH)dnl
])
