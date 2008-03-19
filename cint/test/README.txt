test/README.txt


Description =================================================================

  This directory contains Cint testsuite. In order for users to test Cint
  by themselves, this directory is newly set up from Cint5.15.67. Currently,
  only a subset of Cint testsuite is here.  In process of adding and refining
  this testsuite.


Files ======================================================================

  README.txt   : This file
  testall.cxx  : Test suite main program
  *.cxx,*.h    : Test programs
  testdiff.txt : Test result


How to run the test ========================================================

  Following command outputs test result in 'testdiff.txt'. testall.cxx must be
  interpreted by Cint. 

    $ cint testall.cxx

  For debugging,

    $ cint -DDEBUG testall.cxx


Tested compilers ==========================================================

  Currently, this test suite is tested with following environment.

    Cygwin 5.1      ,  g++ 3.2

  Also, it is likely that the test suite runs under Linux. But in other 
  environments, especially Windows, the test suite may not run properly.

  Windows problem: This test suite relies on file synchronization. 


Known erros ================================================================

  Following errors are known. Those are due to Cint limitations.

    explicitdtor.cxx     : base class dtor isn't called with explicit dtor call
    t694.cxx -DINTERPRET : when to evaluate default parameter value
    t695.cxx             : template specialization is not complete

  Following errors occur due to limitation of compiler or operating system.

  Win32
    t676.cxx             : recursive calls, stack too deep for Win32

  Visual C++ 6.0
    t928.cxx             : function template specialization f<int>() 

  Borland C++ 5.5
    t966.cxx             : reverse_iterator::reverence (cint limitation)

