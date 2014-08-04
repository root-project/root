/* @(#)root/pyroot:$Id$ */
/* Author: Wim Lavrijsen, Apr 2008 */

/*
   Python main program, used to create a 64-bit python executable
   (bin/python64) on MacOS X 64.
   The standard python on MacOS X 64 is 32-bit only and hence cannot
   load any 64-bit python modules, like PyROOT.
*/


#include "Python.h"

int main(int argc, char* argv[])
{
   return Py_Main( argc, argv );
}
