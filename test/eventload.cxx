// @(#)root/test:$Id$
// Author: Rene Brun   10/01/97

{
//  This macro loads the shared library libEvent.so provided in $ROOTSYS/test.
//  Before executing this macro, we assume that:
//     - you have changed your current directory to $ROOTSYS/test.
//     - you have executed Event.
//  If not, go to directory test and issue the commands:
//       make Event      to generate the executable module and shared library
//       Event           to produce the file Event.root
//

//   Load shared library created in $ROOTSYS/test/libEvent.so
   gSystem.Load("libEvent");
}
