// @(#)root/meta:$Id$
// Author: Philippe Canal 15/03/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootIOCtor
#define ROOT_TRootIOCtor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootIOCtor                                                          //
//                                                                      //
// Helper class used to mark a constructor to be used by ROOT instead   //
// of the default constructor.                                          //
// If rootcint sees in a class declaration of the class MyClass:        //
//    MyClass(TRootIOCtor*);                                            //
// This constructor will be used instead of the default constructor.    //
//                                                                      //
// Also the pragma:                                                     //
//   #pragma link C++ ioctortype MyMarker;                              //
// can be used to tell rootcint that a constuctor taking a MyMarker*    //
// should be used whenever available.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TRootIOCtor 
{
   // For now intentionally empty.
};

#endif
