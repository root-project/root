// @(#)root/meta:$Name:  $:$Id: TRootIoCtor.h,v 1.7 2005/06/08 21:13:48 pcanal Exp $
// Author: Philippe Canal 15/03/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootIoCtor
#define ROOT_TRootIoCtor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootIoCtor                                                          //
//                                                                      //
// Helper class used to mark a constructor to be used by ROOT instead   //
// of the default constructor.                                          //
// If rootcint sees in a class declaration of the class MyClass:        //
//    MyClass(TRootIoCtor*);                                            //
// This constructor will be used instead of the default constructor.    //
//                                                                      //
// Also the pragma:                                                     //
//   #pragma link C++ ioctortype MyMarker;                              //
// can be used to tell rootcint that a constuctor taking a MyMarker*    //
// should be used whenever available.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TRootIoCtor 
{
   // For now intentionally empty.
};

#endif
