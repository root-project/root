// @(#)root/metautils:
// Author: Philippe Canal November 2013

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_ESTLType
#define ROOT_ESTLType


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ROOT::ESTLType                                                       //
//                                                                      //
// Enum describing STL collections and some std classes                 //
// This is used in TClassEdit, TStreamerInfo, TClassEdit                //
// and TStreamerElement.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace ROOT {

   enum ESTLType {
      kNotSTL      = 0,
      kSTLvector   = 1,
      kSTLlist     = 2,
      kSTLdeque    = 3,
      kSTLmap      = 4,
      kSTLmultimap = 5,
      kSTLset      = 6,
      kSTLmultiset = 7,
      kSTLbitset   = 8,
      kSTLany      = 300 /* TVirtualStreamerInfo::kSTL */,
      kSTLstring   = 365 /* TVirtualStreamerInfo::kSTLstring */
   };

}

#endif