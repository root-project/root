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
      kNotSTL         = 0,
      kSTLvector      = 1,
      kSTLlist        = 2,
      kSTLforwardlist = 3,
      kSTLdeque       = 4,
      kSTLmap         = 5,
      kSTLmultimap    = 6,
      kSTLset         = 7,
      kSTLmultiset    = 8,
      kSTLbitset      = 9,
      kSTLend         = 10,
      kSTLany         = 300 /* TVirtualStreamerInfo::kSTL */,
      kSTLstring      = 365 /* TVirtualStreamerInfo::kSTLstring */
   };

}

#endif
