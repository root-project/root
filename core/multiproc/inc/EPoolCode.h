/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_EPoolCode
#define ROOT_EPoolCode

//////////////////////////////////////////////////////////////////////////
///
/// An enumeration of the message codes handled by TPool and TPoolServer.
///
//////////////////////////////////////////////////////////////////////////

enum EPoolCode : unsigned {
   kExecFunc = 0,    ///< Execute function without arguments
   kExecFuncWithArg, ///< Execute function with argument
   kFuncResult       ///< Result of a function execution
};

#endif
