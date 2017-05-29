// @(#)root/cont
// Author: Philippe Canal Aug 2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TListOfTypes
#define ROOT_TListOfTypes

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfTypes                                                         //
//                                                                      //
// A collection of TDataType designed to hold the typedef information   //
// and numerical type information.  The collection is populated on      //
// demand.                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "THashTable.h"

class TDataType;

class TListOfTypes : public THashTable
{
public:
   TListOfTypes();

   using THashTable::FindObject;
   virtual TObject   *FindObject(const char *name) const;

   TDataType *FindType(const char *name) const;
};


#endif

