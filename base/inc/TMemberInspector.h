// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   15/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemberInspector
#define ROOT_TMemberInspector


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMemberInspector                                                     //
//                                                                      //
// Abstract base class for accessing the datamembers of a class.        //
// Classes derived from this class can be given as argument to the      //
// ShowMembers() methods of ROOT classes. This feature facilitates      //
// the writing of class browsers and inspectors.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"


class TClass;


class TMemberInspector {

public:
   TMemberInspector() { }
   virtual ~TMemberInspector() { }

   virtual void Inspect(TClass *cl, const char *parent, const char *name, void *addr) = 0;

   ClassDef(TMemberInspector,0)  //ABC for inspecting class data members
};

#endif
