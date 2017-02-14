// @(#)root/base:$Id$
// Author: Philippe Canal 24/06/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers, and al.       *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClassGenerator
#define ROOT_TClassGenerator

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClassGenerator                                                      //
//                                                                      //
// Objects following this interface can be passed onto the TROOT object //
// to implement a user customized way to create the TClass objects.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

class TClass;

class TClassGenerator : public TObject {

protected:
   TClassGenerator() : TObject() { }
   virtual ~TClassGenerator() { }

public:
   virtual TClass *GetClass(const char* classname, Bool_t load) = 0;
   virtual TClass *GetClass(const std::type_info& typeinfo, Bool_t load) = 0;
   virtual TClass *GetClass(const char* classname, Bool_t load, Bool_t silent);
   virtual TClass *GetClass(const std::type_info& typeinfo, Bool_t load, Bool_t silent);

   ClassDef(TClassGenerator,1);  // interface for TClass generators
};

#endif
