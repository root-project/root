// @(#)root/base:$Name:  $:$Id: TObject.h,v 1.25 2003/05/01 07:42:36 brun Exp $
// Author: Philippe Canal 24/06/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers, and al.       *
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

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TClass;
namespace ROOT {

   class TClassGenerator : public TObject {

   public:
      virtual TClass *GetClass(const char* classname, Bool_t load) = 0;
      virtual TClass *GetClass(const type_info& typeinfo, Bool_t load) = 0;

   protected:
      TClassGenerator() : TObject() {};
      ~TClassGenerator() {};
      
      ClassDef(TClassGenerator,1);  // interface for TClass generators
   };

}

#endif // ifndef ROOT_TClassGenerator
