// @(#)root/base:$Id$
// Author: Philippe Canal 24/06/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClassGenerator                                                      //
//                                                                      //
// Objects following this interface can be passed onto the TROOT object //
// to implement a user customized way to create the TClass objects.     //
//                                                                      //
// Use TROOT::AddClassGenerator to register a concrete instance.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TClassGenerator.h"

ClassImp(TClassGenerator);

//////////////////////////////////////////////////////////////////////////
TClass *TClassGenerator::GetClass(const char* classname, Bool_t load, Bool_t /* silent */)
{
   // Default implementation for backward compatibility ignoring the value of 'silent'
   return GetClass(classname,load);
}

//////////////////////////////////////////////////////////////////////////
TClass *TClassGenerator::GetClass(const type_info& typeinfo, Bool_t load, Bool_t /* silent */)
{
   // Default implementation for backward compatibility ignoring the value of 'silent'
   return GetClass(typeinfo,load);
}
