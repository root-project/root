// @(#)root/core:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2010, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// This include must be outside of the code guard because
// Rtypes.h includes TGenericClassInfo.h which includes
// TSchemaHelper.h (this header file) and really need the
// definition of ROOT::TSchemaHelper.   So in this case,
// we need the indirect #include to really do the declaration
// and the direct #include to be a noop.
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOT_TSchemaHelper
#define ROOT_TSchemaHelper

#include <string>

namespace ROOT
{
   struct TSchemaHelper
   {
      TSchemaHelper(): fTarget(), fSourceClass(),
       fSource(), fCode(), fVersion(), fChecksum(),
       fInclude(), fEmbed(kTRUE), fFunctionPtr( 0 ),
       fAttributes() {}
      std::string fTarget;
      std::string fSourceClass;
      std::string fSource;
      std::string fCode;
      std::string fVersion;
      std::string fChecksum;
      std::string fInclude;
      Bool_t      fEmbed;
      void*       fFunctionPtr;
      std::string fAttributes;

      TSchemaHelper(const TSchemaHelper &tsh) :
       fTarget(tsh.fTarget), fSourceClass(tsh.fSourceClass),
       fSource(tsh.fSource), fCode(tsh.fCode), fVersion(tsh.fVersion),fChecksum(tsh.fChecksum),
       fInclude(tsh.fInclude), fEmbed(tsh.fEmbed), fFunctionPtr(tsh.fFunctionPtr),
       fAttributes(tsh.fAttributes) {}

      TSchemaHelper& operator=(const TSchemaHelper &) {return *this;} // Not implemented
   };
}

#endif // ROOT_TSchemaHelper
