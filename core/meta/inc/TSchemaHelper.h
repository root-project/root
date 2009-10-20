// @(#)root/core:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

#ifndef ROOT_TSchemaHelper
#define ROOT_TSchemaHelper

#include <string>
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace ROOT
{
   struct TSchemaHelper
   {
      TSchemaHelper(): fTarget(), fSourceClass(),
       fSource(), fCode(), fVersion(), fChecksum(),
       fInclude(), fEmbed(kTRUE), fFunctionPtr( 0 ) {}
      std::string fTarget;
      std::string fSourceClass;
      std::string fSource;
      std::string fCode;
      std::string fVersion;
      std::string fChecksum;
      std::string fInclude;
      bool        fEmbed;
      void*       fFunctionPtr;

      TSchemaHelper(const TSchemaHelper &tsh) :
       fTarget(tsh.fTarget), fSourceClass(tsh.fSourceClass),
       fSource(tsh.fSource), fCode(tsh.fCode), fVersion(tsh.fVersion), fChecksum(tsh.fChecksum),
       fInclude(tsh.fInclude), fEmbed(tsh.fEmbed), fFunctionPtr(tsh.fFunctionPtr) {}

      TSchemaHelper& operator=(const TSchemaHelper &) {return *this;} // Not implemented
   };
}

#endif // ROOT_TSchemaHelper
