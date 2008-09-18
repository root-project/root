// @(#)root/core:$Id$
// author: Lukasz Janyst <ljanyst@cern.ch>

#ifndef ROOT_TSchemaHelper
#define ROOT_TSchemaHelper

#include <string>

namespace ROOT
{
   struct TSchemaHelper
   {
      TSchemaHelper(): fEmbed(kTRUE), fFunctionPtr( 0 ) {}
      std::string fTarget;
      std::string fSourceClass;
      std::string fSource;
      std::string fCode;
      std::string fVersion;
      std::string fChecksum;
      std::string fInclude;
      bool        fEmbed;
      void*       fFunctionPtr;
   };
}

#endif // ROOT_TSchemaHelper
