// @(#)root/reflex:$Name:  $:$Id: $
// Author: Pere Mato 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_PluginFactoryMap
#define ROOT_Reflex_PluginFactoryMap

#include <string>
#include <list>
#include <map>

namespace ROOT {
   namespace Reflex {  

      /** 
       * @class PluginFactoryMap PluginFactoryMap.h PluginFactoryMap.h
       * @author Pere Mato
       * @date 01/09/2006
       * @ingroup Ref
       */
      class PluginFactoryMap {

      public:

        PluginFactoryMap(const std::string& path = "");

        ~PluginFactoryMap();

        std::list<std::string> GetLibraries(const std::string& name) const;

        void FillMap(const std::string& filename);

        static void SetDebug( int );

        static int Debug();

      private:

        static int fgDebugLevel;

      }; // class PluginFactoryMap

   }  // namespace Reflex
}  // namespace ROOT

#endif // ROOT_Reflex_PluginFactoryMap
