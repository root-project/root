// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_NamespaceBuilder
#define ROOT_Reflex_NamespaceBuilder

// Include files
#include "Reflex/Scope.h"

namespace ROOT{
  namespace Reflex{

    /** 
     * @class NamespaceBuilder NamespaceBuilder.h Reflex/Builder/NamespaceBuilder.h
     * @author Stefan Roiser
     * @ingroup RefBld
     * @date 30/3/2004
     */
    class NamespaceBuilder  {

    public:            

      /** constructor */
      NamespaceBuilder( const char * Name );


      /** destructor */
      virtual ~NamespaceBuilder() {}

      /** AddProperty will add a PropertyNth 
       * @param  key the PropertyNth key
       * @param  value the value of the PropertyNth
       * @return a reference to the building class
       */
      NamespaceBuilder & AddProperty( const char * key, Any value );
      NamespaceBuilder & AddProperty( const char * key, const char * value );

    private:

      /** the namespace */
      Scope fNamespace;

    }; // class NamespaceBuilder

  } // namespace Reflex
} // namespace ROOT


#endif // ROOT_Reflex_NamespaceBuilder
