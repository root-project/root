// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_TemplateInstance
#define ROOT_Reflex_TemplateInstance

// Include files
#include "Reflex/Kernel.h"
#include "Reflex/Type.h"

namespace ROOT {
  namespace Reflex {

    // forward declarations

    /**
     * @class TemplateInstance TemplateInstance.h Reflex/TemplateInstance.h
     * @author Stefan Roiser
     * @date   2004-01-28
     * @ingroup Ref
     */
    class TemplateInstance {

    public:

      /** default constructor */
      TemplateInstance();


      /** constructor */
      TemplateInstance( const std::string & templateArguments );


      /** destructor */
      virtual ~TemplateInstance() {}


      /**
       * Name returns the full Name of the templated collection
       * @param  typedefexp expand typedefs or not
       * @return full Name of template collection
       */
      std::string Name( unsigned int mod = 0 ) const;


      /**
       * TemplateArgumentNth will return a pointer to the nth template argument
       * @param  nth nth template argument
       * @return pointer to nth template argument
       */
      Type TemplateArgumentNth( size_t nth ) const;


      /**
       * templateArgCount will return the number of template arguments
       * @return number of template arguments
       */
      size_t TemplateArgumentCount() const;

    private:

      /** vector of template arguments */
      std::vector < Type > fTemplateArguments;

    }; // class TemplateInstance

  } // namespace Reflex
} // namespace ROOT


//-------------------------------------------------------------------------------
inline ROOT::Reflex::TemplateInstance::TemplateInstance() 
//-------------------------------------------------------------------------------
  : fTemplateArguments( std::vector<Type>()) {}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Type
ROOT::Reflex::TemplateInstance::TemplateArgumentNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  if ( nth < fTemplateArguments.size() ) { return fTemplateArguments[ nth ]; }
  return Type();
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::TemplateInstance::TemplateArgumentCount() const {
//-------------------------------------------------------------------------------
  return fTemplateArguments.size();
}

#endif // ROOT_Reflex_TemplateInstance
