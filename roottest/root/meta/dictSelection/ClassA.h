// Dear emacs, this is -*- c++ -*-
/*
This test has been provided by Attila Krasznahorkay.
*/
#ifndef DICTRULES_CLASSA_H
#define DICTRULES_CLASSA_H

// ROOT include(s):
#include <Rtypes.h>

// Local include(s):
#include "DictSelection.h"
#include "Trait.h"

// Forward declaration for the dictionary rule
ENTER_ROOT_SELECTION_NS
namespace Atlas {
   template< class T, class TRAIT = typename ::Atlas::Trait< T >::Type >
   class ClassA;
}
EXIT_ROOT_SELECTION_NS

namespace Atlas {

   /// Class demonstrating how ATLAS uses Reflex dictionary rules
   ///
   /// This class is meant to demonstrate how we declare different rules
   /// for how Reflex/ROOT should handle template types in the ATLAS EDM
   /// code.
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< class T,
             class TRAIT = typename Trait< T >::Type >
   class ClassA {

   public:
      /// Default constructor
      ClassA();

      /// Get the object owned by this one
      const T* variable() const;
      /// Set the object owned by this object
      void setVariable( T* p );

   private:
      /// Pointer to the specified type
      T* m_variable;

      /// A variable of the type defined by the second template argument
      TRAIT m_trait;

      /// Some persistent variable
      Int_t m_pers;

      /// Instantiate the dictionary selection rule object
      typedef typename ROOT_SELECTION_NS::Atlas::ClassA< T, TRAIT >::self
         DictSel;

   }; // class ClassA

   /// Implementation of the constructor
   template< class T, class TRAIT >
   ClassA< T, TRAIT >::ClassA()
      : m_variable( 0 ), m_trait( 0 ), m_pers( 5 ) {

   }

   /// Implementation of the getter function
   template< class T, class TRAIT >
   const T* ClassA< T, TRAIT >::variable() const {

      return m_variable;
   }

   /// Implementation of the setter function
   template< class T, class TRAIT >
   void ClassA< T, TRAIT >::setVariable( T* p ) {

      if( m_variable ) delete m_variable;
      m_variable = p;
      return;
   }

} // namespace Atlas

/// Implementation for the dictionary selection rule
ENTER_ROOT_SELECTION_NS
namespace Atlas {
#if ROOT_VERSION_CODE < ROOT_VERSION( 5, 99, 0 )
   template< class T, class TRAIT >
   class ClassA {
   public:
      /// A helper typedef
      typedef ClassA< T, TRAIT > self;
      /// Declare the default template parameter
      ROOT_SELECTION_NS::TEMPLATE_DEFAULTS <
         ROOT_SELECTION_NS::NODEFAULT,
         typename ::Atlas::Trait< T >::Type > dummy1;
      /// We need to declare explicitly what types to select in
      /// addition
      ROOT_SELECTION_NS::NO_SELF_AUTOSELECT dummy2;
      /// Declare the transient variable(s)
      ROOT_SELECTION_NS::TRANSIENT m_trait;
      /// Declare that the dictionary for the template parameter
      /// needs to be auto-generated
      ROOT_SELECTION_NS::AUTOSELECT m_variable;
   };
#else
   template< class T, class TRAIT >
   class ClassA : KeepFirstTemplateArguments< 1 > {
   public:
      /// A helper typedef
      typedef ClassA< T, TRAIT > self;
      /// Declare the transient variable(s)
      ROOT_SELECTION_NS::MemberAttributes< kTransient > m_trait;
      /// Declare that the dictionary for the template parameter
      /// needs to be auto-generated
      ROOT_SELECTION_NS::MemberAttributes< kAutoSelected > m_variable;
   };
#endif // ROOT 5
}
EXIT_ROOT_SELECTION_NS

#endif // DICTRULES_CLASSA_H
