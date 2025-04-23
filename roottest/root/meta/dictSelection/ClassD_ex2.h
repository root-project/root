// Dear emacs, this is -*- c++ -*-
#ifndef DICTRULES_CLASSD_H
#define DICTRULES_CLASSD_H

// System include(s):
#include <vector>

// Local include(s):
#include "DictSelection.h"

// Forward declaration for the dictionary rule
ENTER_ROOT_SELECTION_NS
namespace Atlas {
   template< class T, class BASE >
   class ClassD;
}
EXIT_ROOT_SELECTION_NS

namespace Atlas {

   /// Struct marking classes that have no defied base class
   struct NoBase {};

   /// A trait class used by ClassD to implement a "vector inheritance"
   ///
   /// Used pretty much in the same manner as DataVectorBase in the
   /// real Atlas EDM.
   ///
   template< typename T >
   struct Base {
      typedef NoBase Type;
   };

   /// A poor man's version of ATLAS's DataVector class
   ///
   /// This class is meant to demonstrate how ATLAS's DataVector class
   /// is used, and what the dictionary infrastructure needs to be able
   /// to handle.
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< class T, class BASE = typename Base< T >::Type >
   class ClassD : public BASE {

   public:
      /// A convenience type definition
      typedef T* value_type;
      /// A convenience type definition
      typedef typename BASE::size_type size_type;

      /// Default constructor
      ClassD() : BASE() {}

   private:
      /// Instantiate the dictionary selection rule object
      typedef typename ROOT_SELECTION_NS::Atlas::ClassD< T, BASE >::self
         DictSel;

   }; // class ClassD

   /// Specialisation of the class for types not having a base class
   ///
   /// Again, following the DataVector design here...
   ///
   template< class T >
   class ClassD< T, Atlas::NoBase > {

   public:
      /// A convenience type definition
      typedef T* value_type;
      /// A convenience type definition
      typedef size_t size_type;

      /// Default constructor
      ClassD() : m_vec() {}

   protected:
      /// The vector holding the pointers
      std::vector< value_type > m_vec;

      /// Instantiate the dictionary selection rule object
      typedef typename ROOT_SELECTION_NS::Atlas::ClassD< T, Atlas::NoBase >::self
         DictSel;

   }; // class D

} // namespace Atlas

/// Implementation for the dictionary selection rule
ENTER_ROOT_SELECTION_NS
namespace Atlas {
#if ROOT_VERSION_CODE < ROOT_VERSION( 5, 99, 0 )
   template< class T, class BASE >
   class ClassD {
   public:
      /// A helper typedef
      typedef ClassD< T, BASE > self;
      /// Declare the default template parameter
      ROOT_SELECTION_NS::TEMPLATE_DEFAULTS <
         ROOT_SELECTION_NS::NODEFAULT,
         typename ::Atlas::Base< T >::Type > dummy1;
      /// We need to declare explicitly what types to select in
      /// addition
      ROOT_SELECTION_NS::NO_SELF_AUTOSELECT dummy2;
      /// Declare the transient variable(s)
      ROOT_SELECTION_NS::TRANSIENT m_vec;
   };
#else
   template< class T, class BASE >
   class ClassD : KeepFirstTemplateArguments< 1 > {
   public:
      /// A helper typedef
      typedef ClassD< T, BASE > self;
      /// Declare the transient variable(s)
      ROOT_SELECTION_NS::MemberAttributes< kTransient > m_vec;
   };
#endif // ROOT 5
}
EXIT_ROOT_SELECTION_NS

#endif // DICTRULES_CLASSD_H
