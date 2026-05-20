// Dear emacs, this is -*- c++ -*-
#ifndef DICTRULES_CLASSB_H
#define DICTRULES_CLASSB_H

// Local include(s):
#include "Trait.h"
#include "ClassC_ex2.h"
#include "ClassD_ex2.h"

namespace Atlas {

   /// Simple class used as a template parameter for ClassA
   ///
   /// This is just a simple class meant to demonstrate how ClassA and the
   /// Trait struct is meant to operate in broad terms...
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   class ClassB : public ClassC {

   public:
      /// Default constructor
      ClassB();

      /// Get the first variable
      Int_t var1() const;
      /// Set the first variable
      void setVar1( Int_t value );

      /// Get the second variable
      Float_t var2() const;
      /// Set the second variable
      void setVar2( Float_t value );

   private:
      Int_t m_var1; ///< The first variable
      Float_t m_var2; ///< The second variable

   }; // class ClassB

   /// Specialisation of the trait structure
   template<>
   struct Trait< ClassB > {

      /// Custom trait type
      typedef Float_t Type;

   }; // struct Trait< ClassB >

   /// Specialisation of the trait structure
   template<>
   struct Base< ClassB > {

      /// The base class type
      typedef ClassD< ClassC > Type;

   }; // struct Base< ClassB >

} // namespace Atlas


namespace Atlas {

   ClassB::ClassB()
      : m_var1( 0 ), m_var2( 0. ) {

   }

   Int_t ClassB::var1() const {

      return m_var1;
   }

   void ClassB::setVar1( Int_t value ) {

      m_var1 = value;
      return;
   }

   Float_t ClassB::var2() const {

      return m_var2;
   }

   void ClassB::setVar2( Float_t value ) {

      m_var2 = value;
      return;
   }

} // namespace Atlas


#endif // DICTRULES_CLASSB_H
