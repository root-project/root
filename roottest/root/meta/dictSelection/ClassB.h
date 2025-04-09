// Dear emacs, this is -*- c++ -*-
/*
This test has been provided by Attila Krasznahorkay.
*/
#ifndef DICTRULES_CLASSB_H
#define DICTRULES_CLASSB_H

// Local include(s):
#include "Trait.h"

namespace Atlas {

   /// Simple class used as a template parameter for ClassA
   ///
   /// This is just a simple class meant to demonstrate how ClassA and the
   /// Trait struct is meant to operate in broad terms...
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   class ClassB {

   public:
      /// Default constructor
      ClassB(){};

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

} // namespace Atlas

#endif // DICTRULES_CLASSB_H
