// Dear emacs, this is -*- c++ -*-
#ifndef DICTRULES_TRAIT_H
#define DICTRULES_TRAIT_H

// ROOT include(s):
#include <Rtypes.h>

namespace Atlas {

   /// A dummy trait structure used for demonstration purposes
   ///
   /// This class trait structure is meant to demonstrate how we use
   /// traits in the ATLAS EDM in various places. It's not really used
   /// for anything too fancy in this code...
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< class T >
   struct Trait {

      /// Default trait type
      typedef Int_t Type;

   }; // struct Trait

} // namespace Atlas

#endif // DICTRULES_TRAIT_H
