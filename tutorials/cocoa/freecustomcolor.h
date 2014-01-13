#ifndef FREECUSTOMCOLOR_INCLUDED
#define FREECUSTOMCOLOR_INCLUDED

#ifndef ROOT_TError //'ROOT-style' inclusion guards.
#include "TError.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TROOT
#include "TROOT.h"
#endif

namespace ROOT {
namespace CocoaTutorials {

//___________________________________________________________
inline Int_t FindFreeCustomColorIndex()
{
   if (!gROOT) {
      //AH??? WHAT??? Should never happen! :)
      ::Error("FindFreeCustomColorIndex", "gROOT is null");
      return -1;
   }
   //Some (probably stupid) assumption about the TColor -
   //I'm trying to find some 'free' index in the range [1000, 10000).
   for (int i = 1000; i < 10000; ++i)
      if (!gROOT->GetColor(i))
         return i;
   
   return -1;
}

}//CocoaTutorials
}//ROOT

#endif
