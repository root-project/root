#ifndef CUSTOMCOLOR_INCLUDED
#define CUSTOMCOLOR_INCLUDED

#include <algorithm>

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
inline Int_t FindFreeCustomColorIndex(Int_t start = 1000)
{
   if (!gROOT) {
      //AH??? WHAT??? Should never happen! :)
      ::Error("FindFreeCustomColorIndex", "gROOT is null");
      return -1;
   }
   //Some (probably stupid) assumption about the TColor -
   //I'm trying to find some 'free' index in the range [1000, 10000).
   //Int_t(1000) - well, to make some exotic platform happy (if Int_t != int).
   for (Int_t i = std::max(start, Int_t(1000)); i < 10000; ++i)
      if (!gROOT->GetColor(i))
         return i;

   ::Error("FindFreeCustomColorIndex", "no index found");

   return -1;
}

//
//Ho-ho-ho! Say good-bye to CINT and hello CLING ... and good old templates!!!
//___________________________________________________________
template <unsigned N>
inline unsigned FindFreeCustomColorIndices(Int_t (&indices)[N])
{
   //All or none.
   Int_t tmp[N] = {};
   tmp[0] = FindFreeCustomColorIndex();
   if (tmp[0] == -1)//Not found.
      return 0;

   unsigned nFound = 0;
   for (nFound = 1; nFound < N; ++nFound) {
      tmp[nFound] = FindFreeCustomColorIndex(tmp[nFound - 1] + 1);//the next free color index.
      if (tmp[nFound] == -1)
         break;
   }
   
   if (nFound == N)
      std::copy(tmp, tmp + N, indices);

   return nFound;
}

}//CocoaTutorials
}//ROOT

#endif
