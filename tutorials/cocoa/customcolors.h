//Author: Timur Pocheptsov, 26/03/2014.

#ifndef CUSTOMCOLORS_INCLUDED
#define CUSTOMCOLORS_INCLUDED

//'ROOT-style' include guards.
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TError
#include "TError.h"
#endif
#ifndef ROOT_TROOT
#include "TROOT.h"
#endif

//___________________________________________________________
inline Color_t FindFreeCustomColorIndex(Color_t start)
{
   //Some (probably stupid) assumption about the TColor -
   //I'm trying to find some 'free' index in the range [1000, 10000).
   for (Color_t i = start; i < (Color_t)10000; ++i)
      if (!gROOT->GetColor(i))
         return i;

   ::Error("FindFreeCustomColorIndex", "no index found");
   return -1;
}

//___________________________________________________________
inline unsigned FindFreeCustomColorIndices(unsigned nColors, Color_t *indices)
{
   if (!nColors)
      return 0;

   if (!indices) {
      ::Error("FindFreeCustomColorIndices", "parameter 'indices' is null");
      return 0;
   }

   indices[0] = FindFreeCustomColorIndex(1000);
   if (indices[0] == -1)//Not found.
      return 0;

   unsigned nFound = 1;
   for (; nFound < nColors; ++nFound) {
      indices[nFound] = FindFreeCustomColorIndex(indices[nFound - 1] + 1);//the next free color index.
      if (indices[nFound] == -1)
         break;
   }
   
   return nFound;
}

#endif
