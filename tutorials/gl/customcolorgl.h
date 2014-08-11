//Author: Timur Pocheptsov, 02/03/2014.

#ifndef CUSTOMCOLORGL_INCLUDED
#define CUSTOMCOLORGL_INCLUDED

#include <algorithm>

#ifndef ROOT_TError //'ROOT-style' inclusion guards.
#include "TError.h"
#endif
#ifndef ROOT_TROOT
#include "TROOT.h"
#endif

namespace ROOT {
namespace GLTutorials {

//Type T is some integer type - either Int_t or a Color_t as you wish.

//___________________________________________________________
template <typename T>
inline T FindFreeCustomColorIndex(T start = 1000)
{
   if (!gROOT) {
      //AH??? WHAT??? Should never happen! :)
      ::Error("FindFreeCustomColorIndex", "gROOT is null");
      return -1;
   }
   //Some (probably stupid) assumption about the TColor -
   //I'm trying to find some 'free' index in the range [1000, 10000).
   //Int_t(1000) - well, to make some exotic platform happy (if Int_t != int).
   for (Int_t i = std::max(start, T(1000)), e = 10000; i < e; ++i)
      if (!gROOT->GetColor(i))
         return i;

   ::Error("FindFreeCustomColorIndex", "no index found");

   return -1;
}

//
//___________________________________________________________
template <typename T, unsigned N>
inline unsigned FindFreeCustomColorIndices(T (&indices)[N])
{
   //All or none.
   T tmp[N] = {};
   tmp[0] = FindFreeCustomColorIndex<T>();
   if (tmp[0] == -1)//Not found.
      return 0;

   unsigned nFound = 1;
   for (; nFound < N; ++nFound) {
      tmp[nFound] = FindFreeCustomColorIndex(tmp[nFound - 1] + 1);//the next free color index.
      if (tmp[nFound] == -1)
         break;
   }

   if (nFound == N)
      std::copy(tmp, tmp + N, indices);

   return nFound;
}

}//GLTutorials
}//ROOT

#endif
