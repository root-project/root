/*
 * Project: RooFit
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_Algorithms_h
#define RooFit_Detail_Algorithms_h

#include <vector>

namespace RooFit {
namespace Detail {

//_____________________________________________________________________________
// from http://stackoverflow.com/a/5279601
template <typename T>
void cartesianProduct(std::vector<std::vector<T>> &out, std::vector<std::vector<T>> &in)
{
   struct Digits {
      typename std::vector<T>::const_iterator begin;
      typename std::vector<T>::const_iterator end;
      typename std::vector<T>::const_iterator me;
   };

   std::vector<Digits> vd;
   vd.reserve(in.size());

   for (auto it = in.begin(); it != in.end(); ++it) {
      Digits d = {(*it).begin(), (*it).end(), (*it).begin()};
      vd.push_back(d);
   }

   while (true) {
      std::vector<T> result;
      for (auto it = vd.begin(); it != vd.end(); ++it) {
         result.push_back(*(it->me));
      }
      out.push_back(result);

      for (auto it = vd.begin();;) {
         ++(it->me);
         if (it->me == it->end) {
            if (it + 1 == vd.end()) {
               return;
            } else {
               it->me = it->begin;
               ++it;
            }
         } else {
            break;
         }
      }
   }
}

//_____________________________________________________________________________
// from http://stackoverflow.com/a/5097100/8747
template <typename Iterator>
bool nextCombination(const Iterator first, Iterator k, const Iterator last)
{
   if ((first == last) || (first == k) || (last == k)) {
      return false;
   }
   Iterator itr1 = first;
   Iterator itr2 = last;
   ++itr1;
   if (last == itr1) {
      return false;
   }
   itr1 = last;
   --itr1;
   itr1 = k;
   --itr2;
   while (first != itr1) {
      if (*--itr1 < *itr2) {
         Iterator j = k;
         while (!(*itr1 < *j))
            ++j;
         iter_swap(itr1, j);
         ++itr1;
         ++j;
         itr2 = k;
         rotate(itr1, j, last);
         while (last != j) {
            ++j;
            ++itr2;
         }
         rotate(k, itr2, last);
         return true;
      }
   }
   rotate(first, k, last);
   return false;
}

} // namespace Detail
} // namespace RooFit

#endif
