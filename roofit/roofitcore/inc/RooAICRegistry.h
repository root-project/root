/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAICRegistry.h,v 1.11 2007/05/11 09:11:30 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_AIC_REGISTRY
#define ROO_AIC_REGISTRY

#include <array>
#include <memory>
#include <vector>

class RooArgSet;

/// Registry for analytical integration codes.
class RooAICRegistry {

public:
   RooAICRegistry(std::size_t size = 10);
   RooAICRegistry(const RooAICRegistry &other);
   RooAICRegistry &operator=(const RooAICRegistry &other);
   RooAICRegistry(RooAICRegistry &&other) = default;
   RooAICRegistry &operator=(RooAICRegistry &&other) = default;

   int store(const std::vector<int> &codeList, RooArgSet *set1 = nullptr, RooArgSet *set2 = nullptr,
             RooArgSet *set3 = nullptr, RooArgSet *set4 = nullptr);

   struct Output {
      std::vector<int> const &codes;
      std::array<RooArgSet*, 4> sets;
   };

   /// Retrieve the array of integer codes associated with the given master code,
   /// together with the (up to four) RooArgSets associated with this master code.
   Output retrieve(int masterCode)
   {
      Output out{.codes = _clArr[masterCode], .sets = {}};
      for (std::size_t iArr = 0; iArr < 4; ++iArr) {
         out.sets[iArr] = _asArr[iArr][masterCode].get();
      }
      return out;
   }

   /// Number of stored code lists.
   std::size_t size() const { return _clArr.size(); }

protected:
   std::vector<std::vector<int>> _clArr;                          ///<! Array of array of code lists
   std::array<std::vector<std::unique_ptr<RooArgSet>>, 4> _asArr; ///<! Array of RooArgSet pointers
};

#endif
