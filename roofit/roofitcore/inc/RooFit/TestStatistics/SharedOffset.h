/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_ROOFIT_TESTSTATISTICS_SHAREDOFFSET_H
#define ROOT_ROOFIT_TESTSTATISTICS_SHAREDOFFSET_H

#include <Math/Util.h>
#include <vector>
#include <memory> // shared_ptr

class SharedOffset {
public:
   SharedOffset();
   using OffsetVec = std::vector<ROOT::Math::KahanSum<double>>;

   void clear();
   void swap(const std::vector<std::size_t> &component_keys);

   inline OffsetVec &offsets() { return *offsets_; };
   inline OffsetVec &offsets_save() { return *offsets_save_; };

private:
   std::shared_ptr<OffsetVec> offsets_;
   std::shared_ptr<OffsetVec> offsets_save_;
};

#endif // ROOT_ROOFIT_TESTSTATISTICS_SHAREDOFFSET_H
