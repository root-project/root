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

#include "RooFit/TestStatistics/SharedOffset.h"

SharedOffset::SharedOffset() : offsets_(std::make_shared<OffsetVec>()), offsets_save_(std::make_shared<OffsetVec>()) {}

void SharedOffset::clear()
{
   offsets_->clear();
   offsets_save_->clear();
}

/// When calculating an unbinned likelihood with square weights applied, a different offset
/// is necessary. Similar situations may ask for a separate offset as well. This function
/// switches between the two sets of offset values for the given component keys.
/// \note Currently we do not recalculate the offset value, so in practice swapped offsets
///       are zero.
void SharedOffset::swap(const std::vector<std::size_t> &component_keys)
{
   for (auto key : component_keys) {
      std::swap((*offsets_)[key], (*offsets_save_)[key]);
   }
}
