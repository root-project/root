/// \file testspan.cxx
///
/// \brief The file contain unit tests which test the ROOT::RSpan
///
/// \author Ivan Kabadzhov
///
/// \date Feb, 2022
///
/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RSpan.hxx>

#include "gtest/gtest.h"

TEST(SpanConstructors, Vectors)
{
   std::vector<int> v = {1, 2, 3};
   std::span<int> v_span = v;

   const std::vector<int> const_v = {1, 2, 3};
   std::span<int> const_v_span = const_v;

   EXPECT_EQ(v_span, const_v_span);

   std::vector<int> empty = {};
   std::span<int> empty_span = empty;

   const std::vector<int> const_empty = {};
   std::span<int> const_empty_span = const_empty;

   EXPECT_EQ(empty_span, const_empty_span);
}
