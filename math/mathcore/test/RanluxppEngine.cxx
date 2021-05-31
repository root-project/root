// @(#)root/mathcore:$Id$
// Author: Jonas Hahnfeld 11/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// This test uses EXPECT_EQ also for floating point numbers - the expected
// values are entered with enough digits to ensure binary equality.

#include "Math/RanluxppEngine.h"

#include "gtest/gtest.h"

using namespace ROOT::Math;

TEST(RanluxppEngine, random2048)
{
   RanluxppEngine2048 rng(314159265);

   // The following values were obtained without skipping.

   EXPECT_EQ(rng.IntRndm(), 39378223178113);
   EXPECT_EQ(rng.Rndm(), 0.57072241146576274673);

   // Skip ahead in block.
   rng.Skip(8);
   EXPECT_EQ(rng.IntRndm(), 52221857391813);
   EXPECT_EQ(rng.Rndm(), 0.16812543081078956675);

   // The next call needs to advance the state.
   EXPECT_EQ(rng.IntRndm(), 185005245121693);
   EXPECT_EQ(rng.Rndm(), 0.28403302782895423206);

   // Skip ahead to start of next block.
   rng.Skip(10);
   EXPECT_EQ(rng.IntRndm(), 89237874214503);
   EXPECT_EQ(rng.Rndm(), 0.79969842495805920635);

   // Skip ahead across blocks.
   rng.Skip(42);
   EXPECT_EQ(rng.IntRndm(), 49145148745150);
   EXPECT_EQ(rng.Rndm(), 0.74670661284082484599);
}
