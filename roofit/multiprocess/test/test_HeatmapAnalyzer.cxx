/*
 * Project: RooFit
 * Authors:
 *   ZW, Zef Wolffs, Nikhef, zefwolffs@gmail.com
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/MultiProcess/HeatmapAnalyzer.h"

#include "gtest/gtest.h"

#include <iostream>

TEST(TestMPHeatmapAnalyzer, Analyze)
{
   RooFit::MultiProcess::HeatmapAnalyzer hm_analyzer(".");
   std::unique_ptr<TH2I> hm = hm_analyzer.analyze(2); // analyze second gradient

   // indexing starts from 1 in heatmap because of under/overflow bins
   GTEST_ASSERT_EQ(hm->GetBinContent(1, 1), 5);
   GTEST_ASSERT_EQ(hm->GetBinContent(2, 1), 15);
   GTEST_ASSERT_EQ(hm->GetBinContent(1, 2), 10);
   GTEST_ASSERT_EQ(hm->GetBinContent(2, 2), 20);
}
