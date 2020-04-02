/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistStatBox.hxx"

using namespace ROOT::Experimental;


void RHist1StatBox::FillStatistic(RDisplayHistStat &stat, const RPadBase &) const
{
   // need to implement statistic fill for RHist1

   stat.AddLine("Entries = 1");
   stat.AddLine("Mean = 2");
   stat.AddLine("Std dev = 3");
}


void RHist2StatBox::FillStatistic(RDisplayHistStat &stat, const RPadBase &) const
{
   // need to implement statistic fill for RHist2

   stat.AddLine("Entries = 1");
   stat.AddLine("Mean x = 2");
   stat.AddLine("Mean y = 3");
   stat.AddLine("Std dev x = 4");
   stat.AddLine("Std dev y = 5");
}

void RHist3StatBox::FillStatistic(RDisplayHistStat &stat, const RPadBase &) const
{
   // need to implement statistic fill for RHist3

   stat.AddLine("Entries = 1");
   stat.AddLine("Mean x = 2");
   stat.AddLine("Mean y = 3");
   stat.AddLine("Mean z = 4");
   stat.AddLine("Std dev x = 5");
   stat.AddLine("Std dev y = 6");
   stat.AddLine("Std dev z = 7");
}

