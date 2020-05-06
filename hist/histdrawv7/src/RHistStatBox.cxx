/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistStatBox.hxx"

#include "ROOT/RPadBase.hxx"
#include "ROOT/RFrame.hxx"

#include <string>

using namespace ROOT::Experimental;
using namespace std::string_literals;

std::unique_ptr<RDrawableReply> RHistStatRequest::Process()
{
   auto stat = dynamic_cast<RHistStatBoxBase *>(GetContext().GetDrawable());

   auto frame = GetContext().GetPad()->GetFrame();
   RDrawable::RUserRanges ranges;
   if (frame) frame->GetClientRanges(0, ranges);

   auto reply = std::make_unique<RHistStatReply>();

   if (stat) {
      stat->fShowMask = GetMask();

      reply->SetMask(GetMask());

      if (GetMask() & RHistStatBoxBase::kShowTitle)
         reply->GetLines().emplace_back(stat->GetTitle());

      stat->FillStatistic(GetMask(), ranges, reply->GetLines());
   }

   return reply;
}


const std::vector<std::string> &RHistStatBoxBase::GetEntriesNames() const
{
   static const std::vector<std::string> sEntries = {"Title","Entries", "Mean","Deviation", "Ranges"};
   return sEntries;
}

std::unique_ptr<RDisplayItem> RHistStatBoxBase::Display(const RDisplayContext &ctxt)
{
   RDrawable::RUserRanges ranges;

   auto frame = ctxt.GetPad()->GetFrame();
   if (frame) frame->GetClientRanges(0, ranges);

   std::vector<std::string> lines;

   if (fShowMask & RHistStatBoxBase::kShowTitle)
      lines.emplace_back(GetTitle());

   FillStatistic(fShowMask, ranges, lines);

   // do not send stat box itself while it includes histogram which is not required on client side
   return std::make_unique<RDisplayHistStat>(*this, fShowMask, GetEntriesNames(), lines);
}


void RHist1StatBox::FillStatistic(unsigned mask, const RDrawable::RUserRanges &ranges, std::vector<std::string> &lines) const
{
   // TODO: need to implement statistic fill for RHist1

   if (mask & kShowEntries)
      lines.emplace_back("Entries = 1");

   if (mask & kShowMean)
      lines.emplace_back("Mean = 2");

   if (mask & kShowDev)
      lines.emplace_back("Std dev = 3");

   if (mask & kShowRange) {
      lines.emplace_back("X min = "s + std::to_string(ranges.GetMin(0)));
      lines.emplace_back("X max = "s + std::to_string(ranges.GetMax(0)));
   }
}


void RHist2StatBox::FillStatistic(unsigned mask, const RDrawable::RUserRanges &ranges, std::vector<std::string> &lines) const
{
   // TODO: need to implement statistic fill for RHist2

   if (mask & kShowEntries)
      lines.emplace_back("Entries = 1");

   if (mask & kShowMean) {
      lines.emplace_back("Mean x = 2");
      lines.emplace_back("Mean y = 3");
   }

   if (mask & kShowDev) {
      lines.emplace_back("Std dev x = 5");
      lines.emplace_back("Std dev y = 6");
   }

   if (mask & kShowRange) {
      lines.emplace_back("X min = "s + std::to_string(ranges.GetMin(0)));
      lines.emplace_back("X max = "s + std::to_string(ranges.GetMax(0)));
      lines.emplace_back("Y min = "s + std::to_string(ranges.GetMin(1)));
      lines.emplace_back("Y max = "s + std::to_string(ranges.GetMax(1)));
   }
}

void RHist3StatBox::FillStatistic(unsigned mask, const RDrawable::RUserRanges &ranges, std::vector<std::string> &lines) const
{
   // TODO: need to implement statistic fill for RHist3

   if (mask & kShowEntries)
      lines.emplace_back("Entries = 1");

   if (mask & kShowMean) {
      lines.emplace_back("Mean x = 2");
      lines.emplace_back("Mean y = 3");
      lines.emplace_back("Mean z = 4");
   }

   if (mask & kShowDev) {
      lines.emplace_back("Std dev x = 5");
      lines.emplace_back("Std dev y = 6");
      lines.emplace_back("Std dev z = 7");
   }

   if (mask & kShowRange) {
      lines.emplace_back("X min = "s + std::to_string(ranges.GetMin(0)));
      lines.emplace_back("X max = "s + std::to_string(ranges.GetMax(0)));
      lines.emplace_back("Y min = "s + std::to_string(ranges.GetMin(1)));
      lines.emplace_back("Y max = "s + std::to_string(ranges.GetMax(1)));
      lines.emplace_back("Z min = "s + std::to_string(ranges.GetMin(2)));
      lines.emplace_back("Z max = "s + std::to_string(ranges.GetMax(2)));
   }
}
