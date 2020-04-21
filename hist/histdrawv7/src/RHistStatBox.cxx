/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistStatBox.hxx"

#include <string>

using namespace ROOT::Experimental;
using namespace std::string_literals;

std::unique_ptr<RDrawableReply> RHistStatRequest::Process()
{
   auto stat = dynamic_cast<RHistStatBoxBase *>(GetContext().GetDrawable());

   auto reply = std::make_unique<RHistStatReply>();

   if (stat) {
      stat->fShowMask = GetMask();

      if (GetMask() & RHistStatBoxBase::kShowTitle)
         reply->AddLine(stat->GetTitle());

      stat->FillStatistic(*this, *reply);
   }

   return reply;
}


const std::vector<std::string> &RHistStatBoxBase::GetEntriesNames() const
{
   static const std::vector<std::string> sEntries = {"Title","Entries", "Mean","Deviation", "Ranges"};
   return sEntries;
}


void RHist1StatBox::FillStatistic(const RHistStatRequest &req, RHistStatReply &reply) const
{
   // TODO: need to implement statistic fill for RHist1

   if (req.GetMask() & kShowEntries)
      reply.AddLine("Entries = 1");

   if (req.GetMask() & kShowMean)
      reply.AddLine("Mean = 2");

   if (req.GetMask() & kShowDev)
      reply.AddLine("Std dev = 3");

   if (req.GetMask() & kShowRange) {
      reply.AddLine("X min = "s + std::to_string(req.GetMin(0)));
      reply.AddLine("X max = "s + std::to_string(req.GetMax(0)));
   }
}


void RHist2StatBox::FillStatistic(const RHistStatRequest &req, RHistStatReply &reply) const
{
   // TODO: need to implement statistic fill for RHist2

   if (req.GetMask() & kShowEntries)
      reply.AddLine("Entries = 1");

   if (req.GetMask() & kShowMean) {
      reply.AddLine("Mean x = 2");
      reply.AddLine("Mean y = 3");
   }

   if (req.GetMask() & kShowDev) {
      reply.AddLine("Std dev x = 5");
      reply.AddLine("Std dev y = 6");
   }

   if (req.GetMask() & kShowRange) {
      reply.AddLine("X min = "s + std::to_string(req.GetMin(0)));
      reply.AddLine("X max = "s + std::to_string(req.GetMax(0)));
      reply.AddLine("Y min = "s + std::to_string(req.GetMin(1)));
      reply.AddLine("Y max = "s + std::to_string(req.GetMax(1)));
   }
}

void RHist3StatBox::FillStatistic(const RHistStatRequest &req, RHistStatReply &reply) const
{
   // TODO: need to implement statistic fill for RHist3

   if (req.GetMask() & kShowEntries)
      reply.AddLine("Entries = 1");

   if (req.GetMask() & kShowMean) {
      reply.AddLine("Mean x = 2");
      reply.AddLine("Mean y = 3");
      reply.AddLine("Mean z = 4");
   }

   if (req.GetMask() & kShowDev) {
      reply.AddLine("Std dev x = 5");
      reply.AddLine("Std dev y = 6");
      reply.AddLine("Std dev z = 7");
   }

   if (req.GetMask() & kShowRange) {
      reply.AddLine("X min = "s + std::to_string(req.GetMin(0)));
      reply.AddLine("X max = "s + std::to_string(req.GetMax(0)));
      reply.AddLine("Y min = "s + std::to_string(req.GetMin(1)));
      reply.AddLine("Y max = "s + std::to_string(req.GetMax(1)));
      reply.AddLine("Z min = "s + std::to_string(req.GetMin(2)));
      reply.AddLine("Z max = "s + std::to_string(req.GetMax(2)));
   }

}

