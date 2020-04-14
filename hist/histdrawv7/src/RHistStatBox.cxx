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
   auto stat = dynamic_cast<RHistStatBoxBase *>(GetDrawable());

   auto reply = std::make_unique<RHistStatReply>();

   if (stat)
      stat->FillStatistic(*this, *reply.get());

   return reply;
}

void RHist1StatBox::FillStatistic(const RHistStatRequest &req, RHistStatReply &reply) const
{
   // need to implement statistic fill for RHist1

   reply.AddLine("Entries = 1");
   reply.AddLine("Mean = 2");
   reply.AddLine("Std dev = 3");
   reply.AddLine("X min = "s + std::to_string(req.GetMin(0)));
   reply.AddLine("X max = "s + std::to_string(req.GetMax(0)));
}


void RHist2StatBox::FillStatistic(const RHistStatRequest &req, RHistStatReply &reply) const
{
   // need to implement statistic fill for RHist2

   reply.AddLine("Entries = 1");
   reply.AddLine("X min = "s + std::to_string(req.GetMin(0)));
   reply.AddLine("X max = "s + std::to_string(req.GetMax(0)));
   reply.AddLine("Y min = "s + std::to_string(req.GetMin(1)));
   reply.AddLine("Y max = "s + std::to_string(req.GetMax(1)));
}

void RHist3StatBox::FillStatistic(const RHistStatRequest &, RHistStatReply &reply) const
{
   // need to implement statistic fill for RHist3

   reply.AddLine("Entries = 1");
   reply.AddLine("Mean x = 2");
   reply.AddLine("Mean y = 3");
   reply.AddLine("Mean z = 4");
   reply.AddLine("Std dev x = 5");
   reply.AddLine("Std dev y = 6");
   reply.AddLine("Std dev z = 7");
}

