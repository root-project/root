/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2020, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#include <TestStatistics/RooAbsL.h>
#include "RooAbsPdf.h"
#include "RooAbsData.h"

namespace RooFit {
namespace TestStatistics {

RooAbsL::RooAbsL(RooAbsPdf *pdf, RooAbsData *data, bool do_offset, double offset, double offset_carry,
                 std::size_t N_events, std::size_t N_components)
   : pdf(pdf), data(data), _do_offset(do_offset), _offset(offset), _offset_carry(offset_carry), N_events(N_events),
     N_components(N_components)
{
}

RooArgSet *RooAbsL::getParameters()
{
   return pdf->getParameters(*data);
}

void RooAbsL::constOptimizeTestStatistic(RooAbsArg::ConstOpCode /*opcode*/, bool /*doAlsoTrackingOpt*/)
{
   // yet to be implemented
}

std::string RooAbsL::GetName() const
{
   std::string output("likelihood of pdf ");
   output.append(pdf->GetName());
   return output;
}

std::string RooAbsL::GetTitle() const
{
   std::string output("likelihood of pdf ");
   output.append(pdf->GetTitle());
   return output;
}

double RooAbsL::defaultErrorLevel() const
{
   return 0.5;
}
std::size_t RooAbsL::numDataEntries() const
{
   return static_cast<std::size_t>(data->numEntries());
}

bool RooAbsL::is_offsetting() const
{
   return _do_offset;
}
void RooAbsL::enable_offsetting(bool flag)
{
   _do_offset = flag;
   // Clear offset if feature is disabled so that it is recalculated next time it is enabled
   if (!_do_offset) {
      _offset = 0;
      _offset_carry = 0;
   }
}

void RooAbsL::optimize_pdf()
{
   // TODO: implement
}

std::size_t RooAbsL::get_N_events() const
{
   return N_events;
}

std::size_t RooAbsL::get_N_components() const
{
   return N_components;
}

} // namespace TestStatistics
} // namespace RooFit
