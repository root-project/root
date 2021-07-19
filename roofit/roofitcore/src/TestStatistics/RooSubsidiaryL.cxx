// Author: Patrick Bos, Netherlands eScience Center / NIKHEF 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <TestStatistics/RooSubsidiaryL.h>
#include <TestStatistics/kahan_sum.h>
#include <RooAbsPdf.h> // for dynamic cast
#include <RooErrorHandler.h>

namespace RooFit {
namespace TestStatistics {

RooSubsidiaryL::RooSubsidiaryL(const std::string& parent_pdf_name, const RooArgSet &pdfs, const RooArgSet &parameter_set)
   : RooAbsL(nullptr, nullptr, 0, 0, RooAbsL::Extended::No), parent_pdf_name_(parent_pdf_name)
{
   std::unique_ptr<TIterator> inputIter{pdfs.createIterator()};
   RooAbsArg *comp;
   while ((comp = (RooAbsArg *)inputIter->Next())) {
      if (!dynamic_cast<RooAbsPdf *>(comp)) {
         oocoutE((TObject *)0, InputArguments) << "RooSubsidiaryL::ctor(" << GetName() << ") ERROR: component "
                                               << comp->GetName() << " is not of type RooAbsPdf" << std::endl;
         RooErrorHandler::softAbort();
      }
      subsidiary_pdfs_.add(*comp);
   }
   parameter_set_.add(parameter_set);
}

double
RooSubsidiaryL::evaluatePartition(RooAbsL::Section events, std::size_t /*components_begin*/, std::size_t /*components_end*/)
{
   if (events.begin_fraction != 0 || events.end_fraction != 1) {
      oocoutW((TObject *)0, InputArguments) << "RooSubsidiaryL::evaluatePartition can only calculate everything, so "
                                               "section should be {0,1}, but it's not!"
                                            << std::endl;
   }

   double sum = 0, carry = 0;
   RooAbsReal *comp;
   RooFIter setIter1 = subsidiary_pdfs_.fwdIterator();

   while ((comp = (RooAbsReal *)setIter1.next())) {
      double term = -((RooAbsPdf *)comp)->getLogVal(&parameter_set_);
      std::tie(sum, carry) = kahan_add(sum, term, carry);
   }
   eval_carry_ = carry;

   return sum;
}

void RooSubsidiaryL::constOptimizeTestStatistic(RooAbsArg::ConstOpCode /*opcode*/, bool /*doAlsoTrackingOpt*/) {
   // do nothing, there's no dataset here to cache
}

} // namespace TestStatistics
} // namespace RooFit
