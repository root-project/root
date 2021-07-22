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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooAbsL
#define ROOT_ROOFIT_TESTSTATISTICS_RooAbsL

#include "RooArgSet.h"
#include "RooAbsArg.h" // enum ConstOpCode

#include <cstddef> // std::size_t
#include <string>
#include <memory>

// forward declarations
class RooAbsPdf;
class RooAbsData;

namespace RooFit {
namespace TestStatistics {

class RooAbsL {
public:
   enum class Extended { Auto, Yes, No };
   static bool isExtendedHelper(RooAbsPdf *pdf, Extended extended);

   /// wrapper class used to distinguish ctors
   struct ClonePdfData {
      RooAbsPdf *pdf;
      RooAbsData *data;
   };

   //   RooAbsL() = default;
private:
   RooAbsL(std::shared_ptr<RooAbsPdf> pdf, std::shared_ptr<RooAbsData> data, std::size_t N_events,
           std::size_t N_components, Extended extended);

public:
   RooAbsL(RooAbsPdf *pdf, RooAbsData *data, std::size_t N_events, std::size_t N_components,
           Extended extended = Extended::Auto);
   RooAbsL(ClonePdfData in, std::size_t N_events, std::size_t N_components, Extended extended = Extended::Auto);
   RooAbsL(const RooAbsL &other);
   virtual ~RooAbsL() = default;

   void initClones(RooAbsPdf &inpdf, RooAbsData &indata);

   /// A part of some range delimited by two fractional points between 0 and 1 (inclusive).
   struct Section {
      Section(double begin, double end) : begin_fraction(begin), end_fraction(end)
      {
         if ((begin > end) || (begin < 0) || (end > 1)) {
            throw std::domain_error("Invalid input values for section; begin must be >= 0, end <= 1 and begin < end.");
         }
      }

      Section(const Section &section) = default;

      std::size_t begin(std::size_t N_total) const { return static_cast<std::size_t>(N_total * begin_fraction); }

      std::size_t end(std::size_t N_total) const
      {
         if (end_fraction == 1) {
            return N_total;
         } else {
            return static_cast<std::size_t>(N_total * end_fraction);
         }
      }

      double begin_fraction;
      double end_fraction;
   };

   virtual double evaluatePartition(Section events, std::size_t components_begin, std::size_t components_end) = 0;
   inline double getCarry() const { return eval_carry_; }

   // necessary from MinuitFcnGrad to reach likelihood properties:
   virtual RooArgSet *getParameters();
   virtual void constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt);

   virtual std::string GetName() const;
   virtual std::string GetTitle() const;

   // necessary in RooMinimizer (via LikelihoodWrapper)
   inline virtual double defaultErrorLevel() const { return 0.5; }

   // necessary in LikelihoodJob
   virtual std::size_t numDataEntries() const;
   inline std::size_t getNEvents() const { return N_events_; }
   inline std::size_t getNComponents() const { return N_components_; }
   inline bool isExtended() const { return extended_; }
   inline void setSimCount(std::size_t value) { sim_count_ = value; }

protected:
   virtual void optimizePdf();
   // Note: pdf_ and data_ can be constructed in two ways, one of which implies ownership and the other does not.
   // Inspired by this: https://stackoverflow.com/a/61227865/1199693.
   // The owning variant is used for classes that need a pdf/data clone (RooBinnedL and RooUnbinnedL), whereas the
   // non-owning version is used for when a reference to the external pdf/dataset is good enough (RooSumL).
   // This means that pdf_ and data_ are not meant to actually be shared! If there were a unique_ptr with optional
   // ownership, we would have used that instead.
   std::shared_ptr<RooAbsPdf> pdf_;
   std::shared_ptr<RooAbsData> data_;
   std::unique_ptr<RooArgSet> normSet_; // Pointer to set with observables used for normalization

   std::size_t N_events_ = 1;
   std::size_t N_components_ = 1;

   bool extended_ = false;

   std::size_t sim_count_ = 1; // Total number of component p.d.f.s in RooSimultaneous (if any)

   mutable double eval_carry_ = 0; //! carry of Kahan sum in evaluatePartition
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooAbsL
