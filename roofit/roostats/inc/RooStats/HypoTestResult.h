// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke, Sven Kreiss
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/




#ifndef ROOSTATS_HypoTestResult
#define ROOSTATS_HypoTestResult

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOSTATS_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif

#ifndef ROOSTATS_SamplingDistribution
#include "RooStats/SamplingDistribution.h"
#endif

namespace RooStats {


/**

   \ingroup Roostats

   HypoTestResult is a base class for results from hypothesis tests.
   Any tool inheriting from HypoTestCalculator can return a HypoTestResult.
   As such, it stores a p-value for the null-hypothesis (eg. background-only)
   and an alternate hypothesis (eg. signal+background).
   The p-values can also be transformed into confidence levels
   (\f$CL_{b}\f$, \f$CL_{s+b}\f$) in a trivial way.
   The ratio of the \f$CL_{s+b}\f$ to \f$CL_{b}\f$ is often called
   \f$CL_{s}\f$, and is considered useful, though it is not a probability.
   Finally, the p-value of the null can be transformed into a number of
   equivalent Gaussian sigma using the Significance method.

   The p-value of the null for a given test statistic is rigorously defined and
   this is the starting point for the following conventions.

### Conventions used in this class

The p-value for the null and alternate are on the **same side** of the
observed value of the test statistic. This is the more standard
convention and avoids confusion when doing inverted tests.

For exclusion, we also want the formula \f$CL_{s} = CL_{s+b} / CL_{b}\f$
to hold which therefore defines our conventions for \f$CL_{s+b}\f$ and
\f$CL_{b}\f$. \f$CL_{s}\f$ was specifically invented for exclusion
and therefore all quantities need be related through the assignments
as they are for exclusion: \f$CL_{s+b} = p_{s+b}\f$; \f$CL_{b} = p_{b}\f$. This
is derived by considering the scenarios of a powerful and not powerful
inverted test, where for the not so powerful test, \f$CL_{s}\f$ must be
close to one.

For results of Hypothesis tests,
\f$CL_{s}\f$ has no similar direct interpretation as for exclusion and can
be larger than one.

*/

   class HypoTestResult : public TNamed {

   public:

      /// default constructor
      explicit HypoTestResult(const char* name = 0);

      /// copy constructor
      HypoTestResult(const HypoTestResult& other);

      /// constructor from name, null and alternate p values
      HypoTestResult(const char* name, Double_t nullp, Double_t altp);

      /// destructor
      virtual ~HypoTestResult();

      /// assignment operator
      HypoTestResult & operator=(const HypoTestResult& other);

      /// add values from another HypoTestResult
      virtual void Append(const HypoTestResult *other);

      /// Return p-value for null hypothesis
      virtual Double_t NullPValue() const { return fNullPValue; }

      /// Return p-value for alternate hypothesis
      virtual Double_t AlternatePValue() const { return fAlternatePValue; }

      /// Convert  NullPValue into a "confidence level"
      virtual Double_t CLb() const { return !fBackgroundIsAlt ? NullPValue() : AlternatePValue(); }

      /// Convert  AlternatePValue into a "confidence level"
      virtual Double_t CLsplusb() const { return !fBackgroundIsAlt ? AlternatePValue() : NullPValue(); }

      /// \f$CL_{s}\f$ is simply \f$CL_{s+b}/CL_{b}\f$ (not a method, but a quantity)
      virtual Double_t CLs() const {
         double thisCLb = CLb();
         if (thisCLb == 0) {
            std::cout << "Error: Cannot compute CLs because CLb = 0. Returning CLs = -1\n";
            return -1;
         }
         double thisCLsb = CLsplusb();
         return thisCLsb / thisCLb;
      }

      /// familiar name for the Null p-value in terms of 1-sided Gaussian significance
      virtual Double_t Significance() const {return RooStats::PValueToSignificance( NullPValue() ); }

      SamplingDistribution* GetNullDistribution(void) const { return fNullDistr; }
      SamplingDistribution* GetAltDistribution(void) const { return fAltDistr; }
      RooDataSet* GetNullDetailedOutput(void) const { return fNullDetailedOutput; }
      RooDataSet* GetAltDetailedOutput(void) const { return fAltDetailedOutput; }
      RooDataSet* GetFitInfo(void) const { return fFitInfo; }
      Double_t GetTestStatisticData(void) const { return fTestStatisticData; }
      const RooArgList* GetAllTestStatisticsData(void) const { return fAllTestStatisticsData; }
      Bool_t HasTestStatisticData(void) const;

      void SetAltDistribution(SamplingDistribution *alt);
      void SetNullDistribution(SamplingDistribution *null);
      void SetAltDetailedOutput(RooDataSet* d) { fAltDetailedOutput = d; }
      void SetNullDetailedOutput(RooDataSet* d) { fNullDetailedOutput = d; }
      void SetFitInfo(RooDataSet* d) { fFitInfo = d; }
      void SetTestStatisticData(const Double_t tsd);
      void SetAllTestStatisticsData(const RooArgList* tsd);

      void SetPValueIsRightTail(Bool_t pr);
      Bool_t GetPValueIsRightTail(void) const { return fPValueIsRightTail; }

      void SetBackgroundAsAlt(Bool_t l = kTRUE) { fBackgroundIsAlt = l; }
      Bool_t GetBackGroundIsAlt(void) const { return fBackgroundIsAlt; }

      /// The error on the "confidence level" of the null hypothesis
      Double_t CLbError() const;

      /// The error on the "confidence level" of the alternative hypothesis
      Double_t CLsplusbError() const;

      /// The error on the ratio \f$CL_{s+b}/CL_{b}\f$
      Double_t CLsError() const;

      /// The error on the Null p-value
      Double_t NullPValueError() const;

      /// The error on the significance, computed from NullPValueError via error propagation
      Double_t SignificanceError() const;


      void Print(const Option_t* = "") const;

   private:
      void UpdatePValue(const SamplingDistribution* distr, Double_t &pvalue, Double_t &perror,  Bool_t pIsRightTail);


   protected:

      mutable Double_t fNullPValue; // p-value for the null hypothesis (small number means disfavoured)
      mutable Double_t fAlternatePValue; // p-value for the alternate hypothesis (small number means disfavoured)
      mutable Double_t fNullPValueError; // error of p-value for the null hypothesis (small number means disfavoured)
      mutable Double_t fAlternatePValueError; // error of p-value for the alternate hypothesis (small number means disfavoured)
      Double_t fTestStatisticData; // result of the test statistic evaluated on data
      const RooArgList* fAllTestStatisticsData; // for the case of multiple test statistics, holds all the results
      SamplingDistribution *fNullDistr;
      SamplingDistribution *fAltDistr;
      RooDataSet* fNullDetailedOutput;
      RooDataSet* fAltDetailedOutput;
      RooDataSet* fFitInfo;
      Bool_t fPValueIsRightTail;
      Bool_t fBackgroundIsAlt;

      ClassDef(HypoTestResult,3)  // Base class to represent results of a hypothesis test

   };
}


#endif
