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

#include "RooStats/RooStatsUtils.h"
#include "RooStats/SamplingDistribution.h"

#include "TNamed.h"

#include <memory>

namespace RooStats {

   class HypoTestResult : public TNamed {

   public:

      /// default constructor
      explicit HypoTestResult(const char* name = 0);

      /// copy constructor
      HypoTestResult(const HypoTestResult& other);

      /// constructor from name, null and alternate p values
      HypoTestResult(const char* name, double nullp, double altp);

      /// destructor
      ~HypoTestResult() override;

      /// assignment operator
      HypoTestResult & operator=(const HypoTestResult& other);

      /// add values from another HypoTestResult
      virtual void Append(const HypoTestResult *other);

      /// Return p-value for null hypothesis
      virtual double NullPValue() const { return fNullPValue; }

      /// Return p-value for alternate hypothesis
      virtual double AlternatePValue() const { return fAlternatePValue; }

      /// Convert  NullPValue into a "confidence level"
      virtual double CLb() const { return !fBackgroundIsAlt ? NullPValue() : AlternatePValue(); }

      /// Convert  AlternatePValue into a "confidence level"
      virtual double CLsplusb() const { return !fBackgroundIsAlt ? AlternatePValue() : NullPValue(); }

      /// \f$CL_{s}\f$ is simply \f$CL_{s+b}/CL_{b}\f$ (not a method, but a quantity)
      virtual double CLs() const {
         double thisCLb = CLb();
         if (thisCLb == 0) {
            std::cout << "Error: Cannot compute CLs because CLb = 0. Returning CLs = -1\n";
            return -1;
         }
         double thisCLsb = CLsplusb();
         return thisCLsb / thisCLb;
      }

      /// familiar name for the Null p-value in terms of 1-sided Gaussian significance
      virtual double Significance() const {return RooStats::PValueToSignificance( NullPValue() ); }

      SamplingDistribution* GetNullDistribution(void) const { return fNullDistr; }
      SamplingDistribution* GetAltDistribution(void) const { return fAltDistr; }
      RooDataSet* GetNullDetailedOutput(void) const { return fNullDetailedOutput; }
      RooDataSet* GetAltDetailedOutput(void) const { return fAltDetailedOutput; }
      RooDataSet* GetFitInfo() const { return fFitInfo.get(); }
      double GetTestStatisticData(void) const { return fTestStatisticData; }
      const RooArgList* GetAllTestStatisticsData(void) const { return fAllTestStatisticsData; }
      bool HasTestStatisticData(void) const;

      void SetAltDistribution(SamplingDistribution *alt);
      void SetNullDistribution(SamplingDistribution *null);
      void SetAltDetailedOutput(RooDataSet* d) { fAltDetailedOutput = d; }
      void SetNullDetailedOutput(RooDataSet* d) { fNullDetailedOutput = d; }
      void SetFitInfo(RooDataSet* d) { fFitInfo.reset(d); }
      void SetTestStatisticData(const double tsd);
      void SetAllTestStatisticsData(const RooArgList* tsd);

      void SetPValueIsRightTail(bool pr);
      bool GetPValueIsRightTail(void) const { return fPValueIsRightTail; }

      void SetBackgroundAsAlt(bool l = true) { fBackgroundIsAlt = l; }
      bool GetBackGroundIsAlt(void) const { return fBackgroundIsAlt; }

      /// The error on the "confidence level" of the null hypothesis
      double CLbError() const;

      /// The error on the "confidence level" of the alternative hypothesis
      double CLsplusbError() const;

      /// The error on the ratio \f$CL_{s+b}/CL_{b}\f$
      double CLsError() const;

      /// The error on the Null p-value
      double NullPValueError() const;

      /// The error on the significance, computed from NullPValueError via error propagation
      double SignificanceError() const;


      void Print(const Option_t* = "") const override;

   private:
      void UpdatePValue(const SamplingDistribution* distr, double &pvalue, double &perror,  bool pIsRightTail);


   protected:

      mutable double fNullPValue;             ///< p-value for the null hypothesis (small number means disfavoured)
      mutable double fAlternatePValue;        ///< p-value for the alternate hypothesis (small number means disfavoured)
      mutable double fNullPValueError;        ///< error of p-value for the null hypothesis (small number means disfavoured)
      mutable double fAlternatePValueError;   ///< error of p-value for the alternate hypothesis (small number means disfavoured)
      double fTestStatisticData;              ///< result of the test statistic evaluated on data
      const RooArgList* fAllTestStatisticsData; ///< for the case of multiple test statistics, holds all the results
      SamplingDistribution *fNullDistr;
      SamplingDistribution *fAltDistr;
      RooDataSet* fNullDetailedOutput;
      RooDataSet* fAltDetailedOutput;
      std::unique_ptr<RooDataSet> fFitInfo;
      bool fPValueIsRightTail;
      bool fBackgroundIsAlt;

      ClassDefOverride(HypoTestResult,4)  // Base class to represent results of a hypothesis test

   };
}


#endif
