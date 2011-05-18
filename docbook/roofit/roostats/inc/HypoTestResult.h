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

   class HypoTestResult : public TNamed {

   public:
      
      // default constructor
      explicit HypoTestResult(const char* name = 0);

      // constructor from name, null and alternate p values 
      HypoTestResult(const char* name, Double_t nullp, Double_t altp);

      // destructor 
      virtual ~HypoTestResult();

      // add values from another HypoTestResult
      virtual void Append(const HypoTestResult *other);

      // Return p-value for null hypothesis
      virtual Double_t NullPValue() const { return fNullPValue; }

      // Return p-value for alternate hypothesis
      virtual Double_t AlternatePValue() const { return fAlternatePValue; }

      // Convert  NullPValue into a "confidence level"
      virtual Double_t CLb() const { return 1.-NullPValue(); }

      // Convert  AlternatePValue into a "confidence level"
      virtual Double_t CLsplusb() const { return AlternatePValue(); }

      // CLs is simply CLs+b/CLb (not a method, but a quantity)
      virtual Double_t CLs() const {
         double thisCLb = CLb();
         if (thisCLb == 0) {
            std::cout << "Error: Cannot compute CLs because CLb = 0. Returning CLs = -1\n";
            return -1;
         }
         double thisCLsb = CLsplusb();
         return thisCLsb / thisCLb;
      }

      // familiar name for the Null p-value in terms of 1-sided Gaussian significance
      virtual Double_t Significance() const {return RooStats::PValueToSignificance( NullPValue() ); }

      SamplingDistribution* GetNullDistribution(void) const { return fNullDistr; }
      SamplingDistribution* GetAltDistribution(void) const { return fAltDistr; }
      Double_t GetTestStatisticData(void) const { return fTestStatisticData; }
      Bool_t HasTestStatisticData(void) const;

      void SetAltDistribution(SamplingDistribution *alt);
      void SetNullDistribution(SamplingDistribution *null);
      void SetTestStatisticData(const Double_t tsd);

      void SetPValueIsRightTail(Bool_t pr);
      Bool_t GetPValueIsRightTail(void) const { return fPValueIsRightTail; }

      /// The error on the "confidence level" of the null hypothesis
      Double_t CLbError() const;

      /// The error on the "confidence level" of the alternative hypothesis
      Double_t CLsplusbError() const;

      /// The error on the ratio CLs+b/CLb
      Double_t CLsError() const;

      Double_t NullPValueError() const;


      void Print(const Option_t* = "") const {
         // Print out some information about the results

         cout << endl << "Results " << GetName() << ": " << endl;
         if(HasTestStatisticData()  &&  fNullDistr) {
            cout << " - Null p-value = " << NullPValue() << " +/- " << NullPValueError() << endl;
            cout << " - Significance = " << Significance() << " sigma" << endl;
         }
         if(fAltDistr)
            cout << " - Number of S+B toys: " << fAltDistr->GetSize() << std::endl;
         if(fNullDistr)
            cout << " - Number of B toys: " << fNullDistr->GetSize() << std::endl;
         if(HasTestStatisticData())
            cout << " - Test statistic evaluated on data: " << fTestStatisticData << std::endl;
         if(HasTestStatisticData()  &&  fNullDistr)
            cout << " - CL_b: " << CLb() << " +/- " << CLbError() << std::endl;
         if(HasTestStatisticData()  &&  fAltDistr)
            cout << " - CL_s+b: " << CLsplusb() << " +/- " << CLsplusbError() << std::endl;
         if(HasTestStatisticData()  &&  fAltDistr  &&  fNullDistr)
            cout << " - CL_s: " << CLs() << " +/- " << CLsError()  << std::endl;

         return;
      }

   private:
      void UpdatePValue(const SamplingDistribution* distr, Double_t *pvalue, Bool_t pIsRightTail);


   protected:

      mutable Double_t fNullPValue; // p-value for the null hypothesis (small number means disfavored)
      mutable Double_t fAlternatePValue; // p-value for the alternate hypothesis (small number means disfavored)
      Double_t fTestStatisticData; // result of the test statistic evaluated on data
      SamplingDistribution *fNullDistr;
      SamplingDistribution *fAltDistr;
      Bool_t fPValueIsRightTail;

      ClassDef(HypoTestResult,1)  // Base class to represent results of a hypothesis test

   };
}


#endif
