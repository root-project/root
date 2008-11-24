// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HypoTestCalculator
#define ROOSTATS_HypoTestCalculator

//#include "TNamed.h"

//_________________________________________________
/*
BEGIN_HTML
<p>
HypoTestCalculator is an interface class for a tools which produce RooStats HypoTestResults.  
The interface currently assumes that any hypothesis test calculator can be configured by specifying:
<ul>
 <li>a model for the null,</li>
 <li>a model for the alternate,</li>
 <li>a data set, </li>
 <li>a set of parameters of which specify the null (including values and const/non-const status), and </li>
 <li>a set of parameters of which specify the alternate (including values and const/non-const status).</li>
</ul>
The interface allows one to pass the model, data, and parameters via a workspace and then specify them with names.
The interface will be extended so that one does not need to use a workspace.
</p>
<p>
After configuring the calculator, one only needs to ask GetHypoTest, which will return a HypoTestResult pointer.
</p>
<p>
The concrete implementations of this interface should deal with the details of how the nuisance parameters are
dealt with (eg. integration vs. profiling) and which test-statistic is used (perhaps this should be added to the interface).
</p>
<p>
The motivation for this interface is that we hope to be able to specify the problem in a common way for several concrete calculators.
</p>
END_HTML
*/
//

class RooAbsPdf;
class RooArgSet; 
class RooWorkspace; 

namespace RooStats {

   class HypoTestResult;

   class HypoTestCalculator {

   public:

      // Concrete implementations should have a constructor like: 
      // HypoTestCalculator(RooWorkspace*, RooAbsData*,RooAbsPdf*, RooAbsPdf*, RooArgSet*, RooArgSet*) 
      // Concrete implementations should have a constructor like: 
      // HypoTestCalculator(RooAbsData*,RooAbsPdf*, RooAbsPdf*, RooArgSet*, RooArgSet*) 

      virtual ~HypoTestCalculator() {}

      // main interface to get a HypoTestResult, pure virtual
      virtual HypoTestResult* GetHypoTest() const = 0;

      // set a workspace that owns all the necessary components for the analysis
      virtual void SetWorkspace(RooWorkspace* ws) = 0;
      // set the PDF for the null hypothesis
      virtual void SetNullPdf(const char* name) = 0;
      // set the PDF for the alternate hypothesis
      virtual void SetAlternatePdf(const char* name) = 0;
      // set a common PDF for both the null and alternate hypotheses
      virtual void SetCommonPdf(const char* name) = 0;
      // Set a common PDF for both the null and alternate, add to the the workspace if not already there
      virtual void SetCommonPdf(RooAbsPdf* pdf) = 0;
      // Set the PDF for the null, add to the the workspace if not already there
      virtual void SetNullPdf(RooAbsPdf* pdf) = 0;
      // Set the PDF for the alternate hypothesis, add to the the workspace if not already there
      virtual void SetAlternatePdf(RooAbsPdf* pdf) = 0;

      // specify the name of the dataset in the workspace to be used
      virtual void SetData(const char* name) = 0;
      // Set the DataSet, add to the the workspace if not already there
      virtual void SetData(RooAbsData* data) = 0;

      // set parameter values for the null if using a common PDF
      virtual void SetNullParameters(RooArgSet*) = 0;
      // set parameter values for the alternate if using a common PDF
      virtual void SetAlternateParameters(RooArgSet*) = 0;

   protected:
      ClassDef(HypoTestCalculator,1)  // Interface for tools doing hypothesis tests
   };
}


#endif
