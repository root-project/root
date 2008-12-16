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

      // Initialize the calculator from a given data set, a null pdf and an  alternate pdf. 
      // The null parameters and alternate parameters can be optionally passed otherwise by default 
      // the parameters of the pdf's will be used. 
      // In addition, one can give optionally a nuisance pdf with nuisance parameters to be marginalized 
      virtual void Initialize(RooAbsData & data, RooAbsPdf & nullPdf, RooAbsPdf & alternatePdf, 
                              RooArgSet * nullParameters = 0, RooArgSet * alternateParameters = 0, 
                              RooArgSet * nuisanceParameters = 0, RooAbsPdf * nuisancePdf = 0  ) { 
         SetData(data); 
         SetNullPdf(nullPdf); 
         SetAlternatePdf(alternatePdf); 
         if (nullParameters) SetNullParameters(*nullParameters);
         if (alternateParameters) SetAlternateParameters(*alternateParameters); 
         if (nuisanceParameters) SetNuisanceParameters(*nuisanceParameters);  
         if (nuisancePdf) SetNuisancePdf(*nuisancePdf); 
      } 

      // Initialize the calculator from a given data set and a common  pdf for null and alternate hypothesis. 
      // In this case the null parameters and alternate parameters must be given.
      // In addition, one can give optionally a nuisance pdf with nuisance parameters to be marginalized 
      virtual void Initialize(RooAbsData & data, RooAbsPdf & commonPdf,  
                              RooArgSet & nullParameters, RooArgSet & alternateParameters, 
                              RooArgSet * nuisanceParameters = 0, RooAbsPdf * nuisancePdf = 0  ) { 
         Initialize(data, commonPdf, commonPdf, &nullParameters, &alternateParameters, nuisanceParameters, nuisancePdf);
      } 

      // Initialize the calculator from a workspace and names for the  data set, 
      // the null pdf and the alternate pdf. 
      // The null parameters and alternate parameters can be optionally passed otherwise by default 
      // the parameters of the pdf's will be used. 
      // In addition, one can give optionally a nuisance pdf with nuisance parameters to be marginalized 
      virtual void Initialize(RooWorkspace & ws, const char * data, const char * nullPdf, const char *  alternatePdf, 
                              RooArgSet * nullParameters = 0, RooArgSet * alternateParameters = 0, 
                              RooArgSet * nuisanceParameters = 0, const char * nuisancePdf = 0  ) { 
         SetWorkspace(ws);
         SetData(data); 
         SetNullPdf(nullPdf); 
         SetAlternatePdf(alternatePdf); 
         if (nullParameters) SetNullParameters(*nullParameters);
         if (alternateParameters) SetAlternateParameters(*alternateParameters); 
         if (nuisanceParameters) SetNuisanceParameters(*nuisanceParameters);  
         if (nuisancePdf) SetNuisancePdf(nuisancePdf); 
      } 

      // Initialize the calculator from a workspace and names for the  data set and a common pdf 
      // for both the null and the alternate hypothesis.  
      // In this case the null parameters and alternate parameters must be given.
      // In addition, one can give optionally a nuisance pdf with nuisance parameters to be marginalized 
      virtual void Initialize(RooWorkspace & ws, const char * data, const char * commonPdf, 
                              RooArgSet & nullParameters, RooArgSet & alternateParameters, 
                              RooArgSet * nuisanceParameters = 0, const char * nuisancePdf = 0  ) { 
         Initialize(ws, data, commonPdf, commonPdf, &nullParameters, &alternateParameters, nuisanceParameters, nuisancePdf);
      }

      // set a workspace that owns all the necessary components for the analysis
      virtual void SetWorkspace(RooWorkspace& ws) = 0;
      // set the PDF for the null hypothesis
      virtual void SetNullPdf(const char* name) = 0;
      // set the PDF for the alternate hypothesis
      virtual void SetAlternatePdf(const char* name) = 0;
      // set a common PDF for both the null and alternate hypotheses
      virtual void SetCommonPdf(const char* name) { 
         SetNullPdf(name); 
         SetAlternatePdf(name); 
      }
      // Set a common PDF for both the null and alternate, add to the the workspace if not already there
      virtual void SetCommonPdf(RooAbsPdf& pdf) { 
         SetNullPdf(pdf); 
         SetAlternatePdf(pdf); 
      }
      // Set the PDF for the null, add to the the workspace if not already there
      virtual void SetNullPdf(RooAbsPdf& pdf) = 0;
      // Set the PDF for the alternate hypothesis, add to the the workspace if not already there
      virtual void SetAlternatePdf(RooAbsPdf& pdf) = 0;

      // specify the name of the dataset in the workspace to be used
      virtual void SetData(const char* name) = 0;
      // Set the DataSet, add to the the workspace if not already there
      virtual void SetData(RooAbsData& data) = 0;

      // set parameter values for the null if using a common PDF
      virtual void SetNullParameters(RooArgSet&) = 0;
      // set parameter values for the alternate if using a common PDF
      virtual void SetAlternateParameters(RooArgSet&) = 0;

      // set the pdf name for an auxillary measurement of the nuisance which will be marginalized by the calculator
      // needs to be implemented by the derived class only if this feature is supported
      virtual void SetNuisancePdf(const char * ) {}

      // set the pdf for an auxillary measurement which will be marginalized by the calculator
      // needs to be implemented by the derived class only if this feature is supported
      virtual void SetNuisancePdf(RooAbsPdf &) {}

      // set the parameters for the constraned pdf which  will be marginalized by the calculator
      // needs to be implemented by the derived class if this feature is supported
      virtual void SetNuisanceParameters(RooArgSet &) {}

   protected:
      ClassDef(HypoTestCalculator,1)  // Interface for tools doing hypothesis tests
   };
}


#endif
