// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ModelConfig
#define ROOSTATS_ModelConfig


#ifndef ROO_ABS_PDF
#include "RooAbsPdf.h"
#endif

#ifndef ROO_ABS_DATA
#include "RooAbsData.h"
#endif

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif

#ifndef ROO_WORKSPACE
#include "RooWorkspace.h"
#endif


#include <string>

//_________________________________________________
/*
BEGIN_HTML
<p>
ModelConfig is a simple class that holds configuration information specifying how a model
should be used in the context of various RooStats tools.  A single model can be used
in different ways, and this class should carry all that is needed to specify how it should be used.
</p>
END_HTML
*/
//

namespace RooStats {

class ModelConfig : public TNamed {

public:

   ModelConfig() : TNamed(), fWS(0), fOwnsWorkspace(false) {
   }
    
   ModelConfig(const char* name) : TNamed(name, name), fWS(0), fOwnsWorkspace(false) {
   }
    
   ModelConfig(const char* name, const char* title) : TNamed(name, title), fWS(0), fOwnsWorkspace(false) {
      fWS = 0;
      fOwnsWorkspace = false;
   }
    
   // destructor.
   virtual ~ModelConfig(); 

   // set a workspace that owns all the necessary components for the analysis
   virtual void SetWorkspace(RooWorkspace & ws);

   // Set the proto DataSet, add to the the workspace if not already there
   virtual void SetProtoData(RooAbsData & data) {      
      ImportDataInWS(data); 
      SetProtoData( data.GetName() );
   }
    
   // Set the Pdf, add to the the workspace if not already there
   virtual void SetPdf(RooAbsPdf& pdf) {
      ImportPdfInWS(pdf);
      SetPdf( pdf.GetName() );      
   }

   // Set the Prior Pdf, add to the the workspace if not already there
   virtual void SetPriorPdf(RooAbsPdf& pdf) {
      ImportPdfInWS(pdf);
      SetPriorPdf( pdf.GetName() );      
   }
    
   // specify the parameters of interest in the interval
   virtual void SetParameters(RooArgSet& set) {
      fPOIName=std::string(GetName()) + "_POI";
      DefineSetInWS(fPOIName.c_str(), set);
   }
    
   // specify the nuisance parameters (eg. the rest of the parameters)
   virtual void SetNuisanceParameters(RooArgSet& set) {
      fNuisParamsName=std::string(GetName()) + "_NuisParams";
      DefineSetInWS(fNuisParamsName.c_str(), set);
   }

   // set parameter values for the null if using a common PDF
   virtual void SetSnapshot(RooArgSet& set) {
      fSnapshotName=std::string(GetName()) + "_Snapshot";
      DefineSetInWS(fSnapshotName.c_str(), set);
   }    
    
   // specify the name of the PDF in the workspace to be used
   virtual void SetPdf(const char* name) {
      if(!fWS){
         coutE(ObjectHandling) << "workspace not set" << endl;
         return;
      }
      if(fWS->pdf(name))
         fPdfName = name;
      else
         coutE(ObjectHandling) << "pdf "<<name<< " does not exist in workspace"<<endl;
   }

   // specify the name of the PDF in the workspace to be used
   virtual void SetPriorPdf(const char* name) {
      if(!fWS){
         coutE(ObjectHandling) << "workspace not set" << endl;
         return;
      }
      if(fWS->pdf(name))
         fPriorPdfName = name;
      else
         coutE(ObjectHandling) << "pdf "<<name<< " does not exist in workspace"<<endl;
   }


   // specify the name of the dataset in the workspace to be used
   virtual void SetProtoData(const char* name){
      if(!fWS){
         coutE(ObjectHandling) << "workspace not set" << endl;
         return;
      }
      if(fWS->data(name))
         fProtoDataName = name;
      else
         coutE(ObjectHandling) << "dataset "<<name<< " does not exist in workspace"<<endl;
   }


   /* getter methods */


   /// get model PDF (return NULL if pdf has not been specified or does not exist)
   RooAbsPdf * GetPdf() const { return (fWS) ? fWS->pdf(fPdfName.c_str()) : 0;   }

   /// get RooArgSet containing the parameter of interest (return NULL if not existing) 
   const RooArgSet * GetParametersOfInterest() const { return (fWS) ? fWS->set(fPOIName.c_str()) : 0; } 

   /// get RooArgSet containing the nuisance parameters (return NULL if not existing) 
   const RooArgSet * GetNuisanceParameters() const { return (fWS) ? fWS->set(fNuisParamsName.c_str()) : 0; } 

   /// get RooArgSet containing the constraint parameters (return NULL if not existing) 
   const RooArgSet * GetConstraintParameters() const { return (fWS) ? fWS->set(fConstrainedParamName.c_str()) : 0; } 

   /// get parameters prior pdf  (return NULL if not existing) 
   RooAbsPdf * GetPriorPdf() const { return (fWS) ? fWS->pdf(fPriorPdfName.c_str()) : 0; } 

   /// get RooArgSet for observales  (return NULL if not existing) 
   const RooArgSet * GetObservables() const { return (fWS) ? fWS->set(fObservablesName.c_str()) : 0; } 

   /// get RooArgSet for conditional observales  (return NULL if not existing) 
   const RooArgSet * GetConditionalObservables() const { return (fWS) ? fWS->set(fConditionalObservablesName.c_str()) : 0; } 

   /// get Proto data set (return NULL if not existing) 
   RooAbsData * GetProtoData()  const {  return (fWS) ? fWS->data(fProtoDataName.c_str()) : 0; } 

   /// get RooArgSet for parameters for a particular hypothesis  (return NULL if not existing) 
   const RooArgSet * GetSnapshot() const { return (fWS) ? fWS->set(fSnapshotName.c_str()) : 0; } 
 
   const RooWorkspace * GetWS() const { return fWS; }
    
protected:

   
   // helper functions to define a set in the WS
   void DefineSetInWS(const char* name, RooArgSet& set);
    
   // internal function to import Pdf in WS
   void ImportPdfInWS(RooAbsPdf & pdf);
      
   // internal function to import data in WS
   void ImportDataInWS(RooAbsData & data); 
    
   RooWorkspace* fWS; // a workspace that owns all the components to be used by the calculator
   Bool_t fOwnsWorkspace;

   std::string fPdfName; // name of  PDF in workspace
   std::string fDataName; // name of data set in workspace
   std::string fPOIName; // name for RooArgSet specifying parameters of interest
    
   std::string fNuisParamsName; // name for RooArgSet specifying nuisance parameters
   std::string fConstrainedParamName; // name for RooArgSet specifying constrained parameters
   std::string fPriorPdfName; // name for RooAbsPdf specifying a prior on the parameters
    
   std::string fConditionalObservablesName; // name for RooArgSet specifying conditional observables
   std::string fProtoDataName; // name for RooArgSet specifying dataset that should be used as protodata
    
   std::string fSnapshotName; // name for RooArgSet that specifies a particular hypothesis
    
   std::string fObservablesName; // name for RooArgSet specifying observable parameters. 
    
   ClassDef(ModelConfig,1) // A class that holds configuration information for a model using a workspace as a store
      
};

}   // end namespace RooStats


#endif
