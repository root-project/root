// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson and Pourya Vakilipourtakalou. 2016


#ifndef ROOT_TMVA_CrossValidation
#define ROOT_TMVA_CrossValidation


#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TMultiGraph
#include "TMultiGraph.h"
#endif

#ifndef ROOT_TCanvas
#include "TCanvas.h"
#endif


namespace TMVA {

   class CrossValidationResult
   {
   private:
       std::map<UInt_t,Float_t>        fROCs;       //!
       std::shared_ptr<TMultiGraph>    fROCCurves;  //!
   public:
       CrossValidationResult();
       CrossValidationResult(const CrossValidationResult &);
       ~CrossValidationResult();
       
       void SetROCValue(UInt_t fold,Float_t rocint);
       
       std::map<UInt_t,Float_t> GetROCValues(){return fROCs;}
       Float_t GetROCAverage() const;
       std::shared_ptr<TMultiGraph> &GetROCCurves();
       void Print() const ;
       
       TCanvas* Draw(const TString name="CrossValidation") const;
   };
} 


#endif



