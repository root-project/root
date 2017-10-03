// @(#)root/tmva $Id$
// Author: Omar Zapata, Thomas James Stevenson and Pourya Vakilipourtakalou. 2016

#ifndef ROOT_TMVA_CrossValidation
#define ROOT_TMVA_CrossValidation

#include "TString.h"

#include "TMultiGraph.h"

#include "TMVA/IMethod.h"

#include "TMVA/Configurable.h"

#include "TMVA/Types.h"

#include "TMVA/DataSet.h"

#include "TMVA/Event.h"

#include <TMVA/Results.h>

#include <TMVA/Factory.h>

#include <TMVA/DataLoader.h>

#include <TMVA/OptionMap.h>

#include <TMVA/Envelope.h>

/*! \class TMVA::CrossValidationResult
 * Class to save the results of cross validation,
 * the metric for the classification ins ROC and you can ROC curves
 * ROC integrals, ROC average and ROC standard deviation.
\ingroup TMVA
*/

/*! \class TMVA::CrossValidation
 * Class to perform cross validation, splitting the dataloader into folds.
\ingroup TMVA
*/

namespace TMVA {

   class CrossValidationResult {
      friend class CrossValidation;

   private:
      std::map<UInt_t,Float_t> fROCs;
      std::shared_ptr<TMultiGraph> fROCCurves;

      std::vector<Double_t> fSigs;
      std::vector<Double_t> fSeps;
      std::vector<Double_t> fEff01s;
      std::vector<Double_t> fEff10s;
      std::vector<Double_t> fEff30s;
      std::vector<Double_t> fEffAreas;
      std::vector<Double_t> fTrainEff01s;
      std::vector<Double_t> fTrainEff10s;
      std::vector<Double_t> fTrainEff30s;

   public:
      CrossValidationResult();
      CrossValidationResult(const CrossValidationResult &);
      ~CrossValidationResult(){fROCCurves=nullptr;}

      std::map<UInt_t,Float_t> GetROCValues(){return fROCs;}
      Float_t GetROCAverage() const;
      Float_t GetROCStandardDeviation() const;
      TMultiGraph *GetROCCurves(Bool_t fLegend=kTRUE);
      void Print() const ;

      TCanvas* Draw(const TString name="CrossValidation") const;

      std::vector<Double_t> GetSigValues() {return fSigs;}
      std::vector<Double_t> GetSepValues() {return fSeps;}
      std::vector<Double_t> GetEff01Values() {return fEff01s;}
      std::vector<Double_t> GetEff10Values() {return fEff10s;}
      std::vector<Double_t> GetEff30Values() {return fEff30s;}
      std::vector<Double_t> GetEffAreaValues() {return fEffAreas;}
      std::vector<Double_t> GetTrainEff01Values() {return fTrainEff01s;}
      std::vector<Double_t> GetTrainEff10Values() {return fTrainEff10s;}
      std::vector<Double_t> GetTrainEff30Values() {return fTrainEff30s;}
   };


   class CrossValidation : public Envelope {
      UInt_t fNumFolds;                            //!
      std::vector<CrossValidationResult> fResults; //!
      Bool_t fFoldStatus;                          //!
   public:
      explicit CrossValidation(DataLoader *loader);
      ~CrossValidation();

      void SetNumFolds(UInt_t i);
      UInt_t GetNumFolds() {return fNumFolds;}

      virtual void Evaluate();
//    void EvaluateFold(UInt_t fold);//used in ParallelExecution

      const std::vector<CrossValidationResult> &GetResults() const;

   private:
      std::unique_ptr<Factory> fClassifier;
      ClassDef(CrossValidation, 0);
   };

} // namespace TMVA

#endif // ROOT_TMVA_CrossValidation
