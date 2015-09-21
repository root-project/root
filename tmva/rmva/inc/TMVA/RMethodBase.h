// @(#)root/tmva/rmva $Id$
// Author: Omar Zapata,Lorenzo Moneta, Sergei Gleyzer 2015

#ifndef ROOT_TMVA_RMethodBase
#define ROOT_TMVA_RMethodBase

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif

#ifndef ROOT_R_TRInterface
#include<TRInterface.h>
#endif

class TGraph;
class TTree;
class TDirectory;
class TSpline;
class TH1F;
class TH1D;
/**
 @namespace TMVA
 namespace associated TMVA package for ROOT.
 */

namespace TMVA {

   class Ranking;
   class PDF;
   class TSpline1;
   class MethodCuts;
   class MethodBoost;
   class DataSetInfo;
      /**
      \class RMethodBase
         Virtual base class for all TMVA methods based on ROOT::R
         <a href="http://oproject.org/ROOTR">  see http://oproject.org/ROOTR</a><br>
         <h2>Users Guide </h2>
         <a href="http://oproject.org/RMVA"> http://oproject.org/RMVA</a><br>
         \authors Omar Zapata, Lorenzo Moneta, Sergei Gleyzer
         \ingroup TMVA
       */
   class RMethodBase : public MethodBase {

      friend class Factory;
   protected:
      ROOT::R::TRInterface &r;
   public:
         /**
         Default constructor that inherits from TMVA::MethodBase and it have a ROOT::R::TRInterface instance for internal use.
         \param jobName Name taken from method type
         \param methodType Associate TMVA::Types::EMVA (available MVA methods)
         \param methodTitle Sub method associate to method type.
         \param dsi TMVA::DataSetInfo object
         \param theOption Booking options for method
         \param theBaseDir object to TDirectory with the path to calculate histograms and results for current method.
         \param _r ROOTR's object to parse R's code.
         */
      RMethodBase(const TString &jobName,
                  Types::EMVA methodType,
                  const TString &methodTitle,
                  DataSetInfo &dsi,
                  const TString &theOption = "",
                  TDirectory *theBaseDir = 0, ROOT::R::TRInterface &_r = ROOT::R::TRInterface::Instance());

         /**
         Constructor used for Testing + Application of the MVA, only (no training), using given weight file. 
         inherits from TMVA::MethodBase and it have a ROOT::R::TRInterface instance for internal use.
         \param methodType Associate TMVA::Types::EMVA (available MVA methods)
         \param dsi TMVA::DataSetInfo object
         \param theBaseDir object to TDirectory with the path to calculate histograms and results for current method.
         \param _r ROOTR's object to parse R's code.
         */
      RMethodBase(Types::EMVA methodType,
                  DataSetInfo &dsi,
                  const TString &weightFile,
                  TDirectory *theBaseDir = 0, ROOT::R::TRInterface &_r = ROOT::R::TRInterface::Instance());

      // default destructor
      virtual ~RMethodBase() {};
         /**
         Pure abstract method to build the train system
         */
      virtual void     Train() = 0;
         /**
         Pure abstract method for options treatment(some default options initialization)
         */
      virtual void     Init()           = 0;
         /**
         Pure abstract method to declare booking options associate to multivariate algorithm.
         */
      virtual void     DeclareOptions() = 0;
         /**
         Pure abstract method to parse booking options associate to multivariate algorithm.
         */
      virtual void     ProcessOptions() = 0;
         /**
         Pure abstract method to create ranking.
         \return const TMVA::Ranking pointer object
         */
      virtual const Ranking *CreateRanking() = 0;
         /**
         Pure abstract method for classifier response.
         Some methods may return a per-event error estimate
         error calculation is skipped if err==0
         \param errLower lower error estimate
         \param errUpper upper error estimate
         \return mva value for Training/Testing data.
         */
      virtual Double_t GetMvaValue(Double_t *errLower = 0, Double_t *errUpper = 0) = 0;

         /**
         Pure abstract method that return what kind of analysis types support,
         Regression or Classification for two or more classes.
         \param type TMVA::Types::EAnalysisType indicating if can support Classification or Regression.
         \param numberClasses number of classes to evaluate to know if is two class or multi class Classification.
         \param numberTargets number of targets for Regression.
         \return boolean value indicating if the given options are supported
         */
      Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets) = 0;
   protected:
         /**
         Pure abstract method that let you build the actual weights and class attributes information in a xml file
         useful for persistence model, that let you do tests on data without retrain the model. 
         \param parent xml node like a pointer to void.
         */       
      virtual void AddWeightsXMLTo(void *parent) const = 0;
         /**
         Pure abstract method that let you read the  weights and class attributes information from a xml file
         useful for persistence model, that let you do tests on data without retrain the model. 
         \param parent xml node like a pointer to void.
         */       
      virtual void ReadWeightsFromXML(void *wghtnode) = 0;
      virtual void ReadWeightsFromStream(std::istream &) = 0;        // backward compatibility
      virtual void ReadWeightsFromStream(TFile &) {}                 // backward compatibility


      void LoadData();//Read data from Data() Aand DataInfo() to Dataframes and Vectors
   protected:
      ROOT::R::TRDataFrame fDfTrain;//signal and backgrd
      ROOT::R::TRDataFrame fDfTest;
      TVectorD             fWeightTrain;
      TVectorD             fWeightTest;
      std::vector<std::string> fFactorTrain;
      std::vector<std::string> fFactorTest;
      ROOT::R::TRDataFrame fDfSpectators;

   private:
      ClassDef(RMethodBase, 0) // Virtual base class for all TMVA method

   };
} // namespace TMVA

#endif


