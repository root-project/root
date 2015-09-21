// @(#)root/tmva/rmva $Id$
// Author: Omar Zapata,Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RMethodC50                                                            *
 *                                                                                *
 * Description:                                                                   *
 *      R´s Package C50  method based on ROOTR                                    *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_RMethodC50
#define ROOT_TMVA_RMethodC50

#ifndef ROOT_TMVA_RMethodBase
#include "TMVA/RMethodBase.h"
#endif
/**
 @namespace TMVA
 namespace associated TMVA package for ROOT.
 
 */
namespace TMVA {

   class Factory;  // DSMTEST
   class Reader;   // DSMTEST
   class DataSetManager;  // DSMTEST
   class Types;
   
   
         /**
        \class MethodC50
         \brief RMVA class for method C50
                Fit classification tree models or rule-based models using Quinlan’s C5.0 algorithm.
                This method uses the package https://cran.r-project.org/web/packages/C50/index.html
          
          \section BookingC50 The Booking options for C50 are
          <table>
          <tr><td colspan="4"><span style="color:#06F; background-color:">Configuration options reference for MVA method: C50</span></td></tr>
          <tr><td><span style="color:#06F; background-color:">Option</span></td>
              <td><span style="color:#06F; background-color:">Default Value</span></td>
              <td><span style="color:#06F; background-color:">Predefined values</span></td>
              <td><span style="color:#06F; background-color:">Description</span></td></tr>
          <tr><td><strong>nTrials</strong></td><td> 1  </td><td> - </td><td>an integer specifying the number of boosting iterations. A           value of one indicates that a single model is used.   </td></tr>
          <tr><td><strong>Rules</strong></td><td>  kFALSE </td><td> - </td><td>A logical: should the tree be decomposed into a rule-based           model?</td></tr>
          <tr><td colspan="4"><span style="color:#06F; background-color:">Configuration options for C5.0Control : C50</span></td></tr>
          <tr><td><strong>ControlSubset</strong></td><td> kTRUE </td><td> - </td><td>AA logical: should the model evaluate groups of discrete predictors for splits? Note: the C5.0 command line version defaults this parameter to ‘FALSE’, meaning no attempted gropings will be evaluated during the tree growing stage.  </td></tr>
          <tr><td><strong>ControlBands</strong></td><td>0 </td><td>2-1000 </td><td> An integer between 2 and 1000. If ‘TRUE’, the model orders the rules by their affect on the error rate and groups the rules into the specified number of bands. This modifies the output so that the effect on the error rate can be seen for  the groups of rules within a band. If this options is selected and ‘rules = kFALSE’, a warning is issued and ‘rules’ is changed to ‘kTRUE’.  </td></tr>
          <tr><td><strong>ControlWinnow</strong> </td><td> kFALSE </td><td> - </td><td> A logical: should predictor winnowing (i.e feature selection) be used?  </td></tr>
          <tr><td><strong>ControlNoGlobalPruning</strong> </td><td>kFALSE</td><td> -  </td><td> A logical to toggle whether the final, global pruning step to simplify the tree.</td></tr>
          <tr><td><strong>ControlCF</strong></td><td> 0.25 </td><td> - </td><td> A number in (0, 1) for the confidence factor.</td></tr>
          <tr><td><strong>ControlMinCases </strong></td><td> 2 </td><td> - </td><td> An integer for the smallest number of samples that must be put in at least two of the splits.</td></tr>
          <tr><td><strong>ControlFuzzyThreshold</strong> </td><td> kFALSE </td><td> - </td><td> A logical toggle to evaluate possible advanced splits of the data. See Quinlan (1993) for details and examples.</td></tr>
          <tr><td><strong>ControlSample</strong></td><td> 0 </td><td> - </td><td> A value between (0, .999) that specifies the random proportion of the data should be used to train the model. By default, all the samples are used for model training. Samples not used for training are used to evaluate the accuracy of the model in the printed output.</td></tr>
          <tr><td><strong>ControlSeed</strong> </td><td> ? </td><td> -  </td><td>  An integer for the random number seed within the C code.</td></tr>
          <tr><td><strong>ControlEarlyStopping</strong> </td><td> kTRUE </td><td> -  </td><td> A logical to toggle whether the internal method for stopping boosting should be used.</td></tr></table>
          <h3>Example Booking C50 to generate Rule Based Model</h3>
          \code{.cpp}
          factory->BookMethod( TMVA::Types::kC50, "C50",
          "!H:NTrials=10:Rules=kTRUE:ControlSubSet=kFALSE:ControlBands=10:ControlWinnow=kFALSE:ControlNoGlobalPruning=kTRUE:ControlCF=0.25:ControlMinCases=2:ControlFuzzyThreshold=kTRUE:ControlSample=0:ControlEarlyStopping=kTRUE:!V" );
          \endcode
          NOTE: Options Rules=kTRUE and to Control the bands use ControlBands
          <h3>Example Booking C50 to generate Boosted Decision Trees Model</h3>
          \code{.cpp}
          factory->BookMethod( TMVA::Types::kC50, "C50",
         "!H:NTrials=10:Rules=kFALSE:ControlSubSet=kFALSE:ControlWinnow=kFALSE:ControlNoGlobalPruning=kTRUE:ControlCF=0.25:ControlMinCases=2:ControlFuzzyThreshold=kTRUE:ControlSample=0:ControlEarlyStopping=kTRUE:!V" );
          \endcode
          <a href="http://oproject.org/RMVA#C50">  see http://oproject.org/RMVA#C50</a><br>
         
         <h3>Website:</h3>\link http://oproject.org/RMVA
         \ingroup TMVA
       */

   class MethodC50 : public RMethodBase {

   public :
         /**
         Default constructor that inherits from TMVA::RMethodBase and it have a ROOT::R::TRInterface instance for internal use.
         \param jobName Name taken from method type
         \param methodType Associate TMVA::Types::EMVA (available MVA methods)
         \param methodTitle Sub method associate to method type.
         \param dsi TMVA::DataSetInfo object
         \param theOption Booking options for method
         \param theBaseDir object to TDirectory with the path to calculate histograms and results for current method.
         */
      MethodC50(const TString &jobName,
                const TString &methodTitle,
                DataSetInfo &theData,
                const TString &theOption = "",
                TDirectory *theTargetDir = NULL);
         /**
         Constructor used for Testing + Application of the MVA, only (no training), using given weight file. 
         inherits from TMVA::MethodBase and it have a ROOT::R::TRInterface instance for internal use.
         \param methodType Associate TMVA::Types::EMVA (available MVA methods)
         \param dsi TMVA::DataSetInfo object
         \param theBaseDir object to TDirectory with the path to calculate histograms and results for current method.
         */
      MethodC50(DataSetInfo &dsi,
                const TString &theWeightFile,
                TDirectory *theTargetDir = NULL);


      ~MethodC50(void);
         /**
         Pure abstract method to train system,
         it call C5.0 and C5.0Control with the options to created a trained model.
         */
      void     Train();
         /**
         Pure abstract method for options treatment(some default options initialization),
         try to load C50 R's package.
         */
      void     Init();
         /**
         Pure abstract method to declare booking options associate to multivariate algorithm.
         */

      void     DeclareOptions();
         /**
         Pure abstract method to parse booking options associate to multivariate algorithm.
         \see BookingC50
         */      
      void     ProcessOptions();
      // create ranking
      const Ranking *CreateRanking()
      {
         return NULL;  // = 0;
      }


      Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets);

      // performs classifier testing
      virtual void     TestClassification();


      Double_t GetMvaValue(Double_t *errLower = 0, Double_t *errUpper = 0);
      virtual void     MakeClass(const TString &classFileName = TString("")) const;  //required for model persistence
      using MethodBase::ReadWeightsFromStream;
      // the actual "weights"
      virtual void AddWeightsXMLTo(void *parent) const {}        // = 0;
      virtual void ReadWeightsFromXML(void *wghtnode) {}    // = 0;
      virtual void ReadWeightsFromStream(std::istream &) {} //= 0;       // backward compatibility

      void ReadStateFromFile();
   private :
      DataSetManager    *fDataSetManager;     // DSMTEST
      friend class Factory;                   // DSMTEST
      friend class Reader;                    // DSMTEST
   protected:
      //C5.0 function options
      UInt_t fNTrials;//number of trials with boost enabled
      Bool_t fRules;//A logical: should the tree be decomposed into a rule-based model?

      //Control options see C5.0Control
      Bool_t fControlSubset; //A logical: should the model evaluate groups of discrete predictors for splits?
      UInt_t fControlBands;
      Bool_t fControlWinnow;// A logical: should predictor winnowing (i.e feature selection) be used?
      Bool_t fControlNoGlobalPruning; //A logical to toggle whether the final, global pruning step to simplify the tree.
      Double_t fControlCF; //A number in (0, 1) for the confidence factor.
      UInt_t fControlMinCases;//an integer for the smallest number of samples that must be put in at least two of the splits.
      Bool_t fControlFuzzyThreshold;//A logical toggle to evaluate possible advanced splits of the data. See Quinlan (1993) for details and examples.
      Double_t fControlSample;//A value between (0, .999) that specifies the random proportion of the data should be used to train the model.
      Int_t fControlSeed;//An integer for the random number seed within the C code.
      Bool_t fControlEarlyStopping;// logical to toggle whether the internal method for stopping boosting should be used.

      UInt_t fMvaCounter;
      static Bool_t IsModuleLoaded;

      ROOT::R::TRFunctionImport predict;
      ROOT::R::TRFunctionImport C50;
      ROOT::R::TRFunctionImport C50Control;
      ROOT::R::TRFunctionImport asfactor;
      ROOT::R::TRObject *fModel;
      ROOT::R::TRObject fModelControl;
      std::vector <TString > ListOfVariables;


      // get help message text
      void GetHelpMessage() const;

      ClassDef(MethodC50, 0)
   };
} // namespace TMVA
#endif
