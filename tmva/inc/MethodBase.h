// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Kai Voss, Eckhard von Toerne, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodBase                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Virtual base class for all MVA method                                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <peter.speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Jan Therhaag          <Jan.Therhaag@cern.ch>   - U of Bonn, Germany       *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>        - U of Bonn, Germany       *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodBase
#define ROOT_TMVA_MethodBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodBase                                                           //
//                                                                      //
// Virtual base class for all TMVA method                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iosfwd>
#include <vector>
#include <map>
#include "assert.h"

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TMVA_IMethod
#include "TMVA/IMethod.h"
#endif
#ifndef ROOT_TMVA_Configurable
#include "TMVA/Configurable.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_TransformationHandler
#include "TMVA/TransformationHandler.h"
#endif
#ifndef ROOT_TMVA_OptimizeConfigParameters
#include "TMVA/OptimizeConfigParameters.h"
#endif

class TGraph;
class TTree;
class TDirectory;
class TSpline;
class TH1F;
class TH1D;

namespace TMVA {

   class Ranking;
   class PDF;
   class TSpline1;
   class MethodCuts;
   class MethodBoost;
   class DataSetInfo;

   class MethodBase : virtual public IMethod, public Configurable {

      friend class Factory;

   public:

      enum EWeightFileType { kROOT=0, kTEXT };

      // default constructur
      MethodBase( const TString& jobName,
                  Types::EMVA methodType,
                  const TString& methodTitle,
                  DataSetInfo& dsi,
                  const TString& theOption = "",
                  TDirectory* theBaseDir = 0 );

      // constructor used for Testing + Application of the MVA, only (no training),
      // using given weight file
      MethodBase( Types::EMVA methodType,
                  DataSetInfo& dsi,
                  const TString& weightFile,
                  TDirectory* theBaseDir = 0 );

      // default destructur
      virtual ~MethodBase();

      // declaration, processing and checking of configuration options
      void             SetupMethod();
      void             ProcessSetup();
      virtual void     CheckSetup(); // may be overwritten by derived classes

      // ---------- main training and testing methods ------------------------------

      // prepare tree branch with the method's discriminating variable
      void             AddOutput( Types::ETreeType type, Types::EAnalysisType analysisType );

      // performs classifier training
      // calls methods Train() implemented by derived classes
      void             TrainMethod();

      // optimize tuning parameters
      virtual std::map<TString,Double_t> OptimizeTuningParameters(TString fomType="ROCIntegral", TString fitType="FitGA");
      virtual void SetTuneParameters(std::map<TString,Double_t> tuneParameters);

      virtual void     Train() = 0;

      // store and retrieve time used for training
      void             SetTrainTime( Double_t trainTime ) { fTrainTime = trainTime; }
      Double_t         GetTrainTime() const { return fTrainTime; }

      // store and retrieve time used for testing
      void             SetTestTime ( Double_t testTime ) { fTestTime = testTime; }
      Double_t         GetTestTime () const { return fTestTime; }

      // performs classifier testing
      virtual void     TestClassification();

      // performs multiclass classifier testing
      virtual void     TestMulticlass();

      // performs regression testing
      virtual void     TestRegression( Double_t& bias, Double_t& biasT,
                                       Double_t& dev,  Double_t& devT,
                                       Double_t& rms,  Double_t& rmsT,
                                       Double_t& mInf, Double_t& mInfT, // mutual information
                                       Double_t& corr,
                                       Types::ETreeType type );

      // options treatment
      virtual void     Init()           = 0;
      virtual void     DeclareOptions() = 0;
      virtual void     ProcessOptions() = 0;
      virtual void     DeclareCompatibilityOptions(); // declaration of past options

      // reset the Method --> As if it was not yet trained, just instantiated
      //      virtual void     Reset()          = 0;
      //for the moment, I provide a dummy (that would not work) default, just to make
      // compilation/running w/o parameter optimisation still possible
      virtual void     Reset(){return;}

      // classifier response:
      // some methods may return a per-event error estimate
      // error calculation is skipped if err==0
      virtual Double_t GetMvaValue( Double_t* errLower = 0, Double_t* errUpper = 0) = 0;

      // signal/background classification response
      Double_t GetMvaValue( const TMVA::Event* const ev, Double_t* err = 0, Double_t* errUpper = 0 );

   protected:
      // helper function to set errors to -1
      void NoErrorCalc(Double_t* const err, Double_t* const errUpper);

   public:
      // regression response
      virtual const std::vector<Float_t>& GetRegressionValues() {
         std::vector<Float_t>* ptr = new std::vector<Float_t>(0);
         return (*ptr);
      }

      // multiclass classification response
      virtual const std::vector<Float_t>& GetMulticlassValues() {
         std::vector<Float_t>* ptr = new std::vector<Float_t>(0);
         return (*ptr);
      }

      // probability of classifier response (mvaval) to be signal (requires "CreateMvaPdf" option set)
      virtual Double_t GetProba( Double_t mvaVal, Double_t ap_sig );

      // Rarity of classifier response (signal or background (default) is uniform in [0,1])
      virtual Double_t GetRarity( Double_t mvaVal, Types::ESBType reftype = Types::kBackground ) const;

      // create ranking
      virtual const Ranking* CreateRanking() = 0;

      // perfrom extra actions during the boosting at different stages
      virtual Bool_t   MonitorBoost(MethodBoost* /*booster*/) {return kFALSE;};

      // make ROOT-independent C++ class
      virtual void     MakeClass( const TString& classFileName = TString("") ) const;

      // print help message
      void             PrintHelpMessage() const;

      //
      // streamer methods for training information (creates "weight" files) --------
      //
   public:
      void WriteStateToFile     () const;
      void ReadStateFromFile    ();

   protected:
      // the actual "weights"
      virtual void AddWeightsXMLTo      ( void* parent ) const = 0;
      virtual void ReadWeightsFromXML   ( void* wghtnode ) = 0;
      virtual void ReadWeightsFromStream( std::istream& ) = 0;       // backward compatibility
      virtual void ReadWeightsFromStream( TFile& ) {}                // backward compatibility

   private:
      friend class MethodCategory;
      friend class MethodCommittee;
      friend class MethodCompositeBase;
      void WriteStateToXML      ( void* parent ) const;
      void ReadStateFromXML     ( void* parent );
      void WriteStateToStream   ( std::ostream& tf ) const;   // needed for MakeClass
      void WriteVarsToStream    ( std::ostream& tf, const TString& prefix = "" ) const;  // needed for MakeClass


   public: // these two need to be public, they are used to read in-memory weight-files
      void ReadStateFromStream  ( std::istream& tf );         // backward compatibility
      void ReadStateFromStream  ( TFile&        rf );         // backward compatibility
      void ReadStateFromXMLString( const char* xmlstr );      // for reading from memory

   private:
      // the variable information
      void AddVarsXMLTo         ( void* parent  ) const;
      void AddSpectatorsXMLTo   ( void* parent  ) const;
      void AddTargetsXMLTo      ( void* parent  ) const;
      void AddClassesXMLTo      ( void* parent  ) const;
      void ReadVariablesFromXML ( void* varnode );
      void ReadSpectatorsFromXML( void* specnode);
      void ReadTargetsFromXML   ( void* tarnode );
      void ReadClassesFromXML   ( void* clsnode );
      void ReadVarsFromStream   ( std::istream& istr );       // backward compatibility

   public:
      // ---------------------------------------------------------------------------

      // write evaluation histograms into target file
      virtual void     WriteEvaluationHistosToFile(Types::ETreeType treetype);

      // write classifier-specific monitoring information to target file
      virtual void     WriteMonitoringHistosToFile() const;

      // ---------- public evaluation methods --------------------------------------

      // individual initialistion for testing of each method
      // overload this one for individual initialisation of the testing,
      // it is then called automatically within the global "TestInit"

      // variables (and private menber functions) for the Evaluation:
      // get the effiency. It fills a histogram for efficiency/vs/bkg
      // and returns the one value fo the efficiency demanded for 
      // in the TString argument. (Watch the string format)
      virtual Double_t GetEfficiency( const TString&, Types::ETreeType, Double_t& err );
      virtual Double_t GetTrainingEfficiency(const TString& );
      virtual std::vector<Float_t> GetMulticlassEfficiency( std::vector<std::vector<Float_t> >& purity );
      virtual std::vector<Float_t> GetMulticlassTrainingEfficiency(std::vector<std::vector<Float_t> >& purity );
      virtual Double_t GetSignificance() const;
      virtual Double_t GetROCIntegral(TH1F *histS, TH1F *histB) const;
      //      virtual Double_t GetROCIntegral(TH1D *histS, TH1D *histB) const;
      virtual Double_t GetROCIntegral(PDF *pdfS=0, PDF *pdfB=0) const;
      virtual Double_t GetMaximumSignificance( Double_t SignalEvents, Double_t BackgroundEvents, 
                                               Double_t& optimal_significance_value  ) const;
      virtual Double_t GetSeparation( TH1*, TH1* ) const;
      virtual Double_t GetSeparation( PDF* pdfS = 0, PDF* pdfB = 0 ) const;

      virtual void GetRegressionDeviation(UInt_t tgtNum, Types::ETreeType type, Double_t& stddev,Double_t& stddev90Percent ) const;
      // ---------- public accessors -----------------------------------------------

      // classifier naming (a lot of names ... aren't they ;-)
      const TString&   GetJobName       () const { return fJobName; }
      const TString&   GetMethodName    () const { return fMethodName; }
      TString          GetMethodTypeName() const { return Types::Instance().GetMethodName(fMethodType); }
      Types::EMVA      GetMethodType    () const { return fMethodType; }
      const char*      GetName          () const { return fMethodName.Data(); }
      const TString&   GetTestvarName   () const { return fTestvar; }
      const TString    GetProbaName     () const { return fTestvar + "_Proba"; }
      TString          GetWeightFileName() const;

      // build classifier name in Test tree
      // MVA prefix (e.g., "TMVA_")
      void             SetTestvarName  ( const TString & v="" ) { fTestvar = (v=="") ? ("MVA_" + GetMethodName()) : v; }

      // number of input variable used by classifier
      UInt_t           GetNvar()       const { return DataInfo().GetNVariables(); }
      UInt_t           GetNVariables() const { return DataInfo().GetNVariables(); }
      UInt_t           GetNTargets()   const { return DataInfo().GetNTargets(); };

      // internal names and expressions of input variables
      const TString&   GetInputVar  ( Int_t i ) const { return DataInfo().GetVariableInfo(i).GetInternalName(); }
      const TString&   GetInputLabel( Int_t i ) const { return DataInfo().GetVariableInfo(i).GetLabel(); }
      const TString&   GetInputTitle( Int_t i ) const { return DataInfo().GetVariableInfo(i).GetTitle(); }

      // normalisation and limit accessors
      Double_t         GetMean( Int_t ivar ) const { return GetTransformationHandler().GetMean(ivar); }
      Double_t         GetRMS ( Int_t ivar ) const { return GetTransformationHandler().GetRMS(ivar); }
      Double_t         GetXmin( Int_t ivar ) const { return GetTransformationHandler().GetMin(ivar); }
      Double_t         GetXmax( Int_t ivar ) const { return GetTransformationHandler().GetMax(ivar); }

      // sets the minimum requirement on the MVA output to declare an event signal-like
      Double_t         GetSignalReferenceCut() const { return fSignalReferenceCut; }
      Double_t         GetSignalReferenceCutOrientation() const { return fSignalReferenceCutOrientation; }

      // sets the minimum requirement on the MVA output to declare an event signal-like
      void             SetSignalReferenceCut( Double_t cut ) { fSignalReferenceCut = cut; }
      void             SetSignalReferenceCutOrientation( Double_t cutOrientation ) { fSignalReferenceCutOrientation = cutOrientation; }

      // pointers to ROOT directories
      TDirectory*      BaseDir()       const;
      TDirectory*      MethodBaseDir() const;
      void             SetMethodDir ( TDirectory* methodDir ) { fBaseDir = fMethodBaseDir  = methodDir; }
      void             SetBaseDir( TDirectory* methodDir ){ fBaseDir = methodDir; }
      void             SetMethodBaseDir( TDirectory* methodDir ){ fMethodBaseDir = methodDir; }

      // the TMVA version can be obtained and checked using
      //    if (GetTrainingTMVAVersionCode()>TMVA_VERSION(3,7,2)) {...}
      // or
      //    if (GetTrainingROOTVersionCode()>ROOT_VERSION(5,15,5)) {...}
      UInt_t           GetTrainingTMVAVersionCode()   const { return fTMVATrainingVersion; }
      UInt_t           GetTrainingROOTVersionCode()   const { return fROOTTrainingVersion; }
      TString          GetTrainingTMVAVersionString() const;
      TString          GetTrainingROOTVersionString() const;

      TransformationHandler&        GetTransformationHandler(Bool_t takeReroutedIfAvailable=true) 
          { 
	     if(fTransformationPointer && takeReroutedIfAvailable) return *fTransformationPointer; else return fTransformation; 
	  }
      const TransformationHandler&  GetTransformationHandler(Bool_t takeReroutedIfAvailable=true) const 
          { 
	     if(fTransformationPointer && takeReroutedIfAvailable) return *fTransformationPointer; else return fTransformation; 
	  }

      void             RerouteTransformationHandler (TransformationHandler* fTargetTransformation) { fTransformationPointer=fTargetTransformation; }

      // ---------- event accessors ------------------------------------------------

      // returns reference to data set
      DataSetInfo&     DataInfo() const { return fDataSetInfo; }

      mutable const Event*   fTmpEvent; //! temporary event when testing on a different DataSet than the own one

      // event reference and update
      UInt_t           GetNEvents      () const { return Data()->GetNEvents(); }
      const Event*     GetEvent        () const;
      const Event*     GetEvent        ( const TMVA::Event* ev ) const;
      const Event*     GetEvent        ( Long64_t ievt ) const;
      const Event*     GetEvent        ( Long64_t ievt , Types::ETreeType type ) const;
      const Event*     GetTrainingEvent( Long64_t ievt ) const;
      const Event*     GetTestingEvent ( Long64_t ievt ) const;
      const std::vector<TMVA::Event*>& GetEventCollection( Types::ETreeType type );

      // ---------- public auxiliary methods ---------------------------------------

      // this method is used to decide whether an event is signal- or background-like
      // the reference cut "xC" is taken to be where
      // Int_[-oo,xC] { PDF_S(x) dx } = Int_[xC,+oo] { PDF_B(x) dx }
      virtual Bool_t        IsSignalLike();
      virtual Bool_t        IsSignalLike(Double_t mvaVal);

      DataSet* Data() const { return DataInfo().GetDataSet(); }

      Bool_t                HasMVAPdfs() const { return fHasMVAPdfs; }
      virtual void          SetAnalysisType( Types::EAnalysisType type ) { fAnalysisType = type; }
      Types::EAnalysisType  GetAnalysisType() const { return fAnalysisType; }
      Bool_t                DoRegression() const { return fAnalysisType == Types::kRegression; }
      Bool_t                DoMulticlass() const { return fAnalysisType == Types::kMulticlass; }

      // setter method for suppressing writing to XML and writing of standalone classes
      void                  DisableWriting(Bool_t setter){ fDisableWriting = setter; }

   protected:

      // ---------- protected acccessors -------------------------------------------

      //TDirectory*  LocalTDir() const { return Data().LocalRootDir(); }

      // weight file name and directory (given by global config variable)
      void             SetWeightFileName( TString );

      const TString&   GetWeightFileDir() const { return fFileDir; }
      void             SetWeightFileDir( TString fileDir );

      // are input variables normalised ?
      Bool_t           IsNormalised() const { return fNormalise; }
      void             SetNormalised( Bool_t norm ) { fNormalise = norm; }

      // set number of input variables (only used by MethodCuts, could perhaps be removed)
      //      void SetNvar( Int_t n ) { fNvar = n; }

      // verbose and help flags
      Bool_t           Verbose() const { return fVerbose; }
      Bool_t           Help   () const { return fHelp; }

      // ---------- protected event and tree accessors -----------------------------

      // names of input variables (if the original names are expressions, they are 
      // transformed into regexps)
      const TString&   GetInternalVarName( Int_t ivar ) const { return (*fInputVars)[ivar]; }
      const TString&   GetOriginalVarName( Int_t ivar ) const { return DataInfo().GetVariableInfo(ivar).GetExpression(); }

      Bool_t           HasTrainingTree() const { return Data()->GetNTrainingEvents() != 0; }

      // ---------- protected auxiliary methods ------------------------------------

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      virtual void     MakeClassSpecific( std::ostream&, const TString& = "" ) const {}

      // header and auxiliary classes
      virtual void     MakeClassSpecificHeader( std::ostream&, const TString& = "" ) const {}

      // static pointer to this object - required for ROOT finder (to be solved differently)
      static MethodBase* GetThisBase();

      // some basic statistical analysis
      void Statistics( Types::ETreeType treeType, const TString& theVarName,
                       Double_t&, Double_t&, Double_t&, 
                       Double_t&, Double_t&, Double_t& );

      // if TRUE, write weights only to text files 
      Bool_t           TxtWeightsOnly() const { return kTRUE; }

   protected:
      
      // access to event information that needs method-specific information
      
      Float_t GetTWeight( const Event* ev ) const { 
         return (fIgnoreNegWeightsInTraining && (ev->GetWeight() < 0)) ? 0. : ev->GetWeight(); 
      }

      Bool_t           IsConstructedFromWeightFile() const { return fConstructedFromWeightFile; }

   public:
      virtual void SetCurrentEvent( Long64_t ievt ) const {
         Data()->SetCurrentEvent(ievt);
      }


   private:

      // ---------- private definitions --------------------------------------------
      // Initialisation
      void             InitBase();
      void             DeclareBaseOptions();
      void             ProcessBaseOptions();

      // used in efficiency computation
      enum ECutOrientation { kNegative = -1, kPositive = +1 };
      ECutOrientation  GetCutOrientation() const { return fCutOrientation; }

      // ---------- private acccessors ---------------------------------------------

      // reset required for RootFinder
      void             ResetThisBase();

      // ---------- private auxiliary methods --------------------------------------

      // PDFs for classifier response (required to compute signal probability and Rarity)
      void             CreateMVAPdfs();

      // for root finder 
      static Double_t  IGetEffForRoot( Double_t );  // interface
      Double_t         GetEffForRoot ( Double_t );  // implementation

      // used for file parsing
      Bool_t           GetLine( std::istream& fin, char * buf );
      
      // fill test tree with classification or regression results
      virtual void     AddClassifierOutput    ( Types::ETreeType type );
      virtual void     AddClassifierOutputProb( Types::ETreeType type );
      virtual void     AddRegressionOutput    ( Types::ETreeType type );
      virtual void     AddMulticlassOutput    ( Types::ETreeType type );

   private:

      void             AddInfoItem( void* gi, const TString& name, const TString& value) const;

      static void      CreateVariableTransforms(const TString& trafoDefinition, 
						TMVA::DataSetInfo& dataInfo,
						TMVA::TransformationHandler& transformationHandler,
						TMVA::MsgLogger& log );


      // ========== class members ==================================================

   protected:

      // direct accessors
      Ranking*              fRanking;              // pointer to ranking object (created by derived classifiers)
      std::vector<TString>* fInputVars;            // vector of input variables used in MVA

      // histogram binning
      Int_t                 fNbins;                // number of bins in input variable histograms
      Int_t                 fNbinsMVAoutput;       // number of bins in MVA output histograms
      Int_t                 fNbinsH;               // number of bins in evaluation histograms

      Types::EAnalysisType  fAnalysisType;         // method-mode : true --> regression, false --> classification

      std::vector<Float_t>* fRegressionReturnVal;  // holds the return-values for the regression
      std::vector<Float_t>* fMulticlassReturnVal;  // holds the return-values for the multiclass classification

   private:

      // MethodCuts redefines some of the evaluation variables and histograms -> must access private members
      friend class MethodCuts; 

      Bool_t           fDisableWriting;       //! set to true in order to suppress writing to XML

      // data sets
      DataSetInfo&     fDataSetInfo;         //! the data set information (sometimes needed)

      Double_t         fSignalReferenceCut;  // minimum requirement on the MVA output to declare an event signal-like
      Double_t         fSignalReferenceCutOrientation;  // minimum requirement on the MVA output to declare an event signal-like
      Types::ESBType   fVariableTransformType;  // this is the event type (sig or bgd) assumed for variable transform

      // naming and versioning
      TString          fJobName;             // name of job -> user defined, appears in weight files
      TString          fMethodName;          // name of the method (set in derived class)
      Types::EMVA      fMethodType;          // type of method (set in derived class)      
      TString          fTestvar;             // variable used in evaluation, etc (mostly the MVA)
      UInt_t           fTMVATrainingVersion; // TMVA version used for training
      UInt_t           fROOTTrainingVersion; // ROOT version used for training
      Bool_t           fConstructedFromWeightFile; // is it obtained from weight file? 

      // Directory structure: fMethodBaseDir/fBaseDir
      // where the first directory name is defined by the method type
      // and the second is user supplied (the title given in Factory::BookMethod())
      TDirectory*      fBaseDir;             // base directory for the instance, needed to know where to jump back from localDir
      mutable TDirectory* fMethodBaseDir;    // base directory for the method

      TString          fParentDir;           // method parent name, like booster name

      TString          fFileDir;             // unix sub-directory for weight files (default: "weights")
      TString          fWeightFile;          // weight file name

   private:

      TH1*             fEffS;                // efficiency histogram for rootfinder

      PDF*             fDefaultPDF;          // default PDF definitions
      PDF*             fMVAPdfS;             // signal MVA PDF
      PDF*             fMVAPdfB;             // background MVA PDF

      TH1F*            fmvaS;                // PDFs of MVA distribution (signal)
      TH1F*            fmvaB;                // PDFs of MVA distribution (background)
      PDF*             fSplS;                // PDFs of MVA distribution (signal)
      PDF*             fSplB;                // PDFs of MVA distribution (background)
      TSpline*         fSpleffBvsS;          // splines for signal eff. versus background eff.

      PDF*             fSplTrainS;           // PDFs of training MVA distribution (signal)
      PDF*             fSplTrainB;           // PDFs of training MVA distribution (background)
      TSpline*         fSplTrainEffBvsS;     // splines for training signal eff. versus background eff.

   private:

      // basic statistics quantities of MVA
      Double_t         fMeanS;               // mean (signal)
      Double_t         fMeanB;               // mean (background)
      Double_t         fRmsS;                // RMS (signal)
      Double_t         fRmsB;                // RMS (background)
      Double_t         fXmin;                // minimum (signal and background)
      Double_t         fXmax;                // maximum (signal and background)

      // variable preprocessing
      TString          fVarTransformString;          // labels variable transform method

      TransformationHandler* fTransformationPointer;  // pointer to the rest of transformations
      TransformationHandler  fTransformation;         // the list of transformations


      // help and verbosity
      Bool_t           fVerbose;               // verbose flag
      TString          fVerbosityLevelString;  // verbosity level (user input string)
      EMsgType         fVerbosityLevel;        // verbosity level
      Bool_t           fHelp;                  // help flag
      Bool_t           fHasMVAPdfs;            // MVA Pdfs are created for this classifier

      Bool_t           fIgnoreNegWeightsInTraining;// If true, events with negative weights are not used in training

   protected:

      Bool_t           IgnoreEventsWithNegWeightsInTraining() const { return fIgnoreNegWeightsInTraining; }

      // for signal/background
      UInt_t           fSignalClass;           // index of the Signal-class
      UInt_t           fBackgroundClass;       // index of the Background-class

   private:

      // timing variables
      Double_t         fTrainTime;             // for timing measurements
      Double_t         fTestTime;              // for timing measurements

      // orientation of cut: depends on signal and background mean values
      ECutOrientation  fCutOrientation;      // +1 if Sig>Bkg, -1 otherwise

      // for root finder
      TSpline1*        fSplRefS;             // helper splines for RootFinder (signal)
      TSpline1*        fSplRefB;             // helper splines for RootFinder (background)

      TSpline1*        fSplTrainRefS;        // helper splines for RootFinder (signal)
      TSpline1*        fSplTrainRefB;        // helper splines for RootFinder (background)

      mutable std::vector<const std::vector<TMVA::Event*>*> fEventCollections; // if the method needs the complete event-collection, the transformed event coll. ist stored here.

   public:
      Bool_t           fSetupCompleted;      // is method setup

   private:

      // this carrier
      static MethodBase* fgThisBase;         // this pointer


      // ===== depreciated options, kept for backward compatibility  =====
   private:

      Bool_t           fNormalise;                   // normalise input variables
      Bool_t           fUseDecorr;                   // synonymous for decorrelation
      TString          fVariableTransformTypeString; // labels variable transform type
      Bool_t           fTxtWeightsOnly;              // if TRUE, write weights only to text files 
      Int_t            fNbinsMVAPdf;                 // number of bins used in histogram that creates PDF
      Int_t            fNsmoothMVAPdf;               // number of times a histogram is smoothed before creating the PDF

   protected:

      ClassDef(MethodBase,0)  // Virtual base class for all TMVA method

   };
} // namespace TMVA







// ========== INLINE FUNCTIONS =========================================================


//_______________________________________________________________________
inline const TMVA::Event* TMVA::MethodBase::GetEvent( const TMVA::Event* ev ) const 
{
   return GetTransformationHandler().Transform(ev);
}

inline const TMVA::Event* TMVA::MethodBase::GetEvent() const 
{
   if(fTmpEvent)
      return GetTransformationHandler().Transform(fTmpEvent);
   else
      return GetTransformationHandler().Transform(Data()->GetEvent());
}

inline const TMVA::Event* TMVA::MethodBase::GetEvent( Long64_t ievt ) const 
{
   assert(fTmpEvent==0);
   return GetTransformationHandler().Transform(Data()->GetEvent(ievt));
}

inline const TMVA::Event* TMVA::MethodBase::GetEvent( Long64_t ievt, Types::ETreeType type ) const 
{
   assert(fTmpEvent==0);
   return GetTransformationHandler().Transform(Data()->GetEvent(ievt, type));
}

inline const TMVA::Event* TMVA::MethodBase::GetTrainingEvent( Long64_t ievt ) const 
{
   assert(fTmpEvent==0);
   return GetEvent(ievt, Types::kTraining);
}

inline const TMVA::Event* TMVA::MethodBase::GetTestingEvent( Long64_t ievt ) const 
{
   assert(fTmpEvent==0);
   return GetEvent(ievt, Types::kTesting);
}

#endif
