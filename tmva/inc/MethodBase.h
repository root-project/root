// @(#)root/tmva $Id: MethodBase.h,v 1.11 2006/11/23 17:43:38 rdm Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

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
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
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

#include "Riostream.h"
#include <vector>
#include <iostream>

#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif
#ifndef ROOT_TMVA_IMethod
#include "TMVA/IMethod.h"
#endif
#ifndef ROOT_TMVA_PDF
#include "TMVA/PDF.h"
#endif
#ifndef ROOT_TMVA_TSpline1
#include "TMVA/TSpline1.h"
#endif
#ifndef ROOT_TMVA_Option
#include "TMVA/Option.h"
#endif
#ifndef ROOT_TMVA_Configurable
#include "TMVA/Configurable.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_VariableTransformBase
#include "TMVA/VariableTransformBase.h"
#endif

class TTree;
class TDirectory;

namespace TMVA {

   class Ranking;

   class MethodBase : public IMethod, public Configurable {
      
   public:

      enum EWeightFileType { kROOT=0, kTEXT };
      
      // default constructur
      MethodBase( TString jobName,
                  TString methodTitle, 
                  DataSet& theData,
                  TString theOption = "", 
                  TDirectory* theBaseDir = 0 );
      
      // constructor used for Testing + Application of the MVA, only (no training), 
      // using given weight file
      MethodBase( DataSet& theData,
                  TString weightFile, 
                  TDirectory* theBaseDir = 0 );

      // default destructur
      virtual ~MethodBase( void );
      
      // prepare tree branch with the method's discriminating variable
      virtual void PrepareEvaluationTree( TTree* theTestTree );

      void   TrainMethod();
      Bool_t IsMVAPdfs() const { return fIsMVAPdfs; }
      void   CreateMVAPdfs();

      void   WriteStateToFile   () const;
      void   WriteStateToStream ( std::ostream& tf ) const;
      void   WriteStateToStream ( TFile&        rf ) const;
      void   ReadStateFromFile  ();
      void   ReadStateFromStream( std::istream& tf );
      void   ReadStateFromStream( TFile&        rf );
      
      virtual void WriteWeightsToStream ( std::ostream& tf ) const = 0;
      virtual void WriteWeightsToStream ( TFile&      /*rf*/ ) const {};
      virtual void ReadWeightsFromStream( std::istream& tf ) = 0;
      virtual void ReadWeightsFromStream( TFile&      /*rf*/ ){};

      virtual void WriteMonitoringHistosToFile( void ) const;

      virtual Bool_t IsSignalLike() { return GetMvaValue() > GetSignalReferenceCut() ? kTRUE : kFALSE; }     

      // evaluate method (resulting discriminating variable) or input varible
      virtual void TestInit( TTree* theTestTree = 0 );

      // individual initialistion for testing of each method
      // overload this one for individual initialisation of the testing, 
      // it is then called automatically within the global "TestInit" 
      
      // the new way to get the MVA value
      virtual Double_t GetMvaValue() = 0;
      virtual Double_t GetProba( Double_t mvaVal, Double_t ap_sig );

      // test the method
      virtual void Test( TTree* theTestTree = 0 );

      // accessors
      const TString&   GetJobName    ( void ) const { return fJobName; }
      const TString&   GetMethodName ( void ) const { return fMethodName; }
      const char*      GetName       ( void ) const { return GetMethodName().Data(); }
      const TString&   GetMethodTitle( void ) const { return fMethodTitle; }
      Types::EMVA      GetMethodType ( void ) const { return fMethodType; }

      void    SetJobName    ( TString jobName )        { fJobName     = jobName; }
      void    SetMethodName ( TString methodName )     { fMethodName  = methodName; }
      void    SetMethodTitle( TString methodTitle )    { fMethodTitle = methodTitle; }
      void    SetMethodType ( Types::EMVA methodType ) { fMethodType  = methodType; }

      TString GetWeightFileDir( void ) const { return fFileDir; }
      void    SetWeightFileDir( TString fileDir );

      const TString& GetInputVar( int i ) const { return Data().GetInternalVarName(i); }
      const TString& GetInputExp( int i ) const { return Data().GetExpression(i); }

      void    SetWeightFileName( TString );
      TString GetWeightFileName() const;

      Bool_t  HasTrainingTree() const { return Data().GetTrainingTree() != 0; }
      TTree*  GetTrainingTree() const { 
         if (GetVariableTransform()!=Types::kNone) {
            fLogger << kFATAL << "Trying to access correlated Training tree in method " 
                    << GetMethodName() << endl;
         }
         return Data().GetTrainingTree();
      }
      TTree*  GetTestTree() const {
         if (GetVariableTransform()!=Types::kNone) {
            fLogger << kFATAL << "Trying to access correlated Training tree in method " 
                    << GetMethodName() << endl;
         }
         return Data().GetTestTree();
      }

      Int_t   GetNvar( void ) const { return fNvar; }
      void    SetNvar( Int_t n) { fNvar = n; }

      // variables (and private menber functions) for the Evaluation:
      // get the effiency. It fills a histogram for efficiency/vs/bkg
      // and returns the one value fo the efficiency demanded for 
      // in the TString argument. (Watch the string format)
      virtual Double_t  GetEfficiency   ( TString, TTree*, Double_t& err );
      virtual Double_t  GetTrainingEfficiency( TString );
      virtual Double_t  GetSignificance ( void ) const;
      virtual Double_t  GetOptimalSignificance( Double_t SignalEvents, Double_t BackgroundEvents, 
                                                Double_t& optimal_significance_value  ) const;
      virtual Double_t  GetSeparation( TH1*, TH1* ) const;
      virtual Double_t  GetSeparation( PDF* pdfS = 0, PDF* pdfB = 0 ) const;
      virtual Double_t  GetmuTransform( TTree* );

      // normalisation accessors
      Double_t GetRMS( Int_t ivar )          const { return GetVarTransform().Variable(ivar).GetRMS(); }
      Double_t GetXmin( Int_t ivar )         const { return GetVarTransform().Variable(ivar).GetMin(); }
      Double_t GetXmax( Int_t ivar )         const { return GetVarTransform().Variable(ivar).GetMax(); }
      Double_t GetXmin( const TString& var ) const { return GetVarTransform().Variable(var).GetMin(); } // slow
      Double_t GetXmax( const TString& var ) const { return GetVarTransform().Variable(var).GetMax(); } // slow
      void     SetXmin( Int_t ivar, Double_t x )          { GetVarTransform().Variable(ivar).SetMin(x); }
      void     SetXmax( Int_t ivar, Double_t x )          { GetVarTransform().Variable(ivar).SetMax(x); }
      void     SetXmin( const TString& var, Double_t x )  { GetVarTransform().Variable(var).SetMin(x); }
      void     SetXmax( const TString& var, Double_t x )  { GetVarTransform().Variable(var).SetMax(x); }

      // main normalization method is in Tools
      Double_t Norm       ( Int_t ivar,  Double_t x ) const;
      Double_t Norm       ( TString var, Double_t x ) const;

      // member functions for the "evaluation" 
      // accessors
      Bool_t   IsOK     ( void  )  const { return fIsOK; }

      // write method-specific histograms to file
      void WriteEvaluationHistosToFile( TDirectory* targetDir = 0 );

      Types::EVariableTransform GetVariableTransform() const { return fVariableTransform; }
      void SetVariableTransform ( Types::EVariableTransform m ) { fVariableTransform = m; }
      
      Bool_t Verbose( void ) const { return fVerbose; }
      void   SetVerbose( Bool_t v = kTRUE ) { fVerbose = v; }

      DataSet& Data() const { return fData; }
      virtual Bool_t ReadEvent( TTree* tr, UInt_t ievt, Types::ESBType type = Types::kMaxSBType ) const { 
         if (type == Types::kMaxSBType) type = GetVariableTransformType();
         fVarTransform->ReadEvent(tr, ievt, type);
         return kTRUE;
      }

      virtual Bool_t   ReadTrainingEvent( UInt_t ievt, Types::ESBType type = Types::kMaxSBType ) const {
         return ReadEvent( Data().GetTrainingTree(), ievt, type );
      }

      virtual Bool_t ReadTestEvent( UInt_t ievt, Types::ESBType type = Types::kMaxSBType ) const {
         return ReadEvent( Data().GetTestTree(), ievt, type );
      }

      TMVA::Event& GetEvent() const { return GetVarTransform().GetEvent(); }
      Double_t GetEventVal( Int_t ivar ) const { return GetEvent().GetVal(ivar); }
      Double_t GetEventValNormalized(Int_t ivar) const;
      Double_t GetEventWeight() const { return GetEvent().GetWeight(); }

      virtual void DeclareOptions();
      virtual void ProcessOptions();

      // TestVar (the variable name used for the MVA)
      const TString& GetTestvarName() const { return fTestvar; }
      const TString  GetProbaName()   const { return fTestvar + "_Proba"; }

      // retrieve variable transformer
      VariableTransformBase& GetVarTransform() const { return *fVarTransform; }

      // sets the minimum requirement on the MVA output to declare an event signal-like
      Double_t GetSignalReferenceCut() const { return fSignalReferenceCut; }
 
   public:

      // static pointer to this object
      static MethodBase* GetThisBase( void ) { return fgThisBase; }        

   protected:

      // used in efficiency computation
      enum ECutOrientation { kNegative = -1, kPositive = +1 };
      ECutOrientation GetCutOrientation() const { return fCutOrientation; }

      // reset required for RootFinder
      void ResetThisBase( void ) { fgThisBase = this; }

      // sets the minimum requirement on the MVA output to declare an event signal-like
      void     SetSignalReferenceCut( Double_t cut ) { fSignalReferenceCut = cut; }

      // some basic statistical analysis
      void     Statistics( TMVA::Types::ETreeType treeType, const TString& theVarName,
                           Double_t&, Double_t&, Double_t&, 
                           Double_t&, Double_t&, Double_t&, Bool_t norm = kFALSE );
         
      Types::ESBType GetVariableTransformType() const { return fVariableTransformType; }
      void           SetVariableTransformType( Types::ESBType t ) { fVariableTransformType = t; }

      // the versions can be checked using
      // if(GetTrainingTMVAVersionCode()>TMVA_VERSION(3,7,2)) {...}
      // or
      // if(GetTrainingROOTVersionCode()>ROOT_VERSION(5,15,5)) {...}
      UInt_t GetTrainingTMVAVersionCode() const { return fTMVATrainingVersion; }
      UInt_t GetTrainingROOTVersionCode() const { return fROOTTrainingVersion; }
      TString GetTrainingTMVAVersionString() const;
      TString GetTrainingROOTVersionString() const;

   private:

      DataSet &      fData;            //! the data set

      Double_t       fSignalReferenceCut;     // minimum requirement on the MVA output to declare an event signal-like
      Types::ESBType fVariableTransformType;  // this is the event type (sig or bgd) assumed for variable transform

   protected:

      // protected accessors for derived classes
      TDirectory*      BaseDir() const;
      TDirectory*      LocalTDir() const { return Data().LocalRootDir(); }

      // TestVar (the variable name used for the MVA)
      void SetTestvarName( const TString & v="" ) { fTestvar = (v=="")?(fTestvarPrefix + GetMethodTitle()):v; }

      // MVA prefix (e.g., "TMVA_")
      const TString& GetTestvarPrefix() const { return fTestvarPrefix; }
      void SetTestvarPrefix( TString prefix ) { fTestvarPrefix = prefix; }

      // series of sanity checks on input tree (eg, do all the variables really 
      // exist in tree, etc)
      Bool_t CheckSanity( TTree* theTree = 0 );

      // if TRUE, write weights only to text files 
      Bool_t TxtWeightsOnly() const { return fTxtWeightsOnly; }       

      // direct accessors (should be made to functions ...)
      Ranking*         fRanking;        // ranking      
      vector<TString>* fInputVars;      // vector of input variables used in MVA

   private:

      TString     fJobName;             // name of job -> user defined, appears in weight files
      TString     fMethodName;          // name of the method (set in derived class)
      Types::EMVA fMethodType;          // type of method (set in derived class)      
      TString     fMethodTitle;         // user-defined title for method (used for weight-file names)
      TString     fTestvar;             // variable used in evaluation, etc (mostly the MVA)
      TString     fTestvarPrefix;       // 'MVA_' prefix of MVA variable
      UInt_t      fTMVATrainingVersion; // TMVA version used for training
      UInt_t      fROOTTrainingVersion; // ROOT version used for training
 
   private:

      void SetBaseDir( TDirectory* d ) { fBaseDir = d; }
      
      Int_t       fNvar;                // number of input variables
      TDirectory* fBaseDir;             // base director, needed to know where to jump back from localDir


      TString   fFileDir;               // unix sub-directory for weight files (default: "weights")
      TString   fWeightFile;            // weight file name

      VariableTransformBase* fVarTransform; // the variable transformer

   protected:

      Bool_t    fIsOK;                  // status of sanity checks
      TH1*      fHistS_plotbin;         // MVA plots used for graphics representation (signal)
      TH1*      fHistB_plotbin;         // MVA plots used for graphics representation (background)
      TH1*      fProbaS_plotbin;        // P(MVA) plots used for graphics representation (signal)
      TH1*      fProbaB_plotbin;        // P(MVA) plots used for graphics representation (background)
      TH1*      fHistS_highbin;         // MVA plots used for efficiency calculations (signal)    
      TH1*      fHistB_highbin;         // MVA plots used for efficiency calculations (background)
      TH1*      fEffS;                  // efficiency plot (signal)
      TH1*      fEffB;                  // efficiency plot (background)
      TH1*      fEffBvsS;               // background efficiency versus signal efficiency
      TH1*      fRejBvsS;               // background rejection (=1-eff.) versus signal efficiency
      TH1*      finvBeffvsSeff;         // inverse background eff (1/eff.) versus signal efficiency
      TH1*      fHistBhatS;             // working histograms needed for mu-transform (signal)
      TH1*      fHistBhatB;             // working histograms needed for mu-transform (background)
      TH1*      fHistMuS;               // mu-transform (signal)
      TH1*      fHistMuB;               // mu-transform (background)

      TH1*      fTrainEffS;             // Training efficiency plot (signal)
      TH1*      fTrainEffB;             // Training efficiency plot (background)
      TH1*      fTrainEffBvsS;          // Training background efficiency versus signal efficiency
      TH1*      fTrainRejBvsS;          // Training background rejection (=1-eff.) versus signal efficiency

      Int_t      fNbinsMVAPdf;          // number of bins used in histogram that creates PDF
      Int_t      fNsmoothMVAPdf;        // number of times a histogram is smoothed before creating the PDF
      TMVA::PDF* fMVAPdfS;              // signal MVA PDF
      TMVA::PDF* fMVAPdfB;              // background MVA PDF

      // mu-transform
      Double_t  fX;
      Double_t  fMode;

      TGraph*   fGraphS;                // graphs used for splines for efficiency (signal)
      TGraph*   fGraphB;                // graphs used for splines for efficiency (background)
      TGraph*   fGrapheffBvsS;          // graphs used for splines for signal eff. versus background eff.
      PDF*      fSplS;                  // PDFs of MVA distribution (signal)
      PDF*      fSplB;                  // PDFs of MVA distribution (background)
      TSpline*  fSpleffBvsS;            // splines for signal eff. versus background eff.


      TGraph*   fGraphTrainS;           // graphs used for splines for training efficiency (signal)
      TGraph*   fGraphTrainB;           // graphs used for splines for training efficiency (background)
      TGraph*   fGraphTrainEffBvsS;     // graphs used for splines for training signal eff. versus background eff.
      PDF*      fSplTrainS;             // PDFs of training MVA distribution (signal)
      PDF*      fSplTrainB;             // PDFs of training MVA distribution (background)
      TSpline*  fSplTrainEffBvsS;       // splines for training signal eff. versus background eff.

   private:

      // basic statistics quantities of MVA
      Double_t  fMeanS;                 // mean (signal)
      Double_t  fMeanB;                 // mean (background)
      Double_t  fRmsS;                  // RMS (signal)
      Double_t  fRmsB;                  // RMS (background)
      Double_t  fXmin;                  // minimum (signal and background)
      Double_t  fXmax;                  // maximum (signal and background)

      Bool_t    fUseDecorr;                          // kept for backward compatibility
      Types::EVariableTransform fVariableTransform;  // Decorrelation, PCA, etc.
      TString fVarTransformString;                   // labels variable transform method
      TString fVariableTransformTypeString;          // labels variable transform type

      Bool_t    fVerbose;               // verbose flag
      TString   fVerbosityLevelString;  // verbosity level (user input string)
      EMsgType  fVerbosityLevel;        // verbosity level
      Bool_t    fHelp;                  // help flag
      Bool_t    fIsMVAPdfs;             // create MVA Pdfs
      Bool_t    fTxtWeightsOnly;        // if TRUE, write weights only to text files 

   protected:

      Int_t     fNbins;                 // number of bins in representative histograms
      Int_t     fNbinsH;                // number of bins in evaluation histograms

      // orientation of cut: depends on signal and background mean values
      ECutOrientation fCutOrientation;  // +1 if Sig>Bkg, -1 otherwise

      // for root finder
      TSpline1*  fSplRefS;              // helper splines for RootFinder (signal)
      TSpline1*  fSplRefB;              // helper splines for RootFinder (background)

      TSpline1*  fSplTrainRefS;         // helper splines for RootFinder (signal)
      TSpline1*  fSplTrainRefB;         // helper splines for RootFinder (background)

   public:

      // for root finder 
      static Double_t IGetEffForRoot( Double_t );  // interface
      Double_t GetEffForRoot( Double_t );          // implementation

   private:

      // this carrier
      static MethodBase* fgThisBase;    // this pointer

      // Init used in the various constructors
      void Init( void );
      bool GetLine(std::istream& fin, char * buf );
            
   protected:

      // the mutable declaration is needed to use the logger in const methods
      mutable MsgLogger fLogger; // message logger

   public:
      ClassDef(MethodBase,0)  // Virtual base class for all TMVA method
   };
} // namespace TMVA


#endif

