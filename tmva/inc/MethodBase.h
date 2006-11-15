// @(#)root/tmva $Id: MethodBase.h,v 1.55 2006/11/14 23:02:57 stelzer Exp $   
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
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
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
#include <sstream>
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
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

class TTree;
class TDirectory;

namespace TMVA {

   class Ranking;

   class MethodBase : public IMethod {
      
   public:

      enum WeightFileType { kROOT=0, kTEXT };
      
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

      void TrainMethod();
      void WriteStateToFile() const;
      void WriteStateToStream( std::ostream& o ) const;
      void ReadStateFromFile();
      virtual void WriteMonitoringHistosToFile( void ) const;
      virtual void WriteWeightsToStream ( std::ostream& o ) const = 0;
      virtual void ReadWeightsFromStream( std::istream& i ) = 0;
      virtual void ReadStateFromStream  ( std::istream& i );

      virtual Bool_t IsSignalLike() { return GetMvaValue() > GetSignalReferenceCut() ? kTRUE : kFALSE; }     

      // evaluate method (resulting discriminating variable) or input varible
      virtual void TestInit( TTree* theTestTree = 0 );

      // individual initialistion for testing of each method
      // overload this one for individual initialisation of the testing, 
      // it is then called automatically within the global "TestInit" 
      
      // the new way to get the MVA value
      virtual Double_t GetMvaValue() = 0;

      // test the method
      virtual void Test( TTree* theTestTree = 0 );

      // accessors
      const TString&   GetJobName    ( void ) const { return fJobName; }
      const TString&   GetMethodName ( void ) const { return fMethodName; }
      const char*      GetName       ( void ) const { return GetMethodName().Data(); }
      const TString&   GetMethodTitle( void ) const { return fMethodTitle; }
      const Types::MVA GetMethodType ( void ) const { return fMethodType; }

      void    SetJobName    ( TString jobName )       { fJobName     = jobName; }
      void    SetMethodName ( TString methodName )    { fMethodName  = methodName; }
      void    SetMethodTitle( TString methodTitle )   { fMethodTitle = methodTitle; }
      void    SetMethodType ( Types::MVA methodType ) { fMethodType  = methodType; }

      TString GetOptions    ( void ) const { return fOptions; }
      
      TString GetWeightFileExtension( void ) const            { return fFileExtension; }
      void    SetWeightFileExtension( TString fileExtension ) { fFileExtension = fileExtension; } 
      void    SetWeightFileType( WeightFileType w ) { fWeightFileType = w; }
      WeightFileType GetWeightFileType() const { return fWeightFileType; }


      TString GetWeightFileDir( void ) const { return fFileDir; }
      void    SetWeightFileDir( TString fileDir );

      const TString& GetInputVar( int i ) const { return Data().GetInternalVarName(i); }
      const TString& GetInputExp( int i ) const { return Data().GetExpression(i); }

      void     SetWeightFileName( TString );
      TString  GetWeightFileName() const;

      Bool_t   HasTrainingTree() const { return Data().GetTrainingTree() != 0; }
      TTree*   GetTrainingTree() const { 
         if (GetPreprocessingMethod()!=Types::kNone) {
            fLogger << kFATAL << "Trying to access correlated Training tree in method " 
                    << GetMethodName() << endl;
         }
         return Data().GetTrainingTree();
      }
      TTree*   GetTestTree() const {
         if (GetPreprocessingMethod()!=Types::kNone) {
            fLogger << kFATAL << "Trying to access correlated Training tree in method " 
                    << GetMethodName() << endl;
         }
         return Data().GetTestTree();
      }

      Int_t    GetNvar( void ) const { return fNvar; }
      void     SetNvar( Int_t n) { fNvar = n; }

      // variables (and private menber functions) for the Evaluation:
      // get the effiency. It fills a histogram for efficiency/vs/bkg
      // and returns the one value fo the efficiency demanded for 
      // in the TString argument. (Watch the string format)
      virtual Double_t  GetEfficiency   ( TString , TTree*);
      virtual Double_t  GetTrainingEfficiency   ( TString );
      virtual Double_t  GetSignificance ( void );
      virtual Double_t  GetOptimalSignificance( Double_t SignalEvents, Double_t BackgroundEvents, 
                                                Double_t& optimal_significance_value  ) const;
      virtual Double_t  GetSeparation   ( void );
      virtual Double_t  GetmuTransform  ( TTree* );

      // normalisation accessors
      Double_t GetXmin( Int_t ivar, Types::PreprocessingMethod corr = Types::kNone )         const { return fXminNorm[(Int_t) corr][ivar]; }
      Double_t GetXmax( Int_t ivar, Types::PreprocessingMethod corr = Types::kNone )         const { return fXmaxNorm[(Int_t) corr][ivar]; }
      Double_t GetXmin( const TString& var, Types::PreprocessingMethod corr = Types::kNone ) const { return GetXmin(Data().FindVar(var), corr); }
      Double_t GetXmax( const TString& var, Types::PreprocessingMethod corr = Types::kNone ) const { return GetXmax(Data().FindVar(var), corr); }
      void     SetXmin( Int_t ivar, Double_t x, Types::PreprocessingMethod corr = Types::kNone )         { fXminNorm[(Int_t) corr][ivar] = x; }
      void     SetXmax( Int_t ivar, Double_t x, Types::PreprocessingMethod corr = Types::kNone )         { fXmaxNorm[(Int_t) corr][ivar] = x; }
      void     SetXmin( const TString& var, Double_t x, Types::PreprocessingMethod corr = Types::kNone ) { SetXmin(Data().FindVar(var), x, corr); }
      void     SetXmax( const TString& var, Double_t x, Types::PreprocessingMethod corr = Types::kNone ) { SetXmax(Data().FindVar(var), x, corr); }

      // main normalization method is in Tools
      Double_t Norm       ( Int_t ivar,  Double_t x ) const;
      Double_t Norm       ( TString var, Double_t x ) const;

      // member functions for the "evaluation" 
      // accessors
      Bool_t   IsOK     ( void  )  const { return fIsOK; }

      // write method-specific histograms to file
      void WriteEvaluationHistosToFile( TDirectory* targetDir );

      Types::PreprocessingMethod GetPreprocessingMethod() const { return fPreprocessingMethod; }
      void SetPreprocessingMethod ( Types::PreprocessingMethod m ) { fPreprocessingMethod = m; }
      
      Bool_t Verbose( void ) const { return fVerbose; }
      void   SetVerbose( Bool_t v = kTRUE ) { fVerbose = v; }

      DataSet& Data() const { return fData; }
      Bool_t   ReadTrainingEvent( UInt_t ievt, Types::SBType type = Types::kMaxSBType ) { 
         return Data().ReadTrainingEvent( ievt, GetPreprocessingMethod(),
                                          (type == Types::kMaxSBType) ? GetPreprocessingType() : type ); 
      }
      virtual Bool_t   ReadTestEvent( UInt_t ievt, Types::SBType type = Types::kMaxSBType ) { 
         return Data().ReadTestEvent( ievt, GetPreprocessingMethod(),
                                      (type == Types::kMaxSBType) ? GetPreprocessingType() : type ); 
      }

      Double_t GetEventVal( Int_t ivar ) const { return Data().Event().GetVal(ivar); }
      Double_t GetEventValNormalized(Int_t ivar) const;
      Double_t GetEventWeight() const { return Data().Event().GetWeight(); }

      virtual void DeclareOptions();
      virtual void ProcessOptions();

   public:

      // static pointer to this object
      static MethodBase* GetThisBase( void ) { return fgThisBase; }        

   protected:

      // used in efficiency computation
      enum CutOrientation { kNegative = -1, kPositive = +1 };
      CutOrientation GetCutOrientation() const { return fCutOrientation; }

      // reset required for RootFinder
      void ResetThisBase( void ) { fgThisBase = this; }

      // sets the minimum requirement on the MVA output to declare an event 
      // signal-like
      Double_t GetSignalReferenceCut() const { return fSignalReferenceCut; }
      void     SetSignalReferenceCut( Double_t cut ) { fSignalReferenceCut = cut; }

      // some basic statistical analysis
      void     Statistics( TMVA::Types::TreeType treeType, const TString& theVarName,
                           Double_t&, Double_t&, Double_t&, 
                           Double_t&, Double_t&, Double_t&, Bool_t norm = kFALSE );
         
      Types::SBType GetPreprocessingType() const { return fPreprocessingType; }
      void          SetPreprocessingType( Types::SBType t ) { fPreprocessingType = t; }

   private:

      Double_t      fSignalReferenceCut; // minimum requirement on the MVA output to declare an event signal-like
      Types::SBType fPreprocessingType;  // this is the event type (sig or bgd) assumed for preprocessing

   private:

      DataSet&   fData;            //! the data set
      Double_t*  fXminNorm[3];     //! minimum value for correlated/decorrelated/PCA variable
      Double_t*  fXmaxNorm[3];     //! maximum value for correlated/decorrelated/PCA variable

   protected:

      // protected accessors for derived classes
      TDirectory*      BaseDir() const;
      TDirectory*      LocalTDir() const { return Data().LocalRootDir(); }

      // TestVar (the variable name used for the MVA)
      const TString& GetTestvarName() const { return fTestvar; }
      void SetTestvarName( void )      { fTestvar = fTestvarPrefix + GetMethodTitle(); }
      void SetTestvarName( TString v ) { fTestvar = v; }

      // MVA prefix (e.g., "TMVA_")
      const TString& GetTestvarPrefix() const { return fTestvarPrefix; }
      void SetTestvarPrefix( TString prefix ) { fTestvarPrefix = prefix; }

      // series of sanity checks on input tree (eg, do all the variables really 
      // exist in tree, etc)
      Bool_t CheckSanity( TTree* theTree = 0 );
      void   EnableLooseOptions( Bool_t b = kTRUE ) { fLooseOptionCheckingEnabled = b; }

      // direct accessors (should be made to functions ...)
      Ranking*         fRanking;        // ranking      
      vector<TString>* fInputVars;      // vector of input variables used in MVA
 
   private:

      TString          fJobName;        // name of job -> user defined, appears in weight files
      TString          fMethodName;     // name of the method (set in derived class)
      Types::MVA       fMethodType;     // type of method (set in derived class)      
      TString          fMethodTitle;    // user-defined title for method (used for weight-file names)
      TString          fTestvar;        // variable used in evaluation, etc (mostly the MVA)
      TString          fTestvarPrefix;  // 'MVA_' prefix of MVA variable
      TString          fOptions;        // options string
 
   private:

      void SetBaseDir( TDirectory* d ) { fBaseDir = d; }
      
      Int_t            fNvar;           // number of input variables
      TDirectory*      fBaseDir;        // base director, needed to know where to jump back from localDir


      TString     fFileExtension;       // extension used in weight files (default: ".weights")
      TString     fFileDir;             // unix sub-directory for weight files (default: "weights")
      TString     fWeightFile;          // weight file name

      WeightFileType fWeightFileType;   // The type of weight file  {kROOT,kTEXT}

   protected:

      Bool_t    fIsOK;                  // status of sanity checks
      TH1*      fHistS_plotbin;         // MVA plots used for graphics representation (signal)
      TH1*      fHistB_plotbin;         // MVA plots used for graphics representation (background)
      TH1*      fHistS_highbin;         // MVA plots used for efficiency calculations (signal)    
      TH1*      fHistB_highbin;              // MVA plots used for efficiency calculations (background)
      TH1*      fEffS;                  // efficiency plot (signal)
      TH1*      fEffB;                  // efficiency plot (background)
      TH1*      fEffBvsS;               // background efficiency versus signal efficiency
      TH1*      fRejBvsS;               // background rejection (=1-eff.) versus signal efficiency
      TH1*      fHistBhatS;             // working histograms needed for mu-transform (signal)
      TH1*      fHistBhatB;             // working histograms needed for mu-transform (background)
      TH1*      fHistMuS;               // mu-transform (signal)
      TH1*      fHistMuB;               // mu-transform (background)

      TH1*      fTrainEffS;             // Training efficiency plot (signal)
      TH1*      fTrainEffB;             // Training efficiency plot (background)
      TH1*      fTrainEffBvsS;          // Training background efficiency versus signal efficiency
      TH1*      fTrainRejBvsS;          // Training background rejection (=1-eff.) versus signal efficiency

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

      Bool_t    fUseDecorr;             // Use decorrelated Variables (kept for backward compatibility)
      Types::PreprocessingMethod fPreprocessingMethod;  // Decorrelation, PCA, etc.
      TString fPreprocessingString;     // labels preprocessing method
      TString fPreprocessingTypeString; // labels preprocessing type

      // verbose flag (debug messages) 
      Bool_t    fVerbose;               // verbose flag
      Bool_t    fHelp;                  // help flag
      Bool_t    LooseOptionCheckingEnabled() const { return fLooseOptionCheckingEnabled; }
      Bool_t    fLooseOptionCheckingEnabled; // checker for option string

   protected:

      Int_t     fNbins;                 // number of bins in representative histograms
      Int_t     fNbinsH;                // number of bins in evaluation histograms

      // orientation of cut: depends on signal and background mean values
      CutOrientation  fCutOrientation;  // +1 if Sig>Bkg, -1 otherwise

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
            
   protected:

      // classes and method related to easy and flexible option parsing
      OptionBase* fLastDeclaredOption;  // last declared option
      TList       fListOfOptions;       // option list
      const TList& ListOfOptions() const { return fListOfOptions; }

      template <class T>
      void AssignOpt( const TString& name, T& valAssign ) const;

   protected:

      // classes and method related to easy and flexible option parsing
      template<class T> 
      OptionBase* DeclareOption( const TString& name, const TString& desc = "" );

      template<class T> 
      OptionBase* DeclareOptionRef( T& ref, const TString& name, const TString& desc = "" );

      template<class T>
      void AddPreDefVal(const T&);
      
      void ParseOptions( Bool_t verbose = kTRUE);

      void PrintOptions() const;
      void WriteOptionsToStream(ostream& o) const;
      void ReadOptionsFromStream(istream& istr);

      // the mutable declaration is needed to use the logger in const methods
      mutable MsgLogger fLogger; // message logger

      ClassDef(MethodBase,0)  // Virtual base class for all TMVA method
         ;
   };
} // namespace TMVA

// Template Declarations go here

//______________________________________________________________________
template <class T>
TMVA::OptionBase* TMVA::MethodBase::DeclareOption( const TString& name, const TString& desc) 
{
   // declare an option
   OptionBase* o = new Option<T>(name,desc);
   fListOfOptions.Add(o);
   fLastDeclaredOption = o;
   return o;
}

//______________________________________________________________________
template <class T>
TMVA::OptionBase* TMVA::MethodBase::DeclareOptionRef( T& ref, const TString& name, const TString& desc) 
{
   // set the reference for an option
   OptionBase* o = new Option<T>(ref, name, desc);
   fListOfOptions.Add(o);
   fLastDeclaredOption = o;
   return o;
}

//______________________________________________________________________
template<class T>
void TMVA::MethodBase::AddPreDefVal(const T& val) 
{
   // add predefined option value
   Option<T>* oc = dynamic_cast<Option<T>*>(fLastDeclaredOption);
   if(oc!=0) oc->AddPreDefVal(val);
}

//______________________________________________________________________
template <class T>
void TMVA::MethodBase::AssignOpt(const TString& name, T& valAssign) const 
{
   // assign an option
   TObject* opt = fListOfOptions.FindObject(name);
   if (opt!=0) valAssign = ((Option<T>*)opt)->Value();
   else 
      fLogger << kFATAL << "Option \"" << name 
              << "\" not declared, please check the syntax of your option string" << endl;
}

#endif

