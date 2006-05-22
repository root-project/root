// @(#)root/tmva $Id: MethodBase.h,v 1.4 2006/05/22 08:04:39 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodBase                                                            *
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
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
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

#ifndef ROOT_TMVA_PDF
#include "TMVA/PDF.h"
#endif
#ifndef ROOT_TMVA_TSpline1
#include "TMVA/TSpline1.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif

class TTree;
class TDirectory;

namespace TMVA {

  class MethodBase : public TObject {

  public:

    // default constructur
    MethodBase( TString jobName,
		vector<TString>* theVariables, 
		TTree* theTree = 0, 
		TString theOption = "", 
		TDirectory* theBaseDir = 0 );

    // constructor used for Testing + Application of the MVA, only (no training), 
    // using given weight file
    MethodBase( vector<TString> *theVariables, 
		TString weightFile, 
		TDirectory* theBaseDir = NULL  );

    // default destructur
    virtual ~MethodBase( void );

    // training method
    virtual void Train( void ) = 0;

    // write weights to file
    virtual void WriteWeightsToFile( void ) = 0;
  
    // read weights from file
    virtual void ReadWeightsFromFile( void ) = 0;

    // prepare tree branch with the method's discriminating variable
    virtual void PrepareEvaluationTree( TTree* theTestTree );

    // calculate the MVA value
    virtual Double_t GetMvaValue( Event *e ) = 0;

    // evaluate method (resulting discriminating variable) or input varible
    virtual void TestInit(TTree* theTestTree);

    // indivudual initialistion for testing of each method
    // overload this one for individual initialisation of the testing, 
    // it is then called automatically within the global "TestInit" 
    virtual void TestInitLocal(TTree *  /*testTree*/) {
      return ;
    }

    // test the method
    virtual void Test( TTree * theTestTree );

    // write method specific histos to target file
    virtual void WriteHistosToFile( void ) = 0;

    // accessors
    TString GetMethodName( void ) const         { return fMethodName; }
    Types::MVA GetMethod    ( void ) const { return fMethod;     }
    TString GetOptions   ( void ) const         { return fOptions;    }
    void    SetMethodName( TString methodName ) { fMethodName = methodName; }
    void    AppendToMethodName( TString methodNameSuffix );

    TString GetJobName   ( void ) const         { return fJobName; }
    void    SetJobName   ( TString jobName )    { fJobName = jobName; }

    TString GetWeightFileExtension( void ) const            { return fFileExtension; }
    void    SetWeightFileExtension( TString fileExtension ) { fFileExtension = fileExtension; } 

    TString GetWeightFileDir( void ) const      { return fFileDir; }
    void    SetWeightFileDir( TString fileDir );

    vector<TString>*  GetInputVars( void ) const { return fInputVars; }
    void              SetInputVars( vector<TString>* theInputVars ) { fInputVars = theInputVars; }

    void     SetWeightFileName( void );
    void     SetWeightFileName( TString );
    TString  GetWeightFileName( void );
    TTree*   GetTrainingTree  ( void ) const { return fTrainingTree; }

    Int_t    GetNvar          ( void ) const { return fNvar; }

    // variables (and private menber functions) for the Evaluation:
    // get the effiency. It fills a histogram for efficiency/vs/bkg
    // and returns the one value fo the efficiency demanded for 
    // in the TString argument. (Watch the string format)
    virtual Double_t  GetEfficiency   ( TString , TTree *);
    virtual Double_t  GetSignificance ( void );
    virtual Double_t  GetOptimalSignificance( Double_t SignalEvents, Double_t BackgroundEvents, 
					      Double_t & optimal_significance_value  ) const;
    virtual Double_t  GetSeparation   ( void );
    virtual Double_t  GetmuTransform  ( TTree * );

    // normalisation init
    virtual void InitNorm( TTree* theTree );

    // normalisation accessors
    Double_t GetXminNorm( Int_t ivar  ) const { return (*fXminNorm)[ivar]; }
    Double_t GetXmaxNorm( Int_t ivar  ) const { return (*fXmaxNorm)[ivar]; }
    Double_t GetXminNorm( TString var ) const;
    Double_t GetXmaxNorm( TString var ) const;
    void     SetXminNorm( Int_t ivar,  Double_t x ) { (*fXminNorm)[ivar] = x; }
    void     SetXmaxNorm( Int_t ivar,  Double_t x ) { (*fXmaxNorm)[ivar] = x; }
    void     SetXminNorm( TString var, Double_t x );
    void     SetXmaxNorm( TString var, Double_t x );
    void     UpdateNorm ( Int_t ivar,  Double_t x );

    // main normalization method is in Tools
    Double_t Norm       ( Int_t ivar,  Double_t x ) const;
    Double_t Norm       ( TString var, Double_t x ) const;
  
    // member functions for the "evaluation" 
    // accessors
    Bool_t   IsOK     ( void  )  const { return fIsOK; }

    void WriteHistosToFile( TDirectory* targetDir );

    enum CutOrientation { kNegative = -1, kPositive = +1 };
    CutOrientation GetCutOrientation() { return fCutOrientation; }

    enum Type { kSignal = 1, kBackground = 0 };

    Bool_t Verbose( void ) const { return fVerbose; }
    void SetVerbose( Bool_t v = kTRUE ) { fVerbose = v; }

  public:

    // static pointer to this object
    static MethodBase* GetThisBase( void ) { return fgThisBase; }  

  protected:

    // reset required for RootFinder
    void ResetThisBase( void ) { fgThisBase = this; }

  protected:

    TString          fJobName;        // name of job -> user defined, appears in weight files
    TString          fMethodName;     // name of the method (set in derived class)
    Types::MVA       fMethod;         // type of method (set in derived class)
    TTree*           fTrainingTree;   // training tree
    TString          fTestvar;        // variable used in evauation, etc (mostly the MVA)
    TString          fTestvarPrefix;  // 'MVA_' prefix of MVA variable
    vector<TString>* fInputVars;      // vector of input variables used in MVA
    TString          fOptions;        // options string
    TDirectory*      fBaseDir;        // base director, needed to know where to jump back from localDir
    TDirectory*      fLocalTDir;      // local directory, used to save monitoring histograms

    // series of sanity checks on input tree (eg, do all the variables really 
    // exist in tree, etc)
    Bool_t CheckSanity( TTree* theTree = 0 );

    Int_t       fNvar;                // number of input variables

  private:

    TString     fFileExtension;       // extension used in weight files (default: ".weights")
    TString     fFileDir;             // unix sub-directory for weight files (default: "weights")
    TString     fWeightFile;          // weight file name

  protected:

    Bool_t    fIsOK;                  // status of sanity checks
    TH1*      fHistS_plotbin;         // MVA plots used for graphics representation (signal)
    TH1*      fHistB_plotbin;         // MVA plots used for graphics representation (background)
    TH1*      fHistS_highbin;         // MVA plots used for efficiency calculations (signal)    
    TH1*      fHistB_highbin;	      // MVA plots used for efficiency calculations (background)
    TH1*      fEffS;                  // efficiency plot (signal)
    TH1*      fEffB;                  // efficiency plot (background)
    TH1*      fEffBvsS;               // background efficiency versus signal efficiency
    TH1*      fRejBvsS;               // background rejection (=1-eff.) versus signal efficiency
    TH1*      fHistBhatS;             // working histograms needed for mu-transform (signal)
    TH1*      fHistBhatB;             // working histograms needed for mu-transform (background)
    TH1*      fHistMuS;               // mu-transform (signal)
    TH1*      fHistMuB;               // mu-transform (background)

    // mu-transform
    Double_t  fX;
    Double_t  fMode;

    TGraph*   fGraphS;                // graphs used for splines for efficiency (signal)
    TGraph*   fGraphB;                // graphs used for splines for efficiency (background)
    TGraph*   fGrapheffBvsS;          // graphs used for splines for signal eff. versus background eff.
    PDF*      fSplS;                  // PDFs of MVA distribution (signal)
    PDF*      fSplB;                  // PDFs of MVA distribution (background)
    TSpline*  fSpleffBvsS;            // splines for signal eff. versus background eff.

  private:

    // basic statistics quantities of MVA
    Double_t  fMeanS;                 // mean (signal)
    Double_t  fMeanB;                 // mean (background)
    Double_t  fRmsS;                  // RMS (signal)
    Double_t  fRmsB;                  // RMS (background)
    Double_t  fXmin;                  // minimum (signal and background)
    Double_t  fXmax;                  // maximum (signal and background)

    // verbose flag (debug messages) 
    Bool_t    fVerbose;

  protected:

    Int_t     fNbins;                 // number of bins in representative histograms
    Int_t     fNbinsH;                // number of bins in evaluation histograms

    // orientation of cut: depends on signal and background mean values
    CutOrientation  fCutOrientation;  // +1 if Sig>Bkg, -1 otherwise

    // for root finder
    TSpline1*  fSplRefS;              // helper splines for RootFinder (signal)
    TSpline1*  fSplRefB;              // helper splines for RootFinder (background)

  public:

    // for root finder 
    static Double_t IGetEffForRoot( Double_t );  // interface
    Double_t GetEffForRoot( Double_t );          // implementation

  private:

    // normalization
    vector<Double_t>* fXminNorm;      // minimum of input variables
    vector<Double_t>* fXmaxNorm;      // maximum of input variables

    // this carrier
    static MethodBase* fgThisBase;

    // Init used in the various constructors
    void Init( void );

    ClassDef(MethodBase,0)  //Virtual base class for all TMVA method
  };
} // namespace TMVA

#endif

