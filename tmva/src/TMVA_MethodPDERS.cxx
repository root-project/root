// @(#)root/tmva $Id: TMVA_MethodPDERS.cxx,v 1.5 2006/05/09 08:37:06 brun Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodPDERS                                                      *
 *                                                                                *
 * Description:                                                                   *
 *      Multidimensional Likelihood using the "Probability density estimator      *
 *      range search" (PDERS) method suggested in                                 *
 *      T. Carli and B. Koblitz, NIM A 501, 576 (2003)                            *
 *                                                                                *
 *      Implementation (see header file for description)                          *
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

//_______________________________________________________________________
//
// Multidimensional Likelihood using the "Probability density
// estimator range search" (PDERS) method
//
//_______________________________________________________________________

#include "TMVA_MethodPDERS.h"
#include "TMVA_Tools.h"
#include "TMVA_RootFinder.h"
#include "TFile.h"
#include "TObjString.h"
#include "TMath.h"
#include <stdexcept>

#define DEBUG_TMVA_MethodPDERS             kFALSE
#define TMVA_MethodPDERS_UseFindRoot       kTRUE
#define TMVA_MethodPDERS_UseKernelEstimate kFALSE

using std::vector;

ClassImp(TMVA_MethodPDERS)

//_______________________________________________________________________
TMVA_MethodPDERS::TMVA_MethodPDERS( TString jobName, vector<TString>* theVariables,
				    TTree* theTree, TString theOption, TDirectory* theTargetDir )
  : TMVA_MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{
  InitPDERS();

  // interpret options string
  // default settings (should be defined in fOptions string)
  TList*  list  = TMVA_Tools::ParseFormatLine( fOptions );

  // format and syntax of option string: "VolumeRangeMode:options"
  //
  // where:
  //  VolumeRangeMode - all methods defined in private enum "VolumeRangeMode"
  //  options         - deltaFrac in case of VolumeRangeMode=MinMax/RMS
  //                  - nEventsMin/Max, maxVIterations, scale for VolumeRangeMode=Adaptive

  if (list->GetSize() > 0) {
    TString s = ((TObjString*)list->At(0))->GetString();
    s.ToLower();
    if       (s.Contains("minmax")  ) fVRangeMode = TMVA_MethodPDERS::kMinMax;
    else  if (s.Contains("rms")     ) fVRangeMode = TMVA_MethodPDERS::kRMS;
    else  if (s.Contains("adaptive")) fVRangeMode = TMVA_MethodPDERS::kAdaptive;
    else {
      cout  << "--- " << GetName() << ": Fatal error unknown vRangeType type: "
	    << s << " in first option" << endl;
      throw std::invalid_argument( "Abort" );
    }
  }
  if (list->GetSize() > 1 && (fVRangeMode == kMinMax || fVRangeMode == kRMS)) {
    TString s = ((TObjString*)list->At(0))->GetString();
    fDeltaFrac = atof( ((TObjString*)list->At(1))->GetString() );
  }
  else if (fVRangeMode == kAdaptive) {
    if (list->GetSize() > 1) fNEventsMin     = atoi( ((TObjString*)list->At(1))->GetString() );
    if (list->GetSize() > 2) fNEventsMax     = atoi( ((TObjString*)list->At(2))->GetString() );
    if (list->GetSize() > 3) fMaxVIterations = atoi( ((TObjString*)list->At(3))->GetString() );
    if (list->GetSize() > 4) fInitialScale   = atof( ((TObjString*)list->At(4))->GetString() );
  }

  if (Verbose()) {
    cout << "--- " << GetName()
	 << " <verbose>: interpreted option string: vRangeMethod: '"
	 << (const char*)((fVRangeMode == kMinMax) ? "MinMax" :
			  (fVRangeMode == kRMS   ) ? "RMS" : "Adaptive")
	 << "'" << endl;
    if (fVRangeMode == kMinMax || fVRangeMode == kRMS)
      cout << "--- " << GetName() << ": deltaFrac: " << fDeltaFrac << endl;
    else
      cout << "--- " << GetName()
	   << " <verbose>: nEventsMin/Max, maxVIterations, initialScale: "
	   << fNEventsMin << "  " << fNEventsMax
	   << "  " << fMaxVIterations << "  " << fInitialScale << endl;
  }
}

//_______________________________________________________________________
TMVA_MethodPDERS::TMVA_MethodPDERS( vector<TString> *theVariables,
				    TString theWeightFile,
				    TDirectory* theTargetDir )
  : TMVA_MethodBase( theVariables, theWeightFile, theTargetDir )
{
  InitPDERS();
}

void TMVA_MethodPDERS::InitPDERS( void )
{
  fMethodName  = "PDERS";
  fMethod      = TMVA_Types::PDERS;
  fTestvar     = fTestvarPrefix+GetMethodName();
  fFin         = NULL;
  fBinaryTreeS = fBinaryTreeB = NULL;

  fgThisPDERS  = this;

  // default options
  fDeltaFrac      = 3.0;
  fVRangeMode     = kAdaptive;

  // special options for Adaptive mode
  fNEventsMin     = 100;
  fNEventsMax     = 200;
  fMaxVIterations = 50;
  fInitialScale   = 0.99;

  fInitializedVolumeEle = kFALSE;
}

//_______________________________________________________________________
TMVA_MethodPDERS::~TMVA_MethodPDERS( void )
{
  if (NULL != fBinaryTreeS) delete fBinaryTreeS;
  if (NULL != fBinaryTreeB) delete fBinaryTreeB;
  if (NULL != fFin) { fFin->Close(); fFin->Delete(); }
}

//_______________________________________________________________________
void TMVA_MethodPDERS::Train( void )
{
  //--------------------------------------------------------------
  // this is a dummy training: the preparation work to do is the construction
  // of the binary tree as a pointer chain. It is easier to directly save the
  // trainingTree in the weight file, and to rebuild the binary tree in the
  // test phase from scratch

  // default sanity checks
  if (!CheckSanity()) {
    cout << "--- " << GetName() << ": Error: sanity check failed" << endl;
    throw std::invalid_argument( "Abort" );
  }

  // write weights to file
  WriteWeightsToFile();
}

//_______________________________________________________________________
Double_t TMVA_MethodPDERS::GetMvaValue( TMVA_Event *e )
{
  // init the size of a volume element using a defined fraction of the
  // volume containing the entire events
  if (fInitializedVolumeEle == kFALSE) {

    fInitializedVolumeEle = kTRUE;
    SetVolumeElement();

    // create binary trees (global member variables) for signal and background
    Int_t nS = 0, nB = 0;
    fBinaryTreeS = new TMVA_BinarySearchTree();
    fBinaryTreeS->Fill( fTrainingTree, fInputVars, nS, 1 );
    fBinaryTreeB = new TMVA_BinarySearchTree();
    fBinaryTreeB->Fill( fTrainingTree, fInputVars, nB, 0 );

    // sanity check
    if (NULL == fBinaryTreeS || NULL == fBinaryTreeB) {
      cout << "--- " << GetName() << ": Error in ::Train: Create(BinaryTree) returned zero "
	   << "binaryTree pointer(s): " << fBinaryTreeS << "  " << fBinaryTreeB << endl;
      throw std::invalid_argument( "Abort" );
    }

    // these are the signal and background scales for the weights
    fScaleS = 1.0/Float_t(nS);
    fScaleB = 1.0/Float_t(nB);
    if (Verbose()) cout << "--- " << GetName() << " <verbose>: signal and background scales: "
			<< fScaleS << " " << fScaleB << endl;
  }

  return this->RScalc( e );
}

//_______________________________________________________________________
void TMVA_MethodPDERS::SetVolumeElement( void )
{
  // init relative scales
  fDelta = (fNvar > 0) ? new vector<Float_t>( fNvar ) : 0;
  fShift = (fNvar > 0) ? new vector<Float_t>( fNvar ) : 0;
  if (fDelta != 0) {
    for (Int_t ivar=0; ivar<fNvar; ivar++) {
      switch (fVRangeMode) {

      case kRMS:
      case kAdaptive:
	Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;
	TMVA_Tools::ComputeStat( fTrainingTree, (*fInputVars)[ivar],
				 meanS, meanB, rmsS, rmsB, xmin, xmax );
	(*fDelta)[ivar] = (rmsS + rmsB)*0.5*fDeltaFrac;
	if (Verbose())
	  cout << "--- " << GetName() << " <verbose>: delta of var[" << (*fInputVars)[ivar]
	       << "\t]: " << (rmsS + rmsB)*0.5
	       << "\t  |  comp with d|norm|: " << (GetXmaxNorm( ivar ) - GetXminNorm( ivar ))
	       << endl;
	break;

      case kMinMax:
	(*fDelta)[ivar] = (GetXmaxNorm( ivar ) - GetXminNorm( ivar ))*fDeltaFrac;
	break;

      default:
	cout << "--- " << GetName()
	     << ": Error in ::SetVolumeElement: unknown range-set mode: "
	     << fVRangeMode << endl;
	throw std::invalid_argument( "Abort" );
      }

      (*fShift)[ivar] = 0.5; // volume is centered around test value
    }
  }
  else {
    cout << "--- " << GetName() << ": Error: fNvar <= 0: " << fNvar << endl;
    throw std::invalid_argument("");
  }
}

TMVA_MethodPDERS* TMVA_MethodPDERS::fgThisPDERS = NULL;

//_______________________________________________________________________
Double_t TMVA_MethodPDERS::IGetVolumeContentForRoot( Double_t scale )
{
  return TMVA_MethodPDERS::ThisPDERS()->GetVolumeContentForRoot( scale );
}

//_______________________________________________________________________
Double_t TMVA_MethodPDERS::GetVolumeContentForRoot( Double_t scale )
{
  // search for events in rescaled volume
  // retrieve the class object

  TMVA_Volume v( *fHelpVolume );
  v.ScaleInterval( scale );
  Double_t cS = GetBinaryTreeSig()->SearchVolume( &v );
  Double_t cB = GetBinaryTreeBkg()->SearchVolume( &v );
  v.Delete();

  return cS + cB;
}

//_______________________________________________________________________
Float_t TMVA_MethodPDERS::RScalc( TMVA_Event *e )
{
  vector<Double_t> *lb = new vector<Double_t>( fNvar );
  for (Int_t ivar=0; ivar<fNvar; ivar++)
    (*lb)[ivar] = e->GetData(ivar);

  vector<Double_t> *ub = new vector<Double_t>( *lb );
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    (*lb)[ivar] -= (*fDelta)[ivar]*(1.0 - (*fShift)[ivar]);
    (*ub)[ivar] += (*fDelta)[ivar]*(*fShift)[ivar];
  }

  TMVA_Volume* volume = new TMVA_Volume( lb, ub );

  Float_t countS = 0;
  Float_t countB = 0;

  // -------------------------------------------------------------------------
  //
  // ==== test of volume search =====
  //
  // #define TMVA_MethodPDERS__countByHand__Debug__
#ifdef  TMVA_MethodPDERS__countByHand__Debug__
  // starting values
  countS = fBinaryTreeS->SearchVolume( volume );
  countB = fBinaryTreeB->SearchVolume( volume );

  Int_t iS = 0, iB = 0;
  for (Int_t ievt_=0; ievt_<fTrainingTree->GetEntries(); ievt_++) {

    Bool_t inV;
    for (Int_t ivar=0; ivar<fNvar; ivar++) {
      Float_t x = TMVA_Tools::GetValue( fTrainingTree, ievt_, (*fInputVars)[ivar] );
      inV = (x > (*volume->Lower)[ivar] && x <= (*volume->Upper)[ivar]);
      if (!inV) break;
    }
    if (inV) {
      if ((Int_t)TMVA_Tools::GetValue( fTrainingTree, ievt_, "type" ) == 1)
	iS++;
      else
	iB++;
    }
  }
  cout << "--- " << GetName() << ": debug: my test: S/B: "
       << iS << "  " << iB << endl;
  cout << "--- " << GetName() << ": debug: binTree: S/B: "
       << countS << "  " << countB << endl << endl;
#endif
  // -------------------------------------------------------------------------

  // adaptive volume

  if (fVRangeMode == kAdaptive) {

    // -----------------------------------------------------------------------

    if (TMVA_MethodPDERS_UseFindRoot) {

      fHelpVolume = new TMVA_Volume( *volume );

      TMVA_RootFinder rootFinder( &IGetVolumeContentForRoot, 0.01, 50, 50, 10 );
      Double_t scale = rootFinder.Root( (fNEventsMin + fNEventsMax)/2.0 );

      TMVA_Volume v( *volume );
      v.ScaleInterval( scale );

      // improve PDE by estimate of the kernel within volume
      if (TMVA_MethodPDERS_UseKernelEstimate) {
	std::vector<TMVA_Event*> eventsS;
	std::vector<TMVA_Event*> eventsB;
	fBinaryTreeS->SearchVolume( &v, &eventsS );
	fBinaryTreeB->SearchVolume( &v, &eventsB );
	countS = KernelEstimate( *e, eventsS, v );
	countB = KernelEstimate( *e, eventsB, v );
      }
      else {
	countS = fBinaryTreeS->SearchVolume( &v );
	countB = fBinaryTreeB->SearchVolume( &v );
      }

      v.Delete();

      fHelpVolume->Delete();
      delete fHelpVolume; fHelpVolume = NULL;

    }
    // -----------------------------------------------------------------------
    else {

      // starting values
      countS = fBinaryTreeS->SearchVolume( volume );
      countB = fBinaryTreeB->SearchVolume( volume );

      Float_t nEventsO = countS + countB;
      Int_t i_=0;
      while (nEventsO < fNEventsMin) { // this isn't a sain start... try again
	volume->ScaleInterval( 1.15 );
	countS = fBinaryTreeS->SearchVolume( volume );
	countB = fBinaryTreeB->SearchVolume( volume );
	nEventsO = countS + countB;
	i_++;
      }
      if (i_ > 50) cout << "--- " << GetName() << ": Warning in event: " << e
			<< ": adaptive volume pre-adjustment reached "
			<< ">50 iterations in while loop (" << i_ << ")" << endl;

      Float_t nEventsN = nEventsO;
      Float_t nEventsE = 0.5*(fNEventsMin + fNEventsMax);
      Float_t scaleO   = 1.0;
      Float_t scaleN   = fInitialScale;
      Float_t scale    = scaleN;

      Float_t cS = countS;
      Float_t cB = countB;

      for (Int_t ic=1; ic<fMaxVIterations; ic++) {
	if (nEventsN < fNEventsMin || nEventsN > fNEventsMax) {

	  // search for events in rescaled volume
	  TMVA_Volume* v = new TMVA_Volume( *volume );
	  v->ScaleInterval( scale );
	  cS       = fBinaryTreeS->SearchVolume( v );
	  cB       = fBinaryTreeB->SearchVolume( v );
	  nEventsN = cS + cB;

	  // determine next iteration (linear approximation)
	  if (nEventsN > 1 && nEventsN - nEventsO != 0)
	    if (scaleN - scaleO != 0)
	      scale += (scaleN - scaleO)/(nEventsN - nEventsO)*(nEventsE - nEventsN);
	    else
	      scale += (-0.01); // should not actually occur...
	  else
	    scale += 0.5; // use much larger volume

	  // save old scale
	  scaleN   = scale;

	  // take if better (don't accept it if too small number of events)
	  if (TMath::Abs(cS + cB - nEventsE) < TMath::Abs(countS + countB - nEventsE) &&
	      (cS + cB >= fNEventsMin || countS + countB < cS + cB)) {
	    countS = cS; countB = cB;
	  }

	  v->Delete();
	  delete v;
	}
	else break;
      }

      // last sanity check
      nEventsN = countS + countB;
      // include "1" to cover float precision
      if (nEventsN < fNEventsMin-1 || nEventsN > fNEventsMax+1)
	cout << "--- " << GetName() << ": Warning in event " << e
	     << ": adaptive volume adjustment reached "
	     << "max. #iterations (" << fMaxVIterations << ")"
	     << "[ nEvents: " << nEventsN << "  " << fNEventsMin << "  " << fNEventsMax << "]"
	     << endl;
    }

  } // end of adaptive method
  // -----------------------------------------------------------------------

  volume->Delete();
  delete volume;

  if (countS < 1e-20 && countB < 1e-20) return 0.5;
  if (countB < 1e-20) return 1.0;
  if (countS < 1e-20) return 0.0;

  Float_t r = countB*fScaleB/(countS*fScaleS);
  return 1.0/(r + 1.0);
}

//_______________________________________________________________________
Double_t TMVA_MethodPDERS::KernelEstimate( TMVA_Event& event,
					   vector<TMVA_Event*>& events, TMVA_Volume& v )
{
  // define gaussian sigmas
  Double_t fac = 0.2;
  Double_t *sigma = new Double_t[fNvar];
  for (Int_t ivar=0; ivar<fNvar; ivar++)
    sigma[ivar] = ((*v.fUpper)[ivar] - (*v.fLower)[ivar])*fac;

  Double_t pdfSum = 0;
  for (vector<TMVA_Event*>::iterator iev = events.begin(); iev != events.end(); iev++) {

    Double_t pdf = 1;
    for (Int_t ivar=0; ivar<fNvar; ivar++)
      pdf *= TMath::Gaus( event.GetData(ivar), (*iev)->GetData(ivar), sigma[ivar], kTRUE );

    pdfSum += pdf;
  }
  delete [] sigma;

  return pdfSum;
}

//_______________________________________________________________________
Float_t TMVA_MethodPDERS::GetError( Float_t countS, Float_t countB,
				    Float_t sumW2S, Float_t sumW2B ) const
{
  Float_t c = fScaleB/fScaleS;
  Float_t d = countS + c*countB; d *= d;

  if (d < 1e-10) return 1; // Error is zero because of B = S = 0

  Float_t f = c*c/d/d;
  Float_t err = f*countB*countB*sumW2S + f*countS*countS*sumW2B;

  if (err < 1e-10) return 1; // Error is zero because of B or S = 0

  return sqrt(err);
}

//_______________________________________________________________________
void TMVA_MethodPDERS::WriteWeightsToFile( void )
{
  // write coefficients to file
  TString fname = GetWeightFileName() + ".root";
  cout << "--- " << GetName() << ": creating weight file: " << fname << endl;
  TFile *fout = new TFile( fname, "RECREATE" );

  // build TList of input variables, and TVectors for min/max
  // NOTE: the latter values are mandatory for the normalisation
  // in the reader application !!!
  TList    lvar;
  TVectorD vmin( fNvar ), vmax( fNvar );
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    lvar.Add( new TNamed( (*fInputVars)[ivar], TString() ) );
    vmin[ivar] = this->GetXminNorm( ivar );
    vmax[ivar] = this->GetXmaxNorm( ivar );
  }
  // write to file
  lvar.Write();
  vmin.Write( "vmin" );
  vmax.Write( "vmax" );
  lvar.Delete();

  // save other configuration options
  // (best would be to use a TMap here, but found implementation really complicated)
  TVectorD pdersOptions( 6 );
  pdersOptions(0) = (Double_t)fVRangeMode;
  pdersOptions(1) = (Double_t)fDeltaFrac;
  pdersOptions(2) = (Double_t)fNEventsMin;
  pdersOptions(3) = (Double_t)fNEventsMax;
  pdersOptions(4) = (Double_t)fMaxVIterations;
  pdersOptions(5) = (Double_t)fInitialScale;
  pdersOptions.Write( "PdersOptions" );

  // write trainingTree
  // create clone of fTrainingTree in new file
  TObjArrayIter branchIter( fTrainingTree->GetListOfBranches(), kIterForward );
  TBranch*      branch = NULL;
  Float_t      *branchVar[1000]; //please check
  Int_t         theType, jvar = -1;
  while ((branch = (TBranch*)branchIter.Next()) != 0) {

    // note: allowed are only variables with minimum and maximum cut
    //       i.e., no distinct cut regions are supported
    if ((TString)branch->GetName() == "type")
      fTrainingTree->SetBranchAddress( branch->GetName(), &theType );
    else
      fTrainingTree->SetBranchAddress( branch->GetName(), &branchVar[++jvar] );
  }

  fTrainingTree->SetBranchStatus( "*", 1 );
  TTree *trainingTreeClone = fTrainingTree->CloneTree();
  trainingTreeClone->Write( "trainingTree" );

  fout->Close();
  delete fout;
}

//_______________________________________________________________________
void TMVA_MethodPDERS::ReadWeightsFromFile( void )
{
  // read coefficients from file
  TString fname = GetWeightFileName();
  if (!fname.EndsWith( ".root" )) fname += ".root";

  cout << "--- " << GetName() << ": reading weight file: " << fname << endl;
  fFin = new TFile( fname );

  // build TList of input variables, and TVectors for min/max
  // NOTE: the latter values are mandatory for the normalisation
  // in the reader application !!!
  TList lvar;
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    // read variable names
    TNamed t;
    t.Read( (*fInputVars)[ivar] );
    // sanity check
    if (t.GetName() != (*fInputVars)[ivar]) {
      cout << "--- " << GetName() << ": Error while reading weight file; "
	   << "unknown variable: " << t.GetName() << " at position: " << ivar << ". "
	   << "Expected variable: " << (*fInputVars)[ivar] << endl;
      throw std::invalid_argument( "Abort" );
    }
  }

  // read vectors
  TVectorD vmin( fNvar ), vmax( fNvar );
  // unfortunatly the more elegant vmin/max.Read( "vmin/max" ) crash in ROOT V4.04.02g
  TVectorD *tmp = (TVectorD*)fFin->Get( "vmin" );
  vmin = *tmp;
  tmp  = (TVectorD*)fFin->Get( "vmax" );
  vmax = *tmp;

  // initialize min/max
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    this->SetXminNorm( ivar, vmin[ivar] );
    this->SetXmaxNorm( ivar, vmax[ivar] );
  }

  // read other configuration options
  TVectorD* pdersOptions = (TVectorD*)fFin->Get( "PdersOptions" );
  fVRangeMode     = (VolumeRangeMode)(Int_t)(*pdersOptions)(0);
  fDeltaFrac	   = (*pdersOptions)(1);
  fNEventsMin	   = (Int_t)(*pdersOptions)(2);
  fNEventsMax	   = (Int_t)(*pdersOptions)(3);
  fMaxVIterations = (Int_t)(*pdersOptions)(4);
  fInitialScale   = (*pdersOptions)(5);

  // read the trainingTree
  fTrainingTree = (TTree*)fFin->Get( "trainingTree" );

  if (NULL == fTrainingTree) {
    cout << "--- " << GetName() << ": Error while reading 'trainingTree': zero pointer "
	 << endl;
    throw std::invalid_argument( "Abort" );
  }
}

//_______________________________________________________________________
void  TMVA_MethodPDERS::WriteHistosToFile( void )
{
  cout << "--- " << GetName() << ": write " << GetName()
       <<" special histos to file: " << fBaseDir->GetPath() << endl;
}

