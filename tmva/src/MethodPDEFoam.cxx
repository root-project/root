// @(#)root/tmva $Id$
// Author: Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPDEFoam                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      Alexander Voigt  - CERN, Switzerland                                      *
 *      Peter Speckmayer - CERN, Switzerland                                      *
 *                                                                                * 
 * Original author of the TFoam implementation:                                   *
 *      S. Jadach - Institute of Nuclear Physics, Cracow, Poland                  *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________

#include <iomanip>
#include <cassert>

#include "TMath.h"
#include "Riostream.h"
#include "TFile.h"
#include "TXMLFile.h"

#include "TMVA/MethodPDEFoam.h"
#include "TMVA/Tools.h"
#include "TMatrix.h"
#include "TMVA/Ranking.h"
#include "TMVA/Types.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/Config.h"

REGISTER_METHOD(PDEFoam)

ClassImp(TMVA::MethodPDEFoam)

//_______________________________________________________________________
   TMVA::MethodPDEFoam::MethodPDEFoam( const TString& jobName,
				       const TString& methodTitle,
				       DataSetInfo& dsi, 
				       const TString& theOption,
				       TDirectory* theTargetDir ) :
      MethodBase( jobName, Types::kPDEFoam, methodTitle, dsi, theOption, theTargetDir )
{
   // init PDEFoam objects
}

//_______________________________________________________________________
TMVA::MethodPDEFoam::MethodPDEFoam( DataSetInfo& dsi, 
                                    const TString& theWeightFile,  
                                    TDirectory* theTargetDir ) :
   MethodBase( Types::kPDEFoam, dsi, theWeightFile, theTargetDir )
{
   // constructor from weight file
}

//_______________________________________________________________________
Bool_t TMVA::MethodPDEFoam::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // PDEFoam can handle classification with 2 classes and regression with one or more regression-targets
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kRegression) return kTRUE;
   return kFALSE;
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::Init( void )
{
   // default initialization called by all constructors

   // init PDEFoam options
   fSigBgSeparated = kFALSE;   // default values for options
   fFrac           = 0.001;
   fDiscrErrCut    = -1.;
   fVolFrac        = 30;
   fVolFracInv     = 1./30.;
   fnActiveCells   = 500;
   fnCells         = fnActiveCells*2-1;
   fnSampl         = 2000;
   fnBin           = 5;
   fOptRej         = 1;       // use unweighted MC events
   fOptDrive       = 1;       // use variance reduction
   fEvPerBin       = 10000;
   fChat           = 0;
   fCutNmin        = true; 
   fNmin           = 100;
   fCutRMSmin      = false;   // default TFoam method
   fRMSmin         = 0.01;
   
   fKernel         = kNone; // default: use no kernel
   fTargetSelection= kMean; // default: use mean for target selection (only multi target regression!)

   fCompress              = kTRUE;
   fMultiTargetRegression = kFALSE;

   for (int i=0; i<FOAM_NUMBER; i++) foam[i] = NULL;

   PseRan = new TRandom3();  // Create random number generator for PDEFoams
   PseRan->SetSeed(4356);

   SetSignalReferenceCut( 0.0 );
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::DeclareOptions() 
{
   //
   // Declare MethodPDEFoam options
   //
   DeclareOptionRef( fSigBgSeparated = kFALSE, "SigBgSeparate", "Separate foams for signal and background" );
   DeclareOptionRef( fFrac = 0.001,           "TailCut",  "Fraction of outlier events that are excluded from the foam in each dimension" );
   DeclareOptionRef( fVolFracInv = 1./30.,    "VolFrac",  "Size of sampling box, used for density calculation during foam build-up (maximum value: 1.0 is equivalent to volume of entire foam)");
   DeclareOptionRef( fnActiveCells = 500,     "nActiveCells",  "Maximum number of active cells to be created by the foam");
   DeclareOptionRef( fnSampl = 2000,          "nSampl",   "Number of generated MC events per cell");
   DeclareOptionRef( fnBin = 5,               "nBin",     "Number of bins in edge histograms");
   DeclareOptionRef( fCompress = kTRUE,       "Compress", "Compress XML file");
   DeclareOptionRef( fMultiTargetRegression = kFALSE,     "MultiTargetRegression", "Do regression with multiple targets");
   DeclareOptionRef( fCutNmin = true,         "CutNmin",  "Requirement for minimal number of events in cell");
   DeclareOptionRef( fNmin = 100,             "Nmin",     "Number of events in cell required to split cell");
   
   DeclareOptionRef( fKernelStr = "None",     "Kernel",   "Kernel type used");
   AddPreDefVal(TString("None"));
   AddPreDefVal(TString("Gauss"));
   AddPreDefVal(TString("LinNeighbors"));
   DeclareOptionRef( fTargetSelectionStr = "Mean", "TargetSelection", "Target selection method");
   AddPreDefVal(TString("Mean"));
   AddPreDefVal(TString("Mpv"));
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::ProcessOptions() 
{
   // process user options
   if (!(fFrac>0. && fFrac<=1.)) {
      Log() << kWARNING << "TailCut not in [0.,1] ==> using 0.001 instead" << Endl;
      fFrac = 0.001;
   }

   fnCells = fnActiveCells*2-1;

   fVolFrac = Int_t(1./fVolFracInv + 0.5); // round

   if (fCutRMSmin && fRMSmin>1.0) {
      Log() << kWARNING << "RMSmin > 1.0 ==> using 1.0 instead" << Endl;
      fRMSmin = 1.0;
   }
   
   if (fNmin==0)
      fCutNmin = false;

   if (fKernelStr == "None" ) fKernel = kNone;
   else if (fKernelStr == "Gauss" ) fKernel = kGaus;
   else if (fKernelStr == "LinNeighbors") fKernel = kLinN;

   if (fTargetSelectionStr == "Mean" ) fTargetSelection = kMean;
   else                                fTargetSelection = kMpv;
}

//_______________________________________________________________________
TMVA::MethodPDEFoam::~MethodPDEFoam( void )
{
   // destructor
   if(foam[0]) {
      delete foam[0];  
      foam[0]=0;
   }
   if (!DoRegression() && fSigBgSeparated && foam[1]){
      delete foam[1];
      foam[1]=0;
   }
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::CalcXminXmax() 
{
   // determine foam range for all dimensions
   // loop over all testing events -> Get Xmin and Xmax

   Xmin.clear();
   Xmax.clear();
   UInt_t kDim = GetNvar(); // == Data()->GetNVariables();
   UInt_t tDim = Data()->GetNTargets();
   UInt_t vDim = Data()->GetNVariables();
   if (fMultiTargetRegression)
      kDim += tDim;

   Double_t *xmin = new Double_t[kDim];
   Double_t *xmax = new Double_t[kDim];

   // set default values
   for (UInt_t dim=0; dim<kDim; dim++) {
      xmin[dim] =  1.e100;
      xmax[dim] = -1.e100;
   }

   Log() << kDEBUG << "Number of training events: " << Data()->GetNTrainingEvents() << Endl;
   Int_t nevoutside = (Int_t)((Data()->GetNTrainingEvents())*(fFrac)); // number of events that are outside the range
   Int_t rangehistbins = 10000;                               // number of bins in histos
  
   // loop over all testing singnal and BG events and clac minimal and maximal value of every variable
   for (Long64_t i=0; i<(GetNEvents()); i++) { // events loop
      const Event* ev = GetEvent(i);    
      for (UInt_t dim=0; dim<kDim; dim++) { // variables loop
         Double_t val;
         if (fMultiTargetRegression) {
            if (dim < vDim)
               val = ev->GetVal(dim);
            else 
               val = ev->GetTarget(dim-vDim);
         }
         else
            val = ev->GetVal(dim);

         if (val<xmin[dim])
            xmin[dim] = val;
         if (val>xmax[dim])
            xmax[dim] = val;
      }
   }

   // Create and fill histograms for each dimension (with same events as before), to determine range 
   // based on number of events outside the range
   TH1F **range_h = new TH1F*[kDim]; 
   char text[200];
   for (UInt_t dim=0; dim<kDim; dim++) {
      sprintf(text, "range%i", dim);
      range_h[dim]  = new TH1F(text, "range", rangehistbins, xmin[dim], xmax[dim]);
   }

   // fill all testing events into histos 
   for (Long64_t i=0; i<GetNEvents(); i++) {
      const Event* ev = GetEvent(i);
      for (UInt_t dim=0; dim<kDim; dim++) {
         if (fMultiTargetRegression) {
            if (dim < vDim)
               range_h[dim]->Fill(ev->GetVal(dim));
            else
               range_h[dim]->Fill(ev->GetTarget(dim-vDim));
         }
         else
            range_h[dim]->Fill(ev->GetVal(dim));
      }
   };

   // calc Xmin, Xmax from Histos
   for (UInt_t dim=0; dim<kDim; dim++) { 
      for (Int_t i=1; i<(rangehistbins+1); i++) { // loop over bins 
         if (range_h[dim]->Integral(0, i) > nevoutside) { // calc left limit (integral over bins 0..i = nevoutside)
            xmin[dim]=range_h[dim]->GetBinLowEdge(i);
            break;
         }
      }
      for (Int_t i=rangehistbins; i>0; i--) { // calc right limit (integral over bins i..max = nevoutside)
         if (range_h[dim]->Integral(i, (rangehistbins+1)) > nevoutside) {
            xmax[dim]=range_h[dim]->GetBinLowEdge(i+1);
            break;
         }
      }
   }  
   // now xmin[] and xmax[] contain upper/lower limits for every dimension

   // copy xmin[], xmax[] values to the class variable
   Xmin.clear();
   Xmax.clear();
   for (UInt_t dim=0; dim<kDim; dim++) { 
      Xmin.push_back(xmin[dim]);
      Xmax.push_back(xmax[dim]);
   }


   delete[] xmin;
   delete[] xmax;

   // delete histos
   for (UInt_t dim=0; dim<kDim; dim++)
      delete range_h[dim];
   delete[] range_h;

   return;
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::Train( void )
{
   Log() << kDEBUG << "Calculate Xmin and Xmax for every dimension" << Endl;
   CalcXminXmax();


   if (DoRegression()) {
      if (fMultiTargetRegression)
         TrainMultiTargetRegression();
      else
         TrainMonoTargetRegression();
   }
   else {
      if (DataInfo().GetNormalization() != "EQUALNUMEVENTS" ) { 
	 Log() << kINFO << "NormMode=" << DataInfo().GetNormalization() 
	       << " chosen. Note that only NormMode=EqualNumEvents" 
	       << " ensures that Discriminant values correspond to"
	       << " signal probabilities." << Endl;
      }

      Log() << kDEBUG << "N_sig for training events: " << Data()->GetNEvtSigTrain() << Endl;
      Log() << kDEBUG << "N_bg for training events:  " << Data()->GetNEvtBkgdTrain() << Endl;
      Log() << kDEBUG << "User normalization: " << DataInfo().GetNormalization().Data() << Endl;

      if (fSigBgSeparated)
         TrainSeparatedClassification();
      else
         TrainUnifiedClassification();
   }
}

/////////////////////////////////////////////////////////////////////////////
//  First method (signal and bg are in seperate foams).                    //
/////////////////////////////////////////////////////////////////////////////
//_______________________________________________________________________
void TMVA::MethodPDEFoam::TrainSeparatedClassification() 
{
   // Creation of 2 separated foams: one for signal events, one for 
   // backgound events.

   TString foamcaption[2];
   foamcaption[0] = "SignalFoam";
   foamcaption[1] = "BgFoam";

   for(int i=0; i<FOAM_NUMBER; i++) {
      // create 2 PDEFoams
      foam[i] = new PDEFoam(foamcaption[i]);
      InitFoam(foam[i], kSeparate);

      Log() << kINFO << "Filling binary search tree of " << foamcaption[i] 
            << " with events" << Endl;
      // insert event to BinarySearchTree
      for (Long64_t k=0; k<GetNEvents(); k++) {
         const Event* ev = GetEvent(k);
         if ((i==0 && ev->IsSignal()) || (i==1 && !ev->IsSignal()))
            foam[i]->FillBinarySearchTree(ev, kSeparate, IgnoreEventsWithNegWeightsInTraining());
      }

      Log() << kINFO << "Build " << foamcaption[i] << Endl;
      // build foam
      foam[i]->SetNElements(1);  // init space for 1 variable on every cell (number of events in cell)
      foam[i]->Create(fCutNmin);

      // Reset Cell Integrals
      foam[i]->SetNElements(2);  // init space for 2 variables on every cell (N_ev, RMS)
      foam[i]->ResetCellElements();

      Log() << "Filling foam cells with events" << Endl;
      // loop over all events -> fill foam cells
      for (Long64_t k=0; k<GetNEvents(); k++) {
         const Event* ev = GetEvent(k); 
         if ((i==0 && ev->IsSignal()) || (i==1 && !ev->IsSignal()))
            foam[i]->FillFoamCells(ev, kSeparate, IgnoreEventsWithNegWeightsInTraining());
      }

      Log() << kDEBUG << "Check all cells and remove cells with volume 0" << Endl;
      foam[i]->CheckCells(true);
   }
}

//////////////////////////////////////////////////////////////////////////
// second method (only one foam, which contains discriminator).         //
//////////////////////////////////////////////////////////////////////////
void TMVA::MethodPDEFoam::TrainUnifiedClassification() 
{
   // Create only one unified foam which contains discriminator
   // (N_sig)/(N_sig + N_bg)

   foam[0] = new PDEFoam("DiscrFoam");
   InitFoam(foam[0], kDiscr);

   Log() << kINFO << "Filling binary search tree of discriminator foam with events" << Endl;
   // insert event to BinarySearchTree
   for (Long64_t k=0; k<GetNEvents(); k++)
      foam[0]->FillBinarySearchTree(GetEvent(k), kDiscr, IgnoreEventsWithNegWeightsInTraining());

   Log() << kINFO << "Build up discriminator foam" << Endl;
   // build foam with 1 cell element
   foam[0]->SetNElements(1);     // init space for 1 variable on every cell (number of events in cell)
   foam[0]->Create(fCutNmin);    // build foam and create cell elements if Nmin-cut is activated

   Log() << kDEBUG << "Resetting cell integrals" << Endl;
   // Reset cell elements, used after foam build-up
   foam[0]->SetNElements(2);     // init space for 2 variables on every cell
   foam[0]->ResetCellElements();

   Log() << "Filling foam cells with events" << Endl;
   // loop over all training events -> fill foam cells with N_sig and N_Bg
   for (UInt_t k=0; k<GetNEvents(); k++)
      foam[0]->FillFoamCells(GetEvent(k), kDiscr, IgnoreEventsWithNegWeightsInTraining());

   Log() << "Calculate cell discriminator"<< Endl;
   // calc discriminator (and it's error) for each cell
   foam[0]->CalcCellDiscr();

   Log() << kDEBUG << "Check all cells and remove cells with volume 0" << Endl;
   foam[0]->CheckCells(true);
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::TrainMonoTargetRegression() 
{
   // Training for mono target regression
   // - foam density = average Target(0)
   // - dimension of foam = number of non-targets
   // - cell content = average target 0

   if (Data()->GetNTargets() < 1) {
      Log() << kFATAL << "Error: number of targets = " << Data()->GetNTargets() << Endl;
      return;
   }
   else if (Data()->GetNTargets() > 1) {
      Log() << kWARNING << "Warning: number of targets = " << Data()->GetNTargets()
            << "  --> using only first target" << Endl;
   }
   else 
      Log() << kDEBUG << "MethodPDEFoam: number of Targets: " << Data()->GetNTargets() << Endl;

   TString foamcaption = "MonoTargetRegressionFoam";
   foam[0] = new PDEFoam(foamcaption);
   InitFoam(foam[0], kMonoTarget);

   Log() << kINFO << "Filling binary search tree with events" << Endl;
   // insert event to BinarySearchTree
   for (Long64_t k=0; k<GetNEvents(); k++)
      foam[0]->FillBinarySearchTree(GetEvent(k), kMonoTarget, IgnoreEventsWithNegWeightsInTraining());

   Log() << kINFO << "Build mono target regression foam" << Endl;
   // build foam
   foam[0]->SetNElements(1);        // to save N_ev during foam build-up
   foam[0]->Create(fCutNmin);

   Log() << kDEBUG << "Resetting cell elements" << Endl;
   // Reset Cell Integrals
   foam[0]->SetNElements(2);        // to save N_ev and Target(0)
   foam[0]->ResetCellElements();

   Log() << "Filling foam cells with events" << Endl;
   // loop over all events -> fill foam cells with target
   for (UInt_t k=0; k<GetNEvents(); k++)
      foam[0]->FillFoamCells(GetEvent(k), kMonoTarget, IgnoreEventsWithNegWeightsInTraining());

   Log() << kDEBUG << "Calculate cell average targets"<< Endl;
   // calc weight (and it's error) for each cell
   foam[0]->CalcCellTarget();

   Log() << kDEBUG << "Check all cells and remove cells with volume 0" << Endl;
   foam[0]->CheckCells(true);
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::TrainMultiTargetRegression()
{
   // Training multi target regression
   // - foam density = Event density
   // - dimension of foam = number of non-targets + number of targets
   // - cell content = event density

   Log() << kDEBUG << "Number of variables: " << Data()->GetNVariables() << Endl;
   Log() << kDEBUG << "Number of Targets:   " << Data()->GetNTargets()   << Endl;
   Log() << kDEBUG << "Dimension of foam:   " << Data()->GetNVariables()+Data()->GetNTargets() << Endl;
   if (fKernel==kLinN)
      Log() << kFATAL << "LinNeighbors kernel currently not supported" 
	    << " for multi target regression" << Endl;

   TString foamcaption = "MultiTargetRegressionFoam";
   foam[0] = new PDEFoam(foamcaption);
   InitFoam(foam[0], kMultiTarget);

   Log() << kINFO << "Filling binary search tree of multi target regression foam with events" 
         << Endl;
   // insert event to BinarySearchTree
   for (Long64_t k=0; k<GetNEvents(); k++)
      foam[0]->FillBinarySearchTree(GetEvent(k), kMultiTarget, IgnoreEventsWithNegWeightsInTraining());

   Log() << kINFO << "Build multi target regression foam" << Endl;
   // build foam
   foam[0]->SetNElements(1);          // to save N_ev during build-up
   foam[0]->Create(fCutNmin);

   Log() << kDEBUG << "Resetting cell elements" << Endl;
   // Reset Cell values
   foam[0]->SetNElements(2);          // to save N_ev and RMS
   foam[0]->ResetCellElements();

   Log() << kINFO << "Filling foam cells with events" << Endl;
   // loop over all events -> fill foam cells with number of events
   for (UInt_t k=0; k<GetNEvents(); k++)
      foam[0]->FillFoamCells(GetEvent(k), kMultiTarget, IgnoreEventsWithNegWeightsInTraining());

   Log() << kDEBUG << "Check all cells and remove cells with volume 0" << Endl;
   foam[0]->CheckCells(true);
}

//_______________________________________________________________________
Double_t TMVA::MethodPDEFoam::GetMvaValue( Double_t* err )
{
   // Return Mva-Value for both separated and unified foams

   const Event* ev = GetEvent();
   Double_t discr = 0.;
   Double_t discr_error = 0.;

   if (fSigBgSeparated) {
      std::vector<Float_t> xvec = ev->GetValues();

      Double_t density_sig = 0.;
      Double_t density_bg  = 0.;

      density_sig = foam[0]->GetCellDensity(xvec, fKernel); // get signal event density
      density_bg  = foam[1]->GetCellDensity(xvec, fKernel); // get background event density

      // calc disciminator (normed!)
      if ( (density_sig+density_bg) > 0 )
         discr = density_sig/(density_sig+density_bg);
      else
         discr = 0.5; // assume 50% signal probability, if no events found (bad assumption, but can be overruled by cut on error)

      // do error estimation (not jet used in TMVA)
      Double_t neventsB = foam[1]->GetCellEntries(xvec);
      Double_t neventsS = foam[0]->GetCellEntries(xvec);
      Double_t scaleB = 1.;
      Double_t errorS = TMath::Sqrt(neventsS); // estimation of statistical error on counted signal events
      Double_t errorB = TMath::Sqrt(neventsB); // estimation of statistical error on counted background events

      if (neventsS == 0) // no signal events in cell
         errorS = 1.;
      if (neventsB == 0) // no bg events in cell
         errorB = 1.;

      if ( (neventsS>1e-10) || (neventsB>1e-10) ) // eq. (5) in paper T.Carli, B.Koblitz 2002
         discr_error = TMath::Sqrt( TMath::Power ( scaleB*neventsB
                                                   / TMath::Power((neventsS+scaleB*neventsB),2)
                                                   * errorS, 2) +
                                    TMath::Power ( (scaleB*neventsS)
                                                   / TMath::Power((neventsS+scaleB*neventsB),2)
                                                   * errorB, 2) );
      else discr_error = 1.;

      if (discr_error < 1e-10) discr_error = 1.;

      // do not classify, if error too big
      if (fDiscrErrCut>=0.0 && discr_error > fDiscrErrCut) discr = -1.;
   }

   /////////////////////////////////////////////////////////////////////////////
   //  End of first method (signal and bg are in seperate foams).             //
   //  Begin second method (only one foam, which contains discriminator).     //
   /////////////////////////////////////////////////////////////////////////////

   else { // Signal and Bg not separated
      std::vector<Float_t> xvec = ev->GetValues();
      
      // get discriminator direct from the foam
      discr       = foam[0]->GetCellDiscr(xvec, fKernel);
      discr_error = foam[0]->GetCellDiscrError(xvec);
	 
      // do not classify, if error too big
      if (fDiscrErrCut>=0.0 && discr_error > fDiscrErrCut) discr = -1.;
   }

   // attribute error
   if (err != 0) *err = discr_error;

   return discr;
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::SetXminXmax( TMVA::PDEFoam *pdefoam )
{
   // Set Xmin, Xmax in every dimension to pdefoam

   if (!pdefoam){
      Log() << kFATAL << "Null pointer given!" << Endl;
      return;
   }

   UInt_t num_vars = GetNvar();
   if (fMultiTargetRegression)
      num_vars += Data()->GetNTargets();

   for (UInt_t idim=0; idim<num_vars; idim++) { // set upper/ lower limit in foam
      Log()<< kDEBUG << "foam: SetXmin[dim="<<idim<<"]: " << Xmin.at(idim) << Endl;
      Log()<< kDEBUG << "foam: SetXmax[dim="<<idim<<"]: " << Xmax.at(idim) << Endl;
      pdefoam->SetXmin(idim, Xmin.at(idim));
      pdefoam->SetXmax(idim, Xmax.at(idim));
   }
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::InitFoam(TMVA::PDEFoam *pdefoam, EFoamType ft){
   // Set foam options and initialize foam

   if (!pdefoam){
      Log() << kFATAL << "Null pointer given!" << Endl;
      return;
   }

   // set signal and background classes
   if (ft==kSeparate || ft==kDiscr){
      pdefoam->SetSignalClass    (fSignalClass);
      pdefoam->SetBackgroundClass(fBackgroundClass);
   }

   // Set foam fill value
   pdefoam->SetFoamType(ft);
   
   // set Options VolFrac, kDim, ...
   if (ft==kMultiTarget)
      // dimension of foam = number of targets + non-targets
      pdefoam->SetkDim(     Data()->GetNTargets()+Data()->GetNVariables());
   else
      pdefoam->SetkDim(     GetNvar());  // Mandatory!
   pdefoam->SetVolumeFraction(fVolFrac); // Mandatory!
   pdefoam->SetnCells(      fnCells);    // optional
   pdefoam->SetnSampl(      fnSampl);    // optional
   pdefoam->SetnBin(        fnBin);      // optional
   pdefoam->SetOptRej(      fOptRej);    // optional
   pdefoam->SetOptDrive(    fOptDrive);  // optional
   pdefoam->SetEvPerBin(    fEvPerBin);  // optional
   pdefoam->SetChat(        fChat);      // optional

   // cuts
   pdefoam->CutNmin(fCutNmin);     // cut on minimal number of events per cell
   pdefoam->SetNmin(fNmin);
   pdefoam->CutRMSmin(fCutRMSmin); // cut on minimal RMS in cell
   pdefoam->SetRMSmin(fRMSmin);

   // SetPseRan
   pdefoam->SetPseRan(PseRan);

   // Init PDEFoam
   pdefoam->Init();
   
   // Set Xmin, Xmax
   SetXminXmax(pdefoam);
}

//_______________________________________________________________________
const std::vector<Float_t>& TMVA::MethodPDEFoam::GetRegressionValues()
{
   // Return regression values for both multi and mono target regression

   if (fRegressionReturnVal == 0) fRegressionReturnVal = new std::vector<Float_t>;
   fRegressionReturnVal->clear();

   const Event* ev = GetEvent();
   std::vector<Float_t> vals = ev->GetValues(); // get array of event variables (non-targets)   

   if (vals.size() == 0) {
      Log() << kWARNING << "<GetRegressionValues> value vector has size 0. " << Endl;
   }

   if (fMultiTargetRegression) {
      for(UInt_t i=0; i<(Data()->GetNTargets()); i++)
         fRegressionReturnVal->push_back(foam[0]->GetProjectedRegValue(i, vals, fKernel, fTargetSelection));
   }
   else {
      fRegressionReturnVal->push_back(foam[0]->GetCellRegValue0(vals, fKernel));   
   }

   Event * evT = new Event(*ev);
   for (UInt_t itgt = 0; itgt < evT->GetNTargets(); itgt++) {
      evT->SetTarget(itgt, fRegressionReturnVal->at(itgt) );
   }
   const Event* evT2 = GetTransformationHandler().InverseTransform( evT );
   fRegressionReturnVal->clear();
   for (UInt_t itgt = 0; itgt < evT->GetNTargets(); itgt++) {
      fRegressionReturnVal->push_back( evT2->GetTarget(itgt) );
   }

   delete evT;

   return (*fRegressionReturnVal);
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::PrintCoefficients( void ) 
{}
  
//_______________________________________________________________________
void  TMVA::MethodPDEFoam::WriteWeightsToStream( std::ostream& o ) const
{  
   // save options and internal parameters

   o << fSigBgSeparated << endl;                 // Seperate Sig and Bg, or not
   o << std::setprecision(12) << fFrac << endl;  // Fraction used for calc of Xmin, Xmax
   o << std::setprecision(12) << fDiscrErrCut << endl; // cut on discrimant error
   o << fVolFrac << endl;                        // volume fraction (used for density calculation during buildup)
   o << fnCells << endl;                         // Number of Cells  (500)
   o << fnSampl << endl;                         // Number of MC events per cell in build-up (1000)
   o << fnBin << endl;                           // Number of bins in build-up (100)
   o << fOptRej << endl;                         // Wted events for OptRej=0; wt=1 for OptRej=1 (default)
   o << fOptDrive << endl;                       // (D=2) Option, type of Drive =1,2 for Variance,WtMax driven reduction
   o << fEvPerBin << endl;                       // Maximum events (equiv.) per bin in buid-up (1000) 
   o << fChat << endl;                           // Chat level
   o << fCompress << endl;                       // Compress XML file
   o << DoRegression() << endl;                  // regression foam created
   o << fCutNmin << endl;                        // split only cells with number of events > fNmin
   o << fNmin << endl;                           // min number of events in cell
   o << fCutRMSmin << endl;                      // split only cells with RMS > fRMSmin
   o << fRMSmin << endl;                         // min RMS in cell
   o << KernelToUInt(fKernel) << endl;           // used kernel for GetMvaValue()
   o << TargetSelectionToUInt(fTargetSelection) << endl; // used method for target selection

   // save range
   for (UInt_t i=0; i<Xmin.size(); i++) 
      o << std::setprecision(12) << Xmin.at(i) << endl;
   for (UInt_t i=0; i<Xmax.size(); i++) 
      o << std::setprecision(12) << Xmax.at(i) << endl;

   // write internal foam variables to stream
   if (DoRegression())
      o << *(foam[0]);
   else {
      if (fSigBgSeparated) {
         for(int i=0; i<FOAM_NUMBER; i++)
            o << *(foam[i]);
      }
      else
         o << *(foam[0]);
   }

   // write foams to xml file
   WriteFoamsToXMLFile();
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::AddWeightsXMLTo( void* parent ) const 
{
   // create XML output of PDEFoam method variables

   void* wght = gTools().xmlengine().NewChild(parent, 0, "Weights");
   gTools().AddAttr( wght, "SigBgSeparated",  fSigBgSeparated );
   gTools().AddAttr( wght, "Frac",            fFrac );
   gTools().AddAttr( wght, "DiscrErrCut",     fDiscrErrCut );
   gTools().AddAttr( wght, "VolFrac",         fVolFrac );
   gTools().AddAttr( wght, "nCells",          fnCells );
   gTools().AddAttr( wght, "nSampl",          fnSampl );
   gTools().AddAttr( wght, "nBin",            fnBin );
   gTools().AddAttr( wght, "OptRej",          fOptRej );
   gTools().AddAttr( wght, "OptDrive",        fOptDrive );
   gTools().AddAttr( wght, "EvPerBin",        fEvPerBin );
   gTools().AddAttr( wght, "Chat",            fChat );
   gTools().AddAttr( wght, "Compress",        fCompress );
   gTools().AddAttr( wght, "DoRegression",    DoRegression() );
   gTools().AddAttr( wght, "CutNmin",         fCutNmin );
   gTools().AddAttr( wght, "Nmin",            fNmin );
   gTools().AddAttr( wght, "CutRMSmin",       fCutRMSmin );
   gTools().AddAttr( wght, "RMSmin",          fRMSmin );
   gTools().AddAttr( wght, "Kernel",          KernelToUInt(fKernel) );
   gTools().AddAttr( wght, "TargetSelection", TargetSelectionToUInt(fTargetSelection) );
   
   // save foam borders Xmin[i], Xmax[i]
   void *xmin_wrap;
   for (UInt_t i=0; i<Xmin.size(); i++){
      xmin_wrap = gTools().xmlengine().NewChild( wght, 0, "Xmin" );
      gTools().AddAttr( xmin_wrap, "Index", i );
      gTools().AddAttr( xmin_wrap, "Value", Xmin.at(i) );
   }
   void *xmax_wrap;
   for (UInt_t i=0; i<Xmin.size(); i++){
      xmax_wrap = gTools().xmlengine().NewChild( wght, 0, "Xmax" );
      gTools().AddAttr( xmax_wrap, "Index", i );
      gTools().AddAttr( xmax_wrap, "Value", Xmax.at(i) );
   }

   // write internal foam variables
   void *foam_wrap = gTools().xmlengine().NewChild( wght, 0, foam[0]->GetFoamName().Data());
   foam[0]->AddXMLTo(foam_wrap);
   if((!DoRegression()) && fSigBgSeparated){
      void *foam_wrap2 = gTools().xmlengine().NewChild( wght, 0, foam[1]->GetFoamName().Data());
      foam[1]->AddXMLTo(foam_wrap2);
   }

   // write foams to xml file
   WriteFoamsToXMLFile();
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::WriteFoamsToXMLFile() const 
{
   // Write pure foams to xml file

   // fill variable names into foam
   FillVariableNamesToFoam();   

   TString rfname( GetWeightFileName() ); 

   // replace in case of txt weight file
   rfname.ReplaceAll( TString(".") + gConfig().GetIONames().fWeightFileExtension + ".txt", ".xml" );   

   // add foam indicator to distinguish from main weight file
   rfname.ReplaceAll( ".xml", "_foams.xml" );

   TXMLFile *rootFile = 0;
   if (fCompress) rootFile = new TXMLFile(rfname, "RECREATE", "foamfile", 1);
   else           rootFile = new TXMLFile(rfname, "RECREATE");


   foam[0]->Write(foam[0]->GetFoamName().Data());
   if (!DoRegression() && fSigBgSeparated) 
      foam[1]->Write(foam[1]->GetFoamName().Data());
   rootFile->Close();
   Log() << kINFO << "Foams written to XML file: " 
         << gTools().Color("lightblue") << rfname << gTools().Color("reset") << Endl;
};

//_______________________________________________________________________
void  TMVA::MethodPDEFoam::ReadWeightsFromStream( istream& istr )
{
   // read options and internal parameters

   istr >> fSigBgSeparated;                 // Seperate Sig and Bg, or not
   istr >> fFrac;                           // Fraction used for calc of Xmin, Xmax
   istr >> fDiscrErrCut;                    // cut on discrimant error
   istr >> fVolFrac;                        // volume fraction (used for density calculation during buildup)
   istr >> fnCells;                         // Number of Cells  (500)
   istr >> fnSampl;                         // Number of MC events per cell in build-up (1000)
   istr >> fnBin;                           // Number of bins in build-up (100)
   istr >> fOptRej;                         // Wted events for OptRej=0; wt=1 for OptRej=1 (default)
   istr >> fOptDrive;                       // (D=2) Option, type of Drive =1,2 for Variance,WtMax driven reduction
   istr >> fEvPerBin;                       // Maximum events (equiv.) per bin in buid-up (1000) 
   istr >> fChat;   
   istr >> fCompress;                       // compress XML file

   Bool_t regr;
   istr >> regr;                            // regression foam
   SetAnalysisType( (regr ? Types::kRegression : Types::kClassification ) );
   
   istr >> fCutNmin;                        // cut on minimal number of events in cell
   istr >> fNmin;
   istr >> fCutRMSmin;                      // cut on minimal RMS in cell
   istr >> fRMSmin;

   UInt_t ker = 0;
   istr >> ker;                             // used kernel for GetMvaValue()
   fKernel = UIntToKernel(ker);

   UInt_t ts = 0;
   istr >> ts;                             // used method for target selection
   fTargetSelection = UIntToTargetSelection(ts);


   // clear old range and prepare new range
   Xmin.clear();
   Xmax.clear();
   UInt_t kDim = GetNvar();
   if (fMultiTargetRegression)
      kDim += Data()->GetNTargets();

   for (UInt_t i=0; i<kDim; i++) {
      Xmin.push_back(0.);
      Xmax.push_back(0.);
   }
   // read range
   for (UInt_t i=0; i<kDim; i++) 
      istr >> Xmin.at(i);
   for (UInt_t i=0; i<kDim; i++) 
      istr >> Xmax.at(i);

   // create foam(s)
   if (DoRegression()) {
      if (fMultiTargetRegression) foam[0] = new PDEFoam("MultiTargetRegressionFoam");
      else                        foam[0] = new PDEFoam("MonoTargetRegressionFoam");
      istr >> *(foam[0]);
   }
   else {  
      TString foamcaption[2];
      foamcaption[0] = "SignalFoam";
      foamcaption[1] = "BgFoam";
      if (fSigBgSeparated) {
         for(int i=0; i<FOAM_NUMBER; i++) {
            foam[i] = new PDEFoam(foamcaption[i]);
            foam[i]->SetSignalClass    (fSignalClass);
            foam[i]->SetBackgroundClass(fBackgroundClass);
            istr >> *(foam[i]);
         }
      }
      else {
         foam[0] = new PDEFoam("DiscrFoam");
         foam[0]->SetSignalClass    (fSignalClass);
         foam[0]->SetBackgroundClass(fBackgroundClass);
         istr >> *(foam[0]);
      }
   }

   // read pure foams from xml file
   ReadFoamsFromXMLFile();
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::ReadWeightsFromXML( void* wghtnode ) 
{
   // read PDEFoam variables from xml weight file

   gTools().ReadAttr( wghtnode, "SigBgSeparated",  fSigBgSeparated );
   gTools().ReadAttr( wghtnode, "Frac",            fFrac );
   gTools().ReadAttr( wghtnode, "DiscrErrCut",     fDiscrErrCut );
   gTools().ReadAttr( wghtnode, "VolFrac",         fVolFrac );
   gTools().ReadAttr( wghtnode, "nCells",          fnCells );
   gTools().ReadAttr( wghtnode, "nSampl",          fnSampl );
   gTools().ReadAttr( wghtnode, "nBin",            fnBin );
   gTools().ReadAttr( wghtnode, "OptRej",          fOptRej );
   gTools().ReadAttr( wghtnode, "OptDrive",        fOptDrive );
   gTools().ReadAttr( wghtnode, "EvPerBin",        fEvPerBin );
   gTools().ReadAttr( wghtnode, "Chat",            fChat );
   gTools().ReadAttr( wghtnode, "Compress",        fCompress );
   Bool_t regr;
   gTools().ReadAttr( wghtnode, "DoRegression",    regr );
   SetAnalysisType( (regr ? Types::kRegression : Types::kClassification ) );
   gTools().ReadAttr( wghtnode, "CutNmin",         fCutNmin );
   gTools().ReadAttr( wghtnode, "Nmin",            fNmin );
   gTools().ReadAttr( wghtnode, "CutRMSmin",       fCutRMSmin );
   gTools().ReadAttr( wghtnode, "RMSmin",          fRMSmin );
   UInt_t ker = 0;
   gTools().ReadAttr( wghtnode, "Kernel",          ker );
   fKernel = UIntToKernel(ker);
   UInt_t ts = 0;
   gTools().ReadAttr( wghtnode, "TargetSelection", ts );
   fTargetSelection = UIntToTargetSelection(ts);
   
   // clear old range [Xmin, Xmax] and prepare new range for reading
   Xmin.clear();
   Xmax.clear();
   UInt_t kDim = GetNvar();
   if (fMultiTargetRegression)
      kDim += Data()->GetNTargets();

   for (UInt_t i=0; i<kDim; i++) {
      Xmin.push_back(0.);
      Xmax.push_back(0.);
   }

   // read foam range
   void *xmin_wrap = gTools().xmlengine().GetChild( wghtnode );
   for (UInt_t counter=0; counter<kDim; counter++) {
      UInt_t i=0;
      gTools().ReadAttr( xmin_wrap , "Index", i );
      if (i>=kDim)
         Log() << kFATAL << "dimension index out of range:" << i << Endl;
      gTools().ReadAttr( xmin_wrap , "Value", Xmin.at(i) );
      xmin_wrap = gTools().xmlengine().GetNext( xmin_wrap );
   }

   void *xmax_wrap = xmin_wrap;
   for (UInt_t counter=0; counter<kDim; counter++) {
      UInt_t i=0;
      gTools().ReadAttr( xmax_wrap , "Index", i );
      if (i>=kDim)
         Log() << kFATAL << "dimension index out of range:" << i << Endl;
      gTools().ReadAttr( xmax_wrap , "Value", Xmax.at(i) );
      xmax_wrap = gTools().xmlengine().GetNext( xmax_wrap );
   }

   // create foam(s) and read internal variables from xml file
   if (foam[0]) delete foam[0];
   if (foam[1]) delete foam[1];
   if (DoRegression()) {
      if (fMultiTargetRegression) {
         foam[0] = new PDEFoam("MultiTargetRegressionFoam");
      }else {
         foam[0] = new PDEFoam("MonoTargetRegressionFoam");
      }
      void *foam_wrap = xmax_wrap;
      foam[0]->ReadXML( foam_wrap );
   }
   else {  
      TString foamcaption[2];
      foamcaption[0] = "SignalFoam";
      foamcaption[1] = "BgFoam";
      if (fSigBgSeparated) {
         void *foam_wrap = xmax_wrap;
         for(int i=0; i<FOAM_NUMBER; i++) {
            foam[i] = new PDEFoam(foamcaption[i]);
            foam[i]->SetSignalClass    (fSignalClass);
            foam[i]->SetBackgroundClass(fBackgroundClass);
            foam[i]->ReadXML( foam_wrap );
            foam_wrap = gTools().xmlengine().GetNext( foam_wrap );
         }
      }
      else {
         foam[0] = new PDEFoam("DiscrFoam");
         foam[0]->SetSignalClass    (fSignalClass);
         foam[0]->SetBackgroundClass(fBackgroundClass);
         void *foam_wrap = xmax_wrap;
         foam[0]->ReadXML( foam_wrap );
      }
   }
   
   // read pure foams from xml file
   ReadFoamsFromXMLFile();
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::ReadFoamsFromXMLFile()
{
   // read pure foams from xml file

   TString rfname( GetWeightFileName() ); 

   // replace in case of txt weight file
   rfname.ReplaceAll( TString(".") + gConfig().GetIONames().fWeightFileExtension + ".txt", ".xml" );

   // add foam indicator to distinguish from main weight file
   rfname.ReplaceAll( ".xml", "_foams.xml" );

   Log() << kINFO << "Read foams from XML file: " << rfname << Endl;
   TXMLFile *rootFile = new TXMLFile( rfname, "READ" );
   if (rootFile->IsZombie()) Log() << kFATAL << "Cannot open XML file \"" << rfname << "\"" << Endl;

   // read foams from xml file
   if (DoRegression()) {
      if (fMultiTargetRegression) 
	 foam[0] = dynamic_cast<PDEFoam*>(rootFile->Get("MultiTargetRegressionFoam"));
      else                        
	 foam[0] = dynamic_cast<PDEFoam*>(rootFile->Get("MonoTargetRegressionFoam"));
   }
   else {
      if (fSigBgSeparated) {
         foam[0] = dynamic_cast<PDEFoam*>(rootFile->Get("SignalFoam"));
         foam[1] = dynamic_cast<PDEFoam*>(rootFile->Get("BgFoam"));
      }
      else 
	 foam[0] = dynamic_cast<PDEFoam*>(rootFile->Get("DiscrFoam"));
   }
   if (!foam[0] || (!DoRegression() && fSigBgSeparated && !foam[1]))
      Log() << kFATAL << "Could not load foam!" << Endl;
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::FillVariableNamesToFoam() const {
   // fill variable names into foam
   UInt_t nfoams=1;
   if (fSigBgSeparated && !DoRegression()) nfoams=2;
   for (UInt_t ifoam=0; ifoam<nfoams; ifoam++) {
      for (Int_t idim=0; idim<foam[ifoam]->GetTotDim(); idim++) {
         if(fMultiTargetRegression && (UInt_t)idim>=DataInfo().GetNVariables())
            foam[ifoam]->AddVariableName(DataInfo().GetTargetInfo(idim-DataInfo().GetNVariables()).GetExpression().Data());
         else
            foam[ifoam]->AddVariableName(DataInfo().GetVariableInfo(idim).GetExpression().Data());
      }
   }   
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::MakeClassSpecific( std::ostream& /*fout*/, const TString& /*className*/ ) const
{
   // write PDEFoam-specific classifier response
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::GetHelpMessage() const
{
   // provide help message
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "PDE-Foam is a variation of the PDE-RS method using a self-adapting" << Endl;
   Log() << "binning method to divide the multi-dimensional variable space into a" << Endl;
   Log() << "finite number of hyper-rectangles (cells). The binning algorithm " << Endl;
   Log() << "adjusts the size and position of a predefined number of cells such" << Endl;
   Log() << "that the variance of the signal and background densities inside the " << Endl;
   Log() << "cells reaches a minimum" << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Booking options:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "The PDEFoam classifier supports two different algorithms: " << Endl;
   Log() << Endl;
   Log() << "  (1) Create one foam, which stores the signal over background" << Endl;
   Log() << "      probability density.  During foam buildup the variance of the" << Endl;
   Log() << "      discriminant inside the cells is minimised." << Endl;
   Log() << Endl;
   Log() << "      Booking option:   SigBgSeparated=F" << Endl;
   Log() << Endl;
   Log() << "  (2) Create two separate foams, one for the signal events and one for" << Endl;
   Log() << "      background events.  During foam buildup the variance of the" << Endl;
   Log() << "      event density inside the cells is minimised separately for" << Endl;
   Log() << "      signal and background." << Endl;
   Log() << Endl;
   Log() << "      Booking option:   SigBgSeparated=T" << Endl;
   Log() << Endl;
   Log() << "The following options can be set (the listed values are found to be a" << Endl;
   Log() << "good starting point for most applications):" << Endl;
   Log() << Endl;
   Log() << "        SigBgSeparate   False   Separate Signal and Background" << Endl;
   Log() << "              TailCut   0.001   Fraction of outlier events that excluded" << Endl;
   Log() << "                                from the foam in each dimension " << Endl;
   Log() << "              VolFrac  0.0333   Volume fraction (used for density calculation" << Endl;
   Log() << "                                during foam build-up) " << Endl;
   Log() << "         nActiveCells     500   Maximal number of active cells in final foam " << Endl;
   Log() << "               nSampl    2000   Number of MC events per cell in foam build-up " << Endl;
   Log() << "                 nBin       5   Number of bins used in foam build-up " << Endl;
   Log() << "              CutNmin    True   Requirement for minimal number of events in cell " << Endl;
   Log() << "                 Nmin     100   Number of events in cell required to split cell" << Endl;
   Log() << "               Kernel    None   Kernel type used (possible valuses are: None," << Endl;
   Log() << "                                Gauss)" << Endl;
   Log() << "             Compress    True   Compress XML file " << Endl;
   Log() << Endl;
   Log() << "   Additional regression options:" << Endl;
   Log() << Endl;
   Log() << "MultiTargetRegression   False   Do regression with multiple targets " << Endl;
   Log() << "      TargetSelection    Mean   Target selection method (possible valuses are: " << Endl;
   Log() << "                                Mean, Mpv)" << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "The performance of the two implementations was found to be similar for" << Endl;
   Log() << "most examples studied. For the same number of cells per foam, the two-" << Endl;
   Log() << "foam option approximately doubles the amount of computer memory needed" << Endl;
   Log() << "during classification. For special cases where the event-density" << Endl;
   Log() << "distribution of signal and background events is very different, the" << Endl;
   Log() << "two-foam option was found to perform significantly better than the" << Endl;
   Log() << "option with only one foam." << Endl;
   Log() << Endl;
   Log() << "In order to gain better classification performance we recommend to set" << Endl;
   Log() << "the parameter 'nActiveCells' to a high value." << Endl;
   Log() << Endl;
   Log() << "The parameter 'VolFrac' specifies the size of the sampling volume" << Endl;
   Log() << "during foam buildup and should be tuned in order to achieve optimal" << Endl;
   Log() << "performance.  A larger box leads to a reduced statistical uncertainty" << Endl;
   Log() << "for small training samples and to smoother sampling. A smaller box on" << Endl;
   Log() << "the other hand increases the sensitivity to statistical fluctuations" << Endl;
   Log() << "in the training samples, but for sufficiently large training samples" << Endl;
   Log() << "it will result in a more precise local estimate of the sampled" << Endl;
   Log() << "density. In general, higher dimensional problems require larger box" << Endl;
   Log() << "sizes, due to the reduced average number of events per box volume. The" << Endl;
   Log() << "default value of 0.0333 was optimised for an example with 5" << Endl;
   Log() << "observables and training samples of the order of 50000 signal and" << Endl;
   Log() << "background events each." << Endl;
   Log() << Endl;
   Log() << "Furthermore kernel weighting can be activated, which will lead to an" << Endl;
   Log() << "additional performance improvement. Note that Gauss weighting will" << Endl;
   Log() << "significantly increase the response time of the method. LinNeighbors" << Endl;
   Log() << "weighting performs a linear interpolation with direct neighbor cells" << Endl;
   Log() << "for each dimension and is much faster than Gauss weighting." << Endl;
   Log() << Endl;
   Log() << "The classification results were found to be rather insensitive to the" << Endl;
   Log() << "values of the parameters 'nSamples' and 'nBin'." << Endl;
}
