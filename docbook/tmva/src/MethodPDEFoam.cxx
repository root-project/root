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
   , fSigBgSeparated(kFALSE)
   , fFrac(0.001)
   , fDiscrErrCut(-1.0)
   , fVolFrac(30.0)
   , fVolFracInv(1.0/30.0)
   , fnCells(999)
   , fnActiveCells(500)
   , fnSampl(2000)
   , fnBin(5)
   , fEvPerBin(10000)
   , fCompress(kTRUE)
   , fMultiTargetRegression(kFALSE)
   , fNmin(100)
   , fCutNmin(kTRUE)
   , fMaxDepth(0)
   , fKernelStr("None")
   , fKernel(kNone)
   , fTargetSelectionStr("Mean")
   , fTargetSelection(kMean)
   , fFillFoamWithOrigWeights(kFALSE)
   , fUseYesNoCell(kFALSE)
   , fDTLogic("None")
   , fDTSeparation(kFoam)
   , fPeekMax(kTRUE)
   , fXmin(std::vector<Double_t>())
   , fXmax(std::vector<Double_t>())
   , fFoam(std::vector<PDEFoam*>())
{
   // init PDEFoam objects
}

//_______________________________________________________________________
TMVA::MethodPDEFoam::MethodPDEFoam( DataSetInfo& dsi,
                                    const TString& theWeightFile,
                                    TDirectory* theTargetDir ) :
   MethodBase( Types::kPDEFoam, dsi, theWeightFile, theTargetDir )
   , fSigBgSeparated(kFALSE)
   , fFrac(0.001)
   , fDiscrErrCut(-1.0)
   , fVolFrac(30.0)
   , fVolFracInv(1.0/30.0)
   , fnCells(999)
   , fnActiveCells(500)
   , fnSampl(2000)
   , fnBin(5)
   , fEvPerBin(10000)
   , fCompress(kTRUE)
   , fMultiTargetRegression(kFALSE)
   , fNmin(100)
   , fCutNmin(kTRUE)
   , fMaxDepth(0)
   , fKernelStr("None")
   , fKernel(kNone)
   , fTargetSelectionStr("Mean")
   , fTargetSelection(kMean)
   , fFillFoamWithOrigWeights(kFALSE)
   , fUseYesNoCell(kFALSE)
   , fDTLogic("None")
   , fDTSeparation(kFoam)
   , fPeekMax(kTRUE)
   , fXmin(std::vector<Double_t>())
   , fXmax(std::vector<Double_t>())
   , fFoam(std::vector<PDEFoam*>())
{
   // constructor from weight file
}

//_______________________________________________________________________
Bool_t TMVA::MethodPDEFoam::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // PDEFoam can handle classification with 2 classes and regression
   // with one or more regression-targets
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kRegression) return kTRUE;
   return kFALSE;
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::Init( void )
{
   // default initialization called by all constructors

   // init PDEFoam options
   fSigBgSeparated = kFALSE;   // default: unified foam
   fFrac           = 0.001;    // fraction of outlier events
   fDiscrErrCut    = -1.;      // cut on discriminator error
   fVolFrac        = 30.0;     // inverse range searching box size
   fVolFracInv     = 1./30.;   // range searching box size
   fnActiveCells   = 500;      // number of active cells to create
   fnCells         = fnActiveCells*2-1; // total number of cells
   fnSampl         = 2000;     // number of sampling points in cell
   fnBin           = 5;        // number of bins in edge histogram
   fEvPerBin       = 10000;    // number of events per bin
   fNmin           = 100;      // minimum number of events in cell
   fMaxDepth       = 0;        // cell tree depth (default: unlimited)
   fFillFoamWithOrigWeights = kFALSE; // fill orig. weights into foam
   fUseYesNoCell   = kFALSE;   // return -1 or 1 for bg or signal events
   fDTLogic        = "None";   // decision tree algorithmus
   fDTSeparation   = kFoam;    // separation type
   fPeekMax        = kTRUE;    // peek cell with max separation

   fKernel         = kNone; // default: use no kernel
   fTargetSelection= kMean; // default: use mean for target selection (only multi target regression!)

   fCompress              = kTRUE;  // compress ROOT output file
   fMultiTargetRegression = kFALSE; // multi-target regression

   for (UInt_t i=0; i<fFoam.size(); i++) 
      if (fFoam.at(i)) delete fFoam.at(i);
   fFoam.clear();

   if (fUseYesNoCell)
      SetSignalReferenceCut( 0.0 ); // MVA output in [-1, 1]
   else
      SetSignalReferenceCut( 0.5 ); // MVA output in [0, 1]
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
   DeclareOptionRef( fCompress = kTRUE,       "Compress", "Compress foam output file");
   DeclareOptionRef( fMultiTargetRegression = kFALSE,     "MultiTargetRegression", "Do regression with multiple targets");
   DeclareOptionRef( fNmin = 100,             "Nmin",     "Number of events in cell required to split cell");
   DeclareOptionRef( fMaxDepth = 0,           "MaxDepth",  "Maximum depth of cell tree (0=unlimited)");
   DeclareOptionRef( fFillFoamWithOrigWeights = kFALSE, "FillFoamWithOrigWeights", "Fill foam with original or boost weights");
   DeclareOptionRef( fUseYesNoCell = kFALSE, "UseYesNoCell", "Return -1 or 1 for bkg or signal like events");
   DeclareOptionRef( fDTLogic = "None", "DTLogic", "Use decision tree algorithm to split cells");
   AddPreDefVal(TString("None"));
   AddPreDefVal(TString("GiniIndex"));
   AddPreDefVal(TString("MisClassificationError"));
   AddPreDefVal(TString("CrossEntropy"));
   DeclareOptionRef( fPeekMax = kTRUE, "PeekMax", "Peek up cell with max. driver integral for the next split");

   DeclareOptionRef( fKernelStr = "None",     "Kernel",   "Kernel type used");
   AddPreDefVal(TString("None"));
   AddPreDefVal(TString("Gauss"));
   AddPreDefVal(TString("LinNeighbors"));
   DeclareOptionRef( fTargetSelectionStr = "Mean", "TargetSelection", "Target selection method");
   AddPreDefVal(TString("Mean"));
   AddPreDefVal(TString("Mpv"));
}


void TMVA::MethodPDEFoam::DeclareCompatibilityOptions() {
   MethodBase::DeclareCompatibilityOptions();
   DeclareOptionRef(fCutNmin = kTRUE, "CutNmin",  "Requirement for minimal number of events in cell");
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::ProcessOptions()
{
   // process user options
   if (!(fFrac>=0. && fFrac<=1.)) {
      Log() << kWARNING << "TailCut not in [0.,1] ==> using 0.001 instead" << Endl;
      fFrac = 0.001;
   }

   if (fnActiveCells < 1) {
      Log() << kWARNING << "invalid number of active cells specified: "
            << fnActiveCells << "; setting nActiveCells=2" << Endl;
      fnActiveCells = 2;
   }
   fnCells = fnActiveCells*2-1;

   fVolFrac = 1./fVolFracInv;

   // DT logic is only applicable if a single foam is trained
   if (fSigBgSeparated && fDTLogic != "None") {
      Log() << kWARNING << "Decision tree logic works only for a single foam (SigBgSeparate=F)" << Endl;
      fDTLogic = "None";
      fDTSeparation = kFoam;
   }

   // set separation to use
   if (fDTLogic == "None")
      fDTSeparation = kFoam;
   else if (fDTLogic == "GiniIndex")
      fDTSeparation = kGiniIndex;
   else if (fDTLogic == "MisClassificationError")
      fDTSeparation = kMisClassificationError;
   else if (fDTLogic == "CrossEntropy")
      fDTSeparation = kCrossEntropy;
   else {
      Log() << kWARNING << "Unknown separation type: " << fDTLogic 
	    << ", setting to None" << Endl;
      fDTLogic = "None";
      fDTSeparation = kFoam;
   }

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
   for (UInt_t i=0; i<fFoam.size(); i++) {
      if (fFoam.at(i)) delete fFoam.at(i);
   }
   fFoam.clear();
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::CalcXminXmax() 
{
   // Determine foam range [fXmin, fXmax] for all dimensions, such
   // that fFrac events lie outside the foam.

   fXmin.clear();
   fXmax.clear();
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
  
   // loop over all testing singnal and BG events and clac minimal and
   // maximal value of every variable
   for (Long64_t i=0; i<(GetNEvents()); i++) { // events loop
      const Event* ev = GetEvent(i);    
      for (UInt_t dim=0; dim<kDim; dim++) { // variables loop
         Double_t val;
         if (fMultiTargetRegression) {
            if (dim < vDim)
               val = ev->GetValue(dim);
            else 
               val = ev->GetTarget(dim-vDim);
         }
         else
            val = ev->GetValue(dim);

         if (val<xmin[dim])
            xmin[dim] = val;
         if (val>xmax[dim])
            xmax[dim] = val;
      }
   }

   // Create and fill histograms for each dimension (with same events
   // as before), to determine range based on number of events outside
   // the range
   TH1F **range_h = new TH1F*[kDim]; 
   for (UInt_t dim=0; dim<kDim; dim++) {
      range_h[dim]  = new TH1F(Form("range%i", dim), "range", rangehistbins, xmin[dim], xmax[dim]);
   }

   // fill all testing events into histos 
   for (Long64_t i=0; i<GetNEvents(); i++) {
      const Event* ev = GetEvent(i);
      for (UInt_t dim=0; dim<kDim; dim++) {
         if (fMultiTargetRegression) {
            if (dim < vDim)
               range_h[dim]->Fill(ev->GetValue(dim));
            else
               range_h[dim]->Fill(ev->GetTarget(dim-vDim));
         }
         else
            range_h[dim]->Fill(ev->GetValue(dim));
      }
   }

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
   fXmin.clear();
   fXmax.clear();
   for (UInt_t dim=0; dim<kDim; dim++) { 
      fXmin.push_back(xmin[dim]);
      fXmax.push_back(xmax[dim]);
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
   // Train PDE-Foam depending on the set options

   Log() << kVERBOSE << "Calculate Xmin and Xmax for every dimension" << Endl;
   CalcXminXmax();

   // delete foams
   for (UInt_t i=0; i<fFoam.size(); i++) 
      if (fFoam.at(i)) delete fFoam.at(i);
   fFoam.clear();

   // start training
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

   // check cells and delete the binary search tree in order to save
   // memory
   for(UInt_t i=0; i<fFoam.size(); i++) {
      Log() << kVERBOSE << "Check all cells and remove cells with volume 0" << Endl;
      fFoam.at(i)->CheckCells(true);
      if(fFoam.at(i)) fFoam.at(i)->DeleteBinarySearchTree();
   }
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::TrainSeparatedClassification() 
{
   // Creation of 2 separated foams: one for signal events, one for 
   // backgound events.

   TString foamcaption[2];
   foamcaption[0] = "SignalFoam";
   foamcaption[1] = "BgFoam";

   for(int i=0; i<2; i++) {
      // create 2 PDEFoams
      fFoam.push_back( new PDEFoam(foamcaption[i]) );
      InitFoam(fFoam.back(), kSeparate);

      Log() << kVERBOSE << "Filling binary search tree of " << foamcaption[i] 
            << " with events" << Endl;
      // insert event to BinarySearchTree
      for (Long64_t k=0; k<GetNEvents(); k++) {
         const Event* ev = GetEvent(k);
         if ((i==0 && DataInfo().IsSignal(ev)) || (i==1 && !DataInfo().IsSignal(ev)))
            fFoam.back()->FillBinarySearchTree(ev, IgnoreEventsWithNegWeightsInTraining());
      }

      Log() << kINFO << "Build up " << foamcaption[i] << Endl;
      fFoam.back()->Create(); // build foam

      Log() << kVERBOSE << "Filling foam cells with events" << Endl;
      // loop over all events -> fill foam cells
      for (Long64_t k=0; k<GetNEvents(); k++) {
         const Event* ev = GetEvent(k); 
         if ((i==0 && DataInfo().IsSignal(ev)) || (i==1 && !DataInfo().IsSignal(ev)))
            fFoam.back()->FillFoamCells(ev, IgnoreEventsWithNegWeightsInTraining());
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::TrainUnifiedClassification() 
{
   // Create only one unified foam which contains discriminator
   // (N_sig)/(N_sig + N_bg)

   fFoam.push_back( new PDEFoam("DiscrFoam") );
   InitFoam(fFoam.back(), kDiscr);

   Log() << kVERBOSE << "Filling binary search tree of discriminator foam with events" << Endl;
   // insert event to BinarySearchTree
   for (Long64_t k=0; k<GetNEvents(); k++)
      fFoam.back()->FillBinarySearchTree(GetEvent(k), IgnoreEventsWithNegWeightsInTraining());

   Log() << kINFO << "Build up discriminator foam" << Endl;
   fFoam.back()->Create(); // build foam

   Log() << kVERBOSE << "Filling foam cells with events" << Endl;
   // loop over all training events -> fill foam cells with N_sig and N_Bg
   for (UInt_t k=0; k<GetNEvents(); k++)
      fFoam.back()->FillFoamCells(GetEvent(k), IgnoreEventsWithNegWeightsInTraining());

   Log() << kVERBOSE << "Calculate cell discriminator"<< Endl;
   // calc discriminator (and it's error) for each cell
   fFoam.back()->CalcCellDiscr();
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::TrainMonoTargetRegression() 
{
   // Training mono target regression foam
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
   fFoam.push_back( new PDEFoam(foamcaption) );
   InitFoam(fFoam.back(), kMonoTarget);

   Log() << kVERBOSE << "Filling binary search tree with events" << Endl;
   // insert event to BinarySearchTree
   for (Long64_t k=0; k<GetNEvents(); k++)
      fFoam.back()->FillBinarySearchTree(GetEvent(k), IgnoreEventsWithNegWeightsInTraining());

   Log() << kINFO << "Build mono target regression foam" << Endl;
   fFoam.back()->Create(); // build foam

   Log() << kVERBOSE << "Filling foam cells with events" << Endl;
   // loop over all events -> fill foam cells with target
   for (UInt_t k=0; k<GetNEvents(); k++)
      fFoam.back()->FillFoamCells(GetEvent(k), IgnoreEventsWithNegWeightsInTraining());

   Log() << kVERBOSE << "Calculate average cell targets"<< Endl;
   // calc weight (and it's error) for each cell
   fFoam.back()->CalcCellTarget();
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::TrainMultiTargetRegression()
{
   // Training multi target regression foam
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
   fFoam.push_back( new PDEFoam(foamcaption) );
   InitFoam(fFoam.back(), kMultiTarget);

   Log() << kVERBOSE << "Filling binary search tree of multi target regression foam with events" 
         << Endl;
   // insert event to BinarySearchTree
   for (Long64_t k=0; k<GetNEvents(); k++)
      fFoam.back()->FillBinarySearchTree(GetEvent(k), IgnoreEventsWithNegWeightsInTraining());

   Log() << kINFO << "Build multi target regression foam" << Endl;
   fFoam.back()->Create(); // build foam

   Log() << kVERBOSE << "Filling foam cells with events" << Endl;
   // loop over all events -> fill foam cells with number of events
   for (UInt_t k=0; k<GetNEvents(); k++)
      fFoam.back()->FillFoamCells(GetEvent(k), IgnoreEventsWithNegWeightsInTraining());
}

//_______________________________________________________________________
Double_t TMVA::MethodPDEFoam::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   // Return Mva-Value.  In case of 'fSigBgSeparated==false' return
   // the cell content (D = N_sig/(N_bg+N_sig)).  In case of
   // 'fSigBgSeparated==false' return D =
   // Density_sig/(Density_sig+Density_bg).  In both cases the error
   // of the discriminant is stored in 'err'.

   const Event* ev = GetEvent();
   Double_t discr = 0.;
   Double_t discr_error = 0.;

   if (fSigBgSeparated) {
      std::vector<Float_t> xvec = ev->GetValues();

      Double_t density_sig = 0.;
      Double_t density_bg  = 0.;

      density_sig = fFoam.at(0)->GetCellDensity(xvec, fKernel); // get signal event density
      density_bg  = fFoam.at(1)->GetCellDensity(xvec, fKernel); // get background event density

      // calc disciminator (normed!)
      if ( (density_sig+density_bg) > 0 )
         discr = density_sig/(density_sig+density_bg);
      else
         discr = 0.5; // assume 50% signal probability, if no events found (bad assumption, but can be overruled by cut on error)

      // do error estimation (not jet used in TMVA)
      Double_t neventsB = fFoam.at(1)->GetCellValue(xvec, kNev);
      Double_t neventsS = fFoam.at(0)->GetCellValue(xvec, kNev);
      Double_t scaleB = 1.;
      Double_t errorS = TMath::Sqrt(neventsS); // estimation of statistical error on counted signal events
      Double_t errorB = TMath::Sqrt(neventsB); // estimation of statistical error on counted background events

      if (neventsS == 0) // no signal events in cell
         errorS = 1.;
      if (neventsB == 0) // no bg events in cell
         errorB = 1.;

      if ( (neventsS>1e-10) || (neventsB>1e-10) ) // eq. (5) in paper T.Carli, B.Koblitz 2002
         discr_error = TMath::Sqrt( Sqr ( scaleB*neventsB
                                          / Sqr(neventsS+scaleB*neventsB)
                                          * errorS) +
                                    Sqr ( scaleB*neventsS
                                          / Sqr(neventsS+scaleB*neventsB)
                                          * errorB) );
      else discr_error = 1.;

      if (discr_error < 1e-10) discr_error = 1.;
   }
   else { // Signal and Bg not separated
      std::vector<Float_t> xvec = ev->GetValues();
      
      // get discriminator direct from the foam
      discr       = fFoam.at(0)->GetCellDiscr(xvec, fKernel);
      discr_error = fFoam.at(0)->GetCellValue(xvec, kDiscriminatorError);
   }

   // attribute error
   if (err != 0) *err = discr_error;
   if (errUpper != 0) *errUpper = discr_error;

   if (fUseYesNoCell)
      return (discr < 0.5 ? -1 : 1);
   else
      return discr;
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::SetXminXmax( TMVA::PDEFoam *pdefoam )
{
   // Set Xmin, Xmax for every dimension in the given pdefoam object

   if (!pdefoam){
      Log() << kFATAL << "Null pointer given!" << Endl;
      return;
   }

   UInt_t num_vars = GetNvar();
   if (fMultiTargetRegression)
      num_vars += Data()->GetNTargets();

   for (UInt_t idim=0; idim<num_vars; idim++) { // set upper/ lower limit in foam
      Log()<< kDEBUG << "foam: SetXmin[dim="<<idim<<"]: " << fXmin.at(idim) << Endl;
      Log()<< kDEBUG << "foam: SetXmax[dim="<<idim<<"]: " << fXmax.at(idim) << Endl;
      pdefoam->SetXmin(idim, fXmin.at(idim));
      pdefoam->SetXmax(idim, fXmax.at(idim));
   }
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::InitFoam(TMVA::PDEFoam *pdefoam, EFoamType ft)
{
   // Set foam options (incl. Xmin, Xmax) and initialize foam via 
   // pdefoam->Init()

   if (!pdefoam){
      Log() << kFATAL << "Null pointer given!" << Endl;
      return;
   }

   // set fLogger attributes
   pdefoam->Log().SetMinType(this->Log().GetMinType());

   // Set foam fill value
   pdefoam->SetFoamType(ft);
   
   // set Options VolFrac, kDim, ...
   if (ft==kMultiTarget)
      // dimension of foam = number of targets + non-targets
      pdefoam->SetDim(      Data()->GetNTargets()+Data()->GetNVariables());
   else
      pdefoam->SetDim(      GetNvar());  // Mandatory!
   pdefoam->SetVolumeFraction(fVolFrac); // Mandatory!
   pdefoam->SetnCells(      fnCells);    // optional
   pdefoam->SetnSampl(      fnSampl);    // optional
   pdefoam->SetnBin(        fnBin);      // optional
   pdefoam->SetEvPerBin(    fEvPerBin);  // optional
   pdefoam->SetFillFoamWithOrigWeights(fFillFoamWithOrigWeights);
   pdefoam->SetDTSeparation(fDTSeparation);
   pdefoam->SetPeekMax(fPeekMax);

   // cuts
   pdefoam->SetNmin(fNmin);
   pdefoam->SetMaxDepth(fMaxDepth); // maximum cell tree depth

   // Init PDEFoam
   pdefoam->Init();
   
   // Set Xmin, Xmax
   SetXminXmax(pdefoam);
}

//_______________________________________________________________________
const std::vector<Float_t>& TMVA::MethodPDEFoam::GetRegressionValues()
{
   // Return regression values for both multi and mono target regression

   if (fRegressionReturnVal == 0) fRegressionReturnVal = new std::vector<Float_t>();
   fRegressionReturnVal->clear();

   const Event* ev = GetEvent();
   std::vector<Float_t> vals = ev->GetValues(); // get array of event variables (non-targets)   

   if (vals.size() == 0) {
      Log() << kWARNING << "<GetRegressionValues> value vector has size 0. " << Endl;
   }

   if (fMultiTargetRegression) {
      std::vector<Float_t> targets = fFoam.at(0)->GetProjectedRegValue(vals, fKernel, fTargetSelection);
      for(UInt_t i=0; i<(Data()->GetNTargets()); i++)
         fRegressionReturnVal->push_back(targets.at(i));
   }
   else {
      fRegressionReturnVal->push_back(fFoam.at(0)->GetCellRegValue0(vals, fKernel));   
   }

   // apply inverse transformation to regression values
   Event * evT = new Event(*ev);
   for (UInt_t itgt = 0; itgt < Data()->GetNTargets(); itgt++) {
      evT->SetTarget(itgt, fRegressionReturnVal->at(itgt) );
   }
   const Event* evT2 = GetTransformationHandler().InverseTransform( evT );
   fRegressionReturnVal->clear();
   for (UInt_t itgt = 0; itgt < Data()->GetNTargets(); itgt++) {
      fRegressionReturnVal->push_back( evT2->GetTarget(itgt) );
   }

   delete evT;

   return (*fRegressionReturnVal);
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::PrintCoefficients( void ) 
{}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::AddWeightsXMLTo( void* parent ) const 
{
   // create XML output of PDEFoam method variables

   void* wght = gTools().AddChild(parent, "Weights");
   gTools().AddAttr( wght, "SigBgSeparated",  fSigBgSeparated );
   gTools().AddAttr( wght, "Frac",            fFrac );
   gTools().AddAttr( wght, "DiscrErrCut",     fDiscrErrCut );
   gTools().AddAttr( wght, "VolFrac",         fVolFrac );
   gTools().AddAttr( wght, "nCells",          fnCells );
   gTools().AddAttr( wght, "nSampl",          fnSampl );
   gTools().AddAttr( wght, "nBin",            fnBin );
   gTools().AddAttr( wght, "EvPerBin",        fEvPerBin );
   gTools().AddAttr( wght, "Compress",        fCompress );
   gTools().AddAttr( wght, "DoRegression",    DoRegression() );
   gTools().AddAttr( wght, "CutNmin",         fNmin>0 );
   gTools().AddAttr( wght, "Nmin",            fNmin );
   gTools().AddAttr( wght, "CutRMSmin",       false );
   gTools().AddAttr( wght, "RMSmin",          0.0 );
   gTools().AddAttr( wght, "Kernel",          KernelToUInt(fKernel) );
   gTools().AddAttr( wght, "TargetSelection", TargetSelectionToUInt(fTargetSelection) );
   gTools().AddAttr( wght, "FillFoamWithOrigWeights", fFillFoamWithOrigWeights );
   gTools().AddAttr( wght, "UseYesNoCell",    fUseYesNoCell );
   
   // save foam borders Xmin[i], Xmax[i]
   void *xmin_wrap;
   for (UInt_t i=0; i<fXmin.size(); i++){
      xmin_wrap = gTools().AddChild( wght, "Xmin" );
      gTools().AddAttr( xmin_wrap, "Index", i );
      gTools().AddAttr( xmin_wrap, "Value", fXmin.at(i) );
   }
   void *xmax_wrap;
   for (UInt_t i=0; i<fXmax.size(); i++){
      xmax_wrap = gTools().AddChild( wght, "Xmax" );
      gTools().AddAttr( xmax_wrap, "Index", i );
      gTools().AddAttr( xmax_wrap, "Value", fXmax.at(i) );
   }

   // write foams to xml file
   WriteFoamsToFile();
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::WriteFoamsToFile() const 
{
   // Write pure foams to file

   // fill variable names into foam
   FillVariableNamesToFoam();   

   TString rfname( GetWeightFileName() ); 

   // replace in case of txt weight file
   rfname.ReplaceAll( TString(".") + gConfig().GetIONames().fWeightFileExtension + ".txt", ".xml" );   

   // add foam indicator to distinguish from main weight file
   rfname.ReplaceAll( ".xml", "_foams.root" );

   TFile *rootFile = 0;
   if (fCompress) rootFile = new TFile(rfname, "RECREATE", "foamfile", 9);
   else           rootFile = new TFile(rfname, "RECREATE");

   fFoam.at(0)->Write(fFoam.at(0)->GetFoamName().Data());
   if (!DoRegression() && fSigBgSeparated) 
      fFoam.at(1)->Write(fFoam.at(1)->GetFoamName().Data());
   rootFile->Close();
   Log() << kINFO << "Foams written to file: " 
         << gTools().Color("lightblue") << rfname << gTools().Color("reset") << Endl;
}

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
   istr >> fEvPerBin;                       // Maximum events (equiv.) per bin in buid-up (1000) 
   istr >> fCompress;                       // compress output file

   Bool_t regr;
   istr >> regr;                            // regression foam
   SetAnalysisType( (regr ? Types::kRegression : Types::kClassification ) );
   
   Bool_t CutNmin, CutRMSmin; // dummy for backwards compatib.
   Float_t RMSmin;            // dummy for backwards compatib.
   istr >> CutNmin;                         // cut on minimal number of events in cell
   istr >> fNmin;
   istr >> CutRMSmin;                       // cut on minimal RMS in cell
   istr >> RMSmin;

   UInt_t ker = 0;
   istr >> ker;                             // used kernel for GetMvaValue()
   fKernel = UIntToKernel(ker);

   UInt_t ts = 0;
   istr >> ts;                             // used method for target selection
   fTargetSelection = UIntToTargetSelection(ts);

   istr >> fFillFoamWithOrigWeights;        // fill foam with original event weights
   istr >> fUseYesNoCell;                   // return -1 or 1 for bg or signal event

   // clear old range and prepare new range
   fXmin.clear();
   fXmax.clear();
   UInt_t kDim = GetNvar();
   if (fMultiTargetRegression)
      kDim += Data()->GetNTargets();

   for (UInt_t i=0; i<kDim; i++) {
      fXmin.push_back(0.);
      fXmax.push_back(0.);
   }
   // read range
   for (UInt_t i=0; i<kDim; i++) 
      istr >> fXmin.at(i);
   for (UInt_t i=0; i<kDim; i++) 
      istr >> fXmax.at(i);

   // read pure foams from file
   ReadFoamsFromFile();
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
   gTools().ReadAttr( wghtnode, "EvPerBin",        fEvPerBin );
   gTools().ReadAttr( wghtnode, "Compress",        fCompress );
   Bool_t regr;
   gTools().ReadAttr( wghtnode, "DoRegression",    regr );
   SetAnalysisType( (regr ? Types::kRegression : Types::kClassification ) );
   Bool_t CutNmin; // dummy for backwards compatib.
   gTools().ReadAttr( wghtnode, "CutNmin",         CutNmin );
   gTools().ReadAttr( wghtnode, "Nmin",            fNmin );
   Bool_t CutRMSmin; // dummy for backwards compatib.
   Float_t RMSmin;   // dummy for backwards compatib.
   gTools().ReadAttr( wghtnode, "CutRMSmin",       CutRMSmin );
   gTools().ReadAttr( wghtnode, "RMSmin",          RMSmin );
   UInt_t ker = 0;
   gTools().ReadAttr( wghtnode, "Kernel",          ker );
   fKernel = UIntToKernel(ker);
   UInt_t ts = 0;
   gTools().ReadAttr( wghtnode, "TargetSelection", ts );
   fTargetSelection = UIntToTargetSelection(ts);
   if (gTools().HasAttr(wghtnode, "FillFoamWithOrigWeights"))
      gTools().ReadAttr( wghtnode, "FillFoamWithOrigWeights", fFillFoamWithOrigWeights );
   if (gTools().HasAttr(wghtnode, "UseYesNoCell"))
      gTools().ReadAttr( wghtnode, "UseYesNoCell", fUseYesNoCell );
   
   // clear old range [Xmin, Xmax] and prepare new range for reading
   fXmin.clear();
   fXmax.clear();
   UInt_t kDim = GetNvar();
   if (fMultiTargetRegression)
      kDim += Data()->GetNTargets();

   for (UInt_t i=0; i<kDim; i++) {
      fXmin.push_back(0.);
      fXmax.push_back(0.);
   }

   // read foam range
   void *xmin_wrap = gTools().GetChild( wghtnode );
   for (UInt_t counter=0; counter<kDim; counter++) {
      UInt_t i=0;
      gTools().ReadAttr( xmin_wrap , "Index", i );
      if (i>=kDim)
         Log() << kFATAL << "dimension index out of range:" << i << Endl;
      gTools().ReadAttr( xmin_wrap , "Value", fXmin.at(i) );
      xmin_wrap = gTools().GetNextChild( xmin_wrap );
   }

   void *xmax_wrap = xmin_wrap;
   for (UInt_t counter=0; counter<kDim; counter++) {
      UInt_t i=0;
      gTools().ReadAttr( xmax_wrap , "Index", i );
      if (i>=kDim)
         Log() << kFATAL << "dimension index out of range:" << i << Endl;
      gTools().ReadAttr( xmax_wrap , "Value", fXmax.at(i) );
      xmax_wrap = gTools().GetNextChild( xmax_wrap );
   }

   // if foams exist, delete them
   for (UInt_t i=0; i<fFoam.size(); i++)
      if (fFoam.at(i)) delete fFoam.at(i);
   fFoam.clear();
   
   // read pure foams from file
   ReadFoamsFromFile();
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::ReadFoamsFromFile()
{
   // read pure foams from file

   TString rfname( GetWeightFileName() ); 

   // replace in case of txt weight file
   rfname.ReplaceAll( TString(".") + gConfig().GetIONames().fWeightFileExtension + ".txt", ".xml" );

   // add foam indicator to distinguish from main weight file
   rfname.ReplaceAll( ".xml", "_foams.root" );

   Log() << kINFO << "Read foams from file: " << gTools().Color("lightblue") 
         << rfname << gTools().Color("reset") << Endl;
   TFile *rootFile = new TFile( rfname, "READ" );
   if (rootFile->IsZombie()) Log() << kFATAL << "Cannot open file \"" << rfname << "\"" << Endl;

   // read foams from file
   if (DoRegression()) {
      if (fMultiTargetRegression) 
         fFoam.push_back( dynamic_cast<PDEFoam*>(rootFile->Get("MultiTargetRegressionFoam")) );
      else                        
         fFoam.push_back( dynamic_cast<PDEFoam*>(rootFile->Get("MonoTargetRegressionFoam")) );
   }
   else {
      if (fSigBgSeparated) {
         fFoam.push_back( dynamic_cast<PDEFoam*>(rootFile->Get("SignalFoam")) );
         fFoam.push_back( dynamic_cast<PDEFoam*>(rootFile->Get("BgFoam")) );
      }
      else 
         fFoam.push_back( dynamic_cast<PDEFoam*>(rootFile->Get("DiscrFoam")) );
   }
   if (!fFoam.at(0) || (!DoRegression() && fSigBgSeparated && !fFoam.at(1)))
      Log() << kFATAL << "Could not load foam!" << Endl;
}

//_______________________________________________________________________
TMVA::EKernel TMVA::MethodPDEFoam::UIntToKernel(UInt_t iker)
{
   // convert UInt_t to EKernel (used for reading weight files)
   switch(iker) {
   case 0:  return kNone;
   case 1:  return kGaus;
   case 2:  return kLinN;
   default:
      Log() << kWARNING << "<UIntToKernel>: unknown kernel number: " << iker << Endl;
      return kNone;
   }
   return kNone;
}

//_______________________________________________________________________
TMVA::ETargetSelection TMVA::MethodPDEFoam::UIntToTargetSelection(UInt_t its)
{
   // convert UInt_t to ETargetSelection (used for reading weight files)
   switch(its) {
   case 0:  return kMean;
   case 1:  return kMpv;
   default:
      Log() << kWARNING << "<UIntToTargetSelection>: unknown method TargetSelection: " << its << Endl;
      return kMean;
   }
   return kMean;
}

//_______________________________________________________________________
void TMVA::MethodPDEFoam::FillVariableNamesToFoam() const 
{
   // fill variable names into foam(s)
   for (UInt_t ifoam=0; ifoam<fFoam.size(); ifoam++) {
      for (Int_t idim=0; idim<fFoam.at(ifoam)->GetTotDim(); idim++) {
         if(fMultiTargetRegression && (UInt_t)idim>=DataInfo().GetNVariables())
            fFoam.at(ifoam)->AddVariableName(DataInfo().GetTargetInfo(idim-DataInfo().GetNVariables()).GetExpression().Data());
         else
            fFoam.at(ifoam)->AddVariableName(DataInfo().GetVariableInfo(idim).GetExpression().Data());
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
   Log() << gTools().Color("bold") << "--- Use of booking options:" << gTools().Color("reset") << Endl;
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
   Log() << "                 Nmin     100   Number of events in cell required to split cell" << Endl;
   Log() << "               Kernel    None   Kernel type used (possible valuses are: None," << Endl;
   Log() << "                                Gauss)" << Endl;
   Log() << "             Compress    True   Compress foam output file " << Endl;
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
   Log() << "the parameter \"nActiveCells\" to a high value." << Endl;
   Log() << Endl;
   Log() << "The parameter \"VolFrac\" specifies the size of the sampling volume" << Endl;
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
   Log() << "values of the parameters \"nSamples\" and \"nBin\"." << Endl;
}
