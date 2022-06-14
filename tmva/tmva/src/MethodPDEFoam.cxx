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
 *      Alexander Voigt  - TU Dresden, Germany                                    *
 *      Peter Speckmayer - CERN, Switzerland                                      *
 *                                                                                *
 * Original author of the TFoam implementation:                                   *
 *      S. Jadach - Institute of Nuclear Physics, Cracow, Poland                  *
 *                                                                                *
 * Copyright (c) 2008, 2010:                                                      *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::MethodPDEFoam
\ingroup TMVA

The PDEFoam method is an extension of the PDERS method, which
divides the multi-dimensional phase space in a finite number of
hyper-rectangles (cells) of constant event density.  This "foam" of
cells is filled with averaged probability-density information
sampled from a training event sample.

For a given number of cells, the binning algorithm adjusts the size
and position of the cells inside the multidimensional phase space
based on a binary-split algorithm, minimizing the variance of the
event density in the cell.
The binned event density information of the final foam is stored in
binary trees, allowing for a fast and memory-efficient
classification of events.

The implementation of PDEFoam is based on the Monte-Carlo
integration package TFoam included in the analysis package ROOT.
*/

#include "TMVA/MethodPDEFoam.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Config.h"
#include "TMVA/Configurable.h"
#include "TMVA/CrossEntropy.h"
#include "TMVA/DataSet.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/Event.h"
#include "TMVA/GiniIndex.h"
#include "TMVA/GiniIndexWithLaplace.h"
#include "TMVA/IMethod.h"
#include "TMVA/MisClassificationError.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Ranking.h"
#include "TMVA/SdivSqrtSplusB.h"
#include "TMVA/SeparationBase.h"
#include "TMVA/Tools.h"
#include "TMVA/Types.h"
#include "TMVA/VariableInfo.h"

#include "TMath.h"
#include "TH1F.h"
#include "TFile.h"

REGISTER_METHOD(PDEFoam)

ClassImp(TMVA::MethodPDEFoam);

////////////////////////////////////////////////////////////////////////////////
/// init PDEFoam objects

   TMVA::MethodPDEFoam::MethodPDEFoam( const TString& jobName,
                                       const TString& methodTitle,
                                       DataSetInfo& dsi,
                                       const TString& theOption ) :
   MethodBase( jobName, Types::kPDEFoam, methodTitle, dsi, theOption)
   , fSigBgSeparated(kFALSE)
   , fFrac(0.001)
   , fDiscrErrCut(-1.0)
   , fVolFrac(1.0/15.0)
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
   , fKernelEstimator(NULL)
   , fTargetSelectionStr("Mean")
   , fTargetSelection(kMean)
   , fFillFoamWithOrigWeights(kFALSE)
   , fUseYesNoCell(kFALSE)
   , fDTLogic("None")
   , fDTSeparation(kFoam)
   , fPeekMax(kTRUE)
   , fXmin()
   , fXmax()
   , fFoam()
{
}

////////////////////////////////////////////////////////////////////////////////
/// constructor from weight file

TMVA::MethodPDEFoam::MethodPDEFoam( DataSetInfo& dsi,
                                    const TString& theWeightFile) :
   MethodBase( Types::kPDEFoam, dsi, theWeightFile)
   , fSigBgSeparated(kFALSE)
   , fFrac(0.001)
   , fDiscrErrCut(-1.0)
   , fVolFrac(1.0/15.0)
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
   , fKernelEstimator(NULL)
   , fTargetSelectionStr("Mean")
   , fTargetSelection(kMean)
   , fFillFoamWithOrigWeights(kFALSE)
   , fUseYesNoCell(kFALSE)
   , fDTLogic("None")
   , fDTSeparation(kFoam)
   , fPeekMax(kTRUE)
   , fXmin()
   , fXmax()
   , fFoam()
{
}

////////////////////////////////////////////////////////////////////////////////
/// PDEFoam can handle classification with multiple classes and regression
/// with one or more regression-targets

Bool_t TMVA::MethodPDEFoam::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kMulticlass ) return kTRUE;
   if (type == Types::kRegression) return kTRUE;
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// default initialization called by all constructors

void TMVA::MethodPDEFoam::Init( void )
{
   // init PDEFoam options
   fSigBgSeparated = kFALSE;   // default: unified foam
   fFrac           = 0.001;    // fraction of outlier events
   fDiscrErrCut    = -1.;      // cut on discriminator error
   fVolFrac        = 1./15.;   // range searching box size
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

   fKernel         = kNone; // default: use no kernel
   fKernelEstimator= NULL;  // kernel estimator used during evaluation
   fTargetSelection= kMean; // default: use mean for target selection (only multi target regression!)

   fCompress              = kTRUE;  // compress ROOT output file
   fMultiTargetRegression = kFALSE; // multi-target regression

   DeleteFoams();

   if (fUseYesNoCell)
      SetSignalReferenceCut( 0.0 ); // MVA output in [-1, 1]
   else
      SetSignalReferenceCut( 0.5 ); // MVA output in [0, 1]
}

////////////////////////////////////////////////////////////////////////////////
/// Declare MethodPDEFoam options

void TMVA::MethodPDEFoam::DeclareOptions()
{
   DeclareOptionRef( fSigBgSeparated = kFALSE, "SigBgSeparate", "Separate foams for signal and background" );
   DeclareOptionRef( fFrac = 0.001,           "TailCut",  "Fraction of outlier events that are excluded from the foam in each dimension" );
   DeclareOptionRef( fVolFrac = 1./15.,    "VolFrac",  "Size of sampling box, used for density calculation during foam build-up (maximum value: 1.0 is equivalent to volume of entire foam)");
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
   AddPreDefVal(TString("GiniIndexWithLaplace"));
   AddPreDefVal(TString("SdivSqrtSplusB"));

   DeclareOptionRef( fKernelStr = "None",     "Kernel",   "Kernel type used");
   AddPreDefVal(TString("None"));
   AddPreDefVal(TString("Gauss"));
   AddPreDefVal(TString("LinNeighbors"));
   DeclareOptionRef( fTargetSelectionStr = "Mean", "TargetSelection", "Target selection method");
   AddPreDefVal(TString("Mean"));
   AddPreDefVal(TString("Mpv"));
}


////////////////////////////////////////////////////////////////////////////////
/// options that are used ONLY for the READER to ensure backward compatibility

void TMVA::MethodPDEFoam::DeclareCompatibilityOptions() {
   MethodBase::DeclareCompatibilityOptions();
   DeclareOptionRef(fCutNmin = kTRUE, "CutNmin",  "Requirement for minimal number of events in cell");
   DeclareOptionRef(fPeekMax = kTRUE, "PeekMax",  "Peek cell with max. loss for the next split");
}

////////////////////////////////////////////////////////////////////////////////
/// process user options

void TMVA::MethodPDEFoam::ProcessOptions()
{
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

   // DT logic is only applicable if a single foam is trained
   if (fSigBgSeparated && fDTLogic != "None") {
      Log() << kFATAL << "Decision tree logic works only for a single foam (SigBgSeparate=F)" << Endl;
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
   else if (fDTLogic == "GiniIndexWithLaplace")
      fDTSeparation = kGiniIndexWithLaplace;
   else if (fDTLogic == "SdivSqrtSplusB")
      fDTSeparation = kSdivSqrtSplusB;
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
   // sanity check: number of targets > 1 and MultiTargetRegression=F
   // makes no sense --> set MultiTargetRegression=T
   if (DoRegression() && Data()->GetNTargets() > 1 && !fMultiTargetRegression) {
      Log() << kWARNING << "Warning: number of targets > 1"
            << " and MultiTargetRegression=F was set, this makes no sense!"
            << " --> I'm setting MultiTargetRegression=T" << Endl;
      fMultiTargetRegression = kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::MethodPDEFoam::~MethodPDEFoam( void )
{
   DeleteFoams();

   if (fKernelEstimator != NULL)
      delete fKernelEstimator;
}

////////////////////////////////////////////////////////////////////////////////
/// Determine foam range [fXmin, fXmax] for all dimensions, such
/// that a fraction of 'fFrac' events lie outside the foam.

void TMVA::MethodPDEFoam::CalcXminXmax()
{
   fXmin.clear();
   fXmax.clear();
   UInt_t kDim = GetNvar(); // == Data()->GetNVariables();
   UInt_t tDim = Data()->GetNTargets();
   UInt_t vDim = Data()->GetNVariables();
   if (fMultiTargetRegression)
      kDim += tDim;

   Float_t *xmin = new Float_t[kDim];
   Float_t *xmax = new Float_t[kDim];

   // set default values
   for (UInt_t dim=0; dim<kDim; dim++) {
      xmin[dim] = FLT_MAX;
      xmax[dim] = FLT_MIN;
   }

   Log() << kDEBUG << "Number of training events: " << Data()->GetNTrainingEvents() << Endl;
   Int_t nevoutside = (Int_t)((Data()->GetNTrainingEvents())*(fFrac)); // number of events that are outside the range
   Int_t rangehistbins = 10000;                               // number of bins in histos

   // loop over all testing signal and BG events and clac minimal and
   // maximal value of every variable
   for (Long64_t i=0; i<(GetNEvents()); i++) { // events loop
      const Event* ev = GetEvent(i);
      for (UInt_t dim=0; dim<kDim; dim++) { // variables loop
         Float_t val;
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

////////////////////////////////////////////////////////////////////////////////
/// Train PDE-Foam depending on the set options

void TMVA::MethodPDEFoam::Train( void )
{
   Log() << kVERBOSE << "Calculate Xmin and Xmax for every dimension" << Endl;
   CalcXminXmax();

   // delete foams
   DeleteFoams();

   // start training
   if (DoRegression()) {
      if (fMultiTargetRegression)
         TrainMultiTargetRegression();
      else
         TrainMonoTargetRegression();
   }
   else {
      if (DoMulticlass())
         TrainMultiClassification();
      else {
         if (DataInfo().GetNormalization() != "EQUALNUMEVENTS" ) {
            Log() << kHEADER << "NormMode=" << DataInfo().GetNormalization()
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

   // delete the binary search tree in order to save memory
   for(UInt_t i=0; i<fFoam.size(); i++) {
      if(fFoam.at(i))
         fFoam.at(i)->DeleteBinarySearchTree();
   }
   ExitFromTraining();
}

////////////////////////////////////////////////////////////////////////////////
/// Creation of 2 separated foams: one for signal events, one for
/// background events. At the end the foam cells of fFoam[0] will
/// contain the average number of signal events and fFoam[1] will
/// contain the average number of background events.

void TMVA::MethodPDEFoam::TrainSeparatedClassification()
{
   TString foamcaption[2];
   foamcaption[0] = "SignalFoam";
   foamcaption[1] = "BgFoam";

   for(int i=0; i<2; i++) {
      // create 2 PDEFoams
      fFoam.push_back( InitFoam(foamcaption[i], kSeparate) );

      Log() << kVERBOSE << "Filling binary search tree of " << foamcaption[i]
            << " with events" << Endl;
      // insert event to BinarySearchTree
      for (Long64_t k=0; k<GetNEvents(); ++k) {
         const Event* ev = GetEvent(k);
         if ((i==0 && DataInfo().IsSignal(ev)) || (i==1 && !DataInfo().IsSignal(ev)))
            if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
               fFoam.back()->FillBinarySearchTree(ev);
      }

      Log() << kINFO << "Build up " << foamcaption[i] << Endl;
      fFoam.back()->Create(); // build foam

      Log() << kVERBOSE << "Filling foam cells with events" << Endl;
      // loop over all events -> fill foam cells
      for (Long64_t k=0; k<GetNEvents(); ++k) {
         const Event* ev = GetEvent(k);
         Float_t weight = fFillFoamWithOrigWeights ? ev->GetOriginalWeight() : ev->GetWeight();
         if ((i==0 && DataInfo().IsSignal(ev)) || (i==1 && !DataInfo().IsSignal(ev)))
            if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
               fFoam.back()->FillFoamCells(ev, weight);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create only one unified foam (fFoam[0]) whose cells contain the
/// average discriminator (N_sig)/(N_sig + N_bg)

void TMVA::MethodPDEFoam::TrainUnifiedClassification()
{
   fFoam.push_back( InitFoam("DiscrFoam", kDiscr, fSignalClass) );

   Log() << kVERBOSE << "Filling binary search tree of discriminator foam with events" << Endl;
   // insert event to BinarySearchTree
   for (Long64_t k=0; k<GetNEvents(); ++k) {
      const Event* ev = GetEvent(k);
      if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
         fFoam.back()->FillBinarySearchTree(ev);
   }

   Log() << kINFO << "Build up discriminator foam" << Endl;
   fFoam.back()->Create(); // build foam

   Log() << kVERBOSE << "Filling foam cells with events" << Endl;
   // loop over all training events -> fill foam cells with N_sig and N_Bg
   for (Long64_t k=0; k<GetNEvents(); ++k) {
      const Event* ev = GetEvent(k);
      Float_t weight = fFillFoamWithOrigWeights ? ev->GetOriginalWeight() : ev->GetWeight();
      if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
         fFoam.back()->FillFoamCells(ev, weight);
   }

   Log() << kVERBOSE << "Calculate cell discriminator"<< Endl;
   // calc discriminator (and it's error) for each cell
   fFoam.back()->Finalize();
}

////////////////////////////////////////////////////////////////////////////////
/// Create one unified foam (see TrainUnifiedClassification()) for
/// each class, where the cells of foam i (fFoam[i]) contain the
/// average fraction of events of class i, i.e.
///
///   D = number events of class i / total number of events

void TMVA::MethodPDEFoam::TrainMultiClassification()
{
   for (UInt_t iClass=0; iClass<DataInfo().GetNClasses(); ++iClass) {

      fFoam.push_back( InitFoam(Form("MultiClassFoam%u",iClass), kMultiClass, iClass) );

      Log() << kVERBOSE << "Filling binary search tree of multiclass foam "
            << iClass << " with events" << Endl;
      // insert event to BinarySearchTree
      for (Long64_t k=0; k<GetNEvents(); ++k) {
         const Event* ev = GetEvent(k);
         if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
            fFoam.back()->FillBinarySearchTree(ev);
      }

      Log() << kINFO << "Build up multiclass foam " << iClass << Endl;
      fFoam.back()->Create(); // build foam

      Log() << kVERBOSE << "Filling foam cells with events" << Endl;
      // loop over all training events and fill foam cells with signal
      // and background events
      for (Long64_t k=0; k<GetNEvents(); ++k) {
         const Event* ev = GetEvent(k);
         Float_t weight = fFillFoamWithOrigWeights ? ev->GetOriginalWeight() : ev->GetWeight();
         if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
            fFoam.back()->FillFoamCells(ev, weight);
      }

      Log() << kVERBOSE << "Calculate cell discriminator"<< Endl;
      // calc discriminator (and it's error) for each cell
      fFoam.back()->Finalize();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Training one (mono target regression) foam, whose cells contain
/// the average 0th target.  The dimension of the foam = number of
/// non-targets (= number of variables).

void TMVA::MethodPDEFoam::TrainMonoTargetRegression()
{
   if (Data()->GetNTargets() != 1) {
      Log() << kFATAL << "Can't do mono-target regression with "
            << Data()->GetNTargets() << " targets!" << Endl;
   }

   Log() << kDEBUG << "MethodPDEFoam: number of Targets: " << Data()->GetNTargets() << Endl;

   fFoam.push_back( InitFoam("MonoTargetRegressionFoam", kMonoTarget) );

   Log() << kVERBOSE << "Filling binary search tree with events" << Endl;
   // insert event to BinarySearchTree
   for (Long64_t k=0; k<GetNEvents(); ++k) {
      const Event* ev = GetEvent(k);
      if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
         fFoam.back()->FillBinarySearchTree(ev);
   }

   Log() << kINFO << "Build mono target regression foam" << Endl;
   fFoam.back()->Create(); // build foam

   Log() << kVERBOSE << "Filling foam cells with events" << Endl;
   // loop over all events -> fill foam cells with target
   for (Long64_t k=0; k<GetNEvents(); ++k) {
      const Event* ev = GetEvent(k);
      Float_t weight = fFillFoamWithOrigWeights ? ev->GetOriginalWeight() : ev->GetWeight();
      if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
         fFoam.back()->FillFoamCells(ev, weight);
   }

   Log() << kVERBOSE << "Calculate average cell targets"<< Endl;
   // calc weight (and it's error) for each cell
   fFoam.back()->Finalize();
}

////////////////////////////////////////////////////////////////////////////////
/// Training one (multi target regression) foam, whose cells contain
/// the average event density.  The dimension of the foam = number
/// of non-targets + number of targets.

void TMVA::MethodPDEFoam::TrainMultiTargetRegression()
{
   Log() << kDEBUG << "Number of variables: " << Data()->GetNVariables() << Endl;
   Log() << kDEBUG << "Number of Targets:   " << Data()->GetNTargets()   << Endl;
   Log() << kDEBUG << "Dimension of foam:   " << Data()->GetNVariables()+Data()->GetNTargets() << Endl;
   if (fKernel==kLinN)
      Log() << kFATAL << "LinNeighbors kernel currently not supported"
            << " for multi target regression" << Endl;

   fFoam.push_back( InitFoam("MultiTargetRegressionFoam", kMultiTarget) );

   Log() << kVERBOSE << "Filling binary search tree of multi target regression foam with events"
         << Endl;
   // insert event to BinarySearchTree
   for (Long64_t k=0; k<GetNEvents(); ++k) {
      Event *ev = new Event(*GetEvent(k));
      // since in multi-target regression targets are handled like
      // variables --> remove targets and add them to the event variabels
      std::vector<Float_t> targets(ev->GetTargets());
      const UInt_t nVariables = ev->GetValues().size();
      for (UInt_t i = 0; i < targets.size(); ++i)
         ev->SetVal(i+nVariables, targets.at(i));
      ev->GetTargets().clear();
      if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
         fFoam.back()->FillBinarySearchTree(ev);
      // since the binary search tree copies the event, one can delete
      // it
      delete ev;
   }

   Log() << kINFO << "Build multi target regression foam" << Endl;
   fFoam.back()->Create(); // build foam

   Log() << kVERBOSE << "Filling foam cells with events" << Endl;
   // loop over all events -> fill foam cells with number of events
   for (Long64_t k=0; k<GetNEvents(); ++k) {
      Event *ev = new Event(*GetEvent(k));
      // since in multi-target regression targets are handled like
      // variables --> remove targets and add them to the event variabels
      std::vector<Float_t> targets = ev->GetTargets();
      const UInt_t nVariables = ev->GetValues().size();
      Float_t weight = fFillFoamWithOrigWeights ? ev->GetOriginalWeight() : ev->GetWeight();
      for (UInt_t i = 0; i < targets.size(); ++i)
         ev->SetVal(i+nVariables, targets.at(i));
      ev->GetTargets().clear();
      if (!(IgnoreEventsWithNegWeightsInTraining() && ev->GetWeight()<=0))
         fFoam.back()->FillFoamCells(ev, weight);
      // since the PDEFoam copies the event, one can delete it
      delete ev;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return Mva-Value.
///
/// In case of `fSigBgSeparated==false` (one unified PDEFoam was
/// trained) the function returns the content of the cell, which
/// corresponds to the current TMVA::Event, i.e.  D =
/// N_sig/(N_bg+N_sig).
///
/// In case of `fSigBgSeparated==true` (two separate PDEFoams were
/// trained) the function returns
///
///     D = Density_sig/(Density_sig+Density_bg)
///
/// where 'Density_sig' is the content of the cell in the signal
/// PDEFoam (fFoam[0]) and 'Density_bg' is the content of the cell
/// in the background PDEFoam (fFoam[1]).
///
/// In both cases the error on the discriminant is stored in 'err'
/// and 'errUpper'.  (Of course err and errUpper must be non-zero
/// and point to valid address to make this work.)

Double_t TMVA::MethodPDEFoam::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   const Event* ev = GetEvent();
   Double_t discr = 0.;

   if (fSigBgSeparated) {
      std::vector<Float_t> xvec = ev->GetValues();

      Double_t density_sig = 0.; // calc signal event density
      Double_t density_bg  = 0.; // calc background event density
      density_sig = fFoam.at(0)->GetCellValue(xvec, kValueDensity, fKernelEstimator);
      density_bg  = fFoam.at(1)->GetCellValue(xvec, kValueDensity, fKernelEstimator);

      // calc discriminator (normed!)
      if ( (density_sig+density_bg) > 0 )
         discr = density_sig/(density_sig+density_bg);
      else
         discr = 0.5; // assume 50% signal probability, if no events found (bad assumption, but can be overruled by cut on error)
   }
   else { // Signal and Bg not separated
      // get discriminator direct from the foam
      discr = fFoam.at(0)->GetCellValue(ev->GetValues(), kValue, fKernelEstimator);
   }

   // calculate the error
   if (err || errUpper) {
      const Double_t discr_error = CalculateMVAError();
      if (err != 0) *err = discr_error;
      if (errUpper != 0) *errUpper = discr_error;
   }

   if (fUseYesNoCell)
      return (discr < 0.5 ? -1 : 1);
   else
      return discr;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the error on the Mva value
///
/// If `fSigBgSeparated == true` the error is calculated from the
/// number of events in the signal and background PDEFoam cells.
///
/// If `fSigBgSeparated == false`, the error is taken directly from
/// the PDEFoam cell.

Double_t TMVA::MethodPDEFoam::CalculateMVAError()
{
   const Event* ev = GetEvent(); // current event
   Double_t mvaError = 0.0; // the error on the Mva value

   if (fSigBgSeparated) {
      const std::vector<Float_t>& xvec = ev->GetValues();

      const Double_t neventsB = fFoam.at(1)->GetCellValue(xvec, kValue, fKernelEstimator);
      const Double_t neventsS = fFoam.at(0)->GetCellValue(xvec, kValue, fKernelEstimator);
      const Double_t scaleB = 1.;
      // estimation of statistical error on counted signal/background events
      const Double_t errorS = neventsS == 0 ? 1.0 : TMath::Sqrt(neventsS);
      const Double_t errorB = neventsB == 0 ? 1.0 : TMath::Sqrt(neventsB);

      if ((neventsS > 1e-10) || (neventsB > 1e-10)) {
         // eq. (5) in paper T.Carli, B.Koblitz 2002
         mvaError = TMath::Sqrt(Sqr(scaleB * neventsB / Sqr(neventsS + scaleB * neventsB) * errorS) +
                                Sqr(scaleB * neventsS / Sqr(neventsS + scaleB * neventsB) * errorB));
      } else {
         mvaError = 1.0;
      }
   } else { // Signal and Bg not separated
      // get discriminator error direct from the foam
      mvaError = fFoam.at(0)->GetCellValue(ev->GetValues(), kValueError, fKernelEstimator);
   }

   return mvaError;
}

////////////////////////////////////////////////////////////////////////////////
/// Get the multiclass MVA response for the PDEFoam classifier.  The
/// returned MVA values are normalized, i.e. their sum equals 1.

const std::vector<Float_t>& TMVA::MethodPDEFoam::GetMulticlassValues()
{
   const TMVA::Event *ev = GetEvent();
   std::vector<Float_t> xvec = ev->GetValues();

   if (fMulticlassReturnVal == NULL)
      fMulticlassReturnVal = new std::vector<Float_t>();
   fMulticlassReturnVal->clear();
   fMulticlassReturnVal->reserve(DataInfo().GetNClasses());

   std::vector<Float_t> temp;  // temp class. values
   UInt_t nClasses = DataInfo().GetNClasses();
   temp.reserve(nClasses);
   for (UInt_t iClass = 0; iClass < nClasses; ++iClass) {
      temp.push_back(fFoam.at(iClass)->GetCellValue(xvec, kValue, fKernelEstimator));
   }

   for (UInt_t iClass = 0; iClass < nClasses; ++iClass) {
      Float_t norm = 0.0; // normalization
      for (UInt_t j = 0; j < nClasses; ++j) {
         if (iClass != j)
            norm += exp(temp[j] - temp[iClass]);
      }
      fMulticlassReturnVal->push_back(1.0 / (1.0 + norm));
   }

   return *fMulticlassReturnVal;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute ranking of input variables from the number of cuts made
/// in each PDEFoam dimension.  The PDEFoam dimension (the variable)
/// for which the most cuts were done is ranked highest.

const TMVA::Ranking* TMVA::MethodPDEFoam::CreateRanking()
{
   // create the ranking object
   fRanking = new Ranking(GetName(), "Variable Importance");
   std::vector<Float_t> importance(GetNvar(), 0);

   // determine variable importances
   for (UInt_t ifoam = 0; ifoam < fFoam.size(); ++ifoam) {
      // get the number of cuts made in every dimension of foam
      PDEFoamCell *root_cell = fFoam.at(ifoam)->GetRootCell();
      std::vector<UInt_t> nCuts(fFoam.at(ifoam)->GetTotDim(), 0);
      GetNCuts(root_cell, nCuts);

      // fill the importance vector (ignoring the target dimensions in
      // case of a multi-target regression foam)
      UInt_t sumOfCuts = 0;
      std::vector<Float_t> tmp_importance;
      for (UInt_t ivar = 0; ivar < GetNvar(); ++ivar) {
         sumOfCuts += nCuts.at(ivar);
         tmp_importance.push_back( nCuts.at(ivar) );
      }
      // normalization of the variable importances of this foam: the
      // sum of all variable importances equals 1 for this foam
      for (UInt_t ivar = 0; ivar < GetNvar(); ++ivar) {
         if (sumOfCuts > 0)
            tmp_importance.at(ivar) /= sumOfCuts;
         else
            tmp_importance.at(ivar) = 0;
      }
      // the overall variable importance is the average over all foams
      for (UInt_t ivar = 0; ivar < GetNvar(); ++ivar) {
         importance.at(ivar) += tmp_importance.at(ivar) / fFoam.size();
      }
   }

   // fill ranking vector
   for (UInt_t ivar = 0; ivar < GetNvar(); ++ivar) {
      fRanking->AddRank(Rank(GetInputLabel(ivar), importance.at(ivar)));
   }

   return fRanking;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill in 'nCuts' the number of cuts made in every foam dimension,
/// starting at the root cell 'cell'.
///
/// Parameters:
///
///  - cell - root cell to start the counting from
///
///  - nCuts - the number of cuts are saved in this vector

void TMVA::MethodPDEFoam::GetNCuts(PDEFoamCell *cell, std::vector<UInt_t> &nCuts)
{
   if (cell == NULL || cell->GetStat() == 1) // cell is active
      return;

   nCuts.at(cell->GetBest())++;

   if (cell->GetDau0() != NULL)
      GetNCuts(cell->GetDau0(), nCuts);
   if (cell->GetDau1() != NULL)
      GetNCuts(cell->GetDau1(), nCuts);
}

////////////////////////////////////////////////////////////////////////////////
/// Set Xmin, Xmax for every dimension in the given pdefoam object

void TMVA::MethodPDEFoam::SetXminXmax( TMVA::PDEFoam *pdefoam )
{
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

////////////////////////////////////////////////////////////////////////////////
/// Create a new PDEFoam, set the PDEFoam options (nCells, nBin,
/// Xmin, Xmax, etc.) and initialize the PDEFoam by calling
/// pdefoam->Initialize().
///
/// Parameters:
///
///  - foamcaption - name of PDEFoam object
///
///  - ft - type of PDEFoam
///
///    Candidates are:
///      - kSeparate    - creates TMVA::PDEFoamEvent
///      - kDiscr       - creates TMVA::PDEFoamDiscriminant
///      - kMonoTarget  - creates TMVA::PDEFoamTarget
///      - kMultiTarget - creates TMVA::MultiTarget
///      - kMultiClass  - creates TMVA::PDEFoamDiscriminant
///
///    If 'fDTSeparation != kFoam' then a TMVA::PDEFoamDecisionTree
///    is created (the separation type depends on fDTSeparation).
///
///  - cls - marked event class (optional, default value = 0)

TMVA::PDEFoam* TMVA::MethodPDEFoam::InitFoam(TString foamcaption, EFoamType ft, UInt_t cls)
{
   // number of foam dimensions
   Int_t dim = 1;
   if (ft == kMultiTarget)
      // dimension of foam = number of targets + non-targets
      dim = Data()->GetNTargets() + Data()->GetNVariables();
   else
      dim = GetNvar();

   // calculate range-searching box
   std::vector<Double_t> box;
   for (Int_t idim = 0; idim < dim; ++idim) {
      box.push_back((fXmax.at(idim) - fXmin.at(idim))* fVolFrac);
   }

   // create PDEFoam and PDEFoamDensityBase
   PDEFoam *pdefoam = NULL;
   PDEFoamDensityBase *density = NULL;
   if (fDTSeparation == kFoam) {
      // use PDEFoam algorithm
      switch (ft) {
      case kSeparate:
         pdefoam = new PDEFoamEvent(foamcaption);
         density = new PDEFoamEventDensity(box);
         break;
      case kMultiTarget:
         pdefoam = new PDEFoamMultiTarget(foamcaption, fTargetSelection);
         density = new PDEFoamEventDensity(box);
         break;
      case kDiscr:
      case kMultiClass:
         pdefoam = new PDEFoamDiscriminant(foamcaption, cls);
         density = new PDEFoamDiscriminantDensity(box, cls);
         break;
      case kMonoTarget:
         pdefoam = new PDEFoamTarget(foamcaption, 0);
         density = new PDEFoamTargetDensity(box, 0);
         break;
      default:
         Log() << kFATAL << "Unknown PDEFoam type!" << Endl;
         break;
      }
   } else {
      // create a decision tree like PDEFoam

      // create separation type class, which is owned by
      // PDEFoamDecisionTree (i.e. PDEFoamDecisionTree will delete it)
      SeparationBase *sepType = NULL;
      switch (fDTSeparation) {
      case kGiniIndex:
         sepType = new GiniIndex();
         break;
      case kMisClassificationError:
         sepType = new MisClassificationError();
         break;
      case kCrossEntropy:
         sepType = new CrossEntropy();
         break;
      case kGiniIndexWithLaplace:
         sepType = new GiniIndexWithLaplace();
         break;
      case kSdivSqrtSplusB:
         sepType = new SdivSqrtSplusB();
         break;
      default:
         Log() << kFATAL << "Separation type " << fDTSeparation
               << " currently not supported" << Endl;
         break;
      }
      switch (ft) {
      case kDiscr:
      case kMultiClass:
         pdefoam = new PDEFoamDecisionTree(foamcaption, sepType, cls);
         density = new PDEFoamDecisionTreeDensity(box, cls);
         break;
      default:
         Log() << kFATAL << "Decision tree cell split algorithm is only"
               << " available for (multi) classification with a single"
               << " PDE-Foam (SigBgSeparate=F)" << Endl;
         break;
      }
   }

   if (pdefoam) pdefoam->SetDensity(density);
   else Log() << kFATAL << "PDEFoam pointer not set, exiting.." << Endl;

   // create pdefoam kernel
   fKernelEstimator = CreatePDEFoamKernel();

   // set fLogger attributes
   pdefoam->Log().SetMinType(this->Log().GetMinType());

   // set PDEFoam parameters
   pdefoam->SetDim(         dim);
   pdefoam->SetnCells(      fnCells);    // optional
   pdefoam->SetnSampl(      fnSampl);    // optional
   pdefoam->SetnBin(        fnBin);      // optional
   pdefoam->SetEvPerBin(    fEvPerBin);  // optional

   // cuts
   pdefoam->SetNmin(fNmin);
   pdefoam->SetMaxDepth(fMaxDepth); // maximum cell tree depth

   // Init PDEFoam
   pdefoam->Initialize();

   // Set Xmin, Xmax
   SetXminXmax(pdefoam);

   return pdefoam;
}

////////////////////////////////////////////////////////////////////////////////
/// Return regression values for both multi- and mono-target regression

const std::vector<Float_t>& TMVA::MethodPDEFoam::GetRegressionValues()
{
   if (fRegressionReturnVal == 0) fRegressionReturnVal = new std::vector<Float_t>();
   fRegressionReturnVal->clear();
   fRegressionReturnVal->reserve(Data()->GetNTargets());

   const Event* ev = GetEvent();
   std::vector<Float_t> vals = ev->GetValues(); // get array of event variables (non-targets)

   if (vals.empty()) {
      Log() << kWARNING << "<GetRegressionValues> value vector is empty. " << Endl;
   }

   if (fMultiTargetRegression) {
      // create std::map from event variables
      std::map<Int_t, Float_t> xvec;
      for (UInt_t i=0; i<vals.size(); ++i)
         xvec.insert(std::pair<Int_t, Float_t>(i, vals.at(i)));
      // get the targets
      std::vector<Float_t> targets = fFoam.at(0)->GetCellValue( xvec, kValue );

      // sanity check
      if (targets.size() != Data()->GetNTargets())
         Log() << kFATAL << "Something wrong with multi-target regression foam: "
               << "number of targets does not match the DataSet()" << Endl;
      for(UInt_t i=0; i<targets.size(); i++)
         fRegressionReturnVal->push_back(targets.at(i));
   }
   else {
      fRegressionReturnVal->push_back(fFoam.at(0)->GetCellValue(vals, kValue, fKernelEstimator));
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

////////////////////////////////////////////////////////////////////////////////
/// create a pdefoam kernel estimator, depending on the current
/// value of fKernel

TMVA::PDEFoamKernelBase* TMVA::MethodPDEFoam::CreatePDEFoamKernel()
{
   switch (fKernel) {
   case kNone:
      return new PDEFoamKernelTrivial();
   case kLinN:
      return new PDEFoamKernelLinN();
   case kGaus:
      return new PDEFoamKernelGauss(fVolFrac/2.0);
   default:
      Log() << kFATAL << "Kernel: " << fKernel << " not supported!" << Endl;
      return NULL;
   }
   return NULL;
}

////////////////////////////////////////////////////////////////////////////////
/// Deletes all trained foams

void TMVA::MethodPDEFoam::DeleteFoams()
{
   for (UInt_t i=0; i<fFoam.size(); i++)
      if (fFoam.at(i)) delete fFoam.at(i);
   fFoam.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// reset MethodPDEFoam:
///
///  - delete all PDEFoams
///  - delete the kernel estimator

void TMVA::MethodPDEFoam::Reset()
{
   DeleteFoams();

   if (fKernelEstimator != NULL) {
      delete fKernelEstimator;
      fKernelEstimator = NULL;
   }
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::MethodPDEFoam::PrintCoefficients( void )
{}

////////////////////////////////////////////////////////////////////////////////
/// create XML output of PDEFoam method variables

void TMVA::MethodPDEFoam::AddWeightsXMLTo( void* parent ) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Write PDEFoams to file

void TMVA::MethodPDEFoam::WriteFoamsToFile() const
{
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

   // write the foams
   for (UInt_t i=0; i<fFoam.size(); ++i) {
      Log() << "writing foam " << fFoam.at(i)->GetFoamName().Data()
            << " to file" << Endl;
      fFoam.at(i)->Write(fFoam.at(i)->GetFoamName().Data());
   }

   rootFile->Close();
   Log() << kINFO << "Foams written to file: "
         << gTools().Color("lightblue") << rfname << gTools().Color("reset") << Endl;
}

////////////////////////////////////////////////////////////////////////////////
/// read options and internal parameters

void  TMVA::MethodPDEFoam::ReadWeightsFromStream( std::istream& istr )
{
   istr >> fSigBgSeparated;                 // Separate Sig and Bg, or not
   istr >> fFrac;                           // Fraction used for calc of Xmin, Xmax
   istr >> fDiscrErrCut;                    // cut on discriminant error
   istr >> fVolFrac;                        // volume fraction (used for density calculation during buildup)
   istr >> fnCells;                         // Number of Cells  (500)
   istr >> fnSampl;                         // Number of MC events per cell in build-up (1000)
   istr >> fnBin;                           // Number of bins in build-up (100)
   istr >> fEvPerBin;                       // Maximum events (equiv.) per bin in build-up (1000)
   istr >> fCompress;                       // compress output file

   Bool_t regr;
   istr >> regr;                            // regression foam
   SetAnalysisType( (regr ? Types::kRegression : Types::kClassification ) );

   Bool_t CutNmin, CutRMSmin; // dummy for backwards compatible.
   Float_t RMSmin;            // dummy for backwards compatible.
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
   fXmin.assign(kDim, 0);
   fXmax.assign(kDim, 0);

   // read range
   for (UInt_t i=0; i<kDim; i++)
      istr >> fXmin.at(i);
   for (UInt_t i=0; i<kDim; i++)
      istr >> fXmax.at(i);

   // read pure foams from file
   ReadFoamsFromFile();
}

////////////////////////////////////////////////////////////////////////////////
/// read PDEFoam variables from xml weight file

void TMVA::MethodPDEFoam::ReadWeightsFromXML( void* wghtnode )
{
   gTools().ReadAttr( wghtnode, "SigBgSeparated",  fSigBgSeparated );
   gTools().ReadAttr( wghtnode, "Frac",            fFrac );
   gTools().ReadAttr( wghtnode, "DiscrErrCut",     fDiscrErrCut );
   gTools().ReadAttr( wghtnode, "VolFrac",         fVolFrac );
   gTools().ReadAttr( wghtnode, "nCells",          fnCells );
   gTools().ReadAttr( wghtnode, "nSampl",          fnSampl );
   gTools().ReadAttr( wghtnode, "nBin",            fnBin );
   gTools().ReadAttr( wghtnode, "EvPerBin",        fEvPerBin );
   gTools().ReadAttr( wghtnode, "Compress",        fCompress );
   Bool_t regr; // dummy for backwards compatible.
   gTools().ReadAttr( wghtnode, "DoRegression",    regr );
   Bool_t CutNmin; // dummy for backwards compatible.
   gTools().ReadAttr( wghtnode, "CutNmin",         CutNmin );
   gTools().ReadAttr( wghtnode, "Nmin",            fNmin );
   Bool_t CutRMSmin; // dummy for backwards compatible.
   Float_t RMSmin;   // dummy for backwards compatible.
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
   fXmin.assign(kDim, 0);
   fXmax.assign(kDim, 0);

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
   DeleteFoams();

   // read pure foams from file
   ReadFoamsFromFile();

   // recreate the pdefoam kernel estimator
   if (fKernelEstimator != NULL)
      delete fKernelEstimator;
   fKernelEstimator = CreatePDEFoamKernel();
}

////////////////////////////////////////////////////////////////////////////////
/// Reads a foam with name 'foamname' from file, and returns a clone
/// of the foam.  The given ROOT file must be open.  (The ROOT file
/// will not be closed in this function.)
///
/// Parameters:
///
///  - file - an open ROOT file
///
///  - foamname - name of foam to load from the file
///
/// Returns:
///
/// If a foam with name 'foamname' exists in the file, then it is
/// read from the file, cloned and returned.  If a foam with name
/// 'foamname' does not exist in the file or the clone operation
/// does not succeed, then NULL is returned.

TMVA::PDEFoam* TMVA::MethodPDEFoam::ReadClonedFoamFromFile(TFile* file, const TString& foamname)
{
   if (file == NULL) {
      Log() << kWARNING << "<ReadClonedFoamFromFile>: NULL pointer given" << Endl;
      return NULL;
   }

   // try to load the foam from the file
   PDEFoam *foam = (PDEFoam*) file->Get(foamname);
   if (foam == NULL) {
      return NULL;
   }
   // try to clone the foam
   foam = (PDEFoam*) foam->Clone();
   if (foam == NULL) {
      Log() << kWARNING << "<ReadClonedFoamFromFile>: " << foamname
            << " could not be cloned!" << Endl;
      return NULL;
   }

   return foam;
}

////////////////////////////////////////////////////////////////////////////////
/// read foams from file

void TMVA::MethodPDEFoam::ReadFoamsFromFile()
{
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
         fFoam.push_back(ReadClonedFoamFromFile(rootFile, "MultiTargetRegressionFoam"));
      else
         fFoam.push_back(ReadClonedFoamFromFile(rootFile, "MonoTargetRegressionFoam"));
   } else {
      if (fSigBgSeparated) {
         fFoam.push_back(ReadClonedFoamFromFile(rootFile, "SignalFoam"));
         fFoam.push_back(ReadClonedFoamFromFile(rootFile, "BgFoam"));
      } else {
         // try to load discriminator foam
         PDEFoam *foam = ReadClonedFoamFromFile(rootFile, "DiscrFoam");
         if (foam != NULL)
            fFoam.push_back(foam);
         else {
            // load multiclass foams
            for (UInt_t iClass=0; iClass<DataInfo().GetNClasses(); ++iClass) {
               fFoam.push_back(ReadClonedFoamFromFile(rootFile, Form("MultiClassFoam%u",iClass)));
            }
         }
      }
   }

   // Close the root file.  Note, that the foams are still present in
   // memory!
   rootFile->Close();
   delete rootFile;

   for (UInt_t i=0; i<fFoam.size(); ++i) {
      if (!fFoam.at(0))
         Log() << kFATAL << "Could not load foam!" << Endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// convert UInt_t to EKernel (used for reading weight files)

TMVA::MethodPDEFoam::EKernel TMVA::MethodPDEFoam::UIntToKernel(UInt_t iker)
{
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

////////////////////////////////////////////////////////////////////////////////
/// convert UInt_t to ETargetSelection (used for reading weight files)

TMVA::ETargetSelection TMVA::MethodPDEFoam::UIntToTargetSelection(UInt_t its)
{
   switch(its) {
   case 0:  return kMean;
   case 1:  return kMpv;
   default:
      Log() << kWARNING << "<UIntToTargetSelection>: unknown method TargetSelection: " << its << Endl;
      return kMean;
   }
   return kMean;
}

////////////////////////////////////////////////////////////////////////////////
/// store the variable names in all foams

void TMVA::MethodPDEFoam::FillVariableNamesToFoam() const
{
   for (UInt_t ifoam=0; ifoam<fFoam.size(); ifoam++) {
      for (Int_t idim=0; idim<fFoam.at(ifoam)->GetTotDim(); idim++) {
         if(fMultiTargetRegression && (UInt_t)idim>=DataInfo().GetNVariables())
            fFoam.at(ifoam)->AddVariableName(DataInfo().GetTargetInfo(idim-DataInfo().GetNVariables()).GetExpression().Data());
         else
            fFoam.at(ifoam)->AddVariableName(DataInfo().GetVariableInfo(idim).GetExpression().Data());
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// write PDEFoam-specific classifier response
/// NOT IMPLEMENTED YET!

void TMVA::MethodPDEFoam::MakeClassSpecific( std::ostream& /*fout*/, const TString& /*className*/ ) const
{
}

////////////////////////////////////////////////////////////////////////////////
/// provide help message

void TMVA::MethodPDEFoam::GetHelpMessage() const
{
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
   Log() << "              VolFrac  0.0666   Volume fraction (used for density calculation" << Endl;
   Log() << "                                during foam build-up) " << Endl;
   Log() << "         nActiveCells     500   Maximal number of active cells in final foam " << Endl;
   Log() << "               nSampl    2000   Number of MC events per cell in foam build-up " << Endl;
   Log() << "                 nBin       5   Number of bins used in foam build-up " << Endl;
   Log() << "                 Nmin     100   Number of events in cell required to split cell" << Endl;
   Log() << "               Kernel    None   Kernel type used (possible values are: None," << Endl;
   Log() << "                                Gauss)" << Endl;
   Log() << "             Compress    True   Compress foam output file " << Endl;
   Log() << Endl;
   Log() << "   Additional regression options:" << Endl;
   Log() << Endl;
   Log() << "MultiTargetRegression   False   Do regression with multiple targets " << Endl;
   Log() << "      TargetSelection    Mean   Target selection method (possible values are: " << Endl;
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
   Log() << "default value of 0.0666 was optimised for an example with 5" << Endl;
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
