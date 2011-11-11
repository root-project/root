// @(#)root/tmva $Id$
// Author: Rustem Ospanov 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodKNN                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Author:                                                                        *
 *      Rustem Ospanov <rustem@fnal.gov> - U. of Texas at Austin, USA             *
 *                                                                                *
 * Copyright (c) 2007:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      U. of Texas at Austin, USA                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodKNN                                                            //
//                                                                      //
// Analysis of k-nearest neighbor                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// C/C++
#include <cmath>
#include <string>
#include <cstdlib>

// ROOT
#include "TFile.h"
#include "TMath.h"
#include "TTree.h"

// TMVA
#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodKNN.h"
#include "TMVA/Ranking.h"
#include "TMVA/Tools.h"

REGISTER_METHOD(KNN)

ClassImp(TMVA::MethodKNN)

//_______________________________________________________________________
TMVA::MethodKNN::MethodKNN( const TString& jobName,
                            const TString& methodTitle,
                            DataSetInfo& theData, 
                            const TString& theOption,
                            TDirectory* theTargetDir ) 
   : TMVA::MethodBase(jobName, Types::kKNN, methodTitle, theData, theOption, theTargetDir)
   , fSumOfWeightsS(0)
   , fSumOfWeightsB(0)
   , fModule(0)
   , fnkNN(0)
   , fBalanceDepth(0)
   , fScaleFrac(0)
   , fSigmaFact(0)
   , fTrim(kFALSE)
   , fUseKernel(kFALSE)
   , fUseWeight(kFALSE)
   , fUseLDA(kFALSE)
   , fTreeOptDepth(0)
{
   // standard constructor
}

//_______________________________________________________________________
TMVA::MethodKNN::MethodKNN( DataSetInfo& theData, 
                            const TString& theWeightFile,  
                            TDirectory* theTargetDir ) 
   : TMVA::MethodBase( Types::kKNN, theData, theWeightFile, theTargetDir)
   , fSumOfWeightsS(0)
   , fSumOfWeightsB(0)
   , fModule(0)
   , fnkNN(0)
   , fBalanceDepth(0)
   , fScaleFrac(0)
   , fSigmaFact(0)
   , fTrim(kFALSE)
   , fUseKernel(kFALSE)
   , fUseWeight(kFALSE)
   , fUseLDA(kFALSE)
   , fTreeOptDepth(0)
{
   // constructor from weight file
}

//_______________________________________________________________________
TMVA::MethodKNN::~MethodKNN()
{
   // destructor
   if (fModule) delete fModule;
}

//_______________________________________________________________________
void TMVA::MethodKNN::DeclareOptions() 
{
   // MethodKNN options
 
   // fnkNN         = 20;     // number of k-nearest neighbors 
   // fBalanceDepth = 6;      // number of binary tree levels used for tree balancing
   // fScaleFrac    = 0.8;    // fraction of events used to compute variable width
   // fSigmaFact    = 1.0;    // scale factor for Gaussian sigma 
   // fKernel       = use polynomial (1-x^3)^3 or Gaussian kernel
   // fTrim         = false;  // use equal number of signal and background events
   // fUseKernel    = false;  // use polynomial kernel weight function
   // fUseWeight    = true;   // count events using weights
   // fUseLDA       = false

   DeclareOptionRef(fnkNN         = 20,     "nkNN",         "Number of k-nearest neighbors");
   DeclareOptionRef(fBalanceDepth = 6,      "BalanceDepth", "Binary tree balance depth");
   DeclareOptionRef(fScaleFrac    = 0.80,   "ScaleFrac",    "Fraction of events used to compute variable width");
   DeclareOptionRef(fSigmaFact    = 1.0,    "SigmaFact",    "Scale factor for sigma in Gaussian kernel");
   DeclareOptionRef(fKernel       = "Gaus", "Kernel",       "Use polynomial (=Poln) or Gaussian (=Gaus) kernel");
   DeclareOptionRef(fTrim         = kFALSE, "Trim",         "Use equal number of signal and background events");
   DeclareOptionRef(fUseKernel    = kFALSE, "UseKernel",    "Use polynomial kernel weight");
   DeclareOptionRef(fUseWeight    = kTRUE,  "UseWeight",    "Use weight to count kNN events");
   DeclareOptionRef(fUseLDA       = kFALSE, "UseLDA",       "Use local linear discriminant - experimental feature");
}

//_______________________________________________________________________
void TMVA::MethodKNN::DeclareCompatibilityOptions() {
   MethodBase::DeclareCompatibilityOptions();
   DeclareOptionRef(fTreeOptDepth = 6, "TreeOptDepth", "Binary tree optimisation depth");
}

//_______________________________________________________________________
void TMVA::MethodKNN::ProcessOptions() 
{
   // process the options specified by the user
   if (!(fnkNN > 0)) {      
      fnkNN = 10;
      Log() << kWARNING << "kNN must be a positive integer: set kNN = " << fnkNN << Endl;
   }
   if (fScaleFrac < 0.0) {      
      fScaleFrac = 0.0;
      Log() << kWARNING << "ScaleFrac can not be negative: set ScaleFrac = " << fScaleFrac << Endl;
   }
   if (fScaleFrac > 1.0) {
      fScaleFrac = 1.0;
   }
   if (!(fBalanceDepth > 0)) {
      fBalanceDepth = 6;
      Log() << kWARNING << "Optimize must be a positive integer: set Optimize = " << fBalanceDepth << Endl;      
   }

   Log() << kVERBOSE
         << "kNN options: \n" 
         << "  kNN = \n" << fnkNN
         << "  UseKernel = \n" << fUseKernel
         << "  SigmaFact = \n" << fSigmaFact
         << "  ScaleFrac = \n" << fScaleFrac
         << "  Kernel = \n" << fKernel
         << "  Trim = \n" << fTrim 
         << "  Optimize = " << fBalanceDepth << Endl;
}

//_______________________________________________________________________
Bool_t TMVA::MethodKNN::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // FDA can handle classification with 2 classes and regression with one regression-target
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kRegression) return kTRUE;
   return kFALSE;
}

//_______________________________________________________________________
void TMVA::MethodKNN::Init() 
{
   // Initialization

   // fScaleFrac <= 0.0 then do not scale input variables
   // fScaleFrac >= 1.0 then use all event coordinates to scale input variables
   
   fModule = new kNN::ModulekNN();
   fSumOfWeightsS = 0;
   fSumOfWeightsB = 0;
}

//_______________________________________________________________________
void TMVA::MethodKNN::MakeKNN() 
{
   // create kNN
   if (!fModule) {
      Log() << kFATAL << "ModulekNN is not created" << Endl;
   }

   fModule->Clear();

   std::string option;
   if (fScaleFrac > 0.0) {
      option += "metric";
   }
   if (fTrim) {
      option += "trim";
   }

   Log() << kINFO << "Creating kd-tree with " << fEvent.size() << " events" << Endl;

   for (kNN::EventVec::const_iterator event = fEvent.begin(); event != fEvent.end(); ++event) {
      fModule->Add(*event);
   }

   // create binary tree
   fModule->Fill(static_cast<UInt_t>(fBalanceDepth),
                 static_cast<UInt_t>(100.0*fScaleFrac),
                 option);
}

//_______________________________________________________________________
void TMVA::MethodKNN::Train()
{
   // kNN training
   Log() << kINFO << "<Train> start..." << Endl;

   if (IsNormalised()) {
      Log() << kINFO << "Input events are normalized - setting ScaleFrac to 0" << Endl;
      fScaleFrac = 0.0;
   }
   
   if (!fEvent.empty()) {
      Log() << kINFO << "Erasing " << fEvent.size() << " previously stored events" << Endl;
      fEvent.clear();
   }
   if (GetNVariables() < 1)
      Log() << kFATAL << "MethodKNN::Train() - mismatched or wrong number of event variables" << Endl;
 

   Log() << kINFO << "Reading " << GetNEvents() << " events" << Endl;

   for (UInt_t ievt = 0; ievt < GetNEvents(); ++ievt) {
      // read the training event
      const Event*   evt_   = GetEvent(ievt);
      Double_t       weight = evt_->GetWeight();

      // in case event with neg weights are to be ignored
      if (IgnoreEventsWithNegWeightsInTraining() && weight <= 0) continue;          

      kNN::VarVec vvec(GetNVariables(), 0.0);      
      for (UInt_t ivar = 0; ivar < evt_ -> GetNVariables(); ++ivar) vvec[ivar] = evt_->GetValue(ivar);
      
      Short_t event_type = 0;

      if (DataInfo().IsSignal(evt_)) { // signal type = 1
         fSumOfWeightsS += weight;
         event_type = 1;
      }
      else { // background type = 2
         fSumOfWeightsB += weight;
         event_type = 2;
      }

      //
      // Create event and add classification variables, weight, type and regression variables
      // 
      kNN::Event event_knn(vvec, weight, event_type);
      event_knn.SetTargets(evt_->GetTargets());
      fEvent.push_back(event_knn);
      
   }
   Log() << kINFO 
         << "Number of signal events " << fSumOfWeightsS << Endl
         << "Number of background events " << fSumOfWeightsB << Endl;

   // create kd-tree (binary tree) structure
   MakeKNN();
}

//_______________________________________________________________________
Double_t TMVA::MethodKNN::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   // Compute classifier response

   // cannot determine error
   NoErrorCalc(err, errUpper);

   //
   // Define local variables
   //
   const Event *ev = GetEvent();
   const Int_t nvar = GetNVariables();
   const Double_t weight = ev->GetWeight();
   const UInt_t knn = static_cast<UInt_t>(fnkNN);

   kNN::VarVec vvec(static_cast<UInt_t>(nvar), 0.0);
   
   for (Int_t ivar = 0; ivar < nvar; ++ivar) {
      vvec[ivar] = ev->GetValue(ivar);
   }

   // search for fnkNN+2 nearest neighbors, pad with two 
   // events to avoid Monte-Carlo events with zero distance
   // most of CPU time is spent in this recursive function
   const kNN::Event event_knn(vvec, weight, 3);
   fModule->Find(event_knn, knn + 2);

   const kNN::List &rlist = fModule->GetkNNList();
   if (rlist.size() != knn + 2) {
      Log() << kFATAL << "kNN result list is empty" << Endl;
      return -100.0;  
   }
   
   if (fUseLDA) return MethodKNN::getLDAValue(rlist, event_knn);

   //
   // Set flags for kernel option=Gaus, Poln
   //
   Bool_t use_gaus = false, use_poln = false;
   if (fUseKernel) {
      if      (fKernel == "Gaus") use_gaus = true;
      else if (fKernel == "Poln") use_poln = true;
   }

   //
   // Compute radius for polynomial kernel
   //
   Double_t kradius = -1.0;
   if (use_poln) {
      kradius = MethodKNN::getKernelRadius(rlist);

      if (!(kradius > 0.0)) {
         Log() << kFATAL << "kNN radius is not positive" << Endl;
         return -100.0; 
      }
      
      kradius = 1.0/TMath::Sqrt(kradius);
   }
   
   //
   // Compute RMS of variable differences for Gaussian sigma
   //
   std::vector<Double_t> rms_vec;
   if (use_gaus) {
      rms_vec = TMVA::MethodKNN::getRMS(rlist, event_knn);

      if (rms_vec.empty() || rms_vec.size() != event_knn.GetNVar()) {
         Log() << kFATAL << "Failed to compute RMS vector" << Endl;
         return -100.0; 
      }            
   }

   UInt_t count_all = 0;
   Double_t weight_all = 0, weight_sig = 0, weight_bac = 0;

   for (kNN::List::const_iterator lit = rlist.begin(); lit != rlist.end(); ++lit) {

      // get reference to current node to make code more readable
      const kNN::Node<kNN::Event> &node = *(lit->first);
      
      // Warn about Monte-Carlo event with zero distance
      // this happens when this query event is also in learning sample
      if (lit->second < 0.0) {
         Log() << kFATAL << "A neighbor has negative distance to query event" << Endl;
      }
      else if (!(lit->second > 0.0)) {
         Log() << kVERBOSE << "A neighbor has zero distance to query event" << Endl;
      }
      
      // get event weight and scale weight by kernel function
      Double_t evweight = node.GetWeight();
      if      (use_gaus) evweight *= MethodKNN::GausKernel(event_knn, node.GetEvent(), rms_vec);
      else if (use_poln) evweight *= MethodKNN::PolnKernel(TMath::Sqrt(lit->second)*kradius);
      
      if (fUseWeight) weight_all += evweight;
      else          ++weight_all;

      if (node.GetEvent().GetType() == 1) { // signal type = 1
         if (fUseWeight) weight_sig += evweight;
         else          ++weight_sig;
      }
      else if (node.GetEvent().GetType() == 2) { // background type = 2
         if (fUseWeight) weight_bac += evweight;
         else          ++weight_bac;
      }
      else {
         Log() << kFATAL << "Unknown type for training event" << Endl;
      }
      
      // use only fnkNN events
      ++count_all;

      if (count_all >= knn) {
         break;
      }      
   }

   // check that total number of events or total weight sum is positive
   if (!(count_all > 0)) {
      Log() << kFATAL << "Size kNN result list is not positive" << Endl;
      return -100.0;
   }
   
   // check that number of events matches number of k in knn 
   if (count_all < knn) {
      Log() << kDEBUG << "count_all and kNN have different size: " << count_all << " < " << knn << Endl;
   }
   
   // Check that total weight is positive
   if (!(weight_all > 0.0)) {
      Log() << kFATAL << "kNN result total weight is not positive" << Endl;
      return -100.0;
   }
   
   return weight_sig/weight_all;
}

//_______________________________________________________________________
const std::vector< Float_t >& TMVA::MethodKNN::GetRegressionValues()
{
   //
   // Return vector of averages for target values of k-nearest neighbors.
   // Use own copy of the regression vector, I do not like using a pointer to vector.
   //
   if( fRegressionReturnVal == 0 )
      fRegressionReturnVal = new std::vector<Float_t>;
   else 
      fRegressionReturnVal->clear();

   //
   // Define local variables
   //
   const Event *evt = GetEvent();
   const Int_t nvar = GetNVariables();
   const UInt_t knn = static_cast<UInt_t>(fnkNN);
   std::vector<float> reg_vec;

   kNN::VarVec vvec(static_cast<UInt_t>(nvar), 0.0);
   
   for (Int_t ivar = 0; ivar < nvar; ++ivar) {
      vvec[ivar] = evt->GetValue(ivar);
   }   

   // search for fnkNN+2 nearest neighbors, pad with two 
   // events to avoid Monte-Carlo events with zero distance
   // most of CPU time is spent in this recursive function
   const kNN::Event event_knn(vvec, evt->GetWeight(), 3);
   fModule->Find(event_knn, knn + 2);

   const kNN::List &rlist = fModule->GetkNNList();
   if (rlist.size() != knn + 2) {
      Log() << kFATAL << "kNN result list is empty" << Endl;
      return *fRegressionReturnVal;
   }

   // compute regression values
   Double_t weight_all = 0;
   UInt_t count_all = 0;

   for (kNN::List::const_iterator lit = rlist.begin(); lit != rlist.end(); ++lit) {

      // get reference to current node to make code more readable
      const kNN::Node<kNN::Event> &node = *(lit->first);
      const kNN::VarVec &tvec = node.GetEvent().GetTargets();
      const Double_t weight = node.GetEvent().GetWeight();

      if (reg_vec.empty()) {
         reg_vec= kNN::VarVec(tvec.size(), 0.0);
      }
      
      for(UInt_t ivar = 0; ivar < tvec.size(); ++ivar) {
         if (fUseWeight) reg_vec[ivar] += tvec[ivar]*weight;
         else            reg_vec[ivar] += tvec[ivar];
      }

      if (fUseWeight) weight_all += weight;
      else          ++weight_all;

      // use only fnkNN events
      ++count_all;

      if (count_all == knn) {
         break;
      }
   }

   // check that number of events matches number of k in knn 
   if (!(weight_all > 0.0)) {
      Log() << kFATAL << "Total weight sum is not positive: " << weight_all << Endl;
      return *fRegressionReturnVal;
   }

   for (UInt_t ivar = 0; ivar < reg_vec.size(); ++ivar) {
      reg_vec[ivar] /= weight_all;
   }

   // copy result
   fRegressionReturnVal->insert(fRegressionReturnVal->begin(), reg_vec.begin(), reg_vec.end());

   return *fRegressionReturnVal;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodKNN::CreateRanking() 
{
   // no ranking available
   return 0;
}

//_______________________________________________________________________
void TMVA::MethodKNN::AddWeightsXMLTo( void* parent ) const {
   // write weights to XML

   void* wght = gTools().AddChild(parent, "Weights");
   gTools().AddAttr(wght,"NEvents",fEvent.size());
   if (fEvent.size()>0) gTools().AddAttr(wght,"NVar",fEvent.begin()->GetNVar());
   if (fEvent.size()>0) gTools().AddAttr(wght,"NTgt",fEvent.begin()->GetNTgt());

   for (kNN::EventVec::const_iterator event = fEvent.begin(); event != fEvent.end(); ++event) {

      std::stringstream s("");
      s.precision( 16 );
      for (UInt_t ivar = 0; ivar < event->GetNVar(); ++ivar) {
         if (ivar>0) s << " ";
         s << std::scientific << event->GetVar(ivar);
      }

      for (UInt_t itgt = 0; itgt < event->GetNTgt(); ++itgt) {
         s << " " << std::scientific << event->GetTgt(itgt);
      }

      void* evt = gTools().AddChild(wght, "Event", s.str().c_str());
      gTools().AddAttr(evt,"Type", event->GetType());
      gTools().AddAttr(evt,"Weight", event->GetWeight());
   }
}

//_______________________________________________________________________
void TMVA::MethodKNN::ReadWeightsFromXML( void* wghtnode ) {

   void* ch = gTools().GetChild(wghtnode); // first event
   UInt_t nvar = 0, ntgt = 0;
   gTools().ReadAttr( wghtnode, "NVar", nvar );
   gTools().ReadAttr( wghtnode, "NTgt", ntgt );


   Short_t evtType(0);
   Double_t evtWeight(0);

   while (ch) {
      // build event
      kNN::VarVec vvec(nvar, 0);
      kNN::VarVec tvec(ntgt, 0);

      gTools().ReadAttr( ch, "Type",   evtType   );
      gTools().ReadAttr( ch, "Weight", evtWeight );
      std::stringstream s( gTools().GetContent(ch) );
      
      for(UInt_t ivar=0; ivar<nvar; ivar++)
         s >> vvec[ivar];

      for(UInt_t itgt=0; itgt<ntgt; itgt++)
         s >> tvec[itgt];

      ch = gTools().GetNextChild(ch);

      kNN::Event event_knn(vvec, evtWeight, evtType, tvec);
      fEvent.push_back(event_knn);
   }

   // create kd-tree (binary tree) structure
   MakeKNN();
}

//_______________________________________________________________________
void TMVA::MethodKNN::ReadWeightsFromStream(istream& is)
{
   // read the weights
   Log() << kINFO << "Starting ReadWeightsFromStream(istream& is) function..." << Endl;

   if (!fEvent.empty()) {
      Log() << kINFO << "Erasing " << fEvent.size() << " previously stored events" << Endl;
      fEvent.clear();
   }

   UInt_t nvar = 0;

   while (!is.eof()) {
      std::string line;
      std::getline(is, line);
      
      if (line.empty() || line.find("#") != std::string::npos) {
         continue;
      }
      
      UInt_t count = 0;
      std::string::size_type pos=0;
      while( (pos=line.find(',',pos)) != std::string::npos ) { count++; pos++; }

      if (nvar == 0) {
         nvar = count - 2;
      }
      if (count < 3 || nvar != count - 2) {
         Log() << kFATAL << "Missing comma delimeter(s)" << Endl;
      }

      // Int_t ievent = -1;
      Int_t type = -1;
      Double_t weight = -1.0;
      
      kNN::VarVec vvec(nvar, 0.0);
      
      UInt_t vcount = 0;
      std::string::size_type prev = 0;
      
      for (std::string::size_type ipos = 0; ipos < line.size(); ++ipos) {
         if (line[ipos] != ',' && ipos + 1 != line.size()) {
            continue;
         }
         
         if (!(ipos > prev)) {
            Log() << kFATAL << "Wrong substring limits" << Endl;
         }
         
         std::string vstring = line.substr(prev, ipos - prev);
         if (ipos + 1 == line.size()) {
            vstring = line.substr(prev, ipos - prev + 1);
         }
         
         if (vstring.empty()) {
            Log() << kFATAL << "Failed to parse string" << Endl;
         }
         
         if (vcount == 0) {
            // ievent = std::atoi(vstring.c_str());
         }
         else if (vcount == 1) {
            type = std::atoi(vstring.c_str());
         }
         else if (vcount == 2) {
            weight = std::atof(vstring.c_str());
         }
         else if (vcount - 3 < vvec.size()) {
            vvec[vcount - 3] = std::atof(vstring.c_str());
         }
         else {
            Log() << kFATAL << "Wrong variable count" << Endl;
         }
         
         prev = ipos + 1;
         ++vcount;
      }
      
      fEvent.push_back(kNN::Event(vvec, weight, type));
   }
   
   Log() << kINFO << "Read " << fEvent.size() << " events from text file" << Endl;   

   // create kd-tree (binary tree) structure
   MakeKNN();
}

//-------------------------------------------------------------------------------------------
void TMVA::MethodKNN::WriteWeightsToStream(TFile &rf) const
{ 
   // save weights to ROOT file
   Log() << kINFO << "Starting WriteWeightsToStream(TFile &rf) function..." << Endl;
   
   if (fEvent.empty()) {
      Log() << kWARNING << "MethodKNN contains no events " << Endl;
      return;
   }

   kNN::Event *event = new kNN::Event();
   TTree *tree = new TTree("knn", "event tree");
   tree->SetDirectory(0);
   tree->Branch("event", "TMVA::kNN::Event", &event);

   Double_t size = 0.0;
   for (kNN::EventVec::const_iterator it = fEvent.begin(); it != fEvent.end(); ++it) {
      (*event) = (*it);
      size += tree->Fill();
   }

   // !!! hard coded tree name !!!
   rf.WriteTObject(tree, "knn", "Overwrite");

   // scale to MegaBytes
   size /= 1048576.0;

   Log() << kINFO << "Wrote " << size << "MB and "  << fEvent.size() 
         << " events to ROOT file" << Endl;
   
   delete tree;
   delete event; 
}

//-------------------------------------------------------------------------------------------
void TMVA::MethodKNN::ReadWeightsFromStream(TFile &rf)
{ 
   // read weights from ROOT file
   Log() << kINFO << "Starting ReadWeightsFromStream(TFile &rf) function..." << Endl;

   if (!fEvent.empty()) {
      Log() << kINFO << "Erasing " << fEvent.size() << " previously stored events" << Endl;
      fEvent.clear();
   }

   // !!! hard coded tree name !!!
   TTree *tree = dynamic_cast<TTree *>(rf.Get("knn"));
   if (!tree) {
      Log() << kFATAL << "Failed to find knn tree" << Endl;
      return;
   }

   kNN::Event *event = new kNN::Event();
   tree->SetBranchAddress("event", &event);

   const Int_t nevent = tree->GetEntries();

   Double_t size = 0.0;
   for (Int_t i = 0; i < nevent; ++i) {
      size += tree->GetEntry(i);
      fEvent.push_back(*event);
   }

   // scale to MegaBytes
   size /= 1048576.0;

   Log() << kINFO << "Read " << size << "MB and "  << fEvent.size() 
         << " events from ROOT file" << Endl;

   delete event;

   // create kd-tree (binary tree) structure
   MakeKNN();
}

//_______________________________________________________________________
void TMVA::MethodKNN::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   fout << "   // not implemented for class: \"" << className << "\"" << std::endl;
   fout << "};" << std::endl;
}

//_______________________________________________________________________
void TMVA::MethodKNN::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "The k-nearest neighbor (k-NN) algorithm is a multi-dimensional classification" << Endl
         << "and regression algorithm. Similarly to other TMVA algorithms, k-NN uses a set of" << Endl
         << "training events for which a classification category/regression target is known. " << Endl
         << "The k-NN method compares a test event to all training events using a distance " << Endl
         << "function, which is an Euclidean distance in a space defined by the input variables. "<< Endl
         << "The k-NN method, as implemented in TMVA, uses a kd-tree algorithm to perform a" << Endl
         << "quick search for the k events with shortest distance to the test event. The method" << Endl
         << "returns a fraction of signal events among the k neighbors. It is recommended" << Endl
         << "that a histogram which stores the k-NN decision variable is binned with k+1 bins" << Endl
         << "between 0 and 1." << Endl;

   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance tuning via configuration options: " 
         << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "The k-NN method estimates a density of signal and background events in a "<< Endl
         << "neighborhood around the test event. The method assumes that the density of the " << Endl
         << "signal and background events is uniform and constant within the neighborhood. " << Endl
         << "k is an adjustable parameter and it determines an average size of the " << Endl
         << "neighborhood. Small k values (less than 10) are sensitive to statistical " << Endl
         << "fluctuations and large (greater than 100) values might not sufficiently capture  " << Endl
         << "local differences between events in the training set. The speed of the k-NN" << Endl
         << "method also increases with larger values of k. " << Endl;   
   Log() << Endl;
   Log() << "The k-NN method assigns equal weight to all input variables. Different scales " << Endl
         << "among the input variables is compensated using ScaleFrac parameter: the input " << Endl
         << "variables are scaled so that the widths for central ScaleFrac*100% events are " << Endl
         << "equal among all the input variables." << Endl;

   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Additional configuration options: " 
         << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "The method inclues an option to use a Gaussian kernel to smooth out the k-NN" << Endl
         << "response. The kernel re-weights events using a distance to the test event." << Endl;
}

//_______________________________________________________________________
Double_t TMVA::MethodKNN::PolnKernel(const Double_t value) const
{
   // polynomial kernel
   const Double_t avalue = TMath::Abs(value);

   if (!(avalue < 1.0)) {
      return 0.0;
   }

   const Double_t prod = 1.0 - avalue * avalue * avalue;

   return (prod * prod * prod);
}

//_______________________________________________________________________
Double_t TMVA::MethodKNN::GausKernel(const kNN::Event &event_knn,
                                     const kNN::Event &event, const std::vector<Double_t> &svec) const
{
   // Gaussian kernel

   if (event_knn.GetNVar() != event.GetNVar() || event_knn.GetNVar() != svec.size()) {
      Log() << kFATAL << "Mismatched vectors in Gaussian kernel function" << Endl;
      return 0.0;
   }

   //
   // compute exponent
   //
   double sum_exp = 0.0;

   for(unsigned int ivar = 0; ivar < event_knn.GetNVar(); ++ivar) {

      const Double_t diff_ = event.GetVar(ivar) - event_knn.GetVar(ivar);
      const Double_t sigm_ = svec[ivar];
      if (!(sigm_ > 0.0)) {
         Log() << kFATAL << "Bad sigma value = " << sigm_ << Endl;
         return 0.0;
      }

      sum_exp += diff_*diff_/(2.0*sigm_*sigm_);
   }

   //
   // Return unnormalized(!) Gaussian function, because normalization
   // cancels for the ratio of weights.
   //

   return std::exp(-sum_exp);
}

//_______________________________________________________________________
Double_t TMVA::MethodKNN::getKernelRadius(const kNN::List &rlist) const
{
   //
   // Get polynomial kernel radius
   //
   Double_t kradius = -1.0;
   UInt_t kcount = 0;
   const UInt_t knn = static_cast<UInt_t>(fnkNN);

   for (kNN::List::const_iterator lit = rlist.begin(); lit != rlist.end(); ++lit)
      {
         if (!(lit->second > 0.0)) continue;         
      
         if (kradius < lit->second || kradius < 0.0) kradius = lit->second;
      
         ++kcount;
         if (kcount >= knn) break;
      }
   
   return kradius;
}

//_______________________________________________________________________
const std::vector<Double_t> TMVA::MethodKNN::getRMS(const kNN::List &rlist, const kNN::Event &event_knn) const
{
   //
   // Get polynomial kernel radius
   //
   std::vector<Double_t> rvec;
   UInt_t kcount = 0;
   const UInt_t knn = static_cast<UInt_t>(fnkNN);

   for (kNN::List::const_iterator lit = rlist.begin(); lit != rlist.end(); ++lit)
      {
         if (!(lit->second > 0.0)) continue;         
      
         const kNN::Node<kNN::Event> *node_ = lit -> first;
         const kNN::Event &event_ = node_-> GetEvent();
      
         if (rvec.empty()) {
            rvec.insert(rvec.end(), event_.GetNVar(), 0.0);
         }
         else if (rvec.size() != event_.GetNVar()) {
            Log() << kFATAL << "Wrong number of variables, should never happen!" << Endl;
            rvec.clear();
            return rvec;
         }

         for(unsigned int ivar = 0; ivar < event_.GetNVar(); ++ivar) {
            const Double_t diff_ = event_.GetVar(ivar) - event_knn.GetVar(ivar);
            rvec[ivar] += diff_*diff_;
         }

         ++kcount;
         if (kcount >= knn) break;
      }

   if (kcount < 1) {
      Log() << kFATAL << "Bad event kcount = " << kcount << Endl;
      rvec.clear();
      return rvec;
   }

   for(unsigned int ivar = 0; ivar < rvec.size(); ++ivar) {
      if (!(rvec[ivar] > 0.0)) {
         Log() << kFATAL << "Bad RMS value = " << rvec[ivar] << Endl;
         rvec.clear();
         return rvec;
      }

      rvec[ivar] = std::abs(fSigmaFact)*std::sqrt(rvec[ivar]/kcount);
   }   
   
   return rvec;
}

//_______________________________________________________________________
Double_t TMVA::MethodKNN::getLDAValue(const kNN::List &rlist, const kNN::Event &event_knn)
{
   LDAEvents sig_vec, bac_vec;

   for (kNN::List::const_iterator lit = rlist.begin(); lit != rlist.end(); ++lit) {
       
      // get reference to current node to make code more readable
      const kNN::Node<kNN::Event> &node = *(lit->first);
      const kNN::VarVec &tvec = node.GetEvent().GetVars();

      if (node.GetEvent().GetType() == 1) { // signal type = 1
         sig_vec.push_back(tvec);
      }
      else if (node.GetEvent().GetType() == 2) { // background type = 2
         bac_vec.push_back(tvec);
      }
      else {
         Log() << kFATAL << "Unknown type for training event" << Endl;
      }       
   }

   fLDA.Initialize(sig_vec, bac_vec);
    
   return fLDA.GetProb(event_knn.GetVars(), 1);
}
