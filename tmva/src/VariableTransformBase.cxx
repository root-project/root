// @(#)root/tmva $\Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableTransformBase                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      MPI-K Heidelberg, Germany ,                                               *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/VariableTransformBase.h"
#include "TMVA/Ranking.h"
#include "TMVA/Config.h"
#include "TMVA/Tools.h"

#include "TMath.h"
#include "TVectorD.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

ClassImp(TMVA::VariableTransformBase)

//_______________________________________________________________________
TMVA::VariableTransformBase::VariableTransformBase( std::vector<VariableInfo>& varinfo, Types::EVariableTransform tf )
   : TObject(),
     fEvent( 0 ),
     fEventRaw( 0 ),
     fVariableTransform(tf),
     fEnabled( kTRUE ),
     fCreated( kFALSE ),
     fNormalize( kFALSE ),
     fTransformName("TransBase"),
     fVariables( varinfo ),
     fCurrentTree(0), 
     fCurrentEvtIdx(0),
     fOutputBaseDir(0),
     fLogger( GetName(), kINFO )
{
   // standard constructor
   std::vector<VariableInfo>::iterator it = fVariables.begin();
   for( ; it!=fVariables.end(); it++ ) (*it).ResetMinMax();
}

//_______________________________________________________________________
TMVA::VariableTransformBase::~VariableTransformBase()
{
   // destructor
   if (fEvent != fEventRaw && fEvent != 0) { delete fEvent; fEvent = 0; }
   if (fEventRaw != 0)                     { delete fEventRaw; fEventRaw = 0; }
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::ResetBranchAddresses( TTree* tree ) const
{
   // reset the trees branch addresses and have them point to the event
   tree->ResetBranchAddresses();
   fCurrentTree = 0;
   GetEventRaw().SetBranchAddresses(tree);
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::CreateEvent() const {
   // the fEvent is used to hold the event after the
   // transformation. It should not hold any connections to the
   // outside world, all its Variables shold be stored locally, and
   // the external link pointers should be 0 [creating the event from
   // the fVariables list, does not always guarantie that, so it has
   // to be explicitely ensured]
   Bool_t allowExternalLinks = kFALSE;
   fEvent = new TMVA::Event(fVariables, allowExternalLinks); 
}

//_______________________________________________________________________
Bool_t TMVA::VariableTransformBase::ReadEvent( TTree* tr, UInt_t evidx, Types::ESBType type ) const
{
   // read event from a tree into memory
   // after the reading the event transformation is called

   if (tr == 0) fLogger << kFATAL << "<ReadEvent> zero Tree Pointer encountered" << Endl;

   Bool_t needRead = kFALSE;
   if (fEventRaw == 0) {
      needRead = kTRUE;
      GetEventRaw();
      ResetBranchAddresses( tr );
   }   

   if (tr != fCurrentTree) {
      needRead = kTRUE;
      if (fCurrentTree!=0) fCurrentTree->ResetBranchAddresses();
      fCurrentTree = tr;
      ResetBranchAddresses( tr );
   }
   if (evidx != fCurrentEvtIdx) {
      needRead = kTRUE;
      fCurrentEvtIdx = evidx;
   }
   if (!needRead) return kTRUE;

   // this needs to be changed, because we don't want to affect the other branches at all
   // pass this task to the event, which should hold list of branches
   std::vector<TBranch*>::iterator brIt = fEventRaw->Branches().begin();
   for (;brIt!=fEventRaw->Branches().end(); brIt++) (*brIt)->GetEntry(evidx);

   if (type == Types::kTrueType ) type = fEventRaw->IsSignal() ? Types::kSignal : Types::kBackground;
   ApplyTransformation(type);

   return kTRUE;
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::UpdateNorm ( Int_t ivar,  Double_t x ) 
{
   // update min and max of a given variable and a given transformation method
   if (x < fVariables[ivar].GetMin()) fVariables[ivar].SetMin( x );
   if (x > fVariables[ivar].GetMax()) fVariables[ivar].SetMax( x );
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::CalcNorm( TTree * tr )
{
   // method to calculate minimum, maximum, mean, and RMS for all
   // variables used in the MVA

   if(!IsCreated()) return;

   // if PCA has not been succeeded, the tree may be empty
   if (tr == 0) return;

   ResetBranchAddresses( tr );

   UInt_t nvar = GetNVariables();

   UInt_t nevts = tr->GetEntries();

   TVectorD x2( nvar ); x2 *= 0;
   TVectorD x0( nvar ); x0 *= 0;   

   Double_t sumOfWeights = 0;
   for (UInt_t ievt=0; ievt<nevts; ievt++) {
      ReadEvent( tr, ievt, Types::kSignal );

      Double_t weight = GetEvent().GetWeight();
      sumOfWeights += weight;
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         Double_t x = GetEvent().GetVal(ivar);
         UpdateNorm( ivar,  x );
         x0(ivar) += x*weight;
         x2(ivar) += x*x*weight;
      }
   }

   // get Mean and RMS
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      Double_t mean = x0(ivar)/sumOfWeights;
      fVariables[ivar].SetMean( mean ); 
      fVariables[ivar].SetRMS( TMath::Sqrt( x2(ivar)/sumOfWeights - mean*mean) ); 
   }

   fLogger << kVERBOSE << "Set minNorm/maxNorm for variables to: " << Endl;
   fLogger << setprecision(3);
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++)
      fLogger << "    " << fVariables[ivar].GetInternalVarName()
              << "\t: [" << fVariables[ivar].GetMin() << "\t, " << fVariables[ivar].GetMax() << "\t] " << Endl;
   fLogger << setprecision(5); // reset to better value       
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::PlotVariables( TTree* theTree )
{
   // create histograms from the input variables
   // - histograms for all input variables
   // - scatter plots for all pairs of input variables

   if(!IsCreated()) return;

   // if PCA has not been succeeded, the tree may be empty
   if (theTree == 0) return;

   ResetBranchAddresses( theTree );

   // create plots of the input variables and check them
   fLogger << kVERBOSE << "Plot input variables from '" << theTree->GetName() << "'" << Endl;

   // extension for transformation type
   TString transfType = "_"; transfType += GetName();
   const UInt_t nvar = GetNVariables();

   // compute means and RMSs
   TVectorD x2S( nvar ); x2S *= 0;
   TVectorD x2B( nvar ); x2B *= 0;
   TVectorD x0S( nvar ); x0S *= 0;   
   TVectorD x0B( nvar ); x0B *= 0;      
   TVectorD rmsS( nvar ), meanS( nvar ); 
   TVectorD rmsB( nvar ), meanB( nvar ); 
   
   UInt_t nevts = (UInt_t)theTree->GetEntries();
   Double_t nS = 0, nB = 0;
   for (UInt_t ievt=0; ievt<nevts; ievt++) {

      ReadEvent( theTree, ievt, Types::kSignal );

      Double_t weight = GetEvent().GetWeight();
      if (GetEvent().IsSignal()) nS += weight; 
      else                       nB += weight;

      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         Double_t x = GetEvent().GetVal(ivar);
         if (GetEvent().IsSignal()) {
            x0S(ivar) += x*weight;
            x2S(ivar) += x*x*weight;
         }
         else {
            x0B(ivar) += x*weight;
            x2B(ivar) += x*x*weight;
         }
      }
   }
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      meanS(ivar) = x0S(ivar)/nS;
      meanB(ivar) = x0B(ivar)/nB;
      rmsS(ivar) = TMath::Sqrt( x2S(ivar)/nS - x0S(ivar)*x0S(ivar)/nS/nS );   
      rmsB(ivar) = TMath::Sqrt( x2B(ivar)/nB - x0B(ivar)*x0B(ivar)/nB/nB );   
   }

   // Create all histograms
   // do both, scatter and profile plots
   std::vector<TH1F*> vS ( nvar );
   std::vector<TH1F*> vB ( nvar );
   std::vector<std::vector<TH2F*> >     mycorrS( nvar );
   std::vector<std::vector<TH2F*> >     mycorrB( nvar );
   std::vector<std::vector<TProfile*> > myprofS( nvar );
   std::vector<std::vector<TProfile*> > myprofB( nvar );
   for (UInt_t ivar=0; ivar < nvar; ivar++) {
      mycorrS[ivar].resize(nvar);
      mycorrB[ivar].resize(nvar);
      myprofS[ivar].resize(nvar);
      myprofB[ivar].resize(nvar);
   }

   // if there are too many input variables, the creation of correlations plots blows up
   // memory and basically kills the TMVA execution
   // --> avoid above critical number (which can be user defined)
   if (nvar > (UInt_t)gConfig().fVariablePlotting.fMaxNumOfAllowedVariablesForScatterPlots) {
      Int_t nhists = nvar*(nvar - 1)/2;
      fLogger << kWARNING << "<PlotVariables> Problem with creation of scatter and profile plots. "
              << "The number of " << nvar << " input variables would require " << nhists 
              << " two-dimensional histograms, which would blow up the computer's memory. "
              << "The current critical number of input variables is set to " 
              << gConfig().fVariablePlotting.fMaxNumOfAllowedVariablesForScatterPlots
              << ", it can be modified in the user script by the call: "
              << "\"gConfig().fVariablePlotting.fMaxNumOfAllowedVariablesForScatterPlots = <some int>\"." 
              << Endl;
   }

   Float_t timesRMS  = gConfig().fVariablePlotting.fTimesRMS;
   UInt_t  nbins1D   = gConfig().fVariablePlotting.fNbins1D;
   UInt_t  nbins2D   = gConfig().fVariablePlotting.fNbins2D;
   for (UInt_t i=0; i<nvar; i++) {
      TString myVari = Variable(i).GetInternalVarName();  

      // choose reasonable histogram ranges, by removing outliers
      if (Variable(i).VarTypeOriginal() == 'I') {
         // special treatment for integer variables
         Int_t xmin = TMath::Nint( Variable(i).GetMin() );
         Int_t xmax = TMath::Nint( Variable(i).GetMax() + 1 );
         Int_t nbins = xmax - xmin;

         vS[i] = new TH1F( Form("%s__S%s", myVari.Data(), transfType.Data()), Variable(i).GetExpression(), nbins, xmin, xmax );
         vB[i] = new TH1F( Form("%s__B%s", myVari.Data(), transfType.Data()), Variable(i).GetExpression(), nbins, xmin, xmax );
      }
      else {
         Double_t xmin = TMath::Max( Variable(i).GetMin(), TMath::Min( meanS(i) - timesRMS*rmsS(i), meanB(i) - timesRMS*rmsB(i) ) );
         Double_t xmax = TMath::Min( Variable(i).GetMax(), TMath::Max( meanS(i) + timesRMS*rmsS(i), meanB(i) + timesRMS*rmsB(i) ) );

         vS[i] = new TH1F( Form("%s__S%s", myVari.Data(), transfType.Data()), Variable(i).GetExpression(), nbins1D, xmin, xmax );
         vB[i] = new TH1F( Form("%s__B%s", myVari.Data(), transfType.Data()), Variable(i).GetExpression(), nbins1D, xmin, xmax );
      }

      vS[i]->SetXTitle(Variable(i).GetExpression());
      vB[i]->SetXTitle(Variable(i).GetExpression());
      vS[i]->SetLineColor(4);
      vB[i]->SetLineColor(2);
      
      // profile and scatter plots
      if (nvar <= (UInt_t)gConfig().fVariablePlotting.fMaxNumOfAllowedVariablesForScatterPlots) {

         for (UInt_t j=i+1; j<nvar; j++) {
            TString myVarj = Variable(j).GetInternalVarName();  
            
            mycorrS[i][j] = new TH2F( Form( "scat_%s_vs_%s_sig%s", myVarj.Data(), myVari.Data(), transfType.Data() ), 
                                      Form( "%s versus %s (signal)%s", myVarj.Data(), myVari.Data(), transfType.Data() ), 
                                      nbins2D, Variable(i).GetMin(), Variable(i).GetMax(), 
                                      nbins2D, Variable(j).GetMin(), Variable(j).GetMax() );
            mycorrS[i][j]->SetXTitle(Variable(i).GetExpression());
            mycorrS[i][j]->SetYTitle(Variable(j).GetExpression());
            mycorrB[i][j] = new TH2F( Form( "scat_%s_vs_%s_bgd%s", myVarj.Data(), myVari.Data(), transfType.Data() ), 
                                      Form( "%s versus %s (background)%s", myVarj.Data(), myVari.Data(), transfType.Data() ), 
                                      nbins2D, Variable(i).GetMin(), Variable(i).GetMax(), 
                                      nbins2D, Variable(j).GetMin(), Variable(j).GetMax() );
            mycorrB[i][j]->SetXTitle(Variable(i).GetExpression());
            mycorrB[i][j]->SetYTitle(Variable(j).GetExpression());
            
            myprofS[i][j] = new TProfile( Form( "prof_%s_vs_%s_sig%s", myVarj.Data(), myVari.Data(), transfType.Data() ), 
                                          Form( "profile %s versus %s (signal)%s", myVarj.Data(), myVari.Data(), transfType.Data() ), 
                                          nbins1D, Variable(i).GetMin(), Variable(i).GetMax() );
            myprofB[i][j] = new TProfile( Form( "prof_%s_vs_%s_bgd%s", myVarj.Data(), myVari.Data(), transfType.Data() ), 
                                          Form( "profile %s versus %s (background)%s", myVarj.Data(), myVari.Data(), transfType.Data() ), 
                                          nbins1D, Variable(i).GetMin(), Variable(i).GetMax() );
         }
      }   
   }

   // fill the histograms (this approach should be faster than individual projection
   for (Int_t ievt=0; ievt<theTree->GetEntries(); ievt++) {

      ReadEvent( theTree, ievt, Types::kSignal );
      Float_t weight = GetEvent().GetWeight();

      for (UInt_t i=0; i<nvar; i++) {
         Float_t vali = GetEvent().GetVal(i);

         // variable histos
         if (GetEvent().IsSignal()) vS[i]->Fill( vali, weight );
         else                       vB[i]->Fill( vali, weight );
         
         // correlation histos
         if (nvar <= (UInt_t)gConfig().fVariablePlotting.fMaxNumOfAllowedVariablesForScatterPlots) {

            for (UInt_t j=i+1; j<nvar; j++) {
               Float_t valj = GetEvent().GetVal(j);
               if (GetEvent().IsSignal()) {
                  mycorrS[i][j]->Fill( vali, valj, weight );
                  myprofS[i][j]->Fill( vali, valj, weight );
               }
               else {
                  mycorrB[i][j]->Fill( vali, valj, weight );
                  myprofB[i][j]->Fill( vali, valj, weight );
               }
            }
         }
      }
   }
      
   // computes ranking of input variables
   fRanking = new Ranking( GetName(), "Separation" );
   for (UInt_t i=0; i<nvar; i++) {   
      Double_t sep = TMVA::Tools::GetSeparation( vS[i], vB[i] );
      fRanking->AddRank( *new Rank( vS[i]->GetTitle(), sep ) );
   }

   // write histograms

   // create directory in output file
   TString outputDir = TString("InputVariables_") + GetName();
   TObject* o = GetOutputBaseDir()->FindObject(outputDir);
   if (o != 0) {
      fLogger << kFATAL << "A " << o->ClassName() << " already exists in " 
              << GetOutputBaseDir()->GetPath() << Endl;
   }

   TDirectory* localDir = GetOutputBaseDir()->mkdir( outputDir );
   localDir->cd();
   fLogger << kVERBOSE << "Create and switch to directory " << localDir->GetPath() << Endl;
   for (UInt_t i=0; i<nvar; i++) {
      vS[i]->Write();
      vB[i]->Write();
      vS[i]->SetDirectory(0);
      vB[i]->SetDirectory(0);
   }

   // correlation plots have dedicated directory
   if (nvar <= (UInt_t)gConfig().fVariablePlotting.fMaxNumOfAllowedVariablesForScatterPlots) {

      localDir = localDir->mkdir( "CorrelationPlots" );
      localDir ->cd();
      fLogger << kINFO << "Create scatter and profile plots in target-file directory: " << Endl;
      fLogger << kINFO << localDir->GetPath() << Endl;
      
      for (UInt_t i=0; i<nvar; i++) {
         for (UInt_t j=i+1; j<nvar; j++) {
            mycorrS[i][j]->Write();
            mycorrB[i][j]->Write();
            myprofS[i][j]->Write();
            myprofB[i][j]->Write();
            mycorrS[i][j]->SetDirectory(0);
            mycorrB[i][j]->SetDirectory(0);
            myprofS[i][j]->SetDirectory(0);
            myprofB[i][j]->SetDirectory(0);
         }
      }         
   }

   GetOutputBaseDir()->cd();
   theTree->ResetBranchAddresses();
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::PrintVariableRanking() const
{
   // prints ranking of input variables
   fLogger << kINFO << "Ranking input variables..." << Endl;
   fRanking->Print();
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::WriteVarsToStream( std::ostream& o ) const 
{
   // write the list of variables (name, min, max) for a given data
   // transformation method to the stream
   o << "NVar " << GetNVariables() << endl;
   std::vector<VariableInfo>::const_iterator varIt = fVariables.begin();
   for (;varIt!=fVariables.end(); varIt++) varIt->WriteToStream(o);
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::ReadVarsFromStream( std::istream& istr ) 
{
   // Read the variables (name, min, max) for a given data
   // transformation method from the stream. In the stream we only
   // expect the limits which will be set

   TString dummy;
   UInt_t readNVar;
   istr >> dummy >> readNVar;

   if(readNVar!=fVariables.size()) {
      fLogger << kFATAL << "You declared "<< fVariables.size() << " variables in the Reader"
              << " while there are " << readNVar << " variables declared in the file"
              << Endl;
   }

   // we want to make sure all variables are read in the order they are defined
   VariableInfo varInfo;
   std::vector<VariableInfo>::iterator varIt = fVariables.begin();
   int varIdx = 0;
   for (;varIt!=fVariables.end(); varIt++, varIdx++) {
      varInfo.ReadFromStream(istr);
      if(varIt->GetExpression() == varInfo.GetExpression()) {
         varInfo.SetExternalLink((*varIt).GetExternalLink());
         (*varIt) = varInfo;
      } else {
         fLogger << kINFO << "The definition (or the order) of the variables found in the input file is"  << Endl;
         fLogger << kINFO << "is not the same as the one declared in the Reader (which is necessary for" << Endl;
         fLogger << kINFO << "the correct working of the classifier):" << Endl;
         fLogger << kINFO << "   var #" << varIdx <<" declared in Reader: " << varIt->GetExpression() << Endl;
         fLogger << kINFO << "   var #" << varIdx <<" declared in file  : " << varInfo.GetExpression() << Endl;
         fLogger << kFATAL << "The expression declared to the Reader needs to be checked (name or order are wrong)" << Endl;
      }
   }
}
