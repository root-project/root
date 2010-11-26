// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Eckhard v. Toerne

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableGaussTransform                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>  - Uni Bonn, Germany              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Gaussian Transformation of input variables.                           //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <list>
#include <limits>

#include "TVectorF.h"
#include "TVectorD.h"
#include "TMath.h"
#include "TCanvas.h"

#include "TMVA/VariableGaussTransform.h"
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_Version
#include "TMVA/Version.h"
#endif

ClassImp(TMVA::VariableGaussTransform)

//_______________________________________________________________________
TMVA::VariableGaussTransform::VariableGaussTransform( DataSetInfo& dsi, TString strcor )
   : VariableTransformBase( dsi, Types::kGauss, "Gauss" ),
     fFlatNotGauss(kFALSE),
     fPdfMinSmooth(0),
     fPdfMaxSmooth(0),
     fElementsperbin(0)
{ 
   // constructor
   // can only be applied one after the other when they are created. But in order to
   // determine the Gauss transformation
  if (strcor=="Uniform") {fFlatNotGauss = kTRUE;
    SetName("Uniform");
  }
}

//_______________________________________________________________________
TMVA::VariableGaussTransform::~VariableGaussTransform( void )
{
   // destructor
   CleanUpCumulativeArrays();
}

//_______________________________________________________________________
void TMVA::VariableGaussTransform::Initialize()
{

}

//_______________________________________________________________________
Bool_t TMVA::VariableGaussTransform::PrepareTransformation( const std::vector<Event*>& events )
{
   // calculate the cumulative distributions
   Initialize();

   if (!IsEnabled() || IsCreated()) return kTRUE;

   Log() << kINFO << "Preparing the Gaussian transformation..." << Endl;

   SetNVariables(events[0]->GetNVariables());

   if (GetNVariables() > 200) { 
      Log() << kWARNING << "----------------------------------------------------------------------------" 
              << Endl;
      Log() << kWARNING 
              << ": More than 200 variables, I hope you have enough memory!!!!" << Endl;
      Log() << kWARNING << "----------------------------------------------------------------------------" 
              << Endl;
      //      return kFALSE;
   }   

   GetCumulativeDist( events );
   
   SetCreated( kTRUE );

   return kTRUE;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableGaussTransform::Transform(const Event* const ev, Int_t cls ) const
{
   // apply the Gauss transformation

   if (!IsCreated()) Log() << kFATAL << "Transformation not yet created" << Endl;
   //EVT this is a workaround to address the reader problem with transforma and EvaluateMVA(std::vector<float/double> ,...) 
   //EVT if (cls <0 || cls > GetNClasses() ) {
   //EVT   cls = GetNClasses();
   //EVT   if (GetNClasses() == 1 ) cls = (fCumulativePDF[0].size()==1?0:2);
   //EVT}
   if (cls <0 || cls >=  (int) fCumulativePDF[0].size()) cls = fCumulativePDF[0].size()-1;
   //EVT workaround end
 
  // get the variable vector of the current event
   const UInt_t nvar = GetNVariables();
   TVectorD vec( nvar );
   for (UInt_t ivar=0; ivar<nvar; ivar++) vec(ivar) = ev->GetValue(ivar);
   Double_t cumulant;
   //transformation   
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      if (0 != fCumulativePDF[ivar][cls]) { 
         // first make it flat
         if(fTMVAVersion>TMVA_VERSION(3,9,7))
            cumulant = (fCumulativePDF[ivar][cls])->GetVal(vec(ivar)); 
         else
            cumulant = OldCumulant(vec(ivar), fCumulativePDF[ivar][cls]->GetOriginalHist() );
         cumulant = TMath::Min(cumulant,1.-10e-10);
         cumulant = TMath::Max(cumulant,0.+10e-10);

         if (fFlatNotGauss)
            vec(ivar) = cumulant; 
         else {
            // sanity correction for out-of-range values
            Double_t maxErfInvArgRange = 0.99999999;
            Double_t arg = 2.0*cumulant - 1.0;
            arg = TMath::Min(+maxErfInvArgRange,arg);
            arg = TMath::Max(-maxErfInvArgRange,arg);
            
            vec(ivar) = 1.414213562*TMath::ErfInverse(arg);
         }
      }
   }
   
   if (fTransformedEvent==0 || fTransformedEvent->GetNVariables()!=ev->GetNVariables()) {
      if (fTransformedEvent!=0) { delete fTransformedEvent; fTransformedEvent = 0; }
      fTransformedEvent = new Event();
   }

   for (UInt_t itgt = 0; itgt < ev->GetNTargets(); itgt++) fTransformedEvent->SetTarget( itgt, ev->GetTarget(itgt) );
   for (UInt_t ivar=0; ivar<nvar; ivar++)                  fTransformedEvent->SetVal   ( ivar, vec(ivar) );

   fTransformedEvent->SetWeight     ( ev->GetWeight() );
   fTransformedEvent->SetBoostWeight( ev->GetBoostWeight() );
   fTransformedEvent->SetClass      ( ev->GetClass() );

   return fTransformedEvent;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableGaussTransform::InverseTransform( const Event* const ev, Int_t cls ) const
{
   // apply the Gauss transformation
   // TODO: implementation of inverse transformation
   Log() << kFATAL << "Inverse transformation for Gauss transformation not yet implemented. Hence, this transformation cannot be applied together with regression. Please contact the authors if necessary." << Endl;

   if (!IsCreated())
      Log() << kFATAL << "Transformation not yet created" 
              << Endl;

   if (cls <0 || cls >= GetNClasses() ) cls = GetNClasses();
   if (GetNClasses() == 1 ) cls = 0;

   // get the variable vector of the current event
   const UInt_t nvar = GetNVariables();
   TVectorD vec( nvar );
   for (UInt_t ivar=0; ivar<nvar; ivar++) vec(ivar) = ev->GetValue(ivar);
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      if (0 != fCumulativeDist[ivar][cls]) { 
         // first make it flat
         Int_t    thebin   = (fCumulativeDist[ivar][cls])->FindBin(vec(ivar));
         Double_t cumulant = (fCumulativeDist[ivar][cls])->GetBinContent(thebin);
         // now transfor to a gaussian
         // actually, the "sqrt(2) is not really necessary, who cares if it is totally normalised
         //            vec(ivar) =  TMath::Sqrt(2.)*TMath::ErfInverse(2*sum - 1);
         if (fFlatNotGauss) vec(ivar) = cumulant; 
         else {
            // sanity correction for out-of-range values
            Double_t maxErfInvArgRange = 0.99999999;
            Double_t arg = 2.0*cumulant - 1.0;
            arg = TMath::Min(+maxErfInvArgRange,arg);
            arg = TMath::Max(-maxErfInvArgRange,arg);
            
            vec(ivar) = 1.414213562*TMath::ErfInverse(arg);
         }
      }
   }
   
   if (fBackTransformedEvent==0 || fBackTransformedEvent->GetNVariables()!=ev->GetNVariables()) {
      if (fBackTransformedEvent!=0) { delete fBackTransformedEvent; fBackTransformedEvent = 0; }
      fBackTransformedEvent = new Event( *ev );
   }

   for (UInt_t itgt = 0; itgt < ev->GetNTargets(); itgt++) fBackTransformedEvent->SetTarget( itgt, ev->GetTarget(itgt) );
   for (UInt_t ivar=0; ivar<nvar; ivar++)                  fBackTransformedEvent->SetVal   ( ivar, vec(ivar) );

   fBackTransformedEvent->SetWeight     ( ev->GetWeight() );
   fBackTransformedEvent->SetBoostWeight( ev->GetBoostWeight() );
   fBackTransformedEvent->SetClass      ( ev->GetClass() );

   return fBackTransformedEvent;
}

//_______________________________________________________________________
void TMVA::VariableGaussTransform::GetCumulativeDist( const std::vector<Event*>& events )
{
   // fill the cumulative distributions

   const UInt_t nvar = GetNVariables();
   UInt_t nevt = events.size();
   
   const UInt_t nClasses = GetNClasses();
   UInt_t numDist  = nClasses+1; // calculate cumulative distributions for all "event" classes seperately + one where all classes are treated (added) together
      
   if (GetNClasses() == 1 ) numDist = nClasses; // for regression, if there is only one class, there is no "sum" of classes, hence 
 
   UInt_t **nbins = new UInt_t*[numDist];

   std::list< TMVA::TMVAGaussPair >  **listsForBinning = new std::list<TMVA::TMVAGaussPair>* [numDist];
   std::vector< Float_t >   **vsForBinning = new std::vector<Float_t>* [numDist];
   for (UInt_t i=0; i < numDist; i++) {
      listsForBinning[i] = new std::list<TMVA::TMVAGaussPair> [nvar];
      vsForBinning[i]    = new std::vector<Float_t> [nvar];
      nbins[i] = new UInt_t[nvar];  // nbins[0] = number of bins for signal distributions. It depends on the number of entries, thus it's the same for all the input variables, but it isn't necessary for some "weird" reason.
   }


   // perform event loop
   Float_t *sumOfWeights = new Float_t[numDist]; 
   Float_t *minWeight = new Float_t[numDist]; 
   Float_t *maxWeight = new Float_t[numDist]; 
   for (UInt_t i=0; i<numDist; i++) {
      sumOfWeights[i]=0;
      minWeight[i]=10E10;
      maxWeight[i]=0;
   }
   for (UInt_t ievt=0; ievt < nevt; ievt++) {
      const Event* ev= events[ievt];
      Int_t cls = ev->GetClass();
      sumOfWeights[cls] += ev->GetWeight();
      if (minWeight[cls] > ev->GetWeight()) minWeight[cls]=ev->GetWeight();
      if (maxWeight[cls] < ev->GetWeight()) maxWeight[cls]=ev->GetWeight();
      if (numDist>1) sumOfWeights[numDist-1] += ev->GetWeight();
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         listsForBinning[cls][ivar].push_back(TMVA::TMVAGaussPair(ev->GetValue(ivar),ev->GetWeight()));  
         if (numDist>1)listsForBinning[numDist-1][ivar].push_back(TMVA::TMVAGaussPair(ev->GetValue(ivar),ev->GetWeight()));  
      }  
   }
   if (numDist > 1) {
      for (UInt_t icl=0; icl<numDist-1; icl++){
         minWeight[numDist-1] = TMath::Min(minWeight[icl],minWeight[numDist-1]);
         maxWeight[numDist-1] = TMath::Max(maxWeight[icl],maxWeight[numDist-1]);
      }
   }

   // Sorting the lists, getting nbins ...
   const UInt_t nevmin=10;  // minimum number of events per bin (to make sure we get reasonable distributions)
   const UInt_t nbinsmax=2000; // maximum number of bins

   for (UInt_t icl=0; icl< numDist; icl++){
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         listsForBinning[icl][ivar].sort();  
         std::list< TMVA::TMVAGaussPair >::iterator it;
         Float_t sumPerBin = sumOfWeights[icl]/nbinsmax;
         sumPerBin=TMath::Max(minWeight[icl]*nevmin,sumPerBin);
         Float_t sum=0;
         Float_t ev_value=listsForBinning[icl][ivar].begin()->GetValue();
         Float_t lastev_value=ev_value;
         const Float_t eps = 1.e-4;
         vsForBinning[icl][ivar].push_back(ev_value-eps);
         vsForBinning[icl][ivar].push_back(ev_value);

         for (it=listsForBinning[icl][ivar].begin(); it != listsForBinning[icl][ivar].end(); it++){
            sum+= it->GetWeight();
            if (sum >= sumPerBin) {
               ev_value=it->GetValue();
               if (ev_value>lastev_value) {   // protection against bin width of 0
                  vsForBinning[icl][ivar].push_back(ev_value);
                  sum = 0.;
                  lastev_value=ev_value;
               }
            }
         }
         if (sum!=0) vsForBinning[icl][ivar].push_back(listsForBinning[icl][ivar].back().GetValue()); 
         nbins[icl][ivar] = vsForBinning[icl][ivar].size();
      }
   }

   delete[] sumOfWeights;
   delete[] minWeight;
   delete[] maxWeight;

   // create histogram for the cumulative distribution.
   fCumulativeDist.resize(nvar);
   for (UInt_t icls = 0; icls < numDist; icls++) {
      for (UInt_t ivar=0; ivar < nvar; ivar++){
         Float_t* binnings = new Float_t[nbins[icls][ivar]];
         //the binning for this particular histogram:
         for (UInt_t k =0 ; k < nbins[icls][ivar]; k++){
            binnings[k] = vsForBinning[icls][ivar][k];
         }
         fCumulativeDist[ivar].resize(numDist);         
         if (0 != fCumulativeDist[ivar][icls] ) {
            delete fCumulativeDist[ivar][icls]; 
         }
         fCumulativeDist[ivar][icls] = new TH1F(Form("Cumulative_Var%d_cls%d",ivar,icls),
                                                Form("Cumulative_Var%d_cls%d",ivar,icls),
                                                nbins[icls][ivar] -1, // class icls
                                                binnings);
         fCumulativeDist[ivar][icls]->SetDirectory(0);
         delete [] binnings;
      }
   }
   
   // Deallocation
   for (UInt_t i=0; i<numDist; i++) {
      delete [] listsForBinning[numDist-i-1];
      delete [] vsForBinning[numDist-i-1];
      delete [] nbins[numDist-i-1];
   }
   delete [] listsForBinning;
   delete [] vsForBinning;
   delete [] nbins;

   // perform event loop
   std::vector<Int_t> ic(numDist);
   for (UInt_t ievt=0; ievt<nevt; ievt++) {
      
      const Event* ev= events[ievt];
      Int_t cls = ev->GetClass();
      
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         fCumulativeDist[ivar][cls]->Fill(ev->GetValue(ivar),ev->GetWeight());;               
         if (numDist>1) fCumulativeDist[ivar][numDist-1]->Fill(ev->GetValue(ivar),ev->GetWeight());;               
      }
   }         
   
   // clean up 
   CleanUpCumulativeArrays("PDF");

   // now sum up in order to get the real cumulative distribution   
   Double_t  sum = 0, total=0;
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      fCumulativePDF.resize(ivar+1);
      for (UInt_t icls=0; icls<numDist; icls++) {      
         (fCumulativeDist[ivar][icls])->Smooth(); 
         sum = 0;
         total = 0.;
         for (Int_t ibin=1; ibin <=fCumulativeDist[ivar][icls]->GetNbinsX() ; ibin++){
            Float_t val = (fCumulativeDist[ivar][icls])->GetBinContent(ibin);
            if (val>0) total += val;
         }
         for (Int_t ibin=1; ibin <=fCumulativeDist[ivar][icls]->GetNbinsX() ; ibin++){
            Float_t val = (fCumulativeDist[ivar][icls])->GetBinContent(ibin);
            if (val>0) sum += val;
            (fCumulativeDist[ivar][icls])->SetBinContent(ibin,sum/total);
         }
         // create PDf
         fCumulativePDF[ivar].push_back(new PDF( Form("GaussTransform var%d cls%d",ivar,icls),  fCumulativeDist[ivar][icls], PDF::kSpline1, fPdfMinSmooth, fPdfMaxSmooth,kFALSE,kFALSE));
      }
   }
}

//_______________________________________________________________________
void TMVA::VariableGaussTransform::WriteTransformationToStream( std::ostream& ) const
{
   Log() << kFATAL << "VariableGaussTransform::WriteTransformationToStream is obsolete" << Endl; 
}

//_______________________________________________________________________
void TMVA::VariableGaussTransform::CleanUpCumulativeArrays(TString opt) {
   // clean up of cumulative arrays
   if (opt == "ALL" || opt == "PDF"){ 
      for (UInt_t ivar=0; ivar<fCumulativePDF.size(); ivar++) {
         for (UInt_t icls=0; icls<fCumulativePDF[ivar].size(); icls++) { 
            if (0 != fCumulativePDF[ivar][icls]) delete fCumulativePDF[ivar][icls];
         }
      }
      fCumulativePDF.clear();
   }
   if (opt == "ALL" || opt == "Dist"){ 
      for (UInt_t ivar=0; ivar<fCumulativeDist.size(); ivar++) {
         for (UInt_t icls=0; icls<fCumulativeDist[ivar].size(); icls++) { 
            if (0 != fCumulativeDist[ivar][icls]) delete fCumulativeDist[ivar][icls];
         }
      }
      fCumulativeDist.clear();
   }
}
//_______________________________________________________________________
void TMVA::VariableGaussTransform::AttachXMLTo(void* parent) {
   // create XML description of Gauss transformation
   void* trfxml = gTools().AddChild(parent, "Transform");
   gTools().AddAttr(trfxml, "Name",        "Gauss");
   gTools().AddAttr(trfxml, "FlatOrGauss", (fFlatNotGauss?"Flat":"Gauss") );

   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) {
      void* varxml = gTools().AddChild( trfxml, "Variable");
      gTools().AddAttr( varxml, "Name",     Variables()[ivar].GetLabel() );
      gTools().AddAttr( varxml, "VarIndex", ivar );
         
      if ( fCumulativePDF[ivar][0]==0 || fCumulativePDF[ivar][1]==0 )
         Log() << kFATAL << "Cumulative histograms for variable " << ivar << " don't exist, can't write it to weight file" << Endl;
      
      for (UInt_t icls=0; icls<fCumulativePDF[ivar].size(); icls++){
         void* pdfxml = gTools().AddChild( varxml, Form("CumulativePDF_cls%d",icls));
         (fCumulativePDF[ivar][icls])->AddXMLTo(pdfxml);
      }
   }
}

//_______________________________________________________________________
void TMVA::VariableGaussTransform::ReadFromXML( void* trfnode ) {
   // Read the transformation matrices from the xml node

   // clean up first
   CleanUpCumulativeArrays();
   TString FlatOrGauss;
   gTools().ReadAttr(trfnode, "FlatOrGauss", FlatOrGauss );
   if (FlatOrGauss == "Flat") fFlatNotGauss = kTRUE;
   else                       fFlatNotGauss = kFALSE;

   // Read the cumulative distribution
   void* varnode = gTools().GetChild( trfnode );

   TString varname, histname, classname;
   UInt_t ivar;
   while(varnode) {
      gTools().ReadAttr(varnode, "Name", varname);
      gTools().ReadAttr(varnode, "VarIndex", ivar);
      
      void* clsnode = gTools().GetChild( varnode);

      while(clsnode) {
         void* pdfnode = gTools().GetChild( clsnode);
         PDF* pdfToRead = new PDF(TString("tempName"),kFALSE);
         pdfToRead->ReadXML(pdfnode); // pdfnode
         // push_back PDF
         fCumulativePDF.resize( ivar+1 );
         fCumulativePDF[ivar].push_back(pdfToRead);
         clsnode = gTools().GetNextChild(clsnode);
      }
      
      varnode = gTools().GetNextChild(varnode);    
   }
   SetCreated();
}

//_______________________________________________________________________
void TMVA::VariableGaussTransform::ReadTransformationFromStream( std::istream& istr, const TString& classname)
{
   // Read the cumulative distribution
   Bool_t addDirStatus = TH1::AddDirectoryStatus();
   TH1::AddDirectory(0); // this avoids the binding of the hists in TMVA::PDF to the current ROOT file
   char buf[512];
   istr.getline(buf,512);

   TString strvar, dummy;

   while (!(buf[0]=='#'&& buf[1]=='#')) { // if line starts with ## return
      char* p = buf;
      while (*p==' ' || *p=='\t') p++; // 'remove' leading whitespace
      if (*p=='#' || *p=='\0') {
         istr.getline(buf,512);
         continue; // if comment or empty line, read the next line
      }
      std::stringstream sstr(buf);
      sstr >> strvar;
      
      if (strvar=="CumulativeHistogram") {
         UInt_t  type(0), ivar(0);
         TString devnullS(""),hname("");
         Int_t   nbins(0);

         // coverity[tainted_data_argument]
         sstr  >> type >> ivar >> hname >> nbins >> fElementsperbin;

         Float_t *Binnings = new Float_t[nbins+1];
         Float_t val;
         istr >> devnullS; // read the line "BinBoundaries" ..
         for (Int_t ibin=0; ibin<nbins+1; ibin++) {
            istr >> val;
            Binnings[ibin]=val;
         }

         if(ivar>=fCumulativeDist.size()) fCumulativeDist.resize(ivar+1);
         if(type>=fCumulativeDist[ivar].size()) fCumulativeDist[ivar].resize(type+1);

         TH1F * histToRead = fCumulativeDist[ivar][type];
         if ( histToRead !=0 ) delete histToRead;
         // recreate the cumulative histogram to be filled with the values read
         histToRead = new TH1F( hname, hname, nbins, Binnings );
         histToRead->SetDirectory(0);
         fCumulativeDist[ivar][type]=histToRead;

         istr >> devnullS; // read the line "BinContent" .. 
         for (Int_t ibin=0; ibin<nbins; ibin++) {
            istr >> val;
            histToRead->SetBinContent(ibin+1,val);
         }

         PDF* pdf = new PDF(hname,histToRead,PDF::kSpline0, 0, 0, kFALSE, kFALSE);
         // push_back PDF
         fCumulativePDF.resize(ivar+1);
         fCumulativePDF[ivar].resize(type+1);
         fCumulativePDF[ivar][type] = pdf;
         delete [] Binnings;
      }

      //      if (strvar=="TransformToFlatInsetadOfGauss=") { // don't correct this spelling mistake
      if (strvar=="Uniform") { // don't correct this spelling mistake
         sstr >> fFlatNotGauss;
         istr.getline(buf,512);
         break;
      }

      istr.getline(buf,512); // reading the next line   
   }
   TH1::AddDirectory(addDirStatus);

   UInt_t classIdx=(classname=="signal")?0:1;
   for(UInt_t ivar=0; ivar<fCumulativePDF.size(); ++ivar) {
      PDF* src = fCumulativePDF[ivar][classIdx];
      fCumulativePDF[ivar].push_back(new PDF(src->GetName(),fCumulativeDist[ivar][classIdx],PDF::kSpline0, 0, 0, kFALSE, kFALSE) );
   }
   
   SetTMVAVersion(TMVA_VERSION(3,9,7));

   SetCreated();
}

Double_t TMVA::VariableGaussTransform::OldCumulant(Float_t x, TH1* h ) const {

   Int_t bin = h->FindBin(x);
   bin = TMath::Max(bin,1);
   bin = TMath::Min(bin,h->GetNbinsX());

   Double_t cumulant;
   Double_t x0, x1, y0, y1;
   Double_t total = h->GetNbinsX()*fElementsperbin;
   Double_t supmin = 0.5/total;
   
   x0 = h->GetBinLowEdge(TMath::Max(bin,1)); 
   x1 = h->GetBinLowEdge(TMath::Min(bin,h->GetNbinsX())+1); 

   y0 = h->GetBinContent(TMath::Max(bin-1,0)); // Y0 = F(x0); Y0 >= 0
   y1 = h->GetBinContent(TMath::Min(bin, h->GetNbinsX()+1));  // Y1 = F(x1);  Y1 <= 1

   if (bin == 0) {
      y0 = supmin;
      y1 = supmin;
   }
   if (bin == 1) {
      y0 = supmin;
   }
   if (bin > h->GetNbinsX()) { 
      y0 = 1.-supmin;
      y1 = 1.-supmin; 
   }
   if (bin == h->GetNbinsX()) { 
      y1 = 1.-supmin; 
   }

   if (x0 == x1) { 
      cumulant = y1;
   } else {
      cumulant = y0 + (y1-y0)*(x-x0)/(x1-x0);
   }

   if (x <= h->GetBinLowEdge(1)){
      cumulant = supmin;
   }
   if (x >= h->GetBinLowEdge(h->GetNbinsX()+1)){
      cumulant = 1-supmin;
   }
   return cumulant;
}



//_______________________________________________________________________
void TMVA::VariableGaussTransform::PrintTransformation( ostream& ) 
{
   // prints the transformation 
   Int_t cls = 0;
   Log() << kINFO << "I do not know yet how to print this... look in the weight file " << cls << ":" << Endl;
   cls++;
}

//_______________________________________________________________________
void TMVA::VariableGaussTransform::MakeFunction( std::ostream& fout, const TString& fcncName, 
                                                 Int_t part, UInt_t trCounter, Int_t ) 
{
   // creates the transformation function
   //
   const UInt_t nvar = GetNVariables();
   UInt_t numDist  = GetNClasses() + 1;
   Int_t nBins = 1000; 
   // creates the gauss transformation function
   if (part==1) {
      fout << std::endl;
      // declare variables
      fout << "   double  cumulativeDist["<<nvar<<"]["<<numDist<<"]["<<nBins+1<<"];"<<std::endl;
      fout << "   double xMin["<<nvar<<"]["<<numDist<<"];"<<std::endl;
      fout << "   double xMax["<<nvar<<"]["<<numDist<<"];"<<std::endl;
   }
   if (part==2) {
      fout << std::endl;
      fout << "#include \"math.h\"" << std::endl;
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::InitTransform_"<<trCounter<<"()" << std::endl;
      fout << "{" << std::endl;
      // fill meat here
      // loop over nvar , cls, loop over nBins
      // fill cumulativeDist with fCumulativePDF[ivar][cls])->GetValue(vec(ivar)
      for (UInt_t icls=0; icls<numDist; icls++) {
         for (UInt_t ivar=0; ivar<nvar; ivar++) {
            Double_t xmn=(fCumulativePDF[ivar][icls])->GetXmin();
            Double_t xmx=(fCumulativePDF[ivar][icls])->GetXmax();
            fout << "    xMin["<<ivar<<"]["<<icls<<"]="<<xmn<<";"<<std::endl;
            fout << "    xMax["<<ivar<<"]["<<icls<<"]="<<xmx<<";"<<std::endl;
            for (Int_t ibin=0; ibin<=nBins; ibin++) {
               Double_t xval = xmn + (xmx-xmn) / nBins * (ibin+0.5);
               // ibin = (xval -xmin) / (xmax-xmin) *1000 
               fout << "  cumulativeDist[" << ivar << "]["<< icls<< "]["<<ibin<<"]="<< (fCumulativePDF[ivar][icls])->GetVal(xval)<< ";"<<std::endl;
            }
         }
      }
      fout << "}" << std::endl;
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::Transform_"<<trCounter<<"( std::vector<double>& iv, int cls) const" << std::endl;
      fout << "{" << std::endl;
      fout << "   if (cls < 0 || cls > "<<GetNClasses()<<") {"<< std::endl;
      fout << "       if ("<<GetNClasses()<<" > 1 ) cls = "<<GetNClasses()<<";"<< std::endl;
      fout << "       else cls = "<<(fCumulativePDF[0].size()==1?0:2)<<";"<< std::endl;
      fout << "   }"<< std::endl;
      
      fout << "   bool FlatNotGauss = "<< (fFlatNotGauss? "true": "false") <<";"<< std::endl;
      fout << "   double cumulant;"<< std::endl;
      fout << "   const int nvar = "<<GetNVariables()<<";"<< std::endl;
      fout << "   for (int ivar=0; ivar<nvar; ivar++) {"<< std::endl;
      // ibin = (xval -xmin) / (xmax-xmin) *1000 
      fout << "   int ibin1 = (int) ((iv[ivar]-xMin[ivar][cls])/(xMax[ivar][cls]-xMin[ivar][cls])*"<<nBins<<");"<<std::endl;
      fout << "   if (ibin1<=0) { cumulant = cumulativeDist[ivar][cls][0];}"<<std::endl;
      fout << "   else if (ibin1>="<<nBins<<") { cumulant = cumulativeDist[ivar][cls]["<<nBins<<"];}"<<std::endl;
      fout << "   else {"<<std::endl;
      fout << "           int ibin2 = ibin1+1;" << std::endl;
      fout << "           double dx = iv[ivar]-(xMin[ivar][cls]+"<< (1./nBins)
           << "           * ibin1* (xMax[ivar][cls]-xMin[ivar][cls]));"  
           << std::endl;
      fout << "           double eps=dx/(xMax[ivar][cls]-xMin[ivar][cls])*"<<nBins<<";"<<std::endl;
      fout << "           cumulant = eps*cumulativeDist[ivar][cls][ibin1] + (1-eps)*cumulativeDist[ivar][cls][ibin2];" << std::endl;
      fout << "           if (cumulant>1.-10e-10) cumulant = 1.-10e-10;"<< std::endl;
      fout << "           if (cumulant<10e-10)    cumulant = 10e-10;"<< std::endl;
      fout << "           if (FlatNotGauss) iv[ivar] = cumulant;"<< std::endl;
      fout << "           else {"<< std::endl;
      fout << "              double maxErfInvArgRange = 0.99999999;"<< std::endl;
      fout << "              double arg = 2.0*cumulant - 1.0;"<< std::endl;
      fout << "              if (arg >  maxErfInvArgRange) arg= maxErfInvArgRange;"<< std::endl;
      fout << "              if (arg < -maxErfInvArgRange) arg=-maxErfInvArgRange;"<< std::endl;
      fout << "              double inverf=0., stp=1. ;"<<std::endl;
      fout << "              while (stp >1.e-10){;"<<std::endl;
      fout << "                if (erf(inverf)>arg) inverf -=stp ;"<<std::endl;
      fout << "                else if (erf(inverf)<=arg && erf(inverf+stp)>=arg) stp=stp/5. ;"<<std::endl;
      fout << "                else inverf += stp;"<<std::endl;
      fout << "              } ;"<<std::endl;
      fout << "              //iv[ivar] = 1.414213562*TMath::ErfInverse(arg);"<< std::endl;
      fout << "              iv[ivar] = 1.414213562* inverf;"<< std::endl;
      fout << "           }"<< std::endl;
      fout << "       }"<< std::endl;
      fout << "   }"<< std::endl;
      fout << "}" << std::endl;
   }
}
