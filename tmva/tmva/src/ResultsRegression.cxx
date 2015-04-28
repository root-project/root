// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : ResultsRegression                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <vector>

#include "TMVA/ResultsRegression.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/DataSet.h"

//_______________________________________________________________________
TMVA::ResultsRegression::ResultsRegression( const DataSetInfo* dsi, TString resultsName  ) 
   : Results( dsi, resultsName  ),
     fLogger( new MsgLogger(Form("ResultsRegression%s",resultsName.Data()) , kINFO) )
{
   // constructor
}

//_______________________________________________________________________
TMVA::ResultsRegression::~ResultsRegression() 
{
   // destructor
   delete fLogger;
}

//_______________________________________________________________________
void TMVA::ResultsRegression::SetValue( std::vector<Float_t>& value, Int_t ievt )
{
   if (ievt >= (Int_t)fRegValues.size()) fRegValues.resize( ievt+1 );
   fRegValues[ievt] = value; 
}

//_______________________________________________________________________
TH1F*  TMVA::ResultsRegression::QuadraticDeviation( UInt_t tgtNum , Bool_t truncate, Double_t truncvalue )
{
   DataSet* ds = GetDataSet();
   ds->SetCurrentType( GetTreeType() );
   const DataSetInfo* dsi = GetDataSetInfo();
   TString name( Form("tgt_%d",tgtNum) );
   VariableInfo vinf = dsi->GetTargetInfo(tgtNum);
   Float_t xmin=0., xmax=0.;
   if (truncate){
     xmax = truncvalue;
   }
   else{
     for (Int_t ievt=0; ievt<ds->GetNEvents(); ievt++) {
       const Event* ev = ds->GetEvent(ievt);
       std::vector<Float_t> regVal = fRegValues.at(ievt);
       Float_t val = regVal.at( tgtNum ) - ev->GetTarget( tgtNum );
       val *= val;
       xmax = val> xmax? val: xmax;
     } 
   }
   xmax *= 1.1;
   Int_t nbins = 500;
   TH1F* h = new TH1F( name, name, nbins, xmin, xmax);
   h->SetDirectory(0);
   h->GetXaxis()->SetTitle("Quadratic Deviation");
   h->GetYaxis()->SetTitle("Weighted Entries");

   for (Int_t ievt=0; ievt<ds->GetNEvents(); ievt++) {
      const Event* ev = ds->GetEvent(ievt);
      std::vector<Float_t> regVal = fRegValues.at(ievt);
      Float_t val = regVal.at( tgtNum ) - ev->GetTarget( tgtNum );
      val *= val;
      Float_t weight = ev->GetWeight();
      if (!truncate || val<=truncvalue ) h->Fill( val, weight);
   } 
   return h;
}

//_______________________________________________________________________
TH2F*  TMVA::ResultsRegression::DeviationAsAFunctionOf( UInt_t varNum, UInt_t tgtNum )
{
   DataSet* ds = GetDataSet();
   ds->SetCurrentType( GetTreeType() );
   
   TString name( Form("tgt_%d_var_%d",tgtNum, varNum) );
   const DataSetInfo* dsi = GetDataSetInfo();
   Float_t xmin, xmax;
   Bool_t takeTargets = kFALSE;
   if (varNum >= dsi->GetNVariables()) {
      takeTargets = kTRUE;
      varNum -= dsi->GetNVariables();
   }
   if (!takeTargets) {
      VariableInfo vinf = dsi->GetVariableInfo(varNum);
      xmin = vinf.GetMin();
      xmax = vinf.GetMax();

      for (Int_t ievt=0; ievt<ds->GetNEvents(); ievt++) {
         const Event* ev = ds->GetEvent(ievt);
         Float_t val = ev->GetValue(varNum);

         if (val < xmin ) xmin = val;
         if (val > xmax ) xmax = val;
      }

   }
   else {
      VariableInfo vinf = dsi->GetTargetInfo(varNum);
      xmin = vinf.GetMin();
      xmax = vinf.GetMax();

      for (Int_t ievt=0; ievt<ds->GetNEvents(); ievt++) {
         const Event* ev = ds->GetEvent(ievt);
         Float_t val = ev->GetTarget(varNum);

         if (val < xmin ) xmin = val;
         if (val > xmax ) xmax = val;
      }
   }

   Float_t ymin = FLT_MAX;
   Float_t ymax = -FLT_MAX;

   for (Int_t ievt=0; ievt<ds->GetNEvents(); ievt++) {
      const Event* ev = ds->GetEvent(ievt);
      std::vector<Float_t> regVal = fRegValues.at(ievt);

      Float_t diff = regVal.at( tgtNum ) - ev->GetTarget( tgtNum );
      if (diff < ymin ) ymin = diff;
      if (diff > ymax ) ymax = diff;
   }

   Int_t   nxbins = 50;
   Int_t   nybins = 50;

   Float_t epsilon = TMath::Abs(xmax-xmin)/((Float_t)nxbins-1);
   xmin -= 1.01*epsilon;
   xmax += 1.01*epsilon;

   epsilon = (ymax-ymin)/((Float_t)nybins-1);
   ymin -= 1.01*epsilon;
   ymax += 1.01*epsilon;


   TH2F* h = new TH2F( name, name, nxbins, xmin, xmax, nybins, ymin, ymax ); 
   h->SetDirectory(0);

   h->GetXaxis()->SetTitle( (takeTargets ? dsi->GetTargetInfo(varNum).GetTitle() : dsi->GetVariableInfo(varNum).GetTitle() ) );
   TString varName( dsi->GetTargetInfo(tgtNum).GetTitle() );
   TString yName( varName+TString("_{regression} - ") + varName+TString("_{true}") );
   h->GetYaxis()->SetTitle( yName );

   for (Int_t ievt=0; ievt<ds->GetNEvents(); ievt++) {
      const Event* ev = ds->GetEvent(ievt);
      std::vector<Float_t> regVal = fRegValues.at(ievt);

      Float_t xVal = (takeTargets?ev->GetTarget( varNum ):ev->GetValue( varNum ));
      Float_t yVal = regVal.at( tgtNum ) - ev->GetTarget( tgtNum );

      h->Fill( xVal, yVal );
   }

   return h;
}

//_______________________________________________________________________
void  TMVA::ResultsRegression::CreateDeviationHistograms( TString prefix )
{
   Log() << kINFO << "Create variable histograms" << Endl;
   const DataSetInfo* dsi = GetDataSetInfo();

   for (UInt_t ivar = 0; ivar < dsi->GetNVariables(); ivar++) {
      for (UInt_t itgt = 0; itgt < dsi->GetNTargets(); itgt++) {
         TH2F* h = DeviationAsAFunctionOf( ivar, itgt );
         TString name( Form("%s_reg_var%d_rtgt%d",prefix.Data(),ivar,itgt) );
         h->SetName( name );
         h->SetTitle( name );
         Store( h );
      }
   }
   Log() << kINFO << "Create regression target histograms" << Endl;
   for (UInt_t ivar = 0; ivar < dsi->GetNTargets(); ivar++) {
      for (UInt_t itgt = 0; itgt < dsi->GetNTargets(); itgt++) {
         TH2F* h = DeviationAsAFunctionOf( dsi->GetNVariables()+ivar, itgt );
         TString name( Form("%s_reg_tgt%d_rtgt%d",prefix.Data(),ivar,itgt) );
         h->SetName( name );
         h->SetTitle( name );
         Store( h );
      }
   }

   Log() << kINFO << "Create regression average deviation" << Endl;
   for (UInt_t itgt = 0; itgt < dsi->GetNTargets(); itgt++) {
     TH1F* h =  QuadraticDeviation(itgt);
     TString name( Form("%s_Quadr_Deviation_target_%d_",prefix.Data(),itgt) );
     h->SetName( name );
     h->SetTitle( name );
     Double_t yq[1], xq[]={0.9};
     h->GetQuantiles(1,yq,xq);
     Store( h );

     TH1F* htrunc = QuadraticDeviation(itgt, true, yq[0]);
     TString name2( Form("%s_Quadr_Dev_best90perc_target_%d_",prefix.Data(),itgt) );
     htrunc->SetName( name2 );
     htrunc->SetTitle( name2 );
     Store( htrunc );
   }
   Log() << kINFO << "Results created" << Endl;
}
