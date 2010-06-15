// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : ResultsMulticlass                                                     *
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

#include "TMVA/ResultsMulticlass.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/DataSet.h"

//_______________________________________________________________________
TMVA::ResultsMulticlass::ResultsMulticlass( const DataSetInfo* dsi ) 
   : Results( dsi ),
     fLogger( new MsgLogger("ResultsMulticlass", kINFO) )
{
   // constructor
}

//_______________________________________________________________________
TMVA::ResultsMulticlass::~ResultsMulticlass() 
{
   // destructor
   delete fLogger;
}

//_______________________________________________________________________
void TMVA::ResultsMulticlass::SetValue( std::vector<Float_t>& value, Int_t ievt )
{
   if (ievt >= (Int_t)fMultiClassValues.size()) fMultiClassValues.resize( ievt+1 );
   fMultiClassValues[ievt] = value; 
}




//_______________________________________________________________________
void  TMVA::ResultsMulticlass::MakeHistograms()
{
//    DataSet* ds = GetDataSet();
//    ds->SetCurrentType( GetTreeType() );
//    const DataSetInfo* dsi = GetDataSetInfo();

//    TString name( Form("tgt_%d",tgtNum) );

//    VariableInfo vinf = dsi->GetTargetInfo(tgtNum);
//    Float_t xmin=0., xmax=0.;
//    if (truncate){
//      xmax = truncvalue;
//    }
//    else{
//      for (Int_t ievt=0; ievt<ds->GetNEvents(); ievt++) {
//        Event* ev = ds->GetEvent(ievt);
//        std::vector<Float_t> regVal = fMulticlassValues.at(ievt);
//        Float_t val = regVal.at( tgtNum ) - ev->GetTarget( tgtNum );
//        val *= val;
//        xmax = val> xmax? val: xmax;
//      } 
//    }
//    xmax *= 1.1;
//    Int_t nbins = 500;
//    TH1F* h = new TH1F( name, name, nbins, xmin, xmax);
//    h->SetDirectory(0);
//    h->GetXaxis()->SetTitle("Quadratic Deviation");
//    h->GetYaxis()->SetTitle("Weighted Entries");

//    for (Int_t ievt=0; ievt<ds->GetNEvents(); ievt++) {
//       Event* ev = ds->GetEvent(ievt);
//       std::vector<Float_t> regVal = fMulticlassValues.at(ievt);
//       Float_t val = regVal.at( tgtNum ) - ev->GetTarget( tgtNum );
//       val *= val;
//       Float_t weight = ev->GetWeight();
//       if (!truncate || val<=truncvalue ) h->Fill( val, weight);
//    } 
//    return h;
}

