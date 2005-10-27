// @(#)root/minuit2:$Name:  $:$Id: TChi2ExtendedFitData.cxx,v 1.1 2005/10/27 14:11:07 brun Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include <cassert>

#include "TChi2ExtendedFitData.h"

#include "TVirtualFitter.h" 

#if defined(__alpha) 
#define __USE_STD_IOSTREAM
#endif

#include <iostream>

#include "TGraph.h"

// constructor - create fit data from Histogram content

TChi2ExtendedFitData::TChi2ExtendedFitData(const TVirtualFitter & fitter )  {

  fSize = 0;

  TObject * obj = fitter.GetObjectFit(); 
  
  // case of TGraph
  TGraph * graph = dynamic_cast<TGraph*> ( obj );
  if (graph) { 
    GetExtendedFitData(graph, &fitter);    
  } 
  else { 
    std::cout << "other fit on different object than TGraf not yet supported- assert" << std::endl;
    assert(graph); 
  }
}


  // get data for graf with errors 
void TChi2ExtendedFitData::GetExtendedFitData(const TGraph * gr, const TVirtualFitter * /*hFitter*/ ) {

  // fit options
  //Foption_t fitOption = hFitter->GetFitOption();

  int  nPoints = gr->GetN();
  double *gx = gr->GetX();
  double *gy = gr->GetY();
  // return 0 pointer for some graphs, cannot be used 
  //    double *ey = gr->GetEY();
  //    double *exl = gr->GetEXlow();
  //    double *exh = gr->GetEXhigh();
  
  CoordData x = CoordData( 1 );  // 1D graph
  
  //   std::cout << exl << "  " << ey << std::endl;
  //std::cout << "creating data with size " << nPoints << std::endl;
  
  for (int  i = 0; i < nPoints; ++i) { 
    
    x[0] = gx[i];
    SetDataPoint( x, gy[i], gr->GetErrorY(i),  gr->GetErrorXlow(i), gr->GetErrorXhigh(i) );
    
  }
}


void TChi2ExtendedFitData::SetDataPoint( const CoordData & x, double y, double ey, double exl, double exh) { 
  

  fCoordinates.push_back(x);
  fValues.push_back(y);
  fErrorsY.push_back(ey);
  fErrorsXLow.push_back(exl);
  fErrorsXUp.push_back(exh);
  fSize++;
}
