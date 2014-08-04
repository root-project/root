// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  28/07/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdexcept>

#include "TKDEAdapter.h"
#include "TKDEFGT.h"
#include "TError.h"
#include "TGL5D.h"
#include "TAxis.h"

namespace Rgl {
namespace Fgt {

//______________________________________________________________________________
TKDEAdapter::TKDEAdapter()
               : fW(0), fH(0), fD(0),
                 fSliceSize(0),
                 fXMin(0.), fXStep(0.),
                 fYMin(0.), fYStep(0.),
                 fZMin(0.), fZStep(0.),
                 fDE(0),
                 fE(10.)
{
   //Constructor. "Half-baked" object.
}

//______________________________________________________________________________
void TKDEAdapter::SetGeometry(const TGL5DDataSet *dataSet)
{
   //Set grid's dimensions.
   const TAxis *xA = dataSet->GetXAxis();
   const Rgl::Range_t &xMinMax = dataSet->GetXRange();
   const Double_t xRange = xMinMax.second - xMinMax.first;

   const TAxis *yA = dataSet->GetYAxis();
   const Rgl::Range_t &yMinMax = dataSet->GetYRange();
   const Double_t yRange = yMinMax.second - yMinMax.first;

   const TAxis *zA = dataSet->GetZAxis();
   const Rgl::Range_t &zMinMax = dataSet->GetZRange();
   const Double_t zRange = zMinMax.second - zMinMax.first;

   fW = xA->GetNbins();
   fH = yA->GetNbins();
   fD = zA->GetNbins();

   fSliceSize = fW * fH;

   fXMin = (xA->GetBinLowEdge(1) - xMinMax.first) / xRange;
   fXStep = (xA->GetBinUpEdge(xA->GetLast()) - xA->GetBinLowEdge(xA->GetFirst())) / (fW - 1) / xRange;

   fYMin = (yA->GetBinLowEdge(1) - yMinMax.first) / yRange;
   fYStep = (yA->GetBinUpEdge(yA->GetLast()) - yA->GetBinLowEdge(yA->GetFirst())) / (fH - 1) / yRange;

   fZMin = (zA->GetBinLowEdge(1) - zMinMax.first) / zRange;
   fZStep = (zA->GetBinCenter(zA->GetLast()) - zA->GetBinLowEdge(zA->GetFirst())) / (fD - 1) / zRange;
}

//______________________________________________________________________________
void TKDEAdapter::SetE(Double_t e)
{
   //e for kdefgt.
   fE = e;
}

//______________________________________________________________________________
Double_t TKDEAdapter::GetE()const
{
   //e for kdefgt.
   return fE;
}

//______________________________________________________________________________
UInt_t TKDEAdapter::GetW()const
{
   //Number of cells along X.
   return fW;
}

//______________________________________________________________________________
UInt_t TKDEAdapter::GetH()const
{
   //Number of cells along Y.
   return fH;
}

//______________________________________________________________________________
UInt_t TKDEAdapter::GetD()const
{
   //Number of cells along Z.
   return fD;
}

//______________________________________________________________________________
void TKDEAdapter::SetDataSource(const TKDEFGT *de)
{
   //Set density estimator as a data source.
   fDE = de;
}

//______________________________________________________________________________
void TKDEAdapter::FetchDensities()const
{
   //Do some initialization and calculate densities.
   if (!fDE) {
      Error("TKDEAdapter::FetchFirstSlices", "Density estimator is a null pointer."
            " Set it correctly first.");
      throw std::runtime_error("No density estimator.");
   }

   fGrid.resize(fD * fSliceSize * 3);//3 is the number of coordinates: xyz

   //1D index into fGrid array.
   UInt_t ind = 0;
   //The first slice.
   for(UInt_t k = 0; k < fD; ++k) {
      for (UInt_t i = 0; i < fH; ++i) {
         for (UInt_t j = 0; j < fW; ++j, ind += 3) {
            fGrid[ind]     = fXMin + j * fXStep;
            fGrid[ind + 1] = fYMin + i * fYStep;
            fGrid[ind + 2] = fZMin + k * fZStep;
         }
      }
   }

   fDensities.resize(fSliceSize * fD);
   //Ok, now, we can estimate densities.
   fDE->Predict(fGrid, fDensities, fE);
}

//______________________________________________________________________________
Float_t TKDEAdapter::GetData(UInt_t i, UInt_t j, UInt_t k)const
{
   // Get data at given position.
   const UInt_t ind = k * fSliceSize + j * fW + i;
   return fDensities[ind];
}

//______________________________________________________________________________
void TKDEAdapter::FreeVectors()
{
   // Free grid and density vectors.
   vector_t().swap(fGrid);
   vector_t().swap(fDensities);
}

}
}
