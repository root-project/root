// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  28/07/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTKDEAdapter
#define ROOT_TTKDEAdapter

#include <vector>

#ifndef ROOT_TGLIsoMesh
#include "TGLIsoMesh.h"
#endif
#ifndef ROOT_TKDEFGT
#include "TKDEFGT.h"
#endif

//KDE adapter is a "data source" adapter for 
//Marching cubes alhorithm. It produces scalar
//values in regular grid's nodes, using TKDEFGT class
//to estimate a density.
//IMPORTANT: This class is not intended for end-user's code, 
//it's used and _must_ be used only as an argument while 
//instantiating mesh builder's class template.
//That's why all members are protected
//or private - end user cannot create object's of this class.
//But mesh builder, derived from this class,
//knows exactly how to use this class correctly. 
//SetDimenions and SetE/GetE are public members, it will be derived by mesh 
//builder's instantiation and used inside TGL5DPainter class.

namespace Rgl {
namespace Fgt {

class TKDEAdapter : protected virtual Mc::TGridGeometry<Float_t> {
protected:
   typedef Float_t ElementType_t;

   TKDEAdapter();

public:
   void SetGeometry(const TGL5DDataSet *dataSet);

   void SetE(Double_t e);
   Double_t GetE()const;

protected:
   UInt_t GetW()const;
   UInt_t GetH()const;
   UInt_t GetD()const;

   void SetDataSource(const TKDEFGT *dataSource);

   void FetchDensities()const;

   Float_t GetData(UInt_t i, UInt_t j, UInt_t k)const;

   void FreeVectors();
private:
   typedef std::vector<Double_t> vector_t;

   mutable vector_t  fGrid;       //Grid to estimate density on.
   mutable vector_t  fDensities;  //Estimated densities.

   UInt_t    fW;//Number of cells along X.
   UInt_t    fH;//Number of cells along Y.
   UInt_t    fD;//Number of cells along Z.
   UInt_t    fSliceSize;//fW * fH.

   //Grid in a unit cube:
   Double_t  fXMin, fXStep;
   Double_t  fYMin, fYStep;
   Double_t  fZMin, fZStep;

   const TKDEFGT  *fDE;//Density estimator. This class does not own it.

   Double_t  fE;//For KDE.

   TKDEAdapter(const TKDEAdapter &rhs);
   TKDEAdapter &operator = (const TKDEAdapter &rhs);
};

}
}

#endif
