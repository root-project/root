/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :  - date

////////////////////////////////////////////////////////////////////////////////
// A simple geometry checker. Points can be randomly generated inside the 
// bounding  box of a node. For each point the distance to the nearest surface
// and the corresponting point on that surface are computed. These points are 
// stored in a tree and can be directly visualized within ROOT
// A second algoritm is shooting multiple rays from a given point to a geometry
// branch and storing the intersection points with surfaces in same tree. 
// Rays can be traced backwords in order to find overlaps by comparing direct 
// and inverse points.
//Begin_Html
/*
<img src="gif/t_checker.jpg">
*/
//End_Html

#include "TObject.h"
#include "TGeoChecker.h"


// statics and globals

ClassImp(TGeoChecker)

//-----------------------------------------------------------------------------
TGeoChecker::TGeoChecker()
{
// Default constructor
   fCurrentNode  = 0;
   fTreePts      = 0; 
}
//-----------------------------------------------------------------------------
TGeoChecker::TGeoChecker(const char *treename, const char *filename)
{
// constructor
}
//-----------------------------------------------------------------------------
TGeoChecker::~TGeoChecker()
{
// Destructor
}
//-----------------------------------------------------------------------------
void TGeoChecker::CreateTree(const char *treename, const char *filename)
{
// These points are stored in a tree and can be directly visualized within ROOT.
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
//-----------------------------------------------------------------------------
void TGeoChecker::Generate(UInt_t npoint)
{
// Points are randomly generated inside the 
// bounding  box of a node. For each point the distance to the nearest surface
// and the corresponding point on that surface are computed.
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
//-----------------------------------------------------------------------------
void TGeoChecker::Raytrace(Double_t *startpoint, UInt_t npoints)
{
// A second algoritm is shooting multiple rays from a given point to a geometry
// branch and storing the intersection points with surfaces in same tree. 
// Rays can be traced backwords in order to find overlaps by comparing direct 
// and inverse points.   
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
//-----------------------------------------------------------------------------
void TGeoChecker::ShowPoints(Option_t *option)
{
// 
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
