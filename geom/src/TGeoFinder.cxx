// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   18/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// TGeoFinder class
//
//
//
//
//Begin_Html
/*
<img src="gif/t_finder.jpg">
<img src="gif/t_voxelfind.jpg">
<img src="gif/t_voxtree.jpg">
*/
//End_Html


/*************************************************************************
 * TGeoFinder - virtual base class for tracking inside a volume. 
 *  
 *************************************************************************/

#include "TGeoFinder.h"

ClassImp(TGeoFinder)

//-----------------------------------------------------------------------------
TGeoFinder::TGeoFinder()
{
// Default constructor
   fVolume = 0;
}
//-----------------------------------------------------------------------------
TGeoFinder::TGeoFinder(TGeoVolume *vol)
{
// Default constructor
   fVolume = vol;
}
//-----------------------------------------------------------------------------
TGeoFinder::~TGeoFinder()
{
// Destructor
}
//-----------------------------------------------------------------------------
