// @(#)root/graf:$Id$
// Author: Anna-Pia Lohfink 27.3.2014

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TAttBBox2D.h"

ClassImp(TAttBBox2D);

/** \class TAttBBox2D
\ingroup Base
\ingroup GraphicsAtt

Abstract base class for elements drawn in the editor.
Classes inheriting from TAttBBox2D implementing the TAttBBox2D
virtual classes, and using TPad::ShowGuideLines in ExecuteEvent
will automatically get the guide lines drawn when moved in the pad.
All methods work with pixel coordinates.
*/

////////////////////////////////////////////////////////////////////////////////
// TAttBBox2D destructor.

TAttBBox2D::~TAttBBox2D()
{
}
