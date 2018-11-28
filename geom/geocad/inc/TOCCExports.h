// @(#)geom/geocad:$Id$
// Author: Cinzia Luzzi   5/5/2012

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TOCCExports
#define ROOT_TOCCExports

// Combine all OCE headers used by geocad.
// After including them, #undef Handle.

#define Printf Printf_opencascade
#include <TDF_Label.hxx>
#include <TDocStd_Document.hxx>
#include <Standard_Version.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Wire.hxx>
#undef Printf


// ROOT-9837
#if defined(Handle) && !defined(R__Needs_Handle)
#undef Handle
#endif

#endif
