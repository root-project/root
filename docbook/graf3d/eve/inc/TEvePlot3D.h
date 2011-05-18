// @(#)root/eve:$Id$
// Author: Matevz Tadel, 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEvePlot3D
#define ROOT_TEvePlot3D

#include "TEveElement.h"
#include "TAttBBox.h"

class TEvePlot3D : public TEveElementList
{
   friend class TEvePlot3DGL;

private:
   TEvePlot3D(const TEvePlot3D&);            // Not implemented
   TEvePlot3D& operator=(const TEvePlot3D&); // Not implemented

protected:
   TObject     *fPlot;       // Plot object.
   TString      fPlotOption; // Options for the plot-painter.

   Bool_t       fLogX;
   Bool_t       fLogY;
   Bool_t       fLogZ;

public:
   TEvePlot3D(const char* n="TEvePlot3D", const char* t="");
   virtual ~TEvePlot3D() {}

   void SetPlot(TObject* obj, const TString& opt) { fPlot = obj; fPlotOption = opt; }

   TObject* GetPlot()       const { return fPlot;   }
   TString  GetPlotOption() const { return fPlotOption; }

   void     SetLogXYZ(Bool_t lx, Bool_t ly, Bool_t lz) { fLogX = lx; fLogY = ly; fLogZ = lz; }

   void     SetLogX(Bool_t l) { fLogX = l; }
   void     SetLogY(Bool_t l) { fLogY = l; }
   void     SetLogZ(Bool_t l) { fLogZ = l; }

   Bool_t   GetLogX() const { return fLogX; }
   Bool_t   GetLogY() const { return fLogY; }
   Bool_t   GetLogZ() const { return fLogZ; }

   virtual void Paint(Option_t* option="");

   ClassDef(TEvePlot3D, 0); // Short description.
};

#endif
