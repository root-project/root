// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveLine
#define ROOT_TEveLine

#include "TEveUtil.h"
#include "TEvePointSet.h"

#include "TAttLine.h"

class TEveLine : public TEvePointSet,
                 public TAttLine
{
   friend class TEveLineEditor;
   friend class TEveLineGL;

private:
   TEveLine(const TEveLine&);            // Not implemented
   TEveLine& operator=(const TEveLine&); // Not implemented

protected:
   Bool_t  fRnrLine;
   Bool_t  fRnrPoints;

public:
   TEveLine(Int_t n_points=0, ETreeVarType_e tv_type=kTVT_XYZ);
   TEveLine(const Text_t* name, Int_t n_points=0, ETreeVarType_e tv_type=kTVT_XYZ);
   virtual ~TEveLine() {}

   virtual void SetMarkerColor(Color_t col)
   { TAttMarker::SetMarkerColor(col); }
   virtual void SetLineColor(Color_t col)
   { SetMainColor(col); }

   Bool_t GetRnrLine() const   { return fRnrLine;   }
   void SetRnrLine(Bool_t r)   { fRnrLine = r;      }
   Bool_t GetRnrPoints() const { return fRnrPoints; }
   void SetRnrPoints(Bool_t r) { fRnrPoints = r;    }

   ClassDef(TEveLine, 1); // An arbitrary polyline with fixed line and marker attributes.
};

#endif
