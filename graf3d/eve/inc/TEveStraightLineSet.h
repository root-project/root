// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveStraightLineSet
#define ROOT_TEveStraightLineSet

#include "TEveUtil.h"

#include <Gtypes.h>
#include "TNamed.h"
#include "TQObject.h"
#include "TAtt3D.h"
#include "TAttMarker.h"
#include "TAttLine.h"
#include "TAttBBox.h"

#include "TEveUtil.h"
#include "TEveElement.h"
#include "TEveProjectionBases.h"
#include "TEveChunkManager.h"
#include "TEveTrans.h"

class TRandom;

class TEveStraightLineSet : public TEveElement,
                            public TEveProjectable,
                            public TNamed,
                            public TQObject,
                            public TAtt3D,
                            public TAttLine,
                            public TAttMarker,
                            public TAttBBox
{
private:
   TEveStraightLineSet(const TEveStraightLineSet&);            // Not implemented
   TEveStraightLineSet& operator=(const TEveStraightLineSet&); // Not implemented

public:
   struct Line_t
   {
      Float_t        fV1[3];
      Float_t        fV2[3];
      TRef           fRef;

      Line_t(Float_t x1, Float_t y1, Float_t z1,
             Float_t x2, Float_t y2, Float_t z2)
      {
         fV1[0] = x1, fV1[1] = y1, fV1[2] = z1;
         fV2[0] = x2, fV2[1] = y2, fV2[2] = z2;
      }
   };

   struct Marker_t
   {
      Int_t        fLineID;
      Float_t      fPos;
      TRef         fRef;

      Marker_t(Int_t lineID, Float_t pos) : fLineID(lineID), fPos(pos) {};
   };

protected:
   TEveChunkManager  fLinePlex;
   TEveChunkManager  fMarkerPlex;

   Bool_t            fOwnLinesIds;    // Flag specifying if id-objects are owned by the TEveQuadSet
   Bool_t            fOwnMarkersIds;  // Flag specifying if id-objects are owned by the TEveQuadSet

   Bool_t            fRnrMarkers;
   Bool_t            fRnrLines;

   Bool_t            fDepthTest;

   Line_t*           fLastLine; //!

public:
   TEveStraightLineSet(const char* n="StraightLineSet", const char* t="");
   virtual ~TEveStraightLineSet() {}

   virtual void SetLineColor(Color_t col) { SetMainColor(col); }

   void AddLine(Float_t x1, Float_t y1, Float_t z1, Float_t x2, Float_t y2, Float_t z2);
   void AddMarker(Int_t lineID, Float_t pos);

   TEveChunkManager& GetLinePlex()   { return fLinePlex;   }
   TEveChunkManager& GetMarkerPlex() { return fMarkerPlex; }

   virtual Bool_t GetRnrMarkers() { return fRnrMarkers; }
   virtual Bool_t GetRnrLines()   { return fRnrLines;   }
   virtual Bool_t GetDepthTest()  { return fDepthTest;   }

   virtual void SetRnrMarkers(Bool_t x) { fRnrMarkers = x; }
   virtual void SetRnrLines(Bool_t x)   { fRnrLines   = x; }
   virtual void SetDepthTest(Bool_t x)  { fDepthTest   = x; }
   
   virtual void CopyVizParams(const TEveElement* el);
   virtual void WriteVizParams(ostream& out, const TString& var);

   virtual TClass* ProjectedClass(const TEveProjection* p) const;

   virtual void ComputeBBox();
   virtual void Paint(Option_t* option="");

   ClassDef(TEveStraightLineSet, 1); // Set of straight lines with optional markers along the lines.
};


/******************************************************************************/

class TEveStraightLineSetProjected : public TEveStraightLineSet,
                                     public TEveProjected
{
private:
   TEveStraightLineSetProjected(const TEveStraightLineSetProjected&);            // Not implemented
   TEveStraightLineSetProjected& operator=(const TEveStraightLineSetProjected&); // Not implemented

protected:
   virtual void SetDepthLocal(Float_t d);

public:
   TEveStraightLineSetProjected();
   virtual ~TEveStraightLineSetProjected() {}

   virtual void SetProjection(TEveProjectionManager* mng, TEveProjectable* model);
   virtual void UpdateProjection();

   ClassDef(TEveStraightLineSetProjected, 1); // Projected copy of a TEveStraightLineSet.
};

#endif
