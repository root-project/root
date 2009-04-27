// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEvePolygonSetProjected
#define ROOT_TEvePolygonSetProjected

#include "TEveElement.h"
#include "TEveProjectionBases.h"

#include "TNamed.h"
#include "TAtt3D.h"
#include "TAttBBox.h"
#include "TColor.h"
#include "TEveVSDStructs.h"

class TBuffer3D;

class TEveVector;

class TEvePolygonSetProjected :  public TEveElementList,
                                 public TEveProjected,
                                 public TAtt3D,
                                 public TAttBBox
{
   friend class TEvePolygonSetProjectedGL;
   friend class TEvePolygonSetProjectedEditor;
private:
   TEvePolygonSetProjected(const TEvePolygonSetProjected&);            // Not implemented
   TEvePolygonSetProjected& operator=(const TEvePolygonSetProjected&); // Not implemented

protected:
   struct Polygon_t
   {
      Int_t     fNPnts;  // number of points
      Int_t*    fPnts;   // point indices

      Polygon_t() : fNPnts(0), fPnts(0) {}
     virtual ~Polygon_t() { delete [] fPnts; fNPnts=0; fPnts=0;}

      Polygon_t& operator=(const Polygon_t& x)
      { fNPnts = x.fNPnts; fPnts = x.fPnts; return *this; }

      Int_t FindPoint(Int_t pi)
      { for (Int_t i=0; i<fNPnts; ++i) if (fPnts[i] == pi) return i; return -1; }
   };

   typedef std::list<Polygon_t>                    vpPolygon_t;
   typedef vpPolygon_t::iterator                   vpPolygon_i;
   typedef vpPolygon_t::const_iterator             vpPolygon_ci;

private:
   TBuffer3D*   fBuff;   // buffer of projectable object

   Bool_t       IsFirstIdxHead(Int_t s0, Int_t s1);
   Float_t      AddPolygon(std::list<Int_t, std::allocator<Int_t> >& pp, std::list<Polygon_t, std::allocator<Polygon_t> >& p);

   Int_t*       ProjectAndReducePoints();
   Float_t      MakePolygonsFromBP(Int_t* idxMap);
   Float_t      MakePolygonsFromBS(Int_t* idxMap);

protected:
   vpPolygon_t  fPols;     // polygons
   vpPolygon_t  fPolsBS;   // polygons build from TBuffer3D segments
   vpPolygon_t  fPolsBP;   // polygons build from TBuffer3D polygons

   Int_t        fNPnts;    // number of reduced and projected points
   TEveVector*  fPnts;     // reduced and projected points

   Color_t      fFillColor; // fill color of polygons
   Color_t      fLineColor; // outline color of polygons
   Float_t      fLineWidth; // outline width of polygons

public:
   TEvePolygonSetProjected(const char* n="TEvePolygonSetProjected", const char* t="");
   virtual ~TEvePolygonSetProjected();

   virtual void    SetProjection(TEveProjectionManager* mng, TEveProjectable* model);
   virtual void    SetDepth(Float_t d);
   virtual void    UpdateProjection();

   void            ProjectBuffer3D();

   virtual void    ComputeBBox();
   virtual void    Paint(Option_t* option = "");

   virtual void    DumpPolys() const;
   void            DumpBuffer3D();

   // Rendering parameters.
   virtual Bool_t  CanEditMainColor() const { return kTRUE; }
   virtual void    SetMainColor(Color_t color);

   virtual Bool_t  CanEditMainTransparency() const { return kTRUE; }

   virtual Color_t GetFillColor() const { return fFillColor; }
   virtual Color_t GetLineColor() const { return fLineColor; }
   virtual Float_t GetLineWidth() const { return fLineWidth;}

   virtual void    SetFillColor(Color_t c)  { fFillColor = c; }
   virtual void    SetLineColor(Color_t c)  { fLineColor = c; }
   virtual void    SetLineWidth(Float_t lw) { fLineWidth = lw;}

   ClassDef(TEvePolygonSetProjected,0); // Set of projected polygons with outline; typically produced from a TBuffer3D.

};

#endif
