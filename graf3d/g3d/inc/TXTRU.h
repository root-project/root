//@@(#)root/g3d:$Id$
// Author: Robert Hatcher (rhatcher@fnal.gov) 2000.09.06

////////////////////////////////////////////////////////////////////////////
// $Id$
//
// TXTRU
//
// TXTRU is an extrusion with fixed outline shape in x-y and a sequence
// of z extents (segments).  The overall scale of the outline scales
// linearly between z points and the center can have an x-y offset.
//
// Author:  R. Hatcher 2000.04.21
//
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TXTRU
#define ROOT_TXTRU

#include "TShape.h"

class TXTRU : public TShape {
public:
   TXTRU();
   TXTRU(const char *name, const char *title, const char *material,
         Int_t nyx, Int_t nz);
   TXTRU(const TXTRU &xtru);
   virtual ~TXTRU();
   TXTRU& operator=(const TXTRU& rhs);

   virtual void     Copy(TObject &xtru) const;
   virtual void     DefineSection(Int_t secNum, Float_t z, Float_t scale=1.,
                                  Float_t x0=0., Float_t y0=0.);
   virtual void     DefineVertex(Int_t pointNum, Float_t x, Float_t y);
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual const    TBuffer3D &GetBuffer3D(Int_t) const;
   virtual Int_t    GetNxy() const { return fNxy; }
   virtual Int_t    GetNz() const { return fNz; }
   virtual Float_t  GetOutlinePointX(Int_t pointNum) const;
   virtual Float_t  GetOutlinePointY(Int_t pointNum) const;
   virtual Float_t  GetSectionX0(Int_t secNum) const;
   virtual Float_t  GetSectionY0(Int_t secNum) const;
   virtual Float_t  GetSectionScale(Int_t secNum) const;
   virtual Float_t  GetSectionZ(Int_t secNum) const;
   virtual Float_t *GetXvtx() const {return fXvtx; }
   virtual Float_t *GetYvtx() const {return fYvtx; }
   virtual Float_t *GetZ() const {return fZ; }
   virtual Float_t *GetScale() const {return fScale; }
   virtual Float_t *GetX0() const {return fX0; }
   virtual Float_t *GetY0() const {return fY0; }
   virtual void     Print(Option_t *option="") const;
   virtual void     Sizeof3D() const;
   void             SplitConcavePolygon(Bool_t split = kTRUE);
   virtual void     TruncateNxy(Int_t npts);
   virtual void     TruncateNz(Int_t npts);

protected:
   void            CheckOrdering();
   virtual void    SetPoints(Double_t *points) const;

   Int_t       fNxy{0};             // number of x-y points in the cross section
   Int_t       fNxyAlloc{0};        // number of x-y points allocated
   Int_t       fNz{0};              // number of z planes
   Int_t       fNzAlloc{0};         // number of z planes allocated
   Float_t    *fXvtx{nullptr};      //[fNxyAlloc] array of x positions
   Float_t    *fYvtx{nullptr};      //[fNxyAlloc] array of y positions
   Float_t    *fZ{nullptr};         //[fNzAlloc] array of z planes
   Float_t    *fScale{nullptr};     //[fNzAlloc] array of scale factors (for each z)
   Float_t    *fX0{nullptr};        //[fNzAlloc] array of x offsets (for each z)
   Float_t    *fY0{nullptr};        //[fNzAlloc] array of y offsets (for each z)

   enum EXYChecked {kUncheckedXY, kMalformedXY,
                    kConvexCCW,   kConvexCW,
                    kConcaveCCW,  kConcaveCW};
   enum EZChecked  {kUncheckedZ,  kMalformedZ,
                    kConvexIncZ,  kConvexDecZ,
                    kConcaveIncZ, kConcaveDecZ};

   EXYChecked  fPolygonShape{kUncheckedXY};   // CCW vs. CW, convex vs. concave
   EZChecked   fZOrdering{kUncheckedZ};      // increasing or decreasing

   // Concave polygon division (into convex polygons) is not yet supported
   // but if split one gets correct solid rendering but extra lines
   // in wire mode; if not split....the converse.
   Bool_t      fSplitConcave{kFALSE};

private:
   void DumpPoints(int npoints, float *pointbuff) const;
   void DumpSegments(int nsegments, int *segbuff) const;
   void DumpPolygons(int npolygons, int *polybuff, int buffsize) const;

   ClassDef(TXTRU,1)  //TXTRU shape
};

#endif
