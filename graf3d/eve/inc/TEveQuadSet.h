// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveQuadSet
#define ROOT_TEveQuadSet

#include "TEveDigitSet.h"

class TRandom;

class TEveQuadSet : public TEveDigitSet
{
   friend class TEveQuadSetGL;

   TEveQuadSet(const TEveQuadSet&);            // Not implemented
   TEveQuadSet& operator=(const TEveQuadSet&); // Not implemented

public:
   enum EQuadType_e {
      kQT_Undef,                // unknown-ignored
      kQT_FreeQuad,             // arbitrary quad: specify 4*(x,y,z) quad corners
      kQT_RectangleXY,          // rectangle in x-y plane: specify x, y, z, w, h
      kQT_RectangleXZ,          // rectangle in x-z plane: specify x, y, z, w, h
      kQT_RectangleYZ,          // rectangle in y-z plane: specify x, y, z, w, h
      kQT_RectangleXYFixedDim,  // rectangle in x-y plane: specify x, y, z; w, h taken from fDefWidth/Height
      kQT_RectangleXYFixedZ,    // rectangle in x-y plane: specify x, y, w, h; z taken from fDefCoord
      kQT_RectangleXZFixedY,    // rectangle in x-z plane: specify x, z, w, h; y taken from fDefCoord
      kQT_RectangleYZFixedX,    // rectangle in y-z plane: specify y, z, w, h; x taken from fDefWidth/Height/Coord
      kQT_RectangleXYFixedDimZ, // rectangle in x-y plane: specify x, y; w, h, z taken from fDefWidth/Height/Coord
      kQT_RectangleXZFixedDimY, // rectangle in x-z plane: specify x, z; w, h, y taken from fDefWidth/Height/Coord
      kQT_RectangleYZFixedDimX, // rectangle in y-z plane: specify y, z; w, h, x taken from fDefWidth/Height/Coord
      kQT_Rectangle_End,
      // line modes (needed for uniform handling of silicon-strip digits)
      kQT_LineXYFixedZ,         // line in x-y plane: specify x, y, w(dx), h(dy); z taken from fDefCoord
      kQT_LineXZFixedY,         // line in x-z plane: specify x, z, w(dx), h(dz); y taken from fDefCoord
      kQT_Line_End,
      // hexagon modes
      kQT_HexagonXY,            // horizontal hexagon: specify x, y, z, r
      kQT_HexagonYX,            // vertical   hexagon: specify x, y, z, r
      kQT_Hexagon_End
      // circle modes:
      // QT_CircleXY,          // specify r, z
      // QT_CircleXYFixedZ,    // specify r
      // QT_CircleXYFixedR,    // specify z
   };

   struct QFreeQuad_t     : public DigitBase_t     { Float_t fVertices[12]; };

   struct QOrigin_t       : public DigitBase_t     { Float_t fA, fB; };

   struct QRectFixDimC_t  : public QOrigin_t       {};

   struct QRectFixDim_t   : public QRectFixDimC_t  { Float_t fC; };

   struct QRectFixC_t     : public QRectFixDimC_t  { Float_t fW, fH; };

   struct QRect_t         : public QRectFixDim_t   { Float_t fW, fH; };

   struct QLineFixC_t     : public QOrigin_t       { Float_t fDx, fDy; };

   struct QHex_t          : public QOrigin_t       { Float_t fC, fR; };

protected:
   EQuadType_e       fQuadType;

   Float_t           fDefWidth;     // Breadth assigned to first coordinate  (A)
   Float_t           fDefHeight;    // Breadth assigned to second coordinate (B)
   Float_t           fDefCoord;     // Default value for third coordinate    (C)

   static Int_t SizeofAtom(EQuadType_e qt);

public:
   TEveQuadSet(const char* n="TEveQuadSet", const char* t="");
   TEveQuadSet(EQuadType_e quadType, Bool_t valIsCol, Int_t chunkSize,
               const char* n="TEveQuadSet", const char* t="");
   virtual ~TEveQuadSet() {}

   void Reset(EQuadType_e quadType, Bool_t valIsCol, Int_t chunkSize);

   Float_t GetDefWidth()  const { return fDefWidth;  }
   Float_t GetDefHeight() const { return fDefHeight; }
   Float_t GetDefCoord()  const { return fDefCoord;  }

   void SetDefWidth(Float_t v)  { fDefWidth  = v; }
   void SetDefHeight(Float_t v) { fDefHeight = v; }
   void SetDefCoord(Float_t v)  { fDefCoord  = v; }

   // --------------------------------

   void AddQuad(Float_t* verts);

   void AddQuad(Float_t a, Float_t b);
   void AddQuad(Float_t a, Float_t b, Float_t c);
   void AddQuad(Float_t a, Float_t b, Float_t w, Float_t h);
   void AddQuad(Float_t a, Float_t b, Float_t c, Float_t w, Float_t h);

   void AddLine(Float_t a, Float_t b, Float_t w, Float_t h);

   void AddHexagon(Float_t a, Float_t b, Float_t z, Float_t r);

   // Wrappers to make transition to TEveDigitSet as base easier
   void QuadValue(Int_t value) { DigitValue(value); }
   void QuadColor(Color_t ci)  { DigitColor(ci); }
   void QuadColor(UChar_t r, UChar_t g, UChar_t b, UChar_t a=255) { DigitColor(r, g, b, a); }
   void QuadId(TObject* id)    { DigitId(id); }

   // --------------------------------

   // void Test(Int_t nquads);

   virtual void ComputeBBox();

   // virtual void Paint(Option_t* option="");

   ClassDef(TEveQuadSet, 0); // Collection of 2D primitives (rectangles, hexagons, or lines); each primitive can be assigned a signal value and a TRef.
};

#endif
