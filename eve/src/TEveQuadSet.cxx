// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveQuadSet.h>

#include <TEveManager.h>

#include <TColor.h>

#include <TBuffer3D.h>
#include <TBuffer3DTypes.h>
#include <TVirtualPad.h>
#include <TVirtualViewer3D.h>

#include <TROOT.h>
#include <TRandom.h>

//______________________________________________________________________________
// TEveQuadSet
//
// Supports various internal formats that result in rendering of a
// set of planar (lines, rectangles, hegagons with shared normal) objects.
//
// Names of internal structures and their variables use A, B and C as
// names for coordinate value-holders. Typical assignment is A->X,
// B->Y, C->Z but each render mode can override this convention and
// impose y or x as a fixed (third or C) coordinate. Alphabetic order
// is obeyed in this correspondence.
//
// For quad modes the deltas are expected to be positive.
// For line modes negative deltas are ok.

ClassImp(TEveQuadSet)

//______________________________________________________________________________
TEveQuadSet::TEveQuadSet(const Text_t* n, const Text_t* t) :
   TEveDigitSet   (n, t),

   fQuadType  (QT_Undef),
   fDefWidth  (1),
   fDefHeight (1),
   fDefCoord  (0)
{}

//______________________________________________________________________________
TEveQuadSet::TEveQuadSet(QuadType_e quadType, Bool_t valIsCol, Int_t chunkSize,
                         const Text_t* n, const Text_t* t) :
   TEveDigitSet   (n, t),

   fQuadType  (QT_Undef),
   fDefWidth  (1),
   fDefHeight (1),
   fDefCoord  (0)
{
   Reset(quadType, valIsCol, chunkSize);
}

//______________________________________________________________________________
TEveQuadSet::~TEveQuadSet()
{}

/******************************************************************************/

//______________________________________________________________________________
Int_t TEveQuadSet::SizeofAtom(TEveQuadSet::QuadType_e qt)
{
   static const TEveException eH("TEveQuadSet::SizeofAtom ");

   switch (qt) {
      case QT_Undef:                return 0;
      case QT_FreeQuad:             return sizeof(QFreeQuad);
      case QT_RectangleXY:
      case QT_RectangleXZ:
      case QT_RectangleYZ:          return sizeof(QRect);
      case QT_RectangleXYFixedDim:  return sizeof(QRectFixDim);
      case QT_RectangleXYFixedZ:
      case QT_RectangleXZFixedY:
      case QT_RectangleYZFixedX:    return sizeof(QRectFixC);
      case QT_RectangleXYFixedDimZ:
      case QT_RectangleXZFixedDimY:
      case QT_RectangleYZFixedDimX: return sizeof(QRectFixDimC);
      case QT_LineXZFixedY:
      case QT_LineXYFixedZ:         return sizeof(QLineFixC);
      case QT_HexagonXY:
      case QT_HexagonYX:            return sizeof(QHex);
      default:                      throw(eH + "unexpected atom type.");
   }
   return 0;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveQuadSet::Reset(TEveQuadSet::QuadType_e quadType, Bool_t valIsCol, Int_t chunkSize)
{
   fQuadType     = quadType;
   fValueIsColor = valIsCol;
   fDefaultValue = valIsCol ? 0 : kMinInt;
   if (fOwnIds)
      ReleaseIds();
   fPlex.Reset(SizeofAtom(fQuadType), chunkSize);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveQuadSet::AddQuad(Float_t* verts)
{
   static const TEveException eH("TEveQuadSet::AddQuad ");

   if (fQuadType != QT_FreeQuad)
      throw(eH + "expect free quad-type.");

   QFreeQuad* fq = (QFreeQuad*) NewDigit();
   memcpy(fq->fVertices, verts, sizeof(fq->fVertices));
}

//______________________________________________________________________________
void TEveQuadSet::AddQuad(Float_t a, Float_t b)
{
   AddQuad(a, b, fDefCoord, fDefWidth, fDefHeight);
}

//______________________________________________________________________________
void TEveQuadSet::AddQuad(Float_t a, Float_t b, Float_t c)
{
   AddQuad(a, b, c, fDefWidth, fDefHeight);
}

//______________________________________________________________________________
void TEveQuadSet::AddQuad(Float_t a, Float_t b, Float_t w, Float_t h)
{
   AddQuad(a, b, fDefCoord, w, h);
}

//______________________________________________________________________________
void TEveQuadSet::AddQuad(Float_t a, Float_t b, Float_t c, Float_t w, Float_t h)
{
   static const TEveException eH("TEveQuadSet::AddAAQuad ");

   QOrigin& fq = * (QOrigin*) NewDigit();
   fq.fA = a; fq.fB = b;
   switch (fQuadType)
   {
      case QT_RectangleXY:
      case QT_RectangleXZ:
      case QT_RectangleYZ:
      {
         QRect& q = (QRect&) fq;
         q.fC = c; q.fW = w; q.fH = h;
         break;
      }

      case QT_RectangleXYFixedDim:
      {
         QRectFixDim& q =  (QRectFixDim&) fq;
         q.fC = c;
         break;
      }

      case QT_RectangleXYFixedZ:
      case QT_RectangleXZFixedY:
      case QT_RectangleYZFixedX:
      {
         QRectFixC& q = (QRectFixC&) fq;
         q.fW = w; q.fH = h;
         break;
      }

      case QT_RectangleXYFixedDimZ:
      case QT_RectangleXZFixedDimY:
      case QT_RectangleYZFixedDimX:
      {
         break;
      }

      default:
         throw(eH + "expect axis-aligned quad-type.");
   }
}

//______________________________________________________________________________
void TEveQuadSet::AddLine(Float_t a, Float_t b, Float_t w, Float_t h)
{
   static const TEveException eH("TEveQuadSet::AddLine ");

   QOrigin& fq = * (QOrigin*) NewDigit();
   fq.fA = a; fq.fB = b;
   switch (fQuadType)
   {
      case QT_LineXZFixedY:
      case QT_LineXYFixedZ: {
         QLineFixC& q = (QLineFixC&) fq;
         q.fDx = w; q.fDy = h;
         break;
      }
      default:
         throw(eH + "expect line quad-type.");
   }
}

//______________________________________________________________________________
void TEveQuadSet::AddHexagon(Float_t a, Float_t b, Float_t c, Float_t r)
{
   static const TEveException eH("TEveQuadSet::AddHexagon ");

   QOrigin& fq = * (QOrigin*) NewDigit();
   fq.fA = a; fq.fB = b;
   switch (fQuadType)
   {
      case QT_HexagonXY:
      case QT_HexagonYX: {
         QHex& q = (QHex&) fq;
         q.fC = c; q.fR = r;
         break;
      }
      default:
         throw(eH + "expect line quad-type.");
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveQuadSet::ComputeBBox()
{
   // Fill bounding-box information of the base-class TAttBBox (virtual method).
   // If member 'TEveFrameBox* fFrame' is set, frame's corners are used as bbox.

   static const TEveException eH("TEveQuadSet::ComputeBBox ");

   if (fFrame != 0)
   {
      BBoxInit();
      Int_t    n    = fFrame->GetFrameSize() / 3;
      Float_t *bbps = fFrame->GetFramePoints();
      for (int i=0; i<n; ++i, bbps+=3)
         BBoxCheckPoint(bbps);
   }
   else
   {
      if(fPlex.Size() == 0) {
         BBoxZero();
         return;
      }

      BBoxInit();
      if (fQuadType == QT_RectangleXYFixedZ    ||
          fQuadType == QT_RectangleXYFixedDimZ)
      {
         fBBox[4] = fDefCoord;
         fBBox[5] = fDefCoord;
      }
      else if (fQuadType == QT_RectangleXZFixedY    ||
               fQuadType == QT_RectangleXZFixedDimY)
      {
         fBBox[2] = fDefCoord;
         fBBox[3] = fDefCoord;
      }
      else if (fQuadType == QT_RectangleYZFixedX    ||
               fQuadType == QT_RectangleYZFixedDimX)
      {
         fBBox[0] = fDefCoord;
         fBBox[1] = fDefCoord;
      }

      TEveChunkManager::iterator qi(fPlex);

      switch (fQuadType)
      {

         case QT_FreeQuad:
         {
            while (qi.next()) {
               const Float_t* p =  ((QFreeQuad*) qi())->fVertices;
               BBoxCheckPoint(p); p += 3;
               BBoxCheckPoint(p); p += 3;
               BBoxCheckPoint(p); p += 3;
               BBoxCheckPoint(p);
            }
            break;
         }

         case QT_RectangleXY:
         {
            while (qi.next()) {
               QRect& q = * (QRect*) qi();
               if(q.fA        < fBBox[0]) fBBox[0] = q.fA;
               if(q.fA + q.fW > fBBox[1]) fBBox[1] = q.fA + q.fW;
               if(q.fB        < fBBox[2]) fBBox[2] = q.fB;
               if(q.fB + q.fH > fBBox[3]) fBBox[3] = q.fB + q.fH;
               if(q.fC        < fBBox[4]) fBBox[4] = q.fC;
               if(q.fC        > fBBox[5]) fBBox[5] = q.fC;
            }
            break;
         }

         case QT_RectangleXZ:
         {
            while (qi.next()) {
               QRect& q = * (QRect*) qi();
               if(q.fA        < fBBox[0]) fBBox[0] = q.fA;
               if(q.fA + q.fW > fBBox[1]) fBBox[1] = q.fA + q.fW;
               if(q.fB        < fBBox[4]) fBBox[4] = q.fB;
               if(q.fB + q.fH > fBBox[5]) fBBox[5] = q.fB + q.fH;
               if(q.fC        < fBBox[2]) fBBox[2] = q.fC;
               if(q.fC        > fBBox[3]) fBBox[3] = q.fC;
            }
            break;
         }

         case QT_RectangleYZ:
         {
            while (qi.next()) {
               QRect& q = * (QRect*) qi();
               if(q.fA        < fBBox[2]) fBBox[2] = q.fA;
               if(q.fA + q.fW > fBBox[3]) fBBox[3] = q.fA + q.fW;
               if(q.fB        < fBBox[4]) fBBox[4] = q.fB;
               if(q.fB + q.fH > fBBox[5]) fBBox[5] = q.fB + q.fH;
               if(q.fC        < fBBox[0]) fBBox[0] = q.fC;
               if(q.fC        > fBBox[1]) fBBox[1] = q.fC;
            }
            break;
         }

         case QT_RectangleXYFixedDim:
         {
            const Float_t& w = fDefWidth;
            const Float_t& h = fDefHeight;
            while (qi.next()) {
               QRectFixDim& q = * (QRectFixDim*) qi();
               if(q.fA     < fBBox[0]) fBBox[0] = q.fA;
               if(q.fA + w > fBBox[1]) fBBox[1] = q.fA + w;
               if(q.fB     < fBBox[2]) fBBox[2] = q.fB;
               if(q.fB + h > fBBox[3]) fBBox[3] = q.fB + h;
               if(q.fC     < fBBox[4]) fBBox[4] = q.fC;
               if(q.fC     > fBBox[5]) fBBox[5] = q.fC;
            }
            break;
         }

         case QT_RectangleXYFixedZ:
         {
            while (qi.next()) {
               QRectFixC& q = * (QRectFixC*) qi();
               if(q.fA        < fBBox[0]) fBBox[0] = q.fA;
               if(q.fA + q.fW > fBBox[1]) fBBox[1] = q.fA + q.fW;
               if(q.fB        < fBBox[2]) fBBox[2] = q.fB;
               if(q.fB + q.fH > fBBox[3]) fBBox[3] = q.fB + q.fH;
            }
            break;
         }

         case QT_RectangleXZFixedY:
         {
            while (qi.next()) {
               QRectFixC& q = * (QRectFixC*) qi();
               if(q.fA        < fBBox[0]) fBBox[0] = q.fA;
               if(q.fA + q.fW > fBBox[1]) fBBox[1] = q.fA + q.fW;
               if(q.fB        < fBBox[4]) fBBox[4] = q.fB;
               if(q.fB + q.fH > fBBox[5]) fBBox[5] = q.fB + q.fH;
            }
            break;
         }

         case QT_RectangleYZFixedX:
         {
            while (qi.next()) {
               QRectFixC& q = * (QRectFixC*) qi();
               if(q.fA        < fBBox[2]) fBBox[2] = q.fA;
               if(q.fA + q.fW > fBBox[3]) fBBox[3] = q.fA + q.fW;
               if(q.fB        < fBBox[4]) fBBox[4] = q.fB;
               if(q.fB + q.fH > fBBox[5]) fBBox[5] = q.fB + q.fH;
            }
            break;
         }

         case QT_RectangleXYFixedDimZ:
         {
            const Float_t& w = fDefWidth;
            const Float_t& h = fDefHeight;
            while (qi.next()) {
               QRectFixDimC& q = * (QRectFixDimC*) qi();
               if(q.fA     < fBBox[0]) fBBox[0] = q.fA;
               if(q.fA + w > fBBox[1]) fBBox[1] = q.fA + w;
               if(q.fB     < fBBox[2]) fBBox[2] = q.fB;
               if(q.fB + h > fBBox[3]) fBBox[3] = q.fB + h;
            }
            break;
         }

         case QT_RectangleXZFixedDimY:
         {
            const Float_t& w = fDefWidth;
            const Float_t& h = fDefHeight;
            while (qi.next()) {
               QRectFixDimC& q = * (QRectFixDimC*) qi();
               if(q.fA     < fBBox[0]) fBBox[0] = q.fA;
               if(q.fA + w > fBBox[1]) fBBox[1] = q.fA + w;
               if(q.fB     < fBBox[4]) fBBox[4] = q.fB;
               if(q.fB + h > fBBox[5]) fBBox[5] = q.fB + h;
            }
            break;
         }

         case QT_RectangleYZFixedDimX:
         {
            const Float_t& w = fDefWidth;
            const Float_t& h = fDefHeight;
            while (qi.next()) {
               QRectFixDimC& q = * (QRectFixDimC*) qi();
               if(q.fA     < fBBox[2]) fBBox[2] = q.fA;
               if(q.fA + w > fBBox[3]) fBBox[3] = q.fA + w;
               if(q.fB     < fBBox[4]) fBBox[4] = q.fB;
               if(q.fB + h > fBBox[5]) fBBox[5] = q.fB + h;
            }
            break;
         }

         // TEveLine modes

         case QT_LineXYFixedZ:
         {
            while (qi.next()) {
               QLineFixC& q = * (QLineFixC*) qi();
               BBoxCheckPoint(q.fA,         q.fB,         fDefCoord);
               BBoxCheckPoint(q.fA + q.fDx, q.fB + q.fDy, fDefCoord);
            }
            break;
         }

         case QT_LineXZFixedY:
         {
            while (qi.next()) {
               QLineFixC& q = * (QLineFixC*) qi();
               BBoxCheckPoint(q.fA,         fDefCoord, q.fB);
               BBoxCheckPoint(q.fA + q.fDx, fDefCoord, q.fB + q.fDy);
            }
            break;
         }

         // Hexagon modes

         // Ignore 'slight' difference, assume square box for both cases.
         case QT_HexagonXY:
         case QT_HexagonYX:
         {
            while (qi.next()) {
               QHex& q = * (QHex*) qi();
               BBoxCheckPoint(q.fA-q.fR, q.fB-q.fR, q.fC);
               BBoxCheckPoint(q.fA+q.fR, q.fB+q.fR, q.fC);
            }
            break;
         }

         default:
         {
            throw(eH + "unsupported quad-type.");
         }

      } // end switch quad-type
   } // end if frame ... else ...

   AssertBBoxExtents(0.001);
}
