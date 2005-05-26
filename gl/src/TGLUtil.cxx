// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header!!!

#include "TGLUtil.h"
#include "TGLIncludes.h"
#include "Riostream.h"

ClassImp(TGLVertex3)

//______________________________________________________________________________
TGLVertex3::TGLVertex3()
{ 
   Fill(0.0); 
}

//______________________________________________________________________________
TGLVertex3::TGLVertex3(Double_t x, Double_t y, Double_t z) 
{ 
   Set(x,y,z); 
}

//______________________________________________________________________________
TGLVertex3::TGLVertex3(const TGLVertex3 & other)
{ 
   Set(other); 
}

//______________________________________________________________________________
TGLVertex3::~TGLVertex3()
{
}

//______________________________________________________________________________
void TGLVertex3::Dump() const
{
   std::cout << "(" << fVals[0] << "," << fVals[1] << "," << fVals[2] << ")" << std::endl;
}

ClassImp(TGLVector3)
      
//______________________________________________________________________________
TGLVector3::TGLVector3() :
   TGLVertex3()
{ 
}

//______________________________________________________________________________
TGLVector3::TGLVector3(Double_t x, Double_t y, Double_t z) : 
   TGLVertex3(x, y, z) 
{
}

//______________________________________________________________________________
TGLVector3::TGLVector3(const TGLVector3 & other) : 
   TGLVertex3(other.fVals[0], other.fVals[1], other.fVals[2]) 
{
}

//______________________________________________________________________________
TGLVector3::~TGLVector3() 
{
}

ClassImp(TGLRect)

//______________________________________________________________________________
TGLRect::TGLRect() :
   fX(0), fY(0), fWidth(0), fHeight(0)
{ 
}

//______________________________________________________________________________
TGLRect::TGLRect(Int_t x, Int_t y, UInt_t width, UInt_t height) : 
   fX(x), fY(y), fWidth(width), fHeight(height) 
{
}

//______________________________________________________________________________
TGLRect::~TGLRect() 
{
}
  
//______________________________________________________________________________
void TGLRect::Expand(Int_t x, Int_t y) 
{ 
   Int_t delX = x - fX;
   Int_t delY = y - fY;

   if (delX>static_cast<Int_t>(fWidth)) {
      fWidth = delX;
   }      
   if (delY>static_cast<Int_t>(fHeight)) {
      fHeight = delY;
   }      
   
   if (delX<0) {
      fX = x; 
      fWidth += -delX;
   }
   if (delY<0) {
      fY = y; 
      fHeight += -delY;
   }
}     

//______________________________________________________________________________
Double_t TGLRect::Aspect() const 
{ 
   if (fHeight == 0) { 
      return 0.0; 
   } else {
      return static_cast<Double_t>(fWidth) / static_cast<Double_t>(fHeight); 
   }
}

//______________________________________________________________________________
EOverlap TGLRect::Overlap(const TGLRect & other) const
{
   if ((fX <= other.fX) && (fX + fWidth >= other.fX + other.fWidth) &&
       (fY <= other.fY) && (fY +fHeight >= other.fY + other.fHeight)) {
      return kInside;
   } 
   else if ((fX >= other.fX + static_cast<Int_t>(other.fWidth)) || 
             (fX + static_cast<Int_t>(fWidth) <= other.fX) ||
             (fY >= other.fY + static_cast<Int_t>(other.fHeight)) ||
             (fY + static_cast<Int_t>(fHeight) <= other.fY)) {
      return kOutside;
   } else {
      return kPartial;
   }
}

ClassImp(TGLPlane)

//______________________________________________________________________________
TGLPlane::TGLPlane()
{
   fVals[0] = 1.0;
   fVals[1] = 1.0;
   fVals[2] = 1.0;
   fVals[0] = 0.0;
}

//______________________________________________________________________________
TGLPlane::TGLPlane( Double_t a, Double_t b, Double_t c, Double_t d )
{
   fVals[0] = a;
   fVals[1] = b;
   fVals[2] = c;
   fVals[3] = d;
   Normalise();
}

//______________________________________________________________________________
TGLPlane::~TGLPlane()
{
}

ClassImp(TGLMatrix)

//______________________________________________________________________________
TGLMatrix::TGLMatrix() 
{ 
   SetIdentity(); 
}

//______________________________________________________________________________
TGLMatrix::TGLMatrix(Double_t x, Double_t y, Double_t z) 
{ 
   SetTranslation(x,y,z); 
}

//______________________________________________________________________________
TGLMatrix::TGLMatrix(const Double_t vals[16]) 
{ 
   Set(vals); 
}

//______________________________________________________________________________
TGLMatrix::TGLMatrix(const TGLMatrix & other) 
{ 
   *this = other; 
}

//______________________________________________________________________________
TGLMatrix::~TGLMatrix() 
{
}

//______________________________________________________________________________
void TGLMatrix::Set(const Double_t vals[16])
{
   for (UInt_t i=0; i < 16; i++) {
      fVals[i] = vals[i];
   }
}

//______________________________________________________________________________
void TGLMatrix::SetIdentity()
{
   fVals[0] = 1.0; fVals[4] = 0.0; fVals[8 ] = 0.0; fVals[12] = 0.0;
   fVals[1] = 0.0; fVals[5] = 1.0; fVals[9 ] = 0.0; fVals[13] = 0.0;
   fVals[2] = 0.0; fVals[6] = 0.0; fVals[10] = 1.0; fVals[14] = 0.0;
   fVals[3] = 0.0; fVals[7] = 0.0; fVals[11] = 0.0; fVals[15] = 1.0;
}

//______________________________________________________________________________
void TGLMatrix::SetTranslation(Double_t x, Double_t y, Double_t z)
{
   fVals[0] = 1.0; fVals[4] = 0.0; fVals[8 ] = 0.0; fVals[12] = x;
   fVals[1] = 0.0; fVals[5] = 1.0; fVals[9 ] = 0.0; fVals[13] = y;
   fVals[2] = 0.0; fVals[6] = 0.0; fVals[10] = 1.0; fVals[14] = z;
   fVals[3] = 0.0; fVals[7] = 0.0; fVals[11] = 0.0; fVals[15] = 1.0;
}

// TODO: Move this to the TGeo side and remove
//______________________________________________________________________________
void TGLMatrix::InvRot()
{
	Double_t temp = fVals[4];
	fVals[4] = fVals[1];
	fVals[1] = temp;
	temp = fVals[8];
	fVals[8] = fVals[2];
	fVals[2] = temp;
	temp = fVals[9];
	fVals[9] = fVals[6];
	fVals[6] = temp;
}

//______________________________________________________________________________
void TGLMatrix::TransformVertex(TGLVertex3 & vertex) const
{
   TGLVertex3 orig = vertex;
   for (UInt_t i = 0; i < 3; i++) {
      vertex[i] = orig[0] * fVals[0+i] + orig[1] * fVals[4+i] +
                  orig[2] * fVals[8+i] + fVals[12+i];
   }      
}

//______________________________________________________________________________
void TGLMatrix::Dump() const
{
   std::cout.precision(6);
   for (Int_t x = 0; x < 4; x++) {
      std::cout << "[ ";
      for (Int_t y = 0; y < 4; y++) {
         std::cout << fVals[y*4 + x] << " ";
      }
      std::cout << "]" << std::endl;
   }
}

ClassImp(TGLUtil)

//______________________________________________________________________________
void TGLUtil::CheckError()
{
   //TODO: Make for debug only! Which symbol?
   GLenum errCode; // TODO: Type?
   const GLubyte *errString;// TODO: Type?
   
   if ((errCode = glGetError()) != GL_NO_ERROR) {
      errString = gluErrorString(errCode);
      std::cout << "OpenGL Error: " << errString << std::endl;
      //assert(kFALSE);
   }
}
