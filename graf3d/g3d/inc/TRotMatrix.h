// @(#)root/g3d:$Id$
// Author: Rene Brun   14/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRotMatrix
#define ROOT_TRotMatrix


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRotMatrix                                                           //
//                                                                      //
// Rotation Matrix for 3-D geometry objects.                            //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"


class TRotMatrix  : public TNamed {
private:
   virtual      void  SetReflection();   // Set the "reflection" flag if det < 0

protected:
   Int_t        fNumber;      //Rotation matrix number
   Int_t        fType;        //Type of matrix (0=identity, 1=reflexion, 2=otherwise)
   Double_t     fTheta;       //theta angle
   Double_t     fPhi;         //phi angle
   Double_t     fPsi;         //psi angle
   Double_t     fMatrix[9];   //Rotation matrix

public:
   //TRotMatrix status bits
   enum {
      kReflection = BIT(23)   //  "Reflection" bit
   };

   TRotMatrix();
   TRotMatrix(const char *name, const char *title, Double_t *matrix);
   TRotMatrix(const char *name, const char *title, Double_t theta, Double_t phi, Double_t psi);
   TRotMatrix(const char *name, const char *title, Double_t theta1, Double_t phi1,
                                           Double_t theta2, Double_t phi2,
                                           Double_t theta3, Double_t phi3);
   virtual ~TRotMatrix();
   virtual Double_t  Determinant() const ;   // returns the determinant of this matrix
   virtual Double_t* GetMatrix()         {return &fMatrix[0];}
   virtual Int_t     GetNumber()   const {return fNumber;}
   virtual Int_t     GetType()     const {return fType;}
   virtual Double_t  GetTheta()    const {return fTheta;}
   virtual Double_t  GetPhi()      const {return fPhi;}
   virtual Double_t  GetPsi()      const {return fPsi;}
   virtual Double_t* GetGLMatrix(Double_t *rGLMatrix) const ;  // Convert this matrix to the OpenGL [4x4]
   virtual Bool_t    IsReflection() const {return TestBit(kReflection);}  // Return kTRUE if this matrix defines the reflection
   virtual const     Double_t* SetAngles(Double_t theta1, Double_t phi1,Double_t theta2, Double_t phi2, Double_t theta3, Double_t phi3);
   virtual void      SetMatrix(const Double_t *matrix);
   virtual void      SetName(const char *name);

   ClassDef(TRotMatrix,2)  //Rotation Matrix for 3-D geometry objects
};

inline void TRotMatrix::SetName(const char *) { }

#endif
