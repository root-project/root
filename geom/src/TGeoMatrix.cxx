/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - Wed 24 Oct 2001 09:46:13 AM CEST

////////////////////////////////////////////////////////////////////////////////
// Geometrical transformation package
//
//
//
//
//Begin_Html
/*
<img src="gif/t_transf.jpg">
*/
//End_Html
#include "TObjArray.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"

TGeoIdentity *gGeoIdentity = 0;

// statics and globals

ClassImp(TGeoMatrix)
ClassImp(TGeoTranslation)
ClassImp(TGeoRotation)
ClassImp(TGeoScale)
ClassImp(TGeoCombiTrans)
ClassImp(TGeoGenTrans)
ClassImp(TGeoIdentity)

//-----------------------------------------------------------------------------
TGeoMatrix::TGeoMatrix()
{
// dummy constructor
}
//-----------------------------------------------------------------------------
TGeoMatrix::TGeoMatrix(const char *name)
           :TNamed(name, "")
{
// Constructor
   if (!gGeoManager) gGeoManager = new TGeoManager("Geometry", "Default geometry");
}
//-----------------------------------------------------------------------------
Int_t TGeoMatrix::GetByteCount()
{
// Get total size in bytes of this
   Int_t count = 4+28+strlen(GetName())+strlen(GetTitle()); // fId + TNamed
   if (IsTranslation()) count += 12;
   if (IsScale()) count += 12;
   if (IsCombi() || IsGeneral()) count += 4 + 36;
   return count;
}
//-----------------------------------------------------------------------------
void TGeoMatrix::GetHomogenousMatrix(Double_t *hmat)
{
// The homogenous matrix associated with the transformation is used for
// piling up's and visualization. A homogenous matrix is a 4*4 array
// containing the translation, the rotation and the scale components
//
//          | R00*sx  R01    R02    dx |
//          | R10    R11*sy  R12    dy |
//          | R20     R21   R22*sz  dz |
//          |  0       0      0      1 |
//
//   where Rij is the rotation matrix, (sx, sy, sz) is the scale 
// transformation and (dx, dy, dz) is the translation.
   Double_t *hmatrix = hmat;
   const Double_t *mat = GetRotationMatrix();
   for (Int_t i=0; i<3; i++) {
      memcpy(hmatrix, mat, 3*sizeof(Double_t));
      mat     += 3;
      hmatrix += 3;
      *hmatrix = 0.0;
      hmatrix++; 
   }
   memcpy(hmatrix, GetTranslation(), 3*sizeof(Double_t));
   hmatrix = hmat;
   if (IsScale()) {
      for (Int_t i=0; i<3; i++) {
         *hmatrix *= GetScale()[i];
         hmatrix  += 5;
      }
   }
}
//-----------------------------------------------------------------------------
void TGeoMatrix::LocalToMaster(const Double_t *local, Double_t *master)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix inverse
  const Double_t *tr = GetTranslation();
  const Double_t *rot = GetRotationMatrix();
  for (Int_t i=0; i<3; i++) {
      master[i] = tr[i] 
                 + local[0]*rot[3*i]
                 + local[1]*rot[3*i+1]
                 + local[2]*rot[3*i+2];
   }
}
//-----------------------------------------------------------------------------
void TGeoMatrix::LocalToMasterVect(const Double_t *local, Double_t *master)
{
// convert a vector by multiplying its column vector (x, y, z, 1) to matrix inverse
  const Double_t *rot = GetRotationMatrix();
  for (Int_t i=0; i<3; i++) {
      master[i] = local[0]*rot[3*i]
                 + local[1]*rot[3*i+1]
                 + local[2]*rot[3*i+2];
   }
}
//-----------------------------------------------------------------------------
void TGeoMatrix::LocalToMasterBomb(const Double_t *local, Double_t *master)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix inverse
  const Double_t *tr = GetTranslation();
  Double_t bombtr[3];
  gGeoManager->BombTranslation(tr, &bombtr[0]);
  const Double_t *rot = GetRotationMatrix();
  for (Int_t i=0; i<3; i++) {
      master[i] = bombtr[i] 
                 + local[0]*rot[3*i]
                 + local[1]*rot[3*i+1]
                 + local[2]*rot[3*i+2];
   }
}
//-----------------------------------------------------------------------------
void TGeoMatrix::MasterToLocal(const Double_t *master, Double_t *local)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix
  const Double_t *tr = GetTranslation();
  const Double_t *rot = GetRotationMatrix();
    for (Int_t i=0; i<3; i++) {
       local[i] =  (master[0]-tr[0])*rot[i]
                 + (master[1]-tr[1])*rot[i+3]
                 + (master[2]-tr[2])*rot[i+6];
   }
}
//-----------------------------------------------------------------------------
void TGeoMatrix::MasterToLocalVect(const Double_t *master, Double_t *local)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix
  const Double_t *rot = GetRotationMatrix();
    for (Int_t i=0; i<3; i++) {
       local[i] =  master[0]*rot[i]
                 + master[1]*rot[i+3]
                 + master[2]*rot[i+6];
   }
}
//-----------------------------------------------------------------------------
void TGeoMatrix::MasterToLocalBomb(const Double_t *master, Double_t *local)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix
  const Double_t *tr = GetTranslation();
  Double_t bombtr[3];
  gGeoManager->UnbombTranslation(tr, &bombtr[0]);
  const Double_t *rot = GetRotationMatrix();
    for (Int_t i=0; i<3; i++) {
       local[i] =  (master[0]-bombtr[0])*rot[i]
                 + (master[1]-bombtr[1])*rot[i+3]
                 + (master[2]-bombtr[2])*rot[i+6];
   }
}
//-----------------------------------------------------------------------------
void TGeoMatrix::Print(Option_t *) const
{
// print the matrix in 4x4 format
   const Double_t *rot = GetRotationMatrix();
   const Double_t *tr  = GetTranslation();
   const Double_t *sc  = GetScale();
   printf("matrix %s - translation : %i  rotation : %i  scale : %i\n", GetName(),(Int_t)IsTranslation(),
          (Int_t)IsRotation(), (Int_t)IsScale());
   printf(" %g %g %g %g\n", rot[0], rot[1], rot[2], (Double_t)0); 
   printf(" %g %g %g %g\n", rot[3], rot[4], rot[5], (Double_t)0); 
   printf(" %g %g %g %g\n", rot[6], rot[7], rot[8], (Double_t)0); 

   printf(" %g %g %g %g\n", tr[0], tr[1], tr[2], (Double_t)1);
   if (IsScale()) printf("Scale : %g %g %g\n", sc[0], sc[1], sc[2]);
}
//-----------------------------------------------------------------------------
void TGeoMatrix::SetDefaultName()
{
// If no name was supplied in the ctor, the type of transformation is checked.
// A letter will be prepended to the name :
//   t - translation
//   r - rotation
//   s - scale
//   c - combi (translation + rotation)
//   g - general (tr+rot+scale)
// The index of the transformation in gGeoManager list of transformations will
// be appended.
   if (strlen(GetName())) return;
   char type = 'n';
   if (IsTranslation()) type = 't'; 
   if (IsRotation()) type = 'r';
   if (IsScale()) type = 's';
   if (IsCombi()) type = 'c';
   if (IsGeneral()) type = 'g';
   Int_t index = gGeoManager->GetListOfMatrices()->GetSize() - 1;
   Int_t digits = 1;
   Int_t num = 10;
   while ((Int_t)(index/num)) {
      digits++;
      num *= 10;
   }
   char *name = new char[digits+2];
   sprintf(name, "%c%i", type, index);
   SetName(name);
}
//-----------------------------------------------------------------------------
TGeoTranslation::TGeoTranslation()
{
// Default constructor
   SetBit(kGeoTranslation);
   SetDefaultName();
   for (Int_t i=0; i<3; i++) fTranslation[i] = 0;
}
//-----------------------------------------------------------------------------
TGeoTranslation::TGeoTranslation(Double_t dx, Double_t dy, Double_t dz)
                :TGeoMatrix("")
{
// Default constructor defining the translation
   SetBit(kGeoTranslation);
   SetDefaultName();
   SetTranslation(dx, dy, dz);
   gGeoManager->AddTransformation(this);
}
//-----------------------------------------------------------------------------
void TGeoTranslation::Add(TGeoTranslation *other)
{
// Adding a translation to this one
   const Double_t *trans = other->GetTranslation();
   for (Int_t i=0; i<3; i++) 
      fTranslation[i] += trans[i];
}
//-----------------------------------------------------------------------------
void TGeoTranslation::Subtract(TGeoTranslation *other)
{
// Subtracting a translation from this one
   const Double_t *trans = other->GetTranslation();
   for (Int_t i=0; i<3; i++) 
      fTranslation[i] -= trans[i];
}
//-----------------------------------------------------------------------------
void TGeoTranslation::SetTranslation(Double_t dx, Double_t dy, Double_t dz)
{
// Set translation components
   fTranslation[0] = dx;
   fTranslation[1] = dy;
   fTranslation[2] = dz;
}
//-----------------------------------------------------------------------------
void TGeoTranslation::LocalToMaster(const Double_t *local, Double_t *master)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix inverse
  const Double_t *tr = GetTranslation();
  for (Int_t i=0; i<3; i++) 
      master[i] = tr[i] + local[i]; 
}
//-----------------------------------------------------------------------------
void TGeoTranslation::LocalToMasterVect(const Double_t *local, Double_t *master)
{
// convert a vector to MARS
   memcpy(master, local, 3*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
void TGeoTranslation::LocalToMasterBomb(const Double_t *local, Double_t *master)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix inverse
  const Double_t *tr = GetTranslation();
  Double_t bombtr[3];
  gGeoManager->BombTranslation(tr, &bombtr[0]);
  for (Int_t i=0; i<3; i++) 
      master[i] = bombtr[i] + local[i]; 
}
//-----------------------------------------------------------------------------
void TGeoTranslation::MasterToLocal(const Double_t *master, Double_t *local)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix
  const Double_t *tr = GetTranslation();
    for (Int_t i=0; i<3; i++) 
       local[i] =  master[i]-tr[i];
}
//-----------------------------------------------------------------------------
void TGeoTranslation::MasterToLocalVect(const Double_t *master, Double_t *local)
{
// convert a vector from MARS to local
   memcpy(local, master, 3*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
void TGeoTranslation::MasterToLocalBomb(const Double_t *master, Double_t *local)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix
  const Double_t *tr = GetTranslation();
  Double_t bombtr[3];
  gGeoManager->UnbombTranslation(tr, &bombtr[0]);
    for (Int_t i=0; i<3; i++) 
       local[i] =  master[i]-bombtr[i];
}
//-----------------------------------------------------------------------------
TGeoRotation::TGeoRotation()
{
   SetBit(kGeoRotation);
   SetDefaultName();
// dummy ctor
}
//-----------------------------------------------------------------------------
TGeoRotation::TGeoRotation(const char *name)
             :TGeoMatrix(name)
{
// Default rotation constructor
   for (Int_t i=0; i<9; i++) {
      if (i%4) fRotationMatrix[i] = 0;
      else fRotationMatrix[i] = 1.0;
   }
   SetBit(kGeoRotation);
   SetDefaultName();
   gGeoManager->AddTransformation(this);
}
//-----------------------------------------------------------------------------
TGeoRotation::TGeoRotation(const char *name, Double_t alpha, Double_t beta, Double_t gamma)
             :TGeoMatrix(name)
{
// Default rotation constructor with Euler angles. Gamma is the rotation angle about
// Z axis (clockwise) and is done first, beta is the rotation about Y and is done
// second, alpha is the rotation angle about X and is done third. All angles are in
// degrees.
   SetBit(kGeoRotation);
   SetDefaultName();
   SetAngles(alpha, beta, gamma);
   CheckMatrix();
   gGeoManager->AddTransformation(this);
}
//-----------------------------------------------------------------------------
TGeoRotation::TGeoRotation(const char *name, Double_t theta1, Double_t phi1, Double_t theta2, Double_t phi2,
                           Double_t theta3, Double_t phi3)
             :TGeoMatrix(name)
{
// Rotation constructor a la GEANT3. Angles theta(i), phi(i) are the polar and azimuthal
// angles of the (i) axis of the rotated system with respect to the initial non-rotated
// system.
//   Example : the identity matrix (no rotation) is composed by
//      theta1=90, phi1=0, theta2=90, phi2=90, theta3=0, phi3=0
   SetBit(kGeoRotation);
   SetDefaultName();
   SetAngles(theta1, phi1, theta2, phi2, theta3, phi3);
   CheckMatrix();
   gGeoManager->AddTransformation(this);
}
//-----------------------------------------------------------------------------
void TGeoRotation::Clear(Option_t *)
{
// reset data members to 0
   memset(&fRotationMatrix[0], 0, 9*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
void TGeoRotation::FastRotZ(Double_t *sincos)
{
   fRotationMatrix[0] = sincos[1];
   fRotationMatrix[1] = -sincos[0];
   fRotationMatrix[3] = sincos[0];
   fRotationMatrix[4] = sincos[1];
}
//-----------------------------------------------------------------------------
void TGeoRotation::LocalToMaster(const Double_t *local, Double_t *master)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix inverse
  const Double_t *rot = GetRotationMatrix();
  for (Int_t i=0; i<3; i++) {
      master[i] = local[0]*rot[3*i]
                + local[1]*rot[3*i+1]
                + local[2]*rot[3*i+2];
   }
}
//-----------------------------------------------------------------------------
void TGeoRotation::MasterToLocal(const Double_t *master, Double_t *local)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix
  const Double_t *rot = GetRotationMatrix();
    for (Int_t i=0; i<3; i++) {
       local[i] =  master[0]*rot[i]
                 + master[1]*rot[i+3]
                 + master[2]*rot[i+6];
   }
}
//-----------------------------------------------------------------------------
void TGeoRotation::SetAngles(Double_t alpha, Double_t beta, Double_t gamma)
{
// Set matrix elements according to Euler angles
   Double_t degrad = TMath::Pi()/180.;
   Double_t sinalf = TMath::Sin(degrad*alpha);
   Double_t cosalf = TMath::Cos(degrad*alpha);
   Double_t sinbet = TMath::Sin(degrad*beta);
   Double_t cosbet = TMath::Cos(degrad*beta);
   Double_t singam = TMath::Sin(degrad*gamma);
   Double_t cosgam = TMath::Cos(degrad*gamma);

   fRotationMatrix[0] = cosbet*cosgam;
   fRotationMatrix[1] = cosalf*singam + sinalf*sinbet*cosgam;
   fRotationMatrix[2] = sinalf*singam - cosalf*sinbet*cosgam;
   fRotationMatrix[3] = - cosbet*singam;
   fRotationMatrix[4] = cosalf*cosgam - sinalf*sinbet*singam;
   fRotationMatrix[5] = sinalf*cosgam + cosalf*sinbet*singam;
   fRotationMatrix[6] = sinbet;
   fRotationMatrix[7] = - sinalf*cosbet;
   fRotationMatrix[8] = cosalf*cosbet;
   // do the trick to eliminate most of the floating point errors
   for (Int_t i=0; i<9; i++) {
      if (TMath::Abs(fRotationMatrix[i])<1E-15) fRotationMatrix[i] = 0;
      if (TMath::Abs(fRotationMatrix[i]-1)<1E-15) fRotationMatrix[i] = 1;
      if (TMath::Abs(fRotationMatrix[i]+1)<1E-15) fRotationMatrix[i] = -1;
   }   
}
//-----------------------------------------------------------------------------
void TGeoRotation::SetAngles(Double_t theta1, Double_t phi1, Double_t theta2, Double_t phi2,
                             Double_t theta3, Double_t phi3)
{
// Set matrix elements in the GEANT3 way
   Double_t degrad = TMath::Pi()/180.;
   fRotationMatrix[0] = TMath::Cos(degrad*phi1)*TMath::Sin(degrad*theta1);
   fRotationMatrix[3] = TMath::Sin(degrad*phi1)*TMath::Sin(degrad*theta1);
   fRotationMatrix[6] = TMath::Cos(degrad*theta1);
   fRotationMatrix[1] = TMath::Cos(degrad*phi2)*TMath::Sin(degrad*theta2);
   fRotationMatrix[4] = TMath::Sin(degrad*phi2)*TMath::Sin(degrad*theta2);
   fRotationMatrix[7] = TMath::Cos(degrad*theta2);
   fRotationMatrix[2] = TMath::Cos(degrad*phi3)*TMath::Sin(degrad*theta3);
   fRotationMatrix[5] = TMath::Sin(degrad*phi3)*TMath::Sin(degrad*theta3);
   fRotationMatrix[8] = TMath::Cos(degrad*theta3);
   // do the trick to eliminate most of the floating point errors
   for (Int_t i=0; i<9; i++) {
      if (TMath::Abs(fRotationMatrix[i])<1E-15) fRotationMatrix[i] = 0;
      if (TMath::Abs(fRotationMatrix[i]-1)<1E-15) fRotationMatrix[i] = 1;
      if (TMath::Abs(fRotationMatrix[i]+1)<1E-15) fRotationMatrix[i] = -1;
   }   
}
//-----------------------------------------------------------------------------
Double_t TGeoRotation::Determinant()
{
// computes determinant of the rotation matrix
   Double_t 
   det = fRotationMatrix[0]*fRotationMatrix[4]*fRotationMatrix[8] + 
         fRotationMatrix[3]*fRotationMatrix[7]*fRotationMatrix[1] +
         fRotationMatrix[6]*fRotationMatrix[1]*fRotationMatrix[5] - 
         fRotationMatrix[2]*fRotationMatrix[4]*fRotationMatrix[6] -
         fRotationMatrix[5]*fRotationMatrix[7]*fRotationMatrix[0] - 
         fRotationMatrix[7]*fRotationMatrix[1]*fRotationMatrix[3];
   return det;
}
//-----------------------------------------------------------------------------
void TGeoRotation::CheckMatrix()
{
   // performes an orthogonality check and finds if the matrix is a reflection
//   Warning("CheckMatrix", "orthogonality check not performed yet");
   if (Determinant() < 0) {
      SetBit(kGeoReflection);
//      printf("matrix %s is reflection\n", GetName());
   }
}
//-----------------------------------------------------------------------------
void TGeoRotation::GetInverse(Double_t *invmat)
{
// Get the inverse rotation matrix (which is simply the transpose)
   if (!invmat) {
      Error("GetInverse", "no place to store the inverse matrix");
   }
   for (Int_t i=0; i<3; i++) {
      for (Int_t j=0; i<3; i++) {   
         invmat[3*i+j] = fRotationMatrix[3*j+i];
      }
   }
}
//-----------------------------------------------------------------------------
void TGeoRotation::MultiplyBy(TGeoRotation *rot, Bool_t after)
{
   const Double_t *matleft, *matright;
   Double_t  newmat[9];
   if (after) {
      matleft  = &fRotationMatrix[0];
      matright = rot->GetRotationMatrix();
   } else {
      matleft  = rot->GetRotationMatrix();
      matright = &fRotationMatrix[0];
   }
   for (Int_t i=0; i<3; i++) {
      for (Int_t j=0; j<3; j++) { 
         for (Int_t k=0; k<3; k++) {
            newmat[3*i+j] = matleft[3*i+k] * matright[3*k+j];
         }
      }
   }
   memcpy(&fRotationMatrix[0], &newmat[0], 9*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
TGeoScale::TGeoScale()
{
// default constructor
   SetBit(kGeoScale);
   SetDefaultName();
   for (Int_t i=0; i<3; i++) fScale[i] = 0;
}
//-----------------------------------------------------------------------------
TGeoScale::TGeoScale(Double_t sx, Double_t sy, Double_t sz)
          :TGeoMatrix()
{
// default constructor
   SetBit(kGeoScale);
   SetDefaultName();
   SetScale(sx, sy, sz);
   gGeoManager->AddTransformation(this);
}
//-----------------------------------------------------------------------------
TGeoScale::~TGeoScale()
{
// destructor
}
//-----------------------------------------------------------------------------
void TGeoScale::SetScale(Double_t sx, Double_t sy, Double_t sz)
{
// scale setter
   fScale[0] = sx;
   fScale[1] = sy;
   fScale[2] = sz;
   if (!(Normalize())) {
      Error("ctor", "Invalid scale");
      return;
   }
}
//-----------------------------------------------------------------------------
Bool_t TGeoScale::Normalize()
{
// A scale transformation should be normalized by sx*sy*sz factor
   Double_t normfactor = fScale[0]*fScale[1]*fScale[2];
   if (normfactor <= 1E-5) return kFALSE;
   for (Int_t i=0; i<3; i++)
      fScale[i] /= normfactor;
   return kTRUE;
}
//-----------------------------------------------------------------------------
TGeoCombiTrans::TGeoCombiTrans()
{
// dummy ctor
   SetBit(kGeoCombiTrans);
   SetDefaultName();
   for (Int_t i=0; i<3; i++) fTranslation[i] = 0.0;
   fRotation = 0;
}
//-----------------------------------------------------------------------------
TGeoCombiTrans::TGeoCombiTrans(const char *name)
               :TGeoMatrix(name)
{
// ctor
   SetBit(kGeoCombiTrans);
   SetDefaultName();
   for (Int_t i=0; i<3; i++) fTranslation[i] = 0.0;
   fRotation = 0;
   gGeoManager->AddTransformation(this);
}
//-----------------------------------------------------------------------------
TGeoCombiTrans::TGeoCombiTrans(Double_t dx, Double_t dy, Double_t dz, TGeoRotation *rot)
               :TGeoMatrix("")
{
// ctor
   SetBit(kGeoCombiTrans);
   SetDefaultName();
   SetTranslation(dx, dy, dz);
   SetRotation(rot);
   gGeoManager->AddTransformation(this);
}
//-----------------------------------------------------------------------------
TGeoCombiTrans::~TGeoCombiTrans()
{
// destructor
}
//-----------------------------------------------------------------------------
void TGeoCombiTrans::SetTranslation(Double_t dx, Double_t dy, Double_t dz)
{
// set the translation component
   fTranslation[0] = dx;
   fTranslation[1] = dy;
   fTranslation[2] = dz;
}
//-----------------------------------------------------------------------------
void TGeoCombiTrans::SetTranslation(Double_t *vect)
{
// set the translation component
   fTranslation[0] = vect[0];
   fTranslation[1] = vect[1];
   fTranslation[2] = vect[2];
}
//-----------------------------------------------------------------------------
const Double_t *TGeoCombiTrans::GetRotationMatrix() const
{
// get the rotation array
   if (!fRotation) return 0;
   return fRotation->GetRotationMatrix();
}
//-----------------------------------------------------------------------------
TGeoGenTrans::TGeoGenTrans()
{
// dummy ctor
   SetBit(kGeoGenTrans);
   SetDefaultName();
   for (Int_t i=0; i<3; i++) fTranslation[i] = 0.0;
   for (Int_t j=0; j<3; j++) fScale[j] = 1.0;
   fRotation = 0;
}
//-----------------------------------------------------------------------------
TGeoGenTrans::TGeoGenTrans(const char *name)
             :TGeoCombiTrans(name)
{
// ctor
   SetBit(kGeoGenTrans);
   SetDefaultName();
   for (Int_t i=0; i<3; i++) fTranslation[i] = 0.0;
   for (Int_t j=0; j<3; j++) fScale[j] = 1.0;
   fRotation = 0;
}
//-----------------------------------------------------------------------------
TGeoGenTrans::TGeoGenTrans(Double_t dx, Double_t dy, Double_t dz,
                           Double_t sx, Double_t sy, Double_t sz, TGeoRotation *rot)
             :TGeoCombiTrans("")
{
// ctor
   SetBit(kGeoGenTrans);
   SetDefaultName();
   SetTranslation(dx, dy, dz);
   SetScale(sx, sy, sz);
   SetRotation(rot);
}
//-----------------------------------------------------------------------------
TGeoGenTrans::~TGeoGenTrans()
{
// destructor
}
//-----------------------------------------------------------------------------
void TGeoGenTrans::Clear(Option_t *)
{
// clear the fields of this transformation
   memset(&fTranslation[0], 0, 3*sizeof(Double_t));
   memset(&fScale[0], 0, 3*sizeof(Double_t));
   fRotation->Clear();
}
//-----------------------------------------------------------------------------
void TGeoGenTrans::SetScale(Double_t sx, Double_t sy, Double_t sz)
{
// set the scale
   fScale[0] = sx;
   fScale[1] = sy;
   fScale[2] = sz;
   if (!(Normalize())) {
      Error("ctor", "Invalid scale");
      return;
   }
}
//-----------------------------------------------------------------------------
Bool_t TGeoGenTrans::Normalize()
{
// A scale transformation should be normalized by sx*sy*sz factor
   Double_t normfactor = fScale[0]*fScale[1]*fScale[2];
   if (normfactor <= 1E-5) return kFALSE;
   for (Int_t i=0; i<3; i++)
      fScale[i] /= normfactor;
   return kTRUE;
}
//-----------------------------------------------------------------------------
TGeoIdentity::TGeoIdentity()
{
// dummy ctor
}
//-----------------------------------------------------------------------------
TGeoIdentity::TGeoIdentity(const char *name)
             :TGeoMatrix(name)
{
// ctor
   if (!gGeoIdentity) gGeoIdentity = this;
   gGeoManager->AddTransformation(this);
}
//-----------------------------------------------------------------------------
void TGeoIdentity::LocalToMaster(const Double_t *local, Double_t *master)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix inverse
   memcpy(master, local, 3*sizeof(Double_t));
}
//-----------------------------------------------------------------------------
void TGeoIdentity::MasterToLocal(const Double_t *master, Double_t *local)
{
// convert a point by multiplying its column vector (x, y, z, 1) to matrix
   memcpy(local, master, 3*sizeof(Double_t));
}

/*************************************************************************
 * TGeoHMatrix - Matrix class used for computing global transformations  *
 *     Should NOT be used for node definition. An instance of this class *
 *     is generally used to pile-up local transformations starting from  *
 *     the top level physical node, down to the current node.            *
 *************************************************************************/

ClassImp(TGeoHMatrix)
   
//-----------------------------------------------------------------------------
TGeoHMatrix::TGeoHMatrix()
{
// dummy ctor
   memset(&fTranslation[0], 0, 3*sizeof(Double_t));
   SetRotation(&kIdentityMatrix[0]);
   SetScale(&kUnitScale[0]);
}
//-----------------------------------------------------------------------------
TGeoHMatrix::TGeoHMatrix(const char* name)
            :TGeoMatrix(name)
{
// ctor
   memset(&fTranslation[0], 0, 3*sizeof(Double_t));
   SetRotation(&kIdentityMatrix[0]);
   SetScale(&kUnitScale[0]);
}
//-----------------------------------------------------------------------------
TGeoHMatrix::~TGeoHMatrix()
{
// destructor
}
//-----------------------------------------------------------------------------
TGeoHMatrix &TGeoHMatrix::operator=(const TGeoMatrix *matrix)
{
   // assignment
   Clear();
   if (matrix->IsIdentity()) return *this;
   if (matrix->IsTranslation()) {
      SetBit(kGeoTranslation);
      SetTranslation(matrix->GetTranslation());
   }
   if (matrix->IsRotation()) {
      SetBit(kGeoRotation);
      SetRotation(matrix->GetRotationMatrix());
   }
   if (matrix->IsScale()) {
      SetBit(kGeoScale);
      SetScale(matrix->GetScale());
   }
   return *this;
}
//-----------------------------------------------------------------------------
void TGeoHMatrix::Clear(Option_t *)
{
// clear the data for this matrix
   if (IsIdentity()) return;
   if (IsTranslation()) {
      ResetBit(kGeoTranslation);
      SetTranslation(&kNullVector[0]);
   }
   if (IsRotation()) {
      ResetBit(kGeoRotation);
      SetRotation(&kIdentityMatrix[0]);
   }
   if (IsScale()) {      
      ResetBit(kGeoScale);
      SetScale(&kUnitScale[0]);
   }
}
//-----------------------------------------------------------------------------
void TGeoHMatrix::Multiply(TGeoMatrix *right)
{
// multiply to the right with an other transformation
   // if right is identity matrix, just return
   if (right == gGeoIdentity) return;
   const Double_t *r_tra = right->GetTranslation();
   const Double_t *r_rot = right->GetRotationMatrix();
   const Double_t *r_scl = right->GetScale();
   if (IsIdentity()) {
      if (right->IsRotation()) {
         SetBit(kGeoRotation);
         SetRotation(r_rot);
      }
      if (right->IsScale()) {
         SetBit(kGeoScale);
         SetScale(r_scl);
      }
      if (right->IsTranslation()) {
         SetBit(kGeoTranslation);
         SetTranslation(r_tra);
      }
      return;
   }
   Int_t i, j;
   Double_t new_tra[3]; 
   Double_t new_rot[9]; 
   Double_t new_scl[3]; 

   if (right->IsRotation())    SetBit(kGeoRotation);
   if (right->IsScale())       SetBit(kGeoScale);
   if (right->IsTranslation()) SetBit(kGeoTranslation);

   // new translation
   if (IsTranslation()) {  
      for (i=0; i<3; i++) {
         new_tra[i] = fTranslation[i]
                      + fRotationMatrix[3*i]*r_tra[0]
                      + fRotationMatrix[3*i+1]*r_tra[1]
                      + fRotationMatrix[3*i+2]*r_tra[2];
      }
      SetTranslation(&new_tra[0]);
   }
   if (IsRotation()) {
      // new rotation
      for (i=0; i<3; i++) {
         for (j=0; j<3; j++) {
               new_rot[3*i+j] = fRotationMatrix[3*i]*r_rot[j] +
                                fRotationMatrix[3*i+1]*r_rot[3+j] +
                                fRotationMatrix[3*i+2]*r_rot[6+j];
         }
      }
      SetRotation(&new_rot[0]);
   }
   // new scale
   if (IsScale()) {
      for (i=0; i<3; i++) new_scl[i] = fScale[i]*r_scl[i];
      SetScale(&new_scl[0]);
   }
}
 
