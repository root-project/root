#ifdef WIN32
#include "Windows4root.h"
#endif

#include <GL/gl.h>

#include "TGLSceneObject.h"
#include "TGLFrustum.h"

//______________________________________________________________________________
TGLFrustum::TGLFrustum()
{
   fInFrustum = 0;
}

//______________________________________________________________________________
TGLFrustum::~TGLFrustum()
{
}

//______________________________________________________________________________
void TGLFrustum::Update()
{
   fInFrustum = 0;
   GLdouble ProjM[16];  // Projection matrix
   GLdouble ModVM[16];  // ModelView matrix
   GLdouble ClipM[16];  // Object space clip matrix

   glGetDoublev(GL_PROJECTION_MATRIX, ProjM);
   glGetDoublev(GL_MODELVIEW_MATRIX, ModVM);

   // Mult projection by modelview - replace with ROOT matrix lib ops?
   // N.B. OpenGL matricies are COLUMN MAJOR
   ClipM[ 0] = ModVM[ 0] * ProjM[ 0] + ModVM[ 1] * ProjM[ 4] + ModVM[ 2] * ProjM[ 8] + ModVM[ 3] * ProjM[12];
   ClipM[ 1] = ModVM[ 0] * ProjM[ 1] + ModVM[ 1] * ProjM[ 5] + ModVM[ 2] * ProjM[ 9] + ModVM[ 3] * ProjM[13];
   ClipM[ 2] = ModVM[ 0] * ProjM[ 2] + ModVM[ 1] * ProjM[ 6] + ModVM[ 2] * ProjM[10] + ModVM[ 3] * ProjM[14];
   ClipM[ 3] = ModVM[ 0] * ProjM[ 3] + ModVM[ 1] * ProjM[ 7] + ModVM[ 2] * ProjM[11] + ModVM[ 3] * ProjM[15];

   ClipM[ 4] = ModVM[ 4] * ProjM[ 0] + ModVM[ 5] * ProjM[ 4] + ModVM[ 6] * ProjM[ 8] + ModVM[ 7] * ProjM[12];
   ClipM[ 5] = ModVM[ 4] * ProjM[ 1] + ModVM[ 5] * ProjM[ 5] + ModVM[ 6] * ProjM[ 9] + ModVM[ 7] * ProjM[13];
   ClipM[ 6] = ModVM[ 4] * ProjM[ 2] + ModVM[ 5] * ProjM[ 6] + ModVM[ 6] * ProjM[10] + ModVM[ 7] * ProjM[14];
   ClipM[ 7] = ModVM[ 4] * ProjM[ 3] + ModVM[ 5] * ProjM[ 7] + ModVM[ 6] * ProjM[11] + ModVM[ 7] * ProjM[15];

   ClipM[ 8] = ModVM[ 8] * ProjM[ 0] + ModVM[ 9] * ProjM[ 4] + ModVM[10] * ProjM[ 8] + ModVM[11] * ProjM[12];
   ClipM[ 9] = ModVM[ 8] * ProjM[ 1] + ModVM[ 9] * ProjM[ 5] + ModVM[10] * ProjM[ 9] + ModVM[11] * ProjM[13];
   ClipM[10] = ModVM[ 8] * ProjM[ 2] + ModVM[ 9] * ProjM[ 6] + ModVM[10] * ProjM[10] + ModVM[11] * ProjM[14];
   ClipM[11] = ModVM[ 8] * ProjM[ 3] + ModVM[ 9] * ProjM[ 7] + ModVM[10] * ProjM[11] + ModVM[11] * ProjM[15];

   ClipM[12] = ModVM[12] * ProjM[ 0] + ModVM[13] * ProjM[ 4] + ModVM[14] * ProjM[ 8] + ModVM[15] * ProjM[12];
   ClipM[13] = ModVM[12] * ProjM[ 1] + ModVM[13] * ProjM[ 5] + ModVM[14] * ProjM[ 9] + ModVM[15] * ProjM[13];
   ClipM[14] = ModVM[12] * ProjM[ 2] + ModVM[13] * ProjM[ 6] + ModVM[14] * ProjM[10] + ModVM[15] * ProjM[14];
   ClipM[15] = ModVM[12] * ProjM[ 3] + ModVM[13] * ProjM[ 7] + ModVM[14] * ProjM[11] + ModVM[15] * ProjM[15];

   // RIGHT clipping plane
   fClippingPlanes[kRIGHT].fA = ClipM[ 3] - ClipM[ 0];
   fClippingPlanes[kRIGHT].fB = ClipM[ 7] - ClipM[ 4];
   fClippingPlanes[kRIGHT].fC = ClipM[11] - ClipM[ 8];
   fClippingPlanes[kRIGHT].fD = ClipM[15] - ClipM[12];
   fClippingPlanes[kRIGHT].Normalise();
   
   // LEFT clipping plane
   fClippingPlanes[kLEFT].fA = ClipM[ 3] + ClipM[ 0];
   fClippingPlanes[kLEFT].fB = ClipM[ 7] + ClipM[ 4];
   fClippingPlanes[kLEFT].fC = ClipM[11] + ClipM[ 8];
   fClippingPlanes[kLEFT].fD = ClipM[15] + ClipM[12];
   fClippingPlanes[kLEFT].Normalise();

   // BOTTOM clipping plane
   fClippingPlanes[kBOTTOM].fA = ClipM[ 3] + ClipM[ 1];
   fClippingPlanes[kBOTTOM].fB = ClipM[ 7] + ClipM[ 5];
   fClippingPlanes[kBOTTOM].fC = ClipM[11] + ClipM[ 9];
   fClippingPlanes[kBOTTOM].fD = ClipM[15] + ClipM[13];
   fClippingPlanes[kBOTTOM].Normalise();

   // TOP clipping plane
   fClippingPlanes[kTOP].fA = ClipM[ 3] - ClipM[ 1];
   fClippingPlanes[kTOP].fB = ClipM[ 7] - ClipM[ 5];
   fClippingPlanes[kTOP].fC = ClipM[11] - ClipM[ 9];
   fClippingPlanes[kTOP].fD = ClipM[15] - ClipM[13];
   fClippingPlanes[kTOP].Normalise();

   // FAR clipping plane
   fClippingPlanes[kFAR].fA = ClipM[ 3] - ClipM[ 2];
   fClippingPlanes[kFAR].fB = ClipM[ 7] - ClipM[ 6];
   fClippingPlanes[kFAR].fC = ClipM[11] - ClipM[10];
   fClippingPlanes[kFAR].fD = ClipM[15] - ClipM[14];
   fClippingPlanes[kFAR].Normalise();

   // NEAR clipping plane
   fClippingPlanes[kNEAR].fA = ClipM[ 3] + ClipM[ 2];
   fClippingPlanes[kNEAR].fB = ClipM[ 7] + ClipM[ 6];
   fClippingPlanes[kNEAR].fC = ClipM[11] + ClipM[10];
   fClippingPlanes[kNEAR].fD = ClipM[15] + ClipM[14];
   fClippingPlanes[kNEAR].Normalise();
}

//______________________________________________________________________________
Bool_t TGLFrustum::ClipOnBoundingBox(const TGLSceneObject &SceneObject) const
{
   const Double_t *box = SceneObject.GetBBox()->GetData();
   
   Double_t BBVertexes[][3] = {
                               {box[0], box[2], box[4]}, {box[1], box[2], box[4]},
                               {box[0], box[3], box[4]}, {box[1], box[3], box[4]},
                               {box[0], box[2], box[5]}, {box[1], box[2], box[5]},
                               {box[0], box[3], box[5]}, {box[1], box[3], box[5]}
                              };
                              
   ClipResult clip = ClipOnBoundingBox(BBVertexes);
   if (clip == kINSIDE || clip == kPARTIAL) {
      ++fInFrustum;
      return kTRUE;
   } else return kFALSE;
}

//______________________________________________________________________________
TGLFrustum::ClipResult TGLFrustum::ClipOnBoundingBox(const Double_t BBVertexes[8][3]) const
{
   //std::cout << "--------------------------------------------" << std::endl;    
   //std::cout.precision(3);
   //std::cout.width(3);
   // Always draw if clip testing is false
   // Test 8 verticies of BB against 6 frustum planes. 3 cases for EACH plane
   // 1. All outside this SINGLE plane -> BB outside frustum -> skip any remaining planes
   // 2. Some inside, some out this plane -> record and test next plane
   // 3. All inside this plane -> record and test next plane.
   // 
   // This method can result in false positives, where all points lie a plane, 
   // but not a single one. In this case the BB will (incorrectly) be regarded 
   // as intersecting and hence drawn. This is rare and benign aside from 
   // performance cost - GL will clip.
   
   
   Int_t PlanesInside = 0; // Assume outside to start
   for (Int_t PlaneIndex = 0; PlaneIndex < kPLANESPERFRUSTUM; ++PlaneIndex) {

      Int_t VertexesInsidePlane = 8; // Assume inside to start
      for (Int_t VertexIndex = 0; VertexIndex < 8; VertexIndex++) {
         const Double_t *const BBVertex = BBVertexes[VertexIndex];
         
         Double_t PlaneDist = fClippingPlanes[PlaneIndex].fA * BBVertex[0] + fClippingPlanes[PlaneIndex].fB * BBVertex[1] +
                              fClippingPlanes[PlaneIndex].fC * BBVertex[2] + fClippingPlanes[PlaneIndex].fD;
         
         if ( PlaneDist < 0.0 ) {
            VertexesInsidePlane--;
         }
      }
      // Once we find a single plane which all vertexes are outside, we are outside the frustum
      if ( VertexesInsidePlane == 0 ) {
         return kOUTSIDE;  
      } else if ( VertexesInsidePlane == 8 ) {
         PlanesInside++;
      }
   }
   // Completely inside frustum
   if ( PlanesInside == kPLANESPERFRUSTUM ) {
      return kINSIDE;
   } else {
      return kPARTIAL;
   }
}
