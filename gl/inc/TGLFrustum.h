#ifndef ROOT_TGLFrustum
#define ROOT_TGLFrustum

#ifndef ROOT_TMath
#include "TMath.h"
#endif

//Author Richard

class TGLSceneObject;

class TGLFrustum {
public:
   TGLFrustum();
   ~TGLFrustum();

   // Extract the current clipping planes from GL 
   // Project + ModelView matrices
   void Update();
   void Dump() const;  
   Bool_t ClipOnBoundingBox(const TGLSceneObject & sceneObject)const;   

   Int_t GetVisible()const
   {
      return fInFrustum;
   }

private:

   mutable Int_t fInFrustum;

	enum ClipResult { 
      kINSIDE, 
      kPARTIAL, 
      kOUTSIDE 
   };
	
	ClipResult ClipOnBoundingBox( const Double_t bbVertexes[8][3]) const;
	
   // Internal plane representation
   struct Plane {
   public:
      Double_t fA, fB, fC, fD;
      
      void Normalise() 
      {
         Double_t mag = TMath::Sqrt(fA * fA + fB * fB + fC * fC);
         
         if (mag == .0) {
            return;
         }
         
         fA /= mag;
         fB /= mag;
         fC /= mag;
         fD /= mag;
      }
   };
   
   enum {
      kLEFT = 0,
      kRIGHT,
      kTOP,
      kBOTTOM,
      kNEAR,
      kFAR,
      kPLANESPERFRUSTUM
   };
         
   // Cached clipping planes in object drawing space
   Plane fClippingPlanes[kPLANESPERFRUSTUM];
};

#endif
