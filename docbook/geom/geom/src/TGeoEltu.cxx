// @(#)root/geom:$Id$
// Author: Mihaela Gheata   05/06/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_____________________________________________________________________________
// TGeoEltu - elliptical tube class. It takes 3 parameters : 
// semi-axis of the ellipse along x, semi-asix of the ellipse along y
// and half-length dz. 
//
//_____________________________________________________________________________
//Begin_Html
/*
<img src="gif/t_eltu.gif">
*/
//End_Html

#include "Riostream.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoEltu.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TMath.h"

ClassImp(TGeoEltu)
   
//_____________________________________________________________________________
TGeoEltu::TGeoEltu()
{
// Dummy constructor
   SetShapeBit(TGeoShape::kGeoEltu);
}   

//_____________________________________________________________________________
TGeoEltu::TGeoEltu(Double_t a, Double_t b, Double_t dz)
           :TGeoTube()
{
// Default constructor specifying X and Y semiaxis length
   SetShapeBit(TGeoShape::kGeoEltu);
   SetEltuDimensions(a, b, dz);
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoEltu::TGeoEltu(const char *name, Double_t a, Double_t b, Double_t dz)
           :TGeoTube(name,0.,b,dz)
{
// Default constructor specifying X and Y semiaxis length
   SetName(name);
   SetShapeBit(TGeoShape::kGeoEltu);
   SetEltuDimensions(a, b, dz);
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoEltu::TGeoEltu(Double_t *param)
{
// Default constructor specifying minimum and maximum radius
// param[0] =  A
// param[1] =  B
// param[2] = dz
   SetShapeBit(TGeoShape::kGeoEltu);
   SetDimensions(param);
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoEltu::~TGeoEltu()
{
// destructor
}

//_____________________________________________________________________________
Double_t TGeoEltu::Capacity() const
{
// Computes capacity of the shape in [length^3]
   Double_t capacity = 2.*TMath::Pi()*fDz*fRmin*fRmax;
   return capacity;
}   

//_____________________________________________________________________________   
void TGeoEltu::ComputeBBox()
{
// compute bounding box of the tube
   fDX = fRmin;
   fDY = fRmax;
   fDZ = fDz;
}   

//_____________________________________________________________________________   
void TGeoEltu::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Compute normal to closest surface from POINT.
   Double_t a = fRmin;
   Double_t b = fRmax;
   Double_t eps = TMath::Sqrt(point[0]*point[0]/(a*a)+point[1]*point[1]/(b*b))-1.;
   if (eps<1E-4 && TMath::Abs(fDz-TMath::Abs(point[2]))<1E-5) {
      norm[0] = norm[1] = 0;
      norm[2] = TMath::Sign(1.,dir[2]);
      return;
   }   
   norm[2] = 0.;
   Double_t r = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t st = point[1]/r;
   Double_t ct = point[0]/r;
   Double_t rr = TMath::Sqrt(b*b*ct*ct+a*a*st*st);
   norm[0] = b*ct/rr;
   norm[1] = a*st/rr;
   if (norm[0]*dir[0]+norm[1]*dir[1]<0) {
      norm[0] = -norm[0];
      norm[1] = -norm[1];
   }   
}

//_____________________________________________________________________________
Bool_t TGeoEltu::Contains(Double_t *point) const
{
// test if point is inside the elliptical tube
   if (TMath::Abs(point[2]) > fDz) return kFALSE;
   Double_t r2 = (point[0]*point[0])/(fRmin*fRmin)+(point[1]*point[1])/(fRmax*fRmax);
   if (r2>1.)  return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________
Int_t TGeoEltu::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each vertex
   Int_t n = gGeoManager->GetNsegments();
   const Int_t numPoints=4*n;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}   

//_____________________________________________________________________________
Double_t TGeoEltu::DistFromInside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from inside point to surface of the tube
   Double_t a2=fRmin*fRmin;
   Double_t b2=fRmax*fRmax;
   Double_t safz1=fDz-point[2];
   Double_t safz2=fDz+point[2];
   
   if (iact<3 && safe) {
      Double_t x0=TMath::Abs(point[0]);
      Double_t y0=TMath::Abs(point[1]);
      Double_t x1=x0;
      Double_t y1=TMath::Sqrt((fRmin-x0)*(fRmin+x0))*fRmax/fRmin;
      Double_t y2=y0;
      Double_t x2=TMath::Sqrt((fRmax-y0)*(fRmax+y0))*fRmin/fRmax;
      Double_t d1=(x1-x0)*(x1-x0)+(y1-y0)*(y1-y0);
      Double_t d2=(x2-x0)*(x2-x0)+(y2-y0)*(y2-y0);
      Double_t x3,y3;
   
      Double_t safr=TGeoShape::Big();
      Double_t safz = TMath::Min(safz1,safz2);
      for (Int_t i=0; i<8; i++) {
         if (fRmax<fRmin) {
            x3=0.5*(x1+x2);
            y3=TMath::Sqrt((fRmin-x3)*(fRmin+x3))*fRmax/fRmin;;
         } else {
            y3=0.5*(y1+y2);   
            x3=TMath::Sqrt((fRmax-y3)*(fRmax+y3))*fRmin/fRmax;
         }
         if (d1<d2) {
            x2=x3;
            y2=y3;
            d2=(x2-x0)*(x2-x0)+(y2-y0)*(y2-y0);
         } else {
            x1=x3;
            y1=y3;
            d1=(x1-x0)*(x1-x0)+(y1-y0)*(y1-y0);
         }
      }
      safr=TMath::Sqrt(d1)-1.0E-3;   
      *safe = TMath::Min(safz, safr);
      if (iact==0) return TGeoShape::Big();
      if ((iact==1) && (*safe>step)) return TGeoShape::Big();
   }
   // compute distance to surface 
   // Do Z
   Double_t snxt = TGeoShape::Big();
   if (dir[2]>0) {
      snxt=safz1/dir[2];
   } else {
      if (dir[2]<0) snxt=-safz2/dir[2];
   } 
   // do eliptical surface
   Double_t sz = snxt;
   Double_t xz=point[0]+dir[0]*sz;
   Double_t yz=point[1]+dir[1]*sz;
   if ((xz*xz/a2+yz*yz/b2)<=1) return snxt;
   Double_t u=dir[0]*dir[0]*b2+dir[1]*dir[1]*a2;
   Double_t v=point[0]*dir[0]*b2+point[1]*dir[1]*a2;
   Double_t w=point[0]*point[0]*b2+point[1]*point[1]*a2-a2*b2;
   Double_t d=v*v-u*w;
   if (d<0) return snxt;
   if (TGeoShape::IsSameWithinTolerance(u,0)) return snxt;
   Double_t sd=TMath::Sqrt(d);
   Double_t tau1=(-v+sd)/u;
   Double_t tau2=(-v+sd)/u;
   
   if (tau1<0) {
      if (tau2<0) return snxt;
   } else {
      snxt=tau1;
      if ((tau2>0) && (tau2<tau1)) return tau2;
   }
   return snxt;      
}

//_____________________________________________________________________________
Double_t TGeoEltu::DistFromOutside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// compute distance from outside point to surface of the tube and safe distance
   Double_t snxt=TGeoShape::Big();
   Double_t safz=TMath::Abs(point[2])-fDz;
   Double_t a2=fRmin*fRmin;
   Double_t b2=fRmax*fRmax;
   if (iact<3 && safe) {
      Double_t x0=TMath::Abs(point[0]);
      Double_t y0=TMath::Abs(point[1]);
      *safe=0.;
      if ((x0*x0/a2+y0*y0/b2)>=1) {
         Double_t phi1=0;
         Double_t phi2=0.5*TMath::Pi();
         Double_t phi3;
         Double_t x3,y3,d;
         for (Int_t i=0; i<10; i++) {
            phi3=(phi1+phi2)*0.5;
            x3=fRmin*TMath::Cos(phi3);
            y3=fRmax*TMath::Sin(phi3);
            d=y3*a2*(x0-x3)-x3*b2*(y0-y3);
            if (d<0) phi1=phi3;
            else phi2=phi3;
         }
         *safe=TMath::Sqrt((x0-x3)*(x0-x3)+(y0-y3)*(y0-y3));
      }
      if (safz>0) {
         *safe=TMath::Sqrt((*safe)*(*safe)+safz*safz);
      } 
      if (iact==0) return TGeoShape::Big();
      if ((iact==1) && (step<*safe)) return TGeoShape::Big();
   }
   // compute vector distance
   if ((safz>0) && (point[2]*dir[2]>=0)) return TGeoShape::Big();
// Check if the bounding box is crossed within the requested distance
   Double_t sdist = TGeoBBox::DistFromOutside(point,dir, fDX, fDY, fDZ, fOrigin, step);
   if (sdist>=step) return TGeoShape::Big();
   Double_t zi;
   if (!TGeoShape::IsSameWithinTolerance(dir[2],0)) {
      Double_t u=dir[0]*dir[0]*b2+dir[1]*dir[1]*a2;
      if (TGeoShape::IsSameWithinTolerance(u,0)) return TGeoShape::Big();
      Double_t v=point[0]*dir[0]*b2+point[1]*dir[1]*a2;
      Double_t w=point[0]*point[0]*b2+point[1]*point[1]*a2-a2*b2;
      Double_t d=v*v-u*w;
      if (d<0) return TGeoShape::Big();
      Double_t dsq=TMath::Sqrt(d);
      Double_t tau[2];
      tau[0]=(-v+dsq)/u;
      tau[1]=(-v-dsq)/u;
      for (Int_t j=0; j<2; j++) {
         if (tau[j]>=0) {
            zi=point[2]+tau[j]*dir[2];
            if ((TMath::Abs(zi)-fDz)<0)
               snxt=TMath::Min(snxt,tau[j]);
         } 
      }
   }   
   // do z
   zi=TGeoShape::Big();
   if (safz>0) {
      if (TGeoShape::IsSameWithinTolerance(dir[2],0)) return TGeoShape::Big();
      if (point[2]>0) zi=fDz;
      if (point[2]<0) zi=-fDz;     
      Double_t tauz=(zi-point[2])/dir[2];
      Double_t xz=point[0]+dir[0]*tauz;
      Double_t yz=point[1]+dir[1]*tauz;
      if ((xz*xz/a2+yz*yz/b2)<=1) snxt=tauz;
   }
   return snxt;   
}

//_____________________________________________________________________________
TGeoVolume *TGeoEltu::Divide(TGeoVolume * /*voldiv*/, const char * /*divname*/, Int_t /*iaxis*/, Int_t /*ndiv*/, 
                             Double_t /*start*/, Double_t /*step*/) 
{
// Divide the shape along one axis.
   Error("Divide", "Elliptical tubes divisions not implemenetd");
   return 0;
}   

//_____________________________________________________________________________
void TGeoEltu::GetBoundingCylinder(Double_t *param) const
{
//--- Fill vector param[4] with the bounding cylinder parameters. The order
// is the following : Rmin, Rmax, Phi1, Phi2
   param[0] = 0.;                  // Rmin
   param[1] = TMath::Max(fRmin, fRmax); // Rmax
   param[1] *= param[1];
   param[2] = 0.;                  // Phi1
   param[3] = 360.;                // Phi2 
}   

//_____________________________________________________________________________
TGeoShape *TGeoEltu::GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix * /*mat*/) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestShapeBit(kGeoRunTimeShape)) return 0;
   if (!mother->TestShapeBit(kGeoEltu)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t a, b, dz;
   a = fRmin;
   b = fRmax;
   dz = fDz;
   if (fDz<0) dz=((TGeoEltu*)mother)->GetDz();
   if (fRmin<0)
      a = ((TGeoEltu*)mother)->GetA();
   if (fRmax<0) 
      a = ((TGeoEltu*)mother)->GetB();

   return (new TGeoEltu(a, b, dz));
}

//_____________________________________________________________________________
void TGeoEltu::InspectShape() const
{
// print shape parameters
   printf("*** Shape %s: TGeoEltu ***\n", GetName());
   printf("    A    = %11.5f\n", fRmin);
   printf("    B    = %11.5f\n", fRmax);
   printf("    dz   = %11.5f\n", fDz);
   printf(" Bounding box:\n");
   TGeoBBox::InspectShape();
}

//_____________________________________________________________________________
Double_t TGeoEltu::Safety(Double_t *point, Bool_t /*in*/) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t x0 = TMath::Abs(point[0]);
   Double_t y0 = TMath::Abs(point[1]);
   Double_t x1, y1, dx, dy;
   Double_t safr, safz;
   safr = safz = TGeoShape::Big();
   Double_t onepls = 1.+TGeoShape::Tolerance();
   Double_t onemin = 1.-TGeoShape::Tolerance();
   Double_t sqdist = x0*x0/(fRmin*fRmin)+y0*y0/(fRmax*fRmax);
   Bool_t in = kTRUE;
   if (sqdist>onepls) in = kFALSE;
   else if (sqdist<onemin) in = kTRUE;
   else return 0.;
   
   if (in) {
      x1 = fRmin*TMath::Sqrt(1.-(y0*y0)/(fRmax*fRmax));
      y1 = fRmax*TMath::Sqrt(1.-(x0*x0)/(fRmin*fRmin));
      dx = x1-x0;
      dy = y1-y0;
      if (TMath::Abs(dx)<TGeoShape::Tolerance()) return 0;
      safr = dx*dy/TMath::Sqrt(dx*dx+dy*dy);
      safz = fDz - TMath::Abs(point[2]);
      return TMath::Min(safr,safz);
   }   

   if (TMath::Abs(x0)<TGeoShape::Tolerance()) {
      safr = y0 - fRmax;
   } else {
      if (TMath::Abs(y0)<TGeoShape::Tolerance()) {
         safr = x0 - fRmin;
      } else {
         Double_t f = fRmin*fRmax/TMath::Sqrt(x0*x0*fRmax*fRmax+y0*y0*fRmin*fRmin);
         x1 = f*x0;
         y1 = f*y0;
         dx = x0-x1;
         dy = y0-y1;
         Double_t ast = fRmin*y1/fRmax;
         Double_t bct = fRmax*x1/fRmin;
         Double_t d = TMath::Sqrt(bct*bct+ast*ast);
         safr = (dx*bct+dy*ast)/d;
      }
   }
   safz = TMath::Abs(point[2])-fDz;            
   return TMath::Max(safr, safz);
}

//_____________________________________________________________________________
void TGeoEltu::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   if (TObject::TestBit(kGeoSavePrimitive)) return;
   out << "   // Shape: " << GetName() << " type: " << ClassName() << endl;
   out << "   a  = " << fRmin << ";" << endl;
   out << "   b  = " << fRmax << ";" << endl;
   out << "   dz = " << fDz << ";" << endl;
   out << "   TGeoShape *" << GetPointerName() << " = new TGeoEltu(\"" << GetName() << "\",a,b,dz);" << endl;
   TObject::SetBit(TGeoShape::kGeoSavePrimitive);
}

//_____________________________________________________________________________
void TGeoEltu::SetEltuDimensions(Double_t a, Double_t b, Double_t dz)
{
// Set dimensions of the eliptical tube.
   if ((a<=0) || (b<0) || (dz<0)) {
      SetShapeBit(kGeoRunTimeShape);
   }
   fRmin=a;
   fRmax=b;
   fDz=dz;
}   

//_____________________________________________________________________________
void TGeoEltu::SetDimensions(Double_t *param)
{
// Set shape dimensions starting from an array.
   Double_t a    = param[0];
   Double_t b    = param[1];
   Double_t dz   = param[2];
   SetEltuDimensions(a, b, dz);
}   

//_____________________________________________________________________________
void TGeoEltu::SetPoints(Double_t *points) const
{
// Create eliptical tube mesh points
   Double_t dz;
   Int_t j, n;

   n = gGeoManager->GetNsegments();
   Double_t dphi = 360./n;
   Double_t phi = 0;
   Double_t cph,sph;
   dz = fDz;

   Int_t indx = 0;
   Double_t r2,r;
   Double_t a2=fRmin*fRmin;
   Double_t b2=fRmax*fRmax;

   if (points) {
      for (j = 0; j < n; j++) {
         points[indx+6*n] = points[indx] = 0;
         indx++;
         points[indx+6*n] = points[indx] = 0;
         indx++;
         points[indx+6*n] = dz;
         points[indx]     =-dz;
         indx++;
      }
      for (j = 0; j < n; j++) {
         phi = j*dphi*TMath::DegToRad();
         sph=TMath::Sin(phi);
         cph=TMath::Cos(phi);
         r2=(a2*b2)/(b2+(a2-b2)*sph*sph);
         r=TMath::Sqrt(r2);
         points[indx+6*n] = points[indx] = r*cph;
         indx++;
         points[indx+6*n] = points[indx] = r*sph;
         indx++;
         points[indx+6*n]= dz;
         points[indx]    =-dz;
         indx++;
      }
   }
}

//_____________________________________________________________________________
void TGeoEltu::GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
{
// Returns numbers of vertices, segments and polygons composing the shape mesh.
   TGeoTube::GetMeshNumbers(nvert,nsegs,npols);
}

//_____________________________________________________________________________
Int_t TGeoEltu::GetNmeshVertices() const
{
// Returns the number of vertices on the mesh.
   return TGeoTube::GetNmeshVertices();
}   
   
//_____________________________________________________________________________
void TGeoEltu::SetPoints(Float_t *points) const
{
// Create eliptical tube mesh points
   Double_t dz;
   Int_t j, n;

   n = gGeoManager->GetNsegments();
   Double_t dphi = 360./n;
   Double_t phi = 0;
   Double_t cph,sph;
   dz = fDz;

   Int_t indx = 0;
   Double_t r2,r;
   Double_t a2=fRmin*fRmin;
   Double_t b2=fRmax*fRmax;

   if (points) {
      for (j = 0; j < n; j++) {
         points[indx+6*n] = points[indx] = 0;
         indx++;
         points[indx+6*n] = points[indx] = 0;
         indx++;
         points[indx+6*n] = dz;
         points[indx]     =-dz;
         indx++;
      }
      for (j = 0; j < n; j++) {
         phi = j*dphi*TMath::DegToRad();
         sph=TMath::Sin(phi);
         cph=TMath::Cos(phi);
         r2=(a2*b2)/(b2+(a2-b2)*sph*sph);
         r=TMath::Sqrt(r2);
         points[indx+6*n] = points[indx] = r*cph;
         indx++;
         points[indx+6*n] = points[indx] = r*sph;
         indx++;
         points[indx+6*n]= dz;
         points[indx]    =-dz;
         indx++;
      }
   }
}

//_____________________________________________________________________________
const TBuffer3D & TGeoEltu::GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
{
// Fills a static 3D buffer and returns a reference.
   static TBuffer3D buffer(TBuffer3DTypes::kGeneric);
   TGeoBBox::FillBuffer3D(buffer, reqSections, localFrame);

   if (reqSections & TBuffer3D::kRawSizes) {
      Int_t n = gGeoManager->GetNsegments();
      Int_t nbPnts = 4*n;
      Int_t nbSegs = 8*n;
      Int_t nbPols = 4*n;      
      if (buffer.SetRawSizes(nbPnts, 3*nbPnts, nbSegs, 3*nbSegs, nbPols, 6*nbPols)) {
         buffer.SetSectionsValid(TBuffer3D::kRawSizes);
      }
   }
   if ((reqSections & TBuffer3D::kRaw) && buffer.SectionsValid(TBuffer3D::kRawSizes)) {
      SetPoints(buffer.fPnts);
      if (!buffer.fLocalFrame) {
         TransformPoints(buffer.fPnts, buffer.NbPnts());
      }
      SetSegsAndPols(buffer);  
      buffer.SetSectionsValid(TBuffer3D::kRaw);
   }
      
   return buffer;
}
