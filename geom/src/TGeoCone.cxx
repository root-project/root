/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :  Andrei Gheata  - date Thu 31 Jan 2002 01:47:40 PM CET
// TGeoCone::Contains() and DistToOut() implemented by Mihaela Gheata

#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoPainter.h"
#include "TGeoCone.h"


/*************************************************************************
 * TGeoCone - conical tube  class. It has 5 parameters :
 *            dz - half length in z
 *            Rmin1, Rmax1 - inside and outside radii at -dz
 *            Rmin2, Rmax2 - inside and outside radii at +dz
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoCone.gif">
*/
//End_Html

/*************************************************************************
 * TGeoConeSeg - a phi segment of a conical tube. Has 7 parameters :
 *            - the same 5 as a cone;
 *            - first phi limit (in degrees)
 *            - second phi limit 
 *
 *************************************************************************/
//Begin_Html
/*
<img src="gif/TGeoCons.gif">
*/
//End_Html

ClassImp(TGeoCone)
   
//-----------------------------------------------------------------------------
TGeoCone::TGeoCone()
{
// Default constructor
   SetBit(TGeoShape::kGeoCone);
   fDz    = 0.0;
   fRmin1 = 0.0;
   fRmax1 = 0.0;
   fRmin2 = 0.0;
   fRmax2 = 0.0;
}   
//-----------------------------------------------------------------------------
TGeoCone::TGeoCone(Double_t dz, Double_t rmin1, Double_t rmax1,
                   Double_t rmin2, Double_t rmax2)
         :TGeoBBox(0, 0, 0)
{
// Default constructor specifying minimum and maximum radius
   SetBit(TGeoShape::kGeoCone);
   SetConeDimensions(dz, rmin1, rmax1, rmin2, rmax2);
   if ((dz<0) || (rmin1<0) || (rmax1<0) || (rmin2<0) || (rmax2<0)) {
      SetBit(kGeoRunTimeShape);
//      printf("cone : dz=%f, rmin1=%f, rmin2=%f, rmax1=%f, rmax2=%f\n",
//              dz, rmin1, rmax1, rmin2, rmax2);
   }
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoCone::TGeoCone(Double_t *param)
{
// Default constructor specifying minimum and maximum radius
// param[0] = dz
// param[1] = Rmin1
// param[2] = Rmax1
// param[3] = Rmin2
// param[4] = Rmax2
   SetBit(TGeoShape::kGeoCone);
   SetDimensions(param);
   if ((fDz<0) || (fRmin1<0) || (fRmax1<0) || (fRmin2<0) || (fRmax2<0))
      SetBit(kGeoRunTimeShape);
   else ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoCone::~TGeoCone()
{
// destructor
}
//-----------------------------------------------------------------------------   
void TGeoCone::ComputeBBox()
{
// compute bounding box of the sphere
   TGeoBBox *box = (TGeoBBox*)this;
   box->SetBoxDimensions(TMath::Max(fRmax1, fRmax2), TMath::Max(fRmax1, fRmax2), fDz);
   memset(fOrigin, 0, 3*sizeof(Double_t));
}   
//-----------------------------------------------------------------------------
Bool_t TGeoCone::Contains(Double_t *point)
{
// test if point is inside this cone
   if (TMath::Abs(point[2]) > fDz) return kFALSE;
   Double_t r2 = point[0]*point[0]+point[1]*point[1];
   Double_t rl = 0.5*(fRmin2*(point[2]+fDz)+fRmin1*(fDz-point[2]))/fDz;
   Double_t rh = 0.5*(fRmax2*(point[2]+fDz)+fRmax1*(fDz-point[2]))/fDz;
   if ((r2<rl*rl) || (r2>rh*rh)) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
Double_t TGeoCone::DistToOutS(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe,
                              Double_t dz, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2)
{
// compute distance from inside point to surface of the cone (static)
   Double_t saf[3];
   Double_t ro1 = 0.5*(rmin1+rmin2);
   Double_t tg1 = 0.5*(rmin2-rmin1)/dz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(rmax1+rmax2);
   Double_t tg2 = 0.5*(rmax2-rmax1)/dz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   
   if (iact<3 && safe) {
      if (ro1>1E-10) saf[0] = (r-rin)*cr1;
      else saf[0] = kBig;
      saf[1] = (rout-r)*cr2;
      saf[2] = dz-TMath::Abs(point[2]);
      *safe = TMath::Min(saf[0], TMath::Min(saf[1],saf[2]));
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = kBig;
   if (dir[2]>1E-20) 
      sz = (dz-point[2])/dir[2];
   else
      if (dir[2]<-1E-20) sz = -(dz+point[2])/dir[2];
   // Do Rmin
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];  
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];  
   Double_t t3=point[0]*point[0]+point[1]*point[1]; 
   Double_t sr1 = kBig;
   Double_t ds=0;
   Double_t b=0, c=0, d=0;
   Double_t u=0, v=0, w=0;
   if (ro1>0) {
      u=t1-tg1*tg1*dir[2]*dir[2];
      v=t2-tg1*dir[1]*rin;
      w=t3-rin*rin;
      if (u!=0) {
         b=v/u;
         c=w/u;
         d=b*b-c;
         if (d>=0) {
            ds = TMath::Sqrt(d);
            if (ds>=TMath::Abs(b)) sr1=ds-b;
            else if (b<=0)         sr1=-ds-b;
         }
      } else if (v<0) sr1=-0.5*w/v;
   } 
   // Do Rmax
   Double_t sr2=kBig;
   u=t1-tg2*tg2*dir[2]*dir[2];
   v=t2-tg2*dir[2]*rout;
   w=t3-rout*rout;
   if (u!=0) {
      b=v/u;
      c=w/u;
      d=b*b-c;
      if (d>=0) {
         ds = TMath::Sqrt(d);
         if (ds>=TMath::Abs(b)) sr2=ds-b;
         else if (b<=0) sr2 = -ds-b;
      }
   } else if (v>0) sr2=-0.5*w/v;
   return TMath::Min(TMath::Min(sr1, sr2), sz);                   
}
//-----------------------------------------------------------------------------
Double_t TGeoCone::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from inside point to surface of the cone
   Double_t saf[3];
   Double_t ro1 = 0.5*(fRmin1+fRmin2);
   Double_t tg1 = 0.5*(fRmin2-fRmin1)/fDz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(fRmax1+fRmax2);
   Double_t tg2 = 0.5*(fRmax2-fRmax1)/fDz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   
   if (iact<3 && safe) {
      if (ro1>1E-10) saf[0] = (r-rin)*cr1;
      else saf[0] = kBig;
      saf[1] = (rout-r)*cr2;
      saf[2] = fDz-TMath::Abs(point[2]);
      *safe = TMath::Min(saf[0], TMath::Min(saf[1],saf[2]));
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = kBig;
   if (dir[2]>1E-20) 
      sz = (fDz-point[2])/dir[2];
   else
      if (dir[2]<-1E-20) sz = -(fDz+point[2])/dir[2];
   // Do Rmin
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];  
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];  
   Double_t t3=point[0]*point[0]+point[1]*point[1]; 
   Double_t sr1 = kBig;
   Double_t ds=0;
   Double_t b=0, c=0, d=0;
   Double_t u=0, v=0, w=0;
   if (ro1>0) {
      u=t1-tg1*tg1*dir[2]*dir[2];
      v=t2-tg1*dir[1]*rin;
      w=t3-rin*rin;
      if (u!=0) {
         b=v/u;
         c=w/u;
         d=b*b-c;
         if (d>=0) {
            ds = TMath::Sqrt(d);
            if (ds>=TMath::Abs(b)) sr1=ds-b;
            else if (b<=0)         sr1=-ds-b;
         }
      } else if (v<0) sr1=-0.5*w/v;
   } 
   // Do Rmax
   Double_t sr2=kBig;
   u=t1-tg2*tg2*dir[2]*dir[2];
   v=t2-tg2*dir[2]*rout;
   w=t3-rout*rout;
   if (u!=0) {
      b=v/u;
      c=w/u;
      d=b*b-c;
      if (d>=0) {
         ds = TMath::Sqrt(d);
         if (ds>=TMath::Abs(b)) sr2=ds-b;
         else if (b<=0) sr2 = -ds-b;
      }
   } else if (v>0) sr2=-0.5*w/v;
   return TMath::Min(TMath::Min(sr1, sr2), sz);                   
}
//-----------------------------------------------------------------------------
Double_t TGeoCone::DistToInS(Double_t *point, Double_t *dir, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2,
                             Double_t dz, Double_t ro1, Double_t tg1, Double_t cr1, Double_t zv1,
                             Double_t ro2, Double_t tg2, Double_t cr2, Double_t zv2,
                             Double_t r2, Double_t rin, Double_t rout)
{
// compute distance to arbitrary cone from outside point
   Double_t snxt=kBig;
   // intersection with Z planes
   Double_t s, xi, yi, zi, riq, r1q, r2q;
   Double_t *norm=gGeoManager->GetNormalChecked();
   if (TMath::Abs(point[2])>dz) {
      if ((point[2]*dir[2])<0) {
         s=(TMath::Abs(point[2])-dz)/TMath::Abs(dir[2]);
         xi=point[0]+s*dir[0];
         yi=point[1]+s*dir[1];
         riq=xi*xi+yi*yi;
         norm[0]=norm[1]=0;
         if (point[2]<0) {
            r1q=rmin1*rmin1;
            r2q=rmax1*rmax1;
            norm[2]=-1;
         } else {
            r1q=rmin2*rmin2;
            r2q=rmax2*rmax2;
            norm[2]=1;
         }      
         if ((r1q<=riq) && (riq<=r2q)) return s;
      }   
   }
   // intersection with cones
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];
   Double_t t3=r2;
   // outer cone
   Double_t u,v,w,b,c,d,fn;
   if ((zv2*point[2]>zv2*zv2) || (r2>rout*rout)) {
      u=t1-(tg2*tg2*dir[2]*dir[2]);
      v=t2-tg2*dir[2]*(tg2*point[2]+ro2);
      w=t3-rout*rout;
      // track parallel to cone ?
      if (u!=0) {
         b=v/u;
         c=w/u;
         d=b*b-c;
         if (d>=0) {
            s=-b-TMath::Sqrt(d);
            if (s>=0) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=-cr1*tg1;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=xi/fn;
                  norm[1]=yi/fn;
                  snxt=s;
               }
            }
            if (snxt>(-b)) {
               s=-b+TMath::Sqrt(d);
               if (s>=0) {
                  zi=point[2]+s*dir[2];
                  if (TMath::Abs(zi)<dz) {
                     xi=point[0]+s*dir[0];
                     yi=point[1]+s*dir[1];
                     norm[2]=-cr1*tg1;
                     fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                     norm[0]=xi/fn;
                     norm[1]=yi/fn;
                     snxt=s;
                  }
               }
            }
         }   
      } else {
         if (v!=0) {
            s=-0.5*w/v;
            if (s>=0) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<=dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=-cr1*tg1;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=xi/fn;
                  norm[1]=yi/fn;
                  snxt = s;
               }
            }
         }
      }
   }     
   // test inner cone
   if (ro1>0) {
      u=t1-(tg1*tg1*dir[2]*dir[2]);
      v=t2-tg1*dir[2]*(tg1*point[2]+ro1);
      w=t3-rin*rin;
      // track parallel to cone ?
      if (u!=0) {
         b=v/u;
         c=w/u;
         d=b*b-c;
         if (d>=0) {
            s=-b-TMath::Sqrt(d);
            if ((s>=0) && (s<snxt)) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=cr2*tg2;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=-xi/fn;
                  norm[1]=-yi/fn;
                  return s;
               }
            }
            s=-b+TMath::Sqrt(d);
            if ((s>=0) && (s<snxt)) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=cr2*tg2;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=-xi/fn;
                  norm[1]=-yi/fn;
                  return s;
               }
            }
         }
      } else {
         if (v!=0) {
            s=-0.5*w/v;
            if ((s>=0) && (s<snxt)) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<=dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=cr2*tg2;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=-xi/fn;
                  norm[1]=-yi/fn;
                  return s;
               }
            }
         }
      }
   }     
   return snxt;               
}
                             
//-----------------------------------------------------------------------------
Double_t TGeoCone::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from outside point to surface of the tube
   Double_t saf[3];
   Double_t ro1=0.5*(fRmin1+fRmin2);
   Double_t tg1=0.5*(fRmin2-fRmin1)/fDz;
   Double_t cr1=1./TMath::Sqrt(1.0+tg1*tg1);
   Double_t zv1=kBig;
   if (fRmin1!=fRmin2) zv1=-ro1/tg1;
   Double_t ro2=0.5*(fRmax1+fRmax2);
   Double_t tg2=0.5*(fRmax2-fRmax1)/fDz;
   Double_t cr2=1./TMath::Sqrt(1.0+tg2*tg2);
   Double_t zv2=kBig;
   if (fRmax1!=fRmax2) zv2=-ro2/tg2;
   
   Double_t r2=point[0]*point[0]+point[1]*point[1];
   Double_t r=TMath::Sqrt(r2);
   Double_t rin=TMath::Abs(tg1*point[2]+ro1);
   Double_t rout=TMath::Abs(tg2*point[2]+ro2);
   // conmpute safe radius
   if (iact<3 && safe) {
      saf[0]=(rin-r)*cr1;
      saf[1]=(r-rout)*cr2;
      saf[2]=TMath::Abs(point[2])-fDz;
      *safe = saf[TMath::LocMax(3, &saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   return TGeoCone::DistToInS(point, dir,fRmin1,fRmax1,fRmin2,fRmax2,fDz,
                              ro1,tg1,cr1,zv1,ro2,tg2,cr2,zv2,r2,rin,rout);
}
//-----------------------------------------------------------------------------
Int_t TGeoCone::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   Int_t n = TGeoManager::kGeoDefaultNsegments;
   const Int_t numPoints = 4*n;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}
//-----------------------------------------------------------------------------
Double_t TGeoCone::DistToSurf(Double_t *point, Double_t *dir)
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoCone::Draw(Option_t *option)
{
// draw this shape according to option
}
//-----------------------------------------------------------------------------
TGeoShape *TGeoCone::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoCone)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t rmin1, rmax1, rmin2, rmax2, dz;
   rmin1 = fRmin1;
   rmax1 = fRmax1;
   rmin2 = fRmin2;
   rmax2 = fRmax2;
   dz = fDz;
   if (fDz<0) dz=((TGeoCone*)mother)->GetDz();
   if (fRmin1<0) 
      rmin1 = ((TGeoCone*)mother)->GetRmin1();
   if (fRmax1<0)
      rmax1 = ((TGeoCone*)mother)->GetRmax1();
   if (fRmin2<0) 
      rmin2 = ((TGeoCone*)mother)->GetRmin2();
   if (fRmax2<0)
      rmax2 = ((TGeoCone*)mother)->GetRmax2();

   return (new TGeoCone(rmin1, rmax1, rmin2, rmax2, dz));
}
//-----------------------------------------------------------------------------
void TGeoCone::InspectShape()
{
// print shape parameters
   printf("*** TGeoCone parameters ***\n");
   printf("    dz    = %11.5f\n", fDz);
   printf("    Rmin1 = %11.5f\n", fRmin1);
   printf("    Rmax1 = %11.5f\n", fRmax1);
   printf("    Rmin2 = %11.5f\n", fRmin2);
   printf("    Rmax2 = %11.5f\n", fRmax2);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoCone::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoPainter *painter = (TGeoPainter*)gGeoManager->GetMakeDefPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintTube(vol, option);
}
//-----------------------------------------------------------------------------
void TGeoCone::NextCrossing(TGeoParamCurve *c, Double_t *point)
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoCone::Safety(Double_t *point, Double_t *spoint, Option_t *option)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoCone::SetConeDimensions(Double_t dz, Double_t rmin1, Double_t rmax1,
                             Double_t rmin2, Double_t rmax2)
{
   if (rmin1>=0) {
      if (rmax1>0) {
         if (rmin1<rmax1) {
         // normal rmin/rmax
            fRmin1 = rmin1;
            fRmax1 = rmax1;
         } else {
            fRmin1 = rmax1;
            fRmax1 = rmin1;
            Warning("SetConeDimensions", "rmin1>rmax1 Switch rmin1<->rmax1");
         }
      } else {
         // run-time
         fRmin1 = rmin1;
         fRmax1 = rmax1;
      }
   } else {
      // run-time
      fRmin1 = rmin1;
      fRmax1 = rmax1;
   }               
   if (rmin2>=0) {
      if (rmax2>0) {
         if (rmin2<rmax2) {
         // normal rmin/rmax
            fRmin2 = rmin2;
            fRmax2 = rmax2;
         } else {
            fRmin2 = rmax2;
            fRmax2 = rmin2;
            Warning("SetConeDimensions", "rmin2>rmax2 Switch rmin2<->rmax2");
         }
      } else {
         // run-time
         fRmin2 = rmin2;
         fRmax2 = rmax2;
      }
   } else {
      // run-time
      fRmin2 = rmin2;
      fRmax2 = rmax2;
   }               
   
   fDz   = dz;
}   
//-----------------------------------------------------------------------------
void TGeoCone::SetDimensions(Double_t *param)
{
   Double_t dz    = param[0];
   Double_t rmin1 = param[1];
   Double_t rmax1 = param[2];
   Double_t rmin2 = param[3];
   Double_t rmax2 = param[4];
   SetConeDimensions(dz, rmin1, rmax1, rmin2, rmax2);
}   
//-----------------------------------------------------------------------------
void TGeoCone::SetPoints(Double_t *buff) const
{
// create cone mesh points
    Double_t dz, phi, dphi;
    Int_t j, n;

    n = TGeoManager::kGeoDefaultNsegments;
    dphi = 360./n;
    dz    = fDz;
    Int_t indx = 0;

    if (buff) {
        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx++] = fRmin1 * TMath::Cos(phi);
            buff[indx++] = fRmin1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx++] = fRmax1 * TMath::Cos(phi);
            buff[indx++] = fRmax1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }

        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx++] = fRmin2 * TMath::Cos(phi);
            buff[indx++] = fRmin2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }

        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx++] = fRmax2 * TMath::Cos(phi);
            buff[indx++] = fRmax2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoCone::SetPoints(Float_t *buff) const
{
// create cone mesh points
    Double_t dz, phi, dphi;
    Int_t j, n;

    n = TGeoManager::kGeoDefaultNsegments;
    dphi = 360./n;
    dz    = fDz;
    Int_t indx = 0;

    if (buff) {
        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx++] = fRmin1 * TMath::Cos(phi);
            buff[indx++] = fRmin1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx++] = fRmax1 * TMath::Cos(phi);
            buff[indx++] = fRmax1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }

        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx++] = fRmin2 * TMath::Cos(phi);
            buff[indx++] = fRmin2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }

        for (j = 0; j < n; j++) {
            phi = j*dphi*kDegRad;
            buff[indx++] = fRmax2 * TMath::Cos(phi);
            buff[indx++] = fRmax2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoCone::Sizeof3D() const
{
// fill size of this 3-D object
    Int_t n = TGeoManager::kGeoDefaultNsegments;
    gSize3D.numPoints += n*4;
    gSize3D.numSegs   += n*8;
    gSize3D.numPolys  += n*4;
}


ClassImp(TGeoConeSeg)
   
//-----------------------------------------------------------------------------
TGeoConeSeg::TGeoConeSeg()
{
// Default constructor
   SetBit(TGeoShape::kGeoConeSeg);
   fPhi1 = fPhi2 = 0.0;
}   
//-----------------------------------------------------------------------------
TGeoConeSeg::TGeoConeSeg(Double_t dz, Double_t rmin1, Double_t rmax1, 
                          Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2)
            :TGeoCone(dz, rmin1, rmax1, rmin2, rmax2)
{
// Default constructor specifying minimum and maximum radius
   SetBit(TGeoShape::kGeoConeSeg);
   SetConsDimensions(dz, rmin1, rmax1, rmin2, rmax2, phi1, phi2);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoConeSeg::TGeoConeSeg(Double_t *param)
{
// Default constructor specifying minimum and maximum radius
// param[0] = dz
// param[1] = Rmin1
// param[2] = Rmax1
// param[3] = Rmin2
// param[4] = Rmax2
// param[5] = phi1
// param[6] = phi2
   SetBit(TGeoShape::kGeoConeSeg);
   SetDimensions(param);
   ComputeBBox();
}
//-----------------------------------------------------------------------------
TGeoConeSeg::~TGeoConeSeg()
{
// destructor
}
//-----------------------------------------------------------------------------   
void TGeoConeSeg::ComputeBBox()
{
// compute bounding box of the tube segment
   Double_t rmin, rmax;
   rmin = TMath::Min(fRmin1, fRmin2);
   rmax = TMath::Max(fRmax1, fRmax2);

   Double_t xc[4];
   Double_t yc[4];
   xc[0] = rmax*TMath::Cos(fPhi1*kDegRad);
   yc[0] = rmax*TMath::Sin(fPhi1*kDegRad);
   xc[1] = rmax*TMath::Cos(fPhi2*kDegRad);
   yc[1] = rmax*TMath::Sin(fPhi2*kDegRad);
   xc[2] = rmin*TMath::Cos(fPhi1*kDegRad);
   yc[2] = rmin*TMath::Sin(fPhi1*kDegRad);
   xc[3] = rmin*TMath::Cos(fPhi2*kDegRad);
   yc[3] = rmin*TMath::Sin(fPhi2*kDegRad);

   Double_t xmin = xc[TMath::LocMin(4, &xc[0])];
   Double_t xmax = xc[TMath::LocMax(4, &xc[0])]; 
   Double_t ymin = yc[TMath::LocMin(4, &yc[0])]; 
   Double_t ymax = yc[TMath::LocMax(4, &yc[0])];

   Double_t dp = fPhi2-fPhi1;
   if (dp<0) dp+=360;
   Double_t ddp = -fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) xmax = rmax;
   ddp = 90-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) ymax = rmax;
   ddp = 180-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) xmin = -rmax;
   ddp = 270-fPhi1;
   if (ddp<0) ddp+= 360;
   if (ddp>360) ddp-=360;
   if (ddp<=dp) ymin = -rmax;
   fOrigin[0] = (xmax+xmin)/2;
   fOrigin[1] = (ymax+ymin)/2;
   fOrigin[2] = 0;
   fDX = (xmax-xmin)/2;
   fDY = (ymax-ymin)/2;
   fDZ = fDz;
}   
//-----------------------------------------------------------------------------
Bool_t TGeoConeSeg::Contains(Double_t *point)
{
// test if point is inside this sphere
   if (!TGeoCone::Contains(point)) return kFALSE;
   Double_t phi = TMath::ATan2(point[1], point[0]) * kRadDeg;
   if (phi < 0 ) phi+=360.;
   Double_t dphi = fPhi2 - fPhi1;
   if (dphi < 0) dphi+=360.;
   Double_t ddp = phi-fPhi1;
   if (ddp < 0) ddp+=360.; 
//   if (ddp > 360) ddp-=360;
   if (ddp > dphi) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
Double_t TGeoConeSeg::DistToPhiMin(Double_t *point, Double_t *dir, Double_t s1, Double_t c1,
                                   Double_t s2, Double_t c2, Double_t sm, Double_t cm)
{
// compute distance from poin to both phi planes. Return minimum.
   Double_t sfi1=kBig;
   Double_t sfi2=kBig;
   Double_t s=0;
   Double_t un = dir[0]*s1-dir[1]*c1;
   if (un!=0) {
      s=(point[1]*c1-point[0]*s1)/un;
      if (s>=0) {
         if (((point[1]+s*dir[1])*cm-(point[0]+s*dir[0])*sm)<=0) sfi1=s;
      }   
   }
   un = dir[0]*s2-dir[1]*c2;    
   if (un!=0) {
      s=(point[1]*c2-point[0]*s2)/un;
      if (s>=0) {
         if (((point[1]+s*dir[1])*cm-(point[0]+s*dir[0])*sm)>=0) sfi2=s;
      }   
   }
   return TMath::Min(sfi1, sfi2);
}
//-----------------------------------------------------------------------------
Double_t TGeoConeSeg::DistToOutS(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe,
               Double_t dz, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2)
{
// compute distance from inside point to surface of the cone segment (static)
   Double_t saf[4];
   Double_t ph1 = phi1*kDegRad;
   Double_t ph2 = phi2*kDegRad;
   if (ph2<ph1) ph2+=2.*TMath::Pi();
   Double_t phim = 0.5*(ph1+ph2);
   Double_t c1 = TMath::Cos(ph1);
   Double_t c2 = TMath::Cos(ph2);
   Double_t s1 = TMath::Sin(ph1);
   Double_t s2 = TMath::Sin(ph2);
   Double_t cm = TMath::Cos(phim);
   Double_t sm = TMath::Sin(phim);

   Double_t ro1 = 0.5*(rmin1+rmin2);
   Double_t tg1 = 0.5*(rmin2-rmin1)/dz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(rmax1+rmax2);
   Double_t tg2 = 0.5*(rmax2-rmax1)/dz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   
   if (iact<3 && safe) {
      if (ro1>1E-10) saf[0] = (r-rin)*cr1;
      else saf[0] = kBig;
      saf[1] = (rout-r)*cr2;
      saf[2] = dz-TMath::Abs(point[2]);
      if ((point[1]*cm-point[0]*sm)<=0) saf[3]=TMath::Abs(point[0]*s1-point[1]*c1);
      else                              saf[3]=TMath::Abs(point[0]*s2-point[1]*c2);
      *safe = saf[TMath::LocMin(4, &saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = kBig;
   if (dir[2]>1E-20) 
      sz = (dz-point[2])/dir[2];
   else
      if (dir[2]<-1E-20) sz = -(dz+point[2])/dir[2];
   // Do Rmin
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];  
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];  
   Double_t t3=point[0]*point[0]+point[1]*point[1]; 
   Double_t sr1 = kBig;
   Double_t b=0, c=0, d=0;
   Double_t u=0, v=0, w=0;
   Double_t ds=0;
   if (ro1>0) {
      u=t1-tg1*tg1*dir[2]*dir[2];
      v=t2-tg1*dir[2]*rin;
      w=t3-rin*rin;
      if (u!=0) {
         b=v/u;
         c=w/u;
         d=b*b-c;
         if (d>=0) {
            ds = TMath::Sqrt(d);
            if (ds>=TMath::Abs(b)) sr1=ds-b;
            else if (b<=0)         sr1=-ds-b;
         }
      } else if (v<0) sr1=-0.5*w/v;
   } 
   // Do Rmax
   Double_t sr2=kBig;
   u=t1-tg2*tg2*dir[2]*dir[2];
   v=t2-tg2*dir[2]*rout;
   w=t3-rout*rout;
   if (u!=0) {
      b=v/u;
      c=w/u;
      d=b*b-c;
      if (d>=0) {
         ds = TMath::Sqrt(d);
         if (ds>=TMath::Abs(b)) sr2=ds-b;
         else if (b<=0) sr2 = -ds-b;
      }
   } else if (v>0) sr2=-0.5*w/v;

   Double_t sr = TMath::Min(sr1, sr2);

   // phi planes

   Double_t sfmin=TGeoConeSeg::DistToPhiMin(point, dir, s1, c1, s2, c2, sm, cm);
   return TMath::Min(TMath::Min(sz,sr), sfmin);      
}
//-----------------------------------------------------------------------------
Double_t TGeoConeSeg::DistToOut(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from inside point to surface of the tube segment
   Double_t saf[4];
   Double_t phi1 = fPhi1*kDegRad;
   Double_t phi2 = fPhi2*kDegRad;
   if (phi2<phi1) phi2+=2.*TMath::Pi();
   Double_t phim = 0.5*(phi1+phi2);
   Double_t c1 = TMath::Cos(phi1);
   Double_t c2 = TMath::Cos(phi2);
   Double_t s1 = TMath::Sin(phi1);
   Double_t s2 = TMath::Sin(phi2);
   Double_t cm = TMath::Cos(phim);
   Double_t sm = TMath::Sin(phim);

   Double_t ro1 = 0.5*(fRmin1+fRmin2);
   Double_t tg1 = 0.5*(fRmin2-fRmin1)/fDz;
   Double_t cr1 = 1./TMath::Sqrt(1.+tg1*tg1);
   Double_t ro2 = 0.5*(fRmax1+fRmax2);
   Double_t tg2 = 0.5*(fRmax2-fRmax1)/fDz;
   Double_t cr2 = 1./TMath::Sqrt(1.+tg2*tg2);
   
   Double_t r=TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t rin = tg1*point[2]+ro1;
   Double_t rout = tg2*point[2]+ro2;
   
   if (iact<3 && safe) {
      if (ro1>1E-10) saf[0] = (r-rin)*cr1;
      else saf[0] = kBig;
      saf[1] = (rout-r)*cr2;
      saf[2] = fDz-TMath::Abs(point[2]);
      if ((point[1]*cm-point[0]*sm)<=0) saf[3]=TMath::Abs(point[0]*s1-point[1]*c1);
      else                              saf[3]=TMath::Abs(point[0]*s2-point[1]*c2);
      *safe = saf[TMath::LocMin(4, &saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   // compute distance to surface 
   // Do Z
   Double_t sz = kBig;
   if (dir[2]>1E-20) 
      sz = (fDz-point[2])/dir[2];
   else
      if (dir[2]<-1E-20) sz = -(fDz+point[2])/dir[2];
   // Do Rmin
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];  
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];  
   Double_t t3=point[0]*point[0]+point[1]*point[1]; 
   Double_t sr1 = kBig;
   Double_t b=0, c=0, d=0;
   Double_t u=0, v=0, w=0;
   Double_t ds=0;
   if (ro1>0) {
      u=t1-tg1*tg1*dir[2]*dir[2];
      v=t2-tg1*dir[2]*rin;
      w=t3-rin*rin;
      if (u!=0) {
         b=v/u;
         c=w/u;
         d=b*b-c;
         if (d>=0) {
            ds = TMath::Sqrt(d);
            if (ds>=TMath::Abs(b)) sr1=ds-b;
            else if (b<=0)         sr1=-ds-b;
         }
      } else if (v<0) sr1=-0.5*w/v;
   } 
   // Do Rmax
   Double_t sr2=kBig;
   u=t1-tg2*tg2*dir[2]*dir[2];
   v=t2-tg2*dir[2]*rout;
   w=t3-rout*rout;
   if (u!=0) {
      b=v/u;
      c=w/u;
      d=b*b-c;
      if (d>=0) {
         ds = TMath::Sqrt(d);
         if (ds>=TMath::Abs(b)) sr2=ds-b;
         else if (b<=0) sr2 = -ds-b;
      }
   } else if (v>0) sr2=-0.5*w/v;
   Double_t sr = TMath::Min(sr1, sr2);
   // phi planes

   Double_t sfmin=DistToPhiMin(point, dir, s1, c1, s2, c2, sm, cm);
   return TMath::Min(TMath::Min(sz,sr), sfmin);      
}
//-----------------------------------------------------------------------------
Double_t TGeoConeSeg::DistToInS(Double_t *point, Double_t *dir, Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2,
                             Double_t dz, Double_t ro1, Double_t tg1, Double_t cr1, Double_t zv1,
                             Double_t ro2, Double_t tg2, Double_t cr2, Double_t zv2,
                             Double_t r2, Double_t rin, Double_t rout, Double_t c1, Double_t s1,
                             Double_t c2, Double_t s2, Double_t cfio, Double_t sfio, Double_t cdfi)
{
// compute distance from outside point to surface of arbitrary tube
   Double_t snxt=kBig;
   Double_t cpsi=kBig;
   if (r2>0) cpsi=(point[0]*cfio-point[1]*sfio)/TMath::Sqrt(r2);
   // intersection with Z planes
   Double_t s, xi, yi, zi, riq, r1q, r2q, ri;
   Double_t *norm=gGeoManager->GetNormalChecked();
   if (TMath::Abs(point[2])>dz) {
      if ((point[2]*dir[2])<0) {
         s=(TMath::Abs(point[2])-dz)/TMath::Abs(dir[2]);
         xi=point[0]+s*dir[0];
         yi=point[1]+s*dir[1];
         riq=xi*xi+yi*yi;
         norm[0]=norm[1]=0;
         if (point[2]<0) {
            r1q=rmin1*rmin1;
            r2q=rmax1*rmax1;
            norm[2]=-1;
         } else {
            r1q=rmin2*rmin2;
            r2q=rmax2*rmax2;
            norm[2]=1;
         }      
         if ((r1q<=riq) && (riq<=r2q)) {
            if (riq==0) return s;
            cpsi=(xi*cfio+yi*sfio)/TMath::Sqrt(riq);
            if (cpsi>=cdfi) return s;
         }   
      }   
   }
   // intersection with cones
   Double_t t1=dir[0]*dir[0]+dir[1]*dir[1];
   Double_t t2=point[0]*dir[0]+point[1]*dir[1];
   Double_t t3=r2;
   // outer cone
   Double_t u,v,w,b,c,d,fn;
   if ((zv2*point[2]>zv2*zv2) || (r2>rout*rout)) {
      u=t1-(tg2*tg2*dir[2]*dir[2]);
      v=t2-tg2*dir[2]*(tg2*point[2]+ro2);
      w=t3-rout*rout;
      // track parallel to cone ?
      if (u!=0) {
         b=v/u;
         c=w/u;
         d=b*b-c;
         if (d>=0) {
            s=-b-TMath::Sqrt(d);
            if (s>=0) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=-cr1*tg1;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=xi/fn;
                  norm[1]=yi/fn;
                  if (zi==zv2) snxt=s;
                  else {
                     ri=tg2*zi+ro2;
                     cpsi=(xi*cfio+yi*sfio)/ri;
                     if (cpsi>=cdfi) snxt=s;
                  }   
               }
            }
            if (snxt>(-b)) {
               s=-b+TMath::Sqrt(d);
               if (s>=0) {
                  zi=point[2]+s*dir[2];
                  if (TMath::Abs(zi)<dz) {
                     xi=point[0]+s*dir[0];
                     yi=point[1]+s*dir[1];
                     norm[2]=-cr1*tg1;
                     fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                     norm[0]=xi/fn;
                     norm[1]=yi/fn;
                     if (zi==zv2) snxt=s;
                     else {
                        ri=tg2*zi+ro2;
                        cpsi=(xi*cfio+yi*sfio)/ri;
                        if (cpsi>=cdfi) snxt=s;
                     }   
                  }
               }
            }
         }   
      } else {
         if (v!=0) {
            s=-0.5*w/v;
            if (s>=0) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<=dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=-cr1*tg1;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=xi/fn;
                  norm[1]=yi/fn;
                  if (zi==zv2) snxt=s;
                  else {
                     ri=tg2*zi+ro2;
                     cpsi=(xi*cfio+yi*sfio)/ri;
                     if (cpsi>=cdfi) snxt=s;
                  }   
               }
            }
         }
      }
   }     
   // test inner cone
   if (ro1>0) {
      u=t1-(tg1*tg1*dir[2]*dir[2]);
      v=t2-tg1*dir[2]*(tg1*point[2]+ro1);
      w=t3-rin*rin;
      // track parallel to cone ?
      if (u!=0) {
         b=v/u;
         c=w/u;
         d=b*b-c;
         if (d>=0) {
            s=-b-TMath::Sqrt(d);
            if ((s>=0) && (s<snxt)) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=cr2*tg2;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=-xi/fn;
                  norm[1]=-yi/fn;
                  if (zi==zv1) snxt=s;
                  else {
                     ri=tg1*zi+ro1;
                     cpsi=(xi*cfio+yi*sfio)/ri;
                     if (cpsi>=cdfi) snxt=s;
                  }   
               }
            }
            s=-b+TMath::Sqrt(d);
            if ((s>=0) && (s<snxt)) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=cr2*tg2;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=-xi/fn;
                  norm[1]=-yi/fn;
                  if (zi==zv1) snxt=s;
                  else {
                     ri=tg1*zi+ro1;
                     cpsi=(xi*cfio+yi*sfio)/ri;
                     if (cpsi>=cdfi) snxt=s;
                  }   
               }
            }
         }
      } else {
         if (v!=0) {
            s=-0.5*w/v;
            if ((s>=0) && (s<snxt)) {
               zi=point[2]+s*dir[2];
               if (TMath::Abs(zi)<=dz) {
                  xi=point[0]+s*dir[0];
                  yi=point[1]+s*dir[1];
                  norm[2]=cr2*tg2;
                  fn=TMath::Sqrt((xi*xi+yi*yi)/(1.0-norm[2]*norm[2]));
                  norm[0]=-xi/fn;
                  norm[1]=-yi/fn;
                  if (zi==zv1) snxt=s;
                  else {
                     ri=tg1*zi+ro1;
                     cpsi=(xi*cfio+yi*sfio)/ri;
                     if (cpsi>=cdfi) snxt=s;
                  }   
               }
            }
         }
      }
   }     

   // check phi planes
   Double_t un;
   un=dir[0]*s1-dir[1]*c1;
   if (un!=0) {
      s=(point[1]*c1-point[0]*s1)/un;
      if ((s>=0) && (s<snxt)) {
         zi=point[2]+s*dir[2];
         if (TMath::Abs(zi)<=dz) {
            xi=point[0]+s*dir[0];
            yi=point[1]+s*dir[1];
            riq=xi*xi+yi*yi;
            r1q=(tg1*zi+ro1)*(tg1*zi+ro1);
            r2q=(tg2*zi+ro2)*(tg2*zi+ro2);
            if ((r1q<=riq) && (riq<=r2q)) {
               if ((yi*cfio-xi*sfio)<=0) {
                  norm[0] = s1;
                  norm[1] = -c1;
                  norm[2] = 0;
                  snxt = s;
               }
            }
         }
      }
   }               
   un=dir[0]*s2-dir[1]*c2;
   if (un!=0) {
      s=(point[1]*c2-point[0]*s2)/un;
      if ((s>=0) && (s<snxt)) {
         zi=point[2]+s*dir[2];
         if (TMath::Abs(zi)<=dz) {
            xi=point[0]+s*dir[0];
            yi=point[1]+s*dir[1];
            riq=xi*xi+yi*yi;
            r1q=(tg1*zi+ro1)*(tg1*zi+ro1);
            r2q=(tg2*zi+ro2)*(tg2*zi+ro2);
            if ((r1q<=riq) && (riq<=r2q)) {
               if ((yi*cfio-xi*sfio)>=0) {
                  norm[0] = -s2;
                  norm[1] = c2;
                  norm[2] = 0;
                  snxt = s;
               }
            }
         }
      }
   }               
   return snxt;               
}
//-----------------------------------------------------------------------------
Double_t TGeoConeSeg::DistToIn(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe)
{
// compute distance from outside point to surface of the tube
   Double_t saf[4];
   Double_t ro1=0.5*(fRmin1+fRmin2);
   Double_t tg1=0.5*(fRmin2-fRmin1)/fDz;
   Double_t cr1=1./TMath::Sqrt(1.0+tg1*tg1);
   Double_t zv1=kBig;
   if (fRmin1!=fRmin2) zv1=-ro1/tg1;
   Double_t ro2=0.5*(fRmax1+fRmax2);
   Double_t tg2=0.5*(fRmax2-fRmax1)/fDz;
   Double_t cr2=1./TMath::Sqrt(1.0+tg2*tg2);
   Double_t zv2=kBig;
   if (fRmax1!=fRmax2) zv2=-ro2/tg2;

   Double_t phi1 = fPhi1*kDegRad;
   Double_t phi2 = fPhi2*kDegRad;
   if (phi2<phi1) phi2+=2.*TMath::Pi();
   Double_t c1 = TMath::Cos(phi1);
   Double_t s1 = TMath::Sin(phi1);
   Double_t c2 = TMath::Cos(phi2);
   Double_t s2 = TMath::Sin(phi2);
   Double_t fio = 0.5*(phi1+phi2);
   Double_t cfio = TMath::Cos(fio);
   Double_t sfio = TMath::Sin(fio);
   Double_t dfi = 0.5*(phi2-phi1);
   Double_t cdfi = TMath::Cos(dfi);
   Double_t cpsi;

   Double_t r2=point[0]*point[0]+point[1]*point[1];
   Double_t r=TMath::Sqrt(r2);
   Double_t rin=TMath::Abs(tg1*point[2]+ro1);
   Double_t rout=TMath::Abs(tg2*point[2]+ro2);
   // compute safe radius
   if (iact<3 && safe) {
      saf[0]=(rin-r)*cr1;
      saf[1]=(r-rout)*cr2;
      saf[2]=TMath::Abs(point[2])-fDz;
      saf[3] = 0;
      if (r>0) {
         cpsi=(point[0]*cfio-point[1]*sfio)/r;
         if (cpsi<cdfi) saf[3]=TMath::Abs(point[0]*s1-point[1]*c1);
         else saf[3]=TMath::Abs(point[0]*s2-point[1]*c2);
      }   
      *safe = saf[TMath::LocMax(4, &saf[0])];
      if (iact==0) return kBig;
      if ((iact==1) && (*safe>step)) return step;
   }
   return TGeoConeSeg::DistToInS(point, dir,fRmin1,fRmax1,fRmin2,fRmax2,fDz,
                              ro1,tg1,cr1,zv1,ro2,tg2,cr2,zv2,r2,rin,rout,c1,s1,c2,s2,cfio,sfio,cdfi);
}
//-----------------------------------------------------------------------------
Int_t TGeoConeSeg::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each corner
   Int_t n = TGeoManager::kGeoDefaultNsegments+1;
   const Int_t numPoints = 4*n;
   return ShapeDistancetoPrimitive(numPoints, px, py);
}
//-----------------------------------------------------------------------------
Double_t TGeoConeSeg::DistToSurf(Double_t *point, Double_t *dir)
{
// computes the distance to next surface of the sphere along a ray
// starting from given point to the given direction.
   return kBig;
}
//-----------------------------------------------------------------------------
void TGeoConeSeg::Draw(Option_t *option)
{
// draw this shape according to option
}
//-----------------------------------------------------------------------------
TGeoShape *TGeoConeSeg::GetMakeRuntimeShape(TGeoShape *mother) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   if (!TestBit(kGeoRunTimeShape)) return 0;
   if (mother->IsRunTimeShape() || !mother->TestBit(kGeoConeSeg)) {
      Error("GetMakeRuntimeShape", "invalid mother");
      return 0;
   }
   Double_t rmin1, rmax1, rmin2, rmax2, dz;
   rmin1 = fRmin1;
   rmax1 = fRmax1;
   rmin2 = fRmin2;
   rmax2 = fRmax2;
   dz = fDz;
   if (fDz<0) dz=((TGeoCone*)mother)->GetDz();
   if (fRmin1<0) 
      rmin1 = ((TGeoCone*)mother)->GetRmin1();
   if ((fRmax1<0) || (fRmax1<fRmin1))
      rmax1 = ((TGeoCone*)mother)->GetRmax1();
   if (fRmin2<0) 
      rmin2 = ((TGeoCone*)mother)->GetRmin2();
   if ((fRmax2<0) || (fRmax2<fRmin2))
      rmax2 = ((TGeoCone*)mother)->GetRmax2();

   return (new TGeoConeSeg(rmin1, rmax1, rmin2, rmax2, dz, fPhi1, fPhi2));
}
//-----------------------------------------------------------------------------
void TGeoConeSeg::InspectShape()
{
// print shape parameters
   printf("*** TGeoConeSeg parameters ***\n");
   printf("    dz    = %11.5f\n", fDz);
   printf("    Rmin1 = %11.5f\n", fRmin1);
   printf("    Rmax1 = %11.5f\n", fRmax1);
   printf("    Rmin2 = %11.5f\n", fRmin2);
   printf("    Rmax2 = %11.5f\n", fRmax2);
   printf("    phi1  = %11.5f\n", fPhi1);
   printf("    phi2  = %11.5f\n", fPhi2);
   TGeoBBox::InspectShape();
}
//-----------------------------------------------------------------------------
void TGeoConeSeg::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoPainter *painter = (TGeoPainter*)gGeoManager->GetMakeDefPainter();
   if (!painter) return;
   TGeoVolume *vol = gGeoManager->GetCurrentVolume();
   if (vol->GetShape() != (TGeoShape*)this) return;
   painter->PaintTubs(vol, option);
}
//-----------------------------------------------------------------------------
void TGeoConeSeg::NextCrossing(TGeoParamCurve *c, Double_t *point)
{
// computes next intersection point of curve c with this shape
}
//-----------------------------------------------------------------------------
Double_t TGeoConeSeg::Safety(Double_t *point, Double_t *spoint, Option_t *option)
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return 0.0;
}
//-----------------------------------------------------------------------------
void TGeoConeSeg::SetConsDimensions(Double_t dz, Double_t rmin1, Double_t rmax1,
                   Double_t rmin2, Double_t rmax2, Double_t phi1, Double_t phi2)
{
   fDz   = dz;
   fRmin1 = rmin1;
   fRmax1 = rmax1;
   fRmin2 = rmin2;
   fRmax2 = rmax2;
   fPhi1 = phi1;
   if (fPhi1<0) fPhi1+=360.;
   fPhi2 = phi2;
   if (fPhi2<0) fPhi2+=360.;
}   
//-----------------------------------------------------------------------------
void TGeoConeSeg::SetDimensions(Double_t *param)
{
   Double_t dz    = param[0];
   Double_t rmin1 = param[1];
   Double_t rmax1 = param[2];
   Double_t rmin2 = param[3];
   Double_t rmax2 = param[4];
   Double_t phi1  = param[5];
   Double_t phi2  = param[6];
   SetConsDimensions(dz, rmin1, rmax1,rmin2, rmax2, phi1, phi2);
}   
//-----------------------------------------------------------------------------
void TGeoConeSeg::SetPoints(Double_t *buff) const
{
// create cone segment mesh points
    Int_t j, n;
    Float_t dphi,phi,phi1, phi2,dz;

    n = TGeoManager::kGeoDefaultNsegments+1;
    dz    = fDz;
    phi1 = fPhi1;
    phi2 = fPhi2;
    if (phi2<phi1) phi2+=360.;

    dphi = (phi2-phi1)/(n-1);

    Int_t indx = 0;

    if (buff) {
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*kDegRad;
            buff[indx++] = fRmin1 * TMath::Cos(phi);
            buff[indx++] = fRmin1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*kDegRad;
            buff[indx++] = fRmax1 * TMath::Cos(phi);
            buff[indx++] = fRmax1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*kDegRad;
            buff[indx++] = fRmin2 * TMath::Cos(phi);
            buff[indx++] = fRmin2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*kDegRad;
            buff[indx++] = fRmax2 * TMath::Cos(phi);
            buff[indx++] = fRmax2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoConeSeg::SetPoints(Float_t *buff) const
{
// create cone segment mesh points
    Int_t j, n;
    Float_t dphi,phi,phi1, phi2,dz;

    n = TGeoManager::kGeoDefaultNsegments+1;
    dz    = fDz;
    phi1 = fPhi1;
    phi2 = fPhi2;
    if (phi2<phi1) phi2+=360.;

    dphi = (phi2-phi1)/(n-1);

    Int_t indx = 0;

    if (buff) {
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*kDegRad;
            buff[indx++] = fRmin1 * TMath::Cos(phi);
            buff[indx++] = fRmin1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*kDegRad;
            buff[indx++] = fRmax1 * TMath::Cos(phi);
            buff[indx++] = fRmax1 * TMath::Sin(phi);
            buff[indx++] = -dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*kDegRad;
            buff[indx++] = fRmin2 * TMath::Cos(phi);
            buff[indx++] = fRmin2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
        for (j = 0; j < n; j++) {
            phi = (fPhi1+j*dphi)*kDegRad;
            buff[indx++] = fRmax2 * TMath::Cos(phi);
            buff[indx++] = fRmax2 * TMath::Sin(phi);
            buff[indx++] = dz;
        }
    }
}
//-----------------------------------------------------------------------------
void TGeoConeSeg::Sizeof3D() const
{
// fill size of this 3-D object
    Int_t n = TGeoManager::kGeoDefaultNsegments+1;

    gSize3D.numPoints += n*4;
    gSize3D.numSegs   += n*8;
    gSize3D.numPolys  += n*4-2;
}
