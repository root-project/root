// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveCalo2DGL.h"
#include "TEveCalo.h"
#include "TEveProjections.h"
#include "TEveProjectionManager.h"
#include "TEveRGBAPalette.h"

#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLIncludes.h"
#include "TGLUtil.h"
#include "TAxis.h"


//______________________________________________________________________________
// OpenGL renderer class for TEveCalo2D.
//

ClassImp(TEveCalo2DGL);

//______________________________________________________________________________
TEveCalo2DGL::TEveCalo2DGL() :
   TGLObject(),
   fM(0)
{
   // Constructor.

   // fDLCache = kFALSE; // Disable display list.
   fMultiColor = kTRUE;
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveCalo2DGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   if (SetModelCheckClass(obj, TEveCalo2D::Class())) {
      fM = dynamic_cast<TEveCalo2D*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveCalo2DGL::SetBBox()
{
   // Set bounding box.

   SetAxisAlignedBBox(((TEveCalo2D*)fExternalObj)->AssertBBox());
}

/******************************************************************************/

//______________________________________________________________________________
Float_t TEveCalo2DGL::MakeRPhiCell(Float_t phiMin, Float_t phiMax,
                                   Float_t towerH, Float_t offset) const
{
   // Calculate vertices for the calorimeter cell in RPhi projection.
   // Returns outside radius of the tower.

   using namespace TMath;

   Float_t r1 = fM->fBarrelRadius + offset;
   Float_t r2 = r1 + towerH;

   Float_t pnts[8];

   pnts[0] = r1*Cos(phiMin); pnts[1] = r1*Sin(phiMin);
   pnts[2] = r2*Cos(phiMin); pnts[3] = r2*Sin(phiMin);
   pnts[4] = r2*Cos(phiMax); pnts[5] = r2*Sin(phiMax);
   pnts[6] = r1*Cos(phiMax); pnts[7] = r1*Sin(phiMax);

   Float_t x, y, z;
   glBegin(GL_QUADS);
   for (Int_t i = 0; i < 4; ++i)
   {
      x = pnts[2*i];
      y = pnts[2*i+1];
      z = 0.f;
      fM->fManager->GetProjection()->ProjectPoint(x, y, z);
      glVertex3f(x, y, fM->fDepth);
   }
   glEnd();
   return offset + towerH;
}

//______________________________________________________________________________
void TEveCalo2DGL::DrawRPhi(TGLRnrCtx & rnrCtx) const
{
   // Draw calorimeter cells in RPhi projection.

   TEveCaloData* data = fM->GetData();
   Int_t    nSlices  = data->GetNSlices();
   Float_t *sliceVal = new Float_t[nSlices];
   TEveCaloData::CellData_t cellData;
   Float_t towerH;
   Float_t phiMin, phiMax;
   TAxis* ax = fM->fData->GetPhiBins();

   if (rnrCtx.SecSelection()) glPushName(0);

   for(UInt_t vi = 0; vi < fM->fCellLists.size(); ++vi)
   {
      // reset values
      Float_t off = 0;
      for (Int_t s=0; s<nSlices; ++s)
         sliceVal[s] = 0;

      // loop through eta bins
      phiMin = 0;
      phiMax =0;


      TEveCaloData::vCellId_t* cids = fM->fCellLists[vi];
      data->GetCellData(cids->front(), cellData);

      phiMin = cellData.PhiMin();
      phiMax = cellData.PhiMax();

      for (TEveCaloData::vCellId_i it = cids->begin(); it != cids->end(); it++)
      {

         data->GetCellData(*it, cellData);
         sliceVal[(*it).fSlice] += cellData.Value(fM->fPlotEt);
         if(phiMin>cellData.PhiMin()) phiMin=cellData.PhiMin();
         if(phiMax<cellData.PhiMax()) phiMax=cellData.PhiMax();
      }

      // draw
      if (rnrCtx.SecSelection()) {
         glLoadName(vi);
         glPushName(0);
      }

      Int_t bin = fM->fBinIds[vi];
      for (Int_t s = 0; s < nSlices; ++s)
      {
         fM->SetupColorHeight(sliceVal[s], s, towerH);
         off = MakeRPhiCell(ax->GetBinLowEdge(bin), ax->GetBinUpEdge(bin), towerH, off);
      }
      if (rnrCtx.SecSelection()) glPopName(); // slice
   }

   delete [] sliceVal;
}


/*******************************************************************************/
/*******************************************************************************/

//______________________________________________________________________________
void TEveCalo2DGL::MakeRhoZCell(Float_t thetaMin, Float_t thetaMax,
                                Float_t& offset, Bool_t isBarrel,  Bool_t phiPlus, Float_t towerH) const
{
   // Draw cell in RhoZ projection.

   using namespace TMath;

   Float_t pnts[8];

   Float_t sin1 = Sin(thetaMin);
   Float_t cos1 = Cos(thetaMin);
   Float_t sin2 = Sin(thetaMax);
   Float_t cos2 = Cos(thetaMax);

   if (isBarrel)
   {
      Float_t r1 = fM->fBarrelRadius/Abs(Sin(0.5f*(thetaMin+thetaMax))) + offset;
      Float_t r2 = r1 + towerH;

      pnts[0] = r1*sin1; pnts[1] = r1*cos1;
      pnts[2] = r2*sin1; pnts[3] = r2*cos1;
      pnts[4] = r2*sin2; pnts[5] = r2*cos2;
      pnts[6] = r1*sin2; pnts[7] = r1*cos2;
   }
   else
   {
      // endcap
      Float_t r1 = fM->GetEndCapPos()/Abs(Cos(0.5f*(thetaMin+thetaMax))) + offset;
      Float_t r2 = r1 + towerH;

      pnts[0] = r1*sin1; pnts[1] = r1*cos1;
      pnts[2] = r2*sin1; pnts[3] = r2*cos1;
      pnts[4] = r2*sin2; pnts[5] = r2*cos2;
      pnts[6] = r1*sin2; pnts[7] = r1*cos2;
   }

   glPushName(phiPlus);
   glBegin(GL_QUADS);
   Float_t x, y, z;
   for (Int_t i = 0; i < 4; ++i)
   {
      x = 0.f;
      y = phiPlus ? Abs(pnts[2*i]) : -Abs(pnts[2*i]);
      z = pnts[2*i+1];
      fM->fManager->GetProjection()->ProjectPoint(x, y, z);
      glVertex3f(x, y, fM->fDepth);
   }
   glEnd();
   glPopName();

   offset += towerH;
}

//______________________________________________________________________________
void TEveCalo2DGL::DrawRhoZ(TGLRnrCtx & rnrCtx) const
{
   // Draw calorimeter in RhoZ projection.

   TEveCaloData::CellData_t cellData;
   Float_t towerH;
   TEveCaloData* data = fM->GetData();
   Int_t nSlices = data->GetNSlices();

   Float_t *sliceValsUp  = new Float_t[nSlices];
   Float_t *sliceValsLow = new Float_t[nSlices];
   Float_t  thetaMin, thetaMax;
   Bool_t   isBarrel;
   Int_t bin = 0;

   TAxis* ax = fM->fData->GetEtaBins();

   if (rnrCtx.SecSelection()) glPushName(0);

   for (UInt_t vi = 0; vi < fM->fCellLists.size(); ++vi)
   {
      // clear
      Float_t offUp  = 0;
      Float_t offLow = 0;
      for (Int_t s = 0; s < nSlices; ++s) {
         sliceValsUp [s] = 0;
         sliceValsLow[s] = 0;
      }

      // values
      for (TEveCaloData::vCellId_i it = fM->fCellLists[vi]->begin();
           it != fM->fCellLists[vi]->end(); ++it)
      {
         data->GetCellData(*it, cellData);
         if (cellData.Phi() > 0)
            sliceValsUp [it->fSlice] += cellData.Value(fM->fPlotEt);
         else
            sliceValsLow[it->fSlice] += cellData.Value(fM->fPlotEt);
      }

      // draw
      if (rnrCtx.SecSelection())
      {
         glLoadName(vi); // phi bin
         glPushName(0);  // slice
      }

      bin = fM->fBinIds[vi];
      isBarrel = TMath::Abs(ax->GetBinUpEdge(bin)) < fM->GetTransitionEta();
      thetaMin = TEveCaloData::EtaToTheta(ax->GetBinUpEdge(bin));
      thetaMax = TEveCaloData::EtaToTheta(ax->GetBinLowEdge(bin));

      for (Int_t s = 0; s < nSlices; ++s)
      {
         if (rnrCtx.SecSelection()) glLoadName(s);
         //  phi +
         fM->SetupColorHeight(sliceValsUp[s], s, towerH);
         MakeRhoZCell(thetaMin, thetaMax, offUp, isBarrel, kTRUE , towerH);

         // phi -
         fM->SetupColorHeight(sliceValsLow[s], s, towerH);
         MakeRhoZCell(thetaMin, thetaMax, offLow, isBarrel, kFALSE , towerH);
      }

      if (rnrCtx.SecSelection()) glPopName(); // slice
   }

   if (rnrCtx.SecSelection()) glPopName(); // phi bin

   delete [] sliceValsUp;
   delete [] sliceValsLow;
}

//______________________________________________________________________________
void TEveCalo2DGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Render with OpenGL.

   TGLCapabilitySwitch light_off(GL_LIGHTING,  kFALSE);
   TGLCapabilitySwitch cull_off (GL_CULL_FACE, kFALSE);

   if (fM->fCellIdCacheOK == kFALSE)
      fM->BuildCellIdCache();

   fM->AssertPalette();

   TEveProjection::EPType_e pt = fM->fManager->GetProjection()->GetType();
   if (pt == TEveProjection::kPT_RhoZ)
      DrawRhoZ(rnrCtx);
   else if (pt == TEveProjection::kPT_RPhi)
      DrawRPhi(rnrCtx);
}

//______________________________________________________________________________
void TEveCalo2DGL::ProcessSelection(TGLRnrCtx & /*rnrCtx*/, TGLSelectRecord & rec)
{
   // Processes secondary selection from TGLViewer.

   if (rec.GetN() < 2) return;

   Int_t id = rec.GetItem(1);
   Int_t slice = rec.GetItem(2);
   TEveCaloData::CellData_t cellData;

   Int_t n = 0;
   for (TEveCaloData::vCellId_i it =fM->fCellLists[id]->begin(); it!=fM->fCellLists[id]->end(); it++)
   {
      if ((*it).fSlice == slice)
         n++;
   }

   printf("Tower selected in slice %d number of hits: %2d \n", slice, n);
   for (TEveCaloData::vCellId_i it =fM->fCellLists[id]->begin(); it!=fM->fCellLists[id]->end(); it++)
   {
      if ((*it).fSlice == slice)
      {
         fM->fData->GetCellData(*it, cellData);
         cellData.Dump();

      }
   }

   // rho Z
   if (rec.GetN() == 4)
   {
      if(rec.GetItem(3))
         printf("Cell in selected positive phi half \n");
      else
         printf("Cell in selected negative phi half \n");

      for (TEveCaloData::vCellId_i it = fM->fCellLists[id]->begin();
           it != fM->fCellLists[id]->end(); ++it)
      {
         fM->fData->GetCellData(*it, cellData);
         if ((*it).fSlice == slice)
         {
            if ((rec.GetItem(3) && cellData.Phi() > 0) ||
                (rec.GetItem(3) == kFALSE && cellData.Phi() < 0))
            {
               cellData.Dump();
            }
         }
      }
   }
}
