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
#include "TGLPhysicalShape.h"
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
Bool_t TEveCalo2DGL::IsRPhi() const
{
   // Is current projection type RPhi

   return fM->fManager->GetProjection()->GetType() == TEveProjection::kPT_RPhi;
}

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
      fM->fManager->GetProjection()->ProjectPoint(x, y, z, fM->fDepth);
      glVertex3f(x, y, z);
   }
   glEnd();
   return offset + towerH;
}

//______________________________________________________________________________
void TEveCalo2DGL::DrawRPhi(TGLRnrCtx & rnrCtx, TEveCalo2D::vBinCells_t& cellLists) const
{
   // Draw calorimeter cells in RPhi projection.

   TEveCaloData* data = fM->GetData();
   Int_t    nSlices  = data->GetNSlices();
   Float_t *sliceVal = new Float_t[nSlices];
   TEveCaloData::CellData_t cellData;
   Float_t towerH;
   Float_t phiMin, phiMax;

   if (rnrCtx.SecSelection()) glPushName(0);
   for(UInt_t vi = 0; vi < cellLists.size(); ++vi)
   {
      if (cellLists[vi] )
      {
         // reset values
         Float_t off = 0;
         for (Int_t s=0; s<nSlices; ++s)
            sliceVal[s] = 0;

         // loop through eta bins
         phiMin = 0;
         phiMax =0;

         TEveCaloData::vCellId_t* cids = cellLists[vi];
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

         if (rnrCtx.SecSelection()) {
            glLoadName(vi); // phi stack
            glPushName(0);
         }

         for (Int_t s = 0; s < nSlices; ++s)
         {
            if (rnrCtx.SecSelection())  glLoadName(s); // name stack
            fM->SetupColorHeight(sliceVal[s], s, towerH);
            off = MakeRPhiCell(phiMin, phiMax, towerH, off);
         }
         if (rnrCtx.SecSelection()) glPopName(); // slice
      }
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
      fM->fManager->GetProjection()->ProjectPoint(x, y, z, fM->fDepth);
      glVertex3f(x, y, z);
   }
   glEnd();
   glPopName();

   offset += towerH;
}

//______________________________________________________________________________
void TEveCalo2DGL::DrawRhoZ(TGLRnrCtx & rnrCtx, TEveCalo2D::vBinCells_t& cellLists) const
{
   // Draw calorimeter in RhoZ projection.

   if (rnrCtx.SecSelection()) glPushName(0);

   TEveCaloData* data = fM->GetData();
   Int_t nSlices = data->GetNSlices();

   TEveCaloData::CellData_t cellData;
   Float_t *sliceValsUp  = new Float_t[nSlices];
   Float_t *sliceValsLow = new Float_t[nSlices];
   Bool_t   isBarrel;
   Float_t  towerH;

   for (UInt_t vi = 0; vi < cellLists.size(); ++vi)
   {
      if (cellLists[vi] )
      {
         // clear
         Float_t offUp  = 0;
         Float_t offLow = 0;
         for (Int_t s = 0; s < nSlices; ++s) {
            sliceValsUp [s] = 0;
            sliceValsLow[s] = 0;
         }
         // values
         for (TEveCaloData::vCellId_i it = cellLists[vi]->begin();
              it != cellLists[vi]->end(); ++it)
         {
            data->GetCellData(*it, cellData);

            if (cellData.Phi() > 0)
               sliceValsUp [it->fSlice] += cellData.Value(fM->fPlotEt);
            else
            {
               sliceValsLow[it->fSlice] += cellData.Value(fM->fPlotEt);
            }
         }

         // draw
         if (rnrCtx.SecSelection())
         {
            glLoadName(vi); // phi bin
            glPushName(0);  // slice
         }

         isBarrel = TMath::Abs(cellData.EtaMax()) < fM->GetTransitionEta();
         for (Int_t s = 0; s < nSlices; ++s)
         {
            if (rnrCtx.SecSelection()) glLoadName(s);

            //  phi +
            if (sliceValsUp[s])
            {
               fM->SetupColorHeight(sliceValsUp[s], s, towerH);
               MakeRhoZCell(cellData.ThetaMin(), cellData.ThetaMax(), offUp, isBarrel, kTRUE , towerH);
            }

            // phi -
            if (sliceValsLow[s])
            {
               fM->SetupColorHeight(sliceValsLow[s], s, towerH);
               MakeRhoZCell(cellData.ThetaMin(), cellData.ThetaMax(), offLow, isBarrel, kFALSE , towerH);
            }
         }

         if (rnrCtx.SecSelection()) glPopName(); // slice
      }
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

   if (IsRPhi())
      DrawRPhi(rnrCtx, fM->fCellLists);
   else
      DrawRhoZ(rnrCtx, fM->fCellLists);
}

//______________________________________________________________________________
void TEveCalo2DGL::DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp) const
{
   // Draw towers in highlight mode.

   if ((pshp->GetSelected() == 2) && fM->fData->GetCellsSelected().size())
   {
      glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT  | GL_LINE_BIT );
      glDisable(GL_CULL_FACE);
      glDisable(GL_LIGHTING);
      glEnable(GL_LINE_SMOOTH);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      TGLUtil::LineWidth(2);

      glColor4ubv(rnrCtx.ColorSet().Selection(pshp->GetSelected()).CArr());
      TGLUtil::LockColor();

      if (IsRPhi())
         DrawRPhi(rnrCtx, fM->fCellListsSelected);
      else
         DrawRhoZ(rnrCtx, fM->fCellListsSelected);

      TGLUtil::UnlockColor();

      glPopAttrib();
   }
}

 //______________________________________________________________________________
void TEveCalo2DGL::ProcessSelection(TGLRnrCtx & /*rnrCtx*/, TGLSelectRecord & rec)
{
   // Processes tower selection in eta bin or phi bin.
   // Virtual function from TGLogicalShape. Called from TGLViewer.


   Int_t prev = fM->fData->GetCellsSelected().size();
   if (!rec.GetMultiple()) fM->fData->GetCellsSelected().clear();

   Int_t binID = -1;
   if (rec.GetN() > 2)
   {
      binID       = rec.GetItem(1);
      Int_t slice = rec.GetItem(2);
      TEveCaloData::CellData_t cellData;
      for (TEveCaloData::vCellId_i it = fM->fCellLists[binID]->begin();
           it!=fM->fCellLists[binID]->end(); it++)
      {
         if ((*it).fSlice == slice)
         {
            if (!IsRPhi())
            {
               fM->fData->GetCellData(*it, cellData);
               if ((rec.GetItem(3) && cellData.Phi() > 0) || (rec.GetItem(3) == kFALSE && cellData.Phi() < 0))
               {
                  fM->fData->GetCellsSelected().push_back(*it);
               }
            }
            else
               fM->fData->GetCellsSelected().push_back(*it);
         }
      }
   }

   if (prev == 0 && binID >= 0)
      rec.SetSecSelResult(TGLSelectRecord::kEnteringSelection);
   else if (prev  && binID < 0)
      rec.SetSecSelResult(TGLSelectRecord::kLeavingSelection);
   else if (prev  && binID >= 0)
      rec.SetSecSelResult(TGLSelectRecord::kModifyingInternalSelection);

   fM->fData->DataChanged();
}
