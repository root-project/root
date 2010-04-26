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
void TEveCalo2DGL::MakeRPhiCell(Float_t phiMin, Float_t phiMax,
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

   UInt_t nPhi = data->GetPhiBins()->GetNbins();

   for(UInt_t phiBin = 0; phiBin < nPhi; ++phiBin)
   {
      if (cellLists[phiBin] )
      {
         // reset values
         Float_t off = 0;
         for (Int_t s=0; s<nSlices; ++s)
            sliceVal[s] = 0;

         // sum eta cells
         TEveCaloData::vCellId_t* cids = cellLists[phiBin];
         for (TEveCaloData::vCellId_i it = cids->begin(); it != cids->end(); it++)
         {
            data->GetCellData(*it, cellData);
            sliceVal[(*it).fSlice] += cellData.Value(fM->fPlotEt);
         }

         if (rnrCtx.SecSelection()) {
            glLoadName(phiBin); // set name-stack phi bin
            glPushName(0);
         }
         for (Int_t s = 0; s < nSlices; ++s)
         {
            if (rnrCtx.SecSelection())  glLoadName(s); // set name-stack slice
            fM->SetupColorHeight(sliceVal[s], s, towerH);
            MakeRPhiCell(cellData.PhiMin(), cellData.PhiMax(), towerH, off);
            off += towerH;
         }
         if (rnrCtx.SecSelection()) glPopName(); // slice
      }
   }

   delete [] sliceVal;
}

//______________________________________________________________________________
void TEveCalo2DGL::DrawRPhiHighlighted(std::vector<TEveCaloData::vCellId_t*>& cellLists) const
{
   // Draw selected calorimeter cells in RPhi projection.

   static const TEveException eh("TEveCalo2DGL::DrawRPhiHighlighted ");

   TEveCaloData* data = fM->fData;
   TEveCaloData::CellData_t cellData;
   Int_t  nSlices  = data->GetNSlices();
   UInt_t nPhiBins = data->GetPhiBins()->GetNbins();
   Float_t *sliceVal    = new Float_t[nSlices];
   Float_t *sliceValRef = new Float_t[nSlices];
   Float_t  towerH, towerHRef;

   for(UInt_t phiBin = 1; phiBin <= nPhiBins; ++phiBin)
   {
      if (cellLists[phiBin])
      {
         if (!fM->fCellLists[phiBin])
            throw eh + "selected cell not in cell list cache.";

         Float_t off = 0;
         // selected eta sum
         for (Int_t s=0; s<nSlices; ++s) sliceVal[s] = 0;
         TEveCaloData::vCellId_t& cids = *(cellLists[phiBin]);
         for (TEveCaloData::vCellId_i i=cids.begin(); i!=cids.end(); i++) {
            data->GetCellData((*i), cellData);
            sliceVal[i->fSlice] += cellData.Value(fM->fPlotEt);
         }
         // referenced eta sum
         for (Int_t s=0; s<nSlices; ++s) sliceValRef[s] = 0;
         TEveCaloData::vCellId_t& cidsRef = *(fM->fCellLists[phiBin]);
         for (TEveCaloData::vCellId_i i=cidsRef.begin(); i!=cidsRef.end(); i++) {
            data->GetCellData(*i, cellData);
            sliceValRef[i->fSlice] += cellData.Value(fM->fPlotEt);
         }
         // draw
         for (Int_t s = 0; s < nSlices; ++s)  {
            fM->SetupColorHeight(sliceValRef[s], s, towerHRef);
            if (sliceVal[s] > 0)
            {
               fM->SetupColorHeight(sliceVal[s], s, towerH);
               MakeRPhiCell(cellData.PhiMin(), cellData.PhiMax(), towerH, off);
            }
            off += towerHRef;
         }
      }
   }

   delete [] sliceVal;
   delete [] sliceValRef;
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
}

//______________________________________________________________________________
void TEveCalo2DGL::DrawRhoZ(TGLRnrCtx & rnrCtx, TEveCalo2D::vBinCells_t& cellLists) const
{
   // Draw calorimeter in RhoZ projection.

   TEveCaloData* data = fM->GetData();
   Int_t nSlices = data->GetNSlices();

   TEveCaloData::CellData_t cellData;
   Float_t *sliceValsUp  = new Float_t[nSlices];
   Float_t *sliceValsLow = new Float_t[nSlices];
   Bool_t   isBarrel;
   Float_t  towerH;

   UInt_t nEta = data->GetEtaBins()->GetNbins();
   for (UInt_t etaBin = 1; etaBin <= nEta; ++etaBin)
   {
      if (cellLists[etaBin] )
      {
         assert(fM->fCellLists[etaBin]);
         // clear
         Float_t offUp  = 0;
         Float_t offLow = 0;
         for (Int_t s = 0; s < nSlices; ++s) {
            sliceValsUp [s] = 0;
            sliceValsLow[s] = 0;
         }
         // values
         TEveCaloData::vCellId_t* cids = cellLists[etaBin];
         for (TEveCaloData::vCellId_i it = cids->begin(); it != cids->end(); ++it)
         {
            data->GetCellData(*it, cellData);
            if (cellData.Phi() > 0)
               sliceValsUp [it->fSlice] += cellData.Value(fM->fPlotEt);
            else
               sliceValsLow[it->fSlice] += cellData.Value(fM->fPlotEt);
         }
         isBarrel = TMath::Abs(cellData.EtaMax()) < fM->GetTransitionEta();

         // draw
         if (rnrCtx.SecSelection()) glLoadName(etaBin); // name-stack eta bin
         if (rnrCtx.SecSelection()) glPushName(0);

         for (Int_t s = 0; s < nSlices; ++s)
         {
            if (rnrCtx.SecSelection()) glLoadName(s);  // name-stack slice
            if (rnrCtx.SecSelection()) glPushName(0);
            //  phi +
            if (sliceValsUp[s])
            {
               if (rnrCtx.SecSelection()) glLoadName(1);  // name-stack phi sign
               fM->SetupColorHeight(sliceValsUp[s], s, towerH);
               MakeRhoZCell(cellData.ThetaMin(), cellData.ThetaMax(), offUp, isBarrel, kTRUE , towerH);
               offUp += towerH;
            }
            // phi -
            if (sliceValsLow[s])
            {
               if (rnrCtx.SecSelection()) glLoadName(0);  // name-stack phi sign
               fM->SetupColorHeight(sliceValsLow[s], s, towerH);
               MakeRhoZCell(cellData.ThetaMin(), cellData.ThetaMax(), offLow, isBarrel, kFALSE , towerH);
               offLow += towerH;
            }
            if (rnrCtx.SecSelection())  glPopName(); // phi sign is pos
         }
         //
         if (rnrCtx.SecSelection())  glPopName(); // slice
      }
   }

   delete [] sliceValsUp;
   delete [] sliceValsLow;
}

//______________________________________________________________________________
void TEveCalo2DGL::DrawRhoZHighlighted(std::vector<TEveCaloData::vCellId_t*>& cellLists) const
{
   // Draw selected calorimeter cells in RhoZ projection.

   static const TEveException eh("TEveCalo2DGL::DrawRhoZHighlighted ");

   TEveCaloData* data = fM->GetData();
   Int_t  nSlices     = data->GetNSlices();
   UInt_t nEtaBins    = data->GetEtaBins()->GetNbins();

   Float_t *sliceValsUp     = new Float_t[nSlices];
   Float_t *sliceValsLow    = new Float_t[nSlices];
   Float_t *sliceValsUpRef  = new Float_t[nSlices];
   Float_t *sliceValsLowRef = new Float_t[nSlices];

   Bool_t   isBarrel;
   Float_t  towerH, towerHRef, offUp, offLow;
   TEveCaloData::CellData_t cellData;

   for (UInt_t etaBin = 1; etaBin <= nEtaBins; ++etaBin)
   {
      if (cellLists[etaBin])
      {
         if (!fM->fCellLists[etaBin])
            throw(eh + "selected cell not in cell list cache.");

         offUp = 0; offLow =0;
         // selected phi sum
         for (Int_t s = 0; s < nSlices; ++s) {
            sliceValsUp[s] = 0; sliceValsLow[s] = 0;
         }
         TEveCaloData::vCellId_t& cids = *(cellLists[etaBin]);
         for (TEveCaloData::vCellId_i i=cids.begin(); i!=cids.end(); i++) {
            data->GetCellData(*i, cellData);
            if (cellData.Phi() > 0)
               sliceValsUp [i->fSlice] += cellData.Value(fM->fPlotEt);
            else
               sliceValsLow[i->fSlice] += cellData.Value(fM->fPlotEt);
         }

         // reference phi sum
         for (Int_t s = 0; s < nSlices; ++s) {
            sliceValsUpRef[s] = 0; sliceValsLowRef[s] = 0;
         }
         TEveCaloData::vCellId_t& cidsRef = *(fM->fCellLists[etaBin]);
         for (TEveCaloData::vCellId_i i=cidsRef.begin(); i!=cidsRef.end(); i++) {
            data->GetCellData(*i, cellData);
            if (cellData.Phi() > 0)
               sliceValsUpRef [i->fSlice] += cellData.Value(fM->fPlotEt);
            else
               sliceValsLowRef[i->fSlice] += cellData.Value(fM->fPlotEt);
         }

         isBarrel = TMath::Abs(cellData.EtaMax()) < fM->GetTransitionEta();
         for (Int_t s = 0; s < nSlices; ++s)
         {
            //  phi +
            fM->SetupColorHeight(sliceValsUpRef[s], s, towerHRef);
            if (sliceValsUp[s] > 0) {
               fM->SetupColorHeight(sliceValsUp[s], s, towerH);
               MakeRhoZCell(cellData.ThetaMin(), cellData.ThetaMax(), offUp, isBarrel, kTRUE , towerH);
            }
            offUp += towerHRef;

            // phi -
            fM->SetupColorHeight(sliceValsLowRef[s], s, towerHRef);
            if (sliceValsLow[s] > 0) {
               fM->SetupColorHeight(sliceValsLow[s], s, towerH);
               MakeRhoZCell(cellData.ThetaMin(), cellData.ThetaMax(), offLow, isBarrel, kFALSE , towerH);
            }
            offLow += towerHRef;
         } // slices
      } // if eta bin
   } //eta bin

   delete [] sliceValsUp;
   delete [] sliceValsLow;
   delete [] sliceValsUpRef;
   delete [] sliceValsLowRef;
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

   if (rnrCtx.SecSelection()) glPushName(0);
   if (IsRPhi())
      DrawRPhi(rnrCtx, fM->fCellLists);
   else
      DrawRhoZ(rnrCtx, fM->fCellLists);
   if (rnrCtx.SecSelection()) glPopName();
}

//______________________________________________________________________________
void TEveCalo2DGL::DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* /*pshp*/, Int_t /*lvl*/) const
{
   // Draw towers in highlight mode.

   static const TEveException eh("TEveCalo2DGL::DrawHighlight ");

   if (fM->fData->GetCellsSelected().empty() && fM->fData->GetCellsHighlighted().empty())
   {
      return;
   }

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT |GL_POLYGON_BIT );
   glDisable(GL_LIGHTING);
   glDisable(GL_CULL_FACE);
   glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

   TGLUtil::LineWidth(2);
   TGLUtil::LockColor();
   try {
   
      if (!fM->fData->GetCellsHighlighted().empty()) 
      {
         glColor4ubv(rnrCtx.ColorSet().Selection(3).CArr());

         if (IsRPhi())
            DrawRPhiHighlighted(fM->fCellListsHighlighted);
         else
            DrawRhoZHighlighted(fM->fCellListsHighlighted);

      }
      if (!fM->fData->GetCellsSelected().empty())
      {
         Float_t dr[2];
         glGetFloatv(GL_DEPTH_RANGE,dr);
         glColor4ubv(rnrCtx.ColorSet().Selection(1).CArr());
         glDepthRange(dr[0], 0.8*dr[1]);
         if (IsRPhi())
            DrawRPhiHighlighted(fM->fCellListsSelected);
         else
            DrawRhoZHighlighted(fM->fCellListsSelected);

         glDepthRange(dr[0], dr[1]);
      }
   }
   catch (TEveException& exc)
   {
      Warning(eh, exc);
   }

   TGLUtil::UnlockColor();
   glPopAttrib();


  
}

 //______________________________________________________________________________
void TEveCalo2DGL::ProcessSelection(TGLRnrCtx & /*rnrCtx*/, TGLSelectRecord & rec)
{
   // Processes tower selection in eta bin or phi bin.
   // Virtual function from TGLogicalShape. Called from TGLViewer.

   TEveCaloData::vCellId_t& cells = rec.GetHighlight() ? fM->fData->GetCellsHighlighted() : fM->fData->GetCellsSelected() ;
   Int_t prev = cells.size();
   if (!rec.GetMultiple()) cells.clear();

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

            fM->fData->GetCellData(*it, cellData);
            if (!IsRPhi())
            {
               if ((rec.GetItem(3) && cellData.Phi() > 0) || (rec.GetItem(3) == kFALSE && cellData.Phi() < 0)) {
                  cells.push_back(*it);
               }
            }
            else {
               cells.push_back(*it);
            }
         }
      }
   }

   // set secondary selection result
   if (prev == 0 && binID >= 0)
      rec.SetSecSelResult(TGLSelectRecord::kEnteringSelection);
   else if (prev  && binID < 0)
      rec.SetSecSelResult(TGLSelectRecord::kLeavingSelection);
   else if (prev  && binID >= 0)
      rec.SetSecSelResult(TGLSelectRecord::kModifyingInternalSelection);

   fM->fData->CellSelectionChanged();
}
