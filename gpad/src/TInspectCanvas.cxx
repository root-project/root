// @(#)root/gpad:$Name:  $:$Id: TInspectCanvas.cxx,v 1.3 2000/09/08 07:41:00 brun Exp $
// Author: Rene Brun   08/01/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TInspectCanvas.h"
#include "TButton.h"
#include "TClass.h"
#include "TLine.h"
#include "TLink.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TRealData.h"
#include "TLatex.h"

ClassImp(TInspectCanvas)

//______________________________________________________________________________//*-*
//*-*   A InspectCanvas is a canvas specialized to inspect Root objects.
//
//Begin_Html
/*
<img src="gif/InspectCanvas.gif">
*/
//End_Html
//


//______________________________________________________________________________
TInspectCanvas::TInspectCanvas() : TCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*-*InspectCanvas default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

   fBackward   = 0;
   fForward    = 0;
   fCurObject  = 0;
   fObjects    = 0;
}

//_____________________________________________________________________________
TInspectCanvas::TInspectCanvas(UInt_t ww, UInt_t wh)
            : TCanvas("inspect","ROOT Object Inspector",ww,wh)
{
//*-*-*-*-*-*-*-*-*-*-*-*InspectCanvas constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   fBackward   = 0;
   fForward    = 0;
   fCurObject  = 0;
   fObjects    = new TList;
}

//______________________________________________________________________________
TInspectCanvas::~TInspectCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*InspectCanvas default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================

 //  delete fBackward;
 //  delete fForward;
   delete fObjects;
}

//______________________________________________________________________________
void TInspectCanvas::InspectObject(TObject *obj)
{
   // Dump contents of obj in a graphics canvas.
   // Same action as TObject::Dump but in a graphical form.
   // In addition pointers to other objects can be followed.
   //
   // The following picture is the Inspect of a histogram object:
   //Begin_Html
   /*
   <img src="gif/hpxinspect.gif">
   */
   //End_Html

   Int_t cdate = 0;
   Int_t ctime = 0;
   UInt_t *cdatime = 0;
   Bool_t isdate = kFALSE;
   const Int_t kname  = 1;
   const Int_t kvalue = 25;
   const Int_t ktitle = 37;
   const Int_t kline  = 255;
   char line[kline];
   char *pname;

   TClass *cl = obj->IsA();
   if (cl == 0) return;
   if (!cl->GetListOfRealData()) cl->BuildRealData();

   // Count number of data members in order to resize the canvas
   TRealData *rd;
   TIter      next(cl->GetListOfRealData());
   Int_t nreal = cl->GetListOfRealData()->GetSize();

   if (nreal == 0) return;

   Int_t nrows = 33;
   if (nreal+7 > nrows) nrows = nreal+7;
   Int_t nh = nrows*15;
   Int_t nw = 700;
   TVirtualPad *canvas = GetVirtCanvas();
   canvas->Clear();                // remove primitives from canvas
   canvas->SetCanvasSize(nw, nh);  // set new size of drawing area
   canvas->Range(0,-3,20,nreal+4);

   Float_t xvalue = 5;
   Float_t xtitle = 8;
   Float_t dy     = 1;
   Float_t ytext  = Float_t(nreal) - 1.5;
   Float_t tsize  = 0.99/ytext;
   if (tsize < 0.02) tsize = 0.02;
   if (tsize > 0.03) tsize = 0.03;

   // Create text objects
   TText tname, tvalue, ttitle;
   TText *tval;
   tname.SetTextFont(61);
   tname.SetTextAngle(0);
   tname.SetTextAlign(12);
   tname.SetTextColor(1);
   tname.SetTextSize(tsize);
   tvalue.SetTextFont(61);
   tvalue.SetTextAngle(0);
   tvalue.SetTextAlign(12);
   tvalue.SetTextColor(1);
   tvalue.SetTextSize(tsize);
   ttitle.SetTextFont(62);
   ttitle.SetTextAngle(0);
   ttitle.SetTextAlign(12);
   ttitle.SetTextColor(1);
   ttitle.SetTextSize(tsize);

   Float_t x1 = 0.2;
   Float_t x2 = 19.8;
   Float_t y1 = -0.5;
   Float_t y2 = Float_t(nreal) - 0.5;
   Float_t y3 = y2 + 1;
   Float_t y4 = y3 + 1.5;
   Float_t db = 25./GetWh();
   Float_t btop = 0.999;
   // Draw buttons
   fBackward = new TButton("backward","TInspectCanvas::GoBackward();",.01,btop-db,.15,btop);
   fBackward->Draw();
   fBackward->SetToolTipText("Inspect previous object");
   fForward  = new TButton("forward", "TInspectCanvas::GoForward();", .21,btop-db,.35,btop);
   fForward->Draw();
   fForward->SetToolTipText("Inspect next object");

   // Draw surrounding box and title areas
   TLine frame;
   frame.SetLineColor(1);
   frame.SetLineStyle(1);
   frame.SetLineWidth(1);
   frame.DrawLine(x1, y1, x2, y1);
   frame.DrawLine(x2, y1, x2, y4);
   frame.DrawLine(x2, y4, x1, y4);
   frame.DrawLine(x1, y4, x1, y1);
   frame.DrawLine(x1, y2, x2, y2);
   frame.DrawLine(x1, y3, x2, y3);
   frame.DrawLine(xvalue, y1, xvalue, y3);
   frame.DrawLine(xtitle, y1, xtitle, y3);
   ttitle.SetTextSize(0.8*tsize);
   ttitle.SetTextAlign(21);
   ttitle.DrawText(0.5*(x1+xvalue), y2+0.1, "Member Name");
   ttitle.DrawText(0.5*(xvalue+xtitle), y2+0.1, "Value");
   ttitle.DrawText(0.5*(xtitle+x2), y2+0.1, "Title");
   ttitle.SetTextSize(1.2*tsize);
   ttitle.SetTextColor(2);
   ttitle.SetTextAlign(11);
   ttitle.DrawText(x1+0.2, y3+0.1, cl->GetName());
   ttitle.SetTextColor(4);
   sprintf(line,"%s:%d",obj->GetName(),obj->GetUniqueID());
   ttitle.DrawText(xvalue+0.2, y3+0.1, line);
   ttitle.SetTextColor(6);
   ttitle.DrawText(xtitle+2, y3+0.1, obj->GetTitle());
   ttitle.SetTextSize(tsize);
   ttitle.SetTextColor(1);
   ttitle.SetTextFont(11);
   ttitle.SetTextAlign(12);

   //---Now loop on data members-----------------------
   // We make 3 passes. Faster than one single pass because changing
   // font parameters is time consuming
   for (Int_t pass = 0; pass < 3; pass++) {
      ytext  = y2 - 0.5;
      next.Reset();
      while ((rd = (TRealData*) next())) {
         TDataMember *member = rd->GetDataMember();
         if (!member) continue;
         TDataType *membertype = member->GetDataType();
         isdate = kFALSE;
         if (strcmp(member->GetName(),"fDatime") == 0 && strcmp(member->GetTypeName(),"UInt_t") == 0) {
            isdate = kTRUE;
         }

         // Encode data member name
         pname = &line[kname];
         for (Int_t i=0;i<kline;i++) line[i] = ' ';
         line[kline-1] = 0;
         sprintf(pname,"%s ",rd->GetName());

         // Encode data value or pointer value
         tval = &tvalue;
         Int_t offset = rd->GetThisOffset();
         char *pointer = (char*)obj + offset;
         char **ppointer = (char**)(pointer);
         TLink *tlink = 0;

         if (member->IsaPointer()) {
            char **p3pointer = (char**)(*ppointer);
            if (!p3pointer) {
               sprintf(&line[kvalue],"->0");
            } else if (!member->IsBasic()) {
               if (pass == 1) tlink = new TLink(xvalue+0.1, ytext, p3pointer);
            } else if (membertype) {
               if (!strcmp(membertype->GetTypeName(), "char"))
                  sprintf(&line[kvalue], "%s", *ppointer);
               else
                  strcpy(&line[kvalue], membertype->AsString(p3pointer));
            } else if (!strcmp(member->GetFullTypeName(), "char*") ||
                     !strcmp(member->GetFullTypeName(), "const char*")) {
               sprintf(&line[kvalue], "%s", *ppointer);
            } else {
               if (pass == 1) tlink = new TLink(xvalue+0.1, ytext, p3pointer);
            }
         } else if (membertype)
            if (isdate) {
               cdatime = (UInt_t*)pointer;
               TDatime::GetDateTime(cdatime[0],cdate,ctime);
               sprintf(&line[kvalue],"%d/%d",cdate,ctime);
            } else {
               strcpy(&line[kvalue], membertype->AsString(pointer));
            }
         else
            sprintf(&line[kvalue],"->%lx ", (Long_t)pointer);

         // Encode data member title
         Int_t ltit = 0;
         if (isdate == kFALSE && strcmp(member->GetFullTypeName(), "char*") &&
             strcmp(member->GetFullTypeName(), "const char*")) {
            Int_t lentit = strlen(member->GetTitle());
            if (lentit >= kline-ktitle) lentit = kline-ktitle-1;
            strncpy(&line[ktitle],member->GetTitle(),lentit);
            line[ktitle+lentit] = 0;
            ltit = ktitle;
         }

         // Ready to draw the name, value and title columns
         if (pass == 0)tname.DrawText( x1+0.1,  ytext, &line[kname]);
         if (pass == 1) {
            if (tlink) {
               tlink->SetTextFont(61);
               tlink->SetTextAngle(0);
               tlink->SetTextAlign(12);
               tlink->SetTextColor(2);
               tlink->SetTextSize(tsize);
               tlink->SetName(member->GetTypeName());
               tlink->SetBit(kCanDelete);
               tlink->Draw();
            } else {
               tval->DrawText(xvalue+0.1, ytext, &line[kvalue]);
            }
         }
         if (pass == 2 && ltit) ttitle.DrawText(xtitle+0.3, ytext, &line[ltit]);
         ytext -= dy;
      }
   }
   Update();
   fCurObject = obj;
}

//______________________________________________________________________________
void TInspectCanvas::GoBackward()
{
// static function , inspect previous object

  TInspectCanvas *inspect = (TInspectCanvas*)(gROOT->GetListOfCanvases())->FindObject("inspect");
  TObject *cur = inspect->GetCurObject();
  TObject *obj = inspect->GetObjects()->Before(cur);
  if (obj)       inspect->InspectObject(obj);
}

//______________________________________________________________________________
void TInspectCanvas::GoForward()
{
// static function , inspect next object

  TInspectCanvas *inspect = (TInspectCanvas*)(gROOT->GetListOfCanvases())->FindObject("inspect");
  TObject *cur = inspect->GetCurObject();
  TObject *obj = inspect->GetObjects()->After(cur);
  if (obj)       inspect->InspectObject(obj);
}

//______________________________________________________________________________
void TInspectCanvas::Inspector(TObject *obj)
{
// static function , interface to InspectObject.
// Create the InspectCanvas if it does not exist yet.
//

  TVirtualPad *padsav = gPad;
  TInspectCanvas *inspect = (TInspectCanvas*)(gROOT->GetListOfCanvases())->FindObject("inspect");
  if (!inspect) inspect = new TInspectCanvas(700,600);
  else          inspect->cd();

  inspect->InspectObject(obj);
  inspect->GetObjects()->Add(obj);
  obj->SetBit(kMustCleanup);

  if (padsav) padsav->cd();
}

//______________________________________________________________________________
void TInspectCanvas::RecursiveRemove(TObject *obj)
{
//*-*-*-*-*-*-*-*Recursively remove object from the list of objects*-*-*-*-*
//*-*            ==================================================

//printf("Revove obj=%x, name=%s\n",obj,obj->GetName());
   fObjects->Remove(obj);
   TPad::RecursiveRemove(obj);
}

