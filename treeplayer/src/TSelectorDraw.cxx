// @(#)root/treeplayer:$Name:  $:$Id: TSelectorDraw.cxx,v 1.61 2006/05/24 15:10:47 brun Exp $
// Author: Rene Brun   08/01/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelectorDraw                                                        //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSelectorDraw.h"
#include "TROOT.h"
#include "TH2.h"
#include "TH3.h"
#include "TView.h"
#include "TGraph.h"
#include "TPolyMarker3D.h"
#include "TDirectory.h"
#include "TVirtualPad.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TTreeFormulaManager.h"
#include "TEnv.h"
#include "TTree.h"
#include "TCut.h"
#include "TEventList.h"
#include "THLimitsFinder.h"
#include "TStyle.h"
#include "TClass.h"

ClassImp(TSelectorDraw)

const Int_t kCustomHistogram = BIT(17);

//______________________________________________________________________________
TSelectorDraw::TSelectorDraw()
{
   // Default selector constructor.

   fTree           = 0;
   fV1             = 0;
   fV2             = 0;
   fV3             = 0;
   fV4             = 0;
   fW              = 0;
   fVar1           = 0;
   fVar2           = 0;
   fVar3           = 0;
   fVar4           = 0;
   fManager        = 0;
   fMultiplicity   = 0;
   fSelect         = 0;
   fSelectedRows   = 0;
   fDraw           = 0;
   fObject         = 0;
   fOldHistogram   = 0;
   fObjEval        = kFALSE;
   fVar1Multiple   = kFALSE;
   fVar2Multiple   = kFALSE;
   fVar3Multiple   = kFALSE;
   fVar4Multiple   = kFALSE;
   fSelectMultiple = kFALSE;
   fCleanElist     = kFALSE;
   fTreeElist      = 0;
   fAction         = 0;
   fNfill          = 0;
   fDimension      = 0;
   fOldEstimate    = 0;
   fForceRead      = 0;
   fWeight         = 1;
   for (Int_t i=0;i<4;i++) {fNbins[i]=0; fVmin[i] = fVmax[i]= 0;}
}

//______________________________________________________________________________
TSelectorDraw::TSelectorDraw(const TSelectorDraw& sd) :
  TSelector(sd),
  fTree(sd.fTree),
  fVar1(sd.fVar1),
  fVar2(sd.fVar2),
  fVar3(sd.fVar3),
  fVar4(sd.fVar4),
  fSelect(sd.fSelect),
  fManager(sd.fManager),
  fObject(sd.fObject),
  fTreeElist(sd.fTreeElist),
  fOldHistogram(sd.fOldHistogram),
  fAction(sd.fAction),
  fDraw(sd.fDraw),
  fNfill(sd.fNfill),
  fMultiplicity(sd.fMultiplicity),
  fDimension(sd.fDimension),
  fSelectedRows(sd.fSelectedRows),
  fOldEstimate(sd.fOldEstimate),
  fForceRead(sd.fForceRead),
  fWeight(sd.fWeight),
  fV1(sd.fV1),
  fV2(sd.fV2),
  fV3(sd.fV3),
  fV4(sd.fV4),
  fW(sd.fW),
  fVar1Multiple(sd.fVar1Multiple),
  fVar2Multiple(sd.fVar2Multiple),
  fVar3Multiple(sd.fVar3Multiple),
  fVar4Multiple(sd.fVar4Multiple),
  fSelectMultiple(sd.fSelectMultiple),
  fCleanElist(sd.fCleanElist),
  fObjEval(sd.fObjEval)
{
   //copy constructor
   for(Int_t i=0; i<4; i++) {
      fNbins[i]=sd.fNbins[i];
      fVmin[i]=sd.fVmin[i];
      fVmax[i]=sd.fVmax[i];
   }
}

//______________________________________________________________________________
TSelectorDraw& TSelectorDraw::operator=(const TSelectorDraw& sd)
{
   //assignement operator
   if(this!=&sd) {
      TSelector::operator=(sd);
      fTree=sd.fTree;
      fVar1=sd.fVar1;
      fVar2=sd.fVar2;
      fVar3=sd.fVar3;
      fVar4=sd.fVar4;
      fSelect=sd.fSelect;
      fManager=sd.fManager;
      fObject=sd.fObject;
      fTreeElist=sd.fTreeElist;
      fOldHistogram=sd.fOldHistogram;
      fAction=sd.fAction;
      fDraw=sd.fDraw;
      fNfill=sd.fNfill;
      fMultiplicity=sd.fMultiplicity;
      fDimension=sd.fDimension;
      fSelectedRows=sd.fSelectedRows;
      fOldEstimate=sd.fOldEstimate;
      fForceRead=sd.fForceRead;
      for(Int_t i=0; i<4; i++) {
         fNbins[i]=sd.fNbins[i];
         fVmin[i]=sd.fVmin[i];
         fVmax[i]=sd.fVmax[i];
      }
      fWeight=sd.fWeight;
      fV1=sd.fV1;
      fV2=sd.fV2;
      fV3=sd.fV3;
      fV4=sd.fV4;
      fW=sd.fW;
      fVar1Multiple=sd.fVar1Multiple;
      fVar2Multiple=sd.fVar2Multiple;
      fVar3Multiple=sd.fVar3Multiple;
      fVar4Multiple=sd.fVar4Multiple;
      fSelectMultiple=sd.fSelectMultiple;
      fCleanElist=sd.fCleanElist;
      fObjEval=sd.fObjEval;
   }
   return *this;
}

//______________________________________________________________________________
TSelectorDraw::~TSelectorDraw()
{
   // Selector destructor.

   ClearFormula();
   if (fV1)    delete [] fV1;
   if (fV2)    delete [] fV2;
   if (fV3)    delete [] fV3;
   if (fV4)    delete [] fV4;
   if (fW)     delete [] fW;
}

//______________________________________________________________________________
void TSelectorDraw::Begin(TTree *tree)
{
   // Called everytime a loop on the tree(s) starts.

   SetStatus(0);
   ResetBit(kCustomHistogram);
   fSelectedRows   = 0;
   fTree = tree;
   fDimension = 0;

   const char *varexp0   = fInput->FindObject("varexp")->GetTitle();
   const char *selection = fInput->FindObject("selection")->GetTitle();
   const char *option    = GetOption();

   TString  opt;
   char *hdefault = (char *)"htemp";
   char *varexp;
   Int_t i,j,hkeep;
   opt = option;
   opt.ToLower();
   fOldHistogram = 0;
   TEventList *elist = 0;
   char htitle[2560]; htitle[0] = '\0';
   Bool_t profile = kFALSE;
   Bool_t optSame = kFALSE;
   if (opt.Contains("same")) {
      optSame = kTRUE;
      opt.ReplaceAll("same","");
   }
   TCut realSelection(selection);
   TEventList *inElist = fTree->GetEventList();
   fCleanElist = kFALSE;
   fTreeElist = inElist;
   if ( inElist && inElist->GetReapplyCut() ) {
      realSelection *= inElist->GetTitle();
   }

   // what each variable should contain:
   //   varexp0   - original expression eg "a:b>>htest"
   //   hname     - name of new or old histogram
   //   hkeep     - flag if to keep new histogram
   //   hnameplus - flag if to add to current histo
   //   i         - length of variable expression stipped of everything after ">>"
   //   varexp    - variable expression stipped of everything after ">>"
   //   fOldHistogram     - pointer to hist hname
   //   elist     - pointer to selection list of hname

   Bool_t canRebin = kTRUE;
   if (optSame) canRebin = kFALSE;

   Int_t nbinsx=0, nbinsy=0, nbinsz=0;
   Double_t xmin=0, xmax=0, ymin=0, ymax=0, zmin=0, zmax=0;

   fObject  = 0;
   char *hname = 0;
   char *hnamealloc = 0;
   i = 0;
   if (varexp0 && strlen(varexp0)) {
      for(UInt_t k=strlen(varexp0)-1;k>0;k--) {
         if (varexp0[k]=='>' && varexp0[k-1]=='>') {
            i = (int)( &(varexp0[k-1]) - varexp0 );  //  length of varexp0 before ">>"
            hnamealloc = new char[strlen(&(varexp0[k+1]))+1];
            hname = hnamealloc;
            strcpy(hname,&(varexp0[k+1]));
            break;
         }
      }
   }
   //   char *hname = (char*)strstr(varexp0,">>");
   if (hname) {
      hkeep  = 1;
      varexp = new char[i+1];
      varexp[0] = 0; //necessary if i=0
      Bool_t hnameplus = kFALSE;
      while (*hname == ' ') hname++;
      if (*hname == '+') {
         hnameplus = kTRUE;
         hname++;
         while (*hname == ' ') hname++; //skip ' '
      }
      j = strlen(hname) - 1;   // skip ' '  at the end
      while (j) {
         if (hname[j] != ' ') break;
         hname[j] = 0;
         j--;
      }

      if (i) {
         strncpy(varexp,varexp0,i); varexp[i]=0;

         Int_t mustdelete=0;
         SetBit(kCustomHistogram);

         // parse things that follow the name of the histo between '(' and ')'.
         // At this point hname contains the name of the specified histogram.
         //   Now the syntax is exended to handle an hname of the following format
         //   hname(nBIN [[,[xlow]][,xhigh]],...)
         //   so enclosed in brackets is the binning information, xlow, xhigh, and
         //   the same for the other dimensions

         char *pstart;    // pointer to '('
         char *pend;      // pointer to ')'
         char *cdummy;    // dummy pointer
         int ncomma;      // number of commas between '(' and ')', later number of arguments
         int ncols;       // number of columns in varexpr
         Double_t value;  // parsed value (by sscanf)

         const Int_t maxvalues=9;

         pstart= strchr(hname,'(');
         pend =  strchr(hname,')');
         if (pstart != 0 ) {  // found the bracket

            mustdelete=1;

            // check that there is only one open and close bracket
            if (pstart == strrchr(hname,'(')  &&  pend == strrchr(hname,')')) {

               // count number of ',' between '(' and ')'
               ncomma=0;
               cdummy = pstart;
               cdummy = strchr(&cdummy[1],',');
               while (cdummy != 0) {
                  cdummy = strchr(&cdummy[1],',');
                  ncomma++;
               }

               if (ncomma+1 > maxvalues) {
                  Error("DrawSelect","ncomma+1>maxvalues, ncomma=%d, maxvalues=%d",ncomma,maxvalues);
                  ncomma=maxvalues-1;
               }

               ncomma++; // number of arguments
               cdummy = pstart;

               //   number of columns
               ncols  = 1;
               for (j=0;j<i;j++) {
                  if (varexp[j] == ':'
                      && ! ( (j>0&&varexp[j-1]==':') || varexp[j+1]==':' )
                      ) {
                     ncols++;
                  }
               }
               if (ncols > 3 ) {  // max 3 columns
                  Error("DrawSelect","ncols > 3, ncols=%d",ncols);
                  ncols = 0;
               }

               // check dimensions before and after ">>"
               if (ncols*3 < ncomma) {
                  Error("DrawSelect","ncols*3 < ncomma ncols=%d, ncomma=%d",ncols,ncomma);
                  ncomma = ncols*3;
               }

               // scan the values one after the other
               for (j=0;j<ncomma;j++) {
                  cdummy++;  // skip '(' or ','
                  if (sscanf(cdummy," %lf ",&value) == 1) {
                     cdummy=strchr(&cdummy[1],',');

                     switch (j) {  // do certain settings depending on position of argument
                        case 0:  // binning x-axis
                           nbinsx = (Int_t)value;
                           if      (ncols<2) {
                              gEnv->SetValue("Hist.Binning.1D.x",nbinsx);
                           } else if (ncols<3) {
                              gEnv->SetValue("Hist.Binning.2D.x",nbinsx);
                              gEnv->SetValue("Hist.Binning.2D.Prof",nbinsx);
                           } else {
                              gEnv->SetValue("Hist.Binning.3D.x",nbinsx);
                              gEnv->SetValue("Hist.Binning.3D.Profx",nbinsx);
                           }

                           break;
                        case 1:  // lower limit x-axis
                           xmin = value;
                           break;
                        case 2:  // upper limit x-axis
                           xmax = value;
                           break;
                        case 3:  // binning y-axis
                           nbinsy = (Int_t)value;
                           if (ncols<3) gEnv->SetValue("Hist.Binning.2D.y",nbinsy);
                           else {
                              gEnv->SetValue("Hist.Binning.3D.y",nbinsy);
                              gEnv->SetValue("Hist.Binning.3D.Profy",nbinsy);
                           }
                           break;
                        case 4:  // lower limit y-axis
                           ymin = value;
                           break;
                        case 5:  // upper limit y-axis
                           ymax = value;
                           break;
                        case 6:  // binning z-axis
                           nbinsz = (Int_t)value;
                           gEnv->SetValue("Hist.Binning.3D.z",nbinsz);
                           break;
                        case 7:  // lower limit z-axis
                           zmin = value;
                           break;
                        case 8:  // upper limit z-axis
                           zmax = value;
                           break;
                        default:
                           Error("DrawSelect","j>8");
                           break;
                     }
                  }  // if sscanf == 1
               } // for j=0;j<ncomma;j++
            } else {
               Error("Begin","Two open or close brackets found, hname=%s",hname);
            }

            // fix up hname
            pstart[0]='\0';   // removes things after (and including) '('
         } // if '(' is found

         j = strlen(hname) - 1; // skip ' '  at the end
         while (j) {
            if (hname[j] != ' ') break; // skip ' '  at the end
            hname[j] = 0;
            j--;
         }

         TObject *oldObject = gDirectory->Get(hname);  // if hname contains '(...)' the return values is NULL, which is what we want
         fOldHistogram = oldObject ? dynamic_cast<TH1*>(oldObject) : 0;

         if (!fOldHistogram && oldObject && !oldObject->InheritsFrom(TH1::Class())) {
            Error("Begin","An object of type '%s' has the same name as the requested histo (%s)",oldObject->IsA()->GetName(),hname);
            SetStatus(-1);
            return;
         }
         if (fOldHistogram && !hnameplus) fOldHistogram->Reset();  // reset unless adding is wanted

         if (mustdelete) {
            if (gDebug) {
               Warning("Begin","Deleting old histogram, since (possibly new) limits and binnings have been given");
            }
            delete fOldHistogram; fOldHistogram=0;
         }

      } else { // if (i)  // make selection list (i.e. varexp0 starts with ">>")
         TObject *oldObject = gDirectory->Get(hname);
         elist = oldObject ? dynamic_cast<TEventList*>(oldObject) : 0;

         if (!elist && oldObject) {
            Error("Begin","An object of type '%s' has the same name as the requested event list (%s)",
                  oldObject->IsA()->GetName(),hname);
            SetStatus(-1);
            return;
         }
         if (!elist) {
            elist = new TEventList(hname,realSelection.GetTitle(),1000,0);
         }
         if (elist) {
            if (!hnameplus) {
               if (elist==inElist) {
                  // We have been asked to reset the input list!!
                  // Let's set it aside for now ...
                  inElist = new TEventList(*elist);
                  fCleanElist = kTRUE;
                  fTree->SetEventList(inElist);
               }
               elist->Reset();
               elist->SetTitle(realSelection.GetTitle());
            } else {
               TCut old = elist->GetTitle();
               TCut upd = old || realSelection.GetTitle();
               elist->SetTitle(upd.GetTitle());
            }
         }
      }  // if (i)
   } else { // if (hname)
      hname  = hdefault;
      hkeep  = 0;
      varexp = (char*)varexp0;
      if (gDirectory) {
         fOldHistogram = (TH1*)gDirectory->Get(hname);
         if (fOldHistogram) { fOldHistogram->Delete(); fOldHistogram = 0;}
      }
   }

   // Decode varexp and selection
   if (!CompileVariables(varexp, realSelection.GetTitle())) {SetStatus(-1); return;}

   // In case fOldHistogram exists, check dimensionality
   Int_t nsel = strlen(selection);
   if (nsel > 1) {
      sprintf(htitle,"%s {%s}",varexp,selection);
   } else {
      sprintf(htitle,"%s",varexp);
   }
   if (fOldHistogram) {
      Int_t olddim = fOldHistogram->GetDimension();
      Int_t mustdelete = 0;
      if (fOldHistogram->InheritsFrom("TProfile")) {
         profile = kTRUE;
         olddim = 2;
      }
      if (fOldHistogram->InheritsFrom("TProfile2D")) {
         profile = kTRUE;
         olddim = 3;
      }
      if (opt.Contains("prof") && fDimension>1) {
         // ignore "prof" for 1D.
         if (!profile || olddim != fDimension) mustdelete = 1;
      } else {
         if (olddim != fDimension) mustdelete = 1;
      }
      if (mustdelete) {
         Warning("Begin","Deleting old histogram with different dimensions");
         delete fOldHistogram; fOldHistogram = 0;
      }
   }

   // Create a default canvas if none exists
   fDraw = 0;
   if (!gPad && !opt.Contains("goff") && fDimension > 0) {
      if (!gROOT->GetMakeDefCanvas())  {SetStatus(-1); return;}
      (gROOT->GetMakeDefCanvas())();
   }

   // 1-D distribution
   TH1 *hist;
   if (fDimension == 1) {
      fAction = 1;
      if (!fOldHistogram) {
         fNbins[0] = gEnv->GetValue("Hist.Binning.1D.x",100);
         if (gPad && optSame) {
            TListIter np(gPad->GetListOfPrimitives());
            TObject *op;
            TH1 *oldhtemp = 0;
            while ((op = np()) && !oldhtemp) {
               if (op->InheritsFrom("TH1")) oldhtemp = (TH1 *)op;
            }
            if (oldhtemp) {
               fNbins[0] = oldhtemp->GetXaxis()->GetNbins();
               fVmin[0]  = oldhtemp->GetXaxis()->GetXmin();
               fVmax[0]  = oldhtemp->GetXaxis()->GetXmax();
            } else {
               fVmin[0]  = gPad->GetUxmin();
               fVmax[0]  = gPad->GetUxmax();
            }
         } else {
            fAction   = -1;
            fVmin[0] = xmin;
            fVmax[0] = xmax;
            if (xmin < xmax) canRebin = kFALSE;
         }
      }
      if (fOldHistogram) {
         hist    = fOldHistogram;
         fNbins[0] = hist->GetXaxis()->GetNbins();
      } else {
         hist = new TH1F(hname,htitle,fNbins[0],fVmin[0],fVmax[0]);
         hist->SetLineColor(fTree->GetLineColor());
         hist->SetLineWidth(fTree->GetLineWidth());
         hist->SetLineStyle(fTree->GetLineStyle());
         hist->SetFillColor(fTree->GetFillColor());
         hist->SetFillStyle(fTree->GetFillStyle());
         hist->SetMarkerStyle(fTree->GetMarkerStyle());
         hist->SetMarkerColor(fTree->GetMarkerColor());
         hist->SetMarkerSize(fTree->GetMarkerSize());
         if (canRebin)hist->SetBit(TH1::kCanRebin);
         if (!hkeep) {
            hist->GetXaxis()->SetTitle(fVar1->GetTitle());
            hist->SetBit(kCanDelete);
            if (!opt.Contains("goff")) hist->SetDirectory(0);
         }
         if (opt.Length() && opt.Contains("e")) hist->Sumw2();
      }
      fVar1->SetAxis(hist->GetXaxis());
      fObject = hist;

      // 2-D distribution
   } else if (fDimension == 2) {
      fAction = 2;
      if (!fOldHistogram || !optSame) {
         fNbins[0] = gEnv->GetValue("Hist.Binning.2D.y",40);
         fNbins[1] = gEnv->GetValue("Hist.Binning.2D.x",40);
         if (opt.Contains("prof")) fNbins[1] = gEnv->GetValue("Hist.Binning.2D.Prof",100);
         if (optSame) {
            TH1 *oldhtemp = (TH1*)gPad->FindObject(hdefault);
            if (oldhtemp) {
               fNbins[1] = oldhtemp->GetXaxis()->GetNbins();
               fVmin[1]  = oldhtemp->GetXaxis()->GetXmin();
               fVmax[1]  = oldhtemp->GetXaxis()->GetXmax();
               fNbins[0] = oldhtemp->GetYaxis()->GetNbins();
               fVmin[0]  = oldhtemp->GetYaxis()->GetXmin();
               fVmax[0]  = oldhtemp->GetYaxis()->GetXmax();
            } else {
               fNbins[1] = gEnv->GetValue("Hist.Binning.2D.x",40);
               fVmin[1]  = gPad->GetUxmin();
               fVmax[1]  = gPad->GetUxmax();
               fNbins[0] = gEnv->GetValue("Hist.Binning.2D.y",40);
               fVmin[0]  = gPad->GetUymin();
               fVmax[0]  = gPad->GetUymax();
            }
         } else {
            if (!fOldHistogram) fAction = -2;
            fVmin[1] = xmin;
            fVmax[1] = xmax;
            fVmin[0] = ymin;
            fVmax[0] = ymax;
            if (xmin < xmax && ymin < ymax) canRebin = kFALSE;
         }
      }
      if (profile || opt.Contains("prof")) {
         TProfile *hp;
         if (fOldHistogram) {
            fAction = 4;
            hp = (TProfile*)fOldHistogram;
         } else {
            if (fAction < 0) {
               fAction = -4;
               fVmin[1] = xmin;
               fVmax[1] = xmax;
               if (xmin < xmax) canRebin = kFALSE;
            }
            if (fAction == 2) {
               //we come here when option = "same prof"
               fAction = -4;
               TH1 *oldhtemp = (TH1*)gPad->FindObject(hdefault);
               if (oldhtemp) {
                  fNbins[1] = oldhtemp->GetXaxis()->GetNbins();
                  fVmin[1]  = oldhtemp->GetXaxis()->GetXmin();
                  fVmax[1]  = oldhtemp->GetXaxis()->GetXmax();
               }
            }
            if (opt.Contains("profs")) {
               hp = new TProfile(hname,htitle,fNbins[1],fVmin[1], fVmax[1],"s");
            } else if (opt.Contains("profi")) {
               hp = new TProfile(hname,htitle,fNbins[1],fVmin[1], fVmax[1],"i");
            } else if (opt.Contains("profg")) {
               hp = new TProfile(hname,htitle,fNbins[1],fVmin[1], fVmax[1],"g");
            } else {
               hp = new TProfile(hname,htitle,fNbins[1],fVmin[1], fVmax[1],"");
            }
            if (!hkeep) {
               hp->SetBit(kCanDelete);
               if (!opt.Contains("goff")) hp->SetDirectory(0);
            }
            hp->SetLineColor(fTree->GetLineColor());
            hp->SetLineWidth(fTree->GetLineWidth());
            hp->SetLineStyle(fTree->GetLineStyle());
            hp->SetFillColor(fTree->GetFillColor());
            hp->SetFillStyle(fTree->GetFillStyle());
            hp->SetMarkerStyle(fTree->GetMarkerStyle());
            hp->SetMarkerColor(fTree->GetMarkerColor());
            hp->SetMarkerSize(fTree->GetMarkerSize());
            if (canRebin)hp->SetBit(TH1::kCanRebin);
         }
         fVar2->SetAxis(hp->GetXaxis());
         fObject = hp;

      } else {
         TH2F *h2;
         if (fOldHistogram) {
            h2 = (TH2F*)fOldHistogram;
         } else {
            h2 = new TH2F(hname,htitle,fNbins[1],fVmin[1], fVmax[1], fNbins[0], fVmin[0], fVmax[0]);
            h2->SetLineColor(fTree->GetLineColor());
            h2->SetFillColor(fTree->GetFillColor());
            h2->SetFillStyle(fTree->GetFillStyle());
            h2->SetMarkerStyle(fTree->GetMarkerStyle());
            h2->SetMarkerColor(fTree->GetMarkerColor());
            h2->SetMarkerSize(fTree->GetMarkerSize());
            if (canRebin)h2->SetBit(TH1::kCanRebin);
            if (!hkeep) {
               h2->GetXaxis()->SetTitle(fVar2->GetTitle());
               h2->GetYaxis()->SetTitle(fVar1->GetTitle());
               h2->SetBit(TH1::kNoStats);
               h2->SetBit(kCanDelete);
               if (!opt.Contains("goff")) h2->SetDirectory(0);
            }
         }
         fVar1->SetAxis(h2->GetYaxis());
         fVar2->SetAxis(h2->GetXaxis());
         Bool_t graph = kFALSE;
         Int_t l = opt.Length();
         if (l == 0 || optSame) graph = kTRUE;
         if (opt.Contains("p")     || opt.Contains("*")    || opt.Contains("l"))    graph = kTRUE;
         if (opt.Contains("surf")  || opt.Contains("lego") || opt.Contains("cont")) graph = kFALSE;
         if (opt.Contains("col")   || opt.Contains("hist") || opt.Contains("scat")) graph = kFALSE;
         if (opt.Contains("box"))                                                   graph = kFALSE;
         fObject = h2;
         if (graph) {
            fAction = 12;
            if (!fOldHistogram && !optSame) fAction = -12;
         }
      }

      // 3-D distribution
   } else if (fDimension == 3 || fDimension == 4) {
      fAction = 3;
      if (fDimension == 4) fAction = 40;
      if (!fOldHistogram || !optSame) {
         fNbins[0] = gEnv->GetValue("Hist.Binning.3D.z",20);
         fNbins[1] = gEnv->GetValue("Hist.Binning.3D.y",20);
         fNbins[2] = gEnv->GetValue("Hist.Binning.3D.x",20);
         if (fDimension == 3 && opt.Contains("prof")) {
            fNbins[1] = gEnv->GetValue("Hist.Binning.3D.Profy",20);
            fNbins[2] = gEnv->GetValue("Hist.Binning.3D.Profx",20);
         }
         if (optSame) {
            TH1 *oldhtemp = (TH1*)gPad->FindObject(hdefault);
            if (oldhtemp) {
               fNbins[2] = oldhtemp->GetXaxis()->GetNbins();
               fVmin[2]  = oldhtemp->GetXaxis()->GetXmin();
               fVmax[2]  = oldhtemp->GetXaxis()->GetXmax();
               fNbins[1] = oldhtemp->GetYaxis()->GetNbins();
               fVmin[1]  = oldhtemp->GetYaxis()->GetXmin();
               fVmax[1]  = oldhtemp->GetYaxis()->GetXmax();
               fNbins[0] = oldhtemp->GetZaxis()->GetNbins();
               fVmin[0]  = oldhtemp->GetZaxis()->GetXmin();
               fVmax[0]  = oldhtemp->GetZaxis()->GetXmax();
            } else {
               TView *view = gPad->GetView();
               Double_t *rmin = view->GetRmin();
               Double_t *rmax = view->GetRmax();
               fNbins[2] = gEnv->GetValue("Hist.Binning.3D.z",20);
               fVmin[2]  = rmin[0];
               fVmax[2]  = rmax[0];
               fNbins[1] = gEnv->GetValue("Hist.Binning.3D.y",20);
               fVmin[1]  = rmin[1];
               fVmax[1]  = rmax[1];
               fNbins[0] = gEnv->GetValue("Hist.Binning.3D.x",20);
               fVmin[0]  = rmin[2];
               fVmax[0]  = rmax[2];
            }
         } else {
            if (!fOldHistogram && fDimension ==3) fAction = -3;
            fVmin[2] = xmin;
            fVmax[2] = xmax;
            fVmin[1] = ymin;
            fVmax[1] = ymax;
            fVmin[0] = zmin;
            fVmax[0] = zmax;
            if (xmin < xmax && ymin < ymax && zmin < zmax) canRebin = kFALSE;
         }
      }
      if ((fDimension == 3) && (profile || opt.Contains("prof"))) {
         TProfile2D *hp;
         if (fOldHistogram) {
            fAction = 23;
            hp = (TProfile2D*)fOldHistogram;
         } else {
            if (fAction < 0) {
               fAction = -23;
               fVmin[2] = xmin;
               fVmax[2] = xmax;
               fVmin[1] = ymin;
               fVmax[1] = ymax;
               if (xmin < xmax && ymin < ymax) canRebin = kFALSE;
            }
            if (opt.Contains("profs")) {
               hp = new TProfile2D(hname,htitle,fNbins[2],fVmin[2], fVmax[2],fNbins[1],fVmin[1], fVmax[1],"s");
            } else if (opt.Contains("profi")) {
               hp = new TProfile2D(hname,htitle,fNbins[2],fVmin[2], fVmax[2],fNbins[1],fVmin[1], fVmax[1],"i");
            } else if (opt.Contains("profg")) {
               hp = new TProfile2D(hname,htitle,fNbins[2],fVmin[2], fVmax[2],fNbins[1],fVmin[1], fVmax[1],"g");
            } else {
               hp = new TProfile2D(hname,htitle,fNbins[2],fVmin[2], fVmax[2],fNbins[1],fVmin[1], fVmax[1],"");
            }
            if (!hkeep) {
               hp->SetBit(kCanDelete);
               if (!opt.Contains("goff")) hp->SetDirectory(0);
            }
            hp->SetLineColor(fTree->GetLineColor());
            hp->SetLineWidth(fTree->GetLineWidth());
            hp->SetLineStyle(fTree->GetLineStyle());
            hp->SetFillColor(fTree->GetFillColor());
            hp->SetFillStyle(fTree->GetFillStyle());
            hp->SetMarkerStyle(fTree->GetMarkerStyle());
            hp->SetMarkerColor(fTree->GetMarkerColor());
            hp->SetMarkerSize(fTree->GetMarkerSize());
            if (canRebin)hp->SetBit(TH1::kCanRebin);
         }
         fVar2->SetAxis(hp->GetYaxis());
         fVar3->SetAxis(hp->GetXaxis());
         fObject = hp;
      } else if (fDimension == 3 && opt.Contains("col")) {
         TH2F *h2;
         if (fOldHistogram) {
            h2 = (TH2F*)fOldHistogram;
         } else {
            h2 = new TH2F(hname,htitle,fNbins[1],fVmin[1], fVmax[1], fNbins[0], fVmin[0], fVmax[0]);
            h2->SetLineColor(fTree->GetLineColor());
            h2->SetFillColor(fTree->GetFillColor());
            h2->SetFillStyle(fTree->GetFillStyle());
            h2->SetMarkerStyle(fTree->GetMarkerStyle());
            h2->SetMarkerColor(fTree->GetMarkerColor());
            h2->SetMarkerSize(fTree->GetMarkerSize());
            if (canRebin)h2->SetBit(TH1::kCanRebin);
            if (!hkeep) {
               h2->GetXaxis()->SetTitle(fVar2->GetTitle());
               h2->GetZaxis()->SetTitle(fVar1->GetTitle());
               h2->SetBit(TH1::kNoStats);
               h2->SetBit(kCanDelete);
               if (!opt.Contains("goff")) h2->SetDirectory(0);
            }
         }
         fVar1->SetAxis(h2->GetYaxis());
         fVar2->SetAxis(h2->GetXaxis());
         fObject = h2;
         fAction = 33;
      } else {
         TH3F *h3;
         if (fOldHistogram) {
            h3 = (TH3F*)fOldHistogram;
         } else {
            h3 = new TH3F(hname,htitle,fNbins[2],fVmin[2], fVmax[2],fNbins[1],fVmin[1], fVmax[1], fNbins[0], fVmin[0], fVmax[0]);
            h3->SetLineColor(fTree->GetLineColor());
            h3->SetFillColor(fTree->GetFillColor());
            h3->SetFillStyle(fTree->GetFillStyle());
            h3->SetMarkerStyle(fTree->GetMarkerStyle());
            h3->SetMarkerColor(fTree->GetMarkerColor());
            h3->SetMarkerSize(fTree->GetMarkerSize());
            if (canRebin)h3->SetBit(TH1::kCanRebin);
            if (!hkeep) {
               h3->GetXaxis()->SetTitleOffset(1.5);
               h3->GetYaxis()->SetTitleOffset(1.5);
               h3->GetXaxis()->SetTitle(fVar3->GetTitle());
               h3->GetYaxis()->SetTitle(fVar2->GetTitle());
               h3->GetZaxis()->SetTitle(fVar1->GetTitle());
               h3->SetBit(kCanDelete);
               h3->SetBit(TH1::kNoStats);
               if (!opt.Contains("goff")) h3->SetDirectory(0);
            }
         }
         fVar1->SetAxis(h3->GetZaxis());
         fVar2->SetAxis(h3->GetYaxis());
         fVar3->SetAxis(h3->GetXaxis());
         fObject = h3;
         Int_t noscat = strlen(option);
         if (optSame) noscat -= 4;
         if (!noscat && fDimension ==3) {
            fAction = 13;
            if (!fOldHistogram && !optSame) fAction = -13;
         }
      }
      // An Event List
   } else if (elist) {
      fAction = 5;
      fOldEstimate = fTree->GetEstimate();
      fTree->SetEstimate(1);
      fObject = elist;
   }
   if (hkeep) delete [] varexp;
   if (hnamealloc) delete [] hnamealloc;
   fVar1Multiple = kFALSE;
   fVar2Multiple = kFALSE;
   fVar3Multiple = kFALSE;
   fVar4Multiple = kFALSE;
   fSelectMultiple = kFALSE;
   if (fVar1 && fVar1->GetMultiplicity()) fVar1Multiple = kTRUE;
   if (fVar2 && fVar2->GetMultiplicity()) fVar2Multiple = kTRUE;
   if (fVar3 && fVar3->GetMultiplicity()) fVar3Multiple = kTRUE;
   if (fVar4 && fVar4->GetMultiplicity()) fVar4Multiple = kTRUE;
   if (fSelect && fSelect->GetMultiplicity()) fSelectMultiple = kTRUE;

   fForceRead = fTree->TestBit(TTree::kForceRead);
   fWeight  = fTree->GetWeight();
   fNfill   = 0;
   if (!fV1 && fVar1)   fV1 = new Double_t[fTree->GetEstimate()];
   if (!fV2 && fVar2)   fV2 = new Double_t[fTree->GetEstimate()];
   if (!fV3 && fVar3)   fV3 = new Double_t[fTree->GetEstimate()];
   if (!fV4 && fVar4)   fV4 = new Double_t[fTree->GetEstimate()];
   if (!fW)             fW  = new Double_t[fTree->GetEstimate()];

   fVmin[0] = fVmin[1] = fVmin[2] = fVmin[3] =  FLT_MAX; //in float.h
   fVmax[0] = fVmax[1] = fVmax[2] = fVmax[3] = -fVmin[0];
}

//______________________________________________________________________________
void TSelectorDraw::ClearFormula()
{
   // Delete internal buffers.

   ResetBit(kWarn);
   delete fVar1;   fVar1 = 0;
   delete fVar2;   fVar2 = 0;
   delete fVar3;   fVar3 = 0;
   delete fVar4;   fVar4 = 0;
   delete fSelect; fSelect = 0;
   fManager = 0;
   fMultiplicity = 0;
}

//______________________________________________________________________________
Bool_t TSelectorDraw::CompileVariables(const char *varexp, const char *selection)
{
   // Compile input variables and selection expression.
   //
   //  varexp is an expression of the general form e1:e2:e3
   //    where e1,etc is a formula referencing a combination of the columns
   //  Example:
   //     varexp = x  simplest case: draw a 1-Dim distribution of column named x
   //            = sqrt(x)         : draw distribution of sqrt(x)
   //            = x*y/z
   //            = y:sqrt(x) 2-Dim dsitribution of y versus sqrt(x)
   //
   //  selection is an expression with a combination of the columns
   //  Example:
   //      selection = "x<y && sqrt(z)>3.2"
   //       in a selection all the C++ operators are authorized
   //
   //  Return kFALSE if any of the variable is not compilable.

   const Int_t nMAXCOL = 4;
   TString title;
   Int_t i,nch,ncols;
   Int_t index[nMAXCOL];

   // Compile selection expression if there is one
   fDimension = 0;
   ClearFormula();
   fMultiplicity = 0;
   fObjEval = kFALSE;

   if (strlen(selection)) {
      fSelect = new TTreeFormula("Selection",selection,fTree);
      fSelect->SetQuickLoad(kTRUE);
      if (!fSelect->GetNdim()) {delete fSelect; fSelect = 0; return kFALSE; }
   }

   // if varexp is empty, take first column by default
   nch = strlen(varexp);
   if (nch == 0) {
      fDimension = 0;
      fManager = new TTreeFormulaManager();
      if (fSelect) fManager->Add(fSelect);
      fTree->ResetBit(TTree::kForceRead);

      fManager->Sync();

      if (fManager->GetMultiplicity()==-1) fTree->SetBit(TTree::kForceRead);
      if (fManager->GetMultiplicity()>=1) fMultiplicity = fManager->GetMultiplicity();

      return kTRUE;
   }
   title = varexp;

   // otherwise select only the specified columns
   ncols  = 1;
   for (i=0;i<nch;i++)  if (title[i] == ':' && ! ( (i>0&&title[i-1]==':') || title[i+1]==':' ) ) ncols++;
   if (ncols > 4 ) return kFALSE;
   MakeIndex(title,index);

   fManager = new TTreeFormulaManager();
   if (fSelect) fManager->Add(fSelect);
   fTree->ResetBit(TTree::kForceRead);
   if (ncols >= 1) {
      fVar1 = new TTreeFormula("Var1",GetNameByIndex(title,index,0),fTree);
      fVar1->SetQuickLoad(kTRUE);
      if (!fVar1->GetNdim()) { ClearFormula(); return kFALSE;}
      fManager->Add(fVar1);
   }
   if (ncols >= 2) {
      fVar2 = new TTreeFormula("Var2",GetNameByIndex(title,index,1),fTree);
      fVar2->SetQuickLoad(kTRUE);
      if (!fVar2->GetNdim()) { ClearFormula(); return kFALSE;}
      fManager->Add(fVar2);
   }
   if (ncols >= 3) {
      fVar3 = new TTreeFormula("Var3",GetNameByIndex(title,index,2),fTree);
      fVar3->SetQuickLoad(kTRUE);
      if (!fVar3->GetNdim()) { ClearFormula(); return kFALSE;}
      fManager->Add(fVar3);
   }
   if (ncols >= 4) {
      fVar4 = new TTreeFormula("Var4",GetNameByIndex(title,index,3),fTree);
      fVar4->SetQuickLoad(kTRUE);
      if (!fVar4->GetNdim()) { ClearFormula(); return kFALSE;}
      fManager->Add(fVar4);
   }
   fManager->Sync();

   if (fManager->GetMultiplicity()==-1) fTree->SetBit(TTree::kForceRead);
   if (fManager->GetMultiplicity()>=1) fMultiplicity = fManager->GetMultiplicity();

   fDimension    = ncols;

   if (ncols==1) {
      TClass *cl = fVar1->EvalClass();
      if (cl) {
         fObjEval = kTRUE;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
const char *TSelectorDraw::GetNameByIndex(TString &varexp, Int_t *index,Int_t colindex)
{
   // Return name corresponding to colindex in varexp.
   //
   // varexp is a string of names separated by :
   // index is an array with pointers to the start of name[i] in varexp

   Int_t i1,n;
   static TString column;
   if (colindex<0 ) return "";
   i1 = index[colindex] + 1;
   n  = index[colindex+1] - i1;
   column = varexp(i1,n);
   return column.Data();
}

//______________________________________________________________________________
void TSelectorDraw::MakeIndex(TString &varexp, Int_t *index)
{
   // Build Index array for names in varexp.

   Int_t ivar = 1;
   index[0]  = -1;
   for (Int_t i=0;i<varexp.Length();i++) {
      if (varexp[i] == ':'
          && ! ( (i>0&&varexp[i-1]==':') || varexp[i+1]==':' )
          ) {
         index[ivar] = i;
         ivar++;
      }
   }
   index[ivar] = varexp.Length();
}


//______________________________________________________________________________
Bool_t TSelectorDraw::Notify()
{
   // This function is called at the first entry of a new tree in a chain.

   if (fTree) fWeight  = fTree->GetWeight();
   if (fVar1) fVar1->UpdateFormulaLeaves();
   if (fVar2) fVar2->UpdateFormulaLeaves();
   if (fVar3) fVar3->UpdateFormulaLeaves();
   if (fVar4) fVar4->UpdateFormulaLeaves();
   if (fSelect) fSelect->UpdateFormulaLeaves();
   return kTRUE;
}

//______________________________________________________________________________
void TSelectorDraw::ProcessFill(Long64_t entry)
{
   // Called in the entry loop for all entries accepted by Select.

   if (fObjEval) {
      ProcessFillObject(entry);
      return;
   }

   if (fMultiplicity) {
      ProcessFillMultiple(entry);
      return;
   }

   // simple case with no multiplicity
   if ( fForceRead && fManager->GetNdata() <= 0) return;

   if (fSelect) {
      fW[fNfill] = fWeight*fSelect->EvalInstance(0);
      if (!fW[fNfill]) return;
   } else fW[fNfill] = fWeight;
   if (fVar1) {
      fV1[fNfill] = fVar1->EvalInstance(0);
   }
   if (fVar2) {
      fV2[fNfill] = fVar2->EvalInstance(0);
      if (fVar3) {
         fV3[fNfill] = fVar3->EvalInstance(0);
         if (fVar4) {
            fV4[fNfill] = fVar4->EvalInstance(0);
         }
      }
   }
   fNfill++;
   if (fNfill >= fTree->GetEstimate()) {
      TakeAction();
      fNfill = 0;
   }
}

//______________________________________________________________________________
void TSelectorDraw::ProcessFillMultiple(Long64_t /*entry*/)
{
   // Called in the entry loop for all entries accepted by Select.
   // Complex case with multiplicity.

   // Grab the array size of the formulas for this entry
   Int_t ndata = fManager->GetNdata();

   // No data at all, let's move on to the next entry.
   if (!ndata) return;

   Int_t nfill0 = fNfill;

   // Calculate the first values
   if (fSelect) {
      fW[fNfill] = fWeight*fSelect->EvalInstance(0);
      if (!fW[fNfill] && !fSelectMultiple) return;
   } else fW[fNfill] = fWeight;

   // Always call EvalInstance(0) to insure the loading
   // of the branches.
   if (fW[fNfill]) {
      if (fVar1) {
         fV1[fNfill] = fVar1->EvalInstance(0);
         if (fVar2) {
            fV2[fNfill] = fVar2->EvalInstance(0);
            if (fVar3) {
               fV3[fNfill] = fVar3->EvalInstance(0);
               if (fVar4) {
                  fV4[fNfill] = fVar4->EvalInstance(0);
               }
            }
         }
      }
      fNfill++;
      if (fNfill >= fTree->GetEstimate()) {
         TakeAction();
         fNfill = 0;
      }
   } else {
      if (fVar1) {
         fVar1->ResetLoading();
         if (fVar2) {
            fVar2->ResetLoading();
            if (fVar3) {
               fVar3->ResetLoading();
               if (fVar4) {
                  fVar4->ResetLoading();
               }
            }
         }
      }
   }
   Double_t ww = fW[nfill0];

   for (Int_t i=1;i<ndata;i++) {
      if (fSelectMultiple) {
         ww = fWeight*fSelect->EvalInstance(i);
         if (ww == 0) continue;
         if (fNfill == nfill0) {
            if (fVar1) {
               if (!fVar1Multiple) fV1[nfill0] = fVar1->EvalInstance(0);
               if (fVar2) {
                  if (!fVar2Multiple) fV2[nfill0] = fVar2->EvalInstance(0);
                  if (fVar3) {
                     if (!fVar3Multiple) fV3[nfill0] = fVar3->EvalInstance(0);
                     if (fVar4) {
                        if (!fVar4Multiple) fV4[nfill0] = fVar4->EvalInstance(0);
                     }
                  }
               }
            }
         }
      }
      if (fVar1) {
         if (fVar1Multiple) fV1[fNfill] = fVar1->EvalInstance(i);
         else               fV1[fNfill] = fV1[nfill0];
         if (fVar2) {
            if (fVar2Multiple) fV2[fNfill] = fVar2->EvalInstance(i);
            else               fV2[fNfill] = fV2[nfill0];
            if (fVar3) {
               if (fVar3Multiple) fV3[fNfill] = fVar3->EvalInstance(i);
               else               fV3[fNfill] = fV3[nfill0];
               if (fVar4) {
                  if (fVar4Multiple) fV4[fNfill] = fVar4->EvalInstance(i);
                  else               fV4[fNfill] = fV4[nfill0];
               }
            }
         }
      }
      fW[fNfill] = ww;

      fNfill++;
      if (fNfill >= fTree->GetEstimate()) {
         TakeAction();
         fNfill = 0;
      }
   }
}

//______________________________________________________________________________
void TSelectorDraw::ProcessFillObject(Long64_t /*entry*/)
{
   // Called in the entry loop for all entries accepted by Select.
   // Case where the only variable returns an object (or pointer to).

   // Complex case with multiplicity.

   // Grab the array size of the formulas for this entry
   Int_t ndata = fManager->GetNdata();

   // No data at all, let's move on to the next entry.
   if (!ndata) return;

   Int_t nfill0 = fNfill;
   Double_t ww = 0;

   for (Int_t i=0;i<ndata;i++) {
      if (i==0) {
         if (fSelect) {
            fW[fNfill] = fWeight*fSelect->EvalInstance(0);
            if (!fW[fNfill] && !fSelectMultiple) return;
         } else fW[fNfill] = fWeight;
         ww = fW[nfill0];
      } else if (fSelectMultiple) {
         ww = fWeight*fSelect->EvalInstance(i);
         if (ww == 0) continue;
      }
      if (fVar1) {
         TClass *cl = fVar1->EvalClass();
         if (cl==TBits::Class()) {

            void *obj = fVar1->EvalObject(i);

            TBits *bits = (TBits*)obj;
            Int_t nbits = bits->GetNbits();

            Int_t nextbit = -1;
            while(1) {
               nextbit = bits->FirstSetBit(nextbit+1);
               if (nextbit >= nbits) break;
               fV1[fNfill] = nextbit;
               fW[fNfill] =  ww;
               fNfill++;
            }

         } else {

            if (!TestBit(kWarn)) {
               Warning("ProcessFillObject",
                       "Not implemented for %s",
                       cl?cl->GetName():"unknown class");
               SetBit(kWarn);
            }

         }
      }
      if (fNfill >= fTree->GetEstimate()) {
         TakeAction();
         fNfill = 0;
      }
   }

}

//_______________________________________________________________________
void TSelectorDraw::SetEstimate(Long64_t)
{
   // Set number of entries to estimate variable limits.

   delete [] fV1;  fV1 = 0;
   delete [] fV2;  fV2 = 0;
   delete [] fV3;  fV3 = 0;
   delete [] fV4;  fV4 = 0;
   delete [] fW;   fW  = 0;
}

//______________________________________________________________________________
void TSelectorDraw::TakeAction()
{
   // Execute action for object obj fNfill times.

   Int_t i;
   //__________________________1-D histogram_______________________
   if      (fAction ==  1) ((TH1*)fObject)->FillN(fNfill,fV1,fW);
   //__________________________2-D histogram_______________________
   else if (fAction ==  2) {
      TH2 *h2 = (TH2*)fObject;
      for(i=0;i<fNfill;i++) h2->Fill(fV2[i],fV1[i],fW[i]);
   }
   //__________________________Profile histogram_______________________
   else if (fAction ==  4) ((TProfile*)fObject)->FillN(fNfill,fV2,fV1,fW);
   //__________________________Event List______________________________
   else if (fAction ==  5) {
      TEventList *elist = (TEventList*)fObject;
      Long64_t enumb = fTree->GetChainOffset() + fTree->GetReadEntry();
      if (elist->GetIndex(enumb) < 0) elist->Enter(enumb);
   }
   //__________________________2D scatter plot_______________________
   else if (fAction == 12) {
      TH2 *h2 = (TH2*)fObject;
      if (h2->TestBit(TH1::kCanRebin) && h2->TestBit(kCanDelete)) {
         for (i=0;i<fNfill;i++) {
            if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
            if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
            if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
            if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
         }
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h2,fVmin[1],fVmax[1],fVmin[0],fVmax[0]);
      }
      TGraph *pm = new TGraph(fNfill,fV2,fV1);
      pm->SetEditable(kFALSE);
      pm->SetBit(kCanDelete);
      pm->SetMarkerStyle(fTree->GetMarkerStyle());
      pm->SetMarkerColor(fTree->GetMarkerColor());
      pm->SetMarkerSize(fTree->GetMarkerSize());
      pm->SetLineColor(fTree->GetLineColor());
      pm->SetLineStyle(fTree->GetLineStyle());
      pm->SetFillColor(fTree->GetFillColor());
      pm->SetFillStyle(fTree->GetFillStyle());

      if (!fDraw && !strstr(fOption.Data(),"goff")) {
         if (fOption.Length() == 0 || fOption == "same")  pm->Draw("p");
         else                                             pm->Draw(fOption.Data());
      }
      if (!h2->TestBit(kCanDelete)) {
         for (i=0;i<fNfill;i++) h2->Fill(fV2[i],fV1[i],fW[i]);
      }
   }
   //__________________________3D scatter plot_______________________
   else if (fAction ==  3) {
      TH3 *h3 =(TH3*)fObject;
      for(i=0;i<fNfill;i++) h3->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
   }
   else if (fAction == 13) {
      TPolyMarker3D *pm3d = new TPolyMarker3D(fNfill);
      pm3d->SetMarkerStyle(fTree->GetMarkerStyle());
      pm3d->SetMarkerColor(fTree->GetMarkerColor());
      pm3d->SetMarkerSize(fTree->GetMarkerSize());
      for (i=0;i<fNfill;i++) { pm3d->SetPoint(i,fV3[i],fV2[i],fV1[i]);}
      pm3d->Draw();
      TH3 *h3 =(TH3*)fObject;
      for(i=0;i<fNfill;i++) h3->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
   }
   //__________________________3D scatter plot (3rd variable = col)__
   else if (fAction == 33) {
      TH2 *h2 = (TH2*)fObject;
      TakeEstimate();
      Int_t ncolors  = gStyle->GetNumberOfColors();
      TObjArray *grs = (TObjArray*)h2->GetListOfFunctions()->FindObject("graphs");
      Int_t col;
      TGraph *gr;
      if (!grs) {
         grs = new TObjArray(ncolors);
         grs->SetOwner();
         grs->SetName("graphs");
         h2->GetListOfFunctions()->Add(grs, "P");
         for (col=0;col<ncolors;col++) {
            gr = new TGraph();
            gr->SetMarkerColor(gStyle->GetColorPalette(col));
            gr->SetMarkerStyle(fTree->GetMarkerStyle());
            gr->SetMarkerSize(fTree->GetMarkerSize());
            grs->AddAt(gr,col);
         }
      }
      h2->SetEntries(fNfill);
      h2->SetMinimum(fVmin[2]);
      h2->SetMaximum(fVmax[2]);
      // Fill the graphs acording to the color
      for (i=0;i<fNfill;i++) {
         col = Int_t(ncolors*((fV3[i]-fVmin[2])/(fVmax[2]-fVmin[2])));
         if (col < 0) col = 0;
         if (col > ncolors-1) col = ncolors-1;
         gr = (TGraph*)grs->UncheckedAt(col);
         if (gr) gr->SetPoint(gr->GetN(),fV2[i],fV1[i]);
      }
      // Remove potential empty graphs
      for (col=0;col<ncolors;col++) {
         gr = (TGraph*)grs->At(col);
         if (gr && gr->GetN() <= 0) grs->Remove(gr);
      }
   }
   //__________________________2D Profile Histogram__________________
   else if (fAction == 23) {
      TProfile2D *hp2 =(TProfile2D*)fObject;
      for(i=0;i<fNfill;i++) hp2->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
   }
   //__________________________4D scatter plot_______________________
   else if (fAction ==  40) {
      TakeEstimate();
      TH3 *h3 =(TH3*)fObject;
      Int_t ncolors  = gStyle->GetNumberOfColors();
      TObjArray *pms = (TObjArray*)h3->GetListOfFunctions()->FindObject("polymarkers");
      Int_t col;
      TPolyMarker3D *pm3d;
      if (!pms) {
         pms = new TObjArray(ncolors);
         pms->SetOwner();
         pms->SetName("polymarkers");
         h3->GetListOfFunctions()->Add(pms);
         for (col=0;col<ncolors;col++) {
            pm3d = new TPolyMarker3D();
            pm3d->SetMarkerColor(gStyle->GetColorPalette(col));
            pm3d->SetMarkerStyle(fTree->GetMarkerStyle());
            pm3d->SetMarkerSize(fTree->GetMarkerSize());
            pms->AddAt(pm3d,col);
         }
      }
      h3->SetEntries(fNfill);
      h3->SetMinimum(fVmin[3]);
      h3->SetMaximum(fVmax[3]);
      for (i=0;i<fNfill;i++) {
         col = Int_t(ncolors*((fV4[i]-fVmin[3])/(fVmax[3]-fVmin[3])));
         if (col < 0) col = 0;
         if (col > ncolors-1) col = ncolors-1;
         pm3d = (TPolyMarker3D*)pms->UncheckedAt(col);
         pm3d->SetPoint(pm3d->GetLastPoint()+1,fV3[i],fV2[i],fV1[i]);
      }
   }
   //__________________________something else_______________________
   else if (fAction < 0) {
      fAction = -fAction;
      TakeEstimate();
   }

   // Do we need to update screen?
   fSelectedRows += fNfill;
   if (!fTree->GetUpdate()) return;
   if (fSelectedRows > fDraw+fTree->GetUpdate()) {
      if (fDraw) gPad->Modified();
      else       fObject->Draw(fOption.Data());
      gPad->Update();
      fDraw = fSelectedRows;
   }
}

//______________________________________________________________________________
void TSelectorDraw::TakeEstimate()
{
   // Estimate limits for 1-D, 2-D or 3-D objects.

   Int_t i;
   Double_t rmin[3],rmax[3];
   Double_t vminOld[4], vmaxOld[4];
   for (i = 0; i < 4; i++) {
      vminOld[i] = fVmin[i];
      vmaxOld[i] = fVmax[i];
   }
   fVmin[0] = fVmin[1] = fVmin[2] = FLT_MAX; //in float.h
   fVmax[0] = fVmax[1] = fVmax[2] = -fVmin[0];
   //__________________________1-D histogram_______________________
   if      (fAction ==  1) {
      TH1 *h1 = (TH1*)fObject;
      if (fObject->TestBit(TH1::kCanRebin)) {
         for (i=0;i<fNfill;i++) {
            if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
            if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
         }
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h1,fVmin[0],fVmax[0]);
      }
      h1->FillN(fNfill, fV1, fW);
   //__________________________2-D histogram_______________________
   } else if (fAction ==  2) {
      TH2 *h2 = (TH2*)fObject;
      if (fObject->TestBit(TH1::kCanRebin)) {
         for (i=0;i<fNfill;i++) {
            if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
            if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
            if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
            if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
         }
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h2,fVmin[1],fVmax[1],fVmin[0],fVmax[0]);
      }
      for(i=0;i<fNfill;i++) h2->Fill(fV2[i],fV1[i],fW[i]);
   //__________________________Profile histogram_______________________
   } else if (fAction ==  4) {
      TProfile *hp = (TProfile*)fObject;
      if (fObject->TestBit(TH1::kCanRebin)) {
         for (i=0;i<fNfill;i++) {
            if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
            if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
            if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
            if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
         }
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hp,fVmin[1],fVmax[1]);
      }
      hp->FillN(fNfill, fV2, fV1, fW);
   //__________________________2D scatter plot_______________________
   } else if (fAction == 12) {
      TH2 *h2 = (TH2*)fObject;
      if (h2->TestBit(TH1::kCanRebin)) {
         for (i=0;i<fNfill;i++) {
            if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
            if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
            if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
            if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
         }
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h2,fVmin[1],fVmax[1],fVmin[0],fVmax[0]);
      }

      if (!strstr(fOption.Data(),"same") && !strstr(fOption.Data(),"goff")) {
         if (!h2->TestBit(kCanDelete)) {
            // case like: T.Draw("y:x>>myhist")
            // we must draw a copy before filling the histogram h2=myhist
            // because h2 will be filled below and we do not want to show
            // the binned scatter-plot, the TGraph being better.
            TH1 *h2c = h2->DrawCopy(fOption.Data());
            h2c->SetStats(kFALSE);
         } else {
            // case like: T.Draw("y:x")
            // h2 is a temporary histogram (htemp). This histogram
            // will be automatically deleted by TPad::Clear
            h2->Draw();
         }
         gPad->Update();
      }
      TGraph *pm = new TGraph(fNfill,fV2,fV1);
      pm->SetEditable(kFALSE);
      pm->SetBit(kCanDelete);
      pm->SetMarkerStyle(fTree->GetMarkerStyle());
      pm->SetMarkerColor(fTree->GetMarkerColor());
      pm->SetMarkerSize(fTree->GetMarkerSize());
      pm->SetLineColor(fTree->GetLineColor());
      pm->SetLineStyle(fTree->GetLineStyle());
      pm->SetFillColor(fTree->GetFillColor());
      pm->SetFillStyle(fTree->GetFillStyle());
      if (!fDraw && !strstr(fOption.Data(),"goff")) {
         if (fOption.Length() == 0 || fOption == "same")  pm->Draw("p");
         else                                             pm->Draw(fOption.Data());
      }
      if (!h2->TestBit(kCanDelete)) {
         for (i=0;i<fNfill;i++) h2->Fill(fV2[i],fV1[i],fW[i]);
      }
   //__________________________3D scatter plot with option col_______________________
   } else if (fAction == 33) {
      TH2 *h2 = (TH2*)fObject;
      Bool_t process2 = kFALSE;
      if (h2->TestBit(TH1::kCanRebin)) {
         if (vminOld[2] == FLT_MAX)
            process2 = kTRUE;
         for (i = 0; i < 4; i++) {
            fVmin[i] = vminOld[i];
            fVmax[i] = vmaxOld[i];
         }
         for (i=0;i<fNfill;i++) {
            if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
            if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
            if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
            if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
            if (process2) {
               if (fVmin[2] > fV3[i]) fVmin[2] = fV3[i];
               if (fVmax[2] < fV3[i]) fVmax[2] = fV3[i];
            }
         }
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h2,fVmin[1],fVmax[1],fVmin[0],fVmax[0]);
      }
   //__________________________3D scatter plot_______________________
   } else if (fAction == 3 || fAction == 13) {
      TH3 *h3 = (TH3*)fObject;
      if (fObject->TestBit(TH1::kCanRebin)) {
         for (i=0;i<fNfill;i++) {
            if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
            if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
            if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
            if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
            if (fVmin[2] > fV3[i]) fVmin[2] = fV3[i];
            if (fVmax[2] < fV3[i]) fVmax[2] = fV3[i];
         }
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h3,fVmin[2],fVmax[2],fVmin[1],fVmax[1],fVmin[0],fVmax[0]);
      }
      if (fAction == 3) {
         for (i=0;i<fNfill;i++) h3->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
         return;
      }
      if (!strstr(fOption.Data(),"same") && !strstr(fOption.Data(),"goff")) {
         if (!h3->TestBit(kCanDelete)) {
            // case like: T.Draw("y:x>>myhist")
            // we must draw a copy before filling the histogram h3=myhist
            // because h3 will be filled below and we do not want to show
            // the binned scatter-plot, the TGraph being better.
            TH1 *h3c = h3->DrawCopy(fOption.Data());
            h3c->SetStats(kFALSE);
         } else {
            // case like: T.Draw("y:x")
            // h3 is a temporary histogram (htemp). This histogram
            // will be automatically deleted by TPad::Clear
            h3->Draw(fOption.Data());
         }
         gPad->Update();
      } else {
         rmin[0] = fVmin[2]; rmin[1] = fVmin[1]; rmin[2] = fVmin[0];
         rmax[0] = fVmax[2]; rmax[1] = fVmax[1]; rmax[2] = fVmax[0];
         gPad->Clear();
         gPad->Range(-1,-1,1,1);
         new TView(rmin,rmax,1);
      }
      TPolyMarker3D *pm3d = new TPolyMarker3D(fNfill);
      pm3d->SetMarkerStyle(fTree->GetMarkerStyle());
      pm3d->SetMarkerColor(fTree->GetMarkerColor());
      pm3d->SetMarkerSize(fTree->GetMarkerSize());
      for (i=0;i<fNfill;i++) { pm3d->SetPoint(i,fV3[i],fV2[i],fV1[i]);}
      if (!fDraw && !strstr(fOption.Data(),"goff")) pm3d->Draw();
      if (!h3->TestBit(kCanDelete)) {
         for (i=0;i<fNfill;i++) h3->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
      }

   //__________________________2D Profile Histogram__________________
   } else if (fAction == 23) {
      TProfile2D *hp = (TProfile2D*)fObject;
      if (hp->TestBit(TH1::kCanRebin)) {
         for (i=0;i<fNfill;i++) {
            if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
            if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
            if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
            if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
            if (fVmin[2] > fV3[i]) fVmin[2] = fV3[i];
            if (fVmax[2] < fV3[i]) fVmax[2] = fV3[i];
         }
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hp,fVmin[2],fVmax[2],fVmin[1],fVmax[1]);
      }
      for (i=0;i<fNfill;i++) hp->Fill(fV3[i],fV2[i],fV1[i],fW[i]);
   //__________________________4D scatter plot_______________________
   } else if (fAction == 40) {
      TH3 *h3 = (TH3*)fObject;
      if (fObject->TestBit(TH1::kCanRebin)) {
         for (i = 0; i < 4; i++) {
            fVmin[i] = vminOld[i];
            fVmax[i] = vmaxOld[i];
         }
         for (i=0;i<fNfill;i++) {
            if (fVmin[0] > fV1[i]) fVmin[0] = fV1[i];
            if (fVmax[0] < fV1[i]) fVmax[0] = fV1[i];
            if (fVmin[1] > fV2[i]) fVmin[1] = fV2[i];
            if (fVmax[1] < fV2[i]) fVmax[1] = fV2[i];
            if (fVmin[2] > fV3[i]) fVmin[2] = fV3[i];
            if (fVmax[2] < fV3[i]) fVmax[2] = fV3[i];
            if (fVmin[3] > fV4[i]) fVmin[3] = fV4[i];
            if (fVmax[3] < fV4[i]) fVmax[3] = fV4[i];
         }
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(h3,fVmin[2],fVmax[2],fVmin[1],fVmax[1],fVmin[0],fVmax[0]);
      }
   }
}

//______________________________________________________________________________
void TSelectorDraw::Terminate()
{
   // Called at the end of a loop on a TTree.

   if (fNfill) TakeAction();

   if ((fSelectedRows == 0) && (TestBit(kCustomHistogram) == 0)) fDraw = 1; // do not draw

   SetStatus(fSelectedRows);
}
