// $Id$
// Author: Sergey Linev   22/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRootSnifferFull.h"

#include "TH1.h"
#include "TGraph.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TKey.h"
#include "TList.h"
#include "TMemFile.h"
#include "TStreamerInfo.h"
#include "TBufferFile.h"
#include "TBufferJSON.h"
#include "TBufferXML.h"
#include "TROOT.h"
#include "TTimer.h"
#include "TFolder.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TClass.h"
#include "TMethod.h"
#include "TFunction.h"
#include "TMethodArg.h"
#include "TMethodCall.h"
#include "TRealData.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TBaseClass.h"
#include "TObjString.h"
#include "TUrl.h"
#include "TImage.h"
#include "RZip.h"
#include "RVersion.h"
#include "TVirtualMutex.h"
#include "TRootSnifferStore.h"
#include "THttpCallArg.h"

#include <stdlib.h>
#include <vector>
#include <string.h>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSnifferFull                                                     //
//                                                                      //
// Subclass of TRootSniffer, which provides access to different         //
// ROOT collections and containers like TTree, TCanvas, ...             //
//////////////////////////////////////////////////////////////////////////

ClassImp(TRootSnifferFull);

////////////////////////////////////////////////////////////////////////////////
/// constructor

TRootSnifferFull::TRootSnifferFull(const char *name, const char *objpath)
   : TRootSniffer(name, objpath)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TRootSnifferFull::~TRootSnifferFull()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Method to produce image from specified object
///
/// Parameters:
///    kind - image kind TImage::kPng, TImage::kJpeg, TImage::kGif
///    path - path to object
///    options - extra options
///
/// By default, image 300x200 is produced
/// In options string one could provide following parameters:
///    w - image width
///    h - image height
///    opt - draw options
///  For instance:
///     http://localhost:8080/Files/hsimple.root/hpx/get.png?w=500&h=500&opt=lego1
///
///  Return is memory with produced image
///  Memory must be released by user with free(ptr) call

Bool_t TRootSnifferFull::ProduceImage(Int_t kind, const char *path, const char *options, void *&ptr, Long_t &length)
{
   ptr = nullptr;
   length = 0;

   if (!path || (*path == 0)) return kFALSE;
   if (*path == '/') path++;

   TClass *obj_cl(nullptr);
   void *obj_ptr = FindInHierarchy(path, &obj_cl);
   if (!obj_ptr || !obj_cl) return kFALSE;

   if (obj_cl->GetBaseClassOffset(TObject::Class()) != 0) {
      Error("TRootSniffer", "Only derived from TObject classes can be drawn");
      return kFALSE;
   }

   TObject *obj = (TObject *)obj_ptr;

   TImage *img = TImage::Create();
   if (!img) return kFALSE;

   if (obj->InheritsFrom(TPad::Class())) {

      if (gDebug > 1) Info("TRootSniffer", "Crate IMAGE directly from pad");
      img->FromPad((TPad *)obj);
   } else if (IsDrawableClass(obj->IsA())) {

      if (gDebug > 1) Info("TRootSniffer", "Crate IMAGE from object %s", obj->GetName());

      Int_t width(300), height(200);
      TString drawopt = "";

      if ((options != 0) && (*options != 0)) {
         TUrl url;
         url.SetOptions(options);
         url.ParseOptions();
         Int_t w = url.GetIntValueFromOptions("w");
         if (w > 10) width = w;
         Int_t h = url.GetIntValueFromOptions("h");
         if (h > 10) height = h;
         const char *opt = url.GetValueFromOptions("opt");
         if (opt != 0) drawopt = opt;
      }

      Bool_t isbatch = gROOT->IsBatch();
      TVirtualPad *save_gPad = gPad;

      if (!isbatch) gROOT->SetBatch(kTRUE);

      TCanvas *c1 = new TCanvas("__online_draw_canvas__", "title", width, height);
      obj->Draw(drawopt.Data());
      img->FromPad(c1);
      delete c1;

      if (!isbatch) gROOT->SetBatch(kFALSE);
      gPad = save_gPad;

   } else {
      delete img;
      return kFALSE;
   }

   TImage *im = TImage::Create();
   im->Append(img);

   char *png_buffer = nullptr;
   int size(0);

   im->GetImageBuffer(&png_buffer, &size, (TImage::EImageFileTypes)kind);

   if (png_buffer && (size > 0)) {
      ptr = malloc(size);
      length = size;
      memcpy(ptr, png_buffer, length);
   }

   delete[] png_buffer;
   delete im;

   return ptr != nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// produce XML data for specified item
/// For object conversion TBufferXML is used

Bool_t TRootSnifferFull::ProduceXml(const char *path, const char * /*options*/, TString &res)
{
   if (!path || (*path == 0)) return kFALSE;

   if (*path == '/') path++;

   TClass *obj_cl = nullptr;
   void *obj_ptr = FindInHierarchy(path, &obj_cl);
   if (!obj_ptr || !obj_cl) return kFALSE;

   res = TBufferXML::ConvertToXML(obj_ptr, obj_cl);

   return res.Length() > 0;
}


