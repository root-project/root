// @(#)root/gui:$Name:  $:$Id: TGPicture.h,v 1.2 2000/09/29 08:57:05 rdm Exp $
// Author: Fons Rademakers   01/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGPicture
#define ROOT_TGPicture


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGPicture, TGSelectdPicture & TGPicturePool                          //
//                                                                      //
// The TGPicture class implements pictures and icons used in the        //
// different GUI elements and widgets. The TGPicturePool class          //
// implements a TGPicture cache. TGPictures are created, managed and    //
// destroyed by the TGPicturePool.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TRefCnt
#include "TRefCnt.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TGClient
#include "TGClient.h"
#endif
#ifndef ROOT_TGGC
#include "TGGC.h"
#endif

class THashTable;


class TGPicture : public TObject, public TRefCnt {

friend class TGPicturePool;

protected:
   TString             fName;       // name of picture
   Bool_t              fScaled;     // kTRUE if picture is scaled
   Pixmap_t            fPic;        // picture pixmap
   Pixmap_t            fMask;       // picture mask pixmap
   PictureAttributes_t fAttributes; // picture attributes

   TGPicture(const char *name, Bool_t scaled = kFALSE) {
      fName   = name;
      fScaled = scaled;
      fPic    = kNone;
      fMask   = kNone;
      fAttributes.fPixels = 0;
      SetRefCount(1);
   }

   // override default of TObject
   void Draw(Option_t * = "") { MayNotUse("Draw(Option_t*)"); }

   static const char *HashName(const char *name, Int_t width, Int_t height);

public:
   virtual ~TGPicture();

   const char *GetName() const { return fName.Data(); }
   UInt_t      GetWidth() const { return fAttributes.fWidth; }
   UInt_t      GetHeight() const { return fAttributes.fHeight; }
   Pixmap_t    GetPicture() const { return fPic; }
   Pixmap_t    GetMask() const { return fMask; }
   Bool_t      IsScaled() const { return fScaled; }
   ULong_t     Hash() const { return fName.Hash(); }

   virtual void Draw(Handle_t id, GContext_t gc, Int_t x, Int_t y) const;

   ClassDef(TGPicture,0)  // Pictures and icons used by the GUI classes
};


class TGSelectedPicture : public TGPicture {

friend class TGClient;

protected:
   const TGClient *fClient;    // client to which selected picture belongs

   static TGGC fgSelectedGC;

public:
   TGSelectedPicture(const TGClient *client, const TGPicture *p);
   virtual ~TGSelectedPicture();

   ClassDef(TGSelectedPicture,0)  // Selected looking picture
};


class TGPicturePool : public TObject {

friend class TGClient;

protected:
   const TGClient    *fClient;    // client for which we keep icon pool
   TString            fPath;      // icon search path
   THashTable        *fPicList;   // hash table containing the icons

   static Colormap_t  fgDefaultColormap;

public:
   TGPicturePool(const TGClient *client, const char *path) {
      fClient  = client;
      fPath    = path;
      fPicList = 0;
   }
   virtual ~TGPicturePool();

   const char      *GetPath() const { return fPath.Data(); }
   const TGPicture *GetPicture(const char *name);
   const TGPicture *GetPicture(const char *name, UInt_t new_width, UInt_t new_height);
   void             FreePicture(const TGPicture *pic);

   ClassDef(TGPicturePool,0)  // Picture and icon cache
};

#endif
