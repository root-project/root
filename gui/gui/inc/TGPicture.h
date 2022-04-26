// @(#)root/gui:$Id$
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


#include "TObject.h"
#include "TRefCnt.h"
#include "TString.h"
#include "TGClient.h"
#include "TGGC.h"

class THashTable;


class TGPicture : public TObject, public TRefCnt {

friend class TGPicturePool;

protected:
   TString             fName;       ///< name of picture
   Bool_t              fScaled;     ///< kTRUE if picture is scaled
   Pixmap_t            fPic;        ///< picture pixmap
   Pixmap_t            fMask;       ///< picture mask pixmap
   PictureAttributes_t fAttributes; ///< picture attributes

   TGPicture(const char *name, Bool_t scaled = kFALSE):
      fName(name), fScaled(scaled), fPic(kNone), fMask(kNone), fAttributes()
   {
      fAttributes.fPixels = 0;
      SetRefCount(1);
   }

   TGPicture(const char *name, Pixmap_t pxmap, Pixmap_t mask = 0);

   // override default of TObject
   void Draw(Option_t * = "") override { MayNotUse("Draw(Option_t*)"); }

public:
   virtual ~TGPicture();

   const char *GetName() const override { return fName.Data(); }
   UInt_t      GetWidth() const { return fAttributes.fWidth; }
   UInt_t      GetHeight() const { return fAttributes.fHeight; }
   Pixmap_t    GetPicture() const { return fPic; }
   Pixmap_t    GetMask() const { return fMask; }
   Bool_t      IsScaled() const { return fScaled; }
   ULong_t     Hash() const override { return fName.Hash(); }
   static const char *HashName(const char *name, Int_t width, Int_t height);

   virtual void Draw(Handle_t id, GContext_t gc, Int_t x, Int_t y) const;
   void         Print(Option_t *option="") const override;

   ClassDefOverride(TGPicture,0)  // Pictures and icons used by the GUI classes
};


class TGSelectedPicture : public TGPicture {

protected:
   const TGClient *fClient;    // client to which selected picture belongs

   static TGGC *fgSelectedGC;
   static TGGC &GetSelectedGC();

   TGSelectedPicture(const TGSelectedPicture& gp):
     TGPicture(gp), fClient(gp.fClient) { }
   TGSelectedPicture& operator=(const TGSelectedPicture& gp)
     {if(this!=&gp) { TGPicture::operator=(gp); fClient=gp.fClient;}
     return *this;}

public:
   TGSelectedPicture(const TGClient *client, const TGPicture *p);
   virtual ~TGSelectedPicture();

   ClassDefOverride(TGSelectedPicture,0)  // Selected looking picture
};


class TGPicturePool : public TObject {

protected:
   const TGClient    *fClient;    ///< client for which we keep icon pool
   TString            fPath;      ///< icon search path
   THashTable        *fPicList;   ///< hash table containing the icons

   TGPicturePool(const TGPicturePool&);
   TGPicturePool& operator=(const TGPicturePool&);

public:
   TGPicturePool(const TGClient *client, const char *path):
      fClient(client), fPath(path), fPicList(nullptr) { }
   virtual ~TGPicturePool();

   const char      *GetPath() const { return fPath; }
   const TGPicture *GetPicture(const char *name);
   const TGPicture *GetPicture(const char *name, char **xpm);
   const TGPicture *GetPicture(const char *name, UInt_t new_width, UInt_t new_height);
   const TGPicture *GetPicture(const char *name, Pixmap_t pxmap, Pixmap_t mask =  0);
   void             FreePicture(const TGPicture *pic);

   void             Print(Option_t *option="") const override;

   ClassDefOverride(TGPicturePool,0)  // Picture and icon cache
};

#endif
