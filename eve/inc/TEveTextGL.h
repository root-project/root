// @(#)root/eve:$Id$
// Authors: Alja & Matevz Tadel 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTextGL
#define ROOT_TEveTextGL

#include "TGLObject.h"

class TEveText;
class FTFont;

class TEveTextGL : public TGLObject
{
private:
   TEveTextGL(const TEveTextGL&);            // Not implemented
   TEveTextGL& operator=(const TEveTextGL&); // Not implemented

protected:
   mutable Int_t         fSize;     // current font size
   mutable Int_t         fFile;     // current font file
   mutable Int_t         fMode;     // current FTGL class

   mutable FTFont       *fFont;     // FTGL font object

   TEveText             *fM;        // model object.
   mutable Double_t      fX[4][3];  // 3D position of font

   void    SetModelFont( TEveText* model, TGLRnrCtx & rnrCtx) const;
public:
   TEveTextGL();
   virtual ~TEveTextGL() {}

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();

   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const;

   ClassDef(TEveTextGL, 0); // GL renderer class for TEveText.
};

#endif
