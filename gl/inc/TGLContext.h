// @(#)root/gl:$Name:  $:$Id: TGLContext.h,v 1.7 2007/06/18 10:58:33 brun Exp $
// Author:  Timur Pocheptsov, Jun 2007

#include <utility>
#include <list>

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLContext
#define ROOT_TGLContext

class TGLContextIdentity;

#ifndef ROOT_TGLFormat
#include "TGLFormat.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TGLContextPrivate;
class TGLPaintDevice;
//class TGLPBuffer;
class TGLWidget;

class TGLContext {
   friend class TGLContextPrivate;
   friend class TGLWidget;
//   friend class TGLPBuffer;
private:
   TGLPaintDevice *fDevice;
   TGLContextPrivate *fPimpl;

   Bool_t fFromCtor;//To prohibit user's calls of SetContext.
   Bool_t fValid;

   TGLContextIdentity *fIdentity;

public:
   TGLContext(TGLWidget *glWidget);
   TGLContext(TGLWidget *glWidget, const TGLContext *shareList);
//   TGLContext(TGLPBuffer *glPbuf, const TGLContext *shareList = 0);

   TGLContextIdentity *GetIdentity()const;

   virtual ~TGLContext();

   Bool_t           MakeCurrent();
   void             SwapBuffers();

   //This functions are public _ONLY_ for calls via
   //gROOT under win32. Please, DO NOT CALL IT DIRECTLY.
   void             SetContext(TGLWidget *widget, const TGLContext *shareList);
//   void             SetContextPB(TGLPBuffer *pbuff, const TGLContext *shareList);
   void             Release();

   Bool_t           IsValid()const
   {
      return fValid;
   }

   static TGLContext *GetCurrent();

private:
   TGLContext(const TGLContext &);
   TGLContext &operator = (const TGLContext &);

   ClassDef(TGLContext, 0)//This class controls internal gl-context resources.
};


//______________________________________________________________________________

class TGLContextIdentity {
public:
   TGLContextIdentity() : fCnt(0), fClientCnt(0) {}
   virtual ~TGLContextIdentity() {}

   void AddRef(TGLContext* ctx);
   void Release(TGLContext* ctx);

   void AddClientRef()  { ++fClientCnt; }
   void ReleaseClient() { --fClientCnt; CheckDestroy(); }

   Int_t GetRefCnt()       const { return fCnt; }
   Int_t GetClientRefCnt() const { return fClientCnt; }

   Bool_t IsValid() const { return fCnt > 0; }

   void RegisterDLNameRangeToWipe(UInt_t base, Int_t size);
   void DeleteDisplayLists();

   static TGLContextIdentity *GetCurrent();

   static TGLContextIdentity *GetDefaultIdentity();
   static TGLContext         *GetDefaultContextAny();

private:
   Int_t fCnt;
   Int_t fClientCnt;

   void CheckDestroy();

   typedef std::pair<UInt_t, Int_t>  DLRange_t;
   typedef std::list<DLRange_t>      DLTrash_t;
   typedef DLTrash_t::const_iterator DLTrashIt_t;

   typedef std::list<TGLContext*>    CtxList_t;

   DLTrash_t fDLTrash;
   CtxList_t fCtxs;

   static TGLContextIdentity * fgDefaultIdentity;

   ClassDef(TGLContextIdentity, 0) // Identity of a shared GL context.
};

#endif
