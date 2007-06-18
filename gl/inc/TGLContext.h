// @(#)root/gl:$Name:  $:$Id: TGLContext.h,v 1.5 2007/06/18 07:02:16 brun Exp $
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

class TGLPaintDevice;
//class TGLPBuffer;
class TGLWidget;

class TGLContext {
   class TGLContextPrivate;
   friend class TGLContextPrivate; // for solaris cc
   friend class TGLWidget;
//   friend class TGLPBuffer;
private:
   TGLPaintDevice *fDevice;
   class TGLContextPrivate;
   TGLContextPrivate *fPimpl;

   Bool_t fFromCtor;//To prohibit user's calls of SetContext.
   Bool_t fValid;

   TGLContextIdentity *fIdentity;

public:
   TGLContext(TGLWidget *glWidget, const TGLContext *shareList = 0);//2
//   TGLContext(TGLPBuffer *glPbuf, const TGLContext *shareList = 0);//2

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
   TGLContextIdentity() : fCnt(1), fClientCnt(0) {}
   virtual ~TGLContextIdentity() {}

   void AddRef()  { ++fCnt; }
   void Release() { --fCnt; CheckDestroy(); }

   void AddClientRef()  { ++fClientCnt; }
   void ReleaseClient() { --fClientCnt; CheckDestroy(); }

   Int_t GetRefCnt()       const { return fCnt; }
   Int_t GetClientRefCnt() const { return fClientCnt; }

   Bool_t IsValid() const { return fCnt > 0; }

   void RegisterDLNameRangeToWipe(UInt_t base, Int_t size);
   void DeleteDisplayLists();

   static TGLContextIdentity *GetCurrent();

private:
   Int_t fCnt;
   Int_t fClientCnt;

   void CheckDestroy() { if (fCnt <= 0 && fClientCnt <= 0) delete this; }

   typedef std::pair<UInt_t, Int_t>  DLRange_t;
   typedef std::list<DLRange_t>      DLTrash_t;
   typedef DLTrash_t::const_iterator DLTrashIt_t;

   DLTrash_t fDLTrash;

   ClassDef(TGLContextIdentity, 0) // Identity of a shared GL context.
};

#endif
