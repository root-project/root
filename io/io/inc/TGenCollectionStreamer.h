// @(#)root/io:$Id$
// Author: Markus Frank  28/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGenCollectionStreamer
#define ROOT_TGenCollectionStreamer

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGenCollectionStreamer
//
// Streamer around an arbitrary STL like container, which implements basic
// container functionality.
//
//////////////////////////////////////////////////////////////////////////

#include "TGenCollectionProxy.h"
#include "TCollectionProxyFactory.h"

class TGenCollectionStreamer : public TGenCollectionProxy {

protected:
   void ReadMapHelper(StreamHelper *i, Value *v, Bool_t vsn3,  TBuffer &b);
   void ReadMap(int nElements, TBuffer &b, const TClass *onfileClass);
   void ReadPairFromMap(int nElements, TBuffer &b);
   void ReadObjects(int nElements, TBuffer &b, const TClass *onfileClass);
   void ReadPrimitives(int nElements, TBuffer &b, const TClass *onfileClass);
   void WriteMap(int nElements, TBuffer &b);
   void WriteObjects(int nElements, TBuffer &b);
   void WritePrimitives(int nElements, TBuffer &b);

//   typedef void (TGenCollectionStreamer::*ReadBufferConv_t)(TBuffer &b, void *obj, const TClass *onFileClass);
//   ReadBufferConv_t fReadBufferConvFunc;

   typedef void (TGenCollectionStreamer::*ReadBuffer_t)(TBuffer &b, void *obj, const TClass *onFileClass);
   ReadBuffer_t fReadBufferFunc;

   template <typename From, typename To> void ConvertBufferVectorPrimitives(TBuffer &b, void *obj, Int_t nElements);
   template <typename To> void ConvertBufferVectorPrimitivesFloat16(TBuffer &b, void *obj, Int_t nElements);
   template <typename To> void ConvertBufferVectorPrimitivesDouble32(TBuffer &b, void *obj, Int_t nElements);
   template <typename To> void DispatchConvertBufferVectorPrimitives(TBuffer &b, void *obj, Int_t nElements, const TVirtualCollectionProxy *onfileProxy);
   template <typename basictype> void ReadBufferVectorPrimitives(TBuffer &b, void *obj, const TClass *onFileClass);
   void ReadBufferVectorPrimitivesFloat16(TBuffer &b, void *obj, const TClass *onFileClass);
   void ReadBufferVectorPrimitivesDouble32(TBuffer &b, void *obj, const TClass *onFileClass);
   void ReadBufferDefault(TBuffer &b, void *obj, const TClass *onFileClass);
   void ReadBufferGeneric(TBuffer &b, void *obj, const TClass *onFileClass);

private:
   TGenCollectionStreamer &operator=(const TGenCollectionStreamer&); // Not implemented.

public:
   // Virtual copy constructor
   virtual TVirtualCollectionProxy* Generate() const;

   // Copy constructor
   TGenCollectionStreamer(const TGenCollectionStreamer& copy);

   // Initializing constructor
   TGenCollectionStreamer(Info_t typ, size_t iter_size);
   TGenCollectionStreamer(const ROOT::TCollectionProxyInfo &info, TClass *cl);

   // Standard destructor
   virtual ~TGenCollectionStreamer();

   // Streamer I/O overload
   virtual void Streamer(TBuffer &refBuffer);

   // Streamer I/O overload
   virtual void StreamerAsMap(TBuffer &refBuffer);

   // Streamer I/O overload
   virtual void Streamer(TBuffer &buff, void *pObj, int siz)  {
      TGenCollectionProxy::Streamer(buff, pObj, siz);
   }

   // Routine to read the content of the buffer into 'obj'.
   virtual void ReadBuffer(TBuffer &b, void *obj, const TClass *onfileClass);

   // Routine to read the content of the buffer into 'obj'.
   virtual void ReadBuffer(TBuffer &b, void *obj);
};

template <typename T>
struct AnyCollectionStreamer : public TGenCollectionStreamer  {
   AnyCollectionStreamer()
      : TGenCollectionStreamer(typeid(T::Cont_t),sizeof(T::Iter_t))  {
      fValDiff        = sizeof(T::Value_t);
      fValOffset      = T::value_offset();
      fSize.call      = T::size;
      fFirst.call     = T::first;
      fNext.call      = T::next;
      fClear.call     = T::clear;
      fResize         = T::resize;
      fCollect        = T::collect;
      fConstruct      = T::construct;
      fDestruct       = T::destruct;
      fFeed           = T::feed;
      CheckFunctions();
   }
   virtual ~AnyCollectionStreamer() {  }
};

// Forward declaration in the event of later seperation
typedef TGenCollectionStreamer TGenMapStreamer;

#endif
