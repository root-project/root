// @(#)root/cont:$Name:  $:$Id: TCollectionProxy.cxx,v 1.2 2004/11/01 12:26:07 brun Exp $
// Author: Markus Frank 28/10/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGenCollectionProxy
//
// Proxy around an arbitrary container, which implements basic 
// functionality and iteration. The purpose of this implementation 
// is to shield any generated dictionary implementation from the
// underlying streamer/proxy implementation and only expose
// the creation fucntions.
//
// In particular this is used to implement splitting and abstract
// element access of any container. Access to compiled code is necessary
// to implement the abstract iteration sequence and functionality like
// size(), clear(), resize(). resize() may be a void operation.
//
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TClassEdit.h"
#include "TCollectionProxy.h"
#include "TGenCollectionProxy.h"
#include "TGenCollectionStreamer.h"

#include "TEmulatedMapProxy.h"
#include "TEmulatedCollectionProxy.h"

// Do not clutter global namespace with shit....
namespace {
  static TClassEdit::ESTLType stl_type(const char* class_name)  {
    if ( class_name )  {
      int nested = 0;
      std::vector<std::string> inside;
      int num = TClassEdit::GetSplit(class_name,inside,nested);
      if ( num > 1 )  {
        return (TClassEdit::ESTLType)TClassEdit::STLKind(inside[0].c_str());
      }
    }
    return TClassEdit::kNotSTL;
  }

  static TEmulatedCollectionProxy* genEmulation(const char* class_name)  {
    switch ( stl_type(class_name) )  {
      case TClassEdit::kNotSTL:
        return 0;
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
        return new TEmulatedMapProxy(class_name);
      default:
        return new TEmulatedCollectionProxy(class_name);
    }
    return 0;
  }
}

/// Generate emulated collection proxy for a given class
TVirtualCollectionProxy* 
TCollectionProxy::genEmulatedProxy(const char* class_name)  
{
  return genEmulation(class_name);
}

/// Generate emulated class streamer for a given collection class
TClassStreamer* 
TCollectionProxy::genEmulatedClassStreamer(const char* class_name)
{
  TCollectionClassStreamer* s = new TCollectionClassStreamer();
  s->AdoptStreamer(genEmulation(class_name));
  return s;
}

/// Generate emulated member streamer for a given collection class
TMemberStreamer* 
TCollectionProxy::genEmulatedMemberStreamer(const char* class_name)
{
  TCollectionMemberStreamer* s = new TCollectionMemberStreamer();
  s->AdoptStreamer(genEmulation(class_name));
  return s;
}

/// Generate proxy from static functions
TCollectionProxy::Proxy_t* 
TCollectionProxy::genExplicitProxy( Info_t info,
                                    size_t iter_size,
                                    size_t value_diff,
                                    int    value_offset,
                                    void*  (*size_func)(void*),
                                    void*  (*resize_func)(void*),
                                    void*  (*clear_func)(void*),
                                    void*  (*first_func)(void*),
                                    void*  (*next_func)(void*),
                                    void*  (*construct_func)(void*),
                                    void*  (*destruct_func)(void*),
                                    void*  (*feed_func)(void*),
                                    void*  (*collect_func)(void*)
                                    )
{
  TGenCollectionProxy* ptr = new TGenCollectionProxy(info, iter_size);
  ptr->fValDiff        = value_diff;
  ptr->fValOffset      = value_offset;
  ptr->fSize.call      = size_func;
  ptr->fResize.call    = resize_func;
  ptr->fNext.call      = next_func;
  ptr->fFirst.call     = first_func;
  ptr->fClear.call     = clear_func;
  ptr->fConstruct.call = construct_func;
  ptr->fDestruct.call  = destruct_func;
  ptr->fFeed.call      = feed_func;
  ptr->fCollect.call   = collect_func;
  ptr->CheckFunctions();
  return ptr;
}

/// Generate streamer from static functions
TGenCollectionStreamer* 
TCollectionProxy::genExplicitStreamer(  Info_t  info,
                                        size_t  iter_size,
                                        size_t  value_diff,
                                        int     value_offset,
                                        void*  (*size_func)(void*),
                                        void*  (*resize_func)(void*),
                                        void*  (*clear_func)(void*),
                                        void*  (*first_func)(void*),
                                        void*  (*next_func)(void*),
                                        void*  (*construct_func)(void*),
                                        void*  (*destruct_func)(void*),
                                        void*  (*feed_func)(void*),
                                        void*  (*collect_func)(void*)
                                        )
{
  TGenCollectionStreamer* ptr = new TGenCollectionStreamer(info, iter_size);
  ptr->fValDiff        = value_diff;
  ptr->fValOffset      = value_offset;
  ptr->fSize.call      = size_func;
  ptr->fResize.call    = resize_func;
  ptr->fNext.call      = next_func;
  ptr->fFirst.call     = first_func;
  ptr->fClear.call     = clear_func;
  ptr->fConstruct.call = construct_func;
  ptr->fDestruct.call  = destruct_func;
  ptr->fFeed.call      = feed_func;
  ptr->fCollect.call   = collect_func;
  ptr->CheckFunctions();
  return ptr;
}

/// Generate class streamer from static functions
TClassStreamer* 
TCollectionProxy::genExplicitClassStreamer( Info_t info,
                                            size_t iter_size,
                                            size_t value_diff,
                                            int    value_offset,
                                            void*  (*size_func)(void*),
                                            void*  (*resize_func)(void*),
                                            void*  (*clear_func)(void*),
                                            void*  (*first_func)(void*),
                                            void*  (*next_func)(void*),
                                            void*  (*construct_func)(void*),
                                            void*  (*destruct_func)(void*),
                                            void*  (*feed_func)(void*),
                                            void*  (*collect_func)(void*)
                                            )
{
  TCollectionClassStreamer* s = new TCollectionClassStreamer();
  s->AdoptStreamer(genExplicitStreamer(info, 
                                    iter_size,
                                    value_diff,
                                    value_offset,
                                    size_func,
                                    resize_func,
                                    clear_func,
                                    first_func,
                                    next_func,
                                    construct_func,
                                    destruct_func,
                                    feed_func,
                                    collect_func));
  return s;
}

/// Generate member streamer from static functions
TMemberStreamer* 
TCollectionProxy::genExplicitMemberStreamer(Info_t info,
                                            size_t iter_size,
                                            size_t value_diff,
                                            int    value_offset,
                                            void*  (*size_func)(void*),
                                            void*  (*resize_func)(void*),
                                            void*  (*clear_func)(void*),
                                            void*  (*first_func)(void*),
                                            void*  (*next_func)(void*),
                                            void*  (*construct_func)(void*),
                                            void*  (*destruct_func)(void*),
                                            void*  (*feed_func)(void*),
                                            void*  (*collect_func)(void*)
                                            )
{
  TCollectionMemberStreamer* s = new TCollectionMemberStreamer();
  s->AdoptStreamer(genExplicitStreamer(info, 
                                    iter_size,
                                    value_diff,
                                    value_offset,
                                    size_func,
                                    resize_func,
                                    clear_func,
                                    first_func,
                                    next_func,
                                    construct_func,
                                    destruct_func,
                                    feed_func,
                                    collect_func));
  return s;
}

/// Issue Error about invalid proxy
void TCollectionStreamer::InvalidProxyError()   {
  Fatal("TCollectionStreamer>","No proxy available. Data streaming impossible.");
}

/// Initializing constructor
TCollectionStreamer::TCollectionStreamer() : fStreamer(0) {       
}

/// Copy constructor
TCollectionStreamer::TCollectionStreamer(const TCollectionStreamer& c)  {
  if ( c.fStreamer )  {
    fStreamer = dynamic_cast<TGenCollectionProxy*>(c.fStreamer->Generate());
    Assert(fStreamer != 0);
    return;
  }
  InvalidProxyError();
}

/// Standard destructor
TCollectionStreamer::~TCollectionStreamer()    {       
  if ( fStreamer )  {
    delete fStreamer;
  }
}

/// Attach worker proxy
void TCollectionStreamer::AdoptStreamer(TGenCollectionProxy* streamer)  {
  if ( fStreamer )  {
    delete fStreamer;
  }
  fStreamer = streamer;
}

/// Streamer for I/O handling
void TCollectionStreamer::Streamer(TBuffer &buff, void *pObj, int /* siz */ ) {
  if ( fStreamer )  {
    TVirtualCollectionProxy::TPushPop env(fStreamer, pObj);
    fStreamer->Streamer(buff);
    return;
  }
  InvalidProxyError();
}
