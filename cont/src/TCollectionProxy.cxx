// @(#)root/cont:$Name:  $:$Id: TCollectionProxy.cxx,v 1.6 2005/11/16 20:07:50 pcanal Exp $
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
   static TClassEdit::ESTLType stl_type(const std::string& class_name)  {
      // return the STL type.
      int nested = 0;
      std::vector<std::string> inside;
      int num = TClassEdit::GetSplit(class_name.c_str(),inside,nested);
      if ( num > 1 )  {
         return (TClassEdit::ESTLType)TClassEdit::STLKind(inside[0].c_str());
      }
      return TClassEdit::kNotSTL;
   }

   static TEmulatedCollectionProxy* GenEmulation(const char* class_name)  {
      // Generate an emulated collection proxy.

      if ( class_name )  {
         std::string cl = class_name;
         if ( cl.find("stdext::hash_") != std::string::npos )
            cl.replace(3,10,"::");
         if ( cl.find("__gnu_cxx::hash_") != std::string::npos )
            cl.replace(0,16,"std::");
         switch ( stl_type(cl) )  {
            case TClassEdit::kNotSTL:
               return 0;
            case TClassEdit::kMap:
            case TClassEdit::kMultiMap:
               return new TEmulatedMapProxy(class_name);
            default:
               return new TEmulatedCollectionProxy(class_name);
         }
      }
      return 0;
   }
}

TVirtualCollectionProxy*
TCollectionProxy::GenEmulatedProxy(const char* class_name)
{
   // Generate emulated collection proxy for a given class.

   return GenEmulation(class_name);
}

TClassStreamer*
TCollectionProxy::GenEmulatedClassStreamer(const char* class_name)
{
   // Generate emulated class streamer for a given collection class.

   TCollectionClassStreamer* s = new TCollectionClassStreamer();
   s->AdoptStreamer(GenEmulation(class_name));
   return s;
}

TMemberStreamer*
TCollectionProxy::GenEmulatedMemberStreamer(const char* class_name)
{
   // Generate emulated member streamer for a given collection class.

   TCollectionMemberStreamer* s = new TCollectionMemberStreamer();
   s->AdoptStreamer(GenEmulation(class_name));
   return s;
}

TCollectionProxy::Proxy_t*
TCollectionProxy::GenExplicitProxy( Info_t info,
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
   // Generate proxy from static functions.
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

TGenCollectionStreamer*
TCollectionProxy::GenExplicitStreamer(  Info_t  info,
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
   // Generate streamer from static functions.
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

TClassStreamer*
TCollectionProxy::GenExplicitClassStreamer( Info_t info,
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
   // Generate class streamer from static functions.
   TCollectionClassStreamer* s = new TCollectionClassStreamer();
   s->AdoptStreamer(GenExplicitStreamer(info,
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

TMemberStreamer*
TCollectionProxy::GenExplicitMemberStreamer(Info_t info,
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
   // Generate member streamer from static functions.
   TCollectionMemberStreamer* s = new TCollectionMemberStreamer();
   s->AdoptStreamer(GenExplicitStreamer(info,
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

void TCollectionStreamer::InvalidProxyError()   {
   // Issue Error about invalid proxy.
   Fatal("TCollectionStreamer>","No proxy available. Data streaming impossible.");
}

TCollectionStreamer::TCollectionStreamer() : fStreamer(0)
{
   // Initializing constructor.
}

TCollectionStreamer::TCollectionStreamer(const TCollectionStreamer& c)
{
   // Copy constructor.
   if ( c.fStreamer )  {
      fStreamer = dynamic_cast<TGenCollectionProxy*>(c.fStreamer->Generate());
      R__ASSERT(fStreamer != 0);
      return;
   }
   InvalidProxyError();
}

TCollectionStreamer::~TCollectionStreamer()
{
   // Standard destructor.
   if ( fStreamer )  {
      delete fStreamer;
   }
}

void TCollectionStreamer::AdoptStreamer(TGenCollectionProxy* streamer)
{
   // Attach worker proxy.
   if ( fStreamer )  {
      delete fStreamer;
   }
   fStreamer = streamer;
}

void TCollectionStreamer::Streamer(TBuffer &buff, void *pObj, int /* siz */ )
{
   // Streamer for I/O handling.
   if ( fStreamer )  {
      TVirtualCollectionProxy::TPushPop env(fStreamer, pObj);
      fStreamer->Streamer(buff);
      return;
   }
   InvalidProxyError();
}
