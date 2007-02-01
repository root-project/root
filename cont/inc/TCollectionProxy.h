// @(#)root/cont:$Name:  $:$Id: TCollectionProxy.h,v 1.17 2007/01/16 14:31:49 brun Exp $
// Author: Markus Frank  28/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TCollectionProxy
#define ROOT_TCollectionProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Small helper to save proxy environment in the event of
//  recursive calls.
//
//////////////////////////////////////////////////////////////////////////

#include <typeinfo>
#include <vector>

#ifndef ROOT_TCollectionProxyInfo
#include "TCollectionProxyInfo.h"
#endif

#ifndef ROOT_TClassStreamer
#include "TClassStreamer.h"
#endif
#ifndef ROOT_TMemberStreamer
#include "TMemberStreamer.h"
#endif

// Forward declarations
class TBuffer;
class TGenCollectionProxy;
class TGenCollectionStreamer;
class TVirtualCollectionProxy;
class TEmulatedCollectionProxy;

#if defined(_WIN32)
   #if _MSC_VER<1300
      #define TYPENAME
      #define R__VCXX6
   #else
      #define TYPENAME typename
   #endif
#else
   #define TYPENAME typename
#endif


/** @class TCollectionProxy TCollectionProxy.h cont/TCollectionProxy.h
  *
  * TCollectionProxy
  * Interface to collection proxy and streamer generator.
  *
  * Proxy around an arbitrary container, which implements basic
  * functionality and iteration. The purpose of this implementation
  * is to shield any generated dictionary implementation from the
  * underlying streamer/proxy implementation and only expose
  * the creation fucntions.
  *
  * In particular this is used to implement splitting and abstract
  * element access of any container. Access to compiled code is necessary
  * to implement the abstract iteration sequence and functionality like
  * size(), clear(), resize(). resize() may be a void operation.
  *
  * @author  M.Frank
  * @version 1.0
  */
class TCollectionProxy  {
public:

   typedef TVirtualCollectionProxy Proxy_t;
#ifdef R__HPUX
   typedef const type_info&      Info_t;
#else
   typedef const std::type_info& Info_t;
#endif




   /// Generate emulated collection proxy for a given class
   static TVirtualCollectionProxy* GenEmulatedProxy(const char* class_name);

   /// Generate emulated class streamer for a given collection class
   static TClassStreamer* GenEmulatedClassStreamer(const char* class_name);

   /// Generate emulated member streamer for a given collection class
   static TMemberStreamer* GenEmulatedMemberStreamer(const char* class_name);


   /// Generate proxy from static functions
    static Proxy_t* GenExplicitProxy( const ::ROOT::TCollectionProxyInfo &info );

   /// Generate proxy from template
   template <class T> static Proxy_t* GenProxy(const T &arg)  {      
      return GenExplicitProxy( ::ROOT::TCollectionProxyInfo::Get(arg) ); 
   }

   /// Generate streamer from static functions
   static TGenCollectionStreamer*
      GenExplicitStreamer( const ::ROOT::TCollectionProxyInfo &info );

   /// Generate class streamer from static functions
   static TClassStreamer*
      GenExplicitClassStreamer( const ::ROOT::TCollectionProxyInfo &info );

   /// Generate class streamer from template
   template <class T> static TClassStreamer* GenClassStreamer(const T &arg)  {
      return GenExplicitClassStreamer(::ROOT::TCollectionProxyInfo::Get(arg));
   }

   /// Generate member streamer from static functions
   static TMemberStreamer*
      GenExplicitMemberStreamer(const ::ROOT::TCollectionProxyInfo &info);

   /// Generate member streamer from template
   template <class T> static TMemberStreamer* GenMemberStreamer(const T &arg)  {
      return GenExplicitMemberStreamer(::ROOT::TCollectionProxyInfo::Get(arg));
   }
};

/** @class TCollectionStreamer TCollectionProxy.h cont/TCollectionProxy.h
 *
 * TEmulatedClassStreamer
 *
 * Class streamer object to implement TClassStreamr functionality
 * for I/O emulation.
 *
 * @author  M.Frank
 * @version 1.0
 */
class TCollectionStreamer   {
private:
   TCollectionStreamer& operator=(const TCollectionStreamer&);   // not implemented

protected:
   TGenCollectionProxy* fStreamer;   /// Pointer to worker streamer

   /// Issue Error about invalid proxy
   void InvalidProxyError();

public:
   /// Initializing constructor
   TCollectionStreamer();
   /// Copy constructor
   TCollectionStreamer(const TCollectionStreamer& c);
   /// Standard destructor
   virtual ~TCollectionStreamer();
   /// Attach worker proxy
   void AdoptStreamer(TGenCollectionProxy* streamer);
   /// Streamer for I/O handling
   void Streamer(TBuffer &refBuffer, void *pObject, int siz);
};

/** @class TEmulatedClassStreamer TCollectionProxy.h cont/TCollectionProxy.h
  *
  * TEmulatedClassStreamer
  *
  * Class streamer object to implement TClassStreamr functionality
  * for I/O emulation.
  *
  * @author  M.Frank
  * @version 1.0
  */
class TCollectionClassStreamer : public TClassStreamer, public TCollectionStreamer {
public:
   /// Initializing constructor
   TCollectionClassStreamer() : TClassStreamer(0)     {                        }
   /// Copy constructor
   TCollectionClassStreamer(const TCollectionClassStreamer& c)
      : TClassStreamer(c), TCollectionStreamer(c)      {                        }
   /// Standard destructor
   virtual ~TCollectionClassStreamer()                {                        }
   /// Streamer for I/O handling
   virtual void operator()(TBuffer &buff, void *pObj) { Streamer(buff,pObj,0); }

   /// Virtual copy constructor.
   virtual TClassStreamer *Generate() {
      return new TCollectionClassStreamer(*this);
   }

};

/** @class TCollectionMemberStreamer TCollectionProxy.h cont/TCollectionProxy.h
  *
  * TCollectionMemberStreamer
  *
  * Class streamer object to implement TMemberStreamer functionality
  * for I/O emulation.
  *
  * @author  M.Frank
  * @version 1.0
  */
class TCollectionMemberStreamer : public TMemberStreamer, public TCollectionStreamer {
public:
   /// Initializing constructor
   TCollectionMemberStreamer() : TMemberStreamer(0) { }
   /// Copy constructor
   TCollectionMemberStreamer(const TCollectionMemberStreamer& c)
      : TMemberStreamer(c), TCollectionStreamer(c)   { }
   /// Standard destructor
   virtual ~TCollectionMemberStreamer()             { }
   /// Streamer for I/O handling
   virtual void operator()(TBuffer &buff,void *pObj,Int_t siz=0)
   { Streamer(buff, pObj, siz);                       }
};

#endif
