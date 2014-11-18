#ifndef __XRDOUCCACHE_HH__
#define __XRDOUCCACHE_HH__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c C a c h e . h h                         */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdSys/XrdSysPthread.hh"
  
/* The classes defined here can be used to implement a general cache for
   data from an arbitrary source (e.g. files, sockets, etc); as follows:

   1. Create an instance of XrdOucCacheIO. This object is used to actually
      bring in missing data into the cache or write out dirty cache pages.
      There can be many instances of this class, as needed. However, make sure
      that there is a 1-to-1 unique correspondence between data and its CacheIO
      object. Violating this may cause the same data to be cached multiple
      times and if the cache is writable the data may be inconsistent!

   2. Create an instance of XrdOucCache. You can specify various cache
      handling parameters (see the class definition). You can also define
      additional instances if you want more than one cache. The specific cache
      you create will be defined by an implementation that derives from these
      classes. For instance, an implementation of a memory cache is defined
      in "XrdOucCacheDram.hh".

   3. Use the Attach() method in XrdOucCache to attach your XrdOucCacheIO
      object with a cache instance. The method returns a remanufactured
      XrdOucCacheIO object that interposes the cache in front of the original
      XrdOucCacheIO. This allows you to transparently use the cache.

   4. When finished using the remanufactured XrdOucCacheIO object, use its
      Detach() method to remove the association from the cache. Other actions
      are defined by the actual implementation. For instance XrdOucCacheDram
      also releases any assigned cache pages, writes out any dirty pages, and
      may optionally delete the object when all references have been removed.

   5. You may delete cache instances as well. Just be sure that no associations
      still exist using the XrdOucCache::isAttached() method. Otherwise, the
      cache destructor will wait until all attached objects are detached.

   Example:
      class physIO : public XrdOucCacheIO {...}; // Define required methods
      class xCache : public XrdOucCache   {...}; // The cache implementation
      XrdOucCache::Parms myParms;                // Set any desired parameters
      XrdOucCache   *myCache;
      XrdOucCacheIO *cacheIO;
      xCache         theCache;                   // Implementation instance

      myCache = theCache.Create(myParms);        // Create a cache instance
      cacheIO = myCache->Attach(physIO);         // Interpose the cache

      // Use cacheIO (fronted by myCache) instead of physIO. When done...

      delete cacheIO->Detach();                  // Deletes cacheIO and physIO
*/
  
/******************************************************************************/
/*                C l a s s   X r d O u c C a c h e S t a t s                 */
/******************************************************************************/
  
/* The XrdOucCacheStats object holds statistics on cache usage. It is available
   in for each XrdOucCacheIO and each XrdOucCache object. The former usually
   identifies a specific file while the latter provides summary information.
*/

class XrdOucCacheStats
{
public:
long long    BytesPead;  // Bytes read via preread (not included in BytesRead)
long long    BytesRead;  // Total number of bytes read into the cache
long long    BytesGet;   // Number of bytes delivered from the cache
long long    BytesPass;  // Number of bytes read but not cached
long long    BytesWrite; // Total number of bytes written from the cache
long long    BytesPut;   // Number of bytes updated in the cache
int          Hits;       // Number of times wanted data was in the cache
int          Miss;       // Number of times wanted data was *not* in the cache
int          HitsPR;     // Number of pages wanted data was just preread
int          MissPR;     // Number of pages wanted data was just    read

inline void Get(XrdOucCacheStats &Dst)
               {sMutex.Lock();
                Dst.BytesRead   = BytesPead;  Dst.BytesGet    = BytesRead;
                Dst.BytesPass   = BytesPass;
                Dst.BytesWrite  = BytesWrite; Dst.BytesPut    = BytesPut;
                Dst.Hits        = Hits;       Dst.Miss        = Miss;
                Dst.HitsPR      = HitsPR;     Dst.MissPR      = MissPR;
                sMutex.UnLock();
               }

inline void Add(XrdOucCacheStats &Src)
               {sMutex.Lock();
                BytesRead  += Src.BytesPead;  BytesGet   += Src.BytesRead;
                BytesPass  += Src.BytesPass;
                BytesWrite += Src.BytesWrite; BytesPut   += Src.BytesPut;
                Hits       += Src.Hits;       Miss       += Src.Miss;
                HitsPR     += Src.HitsPR;     MissPR     += Src.MissPR;
                sMutex.UnLock();
               }

inline void  Add(long long &Dest, int &Val)
                {sMutex.Lock(); Dest += Val; sMutex.UnLock();}

inline void  Lock()   {sMutex.Lock();}
inline void  UnLock() {sMutex.UnLock();}

             XrdOucCacheStats() : BytesPead(0), BytesRead(0),  BytesGet(0),
                                  BytesPass(0), BytesWrite(0), BytesPut(0),
                                  Hits(0),      Miss(0),
                                  HitsPR(0),    MissPR(0) {}
            ~XrdOucCacheStats() {}
private:
XrdSysMutex sMutex;
};

/******************************************************************************/
/*                   C l a s s   X r d O u c C a c h e I O                    */
/******************************************************************************/

/* The XrdOucCacheIO object is responsible for interacting with the original
   data source/target. It can be used with or without a front-end cache.

   Six abstract methods are provided FSize(), Path(), Read(), Sync(), Trunc(),
   and Write(). You must provide implementations for each as described below.

   Four additional virtual methods are pre-defined: Base(), Detach(), and
   Preread() (2x). Normally, there is no need to over-ride these methods.

   Finally, each object carries with it a XrdOucCacheStats object.
*/

class XrdOucCacheIO
{
public:

// FSize()  returns the current size of the file associated with this object.

//          Success: size of the file in bytes.
//          Failure: -errno associated with the error.
virtual
long long   FSize() = 0;

// Path()   returns the path name associated with this object.
//
virtual
const char *Path() = 0;

// Read()   places Length bytes in Buffer from a data source at Offset.
//          When fronted by a cache, the cache is inspected first.

//          Success: actual number of bytes placed in Buffer.
//          Failure: -errno associated with the error.
virtual
int         Read (char *Buffer, long long Offset, int Length) = 0;

// Sync()   copies any outstanding modified bytes to the target.

//          Success: return 0.
//          Failure: -errno associated with the error.
virtual
int         Sync() = 0;

// Trunc()  truncates the file to the specified offset.

//          Success: return 0.
//          Failure: -errno associated with the error.
virtual
int         Trunc(long long Offset) = 0;


// Write()  takes Length bytes in Buffer and writes to a data target at Offset.
//          When fronted by a cache, the cache is updated as well.

//          Success: actual number of bytes copied from the Buffer.
//          Failure: -errno associated with the error.
virtual
int         Write(char *Buffer, long long Offset, int Length) = 0;

// Base()   returns the underlying XrdOucCacheIO object being used.
//
virtual XrdOucCacheIO *Base()   {return this;}

// Detach() detaches the object from the cache. It must be called instead of
//          using the delete operator since CacheIO objects may have multiple
//          outstanding references and actual deletion may need to be defered.
//          Detach() returns the underlying CacheIO object when the last
//          reference has been removed and 0 otherwise. This allows to say
//          something like "delete ioP->Detach()" if you want to make sure you
//          delete the underlying object as well. Alternatively, use the optADB
//          option when attaching a CacheIO object to a cache. This will delete
//          underlying object and always return 0 to avoid a double delete.
//          When not fronted by a cache, Detach() always returns itself. This
//          makes its use consistent whether or not a cache is employed.
//
virtual XrdOucCacheIO *Detach() {return this;}

// Preread() places Length bytes into the cache from a data source at Offset.
//          When there is no cache or the associated cache does not support or
//          allow pre-reads, it's a no-op. Cache placement limits do not apply.
//          To maximize parallelism, Peread() should called *after* obtaining
//          the wanted bytes using Read(). If the cache implementation supports
//          automatic prereads; you can setup parameters on how this should be
//          done using the next the next structure and method. The following
//          options can be specified:
//
static const int SingleUse = 0x0001; // Mark pages for single use

virtual
void        Preread (long long Offset, int Length, int Opts=0) {}

// The following structure describes automatic preread parameters. These can be
// set at any time for each XrdOucCacheIO object. It can also be specified when
// creating a cache to establish the defaults (see XrdOucCache::Create()).
// Generally, an implementation that supports prereads should disable small
// prereads when minPages or loBound is set to zero; and should disable large
// prereads when maxiRead or maxPages is set to zero. Refer to the actual
// derived class implementation on how the cache handles prereads.
//
struct aprParms
      {int   Trigger;   // preread if (rdln < Trigger)        (0 -> pagesize+1)
       int   prRecalc;  // Recalc pr efficiency every prRecalc bytes   (0->50M)
       int   Reserve4;
       short minPages;  // If rdln/pgsz < min,  preread minPages       (0->off)
       char  minPerf;   // Minimum auto preread performance required   (0->n/a)
       char  Reserve1;

             aprParms() : Trigger(0),  prRecalc(0), Reserve4(0),
                          minPages(0), minPerf(90), Reserve1(0)
                          {}
      };

virtual
void        Preread(aprParms &Parms) {}

//          Here is where the stats about cache and I/O usage reside. There
//          is a summary object in the associated cache as well.
//
XrdOucCacheStats Statistics;

virtual    ~XrdOucCacheIO() {}  // Always use Detach() instead of direct delete!
};

/******************************************************************************/
/*                     C l a s s   X r d O u c C a c h e                      */
/******************************************************************************/
  
/* The XrdOucCache class is used to define an instance of a cache. There can
   be many such instances. Each instance is associated with one or more
   XrdOucCacheIO objects. Use the Attach() method in this class to create 
   such associations.
*/

class XrdOucCache
{
public:

/* Attach()   must be called to obtain a new XrdOucCacheIO object that fronts an
              existing XrdOucCacheIO object with this cache.
              Upon success a pointer to a new XrdOucCacheIO object is returned
              and must be used to read and write data with the cache interposed.
              Upon failure, the original XrdOucCacheIO object is returned with
              errno set. You can continue using the object without any cache.
              The following Attach() options are available and, when specified,
              override the default options associated with the cache, except for
              optRW, optNEW, and optWIN which are valid only for a r/w cache.
*/
static const int optADB = 0x1000; // Automatically delete underlying CacheIO
static const int optFIS = 0x0001; // File is   Structured (e.g. root file)
static const int optFIU = 0x0002; // File is Unstructured (e.g. unix file)
static const int optRW  = 0x0004; // File is read/write   (o/w read/only)
static const int optNEW = 0x0014; // File is new -> optRW (o/w read to write)
static const int optWIN = 0x0024; // File is new -> optRW use write-in cache

virtual
XrdOucCacheIO *Attach(XrdOucCacheIO *ioP, int Options=0) = 0;

/* isAttached()
               Returns the number of CacheIO objects attached to this cache.
               Hence, 0 (false) if none and true otherwise.
*/
virtual
int            isAttached() {return 0;}

/* You must first create an instance of a cache using the Create() method.
   The Parms structure is used to pass parameters about the cache and should
   be filled in with values meaningful to the type of cache being created.
   The fields below, while oriented toward a memory cache, are sufficiently
   generic to apply to almost any kind of cache. Refer to the actual
   implementation in the derived class to see how these values are used.
*/
struct Parms
      {long long CacheSize; // Size of cache in bytes     (default 100MB)
       int       PageSize;  // Size of each page in bytes (default 32KB)
       int       Max2Cache; // Largest read to cache      (default PageSize)
       int       MaxFiles;  // Maximum number of files    (default 256 or 8K)
       int       Options;   // Options as defined below   (default r/o cache)
       int       Reserve1;  // Reserved for future use
       int       Reserve2;  // Reserved for future use

                 Parms() : CacheSize(104857600), PageSize(32768),
                           Max2Cache(0), MaxFiles(0), Options(0),
                           Reserve1(0),  Reserve2(0) {}
      };

// Valid option values in Parms::Options
//
static const int
isServer     = 0x0010; // This is server application (as opposed to a user app).
                       // Appropriate internal optimizations will be used.
static const int
isStructured = 0x0020; // Optimize for structured files (e.g. root).

static const int
canPreRead   = 0x0040; // Enable pre-read operations (o/w ignored)

static const int
logStats     = 0x0080; // Display statistics upon detach

static const int
Serialized   = 0x0004; // Caller ensures MRSW semantics

static const int
ioMTSafe     = 0x0008; // CacheIO object is MT-safe

static const int
Debug        = 0x0003; // Produce some debug messages (levels 0, 1, 2, or 3)

/* Create()    Creates an instance of a cache using the specified parameters.
               You must pass the cache parms and optionally any automatic
               pre-read parameters that will be used as future defaults.
               Upon success, returns a pointer to the cache. Otherwise, a null
               pointer is returned with errno set to indicate the problem.
*/
virtual
XrdOucCache   *Create(Parms &Params, XrdOucCacheIO::aprParms *aprP=0) = 0;

/* The following holds statistics for the cache itself. It is updated as
   associated cacheIO objects are deleted and their statistics are added.
*/
XrdOucCacheStats Stats;

               XrdOucCache() {}
virtual       ~XrdOucCache() {}
};

/******************************************************************************/
/*               C r e a t i n g   C a c h e   P l u g - I n s                */
/******************************************************************************/
  
/* You can create a cache plug-in for those parts of the xrootd system that
   allow a dynamically selectable cache implementation (e.g. the proxy server
   plug-in supports cache plug-ins via the pss.cachelib directive).

   Your plug-in must exist in a shared library and have the following extern C
   function defined:

   extern "C"
   {
   XrdOucCache *XrdOucGetCache(XrdSysLogger *Logger, // Where messages go
                               const char   *Config, // Config file used
                               const char   *Parms); // Optional parm string
   }

   When Logger is null, you should use cerr to output messages. Otherwise,
                        tie an instance XrdSysError to the passed logger.
   When Config is null, no configuration file is present. Otherwise, you need
                        additional configuration information you should get it
                        from that file in order to support single configuration.
   When Parms  is null, no parameter string was specified.

   The call should return an instance of an XrdOucCache object upon success and
   a null pointer otherwise. The instance is used to create actual caches using
   the object's Create() method.
*/
#endif
