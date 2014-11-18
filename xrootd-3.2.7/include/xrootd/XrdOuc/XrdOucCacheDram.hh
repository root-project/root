#ifndef __XRDOUCCACHEDRAM_HH__
#define __XRDOUCCACHEDRAM_HH__
/******************************************************************************/
/*                                                                            */
/*                    X r d O u c C a c h e D r a m . h h                     */
/*                                                                            */
/* (c) 2012 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdOuc/XrdOucCache.hh"
  
/* The class defined here implement a general memory cache for data from an
   arbitrary source (e.g. files, sockets, etc). It is based on the abstract
   definition of a cache. In practice, only one instance of this object needs
   to be created since actual instances of the cache are created using the
   Create() method. There can be many such instances. Each instance is
   associated with one or more XrdOucCacheIO objects. Use the Attach() method
   to create such associations.

   Notes: 1. The minimum PageSize is 4096 (4k) and must be a power of 2.
             The maximum PageSize is 16MB.
          2. The size of the cache is forced to be a multiple PageSize and
             have a minimum size of PageSize * 256.
          3. The minimum external read size is equal to PageSize.
          4. Currently, only write-through caches are supported.
          5. The Max2Cache value avoids placing data in the cache when a read
             exceeds the specified value. The minimum allowed is PageSize, which
             is also the default.
          6. Structured file optimization allows pages whose bytes have been
             fully referenced to be discarded; effectively increasing the cache.
          7. A structured cache treats all files as structured. By default, the
             cache treats files as unstructured. You can over-ride the settings
             on an individual file basis when the file's I/O object is attached
             by passing the XrdOucCache::optFIS or XrdOucCache::optFIU option.
          8. Write-in caches are only supported for files attached with the
             XrdOucCache::optWIN setting. Otherwise, updates are handled
             with write-through operations.
          9. A cache object may be deleted. However, the deletion is delayed
             until all CacheIO objects attached to the cache are detached.
             Use isAttached() to find out if any CacheIO objects are attached.
         10. The default maximum attached files is set to 8192 when isServer
             has been specified. Otherwise, it is set at 256.
         11. When canPreRead is specified, the cache asynchronously handles
             preread requests (see XrdOucCacheIO::Preread()) using 9 threads
             when isServer is in effect. Otherwise, 3 threads are used.
         12. The max queue depth for prereads is 8. When the max is exceeded
             the oldest preread is discarded to make room for the newest one.
         13. If you specify the canPreRead option when creating the cache you
             can also enable automatic prereads if the algorithm is workable.
             Otherwise, you will need to implement your own algorithm and
             issue prereads manually usingthe XrdOucCacheIO::Preread() method.
         14. The automatic preread algorithm is (ref XrdOucCacheIO::aprParms):
             a) A preread operation occurs when all of the following conditions
                are satisfied:
                o The cache CanPreRead option is in effect.
                o The read length < 'miniRead'
                   ||(read length < 'maxiRead' && Offset == next maxi offset)
             b) The preread page count is set to be readlen/pagesize and the
                preread occurs at the page after read_offset+readlen. The page
                is adjusted, as follows:
                o If the count is < minPages, it is set to minPages.
                o The count must be > 0 at this point.
             c) Normally, pre-read pages participate in the LRU scheme. However,
                if the preread was triggered using 'maxiRead' then the pages are
                marked for single use only. This means that the moment data is
                delivered from the page, the page is recycled.
         15. Invalid options silently force the use of the default.
*/

class XrdOucCacheDram : public XrdOucCache
{
public:

/* Attach()   must be called to obtain a new XrdOucCacheIO object that fronts an
              existing XrdOucCacheIO object with this memory cache.
              Upon success a pointer to a new XrdOucCacheIO object is returned
              and must be used to read and write data with the cache interposed.
              Upon failure, the original XrdOucCacheIO object is returned with
              errno set. You can continue using the object without any cache.
*/
virtual
XrdOucCacheIO *Attach(XrdOucCacheIO *ioP, int Options=0) {return 0;}

/* isAttached()
               Returns the number of CacheIO objects attached to this cache.
               Hence, 0 (false) if none and true otherwise.
*/
virtual
int            isAttached() {return 0;}

/* Create()    Creates an instance of a cache using the specified parameters.
               You must pass the cache parms and optionally any automatic
               pre-read parameters that will be used as future defaults.

               Success: returns a pointer to a new instance of the cache.
               Failure: a null pointer is returned with errno set to indicate
                        the problem.
*/
XrdOucCache   *Create(Parms &Params, XrdOucCacheIO::aprParms *aprP=0);

/* The following holds statistics for the cache itself. It is updated as
   associated cacheIO objects are deleted and their statistics are added.
*/
XrdOucCacheStats Stats;

               XrdOucCacheDram() {}
virtual       ~XrdOucCacheDram() {}
};
#endif
