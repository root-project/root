#ifndef XRD_READCACHE_H
#define XRD_READCACHE_H
/******************************************************************************/
/*                                                                            */
/*               X r d C l i e n t R e a d C a c h e . h h                    */
/*                                                                            */
/* Author: Fabrizio Furano (INFN Padova, 2006)                                */
/*                                                                            */
/* This file is part of the XRootD software suite.                            */
/*                                                                            */
/* XRootD is free software: you can redistribute it and/or modify it under    */
/* the terms of the GNU Lesser General Public License as published by the     */
/* Free Software Foundation, either version 3 of the License, or (at your     */
/* option) any later version.                                                 */
/*                                                                            */
/* XRootD is distributed in the hope that it will be useful, but WITHOUT      */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public       */
/* License for more details.                                                  */
/*                                                                            */
/* You should have received a copy of the GNU Lesser General Public License   */
/* along with XRootD in a file called COPYING.LESSER (LGPL license) and file  */
/* COPYING (GPL license).  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                            */
/* The copyright holder's institutional names and contributor's names may not */
/* be used to endorse or promote products derived from this software without  */
/* specific prior written permission of the institution or contributor.       */
/******************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Classes to handle cache reading and cache placeholders               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdClient/XrdClientInputBuffer.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientVector.hh"
#include "XrdClient/XrdClientConst.hh"

//
// XrdClientReadCacheItem
//
// An item is nothing more than an interval of bytes taken from a file.
// Extremes are included.
// Since a cache object is to be associated to a single instance
// of TXNetFile, we do not have to keep here any filehandle
//

class XrdClientReadCacheItem {
private:
    // A placeholder block is a "fake block" used to mark outstanding data
    bool             fIsPlaceholder;  

    long long        fBeginOffset;    // Offset of the first byte of data
    void             *fData;
    long long        fEndOffset;      // Offset of the last byte of data
    long             fTimestampTicks; // timestamp updated each time it's referenced

public:
    XrdClientReadCacheItem(const void *buffer, long long begin_offs, 
			   long long end_offs, long long ticksnow,
			   bool placeholder=false);
    ~XrdClientReadCacheItem();

    inline long long BeginOffset() { return fBeginOffset; }
    inline long long EndOffset() { return fEndOffset; }

    // Is this obj contained in the given interval (which is going to be inserted) ?
    inline bool   ContainedInInterval(long long begin_offs, long long end_offs) {
	return ( (end_offs >= begin_offs) &&
		 (fBeginOffset >= begin_offs) &&
		 (fEndOffset <= end_offs) );
    }

    // Does this obj contain the given interval (which is going to be requested) ?
    inline bool   ContainsInterval(long long begin_offs, long long end_offs) {
	return ( (end_offs > begin_offs) &&
		 (fBeginOffset <= begin_offs) && (fEndOffset >= end_offs) );
    }

    // Are the two intervals intersecting in some way?
    inline bool  IntersectInterval(long long begin_offs, long long end_offs) {
      if ( ContainsOffset( begin_offs ) || ContainsOffset( end_offs ) ) return true;
      if ( (fBeginOffset >= begin_offs) && (fBeginOffset <= end_offs) ) return true;
      return false;
    }


    inline bool ContainsOffset(long long offs) {
	return (fBeginOffset <= offs) && (fEndOffset >= offs);
    }

    void *GetData() { return fData; }
    
    // Get the requested interval, if possible
    inline bool   GetInterval(const void *buffer, long long begin_offs, 
			      long long end_offs) {
	if (!ContainsInterval(begin_offs, end_offs))
	    return FALSE;
	memcpy((void *)buffer, ((char *)fData)+(begin_offs - fBeginOffset),
	       end_offs - begin_offs + 1);
	return TRUE;
    }

    // Get as many bytes as possible, starting from the beginning of the given
    // interval
    inline long   GetPartialInterval(const void *buffer, long long begin_offs,
				     long long end_offs) {

	long long b = -1, e, l;

	if (begin_offs > end_offs) return 0;

	// Try to set the starting point, if contained in the given interval
	if ( (begin_offs >= fBeginOffset) &&
	     (begin_offs <= fEndOffset) )
	    b = begin_offs;

	if (b < 0) return 0;

	// The starting point is in the interval. Let's get the minimum endpoint
	e = xrdmin(end_offs, fEndOffset);

	l = e - b + 1;

	if (buffer && fData)
	    memcpy((void *)buffer, ((char *)fData)+(b - fBeginOffset), l);

	return l;
    }

    inline long long GetTimestampTicks() { return(fTimestampTicks); }

    inline bool IsPlaceholder() { return fIsPlaceholder; }

    long Size() { return (fEndOffset - fBeginOffset + 1); }

    inline void     Touch(long long ticksnow) { fTimestampTicks = ticksnow; }

    bool Pinned;
};

//
// XrdClientReadCache
//
// The content of the cache. Not cache blocks, but
// variable length Items
//
typedef XrdClientVector<XrdClientReadCacheItem *> ItemVect;

// A cache interval, extremes included
struct XrdClientCacheInterval {
    long long beginoffs;
    long long endoffs;
};

typedef XrdClientVector<XrdClientCacheInterval> XrdClientIntvList;

class XrdClientReadCache {
private:

    long long       fBytesHit;         // Total number of bytes read with a cache hit
    long long       fBytesSubmitted;   // Total number of bytes inserted
    float           fBytesUsefulness;
    ItemVect        fItems;
    long long       fMaxCacheSize;
    long long       fMissCount;        // Counter of the cache misses
    float           fMissRate;            // Miss rate
    XrdSysRecMutex  fMutex;
    long long       fReadsCounter;     // Counter of all the attempted reads (hit or miss)
    int             fBlkRemPolicy;     // The algorithm used to remove "old" chunks
    long long       fTimestampTickCounter;        // Aging mechanism yuk!
    long long       fTotalByteCount;

    long long       GetTimestampTick();
    bool            MakeFreeSpace(long long bytes);

    bool            RemoveItem();
    bool            RemoveLRUItem();
    bool            RemoveFirstItem();

    inline void     UpdatePerfCounters() {
	if (fReadsCounter > 0)
	    fMissRate = (float)fMissCount / fReadsCounter;
	if (fBytesSubmitted > 0)
	    fBytesUsefulness = (float)fBytesHit / fBytesSubmitted;
    }

    int             FindInsertionApprox(long long begin_offs);
    int             FindInsertionApprox_rec(int startidx, int endidx,
					long long begin_offs);
public:

    // The algos available for the removal of "old" blocks
    enum {
      kRmBlk_LRU = 0,
      kRmBlk_LeastOffs,
      kRmBlk_FIFO
    };

    XrdClientReadCache();
    ~XrdClientReadCache();
  
    long          GetDataIfPresent(const void *buffer, long long begin_offs,
				   long long end_offs, bool PerfCalc,
				   XrdClientIntvList &missingblks, long &outstandingblks);

  void                       GetInfo(
					  // The actual cache size
					  int &size,

					  // The number of bytes submitted since the beginning
					  long long &bytessubmitted,

					  // The number of bytes found in the cache (estimate)
					  long long &byteshit,

					  // The number of reads which did not find their data
                                          // (estimate)
					  long long &misscount,

					  // miss/totalreads ratio (estimate)
					  float &missrate,

					  // number of read requests towards the cache
					  long long &readreqcnt,

					  // ratio between bytes found / bytes submitted
					  float &bytesusefulness
				     );

    inline long long GetTotalByteCount() {
	XrdSysMutexHelper m(fMutex);
	return fTotalByteCount;
    }

    void PutPlaceholder(long long begin_offs, long long end_offs);

    inline void     PrintPerfCounters() {
	XrdSysMutexHelper m(fMutex);

	cout << "Low level caching info:" << endl;
        cout << " StallsRate=" << fMissRate << endl;
        cout << " StallsCount=" << fMissCount << endl;
        cout << " ReadsCounter=" << fReadsCounter << endl;
	cout << " BytesUsefulness=" << fBytesUsefulness << endl;
        cout << " BytesSubmitted=" << fBytesSubmitted << " BytesHit=" << 
           fBytesHit << endl << endl;
    }


    void            PrintCache();

    void            SubmitXMessage(XrdClientMessage *xmsg, long long begin_offs,
				   long long end_offs);

    bool            SubmitRawData(const void *buffer, long long begin_offs,
				  long long end_offs, bool pinned=false);

    void            RemoveItems(bool leavepinned=true);
    void            RemoveItems(long long begin_offs, long long end_offs, bool remove_overlapped = false);
    void            RemovePlaceholders();


    void            SetSize(int sz) {
      fMaxCacheSize = sz;
    }

    void            SetBlkRemovalPolicy(int p) {
      fBlkRemPolicy = p;
    }

    void UnPinCacheBlk(long long begin_offs, long long end_offs);
    void *FindBlk(long long begin_offs, long long end_offs);

    // To check if a block dimension will fit into the cache
    inline bool   WillFit(long long bc) {
	XrdSysMutexHelper m(fMutex);
	return (bc < fMaxCacheSize);
    }

};
#endif
