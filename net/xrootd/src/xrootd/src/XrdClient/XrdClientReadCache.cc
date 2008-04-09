//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientReadCache                                                   // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2006)                          //
//                                                                      //
// Classes to handle cache reading and cache placeholders               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

#include "XrdClient/XrdClientReadCache.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientEnv.hh"


//________________________________________________________________________
XrdClientReadCacheItem::XrdClientReadCacheItem(const void *buffer, long long begin_offs,
					       long long end_offs, long long ticksnow, bool placeholder)
{
    // Constructor
    fIsPlaceholder = placeholder;

    fData = (void *)0;
    if (!fIsPlaceholder) 
	fData = (void *)buffer;

    Touch(ticksnow);
    fBeginOffset = begin_offs;
    fEndOffset = end_offs;
}

//________________________________________________________________________
XrdClientReadCacheItem::~XrdClientReadCacheItem()
{
    // Destructor

    if (fData)
	free(fData);
}

//
// XrdClientReadCache
//

//________________________________________________________________________
long long XrdClientReadCache::GetTimestampTick()
{
    // Return timestamp

    // Mutual exclusion man!
    XrdSysMutexHelper mtx(fMutex);
    return ++fTimestampTickCounter;
}
  
//________________________________________________________________________
XrdClientReadCache::XrdClientReadCache() : fItems(16384)
{
    // Constructor

    fTimestampTickCounter = 0;
    fTotalByteCount = 0;

    fMissRate = 0.0;
    fMissCount = 0;
    fReadsCounter = 0;

    fBytesSubmitted = 0;
    fBytesHit = 0;
    fBytesUsefulness = 0.0;

    fMaxCacheSize = EnvGetLong(NAME_READCACHESIZE);
    fBlkRemPolicy = EnvGetLong(NAME_READCACHEBLKREMPOLICY);
}

//________________________________________________________________________
XrdClientReadCache::~XrdClientReadCache()
{
  // Destructor

  RemoveItems();

}



//________________________________________________________________________
bool XrdClientReadCache::SubmitRawData(const void *buffer, long long begin_offs,
					long long end_offs)
{
    if (!buffer) return true;
    XrdClientReadCacheItem *itm;


    Info(XrdClientDebug::kHIDEBUG, "Cache",
	 "Submitting " << begin_offs << "->" << end_offs << " to cache.");

    // Mutual exclusion man!
    XrdSysMutexHelper mtx(fMutex);


    //    PrintCache();

    // We remove all the blocks contained in the one we are going to put
    RemoveItems(begin_offs, end_offs);

    if (MakeFreeSpace(end_offs - begin_offs + 1)) {



	// We find the correct insert position to keep the list sorted by
	// BeginOffset
	// A data block will always be inserted BEFORE a true block with
	// equal beginoffset
	int pos = FindInsertionApprox(begin_offs);

	if (pos > 0) pos--;

	for (; pos < fItems.GetSize(); pos++) {
	    if (fItems[pos]->ContainsInterval(begin_offs, end_offs)) {
		pos = -1;
		break;
	    }
	    if (fItems[pos]->BeginOffset() >= begin_offs)
		break;
	}

	if (pos >= 0) {
	    itm = new XrdClientReadCacheItem(buffer, begin_offs, end_offs,
					     GetTimestampTick());
	    fItems.Insert(itm, pos);
	    fTotalByteCount += itm->Size();
	    fBytesSubmitted += itm->Size();
	}

	return true;
    } // if

    return false;

    //    PrintCache();
}


//________________________________________________________________________
void XrdClientReadCache::SubmitXMessage(XrdClientMessage *xmsg, long long begin_offs,
					long long end_offs)
{
    // To populate the cache of items, newly received

    const void *buffer = xmsg->DonateData();

    if (!SubmitRawData(buffer, begin_offs, end_offs))
        free(const_cast<void *>(buffer));
}



//________________________________________________________________________
int XrdClientReadCache::FindInsertionApprox(long long begin_offs) {

    // quickly finds the correct insertion point for a placeholder or for a data block
    // Remember that placeholders are inserted before data blks with
    // identical beginoffs

    if (!fItems.GetSize()) return 0;
    return FindInsertionApprox_rec(0, fItems.GetSize()-1, begin_offs);



}


//________________________________________________________________________
int XrdClientReadCache::FindInsertionApprox_rec(int startidx, int endidx,
					long long begin_offs) {

    // Dicotomic search to quickly find a place where to start scanning
    // for the final destination of a blk
    
    if (endidx - startidx <= 1) {

	
	if (fItems[startidx]->BeginOffset() >= begin_offs) {
	    // The item is to be inserted before the startidx pos
	    return startidx;
	}    
	if (fItems[endidx]->BeginOffset() < begin_offs) {
	    // The item is to be inserted after the endidx pos
	    return endidx+1;
	}

	return endidx;

    }

    int pos2 = (endidx + startidx) / 2;

    if (fItems[startidx]->BeginOffset() >= begin_offs) {
	// The item is not here!
	return startidx;
    }    
    if (fItems[endidx]->BeginOffset() < begin_offs) {
	// The item is not here!
	return endidx+1;
    }

    if (fItems[pos2]->BeginOffset() >= begin_offs) {
	// The item is between startidx and pos2!
	return FindInsertionApprox_rec(startidx, pos2, begin_offs);
    }

    if (fItems[pos2]->BeginOffset() < begin_offs) {
	// The item is between pos2 and endidx!
	return FindInsertionApprox_rec(pos2, endidx, begin_offs);
    }

    return endidx;
}

//________________________________________________________________________
void XrdClientReadCache::PutPlaceholder(long long begin_offs,
					long long end_offs)
{
    // To put a placeholder into the cache

    XrdClientReadCacheItem *itm = 0;

    {
	// Mutual exclusion man!
	XrdSysMutexHelper mtx(fMutex);

	// We find the correct insert position to keep the list sorted by
	// BeginOffset
	int pos = FindInsertionApprox(begin_offs);
	int p = pos;
	if (p > 0) p--;

	for (; p < fItems.GetSize(); p++) {
	    if (fItems[p]->ContainsInterval(begin_offs, end_offs)) {
		return;
	    }

	    if (fItems[p]->BeginOffset() > end_offs)
		break;

	    // We found an item which is overlapping the new candidate.
	    // Here we shrink the candidate at the left 
	    if ( (fItems[p]->BeginOffset() >= begin_offs) &&
		 (fItems[p]->BeginOffset() <= end_offs) ) {

	      itm = 0;
	      if (begin_offs < fItems[p]->BeginOffset()-1)
		itm = new XrdClientReadCacheItem(0, begin_offs, fItems[p]->BeginOffset()-1,
						 GetTimestampTick(), true);
	      begin_offs = fItems[p]->EndOffset()+1;
	      if (itm) {
		fItems.Insert(itm, p);

		// Optimization: we avoid to check the same block twice
		p++;
	      }
	      
	    }

	    if ( (fItems[p]->BeginOffset() <= begin_offs) &&
		 (fItems[p]->EndOffset() >= begin_offs) ) {

	      begin_offs = fItems[p]->EndOffset()+1;
	      
	    }


	    pos = p+1;

	    if (begin_offs >= end_offs) return;


	}

	itm = new XrdClientReadCacheItem(0, begin_offs, end_offs,
					 GetTimestampTick(), true);
	fItems.Insert(itm, pos);

    }

    //    PrintCache();
}

//________________________________________________________________________
long XrdClientReadCache::GetDataIfPresent(const void *buffer,
					  long long begin_offs,
					  long long end_offs,
					  bool PerfCalc, 
					  XrdClientIntvList &missingblks,
					  long &outstandingblks)
{
    // Copies the requested data from the cache. False if not possible
    // Also, this function figures out if:
    // - there are data blocks marked as outstanding
    // - there are sub blocks which should be requested

    int it;
    long bytesgot = 0;

    long long lasttakenbyte = begin_offs-1;
    outstandingblks = 0;
    missingblks.Clear();

    XrdSysMutexHelper mtx(fMutex);

    if (PerfCalc)
	fReadsCounter++;

    // We try to compose the requested data block by concatenating smaller
    //  blocks. 


    // Find a block helping us to go forward
    // The blocks are sorted
    // By scanning the list we also look for:
    //  - the useful blocks which are outstanding
    //  - the useful blocks which are missing, and not outstanding

    // First scan: we get the useful data
    // and remember where we arrived
    it = FindInsertionApprox(lasttakenbyte);
    if (it > 0) it--;

    for (; it < fItems.GetSize(); it++) {
	long l = 0;

	if (!fItems[it]) continue;

	if (fItems[it]->BeginOffset() > lasttakenbyte+1) break;

	if (!fItems[it]->IsPlaceholder())
	    l = fItems[it]->GetPartialInterval(((char *)buffer)+bytesgot,
					       begin_offs+bytesgot, end_offs);
	else break;

	if (l > 0) {
	    bytesgot += l;
	    lasttakenbyte = begin_offs+bytesgot-1;

	    if (fBlkRemPolicy != kRmBlk_FIFO)
	      fItems[it]->Touch(GetTimestampTick());

	    if (PerfCalc) {
		fBytesHit += l;
		UpdatePerfCounters();
	    }

	    if (bytesgot >= end_offs - begin_offs + 1) {
		return bytesgot;
	    }

	}

    }


    // We are here if something is missing to get all the data we need
    // Hence we build a list of what is missing
    // right now what is missing is the interval
    // [lasttakenbyte+1, end_offs]

    XrdClientCacheInterval intv;


    for (; it < fItems.GetSize(); it++) {
	long l;

	if (fItems[it]->BeginOffset() > end_offs) break;

	if (fItems[it]->BeginOffset() > lasttakenbyte+1) {
	    // We found that the interval
	    // [lastbyteseen+1, fItems[it]->BeginOffset-1]
	    // is a hole, which should be requested explicitly

	    intv.beginoffs = lasttakenbyte+1;
	    intv.endoffs = fItems[it]->BeginOffset()-1;
	    missingblks.Push_back( intv );

	    lasttakenbyte = fItems[it]->EndOffset();
	    if (lasttakenbyte >= end_offs) break;
	    continue;
	}

	// Let's see if we can get something from this blk, even if it's a placeholder
	l = fItems[it]->GetPartialInterval(0, lasttakenbyte+1, end_offs);

	if (l > 0) {
	    // We found a placeholder to wait for
	    // or a data block

	    if (fItems[it]->IsPlaceholder()) {
		// Add this interval to the number of blocks to wait for
		outstandingblks++;

	    }


	    lasttakenbyte += l;
	}


    }

     if (lasttakenbyte+1 < end_offs) {
 	    intv.beginoffs = lasttakenbyte+1;
 	    intv.endoffs = end_offs;
 	    missingblks.Push_back( intv );
     }


    if (PerfCalc) {
	fMissCount++;
	UpdatePerfCounters();
    }

    return bytesgot;
}


//________________________________________________________________________
void XrdClientReadCache::PrintCache() {

    XrdSysMutexHelper mtx(fMutex);
    int it;

    Info(XrdClientDebug::kHIDEBUG, "Cache",
	 "Cache Status --------------------------");

    for (it = 0; it < fItems.GetSize(); it++) {

	if (fItems[it]) {

	    if (fItems[it]->IsPlaceholder()) {
		
		Info(XrdClientDebug::kHIDEBUG,
		     "Cache blk", it << "Placeholder " <<
		     fItems[it]->BeginOffset() << "->" << fItems[it]->EndOffset() );

	    }
	    else
		Info(XrdClientDebug::kHIDEBUG,
		     "Cache blk", it << "Data block  " <<
		     fItems[it]->BeginOffset() << "->" << fItems[it]->EndOffset() );

	}
    }
    
    Info(XrdClientDebug::kHIDEBUG, "Cache",
	 "--------------------------------------");

}


//________________________________________________________________________
void XrdClientReadCache::RemoveItems(long long begin_offs, long long end_offs)
{
    // To remove all the items contained in the given interval

    int it;
    XrdSysMutexHelper mtx(fMutex);

    it = FindInsertionApprox(begin_offs);
    if (it > 0) it--;

    // We remove all the blocks contained in the given interval
    while (it < fItems.GetSize()) {
	if (fItems[it]) {

	    if (fItems[it]->BeginOffset() > end_offs) break;

	    if (fItems[it]->ContainedInInterval(begin_offs, end_offs)) {

		if (!fItems[it]->IsPlaceholder())
		    fTotalByteCount -= fItems[it]->Size();
	    
		delete fItems[it];
		fItems.Erase(it);
	    }
	    else it++;

	}
	else it++;

    }
    // Then we resize or split the placeholders overlapping the given interval
    bool changed;
    it = FindInsertionApprox(begin_offs);
    if (it > 0) it--;

    do {
	changed = false;
	for (; it < fItems.GetSize(); it++) {


	    if (fItems[it]) {

		if (fItems[it]->BeginOffset() > end_offs) break;

		if ( fItems[it]->IsPlaceholder() ) {
		    long long plc1_beg = 0;
		    long long plc1_end = 0;
	  
		    long long plc2_beg = 0;
		    long long plc2_end = 0;
	  
		    // We have a placeholder which contains the arrived block
		    plc1_beg = fItems[it]->BeginOffset();
		    plc1_end = begin_offs-1;

		    plc2_beg = end_offs+1;
		    plc2_end = fItems[it]->EndOffset();

		    if ( ( (begin_offs >= fItems[it]->BeginOffset()) &&
			   (begin_offs <= fItems[it]->EndOffset()) ) ||
			 ( (end_offs >= fItems[it]->BeginOffset()) &&
			   (end_offs <= fItems[it]->EndOffset()) ) ) {

			delete fItems[it];
			fItems.Erase(it);
			changed = true;
			it--;

			if (plc1_end - plc1_beg > 32) {
			    PutPlaceholder(plc1_beg, plc1_end);
			}

			if (plc2_end - plc2_beg > 32) {
			    PutPlaceholder(plc2_beg, plc2_end);
			}

			break;
	  
		    }

		


		}

	    }

	}

	it = xrdmax(0, it-2);
    } while (changed);



}

//________________________________________________________________________
void XrdClientReadCache::RemoveItems()
{
    // To remove all the items
    int it;
    XrdSysMutexHelper mtx(fMutex);

    it = 0;

    while (it < fItems.GetSize()) {
	delete fItems[it];
	fItems.Erase(it);
    }

    fTotalByteCount = 0;

}
//________________________________________________________________________
void XrdClientReadCache::RemovePlaceholders() {

    // Finds the LRU item and removes it
    // We  remove placeholders

    int it = 0;

    XrdSysMutexHelper mtx(fMutex);

    if (!fItems.GetSize()) return;

    while (1) {

	if (fItems[it] && fItems[it]->IsPlaceholder()) {
	    delete fItems[it];
	    fItems.Erase(it);
	}
	else
	    it++;

	if (it == fItems.GetSize()) break;
    }

}



//________________________________________________________________________
bool XrdClientReadCache::RemoveFirstItem()
{
    // Finds the first item (lower offset) and removes it
    // We don't remove placeholders

    int it, lruit;
    XrdClientReadCacheItem *item;

    XrdSysMutexHelper mtx(fMutex);

    lruit = -1;

	// Kill the first not placeholder if we have too many blks
	lruit = -1;
	for (it = 0; it < fItems.GetSize(); it++) {
	    // We don't remove placeholders
	    if (!fItems[it]->IsPlaceholder()) {
		lruit = it;
		break;
	    }
	}


    if (lruit >= 0)
	item = fItems[lruit];
    else return false;

    fTotalByteCount -= item->Size();
    delete item;
    fItems.Erase(lruit);


    return true;
}




//________________________________________________________________________
bool XrdClientReadCache::RemoveLRUItem()
{
    // Finds the LRU item and removes it
    // We don't remove placeholders

    int it, lruit;
    long long minticks = -1;
    XrdClientReadCacheItem *item = 0;

    XrdSysMutexHelper mtx(fMutex);

    lruit = -1;

    if (fItems.GetSize() < 1000000)
	for (it = 0; it < fItems.GetSize(); it++) {
	    // We don't remove placeholders
	    if (fItems[it] && !fItems[it]->IsPlaceholder()) {
		if ((minticks < 0) || (fItems[it]->GetTimestampTicks() < minticks)) {
		    minticks = fItems[it]->GetTimestampTicks();
		    lruit = it;
		}
	    }
	}
    else {

	// Kill the first not placeholder if we have tooooo many blks
	lruit = -1;
	for (it = 0; it < fItems.GetSize(); it++) {
	    // We don't remove placeholders
	    if (!fItems[it]->IsPlaceholder()) {
		lruit = it;
		minticks = 0;
		break;
	    }
	}
    }

    if (lruit >= 0)
	item = fItems[lruit];
    else return false;

    if (item) {
      fTotalByteCount -= item->Size();
      delete item;
      fItems.Erase(lruit);
    }

    return true;
}

//________________________________________________________________________
bool XrdClientReadCache::RemoveItem() {

    switch (fBlkRemPolicy) {

    case kRmBlk_LRU:
    case kRmBlk_FIFO:
      return RemoveLRUItem();

    case kRmBlk_LeastOffs:
      return RemoveFirstItem();

    }

    return RemoveLRUItem();
}

//________________________________________________________________________
bool XrdClientReadCache::MakeFreeSpace(long long bytes)
{
    // False if not possible (requested space exceeds max size!)

    if (!WillFit(bytes))
	return false;

    XrdSysMutexHelper mtx(fMutex);

    while (fMaxCacheSize - fTotalByteCount < bytes)
      if (!RemoveItem())
	return false;

    return true;
}


void XrdClientReadCache::GetInfo(int &size, long long &bytessubmitted,
				 long long &byteshit,
				 long long &misscount,
				 float &missrate,
				 long long &readreqcnt,
				 float &bytesusefulness ) {
  size = fMaxCacheSize;
  bytessubmitted = fBytesSubmitted;
  byteshit = fBytesHit;
  misscount = fMissCount;
  missrate = fMissRate;
  readreqcnt = fReadsCounter;
  bytesusefulness = fBytesUsefulness;
}
