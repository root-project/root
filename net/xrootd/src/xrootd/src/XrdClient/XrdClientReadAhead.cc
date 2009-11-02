/////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientReadAhead                                                   //
//                                                                      //
// Author: Fabrizio Furano (CERN IT-DM, 2009)                           //
//                                                                      //
// Classes to implement a selectable read ahead decision maker          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$

const char *XrdClientReadAheadCVSID = "$Id$";

#include "XrdClientReadAhead.hh"
#include "XrdClientConst.hh"
#include "XrdClientVector.hh"



bool XrdClientReadAheadMgr::TrimReadRequest(long long &offs, long &len, long rasize, long blksz) {

    if (!blksz) return true;

    long long newoffs;
    long newlen;

    long long lastbyte;

    newoffs = (long long)(offs / blksz);
    newoffs *= blksz;

    lastbyte = offs+len+blksz-1;
    lastbyte = (long long)(lastbyte / blksz);
    lastbyte *= blksz;
    
    newlen = lastbyte-newoffs;

//    std::cerr << "Trim: " << offs << "," << len << " --> " << newoffs << "," << newlen << std::endl;
    offs = newoffs;
    len = newlen;
    return true;

}





// -----------------------------------------------------------------------






// A basic implementation. Purely sequential read ahead
class XrdClientReadAhead_pureseq : public XrdClientReadAheadMgr {

protected:
   long long RALast;

public:

   XrdClientReadAhead_pureseq() {
      RALast = 0;
   }

   virtual int GetReadAheadHint(long long offset, long len, long long &raoffset, long &ralen, long blksz);

   virtual int Reset() {
      RALast = 0;
      return 0;
   }

};





int XrdClientReadAhead_pureseq::GetReadAheadHint(long long offset, long len, long long &raoffset, long &ralen, long blksz) {

   if (!blksz) blksz = 128*1024;

   // We read ahead only if (offs+len) lies in an interval of RALast not bigger than the readahead size
   if ( (RALast - (offset+len) < RASize) &&
        (RALast - (offset+len) > -RASize) &&
        (RASize > 0) ) {
      
      // This is a HIT case. Async readahead will try to put some data
      // in advance into the cache. The higher the araoffset will be,
      // the best chances we have not to cause overhead
      raoffset = xrdmax(RALast, offset + len);
      ralen = xrdmin(RASize,
                     offset + len + RASize - raoffset);
      
      if (ralen > 0) {
         TrimReadRequest(raoffset, ralen, RASize, blksz);
         RALast = raoffset + ralen;
         return 0;
      }
   }
   
   return 1;
   
};


// -----------------------------------------------------------------------






// Another read ahead schema. A window centered on the recent average slides through the file
//  following the stream of the requests
class XrdClientReadAhead_slidingavg : public XrdClientReadAheadMgr {

protected:
   long long RALast;

   long long LastOffsSum, LastOffsSum2;
   long long LastOffsSumsq, LastOffsSumsq2;
   XrdClientVector<long long> LastOffs;
   XrdClientVector<long long> LastAvgApprox, LastAvgApprox2;
public:

   XrdClientReadAhead_slidingavg() {
      RALast = 0;
      LastOffsSum = LastOffsSum2 = 0;
      LastOffsSumsq = LastOffsSumsq2 = 0;


   }

   virtual int GetReadAheadHint(long long offset, long len, long long &raoffset, long &ralen, long blksz);

   virtual int Reset() {
      RALast = 0;
      LastOffsSum = LastOffsSum2 = 0;
      LastOffsSumsq = LastOffsSumsq2 = 0;
      return 0;
   }

};





int XrdClientReadAhead_slidingavg::GetReadAheadHint(long long offset, long len, long long &raoffset, long &ralen, long blksz) {

   if (!blksz) blksz = 128*1024;

   // Keep the sums up to date, together with the max array size and the sumsqs
   LastOffsSum += offset;
   LastOffsSum2 += offset;
   LastOffs.Push_back(offset);

   if (LastOffs.GetSize() >= 50) {
      LastOffsSum2 -= LastOffs[LastOffs.GetSize()-50];
   }
   if (LastOffs.GetSize() >= 1000) {
      LastOffsSum -= LastOffs[0];
   }

   long long lastavg = LastOffsSum / LastOffs.GetSize();
   long long lastavg2 = LastOffsSum2 / xrdmin(LastOffs.GetSize(), 50);


   // Now the approximations of the std deviation, shifted right by some positions to avoid overflows
   long long sqerr = (((offset >> 20) - (lastavg >> 20))*((offset >> 20) - (lastavg >> 20)));
   LastOffsSumsq += sqerr;
   long long sqerr2 = ( ((offset - lastavg2) >> 20) * ((offset - lastavg2) >> 20) );
   LastOffsSumsq2 += sqerr2;

   LastAvgApprox.Push_back(sqerr);
   LastAvgApprox2.Push_back(sqerr2);

   if (LastAvgApprox2.GetSize() >= 50) {
      LastOffsSumsq2 -= LastAvgApprox2[0];
      LastAvgApprox2.Erase(0);
   }

   if (LastAvgApprox.GetSize() >= 1000) {
      LastOffsSumsq -= LastAvgApprox[0];
      LastAvgApprox.Erase(0);
   }

   if (LastOffs.GetSize() >= 1000) {
      LastOffs.Erase(0);
   }

   long long stddevi = LastOffsSumsq / LastOffs.GetSize();
   long long stddevi2 = LastOffsSumsq2 / LastAvgApprox2.GetSize();

   //std::cerr << "offs:" << offset << " avg:" << lastavg << " avg2:" << lastavg2 << " devi:" << stddevi << " devi2:" << stddevi2;

   // To read ahead, we want at least a few samples
   //if ( LastOffs.GetSize() < 10 ) return 1;

   // If the more stable avg is usable, use it
   if ((stddevi << 20) < 3*RASize) {
      raoffset = xrdmax(RALast, lastavg - RASize/2);

      ralen = xrdmin(RASize,
                     lastavg + RASize/2 - raoffset);

      if (ralen > (1024*1024)) {
         TrimReadRequest(raoffset, ralen, RASize, blksz);
         RALast = raoffset + ralen;
         //std::cerr << " raoffs:" << raoffset << " ralen:" << ralen << " Got avg" << std::endl;
         return 0;
      }
      //std::cerr << std::endl;
   } else
      // If the less stable avg is usable, use it     
      if ((stddevi2 << 20) < 3*RASize) {
         raoffset = xrdmax(RALast, lastavg2 - RASize/2);

         ralen = xrdmin(RASize,
                        lastavg2 + RASize/2 - raoffset);

 
         if (ralen > (1024*1024)) {
            TrimReadRequest(raoffset, ralen, RASize, blksz);
            RALast = raoffset + ralen;
            //std::cerr << " raoffs:" << raoffset << " ralen:" << ralen  << " Got avg2" << std::endl;
            return 0;
         }
         //std::cerr << std::endl;
      }

   //std::cerr << std::endl;
   return 1;

   
};











// ------------------------------------------------------

XrdClientReadAheadMgr *XrdClientReadAheadMgr::CreateReadAheadMgr(XrdClient_RAStrategy strategy) {
   XrdClientReadAheadMgr *ramgr = 0;

   switch (strategy) {

   case RAStr_none:
      break;

   case RAStr_pureseq: {
         ramgr = new XrdClientReadAhead_pureseq();
         break;
      }
   case RAStr_SlidingAvg: {
         ramgr = new XrdClientReadAhead_slidingavg();
         break;
      }
     
   }

   if (ramgr) ramgr->currstrategy = strategy;
   return ramgr;
}
