/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecPreProcess.hh                          */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONDECPREPROCESS_HH
#define XRDMONDECPREPROCESS_HH

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonDecOnePacket.hh"
#include <deque>
#include <fstream>
#include <utility>
#include <vector>
using std::deque;
using std::pair;
using std::vector;

// preprocesses input file, checks for lost packets
// and fixes order of packets. If order has to change,
// it stores output in a tmp file.
// When it returns, theFile is an open file (tmp file or original)

class XrdMonDecPreProcess {
public:
    XrdMonDecPreProcess(fstream& theFile, 
                        kXR_int64 fSize, 
                        sequen_t lastSeq,
                        kXR_int32 ignoreIfBefore,
                        vector< pair<packetlen_t, kXR_int64> >& allPackets);
    void operator()();
    
private:
    void checkFile();
    kXR_char previousSeq() const;
    bool outOfOrder(XrdMonDecOnePacket& packet);
    void keepPacket(XrdMonDecOnePacket& packet);
    void add2TempBuf(XrdMonDecOnePacket& packet);
    int processOnePacket(const char* buf, 
                         int bytesLeft, 
                         kXR_int64 fPos, 
                         kXR_int32& xrdStartTime);
    void reportAndThrowIfTooBad();

private:
    fstream& _file;
    kXR_int64 _fSize;

    enum { TBUFSIZE = 20, MAXTBUFELEM = TBUFSIZE-1 };
    // temporary buffer holding TBUFSIZE last packets    
    XrdMonDecOnePacket _tempBuf[TBUFSIZE];
    short _tempBufPos;
    bool _markNextSlotAsSpecial;

    kXR_int32 _ignoreIfBefore; // ignore packets that arrived before given time
    
    vector< pair<packetlen_t, kXR_int64> >& _allPackets;
    
    // for report purposes
    vector<int> _lostPackets;
    vector<int> _oooPackets;
    
    sequen_t _lastSeq; // last seq from the previous log file
};

#endif /* XRDMONDECPREPROCESS_HH */
