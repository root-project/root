/*****************************************************************************/
/*                                                                           */
/*                          XrdMonCtrArchiver.hh                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONCTRARCHIVER_HH
#define XRDMONCTRARCHIVER_HH

#include "XrdMon/XrdMonTypes.hh"
#include "pthread.h"
#include <vector>
using std::vector;

class XrdMonCtrPacket;
class XrdMonCtrWriter;
class XrdMonDecPacketDecoder;

// Class responsible for archiving packets in log files.
// Manages heartbeat for writers (writers inactive for 24 hours
// are closed). It does not interpret data inside packet.

extern "C" void* decHDFlushHeartBeat(void* arg);
extern "C" void* decRTFlushHeartBeat(void* arg);

class XrdMonCtrArchiver {
public:
    XrdMonCtrArchiver(const char* cBaseDir, 
                      const char* dBaseDir,
                      const char* rtLogDir,
                      kXR_int64 maxFileSize,
                      int ctrBufSize,
                      int rtBufSize,
                      bool onlineDec,
                      bool rtDec);
    ~XrdMonCtrArchiver();
    void operator()();

    static int _decHDFlushDelay; // number of sec between flushes of decoded 
                                 // history data to disk
    static int _decRTFlushDelay; // number of sec between flushes of decoded 
                                 // "current" data to disk

private:
    void check4InactiveSenders();
    void archivePacket(XrdMonCtrPacket* p);
    friend void* decHDFlushHeartBeat(void* arg);
    friend void* decRTFlushHeartBeat(void* arg);
    
private:
    enum { TIMESTAMP_FREQ = 10000,   // re-take time every X packets
           MAX_INACTIVITY = 60*60*24 // kill writer if no activity for 24 hours
    };
    
    vector<XrdMonCtrWriter*> _writers;

    XrdMonDecPacketDecoder* _decoder;
    pthread_t               _decHDFlushThread; // history data
    pthread_t               _decRTFlushThread; // real time data

    long _currentTime;
    int  _heartbeat; // number of packets since the last time check
};

#endif /*  XRDMONCTRARCHIVER_HH */
