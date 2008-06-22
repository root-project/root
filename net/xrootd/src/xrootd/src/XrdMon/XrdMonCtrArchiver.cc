/*****************************************************************************/
/*                                                                           */
/*                           XrdMonCtrArchiver.cc                            */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonCtrAdmin.hh"
#include "XrdMon/XrdMonCtrArchiver.hh"
#include "XrdMon/XrdMonCtrBuffer.hh"
#include "XrdMon/XrdMonCtrPacket.hh"
#include "XrdMon/XrdMonCtrWriter.hh"
#include "XrdMon/XrdMonDecPacketDecoder.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonSenderInfo.hh"

#include "XrdSys/XrdSysHeaders.hh"

#include <sys/time.h>

using std::cout;
using std::endl;

int XrdMonCtrArchiver::_decHDFlushDelay = -1;
int XrdMonCtrArchiver::_decRTFlushDelay = -1;

XrdMonCtrArchiver::XrdMonCtrArchiver(const char* cBaseDir, 
                                     const char* dBaseDir,
                                     const char* rtLogDir,
                                     kXR_int64 maxLogSize,
                                     int ctrBufSize,
                                     int rtBufSize,
                                     bool onlineDec,
                                     bool rtDec)
    : _decoder(0), 
      _currentTime(0),
      _heartbeat(1) // force taking timestamp first time
{
    XrdMonCtrWriter::setBaseDir(cBaseDir);
    XrdMonCtrWriter::setMaxLogSize(maxLogSize);
    XrdMonCtrWriter::setBufferSize(ctrBufSize);
    
    if ( onlineDec ) {
        _decoder = new XrdMonDecPacketDecoder(dBaseDir, rtLogDir, rtBufSize);
        // BTW, MT-safety inside Sink
        if ( 0 != pthread_create(&_decHDFlushThread, 
                                 0, 
                                 decHDFlushHeartBeat,
                                 (void*)_decoder) ) {
            throw XrdMonException(ERR_PTHREADCREATE, 
                                  "Failed to create thread");
        }
        if ( 0 != rtDec ) {
            if ( 0 != pthread_create(&_decRTFlushThread, 
                                     0, 
                                     decRTFlushHeartBeat,
                                     (void*)_decoder) ) {
                throw XrdMonException(ERR_PTHREADCREATE, 
                                      "Failed to create thread");
            }
        }
    }
}

XrdMonCtrArchiver::~XrdMonCtrArchiver()
{
    _decoder->flushRealTimeData();
    delete _decoder;
    _decoder = 0;

    // go through all writers and shut them down
    int i, s = _writers.size();
    for (i=0 ; i<s ; i++) {
        delete _writers[i];
    }
    _writers.clear();

    XrdMonSenderInfo::shutdown();
}

void
XrdMonCtrArchiver::operator()()
{
    XrdMonCtrBuffer* pb = XrdMonCtrBuffer::instance();
    while ( 1 ) {
        try {
            if ( 0 == --_heartbeat ) {
                check4InactiveSenders();
            }
            XrdMonCtrPacket* p = pb->pop_front();
            archivePacket(p);
            delete p;
        } catch (XrdMonException& e) {
            if ( e.err() == SIG_SHUTDOWNNOW ) {
                return;
            }
            e.printItOnce();
        }
    }
}

// this function runs in a separate thread, wakes up
// every now and then and triggers "history data" flushing
extern "C" void*
decHDFlushHeartBeat(void* arg)
{
    if ( XrdMonCtrArchiver::_decHDFlushDelay == -1 ) {
        return (void*)0; // should never happen
    }
    XrdMonDecPacketDecoder* myDecoder = (XrdMonDecPacketDecoder*) arg;
    if ( 0 == myDecoder ) {
        throw XrdMonException(ERR_PTHREADCREATE, 
                              "invalid archiver passed");
    }
    while ( 1 ) {
        sleep(XrdMonCtrArchiver::_decHDFlushDelay);
        myDecoder->flushHistoryData();
    }

    return (void*)0;
}

// this function runs in a separate thread, wakes up
// every now and then and triggers "current data" flushing
extern "C" void*
decRTFlushHeartBeat(void* arg)
{
    if ( XrdMonCtrArchiver::_decRTFlushDelay == -1 ) {
        return (void*)0; // should never happen
    }
    XrdMonDecPacketDecoder* myDecoder = (XrdMonDecPacketDecoder*) arg;
    if ( 0 == myDecoder ) {
        throw XrdMonException(ERR_PTHREADCREATE, 
                              "invalid archiver passed");
    }
    while ( 1 ) {
        sleep(XrdMonCtrArchiver::_decRTFlushDelay);
        if ( 0 != myDecoder ) {
            myDecoder->flushRealTimeData();
        }
    }

    return (void*)0;
}

void
XrdMonCtrArchiver::check4InactiveSenders()
{
    _heartbeat = TIMESTAMP_FREQ;
    struct timeval tv;
    gettimeofday(&tv, 0);
    _currentTime = tv.tv_sec;
    
    long allowed = _currentTime - MAX_INACTIVITY;
    int i, s = _writers.size();
    for (i=0 ; i<s ; i++) {
        if ( _writers[i]->lastActivity() < allowed ) {
            cout << "No activity for " << MAX_INACTIVITY << " sec., "
                 << "closing all files for sender " 
                 << XrdMonSenderInfo::id2HostPortStr(i) << endl;
            _writers[i]->forceClose();
        }
    }
}

void
XrdMonCtrArchiver::archivePacket(XrdMonCtrPacket* p)
{
    XrdMonHeader header;
    header.decode(p->buf);

    if ( XrdMonCtrAdmin::isAdminPacket(header) ) {
        kXR_int16 command = 0, arg = 0;
        XrdMonCtrAdmin::decodeAdminPacket(p->buf, command, arg);
        XrdMonCtrAdmin::doIt(command, arg);
        return;
    }

    senderid_t senderId = XrdMonSenderInfo::convert2Id(p->sender);

    XrdMonCtrWriter* w = 0;
    
    if ( _writers.size() <= senderId ) {
        w = new XrdMonCtrWriter(senderId, header.stod());
        _writers.push_back(w);
    } else {
        w = _writers[senderId];
        if ( w->prevStod() != header.stod() ) {
            cout << "\n* * * *   XRD RESTARTED for " 
                 << XrdMonSenderInfo::id2HostPortStr(senderId) 
                 << ": " << w->prevStod() << " != " << header.stod()
                 << "    * * * *\n" << endl;
            delete w;
            _writers[senderId] = w = 
                new XrdMonCtrWriter(senderId, header.stod());
            _decoder->reset(senderId);
        }
    }
    
    w->operator()(p->buf, header, _currentTime);

    if ( 0 != _decoder ) {
        _decoder->operator()(header, p->buf, senderId);
    }    
}
