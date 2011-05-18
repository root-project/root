/*****************************************************************************/
/*                                                                           */
/*                            XrdMonSndCoder.cc                              */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonSndCoder.hh"
#include "XrdXrootd/XrdXrootdMonData.hh"
#include <sys/time.h>
#include <iomanip>
using std::setw;

kXR_int32 XrdMonSndCoder::_serverStartTime = 0;

XrdMonSndCoder::XrdMonSndCoder()
    : _sequenceNo(0),
      _noDict(0),
      _noOpen(0),
      _noClose(0),
      _noTrace(0),
      _noTime(0)
{
    if ( 0 == _serverStartTime ) {
        struct timeval tv;
        gettimeofday(&tv, 0);
        _serverStartTime = tv.tv_sec;
    }
}

int
XrdMonSndCoder::prepare2Transfer(const XrdMonSndAdminEntry& ae)
{
    kXR_int32 packetSize = HDRLEN + ae.size();

    int ret = reinitXrdMonSndPacket(packetSize, PACKET_TYPE_ADMIN);
    if ( 0 != ret ) {
        return ret;
    }

    add_kXR_int16(ae.command());
    add_kXR_int16(ae.arg());

    return 0;
}

int
XrdMonSndCoder::prepare2Transfer(const vector<XrdMonSndTraceEntry>& vector)
{
    kXR_int16 noElems = vector.size() + 3; // 3: 3 time entries
    if (vector.size() == 0 ) {
        noElems = 0;
    }
    
    kXR_int32 packetSize = HDRLEN + noElems * TRACEELEMLEN;
    if ( packetSize > MAXPACKETSIZE ) {
        cerr << "Internal error: cached too many entries: " << noElems
             << ", MAXPACKETSIZE = " << MAXPACKETSIZE;
        noElems = (MAXPACKETSIZE-HDRLEN) / TRACEELEMLEN;
        cerr << " Will send only " << noElems << endl;
    }

    int ret = reinitXrdMonSndPacket(packetSize, PACKET_TYPE_TRACE);
    if ( 0 != ret ) {
        return ret;
    }

    kXR_int16 middle = noElems/2;
    kXR_int32 curTime = time(0);
    for (kXR_int16 i=0 ; i<noElems-3 ; i++ ) {
        if (i== 0) { // add time entry
            add_Mark(XROOTD_MON_WINDOW);
            add_kXR_int32(curTime); // prev window ended
            add_kXR_int32(curTime);   // this window started
            ++_noTime;
            if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
                cout << "Adding time window {" << curTime << ", " 
                     << curTime << "}" << ", elem no " << i << endl;
            }
        }
        if (i== middle) { // add time entry
            add_Mark(XROOTD_MON_WINDOW);
            add_kXR_int32(curTime); // prev window ended
            add_kXR_int32(curTime);   // this window started
            ++_noTime;
            if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
                cout << "Adding time window {" << curTime << ", " 
                     << curTime << "}" << ", elem no " << i << endl;
            }
        }
        const XrdMonSndTraceEntry& de = vector[i];
        add_kXR_int64(de.offset());
        add_kXR_int32(de.length());
        add_kXR_int32(de.id()    );
        if (i==noElems-4) {
            add_Mark(XROOTD_MON_WINDOW);
            add_kXR_int32(curTime); // prev window ended
            add_kXR_int32(curTime);   // this window started
            ++_noTime;
            if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
                cout << "Adding time window {" << curTime << ", " 
                     << curTime << "}" << ", elem no " << i << endl;
            }
        }
    }
    _noTrace += vector.size();
    
    return 0;
}


pair<char, kXR_unt32>
XrdMonSndCoder::generateBigNumber(const char* descr)
{
    kXR_int64 xOrg = 1000000000000LL + rand();
    char nuToShift = 0;
    kXR_int64 x = xOrg;
    while ( x > 4294967296LL ) {
        ++nuToShift;
        x = x >> 1;
    }
    cout << "Want to send #" << descr << " " << xOrg
         << ", sending " << x << " noShifted " 
         << (int) nuToShift << endl;

    return pair<char, kXR_unt32>(nuToShift, static_cast<kXR_unt32>(x));
}

int 
XrdMonSndCoder::prepare2Transfer(const vector<kXR_int32>& vector)
{
    kXR_int16 noElems = vector.size() + 2; // 2: 2 time entries
    int8_t sizeOfXrdMonSndTraceEntry = sizeof(kXR_int64)+sizeof(kXR_int32)+sizeof(kXR_int32);
    kXR_int32 packetSize = HDRLEN + noElems * sizeOfXrdMonSndTraceEntry;
    if ( packetSize > MAXPACKETSIZE ) {
        cerr << "Internal error: cached too many entries: " << noElems
             << ", MAXPACKETSIZE = " << MAXPACKETSIZE;
        noElems = (MAXPACKETSIZE-HDRLEN) / sizeOfXrdMonSndTraceEntry;
        cerr << " Will send only " << noElems << endl;
    }

    int ret = reinitXrdMonSndPacket(packetSize, PACKET_TYPE_TRACE);
    if ( 0 != ret ) {
        return ret;
    }

    kXR_int32 curTime = time(0);

    struct XrdXrootdMonTrace trace;
    memset(&trace, 0, sizeof(XrdXrootdMonTrace));
    trace.arg0.id[0]  = XROOTD_MON_WINDOW;
    trace.arg1.Window = 
    trace.arg2.Window = htonl(curTime);
    memcpy(writeHere(), &trace, sizeof(XrdXrootdMonTrace));
    _putOffset += sizeof(XrdXrootdMonTrace);
    
    ++_noTime;
    if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
        cout << "Adding time window {" << curTime << ", " 
             << curTime << "}" << ", elem no 0" << endl;
    }

    for (kXR_int16 i=0 ; i<noElems-2 ; i++ ) {
        static int largeNr = 0; // from time to time need to send very large nr
        kXR_unt32 rT, wT;

        memset(&trace, 0, sizeof(XrdXrootdMonTrace));
        trace.arg0.id[0]   = XROOTD_MON_CLOSE;
        if ( ++ largeNr % 11 == 10 ) {
            // generate # bytes read/writen (larger than 2^32, shifted)
            pair<char, kXR_unt32> bigR = generateBigNumber("read");
            pair<char, kXR_unt32> bigW = generateBigNumber("write");

            trace.arg0.id[1] = bigR.first;
            trace.arg0.id[2] = bigW.first;
            rT = bigR.second;
            wT = bigW.second;
        } else {
            // generate # bytes read/writen (smaller than 2^32)
            rT = (kXR_unt32) rand();
            wT = (kXR_unt32) rand() / 512;
        }

        trace.arg0.rTot[1] = htonl(rT);
        trace.arg1.wTot    = htonl(wT);
        trace.arg2.dictid  = htonl(vector[i]);
        memcpy(writeHere(), &trace, sizeof(XrdXrootdMonTrace));
        _putOffset += sizeof(XrdXrootdMonTrace);

        ++_noClose;
        cout << "closing file, dictid " << vector[i] 
             << ", r=" << rT
             << ", w=" << wT << endl;
    }

    memset(&trace, 0, sizeof(XrdXrootdMonTrace));
    trace.arg0.id[0]  = XROOTD_MON_WINDOW;
    trace.arg1.Window = 
    trace.arg2.Window = htonl(curTime);
    memcpy(writeHere(), &trace, sizeof(XrdXrootdMonTrace));
    _putOffset += sizeof(XrdXrootdMonTrace);

    ++_noTime;
    if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
        cout << "Adding time window {" << curTime << ", " 
             << curTime << "}" << ", elem no " << noElems-2 << endl;
    }
    return 0;
}



int
XrdMonSndCoder::prepare2Transfer(const XrdMonSndDictEntry::CompactEntry& ce)
{
    kXR_int32 packetSize = HDRLEN + ce.size();

    int ret = reinitXrdMonSndPacket(packetSize, PACKET_TYPE_DICT);
    if ( 0 != ret ) {
        return ret;
    }

    add_kXR_int32(ce.id);
    add_string  (ce.others);

    ++_noDict;
    
    return 0;
}


int
XrdMonSndCoder::prepare2Transfer(const XrdMonSndStageEntry::CompactEntry& ce)
{
    kXR_int32 packetSize = HDRLEN + ce.size();

    int ret = reinitXrdMonSndPacket(packetSize, PACKET_TYPE_STAGE);
    if ( 0 != ret ) {
        return ret;
    }

    add_kXR_int32(ce.id);
    add_string  (ce.others);

    ++_noDict;
    
    return 0;
}

int
XrdMonSndCoder::reinitXrdMonSndPacket(packetlen_t newSize, char packetCode)
{
    _putOffset = 0;
    int ret = _packet.init(newSize);
    if ( 0 != ret ) {
        return ret;
    }

    if ( XrdMonSndDebug::verbose(XrdMonSndDebug::SPacket) ) {
        cout << "XrdMonSndPacket " << packetCode 
             << ", size " << setw(5) << newSize 
             << ", sequenceNo " << setw(3) << (int) _sequenceNo 
             << ", time " << _serverStartTime
             << " prepared for sending" << endl;
    }
    add_int08_t(packetCode);
    add_int08_t(_sequenceNo++);
    add_kXR_unt16(newSize);
    add_kXR_int32(_serverStartTime);

    return 0;
}

void
XrdMonSndCoder::printStats() const
{
    cout <<   "dict="    << _noDict
         << ", noOpen="  << _noOpen
         << ", noClose=" << _noClose
         << ", noTrace=" << _noTrace
         << ", noTime="  << _noTime << endl;
}

