/*****************************************************************************/
/*                                                                           */
/*                          XrdMonDebugPacketApp.cc                          */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonDecArgParser.hh"
#include "XrdMon/XrdMonDecDictInfo.hh"
#include "XrdMon/XrdMonDecUserInfo.hh"
#include "XrdMon/XrdMonHeader.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdXrootd/XrdXrootdMonData.hh"
#include <fstream>
#include <iomanip>
#include <netinet/in.h>
#include <unistd.h>

using std::cerr;
using std::cout;
using std::endl;
using std::fstream;
using std::ios;
using std::pair;
using std::setw;

typedef pair<kXR_int32, kXR_int32> TimePair; // <beg time, end time>

struct CalcTime {
    CalcTime(float f, kXR_int32 t, int e)
        : timePerTrace(f), begTimeNextWindow(t), endOffset(e) {}
    float  timePerTrace;
    kXR_int32 begTimeNextWindow;
    int    endOffset;
};

TimePair
decodeTime(const char* packet)
{
    struct X {
        kXR_int32 endT;
        kXR_int32 begT;
    } x;
    memcpy(&x, packet+sizeof(kXR_int64), sizeof(X));
    return TimePair(ntohl(x.endT), ntohl(x.begT));
}

CalcTime
prepareTimestamp(const char* packet, 
                 int& offset, 
                 int len, 
                 kXR_int32& begTime)
{
    // look for time window
    int x = offset;
    int noElems = 0;
    while ( static_cast<kXR_char>(*(packet+x)) != XROOTD_MON_WINDOW ) {
        if ( x >= len ) {
            cerr << "Error: expected time window packet (last packet)" << endl;
            exit(1);
        }
        x += TRACELEN;
        ++noElems;
    }

    // decode time window
    TimePair t = decodeTime(packet+x);
    cout << "offset " << setw(5) << x
         << " - location of next timepair: {" 
         << t.first << ", " << t.second << "}. " 
         << noElems << " traces in between" << endl;

    if ( begTime > t.first ) {
        cout << "Error: wrong time: " << begTime 
             << " > " << t.first << " at offset " << x 
             << ", will use begTime == endTime" << endl;
        begTime = t.first;
    }

    float timePerTrace = ((float)(t.first - begTime)) / noElems;
    cout << "will use following time per trace = " << timePerTrace << endl;
    
    return CalcTime(timePerTrace, t.second, x);
}

void
debugRWRequest(const char* packet, kXR_int32 timestamp, kXR_int64 offset)
{
    struct X {
        kXR_int64 tOffset;
        kXR_int32 tLen;
        kXR_int32 dictId;
    } x;
    memcpy(&x, packet, sizeof(X));
    x.tOffset = ntohll(x.tOffset);
    x.tLen = ntohl(x.tLen);
    x.dictId = ntohl(x.dictId);

    if ( x.tOffset < 0 ) {
        cerr << "Error negative offset" << endl;
        exit(1);
    }
    char rwReq = 'r';
    if ( x.tLen<0 ) {
        rwReq = 'w';
        x.tLen *= -1;
    }
    cout << "offset " << setw(5) << offset
         << " --> trace: offset=" << x.tOffset << " len=" << x.tLen
         << " rw=" << rwReq << " timestamp=" << timestamp << endl;
}

void
debugOpen(const char* packet, 
          kXR_int32 timestamp, 
          kXR_int64 offset)
{
    kXR_int32 dictId;
    memcpy(&dictId, 
           packet+sizeof(kXR_int64)+sizeof(kXR_int32), 
           sizeof(kXR_int32));
    dictId = ntohl(dictId);

    cout << "offset " << setw(5) << offset
         << " --> open " << " dictId = " << dictId
         << ", timestamp = " << timestamp << endl;
}

void
debugClose(const char* packet, 
          kXR_int32 timestamp, 
          kXR_int64 offset)
{
    XrdXrootdMonTrace trace;
    memcpy(&trace, packet, sizeof(XrdXrootdMonTrace));
    kXR_unt32 dictId = ntohl(trace.arg2.dictid);
    kXR_unt32 tR     = ntohl(trace.arg0.rTot[1]);
    kXR_unt32 tW     = ntohl(trace.arg1.wTot);
    char rShift      = trace.arg0.id[1];
    char wShift      = trace.arg0.id[2];
    kXR_int64 realR  = tR; realR = realR << rShift;
    kXR_int64 realW  = tW; realW = realW << wShift;

    cout << "offset " << setw(5) << offset
         << " --> close " << " dictId = " << dictId
         << ", timestamp = " << timestamp 
         << ", total r " <<tR<< " shifted " << (int) rShift << ", or " << realR
         << ", total w " <<tW<< " shifted " << (int) wShift << ", or " << realW
         << endl;
}


void
debugDictPacket(const char* packet, int len)
{
    kXR_int32 x32;
    memcpy(&x32, packet, sizeof(kXR_int32));
    dictid_t dictId = ntohl(x32);
    
    XrdMonDecDictInfo de(dictId, -1, 
                         packet+sizeof(kXR_int32), 
                         len-sizeof(kXR_int32), 
                         0);
    cout << "offset " << setw(5) << HDRLEN
         << " --> " << de << endl;
}

void
debugUserPacket(const char* packet, int len)
{
    kXR_int32 x32;
    memcpy(&x32, packet, sizeof(kXR_int32));
    dictid_t dictId = ntohl(x32);
    
    XrdMonDecUserInfo du(dictId, -1, 
                         packet+sizeof(kXR_int32), 
                         len-sizeof(kXR_int32),
                         0);
    cout << "offset " << setw(5) << HDRLEN
         << " --> " << du << endl;
}

void
debugDisconnect(const char* packet, int len)
{
    XrdXrootdMonTrace trace;
    memcpy(&trace, packet, sizeof(XrdXrootdMonTrace));
    kXR_int32 sec    = ntohl(trace.arg1.buflen);
    kXR_unt32 dictId = ntohl(trace.arg2.dictid);

    cout << "offset " << setw(5) << HDRLEN
         << " --> user disconnect, dict " << dictId
         << ", sec = " << sec << endl;
}

void
debugStagePacket(const char* packet, int)
{
    cerr << "DebugStagePacket() not implemented" << endl;
}

void
debugTracePacket(const char* packet, int len)
{
    if ( static_cast<kXR_char>(*packet) != XROOTD_MON_WINDOW ) {
        cerr << "Expected time window packet (1st packet), got " 
             << (int) *packet << endl;
        return;
    }
    TimePair t = decodeTime(packet);
    cout << "offset " << setw(5) << HDRLEN
         << ", timepair: {" << t.first << ", " << t.second << "}" << endl;

    kXR_int32 begTime = t.second;
    int offset = TRACELEN;

    while ( offset < len ) {
        CalcTime ct = prepareTimestamp(packet, offset, len, begTime);
        int elemNo = 0;
        while ( offset<ct.endOffset ) {
            kXR_char infoType = static_cast<kXR_char>(*(packet+offset));
            kXR_int32 timestamp = begTime + (kXR_int32) (elemNo++ * ct.timePerTrace);
            if ( !(infoType & XROOTD_MON_RWREQUESTMASK) ) {
                cout << "offset " << setw(5) << offset 
                     << " --> XROOTD_MON_RWREQUESTMAST" << endl;
                debugRWRequest(packet+offset, timestamp, offset+HDRLEN);
            } else if ( infoType == XROOTD_MON_OPEN ) {
                cout << "offset " << setw(5) << offset 
                     << " --> XROOTD_MON_OPEN" << endl;
                debugOpen(packet+offset, timestamp, offset+HDRLEN);
            } else if ( infoType == XROOTD_MON_CLOSE ) {
                cout << "offset " << setw(5) << offset 
                     << " --> XROOTD_MON_CLOSE" << endl;
                debugClose(packet+offset, timestamp, offset+HDRLEN);
            } else if ( infoType == XROOTD_MON_DISC ) {
                debugDisconnect(packet+offset, offset+HDRLEN);

            } else {
                cerr << "Unsupported infoType of trace packet: " 
                     << infoType << endl;
                return;
            }
            offset += TRACELEN;
        }
        begTime = ct.begTimeNextWindow;
        offset += TRACELEN; // skip window trace which was already read
    }
}

int main(int argc, char* argv[])
{
    if ( argc != 2 ) {
        cerr << "Expected input file path" << endl;
        return 1;
    }
    const char* fName = argv[1];
    if ( 0 != access(fName, F_OK) ) {
        cerr << "Cannot open " << fName << endl;
        return 2;
    }
    
    fstream _file;
    _file.open(fName, ios::in|ios::binary);
    _file.seekg(0, ios::beg);
 
    // read header, dump to file
    char hBuffer[HDRLEN];
    _file.read(hBuffer, HDRLEN);

    // read and decode header
    XrdMonHeader header;
    header.decode(hBuffer);
    cout << "offset " << setw(5) << 0 
         << " header:" << header << endl;

    int len = header.packetLen() - HDRLEN;

    // read and decode packet
    char packet[MAXPACKETSIZE];
    _file.read(packet, len);

    switch (header.packetType() ) {
        case PACKET_TYPE_TRACE: { debugTracePacket(packet, len); break; }
        case PACKET_TYPE_DICT:  { debugDictPacket(packet, len);  break; }
        case PACKET_TYPE_STAGE: { debugStagePacket(packet, len); break; }
        case PACKET_TYPE_USER:  { debugUserPacket(packet, len);  break; }
        default: {
            cerr << "Invalid packet type " << header.packetType() << endl;
            return 1;
        }
    }
    
    _file.close();
    
    return 0;
}
