/*****************************************************************************/
/*                                                                           */
/*                        XrdMonDecPacketDecoder.cc                          */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonHeader.hh"
#include "XrdMon/XrdMonUtils.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonDecPacketDecoder.hh"
#include "XrdMon/XrdMonDecTraceInfo.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdXrootd/XrdXrootdMonData.hh"
#include <stdio.h>
#include <netinet/in.h>
using std::cerr;
using std::cout;
using std::endl;

// for light decoding in real time
XrdMonDecPacketDecoder::XrdMonDecPacketDecoder(const char* baseDir, 
                                               const char* rtLogDir, 
                                               int rtBufSize)
    : _sink(baseDir, rtLogDir, rtBufSize, false, 2),
      _stopNow(false),
      _upToTime(0)
{}

XrdMonDecPacketDecoder::XrdMonDecPacketDecoder(const char* baseDir,
                                               bool saveTraces,
                                               int maxTraceLogSize,
                                               kXR_int32 upToTime)
    : _sink(baseDir, 0, 8, saveTraces, maxTraceLogSize),
      _stopNow(false),
      _upToTime(upToTime)
{}

void
XrdMonDecPacketDecoder::init(dictid_t min, 
                             dictid_t max, 
                             const string& senderHP)
{
    _sink.init(min, max, senderHP);
}    

// true if up-to-time reached and decoding should stop
void
XrdMonDecPacketDecoder::operator()(const XrdMonHeader& header,
                                   const char* packet,
                                   senderid_t senderId)
{
    int len = header.packetLen() - HDRLEN;
    //cout << "header " << header << endl;
    
    if ( len < 1 ) {
        cout << "Warning: Ignoring empty packet" << endl;
        return;
    }

    if ( header.stodChanged(senderId) ) {
        _sink.registerXrdRestart(header.stod(), senderId);
    }

    switch ( header.packetType() ) {
        case PACKET_TYPE_TRACE : {
            decodeTracePacket(packet+HDRLEN, len, senderId);
            break;
        }
        case PACKET_TYPE_DICT : {
            decodeDictPacket(packet+HDRLEN, len, senderId);
            break;
        }
        case PACKET_TYPE_USER : {
            decodeUserPacket(packet+HDRLEN, len, senderId);
            break;
        }
        case PACKET_TYPE_STAGE : {
            decodeStagePacket(packet+HDRLEN, len, senderId);
            break;
        }
        
        default: {
            cerr << "Unsupported packet type: " << header.packetType() << endl;
        }
    }
    _sink.setLastSeq(header.seqNo());
}

void
XrdMonDecPacketDecoder::reset(senderid_t senderId)
{
    _sink.reset(senderId);
}

// packet should point to data after header
void
XrdMonDecPacketDecoder::decodeTracePacket(const char* packet, 
                                          int len, 
                                          senderid_t senderId)
{
    // decode first packet - time window
    if ( static_cast<kXR_char>(*packet) != XROOTD_MON_WINDOW ) {
        char buf[256];
        sprintf(buf, "Expected time window packet (1st packet), got %i, ", (int)*packet);
        throw XrdMonException(ERR_NOTATIMEWINDOW, buf);
    }
    TimePair t = decodeTime(packet);
    if ( _upToTime != 0 && _upToTime <= t.first ) {
        cout << "reached the up-to-time, will stop decoding now" << endl;
        _stopNow = true;
        return;
    }
    kXR_int32 begTime = t.second;
    int offset = TRACELEN;

    //cout << "Decoded time (first) " << t.first << " " << t.second << endl;
    
    while ( offset < len ) {
        CalcTime ct = prepareTimestamp(packet, offset, len, begTime);
        int elemNo = 0;
        while ( offset<ct.endOffset ) {
            kXR_char infoType = static_cast<kXR_char>(*(packet+offset));
            kXR_int32 timestamp = begTime + (kXR_int32) (elemNo++ * ct.timePerTrace);
            if ( !(infoType & XROOTD_MON_RWREQUESTMASK) ) {
                decodeRWRequest(packet+offset, timestamp, senderId);
            } else if ( infoType == XROOTD_MON_OPEN ) {
                decodeOpen(packet+offset, timestamp, senderId);
            } else if ( infoType == XROOTD_MON_CLOSE ) {
                decodeClose(packet+offset, timestamp, senderId);
            } else if ( infoType == XROOTD_MON_DISC ) {
                decodeDisconnect(packet+offset, timestamp, senderId);
            } else {
                char buf[256];
                sprintf(buf, "Unsupported infoType of trace packet: %i", (int) infoType);
                throw XrdMonException(ERR_INVALIDINFOTYPE, buf);
            }
            offset += TRACELEN;
        }
        begTime = ct.begTimeNextWindow;
        offset += TRACELEN; // skip window trace which was already read
    }
}

// packet should point to data after header
void
XrdMonDecPacketDecoder::decodeDictPacket(const char* packet, 
                                         int len, 
                                         senderid_t senderId)
{
    kXR_int32 x32;
    memcpy(&x32, packet, sizeof(kXR_int32));
    dictid_t dictId = ntohl(x32);
    
    _sink.addDictId(dictId, 
                    packet+sizeof(kXR_int32), 
                    len-sizeof(kXR_int32),
                    senderId);
}

// packet should point to data after header
void
XrdMonDecPacketDecoder::decodeUserPacket(const char* packet,
                                         int len,
                                         senderid_t senderId)
{
    kXR_int32 x32;
    memcpy(&x32, packet, sizeof(kXR_int32));
    dictid_t dictId = ntohl(x32);
    
    _sink.addUserId(dictId,
                    packet+sizeof(kXR_int32),
                    len-sizeof(kXR_int32),
                    senderId);
}

// packet should point to data after header
void
XrdMonDecPacketDecoder::decodeStagePacket(const char* packet, 
                                          int len, 
                                          senderid_t senderId)
{
    kXR_int32 x32;
    memcpy(&x32, packet, sizeof(kXR_int32));
    dictid_t dictId = ntohl(x32);

    _sink.addStageInfo(dictId, 
                       packet+sizeof(kXR_int32), 
                       len-sizeof(kXR_int32),
                       senderId);
}

XrdMonDecPacketDecoder::TimePair
XrdMonDecPacketDecoder::decodeTime(const char* packet)
{
    struct X {
        kXR_int32 endT;
        kXR_int32 begT;
    } x;

    memcpy(&x, packet+sizeof(kXR_int64), sizeof(X));
    return TimePair(ntohl(x.endT), ntohl(x.begT));
}

void
XrdMonDecPacketDecoder::decodeRWRequest(const char* packet,
                                        kXR_int32 timestamp, 
                                        senderid_t senderId)
{
    XrdXrootdMonTrace trace;
    memcpy(&trace, packet, sizeof(XrdXrootdMonTrace));
    kXR_int64 tOffset = ntohll(trace.arg0.val);
    kXR_int32 tLen    = ntohl (trace.arg1.buflen);
    kXR_unt32 dictId  = ntohl (trace.arg2.dictid);

    if ( tOffset < 0 ) {
        throw XrdMonException(ERR_NEGATIVEOFFSET);
    }
    char rwReq = 'r';
    if ( tLen<0 ) {
        rwReq = 'w';
        tLen *= -1;
    }

    XrdMonDecTraceInfo traceInfo(tOffset, tLen, rwReq, timestamp);
    _sink.add(dictId, traceInfo, senderId);
}

void
XrdMonDecPacketDecoder::decodeOpen(const char* packet,
                                   kXR_int32 timestamp, 
                                   senderid_t senderId)
{
    XrdXrootdMonTrace trace;
    memcpy(&trace, packet, sizeof(XrdXrootdMonTrace));
    kXR_unt32 dictId = ntohl(trace.arg2.dictid);
    // mask needed to hide id[0] value which is stored
    // in the top most byte
#if _FILE_OFFSET_BITS==64
    kXR_int64 maskHBO = 0x00ffffffffffffffLL;
#else
    kXR_int64 maskHBO = 0x00ffffff;
#endif
    kXR_int64 maskNBO = 0;
    h2nll(maskHBO, maskNBO);
    kXR_int64 maskedVal = trace.arg0.val & maskNBO;
    kXR_int64 fSize = 0;
    n2hll(maskedVal, fSize);
    
    _sink.openFile(dictId, timestamp, senderId, fSize);
}

void
XrdMonDecPacketDecoder::decodeClose(const char* packet,
                                    kXR_int32 timestamp, 
                                    senderid_t senderId)
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

    //cout << "decoded close file, dict " << dictId 
    //     << ", total r " << tR << " shifted " << (int) rShift << ", or " << realR
    //     << ", total w " << tW << " shifted " << (int) wShift << ", or " << realW
    //     << endl;

    _sink.closeFile(dictId, realR, realW, timestamp, senderId);
}

void
XrdMonDecPacketDecoder::decodeDisconnect(const char* packet, 
                                         kXR_int32 timestamp, 
                                         senderid_t senderId)
{
    XrdXrootdMonTrace trace;
    memcpy(&trace, packet, sizeof(XrdXrootdMonTrace));
    kXR_int32 sec    = ntohl(trace.arg1.buflen);
    kXR_unt32 dictId = ntohl(trace.arg2.dictid);

    cout << "decoded user disconnect, dict " << dictId
         << ", sec = " << sec << ", t = " << timestamp << endl;

    _sink.addUserDisconnect(dictId, sec, timestamp, senderId);
}

XrdMonDecPacketDecoder::CalcTime
XrdMonDecPacketDecoder::prepareTimestamp(const char* packet, 
                                         int& offset, 
                                         int len, 
                                         kXR_int32& begTime)
{
    // look for time window
    int x = offset;
    int noElems = 0;
    while ( static_cast<kXR_char>(*(packet+x)) != XROOTD_MON_WINDOW ) {
        if ( x >= len ) {
            throw XrdMonException(ERR_NOTATIMEWINDOW, 
                              "Expected time window packet (last packet)");
        }
        x += TRACELEN;
        ++noElems;
    }

    // cout << "Found timestamp, offset " << x 
    //     << " after " << noElems << " elements" << endl;
    
    // decode time window
    TimePair t = decodeTime(packet+x);

    // cout << "decoded time " << t.first << " " << t.second << endl;
    
    if ( begTime > t.first ) {
        char buf[256];
        sprintf(buf, "Wrong time: %d > %d at offset %d, will fix", begTime, t.first, x);
        cout << buf << endl;
        begTime = t.first;
        //throw XrdMonException(ERR_INVALIDTIME, buf);
    }

    float timePerTrace = ((float)(t.first - begTime)) / noElems;
    //cout << "timepertrace = " << timePerTrace << endl;
    
    return CalcTime(timePerTrace, t.second, x);
}
