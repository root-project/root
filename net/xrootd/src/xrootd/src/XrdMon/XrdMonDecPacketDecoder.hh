/*****************************************************************************/
/*                                                                           */
/*                        XrdMonDecPacketDecoder.hh                          */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$
#ifndef XRDMONDECPACKETDECODER_HH
#define XRDMONDECPACKETDECODER_HH

#include "XrdMon/XrdMonHeader.hh"
#include "XrdMon/XrdMonDecSink.hh"
#include <utility> // for pair
#include <sys/time.h>
using std::pair;

class XrdMonDecPacketDecoder {
public:
    XrdMonDecPacketDecoder(const char* baseDir, 
                           const char* rtLogDir,
                           int rtBufSize);

    XrdMonDecPacketDecoder(const char* baseDir,
                           bool saveTraces,
                           int maxTraceLogSize,
                           kXR_int32 upToTime);

    void init(dictid_t min, dictid_t max, const string& senderHP);
    sequen_t lastSeq() const { return _sink.lastSeq(); }
    
    void operator()(const XrdMonHeader& header,
                    const char* packet,
                    senderid_t senderId=INVALID_SENDER_ID);

    void reset(senderid_t senderId);
    
    bool     stopNow() const { return _stopNow; }

    void flushHistoryData() { return _sink.flushHistoryData(); }
    void flushRealTimeData() { return _sink.flushRealTimeData(); }
    
private:
    typedef pair<kXR_int32, kXR_int32> TimePair; // <beg time, end time>

    struct CalcTime {
        CalcTime(float f, kXR_int32 t, int e)
            : timePerTrace(f), begTimeNextWindow(t), endOffset(e) {}
        float  timePerTrace;
        kXR_int32 begTimeNextWindow;
        int    endOffset;
    };
    
    CalcTime& f();
    
    typedef pair<float, kXR_int32> FloatTime; // <time per trace, beg time next wind>

    void checkLostPackets(const XrdMonHeader& header);
    
    void decodeTracePacket(const char* packet,
                           int packetLen,
                           senderid_t senderId);
    void decodeDictPacket(const char* packet,
                          int packetLen,
                          senderid_t senderId);
    void decodeUserPacket(const char* packet,
                          int packetLen,
                          senderid_t senderId);
    void decodeStagePacket(const char* packet,
                           int packetLen,
                           senderid_t senderId);

    TimePair decodeTime(const char* packet);
    void decodeRWRequest(const char* packet,
                         kXR_int32 timestamp,
                         senderid_t senderId);
    void decodeOpen(const char* packet,
                    kXR_int32 timestamp,
                    senderid_t senderId);
    void decodeClose(const char* packet,
                     kXR_int32 timestamp,
                     senderid_t senderId);
    void decodeDisconnect(const char* packet,
                          kXR_int32 timestamp,
                          senderid_t senderId);

    CalcTime prepareTimestamp(const char* packet, 
                              int& offset, 
                              int len, 
                              kXR_int32& begTime);
private:
    XrdMonDecSink _sink;
    bool          _stopNow;

    kXR_int32     _upToTime; // for decoding parts of log file
};

#endif /* XRDMONDECPACKETDECODER_HH */
