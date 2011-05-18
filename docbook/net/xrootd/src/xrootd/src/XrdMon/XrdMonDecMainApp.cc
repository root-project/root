/*****************************************************************************/
/*                                                                           */
/*                            XrdMonDecMainApp.cc                            */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonDecArgParser.hh"
#include "XrdMon/XrdMonDecOnePacket.hh"
#include "XrdMon/XrdMonDecPacketDecoder.hh"
#include "XrdMon/XrdMonDecPreProcess.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonHeader.hh"
#include "XrdMon/XrdMonTimer.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <unistd.h>   /* access */
#include <fstream>
#include <iomanip>
using std::cout;
using std::ios;
using std::setw;

// use this switch if you want to brute-close all files in the active dir.
// Useful to run after the very last log file

void
closeFiles()
{
    cout << "brute close file not implemented FIXME" << endl;
    ::abort();
}

void
doDecoding()
{
    const char* fName = XrdMonDecArgParser::_fPath.c_str();
    
    if ( 0 != access(fName, F_OK) ) {
        string s("Invalid log file name ");
        s += fName;
        throw XrdMonException(ERR_INVALIDARG, s);
    }

    // here decoder should read all last seqno and unique id from jnl file
    XrdMonDecPacketDecoder decoder(XrdMonDecArgParser::_baseDir.c_str(),
                                   XrdMonDecArgParser::_saveTraces,
                                   XrdMonDecArgParser::_maxTraceLogSize,
                                   XrdMonDecArgParser::_upToTime);

    sequen_t lastSeq = decoder.lastSeq();
        
    // open file, find length and prepare for reading
    fstream _file;
    _file.open(fName, ios::in|ios::binary);
    _file.seekg(0, ios::end);
    kXR_int64 fSize = _file.tellg();
    _file.seekg(0, ios::beg);

    // preprocess: includes catching out of order packets
    vector< pair<packetlen_t, kXR_int64> > allPackets;
    XrdMonDecPreProcess pp(_file, 
                           fSize, 
                           lastSeq, 
                           XrdMonDecArgParser::_ignoreIfBefore,
                           allPackets);
    pp();

    // here it shoudl read all active dicts...
    decoder.init(XrdMonDecOnePacket::minDictId(), 
                 XrdMonDecOnePacket::maxDictId(),
                 XrdMonDecArgParser::_hostPort);         

    XrdMonHeader header;
    char packet[MAXPACKETSIZE];
    int noPackets = allPackets.size();
    for ( int packetNo = 0 ; packetNo<noPackets ; ++ packetNo ) {
        packetlen_t len = allPackets[packetNo].first;
        kXR_int64 pos     = allPackets[packetNo].second;
        if ( pos == -1 && len == 0 ) {
            cout << "Lost #" << packetNo << endl;
            continue;
        }
        cout << "Decoding #" << packetNo 
             << ", tellg " << setw(10) << pos << endl;
        _file.seekg(pos);
        memset(packet, 0, MAXPACKETSIZE);
        _file.read(packet, len);
        header.decode(packet);
        decoder(header, packet+HDRLEN);

        if ( decoder.stopNow() ) {
            break;
        }   
    }

    _file.close();
    //LogMgr::storeLastSeqNo(decoder.lastSeqNo());
}

int main(int argc, char* argv[])
{
    XrdMonTimer t;
    t.start();
    try {
        XrdMonDecArgParser::parseArguments(argc, argv);
    } catch (XrdMonException& e) {
        e.printIt();
        cout << "Expected arguments: <logName> [-isActive <host>:<port> YYYY MM DD HH MM SS] [-ignoreIfBefore <unix_timestamp>] [-forceCloseOnly]\n"
             << "Use \"-isActive\" only when the logfile is active.rcv\n"
             << "up-till-date means that decoding will be stoped when given date is reached\n"
             << "-ignoreIfBefore can be used to ignore packets that arrived before certain date/time (e.g. from \"previous\" xrootd). Useful during xrootd/collector restarts\n"
             << "Use -forceCloseOnly to force closing all active files that are in active dir\n"
             << "\n"
             << "Examples:\n"
             << "  decoder logs/receiver/kan025/61145/20041102_11:57:51.073_kan025:61145.rcv\n"
             << "  decoder logs/receiver/kan025/61145/20041102_11:57:51.073_kan025:61145.rcv -forceCloseOnly\n"
             << "  decoder logs/receiver/kan025/61145/active.rcv -isActive kan025:61145 2004 11 02 12 45 12\n"
             << endl;
        return 1;
    }
    
    try{
        if ( XrdMonDecArgParser::_forceCloseOnly ) {
            closeFiles();
        } else {
            doDecoding();
        }
    } catch (XrdMonException& e) {
        e.printIt();
        return 1;
    }
    
    cout << "Total time: " << t.stop() << endl;
    //cout << "Publishing: " << LogMgr::_t1.getElapsed() << endl;
    //cout << "Flushing:   " << LogMgr::_t2.getElapsed() << endl;
    
    return 0;
}
