/*****************************************************************************/
/*                                                                           */
/*                        XrdMonSndDummyXrootdApp.cc                         */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonArgParser.hh"
#include "XrdMon/XrdMonArgParserConvert.hh"
#include "XrdMon/XrdMonUtils.hh"
#include "XrdMon/XrdMonSndCoder.hh"
#include "XrdMon/XrdMonSndDebug.hh"
#include "XrdMon/XrdMonSndDictEntry.hh"
#include "XrdMon/XrdMonSndDummyXrootd.hh"
#include "XrdMon/XrdMonSndTraceCache.hh"
#include "XrdMon/XrdMonSndTraceEntry.hh"
#include "XrdMon/XrdMonSndTransmitter.hh"

#include <assert.h>
#include <unistd.h>  /* usleep */
#include <sys/time.h>
using namespace XrdMonArgParserConvert;

// known problems with 2 and 4
//const kXR_int64 NOCALLS = 8640000;   24h worth
const kXR_int64 NOCALLS = 1000000;
const kXR_int16 maxNoXrdMonSndPackets = 50;

void
printHelp()
{
    cout << "\nxrdmonDummySender\n"
         << "    [-host <hostName>]\n"
         << "    [-port <portNr>]\n"
         << "\n"
         << "-host <hostName>         Name of the receiver's host.\n"
         << "                         Default value is \"" << DEFAULT_HOST << "\".\n"
         << "-port <portNr>           Port number of the receiver's host\n"
         << "                         Default valus is \"" << DEFAULT_PORT << "\".\n"
         << endl;
}

void
doDictionaryXrdMonSndPacket(XrdMonSndDummyXrootd& xrootd, 
                            XrdMonSndCoder& coder,
                            XrdMonSndTransmitter& transmitter,
                            kXR_int64& noP)
{
    XrdMonSndDictEntry m = xrootd.newXrdMonSndDictEntry();
    cout << m << endl;

    XrdMonSndDictEntry::CompactEntry ce = m.code();
    
    if ( 0 == coder.prepare2Transfer(ce) ) {
        transmitter(coder.packet());
        coder.reset();
        ++noP;
    }
}

void
doStageXrdMonSndPacket(XrdMonSndDummyXrootd& xrootd, 
                       XrdMonSndCoder& coder,
                       XrdMonSndTransmitter& transmitter,
                       kXR_int64& noP)
{
    XrdMonSndStageEntry m = xrootd.newXrdMonSndStageEntry();
    cout << "about to send this stage package:"  << m << endl;

    XrdMonSndStageEntry::CompactEntry ce = m.code();
    
    if ( 0 == coder.prepare2Transfer(ce) ) {
        transmitter(coder.packet());
        coder.reset();
        ++noP;
    }
}

void
doTraceXrdMonSndPacket(XrdMonSndDummyXrootd& xrootd,
                       XrdMonSndCoder& coder, 
                       XrdMonSndTransmitter& transmitter,
                       XrdMonSndTraceCache& cache, 
                       kXR_int64& noP)
{
    XrdMonSndTraceEntry de = xrootd.newXrdMonSndTraceEntry();
    // add to buffer, perhaps transmit
    cache.add(de);
    if ( ! cache.bufferFull() ) {
        return;
    }
    
    if ( 0 == coder.prepare2Transfer(cache.getVector()) ) {
        cache.clear();
        transmitter(coder.packet());
        coder.reset();
        noP++;
    }
}

void
closeFiles(XrdMonSndDummyXrootd& xrootd,
           XrdMonSndCoder& coder, 
           XrdMonSndTransmitter& transmitter,
           kXR_int64& noP,
           bool justOne)
{
    vector<kXR_int32> closedFiles;
    if ( justOne ) {
        kXR_int32 id = xrootd.closeOneFile();
        closedFiles.push_back(id);
    } else {
        xrootd.closeFiles(closedFiles);
    }
    
    int s = closedFiles.size();
    int pos = 0;
    unsigned int i;
    while ( pos < s ) {
        vector<kXR_int32> v;
        for (i=0 ; i<XrdMonSndTraceCache::NODATAELEMS-2 && pos<s ; ++i, ++pos) {
            v.push_back(closedFiles.back());
            closedFiles.pop_back();
        }
        coder.prepare2Transfer(v);
        transmitter(coder.packet());
        noP++;
    }
}

// XrdMonSndDummyXrootd - main class

int main(int argc, char* argv[])
{
    XrdMonArgParser::ArgImpl<const char*, Convert2String>
        arg_host("-host", DEFAULT_HOST);
    XrdMonArgParser::ArgImpl<int, Convert2Int> 
        arg_port("-port", DEFAULT_PORT);

    try {
        XrdMonArgParser argParser;
        argParser.registerExpectedArg(&arg_host);
        argParser.registerExpectedArg(&arg_port);
        argParser.parseArguments(argc, argv);
    } catch (XrdMonException& e) {
        e.printIt();
        printHelp();
        return 1;
    }

    const char* inputPathFile = "./paths.txt";

    kXR_int32 seed = 12345;

    srand(seed);
    
    XrdMonSndDummyXrootd::NEWUSERFREQUENCY  =  200;
    XrdMonSndDummyXrootd::NEWPROCFREQUENCY  =   50;
    kXR_int16 NEWDICTENTRYFREQUENCY =  8000;
    kXR_int16 calls2NewXrdMonSndDictEntry  =     1;    
    
    XrdMonSndDebug::initialize();

    XrdMonSndDummyXrootd xrootd;
    assert ( !xrootd.initialize(inputPathFile) );
    
    XrdMonSndTraceCache cache;
    XrdMonSndCoder coder;
    XrdMonSndTransmitter transmitter;

    assert ( !transmitter.initialize(arg_host.myVal(), arg_port.myVal()) );
    kXR_int64 noP = 0;

    while ( 0 != access("start.txt", F_OK) ) {
        static bool warned = false;
        if ( ! warned ) {
            cout << "Waiting for start.txt file\n";
            warned = true;
        }
        sleep(1);
    }

    bool sendLight = true;
    
    if ( sendLight ) { // use this loop to test light decoder
        cout << "\n***** sending LIGHT data *****\n" << endl;
        for ( kXR_int64 i=0 ; i<NOCALLS ; i++ ) {
            calls2NewXrdMonSndDictEntry = NEWDICTENTRYFREQUENCY;
            doDictionaryXrdMonSndPacket(xrootd, coder, transmitter, noP);
            doStageXrdMonSndPacket(xrootd, coder, transmitter, noP);
            if ( i % 3 ) { // every 3 opens, close one file...
                closeFiles(xrootd, coder, transmitter, noP, true);
            }
            if ( noP >= maxNoXrdMonSndPackets-2 ) {
                break;
            }
            if ( 0 == access("stop.txt", F_OK) ) {
                break;
            }
            if ( i%1001 == 1000 ) {
                usleep(1);
            }
        }
    } else { //  use this loop to test full tracing
        cout << "\n***** sending BULK data *****\n" << endl;
        for ( kXR_int64 i=0 ; i<NOCALLS ; i++ ) {
            if ( ! --calls2NewXrdMonSndDictEntry ) {
                calls2NewXrdMonSndDictEntry = NEWDICTENTRYFREQUENCY;
                doDictionaryXrdMonSndPacket(xrootd, coder, transmitter, noP);
            } else {
                doTraceXrdMonSndPacket(xrootd, coder, transmitter, cache, noP);            
            }
            if ( noP >= maxNoXrdMonSndPackets-2 ) {
                break;
            }
            if ( 0 == access("stop.txt", F_OK) ) {
                break;
            }
            if ( i%1001 == 1000 ) {
                usleep(1);
            }
        }
    }

    if ( XrdMonSndDebug::verbose(XrdMonSndDebug::Sending) ) {
        cout << "Flushing cache" << endl;
    }
    if ( 0 == coder.prepare2Transfer(cache.getVector()) ) {
        cache.clear();
        transmitter(coder.packet());
        coder.reset();
        noP++;
    }

    closeFiles(xrootd, coder, transmitter, noP, false);

    // set shutdown signal
    //XrdMonSndAdminEntry ae;
    //ae.setShutdown();
    //coder.prepare2Transfer(ae);
    //transmitter(coder.packet());

    transmitter.shutdown();
    coder.printStats();
    
    return 0;
}
