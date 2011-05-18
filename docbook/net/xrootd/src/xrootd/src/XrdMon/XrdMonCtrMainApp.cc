/*****************************************************************************/
/*                                                                           */
/*                           XrdMonCtrMainApp.cc                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonArgParser.hh"
#include "XrdMon/XrdMonArgParserConvert.hh"
#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonTypes.hh"
#include "XrdMon/XrdMonUtils.hh"
#include "XrdMon/XrdMonCtrArchiver.hh"
#include "XrdMon/XrdMonCtrDebug.hh"
#include "XrdMon/XrdMonCtrCollector.hh"
#include "XProtocol/XPtypes.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <iomanip>
using std::cerr;
using std::cout;
using std::endl;
using std::setfill;
using std::setw;
using namespace XrdMonArgParserConvert;

const bool      defaultOnlineDecOn   = true;    // online decoding on
const bool      defaultRTOn          = true;    // real time decoding on
const char*     defaultCtrLogDir     = "./logs/collector";
const char*     defaultDecLogDir     = "./logs/decoder";
const char*     defaultRTLogDir      = "./logs/rt";
const int       defaultDecHDFlushDelay = 600;           // [sec]
const int       defaultDecRTFlushDelay = 5;             // [sec]
const kXR_int64 defaultMaxCtrLogSize = 1024*1024*1024;  // 1GB
const kXR_int32 defaultCtrBufSize    = 64*1024;         // 64 KB
const int       defaultRTBufSize     = 128*1024;        // 128 KB

void
printHelp()
{
    cout << "\nxrdmonCollector\n"
         << "    [-onlineDec <on|off>]\n"
         << "    [-rt <on|off>]\n"
         << "    [-ctrLogDir <path>]\n"
         << "    [-decLogDir <path>]\n"
         << "    [-rtLogDir <path>]\n"
         << "    [-decHDFlushDelay <value>]\n"
         << "    [-maxCtrLogSize <value>]\n"
         << "    [-ctrBufSize <value>]\n"
         << "    [-rtBufSize <value>]\n"
         << "    [-port <portNr>]\n"
         << "    [-ver]\n"
         << "    [-help]\n"
         << "\n"
         << "-onlineDec <on|off>      Turns on/off online decoding.\n"
         << "                         Default value is \"" << (defaultOnlineDecOn?"on":"off") << "\".\n"
         << "-rt <on|off>             Turns on/off real time monitoring. Online decoding has to be\n"
         << "                         Default value is \"" << (defaultRTOn?"on":"off") << "\".\n"
         << "-ctrLogDir <path>        Directory where collector's log file are stored.\n"
         << "                         Default value is \"" << defaultCtrLogDir << "\".\n"
         << "-decLogDir <path>        Directory where decoder's log file are stored.\n"
         << "                         Default value is \"" << defaultDecLogDir << "\".\n"
         << "-rtLogDir <path>         Directory where real time log file are stored.\n"
         << "                         Default value is \"" << defaultRTLogDir << "\".\n"
         << "-decHDFlushDelay <delay> Value in sec specifying how often history data is\n"
         << "                         flushed to collector's log files. History data means\n"
         << "                         the data corresponding to *closed* sessions and files.\n" 
         << "                         Default value is \"" << defaultDecHDFlushDelay << "\".\n"
         << "-decRTFlushDelay <delay> Value in sec specifying how often real time data is\n"
         << "                         flushed to log file.\n"
         << "                         Default value is \"" << defaultDecRTFlushDelay << "\".\n"
         << "-maxCtrLogSize <size>    Max size of collector's log file.\n"
         << "                         Default value is \"" << defaultMaxCtrLogSize << "\".\n"
         << "-ctrBufSize <size>       Size of transient buffer of collected packets. It has to be\n"
         << "                         larger than or equal to max page size (64K).\n"
         << "                         Default value is \"" << defaultCtrBufSize << "\".\n"
         << "-rtBufSize <size>        Size of transient buffer of collected real time data.\n"
         << "                         Default value is \"" << defaultRTBufSize << "\".\n"
         << "-port <portNr>           Port number to be used.\n"
         << "                         Default valus is \"" << DEFAULT_PORT << "\".\n"
         << "-ver                     Reports version and exits. Don't specify any other\n"
         << "                         option with this one.\n"
         << endl;
}

int main(int argc, char* argv[])
{
    XrdMonCtrDebug::initialize();

    if ( argc == 2 && !strcmp(argv[1], "-ver") ) {
        cout << "v\t" << setw(3) << setfill('0') << XRDMON_VERSION << endl;
        return 0;
    }
         
    XrdMonArgParser::ArgImpl<bool, ConvertOnOff>
         arg_onlineDecOn("-onlineDec", defaultOnlineDecOn);
    XrdMonArgParser::ArgImpl<bool, ConvertOnOff>
         arg_rtOn       ("-rt", defaultRTOn);
    XrdMonArgParser::ArgImpl<const char*, Convert2String> 
         arg_ctrLogDir  ("-ctrLogDir", defaultCtrLogDir);
    XrdMonArgParser::ArgImpl<const char*, Convert2String> 
         arg_decLogDir  ("-decLogDir", defaultDecLogDir);
    XrdMonArgParser::ArgImpl<const char*, Convert2String>
         arg_rtLogDir   ("-rtLogDir", defaultRTLogDir);
    XrdMonArgParser::ArgImpl<int, Convert2Int>
         arg_decRTFlushDel("-decRTFlushDelay", defaultDecRTFlushDelay);
    XrdMonArgParser::ArgImpl<int, Convert2Int>
         arg_decHDFlushDel("-decHDFlushDelay", defaultDecHDFlushDelay);
    XrdMonArgParser::ArgImpl<kXR_int64, Convert2LL> 
        arg_maxFSize   ("-maxCtrLogSize", defaultMaxCtrLogSize);
    XrdMonArgParser::ArgImpl<int, Convert2Int> 
        arg_ctrBufSize("-ctrBufSize", defaultCtrBufSize);
    XrdMonArgParser::ArgImpl<int, Convert2Int> 
        arg_port("-port", DEFAULT_PORT);
    XrdMonArgParser::ArgImpl<int, Convert2Int> 
        arg_rtBufSize   ("-maxCtrLogSize", defaultRTBufSize);

    try {
        XrdMonArgParser argParser;
        argParser.registerExpectedArg(&arg_onlineDecOn);
        argParser.registerExpectedArg(&arg_rtOn);
        argParser.registerExpectedArg(&arg_ctrLogDir);
        argParser.registerExpectedArg(&arg_decLogDir);
        argParser.registerExpectedArg(&arg_rtLogDir);
        argParser.registerExpectedArg(&arg_decRTFlushDel);
        argParser.registerExpectedArg(&arg_decHDFlushDel);
        argParser.registerExpectedArg(&arg_maxFSize);
        argParser.registerExpectedArg(&arg_ctrBufSize);
        argParser.registerExpectedArg(&arg_port);
        argParser.registerExpectedArg(&arg_rtBufSize);
        argParser.parseArguments(argc, argv);
    } catch (XrdMonException& e) {
        e.printIt();
        printHelp();
        return 1;
    }
    if ( arg_onlineDecOn.myVal() == false && arg_rtOn.myVal() ) {
        cerr << "\nError: you can not turn on rt monitoring if"
             << " online decoding is off." << endl;
        printHelp();
        return 1;
    }
    if ( arg_ctrBufSize.myVal() < 64*1024 ) {
        cerr << "\nError: collector's buffer size too small" << endl;
        printHelp();
        return 2;
    }
    if ( arg_port.myVal() < 1 ) {
        cerr << "\nError: invalid port number" << endl;
        printHelp();
        return 3;
    }
    
    cout << "online decoding  is " << (arg_onlineDecOn.myVal()?"on":"off") <<'\n'
         << "rt monitoring    is " << (arg_rtOn.myVal()?"on":"off") << '\n'
         << "ctrLogDir        is " << arg_ctrLogDir.myVal() << '\n'
         << "decLogDir        is " << arg_decLogDir.myVal() << '\n'
         << "rtLogDir         is " << arg_rtLogDir.myVal()  << '\n'
         << "decRTFlushDelay  is " << arg_decRTFlushDel.myVal() << '\n'
         << "decHDFlushDelay  is " << arg_decHDFlushDel.myVal() << '\n'
         << "maxCtrLogSize    is " << arg_maxFSize.myVal() << '\n'
         << "ctrBufSize       is " << arg_ctrBufSize.myVal() << '\n'
         << "rtBufSize        is " << arg_rtBufSize.myVal() << '\n'
         << "port             is " << arg_port.myVal()
         << endl;

    try {
        mkdirIfNecessary(arg_ctrLogDir.myVal());
        XrdMonCtrArchiver::_decHDFlushDelay= arg_decHDFlushDel.myVal();
        if ( arg_onlineDecOn.myVal() ) {
            mkdirIfNecessary(arg_decLogDir.myVal());
            if ( arg_rtOn.myVal() ) {
                mkdirIfNecessary(arg_rtLogDir.myVal());
                XrdMonCtrArchiver::_decRTFlushDelay= arg_decRTFlushDel.myVal();
            }
        }
    } catch (XrdMonException& e) {
        e.printIt();
        return 2;
    }
    XrdMonCtrCollector::port = arg_port.myVal();
    
    // start thread for receiving data
    pthread_t recThread;
    if ( 0 != pthread_create(&recThread, 
                             0, 
                             receivePackets,
                             0) ) {
        cerr << "Failed to create a collector thread" << endl;
        return 1;
    }

    try {
        // store received packets until admin packet with sigterm arrives
        XrdMonCtrArchiver archiver(arg_ctrLogDir.myVal(), 
                                   arg_decLogDir.myVal(),
                                   arg_rtLogDir.myVal(),
                                   arg_maxFSize.myVal(),
                                   arg_ctrBufSize.myVal(),
                                   arg_rtBufSize.myVal(),
                                   arg_onlineDecOn.myVal(), 
                                   arg_rtOn.myVal());
        archiver();
    } catch (XrdMonException& e) {
        e.printIt();
        return 2;
    }
    
    return 0;
}


