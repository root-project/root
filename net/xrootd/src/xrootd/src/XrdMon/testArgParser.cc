/*****************************************************************************/
/*                                                                           */
/*                             testArgParser.cc                              */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonArgParserConvert.hh"
#include "XrdMon/XrdMonArgParser.hh"
#include "XrdMon/XrdMonException.hh"
#include "XrdSys/XrdSysHeaders.hh"

using std::cout;
using std::endl;
using namespace XrdMonArgParserConvert;


int main(int argc, char* argv[])
{
#define OA XrdMonArgParser::ArgImpl

    OA<const char*, Convert2String> arg_ctrLogDir  ("-ctrLogDir", "./logs/collector");
    OA<const char*, Convert2String> arg_decLogDir  ("-decLogDir", "./logs/decoder");
    OA<const char*, Convert2String> arg_rtLogDir   ("-rtLogDir", "./logs/rt");
    OA<int,         Convert2Int>    arg_decFlushDel("-decFlushDelay", 600); // [sec]
    OA<bool,        ConvertOnOff>   arg_rtOnOff    ("-rt", 600); // [sec]
#undef OA

    try {
        XrdMonArgParser argParser;
        argParser.registerExpectedArg(&arg_ctrLogDir);
        argParser.registerExpectedArg(&arg_decLogDir);
        argParser.registerExpectedArg(&arg_rtLogDir);
        argParser.registerExpectedArg(&arg_decFlushDel);
        argParser.registerExpectedArg(&arg_rtOnOff);
        argParser.parseArguments(argc, argv);
    } catch (XrdMonException& e) {
        e.printIt();
        cout << "Expected arguments:"
             << " [-ctrLogDir <path>]"
             << " [-decLogDir <path>]"
             << " [-rtLogDir <path>]"
             << " [-decFlushDelay <number>]"
             << " [-rt <on|off>]\n"
             << "where:\n"
             << "  ctrLogDir is a directory where collector's log file are stored\n"
             << "  decLogDir is a directory where decoder's log file are stored\n"
             << "  rtLogDir is a directory where real time log file are stored\n"
             << "  decFlushDelay is a value in sec specifying how often data is "
             << "flushed to collector's log files\n"
             << "  -rt turns on/off real time monitoring. If off, rtLogDir ignored\n"
             << endl;
        
        return 1;
    }
    
    cout << "ctrLogDir     is " << arg_ctrLogDir.myVal() << endl;
    cout << "decLogDir     is " << arg_decLogDir.myVal() << endl;
    cout << "rtLogDir      is " << arg_rtLogDir.myVal()  << endl;
    cout << "decFlushDelay is " << arg_decFlushDel.myVal() << endl;
    cout << "rt monitoring is " << ( arg_rtOnOff.myVal()? "on" : " off") << endl;

    return 0;
}
