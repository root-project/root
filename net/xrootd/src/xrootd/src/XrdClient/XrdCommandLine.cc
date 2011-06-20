//------------------------------------------------------------------------------
// XrdCommandLine
//
// Author: Fabrizio Furano (INFN Padova, 2005)
// Rewritten by Elvin Sindrilaru <elvin.alin.sindrilaru@cern.ch>
// with mods from Lukasz Janyst <ljanyst@cern.ch> (CERN, 2010)
//
// A command line tool for xrootd environments. The executable normally
// is named xrd.
//------------------------------------------------------------------------------

#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>
#include <sstream>
#include <string>
#include <signal.h>
#include <iomanip>

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>

// define NOMACROS prevents the insanity of some courses implementations from
// breaking this app
#define NOMACROS
#include <curses.h>
#undef NOMACROS

#include <term.h>
#endif

#define XRDCLI_VERSION "(C) 2004-2010 by the Xrootd group. Xrootd version: "XrdVSTRING

//------------------------------------------------------------------------------
// Some globals
//------------------------------------------------------------------------------
char           *opaqueinfo   = 0; // opaque info to be added to urls
kXR_unt16       xrd_wr_flags = kXR_async | kXR_mkpath | kXR_open_updt | kXR_new;
char           *initialhost  = 0;
XrdClient      *genclient    = 0;
XrdClientAdmin *genadmin     = 0;
XrdOucString    currentpath  = "/";
XrdOucString    cmdline_cmd;

//------------------------------------------------------------------------------
// Interrupt signal handler
//------------------------------------------------------------------------------
void CtrlCHandler(int sig)
{
    std::cerr << std::endl;
    std::cerr << "Please use 'exit' to terminate this program." << std::endl;
}

//------------------------------------------------------------------------------
// Commandline help message
//------------------------------------------------------------------------------
void PrintUsage()
{
    std::cerr << "usage: xrd [host]";
    std::cerr << "[-DSparmname stringvalue] ... [-DIparmname intvalue] ";
    std::cerr << "[-O<opaque info>] [command]" << std::endl;

    std::cerr << " -DSparmname stringvalue     :         ";
    std::cerr << "override the default value of an internal XrdClient setting ";
    std::cerr << "(of string type)" << std::endl;

    std::cerr << " -DIparmname intvalue        :         ";
    std::cerr << "override the default value of an internal  XrdClient setting ";
    std::cerr << "(of int type)" << std::endl;

    std::cerr << " -O     :         adds some opaque information to any used ";
    std::cerr << "xrootd url" << std::endl;

    std::cerr << " -h     :         this help screen" << std::endl;

    std::cerr <<std::endl << " where:" << std::endl;

    std::cerr << "   parmname     is the name of an internal parameter";
    std::cerr << std::endl;

    std::cerr << "   stringvalue  is a string to be assigned to an internal ";
    std::cerr << "parameter" << std::endl;

    std::cerr << "   intvalue     is an int to be assigned to an internal ";
    std::cerr << "parameter" << std::endl;

    std::cerr << "   command      is a command line to be executed. in this ";
    std::cerr << "case the host is mandatory." << std::endl;

    std::cerr << std::endl;
}

//------------------------------------------------------------------------------
// Commandline prompt
//------------------------------------------------------------------------------
void BuildPrompt( std::stringstream &s )
{
    s.clear();
    if (genadmin)
    {
        s << "root://" << genadmin->GetCurrentUrl().Host;
        s << ":" << genadmin->GetCurrentUrl().Port;
        s << "/" << currentpath;
    }
    s << "> ";
}

//------------------------------------------------------------------------------
// Out own primitive implementation of GNU readline
//------------------------------------------------------------------------------
#ifndef HAVE_READLINE
char *readline(const char *prompt)
{
    std::cout << prompt << std::flush;
    std::string input;
    std::getline( std::cin, input );

    if( !cin.good() )
        return 0;

    char *linebuf = (char *)malloc( input.size()+1 );
    strncpy( linebuf, input.c_str(), input.size()+1 );

    return linebuf;
}
#endif

//------------------------------------------------------------------------------
// Print help
//------------------------------------------------------------------------------
void PrintHelp()
{
    std::cout << std::endl << XRDCLI_VERSION << std::endl << std::endl;
    std::cout << "Usage: xrd [-O<opaque_info>] [-DS<var_name> stringvalue] ";
    std::cout << "[-DI<var_name> integervalue] [host[:port]] [batchcommand]";
    std::cout << std::endl << std::endl;

    std::cout << "List of available commands:" << std::endl << std::endl;
    std::cout << std::left;
    std::cout << std::setw(55) << "cat <filename> [xrdcp parameters]";
    std::cout << "Output a file on standard output using xrdcp. <filename> ";
    std::cout << "can be a root:// URL." << std::endl;

    std::cout << std::setw(55)  << "cd <dirname>";
    std::cout << "Change the current directory. Note: no existence check is ";
    std::cout << "performed." << std::endl;

    std::cout << std::setw(55)  << "chmod <fileordirname> <user> <group> <other>";
    std::cout << "Modify file permissions." << std::endl;

    std::cout << std::setw(55)  << "connect [hostname[:port]]";
    std::cout << "Connect to the specified host." << std::endl;

    std::cout << std::setw(55)  << "cp <fileordirname> <fileordirname> [xrdcp parameters]";
    std::cout << "Copies a file using xrdcp. <fileordirname> are always ";
    std::cout << "relative to the" << std::endl;
    std::cout << std::setw(55)  << " " << "current remote path. Also, they ";
    std::cout << "can be root:// URLs specifying any other host." << std::endl;

    std::cout << std::setw(55)  << "dirlist [dirname]";
    std::cout << "Get the requested directory listing." << std::endl;

    std::cout << std::setw(55)  << "dirlistrec [dirname]";
    std::cout << "Get the requested recursive directory listing.";
    std::cout << std::endl;

    std::cout << std::setw(55)  << "envputint <varname> <intval>";
    std::cout << "Put an integer in the internal environment." << std::endl;

    std::cout << std::setw(55)  << "envputstring <varname> <stringval>";
    std::cout << "Put a string in the internal environment." << std::endl;

    std::cout << std::setw(55)  << "exit";
    std::cout << "Exits from the program." << std::endl;

    std::cout << std::setw(55) << "help";
    std::cout << "This help screen." << std::endl;

    std::cout << std::setw(55) << "stat [fileordirname]";
    std::cout << "Get info about the given file or directory path.";
    std::cout << std::endl;

    std::cout << std::setw(55) << "statvfs [vfilesystempath]";
    std::cout << "Get info about a virtual file system." << std::endl;

    std::cout << std::setw(55) << "existfile <filename>";
    std::cout << "Test if the specified file exists." << std::endl;

    std::cout << std::setw(55) << "existdir <dirname>";
    std::cout << "Test if the specified directory exists." << std::endl;

    std::cout << std::setw(55) << "getchecksum <filename>";
    std::cout << "Get the checksum for the specified file." << std::endl;

    std::cout << std::setw(55) << "isfileonline <filename>";
    std::cout << "Test if the specified file is online." << std::endl;

    std::cout << std::setw(55) << "locatesingle <filename> <writable>";
    std::cout << "Give a location of the given file in the currently ";
    std::cout << "connected cluster." << std::endl;
    std::cout << std::setw(55) << " " << "if writable is true only a ";
    std::cout << "writable location is searched" << std::endl;
    std::cout << std::setw(55) << " " << "but, if no writable locations ";
    std::cout << "are found, the result is negative but may" << std::endl;
    std::cout << std::setw(55) << " " << "propose a non writable one.";
    std::cout << std::endl;

    std::cout << std::setw(55) << "locateall <filename>";
    std::cout << "Give all the locations of the given file in the currently";
    std::cout << "connected cluster." << std::endl;

    std::cout << std::setw(55) << "mv <filename1> <filename2>";
    std::cout << "Move filename1 to filename2 locally to the same server.";
    std::cout << std::endl;

    std::cout << std::setw(55) << "mkdir <dirname> [user] [group] [other]";
    std::cout << "Creates a directory." << std::endl;

    std::cout << std::setw(55) << "rm <filename>";
    std::cout << "Remove a file." << std::endl;

    std::cout << std::setw(55) << "rmdir <dirname>";
    std::cout << "Removes a directory." << std::endl;

    std::cout << std::setw(55) << "prepare <filename> <options> <priority>";
    std::cout << "Stage a file in." << std::endl;

    std::cout << std::setw(55)  << "query <reqcode> <parms>";
    std::cout << "Obtain server information." << std::endl;

    std::cout << std::setw(55)  << "queryspace <logicalname>";
    std::cout << "Obtain space information." << std::endl;

    std::cout << std::setw(55)  << "truncate <filename> <length>";
    std::cout << "Truncate a file." << std::endl;

    std::cout << std::endl << "For further information, please read the ";
    std::cout << "xrootd protocol documentation." << std::endl;

    std::cout << std::endl;
}

//------------------------------------------------------------------------------
// Check the answer of the server
//------------------------------------------------------------------------------
bool CheckAnswer(XrdClientAbs *gencli)
{
    if (!gencli->LastServerResp()) return false;

    switch (gencli->LastServerResp()->status)
    {
        case kXR_ok:
            return true;

        case kXR_error:
            std::cout << "Error " << gencli->LastServerError()->errnum;
            std::cout << ": " << gencli->LastServerError()->errmsg << std::endl;
            std::cout << std::endl;
            return false;

        default:
            std::cout << "Server response: " << gencli->LastServerResp()->status;
            std::cout << std::endl;
            return true;

    }
}

//------------------------------------------------------------------------------
// Print the output from locate
//------------------------------------------------------------------------------
void PrintLocateInfo(XrdClientLocate_Info &loc)
{
    std::cout << "InfoType: ";
    switch (loc.Infotype)
    {
        case XrdClientLocate_Info::kXrdcLocNone:
            std::cout << "none" << std::endl;
            break;
        case XrdClientLocate_Info::kXrdcLocDataServer:
            std::cout << "kXrdcLocDataServer" << std::endl;
            break;
        case XrdClientLocate_Info::kXrdcLocDataServerPending:
            std::cout << "kXrdcLocDataServerPending" << std::endl;
            break;
        case XrdClientLocate_Info::kXrdcLocManager:
            std::cout << "kXrdcLocManager" << std::endl;
            break;
        case XrdClientLocate_Info::kXrdcLocManagerPending:
            std::cout << "kXrdcLocManagerPending" << std::endl;
            break;
        default:
            std::cout << "Invalid Infotype" << std::endl;
    }
    std::cout << "CanWrite: ";
    if (loc.CanWrite) std::cout << "true" << std::endl;
    else std::cout << "false" << std::endl;
    std::cout << "Location: '" << loc.Location << "'" << std::endl;
}

//------------------------------------------------------------------------------
// process the "EXISTDIR" command
//------------------------------------------------------------------------------
void executeExistDir(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;
        }
        else pathname = currentpath;

        // Try to issue the request
        vecBool vb;
        vecString vs;
        vs.Push_back(pathname);
        genadmin->ExistDirs(vs, vb);

        // Check the answer
        if (vb[0] && (vb.GetSize() >= 1))
            std::cout << "The directory exists." << std::endl;
        else
            std::cout << "Directory not found." << std::endl;
        std::cout << std::endl;
        return;
    }
}

//------------------------------------------------------------------------------
// process the "CD" command
//------------------------------------------------------------------------------
void executeCd(XrdOucTokenizer &tkzer)
{
    char *parmname = tkzer.GetToken(0, 0);
    XrdOucString pathname;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        return;
    }

    if (!parmname || !strlen(parmname))
    {
        std::cout << "A directory name is needed." << std::endl << std::endl;
    }
    else
    {

        // Quite trivial directory processing
        if (!strcmp(parmname, ".."))
        {
            if (currentpath == "/") return;

            int pos = currentpath.rfind('/');

            if (pos != STR_NPOS)
                currentpath.erase(pos);

            if (!currentpath.length())
                currentpath = "/";

            return;
        }
        else if (!strcmp(parmname, "."))
        {
            return;
        }

        if (!currentpath.length() || (currentpath[currentpath.length()-1] != '/'))
            currentpath += "/";

        XrdOucString tmpPath;
        if (parmname[0] == '/')
            tmpPath = parmname;
        else
            tmpPath = currentpath + parmname;

        // Verify if tmpPath really exists
        vecBool vb;
        vecString vs;
        vs.Push_back(tmpPath);
        genadmin->ExistDirs(vs, vb);

        // Now check the answer
        if (CheckAnswer(genadmin))
            currentpath = tmpPath;
        else
            std::cout << "The directory does not exist." << std::endl << std::endl;
        return;
    }
}

//------------------------------------------------------------------------------
// process the "ENVPUTINT" command
//------------------------------------------------------------------------------
void executeEnvPutInt(XrdOucTokenizer &tkzer)
{
    char *parmname = tkzer.GetToken(0, 0),
          *val = tkzer.GetToken(0, 1);

    if (!parmname || !val)
    {
        std::cout << "Please provide command parameters:envputint <varname> ";
        std::cout << "<intvalue>" << std::endl << std::endl;
    }
    else
    {
        EnvPutInt(parmname, atoi(val));
        DebugSetLevel(EnvGetLong(NAME_DEBUG));
    }
    return;
}

//------------------------------------------------------------------------------
// process the "ENVPUTSTRING" command
//------------------------------------------------------------------------------
void executeEnvPutString(XrdOucTokenizer &tkzer)
{
    char *parmname = tkzer.GetToken(0, 0),
          *val = tkzer.GetToken(0, 1);

    if (!parmname || !val)
    {
        std::cout << "Please provide command parameters:envputstring <varname> ";
        std::cout << "<stringvalue>" << std::endl << std::endl;
    }
    else
        EnvPutString(parmname, val);
    return;
}

//------------------------------------------------------------------------------
// process the "HELP" command
//------------------------------------------------------------------------------
void executeHelp(XrdOucTokenizer &)
{
    PrintHelp();
}

//------------------------------------------------------------------------------
// process the "CONNECT" command
//------------------------------------------------------------------------------
void executeConnect(XrdOucTokenizer &tkzer)
{
    int retval = 0;
    char *host = initialhost;

    // If no host was given, then pretend one
    if (!host)
    {
        host = tkzer.GetToken(0, 1);
        if (!host || !strlen(host))
        {
            std::cout << "A hostname is needed to connect somewhere.";
            std::cout << std::endl << std::endl;
            retval = 1;
        }
    }

    if (!retval)
    {
        // Init the instance
        if (genadmin) delete genadmin;
        XrdOucString h(host);
        h  = "root://" + h;
        h += "//dummy";

        genadmin = new XrdClientAdmin(h.c_str());

        // Then connect
        if (!genadmin->Connect())
        {
            delete genadmin;
            genadmin = 0;
        }
    }
}

//------------------------------------------------------------------------------
// process the "DIRLISTREC" command
//------------------------------------------------------------------------------
void executeDirListRec(XrdOucTokenizer &tkzer)
{
    XrdClientVector<XrdOucString> pathq;
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *dirname = tkzer.GetToken(0, 0);
        XrdOucString path;

        if (dirname)
        {
            if (dirname[0] == '/')
                path = dirname;
            else
            {
                if ((currentpath.length() > 0) &&
                        (currentpath[currentpath.length()-1] != '/'))
                    path = currentpath + "/" + dirname;
                else
                    path = currentpath + dirname;

            }
        }
        else path = currentpath;

        if (!path.length())
        {
            std::cout << "The current path is an empty string. Assuming '/'.";
            std::cout << std::endl << std::endl;
            path = '/';
        }

        // Initialize the queue with this path
        pathq.Push_back(path);

        while (pathq.GetSize() > 0)
        {
            XrdOucString pathtodo = pathq.Pop_back();

            // Now try to issue the request
            XrdClientVector<XrdClientAdmin::DirListInfo> nfo;
            genadmin->DirList(pathtodo.c_str(), nfo, true);

            // Now check the answer
            if (!CheckAnswer(genadmin))
            {
                retval = 1;
                std::cout << "In server ";
                std::cout << genadmin->GetCurrentUrl().HostWPort;
                std::cout << " or in some of its child nodes." << std::endl;
                break;
            }

            for (int i = 0; i < nfo.GetSize(); i++)
            {

                if ((nfo[i].flags & kXR_isDir) &&
                        (nfo[i].flags & kXR_readable) &&
                        (nfo[i].flags & kXR_xset))
                {

                    // The path has not to be pushed if it's already present
                    // This may happen if several servers have the same path
                    bool foundpath = false;
                    for (int ii = 0; ii < pathq.GetSize(); ii++)
                    {
                        if (nfo[i].fullpath == pathq[ii])
                        {
                            foundpath = true;
                            break;
                        }
                    }

                    if (!foundpath)
                        pathq.Push_back(nfo[i].fullpath);
                    else
                        // If the path is already present in the queue
                        // then it was already printed as well.
                        continue;

                }

                char ts[256];
                strcpy(ts, "n/a");

                struct tm *t = gmtime(&nfo[i].modtime);
                strftime(ts, 255, "%F %T", t);

                char cflgs[16];
                memset(cflgs, 0, 16);

                if (nfo[i].flags & kXR_isDir)
                    strcat(cflgs, "d");
                else strcat(cflgs, "-");

                if (nfo[i].flags & kXR_readable)
                    strcat(cflgs, "r");
                else strcat(cflgs, "-");

                if (nfo[i].flags & kXR_writable)
                    strcat(cflgs, "w");
                else strcat(cflgs, "-");

                if (nfo[i].flags & kXR_xset)
                    strcat(cflgs, "x");
                else strcat(cflgs, "-");

                printf( "%s(%03ld) %12lld %s %s\n",
                        cflgs, nfo[i].flags, nfo[i].size,
                        ts, nfo[i].fullpath.c_str());
            }
        }
    }
    std::cout << std::endl;
    return;
}

//------------------------------------------------------------------------------
// process the "DIRLIST" command
//------------------------------------------------------------------------------
void executeDirList(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *dirname = tkzer.GetToken(0, 0);
        XrdOucString path;

        if (dirname)
        {
            if (dirname[0] == '/')
                path = dirname;
            else
            {
                if ((currentpath.length() > 0) &&
                        (currentpath[currentpath.length()-1] != '/'))
                    path = currentpath + "/" + dirname;
                else
                    path = currentpath + dirname;

            }
        }
        else path = currentpath;

        if (!path.length())
        {
            std::cout << "The current path is an empty string. Assuming '/'.";
            std::cout << std::endl << std::endl;
            path = '/';
        }

        // Now try to issue the request
        XrdClientVector<XrdClientAdmin::DirListInfo> nfo;
        genadmin->DirList(path.c_str(), nfo, true);

        // Now check the answer
        if (!CheckAnswer(genadmin))
        {
            retval = 1;
            std::cout << "In server " << genadmin->GetCurrentUrl().HostWPort;
            std::cout << " or in some of its child nodes." << std::endl;
            //nfo.Clear();
        }

        for (int i = 0; i < nfo.GetSize(); i++)
        {
            char ts[256];
            strcpy(ts, "n/a");

            struct tm *t = gmtime(&nfo[i].modtime);
            strftime(ts, 255, "%F %T", t);

            char cflgs[16];
            memset(cflgs, 0, 16);

            if (nfo[i].flags & kXR_isDir)
                strcat(cflgs, "d");
            else strcat(cflgs, "-");

            if (nfo[i].flags & kXR_readable)
                strcat(cflgs, "r");
            else strcat(cflgs, "-");

            if (nfo[i].flags & kXR_writable)
                strcat(cflgs, "w");
            else strcat(cflgs, "-");

            if (nfo[i].flags & kXR_xset)
                strcat(cflgs, "x");
            else strcat(cflgs, "-");

            printf( "%s(%03ld) %12lld %s %s\n",
                    cflgs, nfo[i].flags, nfo[i].size, ts,
                    nfo[i].fullpath.c_str());
        }
        std::cout << std::endl;
    }
    return;
}

//------------------------------------------------------------------------------
// process the "LOCATESINGLE" command
//------------------------------------------------------------------------------
void executeLocateSingle(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;
        }
        else pathname = currentpath;

        char *writable = tkzer.GetToken(0, 1);
        bool wrt = false;

        if (writable)
        {
            wrt = true;
            if (!strcmp(writable, "false") ||
                    !strcmp(writable, "0")) wrt = false;
            else
                std::cout << "Checking for a writable location." <<std::endl;
        }

        // Now try to issue the request
        XrdClientLocate_Info loc;
        bool r;
        r = genadmin->Locate((kXR_char *)pathname.c_str(), loc, wrt);
        if (!r)
            std::cout << "No matching files were found." <<std::endl;

        PrintLocateInfo(loc);
        std::cout << std::endl;
    }
    return;

}

//------------------------------------------------------------------------------
// process the "LOCATEALL" command
//------------------------------------------------------------------------------
void executeLocateAll(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;
        }
        else pathname = currentpath;

        // Now try to issue the request
        XrdClientVector<XrdClientLocate_Info> loc;
        bool r;
        r = genadmin->Locate((kXR_char *)pathname.c_str(), loc);
        if (!r)
            std::cout << "No matching files were found." <<std::endl;

        for (int ii = 0; ii < loc.GetSize(); ii++)
        {
            std::cout << std::endl << std::endl;
            std::cout << "------------- Location #" << ii+1 << std::endl;
            PrintLocateInfo(loc[ii]);
        }
        std::cout << std::endl;
    }
}

//------------------------------------------------------------------------------
// process the "STAT"  command
//------------------------------------------------------------------------------
void executeStat(XrdOucTokenizer &tkzer)
{

    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;
        }
        else pathname = currentpath;

        // Now try to issue the request
        long id, flags, modtime;
        long long size;
        genadmin->Stat(pathname.c_str(), id, size, flags, modtime);

        // Now check the answer
        if (!CheckAnswer(genadmin))
        {
            std::cout << "The command returned an error.";
            std::cout << std::endl << std::endl;
        }
        else
        {
            std::cout << "Id: " << id << " Size: " << size << " Flags: ";
            std::cout << flags << " Modtime: " << modtime << std::endl;
            std::cout <<std::endl;
        }
    }
}

//------------------------------------------------------------------------------
// process the "STATVFS" command
//------------------------------------------------------------------------------
void executeStatvfs(XrdOucTokenizer &tkzer)
{

    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;
        }
        else pathname = currentpath;

        // Now try to issue the request
        int rwservers = 0;
        long long rwfree = 0;
        int rwutil = 0;
        int stagingservers = 0;
        long long stagingfree = 0;
        int stagingutil = 0;


        genadmin->Stat_vfs(pathname.c_str(), rwservers, rwfree, rwutil,
                           stagingservers, stagingfree, stagingutil);

        // Now check the answer
        if (!CheckAnswer(genadmin))
        {
            std::cout << "The command returned an error.";
            std::cout << std::endl <<std::endl;
        }
        else
        {
            std::cout << "r/w nodes: " << rwservers;
            std::cout << " r/w free space: " << rwfree;
            std::cout << " r/w utilization: " << rwutil;
            std::cout << " staging nodes: " << rwservers;
            std::cout << " staging free space: " << rwfree;
            std::cout << " staging utilization: " << rwutil;
            std::cout << std::endl << std::endl;
        }
    }
}

//------------------------------------------------------------------------------
// process the "EXISTFILE" command
//------------------------------------------------------------------------------
void executeExistFile(XrdOucTokenizer &tkzer)
{
    int retval = 0;
    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;
        }
        else pathname = currentpath;

        // Now try to issue the request
        vecBool vb;
        vecString vs;
        vs.Push_back(pathname);
        genadmin->ExistFiles(vs, vb);

        // Now check the answer
        if (!CheckAnswer(genadmin))
            retval = 1;

        if (vb[0] && (vb.GetSize() >= 1))
            std::cout << "The file exists." << std::endl;
        else
            std::cout << "File not found." << std::endl;
        std::cout <<std::endl;
    }
}

//------------------------------------------------------------------------------
// process the "GETCHECKSUM" command
//------------------------------------------------------------------------------
void executeGetCheckSum(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;

            // Now try to issue the request
            kXR_char *ans;

            genadmin->GetChecksum((kXR_char *)pathname.c_str(), &ans);

            // Now check the answer
            if (!CheckAnswer(genadmin))
            {
                std::cout << "The command returned an error.";
                std::cout << std::endl << std::endl;
            }
            else
            {
                std::cout << "Checksum: " << ans << std::endl;
                free(ans);
                std::cout << std::endl;
            }
        }
        else
        {
            std::cout << "Please provide command parameter: getchecksum ";
            std::cout << "<filename>" << std::endl << std::endl;
        }
    }
}

//------------------------------------------------------------------------------
// process the "ISFILEONLINE" command
//------------------------------------------------------------------------------
void executeIsFileOnline(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;

            // Now try to issue the request
            vecBool vb;
            vecString vs;
            vs.Push_back(pathname);
            genadmin->IsFileOnline(vs, vb);

            // Now check the answer
            if (!CheckAnswer(genadmin))
            {
                std::cout << "The command returned an error.";
                std::cout << std::endl << std::endl;
            }
            else
            {
                if (vb[0] && (vb.GetSize() >= 1))
                    std::cout << "The file is online." << std::endl;
                else
                    std::cout << "The file is not online." << std::endl;
                std::cout <<std::endl;
            }
        }
        else
        {
            std::cout << "Please provide command parameter: isfileoneline ";
            std::cout << "<filename>" << std::endl << std::endl;
        }
    }
}

//------------------------------------------------------------------------------
// process the "MV" command
//------------------------------------------------------------------------------
void executeMv(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname1 = tkzer.GetToken(0, 0);
        char *fname2 = tkzer.GetToken(0, 0);
        XrdOucString pathname1, pathname2;

        if (!fname1 || !fname2)
        {
            std::cout << "Please provide command parameteres: mv <source> ";
            std::cout << "<destination>" <<std::endl <<std::endl;
        }
        else
        {

            if (fname1[0] == '/')
                pathname1 = fname1;
            else
                pathname1 = currentpath + "/" + fname1;

            if (fname2[0] == '/')
                pathname2 = fname2;
            else
                pathname2 = currentpath + "/" + fname2;

            // Now try to issue the request
            genadmin->Mv(pathname1.c_str(), pathname2.c_str());

            // Now check the answer
            if (!CheckAnswer(genadmin))
                std::cout << "The command returned an error." <<std::endl <<std::endl;
        }
    }
    return;
}

//------------------------------------------------------------------------------
// process the "MKDIR" command
//------------------------------------------------------------------------------
void executeMkDir(XrdOucTokenizer &tkzer)
{

    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname1 = tkzer.GetToken(0, 0);
        char *userc = tkzer.GetToken(0, 0);
        char *groupc = tkzer.GetToken(0, 0);
        char *otherc = tkzer.GetToken(0, 0);

        if (!fname1)
        {
            std::cout << "Please provide command parameters: mkdir <filename> ";
            std::cout << "[<user> <group> <other>]" << std::endl << std::endl;
        }
        else
        {
            int user = 0, group = 0, other = 0;
            if (userc) user = atoi(userc);
            if (groupc) group = atoi(groupc);
            if (otherc) other = atoi(otherc);

            XrdOucString pathname1;

            if (fname1[0] == '/')
                pathname1 = fname1;
            else
                pathname1 = currentpath + "/" + fname1;

            // Now try to issue the request
            genadmin->Mkdir(pathname1.c_str(), user, group, other);

            // Now check the answer
            if (!CheckAnswer(genadmin))
            {
                std::cout << "The command returned an error.";
                std::cout << std::endl << std::endl;
            }
        }
    }
}

//------------------------------------------------------------------------------
// process the "CHMOD" command
//------------------------------------------------------------------------------
void executeChmod(XrdOucTokenizer &tkzer)
{

    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl <<std::endl;
        retval = 1;
    }

    if (!retval)
    {

        char *fname1 = tkzer.GetToken(0, 0);
        char *userc = tkzer.GetToken(0, 0);
        char *groupc = tkzer.GetToken(0, 0);
        char *otherc = tkzer.GetToken(0, 0);

        if (!fname1 || !userc || !groupc || !otherc)
        {
            std::cout << "Please provide command parameters: chmod <filename> ";
            std::cout << "<user> <group> <other>" << std::endl << std::endl;
        }
        else
        {
            int user = 0, group = 0, other = 0;
            if (userc) user = atoi(userc);
            if (groupc) group = atoi(groupc);
            if (otherc) other = atoi(otherc);

            XrdOucString pathname1;

            if (fname1[0] == '/')
                pathname1 = fname1;
            else
                pathname1 = currentpath + "/" + fname1;

            // Now try to issue the request
            genadmin->Chmod(pathname1.c_str(), user, group, other);

            // Now check the answer
            if (!CheckAnswer(genadmin))
                std::cout << "The command returned an error." <<std::endl <<std::endl;
        }
    }
}

//------------------------------------------------------------------------------
// process the "truncate" command
//------------------------------------------------------------------------------
void executeTruncate(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl <<std::endl;
        retval = 1;
    }

    if (!retval)
    {

        char *fname = tkzer.GetToken(0, 0);
        char *slen = tkzer.GetToken(0, 0);

        long long len = 0;
        XrdOucString pathname1;

        if (fname)
        {
            if (fname[0] == '/')
                pathname1 = fname;
            else
                pathname1 = currentpath + "/" + fname;

            if (slen) len = atoll(slen);
            else
            {
                std::cout << "Missing parameter length." <<std::endl;
                return;
            }

            if (len <= 0)
            {
                std::cout << "Bad length." <<std::endl;
                return;
            }

            // Now try to issue the request
            genadmin->Truncate(pathname1.c_str(), len);

            // Now check the answer
            if (!CheckAnswer(genadmin))
                std::cout << "The command returned an error." <<std::endl <<std::endl;
        }
        else
        {
            std::cout << "Please provide command parameters: truncate <filename> ";
            std::cout << "<length>" << std::endl << std::endl;
            return;
        }
    }
}

//------------------------------------------------------------------------------
// process the "RM" command
//------------------------------------------------------------------------------
void executeRm(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl <<std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;

            // Now try to issue the request
            genadmin->Rm(pathname.c_str());

            // Now check the answer
            if (!CheckAnswer(genadmin))
            {
                std::cout << "The command returned an error.";
                std::cout << std::endl << std::endl;
            }
        }
        else
        {
            std::cout << "Please provide command parameter: rm <filename>";
            std::cout << std::endl << std::endl;
        }
    }
}

//------------------------------------------------------------------------------
// process the "RMDIR" command
//------------------------------------------------------------------------------
void executeRmDir(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl <<std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname = tkzer.GetToken(0, 0);
        XrdOucString pathname;

        if (fname)
        {
            if (fname[0] == '/')
                pathname = fname;
            else
                pathname = currentpath + "/" + fname;
            // Now try to issue the request
            genadmin->Rmdir(pathname.c_str());

            // Now check the answer
            if (!CheckAnswer(genadmin))
            {
                std::cout << "The command returned an error.";
                std::cout << std::endl << std::endl;
            }
        }
        else
        {
            std::cout << "Please provide command parameter: rmdir <dirname>";
            std::cout << std::endl << std::endl;
        }
    }
}

//------------------------------------------------------------------------------
// process the "PREPARE" command
//------------------------------------------------------------------------------
void executePrepare(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname1 = tkzer.GetToken(0, 0);
        char *optsc = tkzer.GetToken(0, 0);
        char *prioc = tkzer.GetToken(0, 0);

        if (!fname1)
        {
            std::cout << "Please provide command parameters: prepare ";
            std::cout << "<filename> <options> <priority>";
            std::cout << std::endl << std::endl;
        }
        else
        {
            int opts = 0, prio = 0;
            if (optsc) opts = atoi(optsc);
            if (prioc) prio = atoi(prioc);

            XrdOucString pathname1;

            if (fname1[0] == '/')
                pathname1 = fname1;
            else
                pathname1 = currentpath + "/" + fname1;

            // Now try to issue the request
            vecString vs;
            vs.Push_back(pathname1);
            genadmin->Prepare(vs, (kXR_char)opts, (kXR_char)prio);

            // Now check the answer
            if (!CheckAnswer(genadmin))
            {
                std::cout << "The command returned an error.";
                std::cout << std::endl << std::endl;
            }
        }
    }
}

//------------------------------------------------------------------------------
// process the "CAT" command
//------------------------------------------------------------------------------
void executeCat(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." << std::endl << std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname1 = tkzer.GetToken(0, 0);
        char *tk;
        XrdOucString pars;

        while ((tk = tkzer.GetToken(0, 0)))
        {
            pars += " ";
            pars += tk;
        }

        XrdOucString pathname1;

        if (fname1)
        {
            if ( (strstr(fname1, "root://") == fname1) ||
                    (strstr(fname1, "xroot://") == fname1) )
                pathname1 = fname1;
            else if (fname1[0] == '/')
            {
                pathname1 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                pathname1 += "/";
                pathname1 += fname1;
            }
            else
            {
                pathname1 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                pathname1 += "/";
                pathname1 += currentpath;
                pathname1 += "/";
                pathname1 += fname1;
            }
        }
        else
            std::cout << "Missing parameter." <<std::endl;

        XrdOucString cmd;
        cmd = "xrdcp -s ";
        cmd += pathname1;
        cmd += pars;
        cmd += " -";

        int rt = system(cmd.c_str());

        std::cout << "cat returned " << rt <<std::endl <<std::endl;
    }
}

//------------------------------------------------------------------------------
// process the "CP" command
//------------------------------------------------------------------------------
void executeCp(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl <<std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *fname1 = tkzer.GetToken(0, 0);
        char *fname2 = tkzer.GetToken(0, 0);
        char *tk;
        XrdOucString pars;

        while ((tk = tkzer.GetToken(0, 0)))
        {
            pars += " ";
            pars += tk;
        }

        XrdOucString pathname1, pathname2;

        if (fname1)
        {
            if ( (strstr(fname1, "root://") == fname1) ||
                    (strstr(fname1, "xroot://") == fname1) )
                pathname1 = fname1;
            else if (fname1[0] == '/')
            {
                pathname1 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                pathname1 += "/";
                pathname1 += fname1;
            }
            else
            {
                pathname1 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                pathname1 += "/";
                pathname1 += currentpath;
                pathname1 += "/";
                pathname1 += fname1;
            }

            if (fname2)
            {

                if ( (strstr(fname2, "root://") == fname2) ||
                        (strstr(fname2, "xroot://") == fname2) )
                    pathname2 = fname2;
                else if (fname2[0] == '/')
                {
                    pathname2 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                    pathname2 += "/";
                    pathname2 += fname2;
                }
                else
                {
                    pathname2 = "root://" + genadmin->GetCurrentUrl().HostWPort;
                    pathname2 += "/";
                    pathname2 += currentpath;
                    pathname2 += "/";
                    pathname2 += fname2;
                }

            }
            else
            {
                std::cout << "Please provide command parameters: cp <source>";
                std::cout << "<destination> [<params>]" <<std::endl <<std::endl;
            }
        }
        else
        {
            std::cout << "Please provide command parameters: cp <source> ";
            std::cout << "<destination> [<params>]" <<std::endl <<std::endl;
        }

        XrdOucString cmd;
        cmd = "xrdcp ";
        cmd += pathname1;
        cmd += " ";
        cmd += pathname2 + pars;

        int rt = system(cmd.c_str());

        std::cout << "cp returned " << rt <<std::endl <<std::endl;;
    }
}

//------------------------------------------------------------------------------
// process the "QUERY" command
//------------------------------------------------------------------------------
void executeQuery(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl <<std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *reqcode = tkzer.GetToken(0, 0);

        if (!reqcode)
        {
            std::cout << "Please provide command parameters: query ";
            std::cout << "<reqcode> <params>" << std::endl << std::endl;
        }
        else
        {

            const kXR_char *args = (const kXR_char *)tkzer.GetToken(0, 0);
            kXR_char Resp[1024];

            genadmin->Query(atoi(reqcode), args, Resp, 1024);

            // Now check the answer
            if (!CheckAnswer(genadmin))
                retval = 1;

            std::cout << Resp << std::endl << std::endl;
        }
    }
}

//------------------------------------------------------------------------------
// process the "QUERYSPACE" command
//------------------------------------------------------------------------------
void executeQuerySpace(XrdOucTokenizer &tkzer)
{
    int retval = 0;

    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl <<std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *ns = tkzer.GetToken(0, 0);

        if (!ns)
        {
            std::cout << "Please provide command parameters: query ";
            std::cout << "<logicalname>" << std::endl << std::endl;
        }
        else
        {

            long long totspace;
            long long totfree;
            long long totused;
            long long largestchunk;

            genadmin->GetSpaceInfo(ns, totspace, totfree, totused, largestchunk);

            // Now check the answer
            if (!CheckAnswer(genadmin))
                retval = 1;

            std::cout << "Disk space approximations (MB):" << std::endl;
            std::cout << "Total         : " << totspace/(1024*1024) << std::endl;
            std::cout << "Free          : " << totfree/(1024*1024) << std::endl;
            std::cout << "Used          : " << totused/(1024*1024) << std::endl;
            std::cout << "Largest chunk : " << largestchunk/(1024*1024);
            std::cout << std::endl << std::endl;
        }
    }
}

//------------------------------------------------------------------------------
// process the "DEGBUG" command
//------------------------------------------------------------------------------
void executeDebug(XrdOucTokenizer &tkzer)
{

    int level = -2, retval = 0;
    string delim = "=";
    if (!genadmin)
    {
        std::cout << "Not connected to any server." <<std::endl <<std::endl;
        retval = 1;
    }

    if (!retval)
    {
        char *str = tkzer.GetToken(0, 0);

        if(str)
        {
            char *tok;
            tok = strtok(str,"=");

            if (!strcmp(tok, "-set-level"))
            {
                tok = strtok(NULL,"=");
                if (tok)
                    if (strspn(tok, "-0123456789") == strlen(tok))
                    {
                        level = atoi(tok);
                        std::cout << "The value of the debug level is: ";
                        std::cout << level <<std::endl <<std::endl;
                        EnvPutInt(NAME_DEBUG, level);
                        DebugSetLevel(EnvGetLong(NAME_DEBUG));
                        return;
                    }
            }
        }
    }
    std::cout << "Please provide correct parameters, eg. debug ";
    std::cout << "-set-level=<debug_level>" << std::endl << std::endl;
}

//------------------------------------------------------------------------------
// Function lookup
//------------------------------------------------------------------------------
typedef void (*CommandCallback)(XrdOucTokenizer &);

struct LookupItem
{
    const char      *name;
    CommandCallback  callback;
};

LookupItem lookupTable[] =
{
    {"cd",           executeCd          },
    {"envputint",    executeEnvPutInt   },
    {"envputstring", executeEnvPutString},
    {"help",         executeHelp        },
    {"connect",      executeConnect     },
    {"dirlistrec",   executeDirListRec  },
    {"dirlist",      executeDirList     },
    {"ls",           executeDirList     },
    {"locatesingle", executeLocateSingle},
    {"locateall",    executeLocateAll   },
    {"stat",         executeStat        },
    {"statvfs",      executeStatvfs     },
    {"existfile",    executeExistFile   },
    {"existdir",     executeExistDir    },
    {"getchecksum",  executeGetCheckSum },
    {"isfileonline", executeIsFileOnline},
    {"mv",           executeMv          },
    {"mkdir",        executeMkDir       },
    {"chmod",        executeChmod       },
    {"truncate",     executeTruncate    },
    {"rm",           executeRm          },
    {"rmdir",        executeRmDir       },
    {"prepare",      executePrepare     },
    {"cat",          executeCat         },
    {"cp",           executeCp          },
    {"query",        executeQuery       },
    {"queryspace",   executeQuerySpace  },
    {"debug",        executeDebug       },
    {0,              0                  }
};

CommandCallback lookup( char *command )
{
    LookupItem *it = lookupTable;
    while( it->name != 0 )
    {
        if( strcmp( command, it->name ) == 0 )
            return it->callback;
        ++it;
    }
    return 0;
}

//------------------------------------------------------------------------------
// Main program
//------------------------------------------------------------------------------
int main(int argc, char**argv)
{

    int retval = 0;

    DebugSetLevel(0);

    // We want this tool to be able to connect everywhere
    // Note that the side effect of these calls here is to initialize the
    // XrdClient environment.
    // This is crucial if we want to later override its default values
    EnvPutString( NAME_REDIRDOMAINALLOW_RE, "*" );
    EnvPutString( NAME_CONNECTDOMAINALLOW_RE, "*" );
    EnvPutString( NAME_REDIRDOMAINDENY_RE, "" );
    EnvPutString( NAME_CONNECTDOMAINDENY_RE, "" );

    EnvPutInt( NAME_DEBUG, -1);

    //--------------------------------------------------------------------------
    // Parse the commandline
    //--------------------------------------------------------------------------
    for (int i=1; i < argc; i++)
    {
        if ( (strstr(argv[i], "-O") == argv[i]))
        {
            opaqueinfo=argv[i]+2;
            continue;
        }

        if ( (strstr(argv[i], "-h") == argv[i]) ||
                (strstr(argv[i], "--help") == argv[i]) )
        {
            PrintUsage();
            exit(0);
            continue;
        }

        if ( (strstr(argv[i], "-DS") == argv[i]) &&
                (argc >= i+2) )
        {
            std::cerr << "Overriding " << argv[i]+3 << " with value ";
            std::cerr << argv[i+1] << ". ";
            EnvPutString( argv[i]+3, argv[i+1] );
            std::cerr << " Final value: " << EnvGetString(argv[i]+3);
            std::cerr << std::endl;
            i++;
            continue;
        }

        if ( (strstr(argv[i], "-DI") == argv[i]) &&
                (argc >= i+2) )
        {
            std::cerr << "Overriding '" << argv[i]+3 << "' with value ";
            std::cerr << argv[i+1] << ". ";
            EnvPutInt( argv[i]+3, atoi(argv[i+1]) );
            std::cerr << " Final value: " << EnvGetLong(argv[i]+3);
            std::cerr << std::endl;
            i++;
            continue;
        }

        // Any other par is ignored
        if ( (strstr(argv[i], "-") == argv[i]) && (strlen(argv[i]) > 1) )
        {
            std::cerr << "Unknown parameter " << argv[i] << std::endl;
            continue;
        }

        if (!initialhost) initialhost = argv[i];
        else
        {
            cmdline_cmd += argv[i];
            cmdline_cmd += " ";
        }
    }


    //--------------------------------------------------------------------------
    // Initialize the client
    //--------------------------------------------------------------------------
    DebugSetLevel(EnvGetLong(NAME_DEBUG));

    // if there's no command to execute from the cmdline...
    if (cmdline_cmd.length() == 0)
    {
        std::cout << XRDCLI_VERSION << std::endl;
        std::cout << "Welcome to the xrootd command line interface.";
        std::cout << std::endl;
        std::cout << "Type 'help' for a list of available commands.";
        std::cout << std::endl;
    }

    if (initialhost)
    {
        XrdOucString s = "root://";
        s += initialhost;
        s += "//dummy";
        genadmin = new XrdClientAdmin(s.c_str());

        // Then connect
        if (!genadmin->Connect())
        {
            delete genadmin;
            genadmin = 0;
        }
    }

    //--------------------------------------------------------------------------
    // Get the command from the std input
    //--------------------------------------------------------------------------
    while( true )
    {
        //----------------------------------------------------------------------
        // Parse the string
        //----------------------------------------------------------------------
        stringstream prompt;
        char *linebuf=0;

        if (cmdline_cmd.length() == 0)
        {
            BuildPrompt(prompt);
            linebuf = readline(prompt.str().c_str());
            if( !linebuf )
            {
                std::cout << "Goodbye." << std::endl << std::endl;
                break;
            }
            if( ! *linebuf)
            {
                free(linebuf);
                continue;
            }
#ifdef HAVE_READLINE
            add_history(linebuf);
#endif
        }
        else linebuf = strdup(cmdline_cmd.c_str());

        XrdOucTokenizer tkzer(linebuf);
        if (!tkzer.GetLine()) continue;

        char *cmd = tkzer.GetToken(0, 1);

        if (!cmd) continue;

        //----------------------------------------------------------------------
        // Execute the command
        //----------------------------------------------------------------------
        CommandCallback callback = lookup( cmd );
        if( !callback )
        {
            if( strcmp( cmd, "exit" ) == 0 )
            {
                std::cout << "Goodbye." << std::endl << std::endl;
                free( linebuf );
                break;
            }
            std::cout << "Command not recognized." << std::endl;
            std::cout << "Type \"help\" for a list of commands.";
            std::cout <<std::endl <<std::endl;
        }
        else
            (*callback)( tkzer );


        free( linebuf );

        // if it was a cmd from the commandline...
        if (cmdline_cmd.length() > 0) break;

    }

    if (genadmin)
        delete genadmin;

    return retval;
}
