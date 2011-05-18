/*****************************************************************************/
/*                                                                           */
/*                         XrdMonSndDummyXrootd.hh                           */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONSNDDUMMYXROOTD_HH
#define XRDMONSNDDUMMYXROOTD_HH

#include <vector>
#include <string>
#include "XrdMon/XrdMonTypes.hh"
#include "XrdMon/XrdMonSndTraceEntry.hh"
#include "XrdMon/XrdMonSndDictEntry.hh"
#include "XrdMon/XrdMonSndStageEntry.hh"
using std::vector;
using std::string;

class XrdMonSndDummyXrootd {
public:
    static kXR_int16 NEWUSERFREQUENCY;
    static kXR_int16 NEWPROCFREQUENCY;
    static kXR_int16 NEWFILEFREQUENCY;
    static kXR_int16 MAXHOSTS;
    
    XrdMonSndDummyXrootd();
    ~XrdMonSndDummyXrootd();

    int initialize(const char* pathFile);
    XrdMonSndDictEntry newXrdMonSndDictEntry();
    XrdMonSndStageEntry newXrdMonSndStageEntry();
    XrdMonSndTraceEntry newXrdMonSndTraceEntry();
    kXR_int32 closeOneFile();
    void closeFiles(vector<kXR_int32>& closedFiles);
    
private:
    int readPaths(const char* pathFile);
    void createUser();
    void createProcess();
    void createFile();
    string generateUserName(kXR_int16 uid);
    string generateHostName();
    
    struct User {
        struct HostAndPid {
            string name;
            kXR_int16 pid;
            vector<kXR_int16> myFiles; // offsets in _paths vector
            HostAndPid(string n, kXR_int16 id) 
                : name(n), pid(id) {};
        };

        kXR_int16 uid;
        vector<HostAndPid> myProcesses;
        User(kXR_int16 id) : uid(id) {}
    };

    vector<User> _users;

    kXR_int32 _noCalls2NewUser;
    kXR_int32 _noCalls2NewProc;
    kXR_int32 _noCalls2NewFile;

    kXR_int16 _activeUser;
    kXR_int16 _activeProcess;
    kXR_int16 _activeFile;
    bool    _newFile;

    struct PathData {
        string path;
        kXR_int16 fd;
        PathData(const char* s, kXR_int16 id) : path(s), fd(id) {}
    };

    // input data to pick from, loaded from ascii file
    // Yes, this might be a lot of memory
    vector<PathData> _paths;

    kXR_int32 _firstAvailId;
    vector<kXR_unt32> _noTracesPerDict;

    vector<bool> _openFiles; // true: open, false: close
};

#endif /* XRDMONSNDDUMMYXROOTD_HH */
