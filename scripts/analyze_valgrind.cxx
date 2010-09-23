#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <list>
#include <sstream>
#include <stdlib.h>

using namespace std;

int GetPID(string& line) {
   string::size_type posEqual = line.find("==");
   if (posEqual == string::npos) return -1;
   line.erase(0, posEqual + 2);
   stringstream ssline(line);
   int pid;
   ssline >> pid;
   line.erase(0, line.find("==") + 2);
   return pid;
}

class PIDInfo {
public:
   enum EPrintWhat {
      kErrors = 1,
      kLeaks  = 2,
      kAll    = 0xff
   };

   PIDInfo(int pid): fPID(pid) {}

   void Print(EPrintWhat what) {
      bool errors =  (what & kErrors) && fErrors > 0;
      bool leaks =   (what & kLeaks)
         && fLeakBytesDefinitely + fLeakBytesPossibly + fLeakBytesReachable > 0;

      if (!errors && !leaks) return;

      cout << "Test " << fName << ": ";
      if (errors)
         cout << fErrors << endl;
      if (leaks)
         cout << fLeakBytesDefinitely << " / " 
           << fLeakBytesPossibly << " / " << fLeakBytesReachable << endl;
   }

   int fPID;
   string fName;
   long fErrors;
   long fLeakBytesDefinitely;
   long fLeakBytesPossibly;
   long fLeakBytesReachable;
};

long ParseNumber(string& line, const char* tag) {
   line.erase(0, line.find(':') + 1);
   line.erase(line.find(tag));
   stringstream sslostbytes(line);
   long bytes = 0;
   do {
      int digits = 0;
      sslostbytes >> digits;
      if (sslostbytes) {
         bytes *= 1000;
         bytes += digits;
      }
   } while (sslostbytes);
   return bytes;
}

void Parse(long leakOffset) {
   istream* pin = &cin;
   ifstream fin;
   if (!cin) {
      fin.open("test.log");
      pin = &fin;
   }
   istream& in(*pin);

   map<int, PIDInfo*> mapPIDInfos;
   list<PIDInfo*> logPIDInfos;

   // these PIDs are waiting for the test name -
   // last argument in the list of args after "My PID"
   set<int> testsWaitingForName;

   do {
      string line;
      getline(in, line);
      int pid = GetPID(line);
      while (isspace(line[0]))
         line.erase(0,1);
      while (!line.empty() && isspace(line[line.length() - 1]))
             line.erase(line.length() - 1);

      if (line.find("My PID =") != std::string::npos || line.find("Memcheck, a memory error detector") != std::string::npos ) {
         if (mapPIDInfos.find(pid) == mapPIDInfos.end())
            delete mapPIDInfos[pid];
         PIDInfo* pidinfo = new PIDInfo(pid);
         mapPIDInfos[pid] = pidinfo;
         testsWaitingForName.insert(pid);
         continue;
      } else if (!line.empty()
                 && testsWaitingForName.find(pid) != testsWaitingForName.end()) {
         mapPIDInfos[pid]->fName = line;
         continue;
      }

      if (line.empty()) {
         set<int>::iterator iWaiting = testsWaitingForName.find(pid);
         if (iWaiting != testsWaitingForName.end()) {
            testsWaitingForName.erase(iWaiting);
         }
      }

      if (line.find("ERROR SUMMARY") != string::npos) {
         PIDInfo* pidinfo = mapPIDInfos[pid];
         if (pidinfo) {
            mapPIDInfos[pid]->fErrors = ParseNumber(line, " errors");

            // this also marks the end of life for this PID.
            if (pidinfo->fErrors > 0
                || pidinfo->fLeakBytesDefinitely > leakOffset)
               logPIDInfos.push_back(pidinfo);
            mapPIDInfos.erase(pid);
         }
      } else if (line.find("definitely lost: ") != string::npos) {
         mapPIDInfos[pid]->fLeakBytesDefinitely = ParseNumber(line, " bytes");
      } else if (line.find("possibly lost: ") != string::npos) {
         mapPIDInfos[pid]->fLeakBytesPossibly = ParseNumber(line, " bytes");
      } else if (line.find("still reachable: ") != string::npos) {
         mapPIDInfos[pid]->fLeakBytesReachable = ParseNumber(line, " bytes");
      }
   } while (in);

   cout << endl << " === ERRORS === " << endl;
   for (list<PIDInfo*>::const_iterator iInfo = logPIDInfos.begin();
        iInfo != logPIDInfos.end(); ++iInfo) {
      PIDInfo* pidinfo = *iInfo;
      if (pidinfo->fErrors > 0)
      pidinfo->Print(PIDInfo::kErrors);
   }  
   cout << endl << " === LEAKS (definitely / possibly / reachable) === " << endl;
   for (list<PIDInfo*>::const_iterator iInfo = logPIDInfos.begin();
        iInfo != logPIDInfos.end(); ++iInfo) {
      PIDInfo* pidinfo = *iInfo;
      if (pidinfo->fLeakBytesDefinitely > leakOffset)
         pidinfo->Print(PIDInfo::kLeaks);
      delete pidinfo;
   }  
}


int main(int argc, char* argv[]) {
   long leakOffset = 0;
   if (argc > 1) {
      string arg1(argv[1]);
      if (arg1.find("--leakoffset=") == 0) {
         stringstream ssarg1(arg1.substr(13));
         ssarg1 >> leakOffset;
         cout << "Using minimal leakage = " << leakOffset << endl;
      } else {
         cerr << "Analyzes a roottest valgrind log file." << endl
              << "USAGE: analyze_valgrind [--leakoffset=<val>]" << endl
              << "  --leakoffset: leakage bytes to ignore, default: 0" << endl
              << endl;
         exit(1);
      }
   }
   Parse(leakOffset);
   return 0;
}
