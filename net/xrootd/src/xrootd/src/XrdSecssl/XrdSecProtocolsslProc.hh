//         $Id$

#ifndef __SECPROTOCOLSSL_PROC__
#define __SECPROTOCOLSSL_PROC__

#include "XrdOuc/XrdOucString.hh"
#include "XrdOuc/XrdOucHash.hh"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>

class XrdSecProtocolsslProcFile
{
private:
  int fd;
  XrdOucString fname;
  bool procsync;
  time_t lastwrite;

public:
  bool Open();
  bool Close() { if (fd>=0) close(fd);return true;}
  bool Write(long long val, int writedelay=0);
  bool Write(double val, int writedelay=0);
  bool Write(const char* str, int writedelay=0);
  bool WriteKeyVal(const char* key, unsigned long long value, int writedelay, bool truncate=0);
  long long Read();
  bool Read(XrdOucString &str);
  

  XrdSecProtocolsslProcFile(const char* name, bool syncit=false){fname = name;fd=0;procsync = syncit;lastwrite=0;};
  virtual ~XrdSecProtocolsslProcFile() {Close();};
};

class XrdSecProtocolsslProc
{
private:
  bool procsync;
  XrdOucString procdirectory;
  XrdOucHash<XrdSecProtocolsslProcFile> files;

public:
  
  XrdSecProtocolsslProcFile* Handle(const char* name);

  XrdSecProtocolsslProc(const char* procdir, bool syncit) { 
    procdirectory = procdir; 
    procdirectory += "/ssl/";
    while((procdirectory.find("//")!=STR_NPOS)) { procdirectory.replace("//","/");}
    procsync = syncit;
  };

  bool Open() {
    XrdOucString doit="mkdir -p ";
    doit+=procdirectory;
    int rc = system(doit.c_str());
    if (rc == -1)
      return false;

    DIR* pd=opendir(procdirectory.c_str());
    if (!pd) {
      return false;
    } else {
      closedir(pd);
      return true;
    }
  }

  virtual ~XrdSecProtocolsslProc() {};
};
#endif

