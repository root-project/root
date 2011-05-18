//          $Id$

#include "XrdSecssl/XrdSecProtocolsslProc.hh"
#include <string.h>

bool
XrdSecProtocolsslProcFile::Open() {
  if (procsync) {
    fd = open(fname.c_str(),O_CREAT| O_SYNC|O_RDWR, S_IRWXU | S_IROTH | S_IRGRP );
  } else {
    fd = open(fname.c_str(),O_CREAT|O_RDWR, S_IRWXU | S_IROTH | S_IRGRP);
  }

  if (fd<0) {
    return false;
  }
  return true;
}

bool 
XrdSecProtocolsslProcFile::Write(long long val, int writedelay) {
  char pbuf[1024];
  sprintf(pbuf,"%lld\n",val);
  return Write(pbuf,writedelay);
}

bool 
XrdSecProtocolsslProcFile::Write(double val, int writedelay) {
  char pbuf[1024];
  sprintf(pbuf,"%.02f\n",val);
  return Write(pbuf,writedelay);
}

bool 
XrdSecProtocolsslProcFile::Write(const char* pbuf, int writedelay) {
  time_t now = time(NULL);
  if (writedelay) { 

    if (now-lastwrite <writedelay) {
      return true;
    }
  }

  int result;
  lseek(fd,0,SEEK_SET);
  while ( (result=::ftruncate(fd,0)) && (errno == EINTR ) ) {}
  lastwrite = now;
  if ( (write(fd,pbuf,strlen(pbuf))) == (int)(strlen(pbuf))) {
    return true;
  } else {
    return false;
  }
}

bool
XrdSecProtocolsslProcFile::WriteKeyVal(const char* key, unsigned long long value, int writedelay, bool dotruncate) {
  if (dotruncate) {
    time_t now = time(NULL);
    if (writedelay) {
      
      if (now-lastwrite <writedelay) {
	return false;
      }
    }

    //    printf("Truncating FD %d for %s\n",fd,key);
    lseek(fd,0,SEEK_SET);
    while ( (::ftruncate(fd,0)) && (errno == EINTR ) ) {}
    lastwrite = now;
  }
  char pbuf[1024];
  sprintf(pbuf,"%u %-32s %lld\n",(unsigned int)time(NULL),key,value);
  if ( (write(fd,pbuf,strlen(pbuf))) == (int)(strlen(pbuf))) {
    return true;
  } else {
    return false;
  }
}

long long
XrdSecProtocolsslProcFile::Read() {
  char pbuf[1024];
  lseek(fd,0,SEEK_SET);
  ssize_t rb = read(fd,pbuf,sizeof(pbuf));
  if (rb<=0) 
    return -1;

  return strtoll(pbuf,(char**)NULL,10);
}

bool 
XrdSecProtocolsslProcFile::Read(XrdOucString &str) {
  char pbuf[1024];
  pbuf[0] = 0;
  lseek(fd,0,SEEK_SET);
  ssize_t rb = read(fd,pbuf,sizeof(pbuf));
  str = pbuf;
  if (rb<=0)
    return false;
  else
    return true;
}

XrdSecProtocolsslProcFile*
XrdSecProtocolsslProc::Handle(const char* name) {
  XrdSecProtocolsslProcFile* phandle=0;
  if (( phandle = files.Find(name))) {
    return phandle;
  } else {
    XrdOucString pfname=procdirectory;
    pfname += "/";
    pfname += name;
    phandle = new XrdSecProtocolsslProcFile(pfname.c_str());
    if (phandle && phandle->Open()) {
      files.Add(name,phandle);
      return phandle;
    }
  }
  return NULL;
}


