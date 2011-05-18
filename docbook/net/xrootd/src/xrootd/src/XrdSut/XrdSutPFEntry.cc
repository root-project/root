// $Id$

const char *XrdSutPFEntryCVSID = "$Id$";
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "XrdSutAux.hh"
#include "XrdSutPFEntry.hh"

//__________________________________________________________________
XrdSutPFBuf::XrdSutPFBuf(char *b, kXR_int32 l)
{
   // Constructor

   len = 0;
   buf = 0; 
   if (b) { 
      buf = b;
      len = l;
   }
}

//__________________________________________________________________
XrdSutPFBuf::XrdSutPFBuf(const XrdSutPFBuf &b)
{
   //Copy constructor

   buf = 0;
   len = 0; 
   if (b.buf) {
      buf = new char[b.len];
      if (buf) {
         memcpy(buf,b.buf,b.len);
         len = b.len;
      }
   }
}

//__________________________________________________________________
void XrdSutPFBuf::SetBuf(const char *b, kXR_int32 l)
{
   // Set the buffer

   len = 0;
   if (buf) {
      delete[] buf;
      buf = 0; 
   }
   if (b && l > 0) {
      buf = new char[l];
      if (buf) {
         memcpy(buf,b,l);
         len = l;
      }
   }
}

//____________________________________________________________________
XrdSutPFEntry::XrdSutPFEntry(const char *n, short st, short cn,
                             kXR_int32 mt)
{
   // Constructor

   name = 0;
   status = st;
   cnt    = cn;
   mtime  = (mt > 0) ? mt : (kXR_int32)time(0);
   if (n) { 
      name = new char[strlen(n)+1];
      if (name)
         strcpy(name,n);
   }
}

//_____________________________________________________________________
XrdSutPFEntry::XrdSutPFEntry(const XrdSutPFEntry &e) : buf1(e.buf1),
                     buf2(e.buf2), buf3(e.buf3), buf4(e.buf4)
{
   // Copy constructor

   name = 0;
   status = e.status;
   cnt    = e.cnt;
   mtime  = e.mtime;
   if (e.name) {
      name = new char[strlen(e.name)+1];
      if (name)
         strcpy(name,e.name);
   }
}

//____________________________________________________________________
void XrdSutPFEntry::Reset()
{
   // Resetting entry

   if (name)
      delete[] name;
   name = 0;
   status = 0;
   cnt    = 0;
   mtime  = (kXR_int32)time(0);
   buf1.SetBuf();
   buf2.SetBuf();
   buf3.SetBuf();
   buf4.SetBuf();
}

//_____________________________________________________________________
void XrdSutPFEntry::SetName(const char *n)
{
   // Set the name

   if (name) { 
      delete[] name;
      name = 0;
   }
   if (n) {
      name = new char[strlen(n)+1];
      if (name)
         strcpy(name,n);
   }
}

//_____________________________________________________________________
char *XrdSutPFEntry::AsString() const
{
   // Return a string with serialized information
   // For print purposes
   // The output string points to a static buffer, so it must
   // not be deleted by the caller
   static char pbuf[2048];

   char smt[20] = {0};
   XrdSutTimeString(mtime,smt);

   sprintf(pbuf,"st:%d cn:%d  buf:%d,%d,%d,%d modified:%s name:%s",
           status,cnt,buf1.len,buf2.len,buf3.len,buf4.len,smt,name);
 
   return pbuf;
}

//______________________________________________________________________________
XrdSutPFEntry& XrdSutPFEntry::operator=(const XrdSutPFEntry &e)
{
   // Assign entry e to local entry.

   SetName(name);
   status = e.status;
   cnt    = e.cnt;            // counter
   mtime  = e.mtime;          // time of last modification / creation
   buf1.SetBuf(e.buf1.buf);
   buf2.SetBuf(e.buf2.buf);
   buf3.SetBuf(e.buf3.buf);
   buf4.SetBuf(e.buf4.buf);

   return (*this);
}
