// $Id$
// Author: Sergey Linev   21/05/2015

#include "THttpCallArg.h"

#include <string.h>
#include "RZip.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpCallArg                                                         //
//                                                                      //
// Contains arguments for single HTTP call                              //
// Must be used in THttpEngine to process incoming http requests        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
THttpCallArg::THttpCallArg() :
   TObject(),
   fTopName(),
   fMethod(),
   fPathName(),
   fFileName(),
   fUserName(),
   fQuery(),
   fPostData(0),
   fPostDataLength(0),
   fCond(),
   fContentType(),
   fRequestHeader(),
   fHeader(),
   fContent(),
   fZipping(0),
   fBinData(0),
   fBinDataLength(0)
{
   // constructor
}

//______________________________________________________________________________
THttpCallArg::~THttpCallArg()
{
   // destructor

   if (fPostData) {
      free(fPostData);
      fPostData = 0;
   }

   if (fBinData) {
      free(fBinData);
      fBinData = 0;
   }
}

//______________________________________________________________________________
TString THttpCallArg::AccessHeader(TString& buf, const char* name, const char* value, Bool_t doing_set)
{
   // method used to get or set http header in the string buffer
   // Header has following format:
   //   field1 : value1\r\n
   //   field2 : value2\r\n
   // Such format corresponds to header format in HTTP requests

   if (name==0) return TString();

   Int_t curr = 0;

   while (curr < buf.Length()-2) {

      Int_t next = buf.Index("\r\n", curr);
      if (next == kNPOS) break; // should never happen

      if (buf.Index(name, curr) != curr) {
         curr = next + 2;
         continue;
      }

      if ((value==0) && doing_set) {
         // special case - empty value means that field must be removed completely
         buf.Remove(curr, next-curr+2);
         return TString();
      }

      curr += strlen(name);
      while ((curr < next) && (buf[curr] != ':')) curr++;
      curr++;
      while ((curr < next) && (buf[curr] == ' ')) curr++;

      if (value==0) return buf(curr, next-curr);
      buf.Remove(curr, next-curr);
      buf.Insert(curr, value);
      return TString(value);
   }

   if (value==0) return TString();

   buf.Append(TString::Format("%s: %s\r\n", name, value));
   return TString(value);
}

//______________________________________________________________________________
TString THttpCallArg::CountHeader(const TString& buf, Int_t number) const
{
   // method used to counter number of headers or returns name of specified header

   Int_t curr(0), cnt(0);

   while (curr < buf.Length()-2) {

      Int_t next = buf.Index("\r\n", curr);
      if (next == kNPOS) break; // should never happen

      if (cnt == number) {
         // we should extract name of header
         Int_t separ = curr + 1;
         while ((separ < next) && (buf[separ] != ':')) separ++;
         return buf(curr, separ-curr);
      }

      curr = next + 2;
      cnt++;
   }

   // return total number of headers
   if (number == -1111) return TString::Format("%d", cnt);
   return TString();
}

//______________________________________________________________________________
void THttpCallArg::SetPostData(void *data, Long_t length)
{
   // set data, posted with the request
   // buffer should be allocated with malloc(length+1) call,
   // while last byte will be set to 0
   // Than one could use post data as null-terminated string

   if (fPostData) free(fPostData);
   if (data!=0) *(((char*) data) + length) = 0;
   fPostData = data;
   fPostDataLength = length;
}

//______________________________________________________________________________
void THttpCallArg::SetBinData(void *data, Long_t length)
{
   // set binary data, which will be returned as reply body

   if (fBinData) free(fBinData);
   fBinData = data;
   fBinDataLength = length;

   // string content must be cleared in any case
   fContent.Clear();
}

//______________________________________________________________________________
void THttpCallArg::SetPathAndFileName(const char *fullpath)
{
   // set complete path of requested http element
   // For instance, it could be "/folder/subfolder/get.bin"
   // Here "/folder/subfolder/" is element path and "get.bin" requested file.
   // One could set path and file name separately

   fPathName.Clear();
   fFileName.Clear();

   if (fullpath == 0) return;

   const char *rslash = strrchr(fullpath, '/');
   if (rslash == 0) {
      fFileName = fullpath;
   } else {
      while ((fullpath != rslash) && (*fullpath == '/')) fullpath++;
      fPathName.Append(fullpath, rslash - fullpath);
      if (fPathName == "/") fPathName.Clear();
      fFileName = rslash + 1;
   }
}

//______________________________________________________________________________
TString THttpCallArg::GetHeader(const char* name)
{
   // return specified header

   if ((name == 0) || (*name == 0)) return TString();

   if (strcmp(name,"Content-Type") == 0) return fContentType;
   if (strcmp(name,"Content-Length") == 0) return TString::Format("%ld", GetContentLength());

   return AccessHeader(fHeader, name);
}

//______________________________________________________________________________
void THttpCallArg::AddHeader(const char *name, const char *value)
{
   // Set name: value pair to reply header
   // Content-Type field handled separately - one should use SetContentType() method
   // Content-Length field cannot be set at all;

   if ((name == 0) || (*name == 0) || (strcmp(name,"Content-Length") == 0)) return;

   if (strcmp(name,"Content-Type") == 0)
      SetContentType(value);
   else
      AccessHeader(fHeader, name, value, kTRUE);
}

//______________________________________________________________________________
void THttpCallArg::FillHttpHeader(TString &hdr, const char *kind)
{
   // fill HTTP header

   if (kind == 0) kind = "HTTP/1.1";

   if ((fContentType.Length() == 0) || Is404()) {
      hdr.Form("%s 404 Not Found\r\n"
               "Content-Length: 0\r\n"
               "Connection: close\r\n\r\n", kind);
   } else {
      hdr.Form("%s 200 OK\r\n"
               "Content-Type: %s\r\n"
               "Connection: keep-alive\r\n"
               "Content-Length: %ld\r\n"
               "%s\r\n",
               kind,
               GetContentType(),
               GetContentLength(),
               fHeader.Data());
   }
}

//______________________________________________________________________________
Bool_t THttpCallArg::CompressWithGzip()
{
   // compress reply data with gzip compression

   char *objbuf = (char *) GetContent();
   Long_t objlen = GetContentLength();

   unsigned long objcrc = R__crc32(0, NULL, 0);
   objcrc = R__crc32(objcrc, (const unsigned char *) objbuf, objlen);

   // 10 bytes (ZIP header), compressed data, 8 bytes (CRC and original length)
   Int_t buflen = 10 + objlen + 8;
   if (buflen < 512) buflen = 512;

   void *buffer = malloc(buflen);

   char *bufcur = (char *) buffer;

   *bufcur++ = 0x1f;  // first byte of ZIP identifier
   *bufcur++ = 0x8b;  // second byte of ZIP identifier
   *bufcur++ = 0x08;  // compression method
   *bufcur++ = 0x00;  // FLAG - empty, no any file names
   *bufcur++ = 0;    // empty timestamp
   *bufcur++ = 0;    //
   *bufcur++ = 0;    //
   *bufcur++ = 0;    //
   *bufcur++ = 0;    // XFL (eXtra FLags)
   *bufcur++ = 3;    // OS   3 means Unix
   //strcpy(bufcur, "item.json");
   //bufcur += strlen("item.json")+1;

   char dummy[8];
   memcpy(dummy, bufcur - 6, 6);

   // R__memcompress fills first 6 bytes with own header, therefore just overwrite them
   unsigned long ziplen = R__memcompress(bufcur - 6, objlen + 6, objbuf, objlen);

   memcpy(bufcur - 6, dummy, 6);

   bufcur += (ziplen - 6); // jump over compressed data (6 byte is extra ROOT header)

   *bufcur++ = objcrc & 0xff;    // CRC32
   *bufcur++ = (objcrc >> 8) & 0xff;
   *bufcur++ = (objcrc >> 16) & 0xff;
   *bufcur++ = (objcrc >> 24) & 0xff;

   *bufcur++ = objlen & 0xff;  // original data length
   *bufcur++ = (objlen >> 8) & 0xff;  // original data length
   *bufcur++ = (objlen >> 16) & 0xff;  // original data length
   *bufcur++ = (objlen >> 24) & 0xff;  // original data length

   SetBinData(buffer, bufcur - (char *) buffer);

   SetEncoding("gzip");

   return kTRUE;
}
