// $Id$
// Author: Sergey Linev   21/05/2015

#include "THttpCallArg.h"

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
   fQuery(),
   fPostData(0),
   fPostDataLength(0),
   fCond(),
   fContentType(),
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
void THttpCallArg::SetPostData(void *data, Long_t length)
{
   // set data, posted with the request

   if (fPostData) free(fPostData);
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
