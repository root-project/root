// $Id$
// Author: Sergey Linev   21/05/2015

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THttpCallArg.h"

#include <cstring>

#include "RZip.h"
#include "THttpWSEngine.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpCallArg                                                         //
//                                                                      //
// Contains arguments for single HTTP call                              //
// Must be used in THttpEngine to process incoming http requests        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(THttpCallArg);

////////////////////////////////////////////////////////////////////////////////
/// destructor

THttpCallArg::~THttpCallArg()
{
}

////////////////////////////////////////////////////////////////////////////////
/// method used to get or set http header in the string buffer
/// Header has following format:
///   field1 : value1\\r\\n
///   field2 : value2\\r\\n
/// Such format corresponds to header format in HTTP requests

TString THttpCallArg::AccessHeader(TString &buf, const char *name, const char *value, Bool_t doing_set)
{
   if (name == 0)
      return TString();

   Int_t curr = 0;

   while (curr < buf.Length() - 2) {

      Int_t next = buf.Index("\r\n", curr);
      if (next == kNPOS)
         break; // should never happen

      if (buf.Index(name, curr) != curr) {
         curr = next + 2;
         continue;
      }

      if ((value == 0) && doing_set) {
         // special case - empty value means that field must be removed completely
         buf.Remove(curr, next - curr + 2);
         return TString();
      }

      curr += strlen(name);
      while ((curr < next) && (buf[curr] != ':'))
         curr++;
      curr++;
      while ((curr < next) && (buf[curr] == ' '))
         curr++;

      if (value == 0)
         return buf(curr, next - curr);
      buf.Remove(curr, next - curr);
      buf.Insert(curr, value);
      return TString(value);
   }

   if (value == 0)
      return TString();

   buf.Append(TString::Format("%s: %s\r\n", name, value));
   return TString(value);
}

////////////////////////////////////////////////////////////////////////////////
/// method used to counter number of headers or returns name of specified header

TString THttpCallArg::CountHeader(const TString &buf, Int_t number) const
{
   Int_t curr(0), cnt(0);

   while (curr < buf.Length() - 2) {

      Int_t next = buf.Index("\r\n", curr);
      if (next == kNPOS)
         break; // should never happen

      if (cnt == number) {
         // we should extract name of header
         Int_t separ = curr + 1;
         while ((separ < next) && (buf[separ] != ':'))
            separ++;
         return buf(curr, separ - curr);
      }

      curr = next + 2;
      cnt++;
   }

   // return total number of headers
   if (number == -1111)
      return TString::Format("%d", cnt);
   return TString();
}


////////////////////////////////////////////////////////////////////////////////
/// Set content as text.
/// Content will be copied by THttpCallArg
void THttpCallArg::SetContent(const char *cont)
{
   if (cont)
      fContent = cont;
   else
      fContent.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Set text or binary content directly
/// After method call argument cont will be in undefined state

void THttpCallArg::SetContent(std::string &&cont)
{
   fContent = cont;
}

////////////////////////////////////////////////////////////////////////////////
/// Set content type as "text/plain"

void THttpCallArg::SetText()
{
   SetContentType("text/plain");
}

////////////////////////////////////////////////////////////////////////////////
/// Set content type as "text/plain" and also assigns content
/// After method call argument \param txt will be in undefined state

void THttpCallArg::SetTextContent(std::string &&txt)
{
   SetText();
   fContent = txt;
}

////////////////////////////////////////////////////////////////////////////////
/// Set content type as "text/xml"

void THttpCallArg::SetXml()
{
   SetContentType("text/xml");
}

////////////////////////////////////////////////////////////////////////////////
/// Set content type as "text/xml" and also assigns content
/// After method call argument \param xml will be in undefined state

void THttpCallArg::SetXmlContent(std::string &&xml)
{
   SetXml();
   fContent = xml;
}

////////////////////////////////////////////////////////////////////////////////
/// Set content type as "application/json"

void THttpCallArg::SetJson()
{
   SetContentType("application/json");
}

////////////////////////////////////////////////////////////////////////////////
/// Set content type as "application/json" and also assigns content
/// After method call argument \param json will be in undefined state

void THttpCallArg::SetJsonContent(std::string &&json)
{
   SetJson();
   fContent = json;
}

////////////////////////////////////////////////////////////////////////////////
/// Set content type as "application/x-binary"

void THttpCallArg::SetBinary()
{
   SetContentType("application/x-binary");
}

////////////////////////////////////////////////////////////////////////////////
/// Set content type as "application/x-binary" and also assigns content
/// After method call argument \param bin will be in undefined state

void THttpCallArg::SetBinaryContent(std::string &&bin)
{
   SetBinary();
   fContent = bin;
}

////////////////////////////////////////////////////////////////////////////////
/// \deprecated  Use signature with std::string
/// Set data, posted with the request
/// If make_copy==kFALSE, data will be released with free(data) call

void THttpCallArg::SetPostData(void *data, Long_t length, Bool_t make_copy)
{
   fPostData.resize(length);

   if (data && length) {
      std::copy((const char *)data, (const char *)data + length, fPostData.begin());
      if (!make_copy) free(data); // it supposed to get ownership over the buffer
   }
}

////////////////////////////////////////////////////////////////////////////////
/// set data, which is posted with the request
/// Although std::string is used, not only text data can be assigned -
/// std::string can contain any sequence of symbols

void THttpCallArg::SetPostData(std::string &&data)
{
   fPostData = data;
}

////////////////////////////////////////////////////////////////////////////////
/// Assign websocket identifier from the engine

void THttpCallArg::AssignWSId()
{
   SetWSId(fWSEngine->GetId());
}

////////////////////////////////////////////////////////////////////////////////
/// takeout websocket handle with HTTP call
/// can be done only once

std::shared_ptr<THttpWSEngine> THttpCallArg::TakeWSEngine()
{
   auto res = fWSEngine;
   fWSEngine.reset();
   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Replace all occurrences of \param from by \param to in content
/// \param once set to true to stop after first occurrence is replaced
/// \note Used only internally

void THttpCallArg::ReplaceAllinContent(const std::string &from, const std::string &to, bool once)
{
   std::size_t start_pos = 0;
   while((start_pos = fContent.find(from, start_pos)) != std::string::npos) {
      fContent.replace(start_pos, from.length(), to);
      if (once) return;
      start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
   }
}

////////////////////////////////////////////////////////////////////////////////
/// set complete path of requested http element
/// For instance, it could be "/folder/subfolder/get.bin"
/// Here "/folder/subfolder/" is element path and "get.bin" requested file.
/// One could set path and file name separately

void THttpCallArg::SetPathAndFileName(const char *fullpath)
{
   fPathName.Clear();
   fFileName.Clear();

   if (fullpath == 0)
      return;

   const char *rslash = strrchr(fullpath, '/');
   if (rslash == 0) {
      fFileName = fullpath;
   } else {
      while ((fullpath != rslash) && (*fullpath == '/'))
         fullpath++;
      fPathName.Append(fullpath, rslash - fullpath);
      if (fPathName == "/")
         fPathName.Clear();
      fFileName = rslash + 1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// return specified header

TString THttpCallArg::GetHeader(const char *name)
{
   if ((name == 0) || (*name == 0))
      return TString();

   if (strcmp(name, "Content-Type") == 0)
      return fContentType;
   if (strcmp(name, "Content-Length") == 0)
      return TString::Format("%ld", GetContentLength());

   return AccessHeader(fHeader, name);
}

////////////////////////////////////////////////////////////////////////////////
/// Set name: value pair to reply header
/// Content-Type field handled separately - one should use SetContentType() method
/// Content-Length field cannot be set at all;

void THttpCallArg::AddHeader(const char *name, const char *value)
{
   if ((name == 0) || (*name == 0) || (strcmp(name, "Content-Length") == 0))
      return;

   if (strcmp(name, "Content-Type") == 0)
      SetContentType(value);
   else
      AccessHeader(fHeader, name, value, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Set CacheControl http header to disable browser caching

void THttpCallArg::AddNoCacheHeader()
{
   AddHeader("Cache-Control", "private, no-cache, no-store, must-revalidate, max-age=0, proxy-revalidate, s-maxage=0");
}

////////////////////////////////////////////////////////////////////////////////
/// Fills HTTP header, which can be send at the beggining of reply on the http request
/// \param name is HTTP protocol name (default "HTTP/1.1")

std::string THttpCallArg::FillHttpHeader(const char *name)
{
   std::string hdr(name ? name : "HTTP/1.1");

   if ((fContentType.Length() == 0) || Is404())
      hdr.append(" 404 Not Found\r\n"
                 "Content-Length: 0\r\n"
                 "Connection: close\r\n\r\n");
   else
      hdr.append(Form(" 200 OK\r\n"
                      "Content-Type: %s\r\n"
                      "Connection: keep-alive\r\n"
                      "Content-Length: %ld\r\n"
                      "%s\r\n",
                      GetContentType(), GetContentLength(), fHeader.Data()));

   return hdr;
}

////////////////////////////////////////////////////////////////////////////////
/// compress reply data with gzip compression

Bool_t THttpCallArg::CompressWithGzip()
{
   char *objbuf = (char *)GetContent();
   Long_t objlen = GetContentLength();

   unsigned long objcrc = R__crc32(0, NULL, 0);
   objcrc = R__crc32(objcrc, (const unsigned char *)objbuf, objlen);

   // 10 bytes (ZIP header), compressed data, 8 bytes (CRC and original length)
   Int_t buflen = 10 + objlen + 8;
   if (buflen < 512)
      buflen = 512;

   std::string buffer;
   buffer.resize(buflen);

   char *bufcur = (char *)buffer.data();

   *bufcur++ = 0x1f; // first byte of ZIP identifier
   *bufcur++ = 0x8b; // second byte of ZIP identifier
   *bufcur++ = 0x08; // compression method
   *bufcur++ = 0x00; // FLAG - empty, no any file names
   *bufcur++ = 0;    // empty timestamp
   *bufcur++ = 0;    //
   *bufcur++ = 0;    //
   *bufcur++ = 0;    //
   *bufcur++ = 0;    // XFL (eXtra FLags)
   *bufcur++ = 3;    // OS   3 means Unix
   // strcpy(bufcur, "item.json");
   // bufcur += strlen("item.json")+1;

   char dummy[8];
   memcpy(dummy, bufcur - 6, 6);

   // R__memcompress fills first 6 bytes with own header, therefore just overwrite them
   unsigned long ziplen = R__memcompress(bufcur - 6, objlen + 6, objbuf, objlen);

   memcpy(bufcur - 6, dummy, 6);

   bufcur += (ziplen - 6); // jump over compressed data (6 byte is extra ROOT header)

   // write CRC32
   *bufcur++ = objcrc & 0xff;
   *bufcur++ = (objcrc >> 8) & 0xff;
   *bufcur++ = (objcrc >> 16) & 0xff;
   *bufcur++ = (objcrc >> 24) & 0xff;

   // write original data length
   *bufcur++ = objlen & 0xff;
   *bufcur++ = (objlen >> 8) & 0xff;
   *bufcur++ = (objlen >> 16) & 0xff;
   *bufcur++ = (objlen >> 24) & 0xff;

   buffer.resize(bufcur - (char *)buffer.data());

   SetContent(std::move(buffer));

   SetEncoding("gzip");

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// method used to notify condition which waiting when operation will complete
/// Condition notified only if not-postponed state is set

void THttpCallArg::NotifyCondition()
{
   if (!fNotifyFlag && !IsPostponed()) {
      fNotifyFlag = kTRUE;
      HttpReplied();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// virtual method to inform object that http request is processed
/// Normally condition is notified and waiting thread will be awaked
/// One could reimplement this method in sub-class

void THttpCallArg::HttpReplied()
{
   fCond.notify_one();
}
