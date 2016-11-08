// @(#)root/io:$Id$
// Author: Rene Brun 17/01/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBufferFile                                                          //
//                                                                      //
// The concrete implementation of TBuffer for writing/reading to/from a //
// ROOT file or socket.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string.h>
#include <typeinfo>
#include <string>

#include "TFile.h"
#include "TBufferFile.h"
#include "TExMap.h"
#include "TClass.h"
#include "TProcessID.h"
#include "TRefTable.h"
#include "TStorage.h"
#include "TError.h"
#include "TClonesArray.h"
#include "TStreamer.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TSchemaRuleSet.h"
#include "TStreamerInfoActions.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TArrayC.h"

#if (defined(__linux) || defined(__APPLE__)) && defined(__i386__) && \
     defined(__GNUC__)
#define USE_BSWAPCPY
#endif

#ifdef USE_BSWAPCPY
#include "Bswapcpy.h"
#endif


const UInt_t kNullTag           = 0;
const UInt_t kNewClassTag       = 0xFFFFFFFF;
const UInt_t kClassMask         = 0x80000000;  // OR the class index with this
const UInt_t kByteCountMask     = 0x40000000;  // OR the byte count with this
const UInt_t kMaxMapCount       = 0x3FFFFFFE;  // last valid fMapCount and byte count
const Version_t kByteCountVMask = 0x4000;      // OR the version byte count with this
const Version_t kMaxVersion     = 0x3FFF;      // highest possible version number
const Int_t  kMapOffset         = 2;   // first 2 map entries are taken by null obj and self obj

Int_t TBufferFile::fgMapSize   = kMapSize;


ClassImp(TBufferFile)

//______________________________________________________________________________
static inline ULong_t Void_Hash(const void *ptr)
{
   // Return hash value for this object.

   return TString::Hash(&ptr, sizeof(void*));
}

//______________________________________________________________________________
static inline bool Class_Has_StreamerInfo(const TClass* cl)
{
   // Thread-safe check on StreamerInfos of a TClass

   // NOTE: we do not need a R__LOCKGUARD2 since we know the
   //   mutex is available since the TClass constructor will make
   //   it
   R__LOCKGUARD(gCINTMutex);
   return cl->GetStreamerInfos()->GetLast()>1;
}

//______________________________________________________________________________
TBufferFile::TBufferFile(TBuffer::EMode mode)
            :TBuffer(mode),
             fDisplacement(0),fPidOffset(0), fMap(0), fClassMap(0),
             fInfo(0), fInfoStack()
{
   // Create an I/O buffer object. Mode should be either TBuffer::kRead or
   // TBuffer::kWrite. By default the I/O buffer has a size of
   // TBuffer::kInitialSize (1024) bytes.

   fMapCount     = 0;
   fMapSize      = fgMapSize;
   fMap          = 0;
   fClassMap     = 0;
   fParent       = 0;
   fDisplacement = 0;
}

//______________________________________________________________________________
TBufferFile::TBufferFile(TBuffer::EMode mode, Int_t bufsiz)
            :TBuffer(mode,bufsiz),
             fDisplacement(0),fPidOffset(0), fMap(0), fClassMap(0),
             fInfo(0), fInfoStack()
{
   // Create an I/O buffer object. Mode should be either TBuffer::kRead or
   // TBuffer::kWrite.

   fMapCount = 0;
   fMapSize  = fgMapSize;
   fMap      = 0;
   fClassMap = 0;
   fDisplacement = 0;
}

//______________________________________________________________________________
TBufferFile::TBufferFile(TBuffer::EMode mode, Int_t bufsiz, void *buf, Bool_t adopt, ReAllocCharFun_t reallocfunc) :
   TBuffer(mode,bufsiz,buf,adopt,reallocfunc),
   fDisplacement(0),fPidOffset(0), fMap(0), fClassMap(0),
   fInfo(0), fInfoStack()
{
   // Create an I/O buffer object. Mode should be either TBuffer::kRead or
   // TBuffer::kWrite. By default the I/O buffer has a size of
   // TBuffer::kInitialSize (1024) bytes. An external buffer can be passed
   // to TBuffer via the buf argument. By default this buffer will be adopted
   // unless adopt is false.
   // If the new buffer is _not_ adopted and no memory allocation routine
   // is provided, a Fatal error will be issued if the Buffer attempts to
   // expand.

   fMapCount = 0;
   fMapSize  = fgMapSize;
   fMap      = 0;
   fClassMap = 0;
   fDisplacement = 0;
}

//______________________________________________________________________________
TBufferFile::~TBufferFile()
{
   // Delete an I/O buffer object.

   delete fMap;
   delete fClassMap;
}

//______________________________________________________________________________
Int_t TBufferFile::GetVersionOwner() const
{
   // Return the version number of the owner file.

   TFile *file = (TFile*)GetParent();
   if (file) return file->GetVersion();
   else return 0;
}

//______________________________________________________________________________
void TBufferFile::TagStreamerInfo(TVirtualStreamerInfo* info)
{
   // Mark the classindex of the current file as using this TStreamerInfo

   TFile *file = (TFile*)GetParent();
   if (file) {
      TArrayC *cindex = file->GetClassIndex();
      Int_t nindex = cindex->GetSize();
      Int_t number = info->GetNumber();
      if (number < 0 || number >= nindex) {
         Error("TagStreamerInfo","StreamerInfo: %s number: %d out of range[0,%d] in file: %s",
               info->GetName(),number,nindex,file->GetName());
         return;
      }
      if (cindex->fArray[number] == 0) {
         cindex->fArray[0]       = 1;
         cindex->fArray[number] = 1;
      }
   }
}

//______________________________________________________________________________
void TBufferFile::IncrementLevel(TVirtualStreamerInfo* info)
{
   // Increment level.

   fInfoStack.push_back(fInfo);
   fInfo = (TStreamerInfo*)info;
}

//______________________________________________________________________________
void TBufferFile::DecrementLevel(TVirtualStreamerInfo* /*info*/)
{
   // Decrement level.

   fInfo = fInfoStack.back();
   fInfoStack.pop_back();
}

//______________________________________________________________________________
static void frombufOld(char *&buf, Long_t *x)
{
   // Files written with versions older than 3.00/06 had a non-portable
   // implementation of Long_t/ULong_t. These types should not have been
   // used at all. However, because some users had already written many
   // files with these types we provide this dirty patch for "backward
   // compatibility"

#ifdef R__BYTESWAP
#ifdef R__B64
   char *sw = (char *)x;
   sw[0] = buf[7];
   sw[1] = buf[6];
   sw[2] = buf[5];
   sw[3] = buf[4];
   sw[4] = buf[3];
   sw[5] = buf[2];
   sw[6] = buf[1];
   sw[7] = buf[0];
#else
   char *sw = (char *)x;
   sw[0] = buf[3];
   sw[1] = buf[2];
   sw[2] = buf[1];
   sw[3] = buf[0];
#endif
#else
   memcpy(x, buf, sizeof(Long_t));
#endif
   buf += sizeof(Long_t);
}

//______________________________________________________________________________
void TBufferFile::ReadLong(Long_t &l)
{
   // Read Long from TBuffer.

   TFile *file = (TFile*)fParent;
   if (file && file->GetVersion() < 30006) {
      frombufOld(fBufCur, &l);
   } else {
      frombuf(fBufCur, &l);
   }
}

//_______________________________________________________________________
void TBufferFile::ReadTString(TString &s)
{
   // Read TString from TBuffer.

   Int_t   nbig;
   UChar_t nwh;
   *this >> nwh;
   if (nwh == 0) {
      s.UnLink();
      s.Zero();
   } else {
      if (nwh == 255)
         *this >> nbig;
      else
         nbig = nwh;

      s.Clobber(nbig);
      char *data = s.GetPointer();
      data[nbig] = 0;
      s.SetSize(nbig);
      ReadFastArray(data, nbig);
   }
}

//_______________________________________________________________________
void TBufferFile::WriteTString(const TString &s)
{
   // Write TString to TBuffer.

   Int_t nbig = s.Length();
   UChar_t nwh;
   if (nbig > 254) {
      nwh = 255;
      *this << nwh;
      *this << nbig;
   } else {
      nwh = UChar_t(nbig);
      *this << nwh;
   }
   const char *data = s.GetPointer();
   WriteFastArray(data, nbig);
}

//_______________________________________________________________________
void TBufferFile::ReadStdString(std::string *obj)
{
   // Read std::string from TBuffer.

   if (obj == 0) {
      Error("TBufferFile::ReadStdString","The std::string address is nullptr but should not");
   }
   Int_t   nbig;
   UChar_t nwh;
   *this >> nwh;
   if (nwh == 0)  {
       obj->clear();
   } else {
      if( obj->size() ) {
         // Insure that the underlying data storage is not shared
         (*obj)[0] = '\0';
      }
      if (nwh == 255)  {
         *this >> nbig;
         obj->resize(nbig,'\0');
         ReadFastArray((char*)obj->data(),nbig);
      }
      else  {
         obj->resize(nwh,'\0');
         ReadFastArray((char*)obj->data(),nwh);
      }
   }
}

//_______________________________________________________________________
void TBufferFile::WriteStdString(const std::string *obj)
{
   // Write std::string to TBuffer.

   if (obj==0) {
      *this << (UChar_t)0;
      WriteFastArray("",0);
      return;
   }

   UChar_t nwh;
   Int_t nbig = obj->length();
   if (nbig > 254) {
      nwh = 255;
      *this << nwh;
      *this << nbig;
   } else {
      nwh = UChar_t(nbig);
      *this << nwh;
   }
   WriteFastArray(obj->data(),nbig);
}


//______________________________________________________________________________
void TBufferFile::ReadCharStar(char* &s)
{
   // Read char* from TBuffer.

   delete [] s;
   s = 0;

   Int_t nch;
   *this >> nch;
   if (nch > 0) {
      s = new char[nch+1];
      ReadFastArray(s, nch);
      s[nch] = 0;
   }
}

//______________________________________________________________________________
void TBufferFile::WriteCharStar(char *s)
{
   // Write char* into TBuffer.

   Int_t nch = 0;
   if (s) {
      nch = strlen(s);
      *this  << nch;
      WriteFastArray(s,nch);
   } else {
      *this << nch;
   }

}

//______________________________________________________________________________
void TBufferFile::SetByteCount(UInt_t cntpos, Bool_t packInVersion)
{
   // Set byte count at position cntpos in the buffer. Generate warning if
   // count larger than kMaxMapCount. The count is excluded its own size.

   UInt_t cnt = UInt_t(fBufCur - fBuffer) - cntpos - sizeof(UInt_t);
   char  *buf = (char *)(fBuffer + cntpos);

   // if true, pack byte count in two consecutive shorts, so it can
   // be read by ReadVersion()
   if (packInVersion) {
      union {
         UInt_t    cnt;
         Version_t vers[2];
      } v;
      v.cnt = cnt;
#ifdef R__BYTESWAP
      tobuf(buf, Version_t(v.vers[1] | kByteCountVMask));
      tobuf(buf, v.vers[0]);
#else
      tobuf(buf, Version_t(v.vers[0] | kByteCountVMask));
      tobuf(buf, v.vers[1]);
#endif
   } else
      tobuf(buf, cnt | kByteCountMask);

   if (cnt >= kMaxMapCount) {
      Error("WriteByteCount", "bytecount too large (more than %d)", kMaxMapCount);
      // exception
   }
}

//______________________________________________________________________________
Int_t TBufferFile::CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss, const char *classname)
{
   // Check byte count with current buffer position. They should
   // match. If not print warning and position buffer in correct
   // place determined by the byte count. Startpos is position of
   // first byte where the byte count is written in buffer.
   // Returns 0 if everything is ok, otherwise the bytecount offset
   // (< 0 when read too little, >0 when read too much).

   if (!bcnt) return 0;

   Int_t  offset = 0;

   Long_t endpos = Long_t(fBuffer) + startpos + bcnt + sizeof(UInt_t);

   if (Long_t(fBufCur) != endpos) {
      offset = Int_t(Long_t(fBufCur) - endpos);

      const char *name = clss ? clss->GetName() : classname ? classname : 0;

      if (name) {
         if (offset < 0) {
            Error("CheckByteCount", "object of class %s read too few bytes: %d instead of %d",
                  name,bcnt+offset,bcnt);
         }
         if (offset > 0) {
            Error("CheckByteCount", "object of class %s read too many bytes: %d instead of %d",
                  name,bcnt+offset,bcnt);
            if (fParent)
               Warning("CheckByteCount","%s::Streamer() not in sync with data on file %s, fix Streamer()",
                       name, fParent->GetName());
            else
               Warning("CheckByteCount","%s::Streamer() not in sync with data, fix Streamer()",
                       name);
         }
      }
      if ( ((char *)endpos) > fBufMax ) {
         offset = fBufMax-fBufCur;
         Error("CheckByteCount",
               "Byte count probably corrupted around buffer position %d:\n\t%d for a possible maximum of %d",
               startpos, bcnt, offset);
         fBufCur = fBufMax;

      } else {

         fBufCur = (char *) endpos;

      }
   }
   return offset;
}

//______________________________________________________________________________
Int_t TBufferFile::CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss)
{
   // Check byte count with current buffer position. They should
   // match. If not print warning and position buffer in correct
   // place determined by the byte count. Startpos is position of
   // first byte where the byte count is written in buffer.
   // Returns 0 if everything is ok, otherwise the bytecount offset
   // (< 0 when read too little, >0 when read too much).

   if (!bcnt) return 0;
   return CheckByteCount( startpos, bcnt, clss, 0);
}

//______________________________________________________________________________
Int_t TBufferFile::CheckByteCount(UInt_t startpos, UInt_t bcnt, const char *classname)
{
   // Check byte count with current buffer position. They should
   // match. If not print warning and position buffer in correct
   // place determined by the byte count. Startpos is position of
   // first byte where the byte count is written in buffer.
   // Returns 0 if everything is ok, otherwise the bytecount offset
   // (< 0 when read too little, >0 when read too much).

   if (!bcnt) return 0;
   return CheckByteCount( startpos, bcnt, 0, classname);
}

//______________________________________________________________________________
void TBufferFile::ReadFloat16(Float_t *f, TStreamerElement *ele)
{
   // Read a Float16_t from the buffer,
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16().

   if (ele && ele->GetFactor() != 0) {
      ReadWithFactor(f, ele->GetFactor(), ele->GetXmin());
   } else {
      Int_t nbits = 0;
      if (ele) nbits = (Int_t)ele->GetXmin();
      if (!nbits) nbits = 12;
      ReadWithNbits(f, nbits);
   }
}

//______________________________________________________________________________
void TBufferFile::ReadDouble32(Double_t *d, TStreamerElement *ele)
{
   // Read a Double32_t from the buffer,
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().

   if (ele && ele->GetFactor() != 0) {
      ReadWithFactor(d, ele->GetFactor(), ele->GetXmin());
   } else {
      Int_t nbits = 0;
      if (ele) nbits = (Int_t)ele->GetXmin();
      if (!nbits) {
         //we read a float and convert it to double
         Float_t afloat;
         *this >> afloat;
         d[0] = (Double_t)afloat;
      } else {
         ReadWithNbits(d, nbits);
      }
   }
}

//______________________________________________________________________________
void TBufferFile::ReadWithFactor(Float_t *ptr, Double_t factor, Double_t minvalue)
{
   // Read a Float16_t from the buffer when the factor and minimun value have been specified
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().

   //a range was specified. We read an integer and convert it back to a double.
   UInt_t aint;
   frombuf(this->fBufCur,&aint);
   ptr[0] = (Float_t)(aint/factor + minvalue);
}

//______________________________________________________________________________
void TBufferFile::ReadWithNbits(Float_t *ptr, Int_t nbits)
{
   // Read a Float16_t from the buffer when the number of bits is specified (explicitly or not)
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16().

   //we read the exponent and the truncated mantissa of the float
   //and rebuild the float.
   union {
      Float_t fFloatValue;
      Int_t   fIntValue;
   } temp;
   UChar_t  theExp;
   UShort_t theMan;
   frombuf(this->fBufCur,&theExp);
   frombuf(this->fBufCur,&theMan);
   temp.fIntValue = theExp;
   temp.fIntValue <<= 23;
   temp.fIntValue |= (theMan & ((1<<(nbits+1))-1)) <<(23-nbits);
   if(1<<(nbits+1) & theMan) temp.fFloatValue = -temp.fFloatValue;
   ptr[0] = temp.fFloatValue;
}

//______________________________________________________________________________
void TBufferFile::ReadWithFactor(Double_t *ptr, Double_t factor, Double_t minvalue)
{
   // Read a Double32_t from the buffer when the factor and minimun value have been specified
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().

   //a range was specified. We read an integer and convert it back to a double.
   UInt_t aint;
   frombuf(this->fBufCur,&aint);
   ptr[0] = (Double_t)(aint/factor + minvalue);
}

//______________________________________________________________________________
void TBufferFile::ReadWithNbits(Double_t *ptr, Int_t nbits)
{
   // Read a Double32_t from the buffer when the number of bits is specified (explicitly or not)
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().

   //we read the exponent and the truncated mantissa of the float
   //and rebuild the float.
   union {
      Float_t fFloatValue;
      Int_t   fIntValue;
   } temp;
   UChar_t  theExp;
   UShort_t theMan;
   frombuf(this->fBufCur,&theExp);
   frombuf(this->fBufCur,&theMan);
   temp.fIntValue = theExp;
   temp.fIntValue <<= 23;
   temp.fIntValue |= (theMan & ((1<<(nbits+1))-1)) <<(23-nbits);
   if(1<<(nbits+1) & theMan) temp.fFloatValue = -temp.fFloatValue;
   ptr[0] = (Double_t)temp.fFloatValue;
}

//______________________________________________________________________________
void TBufferFile::WriteFloat16(Float_t *f, TStreamerElement *ele)
{
   // write a Float16_t to the buffer.
   // The following cases are supported for streaming a Float16_t type
   // depending on the range declaration in the comment field of the data member:
   //  A-    Float16_t     fNormal;
   //  B-    Float16_t     fTemperature; //[0,100]
   //  C-    Float16_t     fCharge;      //[-1,1,2]
   //  D-    Float16_t     fVertex[3];   //[-30,30,10]
   //  E-    Float16_t     fChi2;        //[0,0,6]
   //  F-    Int_t          fNsp;
   //        Float16_t*    fPointValue;   //[fNsp][0,3]
   //
   // In case A fNormal is converted from a Float_t to a Float_t with mantissa truncated to 12 bits
   // In case B fTemperature is converted to a 32 bit unsigned integer
   // In case C fCharge is converted to a 2 bits unsigned integer
   // In case D the array elements of fVertex are converted to an unsigned 10 bits integer
   // In case E fChi2 is converted to a Float_t with truncated precision at 6 bits
   // In case F the fNsp elements of array fPointvalue are converted to an unsigned 32 bit integer
   //           Note that the range specifier must follow the dimension specifier.
   // the case B has more precision (9 to 10 significative digits than case A (6 to 7 digits).
   //
   // The range specifier has the general format: [xmin,xmax] or [xmin,xmax,nbits]
   //  [0,1]
   //  [-10,100];
   //  [-pi,pi], [-pi/2,pi/4],[-2pi,2*pi]
   //  [-10,100,16]
   //  [0,0,8]
   // if nbits is not specified, or nbits <2 or nbits>16 it is set to 16
   // if (xmin==0 and xmax==0 and nbits <=14) the float word will have
   // its mantissa truncated to nbits significative bits.
   //
   // IMPORTANT NOTE
   // --------------
   // --NOTE 1
   // Lets assume an original variable float x:
   // When using the format [0,0,8] (ie range not specified) you get the best
   // relative precision when storing and reading back the truncated x, say xt.
   // The variance of (x-xt)/x will be better than when specifying a range
   // for the same number of bits. However the precision relative to the
   // range (x-xt)/(xmax-xmin) will be worst, and vice-versa.
   // The format [0,0,8] is also interesting when the range of x is infinite
   // or unknown.
   //
   // --NOTE 2
   // It is important to understand the difference with the meaning of nbits
   //  -in case of [-1,1,nbits], nbits is the total number of bits used to make
   //    the conversion from a float to an integer
   //  -in case of [0,0,nbits], nbits is the number of bits used for the mantissa
   //
   //  see example of use of the Float16_t data type in tutorial double32.C
   //
   //Begin_Html
   /*
     <img src="gif/double32.gif">
   */
   //End_Html

   if (ele && ele->GetFactor() != 0) {
      //A range is specified. We normalize the double to the range and
      //convert it to an integer using a scaling factor that is a function of nbits.
      //see TStreamerElement::GetRange.
      Double_t x = f[0];
      Double_t xmin = ele->GetXmin();
      Double_t xmax = ele->GetXmax();
      if (x < xmin) x = xmin;
      if (x > xmax) x = xmax;
      UInt_t aint = UInt_t(0.5+ele->GetFactor()*(x-xmin)); *this << aint;
   } else {
      Int_t nbits = 0;
      //number of bits stored in fXmin (see TStreamerElement::GetRange)
      if (ele) nbits = (Int_t)ele->GetXmin();
      if (!nbits) nbits = 12;
      //a range is not specified, but nbits is.
      //In this case we truncate the mantissa to nbits and we stream
      //the exponent as a UChar_t and the mantissa as a UShort_t.
      union {
         Float_t fFloatValue;
         Int_t   fIntValue;
      };
      fFloatValue = f[0];
      UChar_t  theExp = (UChar_t)(0x000000ff & ((fIntValue<<1)>>24));
      UShort_t theMan = ((1<<(nbits+1))-1) & (fIntValue>>(23-nbits-1));
      theMan++;
      theMan = theMan>>1;
      if (theMan&1<<nbits) theMan = (1<<nbits) - 1;
      if (fFloatValue < 0) theMan |= 1<<(nbits+1);
      *this << theExp;
      *this << theMan;
   }
}

//______________________________________________________________________________
void TBufferFile::WriteDouble32(Double_t *d, TStreamerElement *ele)
{
   // write a Double32_t to the buffer.
   // The following cases are supported for streaming a Double32_t type
   // depending on the range declaration in the comment field of the data member:
   //  A-    Double32_t     fNormal;
   //  B-    Double32_t     fTemperature; //[0,100]
   //  C-    Double32_t     fCharge;      //[-1,1,2]
   //  D-    Double32_t     fVertex[3];   //[-30,30,10]
   //  E-    Double32_t     fChi2;        //[0,0,6]
   //  F-    Int_t          fNsp;
   //        Double32_t*    fPointValue;   //[fNsp][0,3]
   //
   // In case A fNormal is converted from a Double_t to a Float_t
   // In case B fTemperature is converted to a 32 bit unsigned integer
   // In case C fCharge is converted to a 2 bits unsigned integer
   // In case D the array elements of fVertex are converted to an unsigned 10 bits integer
   // In case E fChi2 is converted to a Float_t with mantissa truncated precision at 6 bits
   // In case F the fNsp elements of array fPointvalue are converted to an unsigned 32 bit integer
   //           Note that the range specifier must follow the dimension specifier.
   // the case B has more precision (9 to 10 significative digits than case A (6 to 7 digits).
   //
   // The range specifier has the general format: [xmin,xmax] or [xmin,xmax,nbits]
   //  [0,1]
   //  [-10,100];
   //  [-pi,pi], [-pi/2,pi/4],[-2pi,2*pi]
   //  [-10,100,16]
   //  [0,0,8]
   // if nbits is not specified, or nbits <2 or nbits>32 it is set to 32
   // if (xmin==0 and xmax==0 and nbits <=14) the double word will be converted
   // to a float and its mantissa truncated to nbits significative bits.
   //
   // IMPORTANT NOTEs
   // --------------
   // --NOTE 1
   // Lets assume an original variable double x:
   // When using the format [0,0,8] (ie range not specified) you get the best
   // relative precision when storing and reading back the truncated x, say xt.
   // The variance of (x-xt)/x will be better than when specifying a range
   // for the same number of bits. However the precision relative to the
   // range (x-xt)/(xmax-xmin) will be worst, and vice-versa.
   // The format [0,0,8] is also interesting when the range of x is infinite
   // or unknown.
   //
   // --NOTE 2
   // It is important to understand the difference with the meaning of nbits
   //  -in case of [-1,1,nbits], nbits is the total number of bits used to make
   //    the conversion from a double to an integer
   //  -in case of [0,0,nbits], nbits is the number of bits used for the mantissa
   //
   //  see example of use of the Double32_t data type in tutorial double32.C
   //
   //Begin_Html
   /*
     <img src="gif/double32.gif">
   */
   //End_Html

   if (ele && ele->GetFactor() != 0) {
      //A range is specified. We normalize the double to the range and
      //convert it to an integer using a scaling factor that is a function of nbits.
      //see TStreamerElement::GetRange.
      Double_t x = d[0];
      Double_t xmin = ele->GetXmin();
      Double_t xmax = ele->GetXmax();
      if (x < xmin) x = xmin;
      if (x > xmax) x = xmax;
      UInt_t aint = UInt_t(0.5+ele->GetFactor()*(x-xmin)); *this << aint;
   } else {
      Int_t nbits = 0;
      //number of bits stored in fXmin (see TStreamerElement::GetRange)
      if (ele) nbits = (Int_t)ele->GetXmin();
      if (!nbits) {
         //if no range and no bits specified, we convert from double to float
         Float_t afloat = (Float_t)d[0];
         *this << afloat;
      } else {
         //a range is not specified, but nbits is.
         //In this case we truncate the mantissa to nbits and we stream
         //the exponent as a UChar_t and the mantissa as a UShort_t.
         union {
            Float_t fFloatValue;
            Int_t   fIntValue;
         };
         fFloatValue = (Float_t)d[0];
         UChar_t  theExp = (UChar_t)(0x000000ff & ((fIntValue<<1)>>24));
         UShort_t theMan = ((1<<(nbits+1))-1) & (fIntValue>>(23-nbits-1)) ;
         theMan++;
         theMan = theMan>>1;
         if (theMan&1<<nbits) theMan = (1<<nbits)-1 ;
         if (fFloatValue < 0) theMan |= 1<<(nbits+1);
         *this << theExp;
         *this << theMan;
      }
   }
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArray(Bool_t *&b)
{
   // Read array of bools from the I/O buffer. Returns the number of
   // bools read. If argument is a 0 pointer then space will be
   // allocated for the array.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;

   if (n <= 0 || n > fBufSize) return 0;

   if (!b) b = new Bool_t[n];

   if (sizeof(Bool_t) > 1) {
      for (int i = 0; i < n; i++)
         frombuf(fBufCur, &b[i]);
   } else {
      Int_t l = sizeof(Bool_t)*n;
      memcpy(b, fBufCur, l);
      fBufCur += l;
   }

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArray(Char_t *&c)
{
   // Read array of characters from the I/O buffer. Returns the number of
   // characters read. If argument is a 0 pointer then space will be
   // allocated for the array.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Char_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!c) c = new Char_t[n];

   memcpy(c, fBufCur, l);
   fBufCur += l;

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArray(Short_t *&h)
{
   // Read array of shorts from the I/O buffer. Returns the number of shorts
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Short_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!h) h = new Short_t[n];

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(h, fBufCur, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &h[i]);
# endif
#else
   memcpy(h, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArray(Int_t *&ii)
{
   // Read array of ints from the I/O buffer. Returns the number of ints
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Int_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!ii) ii = new Int_t[n];

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(ii, fBufCur, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &ii[i]);
# endif
#else
   memcpy(ii, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArray(Long_t *&ll)
{
   // Read array of longs from the I/O buffer. Returns the number of longs
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Long_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!ll) ll = new Long_t[n];

   TFile *file = (TFile*)fParent;
   if (file && file->GetVersion() < 30006) {
      for (int i = 0; i < n; i++) frombufOld(fBufCur, &ll[i]);
   } else {
      for (int i = 0; i < n; i++) frombuf(fBufCur, &ll[i]);
   }
   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArray(Long64_t *&ll)
{
   // Read array of long longs from the I/O buffer. Returns the number of
   // long longs read. If argument is a 0 pointer then space will be
   // allocated for the array.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Long64_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!ll) ll = new Long64_t[n];

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &ll[i]);
#else
   memcpy(ll, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArray(Float_t *&f)
{
   // Read array of floats from the I/O buffer. Returns the number of floats
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Float_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!f) f = new Float_t[n];

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(f, fBufCur, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &f[i]);
# endif
#else
   memcpy(f, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArray(Double_t *&d)
{
   // Read array of doubles from the I/O buffer. Returns the number of doubles
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Double_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!d) d = new Double_t[n];

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &d[i]);
#else
   memcpy(d, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArrayFloat16(Float_t *&f, TStreamerElement *ele)
{
   // Read array of floats (written as truncated float) from the I/O buffer.
   // Returns the number of floats read.
   // If argument is a 0 pointer then space will be allocated for the array.
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;

   if (n <= 0 || 3*n > fBufSize) return 0;

   if (!f) f = new Float_t[n];

   ReadFastArrayFloat16(f,n,ele);

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadArrayDouble32(Double_t *&d, TStreamerElement *ele)
{
   // Read array of doubles (written as float) from the I/O buffer.
   // Returns the number of doubles read.
   // If argument is a 0 pointer then space will be allocated for the array.
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;

   if (n <= 0 || 3*n > fBufSize) return 0;

   if (!d) d = new Double_t[n];

   ReadFastArrayDouble32(d,n,ele);

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArray(Bool_t *b)
{
   // Read array of bools from the I/O buffer. Returns the number of bools
   // read.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;

   if (n <= 0 || n > fBufSize) return 0;

   if (!b) return 0;

   if (sizeof(Bool_t) > 1) {
      for (int i = 0; i < n; i++)
         frombuf(fBufCur, &b[i]);
   } else {
      Int_t l = sizeof(Bool_t)*n;
      memcpy(b, fBufCur, l);
      fBufCur += l;
   }

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArray(Char_t *c)
{
   // Read array of characters from the I/O buffer. Returns the number of
   // characters read.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Char_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!c) return 0;

   memcpy(c, fBufCur, l);
   fBufCur += l;

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArray(Short_t *h)
{
   // Read array of shorts from the I/O buffer. Returns the number of shorts
   // read.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Short_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!h) return 0;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(h, fBufCur, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &h[i]);
# endif
#else
   memcpy(h, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArray(Int_t *ii)
{
   // Read array of ints from the I/O buffer. Returns the number of ints
   // read.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Int_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!ii) return 0;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(ii, fBufCur, n);
   fBufCur += sizeof(Int_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &ii[i]);
# endif
#else
   memcpy(ii, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArray(Long_t *ll)
{
   // Read array of longs from the I/O buffer. Returns the number of longs
   // read.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Long_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!ll) return 0;

   TFile *file = (TFile*)fParent;
   if (file && file->GetVersion() < 30006) {
      for (int i = 0; i < n; i++) frombufOld(fBufCur, &ll[i]);
   } else {
      for (int i = 0; i < n; i++) frombuf(fBufCur, &ll[i]);
   }
   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArray(Long64_t *ll)
{
   // Read array of long longs from the I/O buffer. Returns the number of
   // long longs read.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Long64_t)*n;

   if (l <= 0 || l > fBufSize) return 0;

   if (!ll) return 0;

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &ll[i]);
#else
   memcpy(ll, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArray(Float_t *f)
{
   // Read array of floats from the I/O buffer. Returns the number of floats
   // read.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Float_t)*n;

   if (n <= 0 || l > fBufSize) return 0;

   if (!f) return 0;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(f, fBufCur, n);
   fBufCur += sizeof(Float_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &f[i]);
# endif
#else
   memcpy(f, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArray(Double_t *d)
{
   // Read array of doubles from the I/O buffer. Returns the number of doubles
   // read.

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;
   Int_t l = sizeof(Double_t)*n;

   if (n <= 0 || l > fBufSize) return 0;

   if (!d) return 0;

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &d[i]);
#else
   memcpy(d, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArrayFloat16(Float_t *f, TStreamerElement *ele)
{
   // Read array of floats (written as truncated float) from the I/O buffer.
   // Returns the number of floats read.
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;

   if (n <= 0 || 3*n > fBufSize) return 0;

   if (!f) return 0;

   ReadFastArrayFloat16(f,n,ele);

   return n;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadStaticArrayDouble32(Double_t *d, TStreamerElement *ele)
{
   // Read array of doubles (written as float) from the I/O buffer.
   // Returns the number of doubles read.
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32

   R__ASSERT(IsReading());

   Int_t n;
   *this >> n;

   if (n <= 0 || 3*n > fBufSize) return 0;

   if (!d) return 0;

   ReadFastArrayDouble32(d,n,ele);

   return n;
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(Bool_t *b, Int_t n)
{
   // Read array of n bools from the I/O buffer.

   if (n <= 0 || n > fBufSize) return;

   if (sizeof(Bool_t) > 1) {
      for (int i = 0; i < n; i++)
         frombuf(fBufCur, &b[i]);
   } else {
      Int_t l = sizeof(Bool_t)*n;
      memcpy(b, fBufCur, l);
      fBufCur += l;
   }
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(Char_t *c, Int_t n)
{
   // Read array of n characters from the I/O buffer.

   if (n <= 0 || n > fBufSize) return;

   Int_t l = sizeof(Char_t)*n;
   memcpy(c, fBufCur, l);
   fBufCur += l;
}

//______________________________________________________________________________
void TBufferFile::ReadFastArrayString(Char_t *c, Int_t n)
{
   // Read array of n characters from the I/O buffer.

   Int_t len;
   UChar_t lenchar;
   *this >> lenchar;
   if (lenchar < 255) {
      len = lenchar;
   } else {
      *this >> len;
   }
   if (len) {
      if (len <= 0 || len > fBufSize) return;
      Int_t blen = len;
      if (len >= n) len = n-1;

      Int_t l = sizeof(Char_t)*len;
      memcpy(c, fBufCur, l);
      fBufCur += blen;

      c[len] = 0;
   } else {
      c[0] = 0;
   }
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(Short_t *h, Int_t n)
{
   // Read array of n shorts from the I/O buffer.

   Int_t l = sizeof(Short_t)*n;
   if (n <= 0 || l > fBufSize) return;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(h, fBufCur, n);
   fBufCur += sizeof(Short_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &h[i]);
# endif
#else
   memcpy(h, fBufCur, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(Int_t *ii, Int_t n)
{
   // Read array of n ints from the I/O buffer.

   Int_t l = sizeof(Int_t)*n;
   if (l <= 0 || l > fBufSize) return;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(ii, fBufCur, n);
   fBufCur += sizeof(Int_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &ii[i]);
# endif
#else
   memcpy(ii, fBufCur, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(Long_t *ll, Int_t n)
{
   // Read array of n longs from the I/O buffer.

   Int_t l = sizeof(Long_t)*n;
   if (l <= 0 || l > fBufSize) return;

   TFile *file = (TFile*)fParent;
   if (file && file->GetVersion() < 30006) {
      for (int i = 0; i < n; i++) frombufOld(fBufCur, &ll[i]);
   } else {
      for (int i = 0; i < n; i++) frombuf(fBufCur, &ll[i]);
   }
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(Long64_t *ll, Int_t n)
{
   // Read array of n long longs from the I/O buffer.

   Int_t l = sizeof(Long64_t)*n;
   if (l <= 0 || l > fBufSize) return;

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &ll[i]);
#else
   memcpy(ll, fBufCur, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(Float_t *f, Int_t n)
{
   // Read array of n floats from the I/O buffer.

   Int_t l = sizeof(Float_t)*n;
   if (l <= 0 || l > fBufSize) return;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(f, fBufCur, n);
   fBufCur += sizeof(Float_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &f[i]);
# endif
#else
   memcpy(f, fBufCur, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(Double_t *d, Int_t n)
{
   // Read array of n doubles from the I/O buffer.

   Int_t l = sizeof(Double_t)*n;
   if (l <= 0 || l > fBufSize) return;

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &d[i]);
#else
   memcpy(d, fBufCur, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::ReadFastArrayFloat16(Float_t *f, Int_t n, TStreamerElement *ele)
{
   // Read array of n floats (written as truncated float) from the I/O buffer.
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16

   if (n <= 0 || 3*n > fBufSize) return;

   if (ele && ele->GetFactor() != 0) {
      //a range was specified. We read an integer and convert it back to a float
      Double_t xmin = ele->GetXmin();
      Double_t factor = ele->GetFactor();
      for (int j=0;j < n; j++) {
         UInt_t aint; *this >> aint; f[j] = (Float_t)(aint/factor + xmin);
      }
   } else {
      Int_t i;
      Int_t nbits = 0;
      if (ele) nbits = (Int_t)ele->GetXmin();
      if (!nbits) nbits = 12;
      //we read the exponent and the truncated mantissa of the float
      //and rebuild the new float.
      union {
         Float_t fFloatValue;
         Int_t   fIntValue;
      };
      UChar_t  theExp;
      UShort_t theMan;
      for (i = 0; i < n; i++) {
         *this >> theExp;
         *this >> theMan;
         fIntValue = theExp;
         fIntValue <<= 23;
         fIntValue |= (theMan & ((1<<(nbits+1))-1)) <<(23-nbits);
         if(1<<(nbits+1) & theMan) fFloatValue = -fFloatValue;
         f[i] = fFloatValue;
      }
   }
}

//______________________________________________________________________________
void TBufferFile::ReadFastArrayWithFactor(Float_t *ptr, Int_t n, Double_t factor, Double_t minvalue)
{
   // Read array of n floats (written as truncated float) from the I/O buffer.
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16

   if (n <= 0 || 3*n > fBufSize) return;

   //a range was specified. We read an integer and convert it back to a float
   for (int j=0;j < n; j++) {
      UInt_t aint; *this >> aint; ptr[j] = (Float_t)(aint/factor + minvalue);
   }
}

//______________________________________________________________________________
void TBufferFile::ReadFastArrayWithNbits(Float_t *ptr, Int_t n, Int_t nbits)
{
   // Read array of n floats (written as truncated float) from the I/O buffer.
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16

   if (n <= 0 || 3*n > fBufSize) return;

   if (!nbits) nbits = 12;
   //we read the exponent and the truncated mantissa of the float
   //and rebuild the new float.
   union {
      Float_t fFloatValue;
      Int_t   fIntValue;
   };
   UChar_t  theExp;
   UShort_t theMan;
   for (Int_t i = 0; i < n; i++) {
      *this >> theExp;
      *this >> theMan;
      fIntValue = theExp;
      fIntValue <<= 23;
      fIntValue |= (theMan & ((1<<(nbits+1))-1)) <<(23-nbits);
      if(1<<(nbits+1) & theMan) fFloatValue = -fFloatValue;
      ptr[i] = fFloatValue;
   }
}

//______________________________________________________________________________
void TBufferFile::ReadFastArrayDouble32(Double_t *d, Int_t n, TStreamerElement *ele)
{
   // Read array of n doubles (written as float) from the I/O buffer.
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32

   if (n <= 0 || 3*n > fBufSize) return;

   if (ele && ele->GetFactor() != 0) {
      //a range was specified. We read an integer and convert it back to a double.
      Double_t xmin = ele->GetXmin();
      Double_t factor = ele->GetFactor();
      for (int j=0;j < n; j++) {
         UInt_t aint; *this >> aint; d[j] = (Double_t)(aint/factor + xmin);
      }
   } else {
      Int_t i;
      Int_t nbits = 0;
      if (ele) nbits = (Int_t)ele->GetXmin();
      if (!nbits) {
         //we read a float and convert it to double
         Float_t afloat;
         for (i = 0; i < n; i++) {
            *this >> afloat;
            d[i] = (Double_t)afloat;
         }
      } else {
         //we read the exponent and the truncated mantissa of the float
         //and rebuild the double.
         union {
            Float_t fFloatValue;
            Int_t   fIntValue;
         };
         UChar_t  theExp;
         UShort_t theMan;
         for (i = 0; i < n; i++) {
            *this >> theExp;
            *this >> theMan;
            fIntValue = theExp;
            fIntValue <<= 23;
            fIntValue |= (theMan & ((1<<(nbits+1))-1)) <<(23-nbits);
            if (1<<(nbits+1) & theMan) fFloatValue = -fFloatValue;
            d[i] = (Double_t)fFloatValue;
         }
      }
   }
}

//______________________________________________________________________________
void TBufferFile::ReadFastArrayWithFactor(Double_t *d, Int_t n, Double_t factor, Double_t minvalue)
{
   // Read array of n doubles (written as float) from the I/O buffer.
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32

   if (n <= 0 || 3*n > fBufSize) return;

   //a range was specified. We read an integer and convert it back to a double.
   for (int j=0;j < n; j++) {
      UInt_t aint; *this >> aint; d[j] = (Double_t)(aint/factor + minvalue);
   }
}

//______________________________________________________________________________
void TBufferFile::ReadFastArrayWithNbits(Double_t *d, Int_t n, Int_t nbits)
{
   // Read array of n doubles (written as float) from the I/O buffer.
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32

   if (n <= 0 || 3*n > fBufSize) return;

   if (!nbits) {
      //we read a float and convert it to double
      Float_t afloat;
      for (Int_t i = 0; i < n; i++) {
         *this >> afloat;
         d[i] = (Double_t)afloat;
      }
   } else {
      //we read the exponent and the truncated mantissa of the float
      //and rebuild the double.
      union {
         Float_t fFloatValue;
         Int_t   fIntValue;
      };
      UChar_t  theExp;
      UShort_t theMan;
      for (Int_t i = 0; i < n; i++) {
         *this >> theExp;
         *this >> theMan;
         fIntValue = theExp;
         fIntValue <<= 23;
         fIntValue |= (theMan & ((1<<(nbits+1))-1)) <<(23-nbits);
         if (1<<(nbits+1) & theMan) fFloatValue = -fFloatValue;
         d[i] = (Double_t)fFloatValue;
      }
   }
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(void  *start, const TClass *cl, Int_t n,
                                TMemberStreamer *streamer, const TClass* onFileClass )
{
   // Read an array of 'n' objects from the I/O buffer.
   // Stores the objects read starting at the address 'start'.
   // The objects in the array are assume to be of class 'cl'.

   if (streamer) {
      streamer->SetOnFileClass(onFileClass);
      (*streamer)(*this,start,0);
      return;
   }

   int objectSize = cl->Size();
   char *obj = (char*)start;
   char *end = obj + n*objectSize;

   for(; obj<end; obj+=objectSize) ((TClass*)cl)->Streamer(obj,*this, onFileClass);
}

//______________________________________________________________________________
void TBufferFile::ReadFastArray(void **start, const TClass *cl, Int_t n,
                                Bool_t isPreAlloc, TMemberStreamer *streamer, const TClass* onFileClass)
{
   // Read an array of 'n' objects from the I/O buffer.
   // The objects read are stored starting at the address '*start'
   // The objects in the array are assumed to be of class 'cl' or a derived class.
   // 'mode' indicates whether the data member is marked with '->'

   // if isPreAlloc is true (data member has a ->) we can assume that the pointer (*start)
   // is never 0.

   if (streamer) {
      if (isPreAlloc) {
         for (Int_t j=0;j<n;j++) {
            if (!start[j]) start[j] = cl->New();
         }
      }
      streamer->SetOnFileClass(onFileClass);
      (*streamer)(*this,(void*)start,0);
      return;
   }

   if (!isPreAlloc) {

      for (Int_t j=0; j<n; j++){
         //delete the object or collection
         void *old = start[j];
         start[j] = ReadObjectAny(cl);
         if (old && old!=start[j] &&
             TStreamerInfo::CanDelete()
             // There are some cases where the user may set up a pointer in the (default)
             // constructor but not mark this pointer as transient.  Sometime the value
             // of this pointer is the address of one of the object with just created
             // and the following delete would result in the deletion (possibly of the
             // top level object we are goint to return!).
             // Eventhough this is a user error, we could prevent the crash by simply
             // adding:
             // && !CheckObject(start[j],cl)
             // However this can increase the read time significantly (10% in the case
             // of one TLine pointer in the test/Track and run ./Event 200 0 0 20 30000
             //
             // If ReadObjectAny returned the same value as we previous had, this means
             // that when writing this object (start[j] had already been written and
             // is indeed pointing to the same object as the object the user set up
             // in the default constructor).
             ) {
            ((TClass*)cl)->Destructor(old,kFALSE); // call delete and desctructor
         }
      }

   } else {
      //case //-> in comment

      for (Int_t j=0; j<n; j++){
         if (!start[j]) start[j] = ((TClass*)cl)->New();
         ((TClass*)cl)->Streamer(start[j],*this,onFileClass);
      }

   }
}

//______________________________________________________________________________
void TBufferFile::WriteArray(const Bool_t *b, Int_t n)
{
   // Write array of n bools into the I/O buffer.

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(b);

   Int_t l = sizeof(UChar_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   if (sizeof(Bool_t) > 1) {
      for (int i = 0; i < n; i++)
         tobuf(fBufCur, b[i]);
   } else {
      memcpy(fBufCur, b, l);
      fBufCur += l;
   }
}

//______________________________________________________________________________
void TBufferFile::WriteArray(const Char_t *c, Int_t n)
{
   // Write array of n characters into the I/O buffer.

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(c);

   Int_t l = sizeof(Char_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   memcpy(fBufCur, c, l);
   fBufCur += l;
}

//______________________________________________________________________________
void TBufferFile::WriteArray(const Short_t *h, Int_t n)
{
   // Write array of n shorts into the I/O buffer.

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(h);

   Int_t l = sizeof(Short_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(fBufCur, h, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, h[i]);
# endif
#else
   memcpy(fBufCur, h, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteArray(const Int_t *ii, Int_t n)
{
   // Write array of n ints into the I/O buffer.

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(ii);

   Int_t l = sizeof(Int_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(fBufCur, ii, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, ii[i]);
# endif
#else
   memcpy(fBufCur, ii, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteArray(const Long_t *ll, Int_t n)
{
   // Write array of n longs into the I/O buffer.

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(ll);

   Int_t l = 8*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);
   for (int i = 0; i < n; i++) tobuf(fBufCur, ll[i]);
}

//______________________________________________________________________________
void TBufferFile::WriteArray(const ULong_t *ll, Int_t n)
{
   // Write array of n unsigned longs into the I/O buffer.
   // This is an explicit case for unsigned longs since signed longs
   // have a special tobuf().

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(ll);

   Int_t l = 8*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);
   for (int i = 0; i < n; i++) tobuf(fBufCur, ll[i]);
}

//______________________________________________________________________________
void TBufferFile::WriteArray(const Long64_t *ll, Int_t n)
{
   // Write array of n long longs into the I/O buffer.

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(ll);

   Int_t l = sizeof(Long64_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, ll[i]);
#else
   memcpy(fBufCur, ll, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteArray(const Float_t *f, Int_t n)
{
   // Write array of n floats into the I/O buffer.

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(f);

   Int_t l = sizeof(Float_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(fBufCur, f, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, f[i]);
# endif
#else
   memcpy(fBufCur, f, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteArray(const Double_t *d, Int_t n)
{
   // Write array of n doubles into the I/O buffer.

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(d);

   Int_t l = sizeof(Double_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, d[i]);
#else
   memcpy(fBufCur, d, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteArrayFloat16(const Float_t *f, Int_t n, TStreamerElement *ele)
{
   // Write array of n floats (as truncated float) into the I/O buffer.
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(f);

   Int_t l = sizeof(Float_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   WriteFastArrayFloat16(f,n,ele);
}

//______________________________________________________________________________
void TBufferFile::WriteArrayDouble32(const Double_t *d, Int_t n, TStreamerElement *ele)
{
   // Write array of n doubles (as float) into the I/O buffer.
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32

   R__ASSERT(IsWriting());

   *this << n;

   if (n <= 0) return;

   R__ASSERT(d);

   Int_t l = sizeof(Float_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   WriteFastArrayDouble32(d,n,ele);
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(const Bool_t *b, Int_t n)
{
   // Write array of n bools into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(UChar_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   if (sizeof(Bool_t) > 1) {
      for (int i = 0; i < n; i++)
         tobuf(fBufCur, b[i]);
   } else {
      memcpy(fBufCur, b, l);
      fBufCur += l;
   }
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(const Char_t *c, Int_t n)
{
   // Write array of n characters into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Char_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   memcpy(fBufCur, c, l);
   fBufCur += l;
}

//______________________________________________________________________________
void TBufferFile::WriteFastArrayString(const Char_t *c, Int_t n)
{
   // Write array of n characters into the I/O buffer.

   if (n < 255) {
      *this << (UChar_t)n;
   } else {
      *this << (UChar_t)255;
      *this << n;
   }

   if (n <= 0) return;

   Int_t l = sizeof(Char_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   memcpy(fBufCur, c, l);
   fBufCur += l;
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(const Short_t *h, Int_t n)
{
   // Write array of n shorts into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Short_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(fBufCur, h, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, h[i]);
# endif
#else
   memcpy(fBufCur, h, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(const Int_t *ii, Int_t n)
{
   // Write array of n ints into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Int_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(fBufCur, ii, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, ii[i]);
# endif
#else
   memcpy(fBufCur, ii, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(const Long_t *ll, Int_t n)
{
   // Write array of n longs into the I/O buffer.

   if (n <= 0) return;

   Int_t l = 8*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   for (int i = 0; i < n; i++) tobuf(fBufCur, ll[i]);
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(const ULong_t *ll, Int_t n)
{
   // Write array of n unsigned longs into the I/O buffer.
   // This is an explicit case for unsigned longs since signed longs
   // have a special tobuf().

   if (n <= 0) return;

   Int_t l = 8*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   for (int i = 0; i < n; i++) tobuf(fBufCur, ll[i]);
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(const Long64_t *ll, Int_t n)
{
   // Write array of n long longs into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Long64_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, ll[i]);
#else
   memcpy(fBufCur, ll, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(const Float_t *f, Int_t n)
{
   // Write array of n floats into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Float_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(fBufCur, f, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, f[i]);
# endif
#else
   memcpy(fBufCur, f, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(const Double_t *d, Int_t n)
{
   // Write array of n doubles into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Double_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, d[i]);
#else
   memcpy(fBufCur, d, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBufferFile::WriteFastArrayFloat16(const Float_t *f, Int_t n, TStreamerElement *ele)
{
   // Write array of n floats (as truncated float) into the I/O buffer.
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16

   if (n <= 0) return;

   Int_t l = sizeof(Float_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   if (ele && ele->GetFactor()) {
      //A range is specified. We normalize the float to the range and
      //convert it to an integer using a scaling factor that is a function of nbits.
      //see TStreamerElement::GetRange.
      Double_t factor = ele->GetFactor();
      Double_t xmin = ele->GetXmin();
      Double_t xmax = ele->GetXmax();
      for (int j = 0; j < n; j++) {
         Float_t x = f[j];
         if (x < xmin) x = xmin;
         if (x > xmax) x = xmax;
         UInt_t aint = UInt_t(0.5+factor*(x-xmin)); *this << aint;
      }
   } else {
      Int_t nbits = 0;
      //number of bits stored in fXmin (see TStreamerElement::GetRange)
      if (ele) nbits = (Int_t)ele->GetXmin();
      if (!nbits) nbits = 12;
      Int_t i;
      //a range is not specified, but nbits is.
      //In this case we truncate the mantissa to nbits and we stream
      //the exponent as a UChar_t and the mantissa as a UShort_t.
      union {
         Float_t fFloatValue;
         Int_t   fIntValue;
      };
      for (i = 0; i < n; i++) {
         fFloatValue = f[i];
         UChar_t  theExp = (UChar_t)(0x000000ff & ((fIntValue<<1)>>24));
         UShort_t theMan = ((1<<(nbits+1))-1) & (fIntValue>>(23-nbits-1));
         theMan++;
         theMan = theMan>>1;
         if (theMan&1<<nbits) theMan = (1<<nbits) - 1;
         if (fFloatValue < 0) theMan |= 1<<(nbits+1);
         *this << theExp;
         *this << theMan;
      }
   }
}

//______________________________________________________________________________
void TBufferFile::WriteFastArrayDouble32(const Double_t *d, Int_t n, TStreamerElement *ele)
{
   // Write array of n doubles (as float) into the I/O buffer.
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32

   if (n <= 0) return;

   Int_t l = sizeof(Float_t)*n;
   if (fBufCur + l > fBufMax) AutoExpand(fBufSize+l);

   if (ele && ele->GetFactor()) {
      //A range is specified. We normalize the double to the range and
      //convert it to an integer using a scaling factor that is a function of nbits.
      //see TStreamerElement::GetRange.
      Double_t factor = ele->GetFactor();
      Double_t xmin = ele->GetXmin();
      Double_t xmax = ele->GetXmax();
      for (int j = 0; j < n; j++) {
         Double_t x = d[j];
         if (x < xmin) x = xmin;
         if (x > xmax) x = xmax;
         UInt_t aint = UInt_t(0.5+factor*(x-xmin)); *this << aint;
      }
   } else {
      Int_t nbits = 0;
      //number of bits stored in fXmin (see TStreamerElement::GetRange)
      if (ele) nbits = (Int_t)ele->GetXmin();
      Int_t i;
      if (!nbits) {
         //if no range and no bits specified, we convert from double to float
         for (i = 0; i < n; i++) {
            Float_t afloat = (Float_t)d[i];
            *this << afloat;
         }
      } else {
         //a range is not specified, but nbits is.
         //In this case we truncate the mantissa to nbits and we stream
         //the exponent as a UChar_t and the mantissa as a UShort_t.
         union {
            Float_t fFloatValue;
            Int_t   fIntValue;
         };
         for (i = 0; i < n; i++) {
            fFloatValue = (Float_t)d[i];
            UChar_t  theExp = (UChar_t)(0x000000ff & ((fIntValue<<1)>>24));
            UShort_t theMan = ((1<<(nbits+1))-1) & (fIntValue>>(23-nbits-1));
            theMan++;
            theMan = theMan>>1;
            if(theMan&1<<nbits) theMan = (1<<nbits) - 1;
            if (fFloatValue < 0) theMan |= 1<<(nbits+1);
            *this << theExp;
            *this << theMan;
         }
      }
   }
}

//______________________________________________________________________________
void TBufferFile::WriteFastArray(void  *start, const TClass *cl, Int_t n,
                                 TMemberStreamer *streamer)
{
   // Write an array of object starting at the address 'start' and of length 'n'
   // the objects in the array are assumed to be of class 'cl'

   if (streamer) {
      (*streamer)(*this, start, 0);
      return;
   }

   char *obj = (char*)start;
   if (!n) n=1;
   int size = cl->Size();

   for(Int_t j=0; j<n; j++,obj+=size) {
      ((TClass*)cl)->Streamer(obj,*this);
   }
}

//______________________________________________________________________________
Int_t TBufferFile::WriteFastArray(void **start, const TClass *cl, Int_t n,
                                  Bool_t isPreAlloc, TMemberStreamer *streamer)
{
   // Write an array of object starting at the address '*start' and of length 'n'
   // the objects in the array are of class 'cl'
   // 'isPreAlloc' indicates whether the data member is marked with '->'
   // Return:
   //  0: success
   //  2: truncated success (i.e actual class is missing. Only ptrClass saved.)

   // if isPreAlloc is true (data member has a ->) we can assume that the pointer
   // is never 0.

   if (streamer) {
      (*streamer)(*this,(void*)start,0);
      return 0;
   }

   int strInfo = 0;

   Int_t res = 0;

   if (!isPreAlloc) {

      for (Int_t j=0;j<n;j++) {
         //must write StreamerInfo if pointer is null
         if (!strInfo && !start[j]) {
            if (cl->Property() & kIsAbstract) {
               // Do not try to generate the StreamerInfo for an abstract class
            } else {
               TStreamerInfo *info = (TStreamerInfo*)((TClass*)cl)->GetStreamerInfo();
               ForceWriteInfo(info,kFALSE);
            }
         }
         strInfo = 2003;
         res |= WriteObjectAny(start[j],cl);
      }

   } else {
      //case //-> in comment

      for (Int_t j=0;j<n;j++) {
         if (!start[j]) start[j] = ((TClass*)cl)->New();
         ((TClass*)cl)->Streamer(start[j],*this);
      }

   }
   return res;
}

//______________________________________________________________________________
TObject *TBufferFile::ReadObject(const TClass * /*clReq*/)
{
   // Read object from I/O buffer. clReq is NOT used.
   // The value returned is the address of the actual start in memory of
   // the object. Note that if the actual class of the object does not
   // inherit first from TObject, the type of the pointer is NOT 'TObject*'.
   // [More accurately, the class needs to start with the TObject part, for
   // the pointer to be a real TObject*].
   // We recommend using ReadObjectAny instead of ReadObject

   return (TObject*) ReadObjectAny(0);
}

//______________________________________________________________________________
void TBufferFile::SkipObjectAny()
{
   // Skip any kind of object from buffer

   UInt_t start, count;
   ReadVersion(&start, &count);
   SetBufferOffset(start+count+sizeof(UInt_t));
}

//______________________________________________________________________________
void *TBufferFile::ReadObjectAny(const TClass *clCast)
{
   // Read object from I/O buffer.
   // A typical use for this function is:
   //    MyClass *ptr = (MyClass*)b.ReadObjectAny(MyClass::Class());
   // I.e. clCast should point to a TClass object describing the class pointed
   // to by your pointer.
   // In case of multiple inheritance, the return value might not be the
   // real beginning of the object in memory.  You will need to use a
   // dynamic_cast later if you need to retrieve it.

   R__ASSERT(IsReading());

   // make sure fMap is initialized
   InitMap();

   // before reading object save start position
   UInt_t startpos = UInt_t(fBufCur-fBuffer);

   // attempt to load next object as TClass clCast
   UInt_t tag;       // either tag or byte count
   TClass *clRef = ReadClass(clCast, &tag);
   TClass *clOnfile = 0;
   Int_t baseOffset = 0;
   if (clRef && (clRef!=(TClass*)(-1)) && clCast) {
      //baseOffset will be -1 if clRef does not inherit from clCast.
      baseOffset = clRef->GetBaseClassOffset(clCast);
      if (baseOffset == -1) {
         // The 2 classes are unrelated, maybe there is a converter between the 2.

         if (!clCast->GetSchemaRules() ||
             !clCast->GetSchemaRules()->HasRuleWithSourceClass(clRef->GetName()))
         {
            // There is no converter
            Error("ReadObject", "got object of wrong class! requested %s but got %s",
                  clCast->GetName(), clRef->GetName());

            CheckByteCount(startpos, tag, (TClass*)0); // avoid mis-leading byte count error message
            return 0; // We better return at this point
         }
         baseOffset = 0; // For now we do not support requesting from a class that is the base of one of the class for which there is transformation to ....

         Info("ReadObjectAny","Using Converter StreamerInfo from %s to %s",clRef->GetName(),clCast->GetName());
         clRef = const_cast<TClass*>(clCast);

      }
      if (clCast->GetClassInfo() && !clRef->GetClassInfo()) {
         //we cannot mix a compiled class with an emulated class in the inheritance
         Error("ReadObject", "trying to read an emulated class (%s) to store in a compiled pointer (%s)",
               clRef->GetName(),clCast->GetName());
         CheckByteCount(startpos, tag, (TClass*)0); // avoid mis-leading byte count error message
         return 0;
      }
   }

   // check if object has not already been read
   // (this can only happen when called via CheckObject())
   char *obj;
   if (fVersion > 0) {
      obj = (char *) (Long_t)fMap->GetValue(startpos+kMapOffset);
      if (obj == (void*) -1) obj = 0;
      if (obj) {
         CheckByteCount(startpos, tag, (TClass*)0);
         return (obj+baseOffset);
      }
   }

   // unknown class, skip to next object and return 0 obj
   if (clRef == (TClass*) -1) {
      if (fBufCur >= fBufMax) return 0;
      if (fVersion > 0)
         MapObject((TObject*) -1, startpos+kMapOffset);
      else
         MapObject((void*)0, 0, fMapCount);
      CheckByteCount(startpos, tag, (TClass*)0);
      return 0;
   }

   if (!clRef) {

      // got a reference to an already read object
      if (fVersion > 0) {
         tag += fDisplacement;
         tag = CheckObject(tag, clCast);
      } else {
         if (tag > (UInt_t)fMap->GetSize()) {
            Error("ReadObject", "object tag too large, I/O buffer corrupted");
            return 0;
            // exception
         }
      }
      obj = (char *) (Long_t)fMap->GetValue(tag);
      clRef = (TClass*) (Long_t)fClassMap->GetValue(tag);

      if (clRef && (clRef!=(TClass*)(-1)) && clCast) {
         //baseOffset will be -1 if clRef does not inherit from clCast.
         baseOffset = clRef->GetBaseClassOffset(clCast);
         if (baseOffset == -1) {
            Error("ReadObject", "Got object of wrong class (Got %s while expecting %s)",
                  clRef->GetName(),clCast->GetName());
            // exception
            baseOffset = 0;
         }
      }

      // There used to be a warning printed here when:
      //   obj && isTObject && !((TObject*)obj)->IsA()->InheritsFrom(clReq)
      // however isTObject was based on clReq (now clCast).
      // If the test was to fail, then it is as likely that the object is not a TObject
      // and then we have a potential core dump.
      // At this point (missing clRef), we do NOT have enough information to really
      // answer the question: is the object read of the type I requested.

   } else {

      // allocate a new object based on the class found
      obj = (char*)clRef->New();
      if (!obj) {
         Error("ReadObject", "could not create object of class %s",
               clRef->GetName());
         // exception
         return 0;
      }

      // add to fMap before reading rest of object
      if (fVersion > 0)
         MapObject(obj, clRef, startpos+kMapOffset);
      else
         MapObject(obj, clRef, fMapCount);

      // let the object read itself
      clRef->Streamer( obj, *this, clOnfile );

      CheckByteCount(startpos, tag, clRef);
   }

   return obj+baseOffset;
}

//______________________________________________________________________________
void TBufferFile::WriteObject(const TObject *obj)
{
   // Write object to I/O buffer.

   WriteObjectAny(obj, TObject::Class());
}

//______________________________________________________________________________
void TBufferFile::WriteObjectClass(const void *actualObjectStart, const TClass *actualClass)
{
   // Write object to I/O buffer.
   // This function assumes that the value of 'actualObjectStart' is the actual start of
   // the object of class 'actualClass'

   R__ASSERT(IsWriting());

   if (!actualObjectStart) {

      // save kNullTag to represent NULL pointer
      *this << kNullTag;

   } else {

      // make sure fMap is initialized
      InitMap();

      ULong_t idx;
      UInt_t slot;
      ULong_t hash = Void_Hash(actualObjectStart);

      if ((idx = (ULong_t)fMap->GetValue(hash, (Long_t)actualObjectStart, slot)) != 0) {

         // truncation is OK the value we did put in the map is an 30-bit offset
         // and not a pointer
         UInt_t objIdx = UInt_t(idx);

         // save index of already stored object
         *this << objIdx;

      } else {

         // A warning to let the user know it will need to change the class code
         // to  be able to read this back.
         if (actualClass->HasDefaultConstructor() == 0) {
            Warning("WriteObjectAny", "since %s has no public constructor\n"
               "\twhich can be called without argument, objects of this class\n"
               "\tcan not be read with the current library. You will need to\n"
               "\tadd a default constructor before attempting to read it.",
               actualClass->GetName());
         }

         // reserve space for leading byte count
         UInt_t cntpos = UInt_t(fBufCur-fBuffer);
         fBufCur += sizeof(UInt_t);

         // write class of object first
         Int_t mapsize = fMap->Capacity(); // The slot depends on the capacity and WriteClass might induce an increase.
         WriteClass(actualClass);

         // add to map before writing rest of object (to handle self reference)
         // (+kMapOffset so it's != kNullTag)
         //MapObject(actualObjectStart, actualClass, cntpos+kMapOffset);
         UInt_t offset = cntpos+kMapOffset;
         if (mapsize == fMap->Capacity()) {
            fMap->AddAt(slot, hash, (Long_t)actualObjectStart, offset);
         } else {
            // The slot depends on the capacity and WriteClass has induced an increase.
            fMap->Add(hash, (Long_t)actualObjectStart, offset);
         }
         // No need to keep track of the class in write mode
         // fClassMap->Add(hash, (Long_t)obj, (Long_t)((TObject*)obj)->IsA());
         fMapCount++;

         ((TClass*)actualClass)->Streamer((void*)actualObjectStart,*this);

         // write byte count
         SetByteCount(cntpos);
      }
   }
}

namespace {
   struct DynamicType {
      // Helper class to enable typeid on any address
      // Used in code similar to:
      //    typeid( * (DynamicType*) void_ptr );
      virtual ~DynamicType() {}
   };
}

//______________________________________________________________________________
Int_t TBufferFile::WriteObjectAny(const void *obj, const TClass *ptrClass)
{
   // Write object to I/O buffer.
   // This function assumes that the value in 'obj' is the value stored in
   // a pointer to a "ptrClass". The actual type of the object pointed to
   // can be any class derived from "ptrClass".
   // Return:
   //  0: failure
   //  1: success
   //  2: truncated success (i.e actual class is missing. Only ptrClass saved.)

   if (!obj) {
      WriteObjectClass(0, 0);
      return 1;
   }

   if (!ptrClass) {
      Error("WriteObjectAny", "ptrClass argument may not be 0");
      return 0;
   }

   TClass *clActual = ptrClass->GetActualClass(obj);

   if (clActual==0) {
      // The ptrClass is a class with a virtual table and we have no
      // TClass with the actual type_info in memory.

      DynamicType* d_ptr = (DynamicType*)obj;
      Warning("WriteObjectAny",
              "An object of type %s (from type_info) passed through a %s pointer was truncated (due a missing dictionary)!!!",
              typeid(*d_ptr).name(),ptrClass->GetName());
      WriteObjectClass(obj, ptrClass);
      return 2;
   } else if (clActual && (clActual != ptrClass)) {
      const char *temp = (const char*) obj;
      temp -= clActual->GetBaseClassOffset(ptrClass);
      WriteObjectClass(temp, clActual);
      return 1;
   } else {
      WriteObjectClass(obj, ptrClass);
      return 1;
   }
}

//______________________________________________________________________________
TClass *TBufferFile::ReadClass(const TClass *clReq, UInt_t *objTag)
{
   // Read class definition from I/O buffer. clReq can be used to cross check
   // if the actually read object is of the requested class. objTag is
   // set in case the object is a reference to an already read object.

   R__ASSERT(IsReading());

   // read byte count and/or tag (older files don't have byte count)
   TClass *cl;
   if (fBufCur < fBuffer || fBufCur > fBufMax) {
      fBufCur = fBufMax;
      cl = (TClass*)-1;
      return cl;
   }
   UInt_t bcnt, tag, startpos = 0;
   *this >> bcnt;
   if (!(bcnt & kByteCountMask) || bcnt == kNewClassTag) {
      tag  = bcnt;
      bcnt = 0;
   } else {
      fVersion = 1;
      startpos = UInt_t(fBufCur-fBuffer);
      *this >> tag;
   }

   // in case tag is object tag return tag
   if (!(tag & kClassMask)) {
      if (objTag) *objTag = tag;
      return 0;
   }

   if (tag == kNewClassTag) {

      // got a new class description followed by a new object
      // (class can be 0 if class dictionary is not found, in that
      // case object of this class must be skipped)
      cl = TClass::Load(*this);

      // add class to fMap for later reference
      if (fVersion > 0) {
         // check if class was already read
         TClass *cl1 = (TClass *)(Long_t)fMap->GetValue(startpos+kMapOffset);
         if (cl1 != cl)
            MapObject(cl ? cl : (TObject*) -1, startpos+kMapOffset);
      } else
         MapObject(cl, fMapCount);

   } else {

      // got a tag to an already seen class
      UInt_t clTag = (tag & ~kClassMask);

      if (fVersion > 0) {
         clTag += fDisplacement;
         clTag = CheckObject(clTag, clReq, kTRUE);
      } else {
         if (clTag == 0 || clTag > (UInt_t)fMap->GetSize()) {
            Error("ReadClass", "illegal class tag=%d (0<tag<=%d), I/O buffer corrupted",
                  clTag, fMap->GetSize());
            // exception
         }
      }

      // class can be 0 if dictionary was not found
      cl = (TClass *)(Long_t)fMap->GetValue(clTag);
   }

   if (cl && clReq &&
       (!cl->InheritsFrom(clReq) &&
        !(clReq->GetSchemaRules() &&
          clReq->GetSchemaRules()->HasRuleWithSourceClass(cl->GetName()) )
        ) ) {
      Error("ReadClass", "got wrong class: %s", cl->GetName());
      // exception
   }

   // return bytecount in objTag
   if (objTag) *objTag = (bcnt & ~kByteCountMask);

   // case of unknown class
   if (!cl) cl = (TClass*)-1;

   return cl;
}

//______________________________________________________________________________
void TBufferFile::WriteClass(const TClass *cl)
{
   // Write class description to I/O buffer.

   R__ASSERT(IsWriting());

   ULong_t idx;
   ULong_t hash = Void_Hash(cl);
   UInt_t slot;

   if ((idx = (ULong_t)fMap->GetValue(hash, (Long_t)cl,slot)) != 0) {

      // truncation is OK the value we did put in the map is an 30-bit offset
      // and not a pointer
      UInt_t clIdx = UInt_t(idx);

      // save index of already stored class
      *this << (clIdx | kClassMask);

   } else {

      // offset in buffer where class info is written
      UInt_t offset = UInt_t(fBufCur-fBuffer);

      // save new class tag
      *this << kNewClassTag;

      // write class name
      cl->Store(*this);

      // store new class reference in fMap (+kMapOffset so it's != kNullTag)
      CheckCount(offset+kMapOffset);
      fMap->AddAt(slot, hash, (Long_t)cl, offset+kMapOffset);
      fMapCount++;
   }
}

//______________________________________________________________________________
void TBufferFile::SkipVersion(const TClass *cl)
{
   // Skip class version from I/O buffer.

   Version_t version;

   // not interested in byte count
   frombuf(this->fBufCur,&version);

   // if this is a byte count, then skip next short and read version
   if (version & kByteCountVMask) {
      frombuf(this->fBufCur,&version);
      frombuf(this->fBufCur,&version);
   }

   if (cl && cl->GetClassVersion() != 0  && version<=1) {
      if (version <= 0)  {
         UInt_t checksum = 0;
         //*this >> checksum;
         frombuf(this->fBufCur,&checksum);
         TStreamerInfo *vinfo = (TStreamerInfo*)cl->FindStreamerInfo(checksum);
         if (vinfo) {
            return;
         } else {
            // There are some cases (for example when the buffer was stored outside of
            // a ROOT file) where we do not have a TStreamerInfo.  If the checksum is
            // the one from the current class, we can still assume that we can read
            // the data so let use it.
            if (checksum==cl->GetCheckSum() || cl->MatchLegacyCheckSum(checksum)) {
               version = cl->GetClassVersion();
            } else {
               if (fParent) {
                  Error("ReadVersion", "Could not find the StreamerInfo with a checksum of %d for the class \"%s\" in %s.",
                        checksum, cl->GetName(), ((TFile*)fParent)->GetName());
               } else {
                  Error("ReadVersion", "Could not find the StreamerInfo with a checksum of %d for the class \"%s\" (buffer with no parent)",
                        checksum, cl->GetName());
               }
               return;
            }
         }
      }  else if (version == 1 && fParent && ((TFile*)fParent)->GetVersion()<40000 ) {
         // We could have a file created using a Foreign class before
         // the introduction of the CheckSum.  We need to check
         if ((!cl->IsLoaded() || cl->IsForeign()) &&
             Class_Has_StreamerInfo(cl) ) {

            const TList *list = ((TFile*)fParent)->GetStreamerInfoCache();
            const TStreamerInfo *local = list ? (TStreamerInfo*)list->FindObject(cl->GetName()) : 0;
            if ( local )  {
               UInt_t checksum = local->GetCheckSum();
               TStreamerInfo *vinfo = (TStreamerInfo*)cl->FindStreamerInfo(checksum);
               if (vinfo) {
                  version = vinfo->GetClassVersion();
               } else {
                  Error("ReadVersion", "Could not find the StreamerInfo with a checksum of %d for the class \"%s\" in %s.",
                        checksum, cl->GetName(), ((TFile*)fParent)->GetName());
                  return;
               }
            }
            else  {
               Error("ReadVersion", "Class %s not known to file %s.",
                     cl->GetName(), ((TFile*)fParent)->GetName());
               version = 0;
            }
         }
      }
   }
}

//______________________________________________________________________________
Version_t TBufferFile::ReadVersion(UInt_t *startpos, UInt_t *bcnt, const TClass *cl)
{
   // Read class version from I/O buffer.

   Version_t version;

   if (startpos) {
      // before reading object save start position
      *startpos = UInt_t(fBufCur-fBuffer);
   }

   // read byte count (older files don't have byte count)
   // byte count is packed in two individual shorts, this to be
   // backward compatible with old files that have at this location
   // only a single short (i.e. the version)
   union {
      UInt_t     cnt;
      Version_t  vers[2];
   } v;
#ifdef R__BYTESWAP
   frombuf(this->fBufCur,&v.vers[1]);
   frombuf(this->fBufCur,&v.vers[0]);
#else
   frombuf(this->fBufCur,&v.vers[0]);
   frombuf(this->fBufCur,&v.vers[1]);
#endif

   // no bytecount, backup and read version
   if (!(v.cnt & kByteCountMask)) {
      fBufCur -= sizeof(UInt_t);
      v.cnt = 0;
   }
   if (bcnt) *bcnt = (v.cnt & ~kByteCountMask);
   frombuf(this->fBufCur,&version);

   if (version<=1) {
      if (version <= 0)  {
         if (cl) {
            if (cl->GetClassVersion() != 0
                // If v.cnt < 6 then we have a class with a version that used to be zero and so there is no checksum.
                && (v.cnt && v.cnt >= 6)
                ) {
               UInt_t checksum = 0;
               //*this >> checksum;
               frombuf(this->fBufCur,&checksum);
               TStreamerInfo *vinfo = (TStreamerInfo*)cl->FindStreamerInfo(checksum);
               if (vinfo) {
                  return vinfo->TStreamerInfo::GetClassVersion(); // Try to get inlining.
               } else {
                  // There are some cases (for example when the buffer was stored outside of
                  // a ROOT file) where we do not have a TStreamerInfo.  If the checksum is
                  // the one from the current class, we can still assume that we can read
                  // the data so let use it.
                  if (checksum==cl->GetCheckSum() || cl->MatchLegacyCheckSum(checksum)) {
                     version = cl->GetClassVersion();
                  } else {
                     if (fParent) {
                        Error("ReadVersion", "Could not find the StreamerInfo with a checksum of 0x%x for the class \"%s\" in %s.",
                              checksum, cl->GetName(), ((TFile*)fParent)->GetName());
                     } else {
                        Error("ReadVersion", "Could not find the StreamerInfo with a checksum of 0x%x for the class \"%s\" (buffer with no parent)",
                              checksum, cl->GetName());
                     }
                     return 0;
                  }
               }
            }
         } else { // of if (cl) {
            UInt_t checksum = 0;
            //*this >> checksum;
            // If *bcnt < 6 then we have a class with 'just' version zero and no checksum
            if (v.cnt && v.cnt >= 6)
               frombuf(this->fBufCur,&checksum);
         }
      }  else if (version == 1 && fParent && ((TFile*)fParent)->GetVersion()<40000 && cl && cl->GetClassVersion() != 0) {
         // We could have a file created using a Foreign class before
         // the introduction of the CheckSum.  We need to check
         if ((!cl->IsLoaded() || cl->IsForeign()) &&
             Class_Has_StreamerInfo(cl) ) {

            const TList *list = ((TFile*)fParent)->GetStreamerInfoCache();
            const TStreamerInfo *local = list ? (TStreamerInfo*)list->FindObject(cl->GetName()) : 0;
            if ( local )  {
               UInt_t checksum = local->GetCheckSum();
               TStreamerInfo *vinfo = (TStreamerInfo*)cl->FindStreamerInfo(checksum);
               if (vinfo) {
                  version = vinfo->GetClassVersion();
               } else {
                  Error("ReadVersion", "Could not find the StreamerInfo with a checksum of 0x%x for the class \"%s\" in %s.",
                        checksum, cl->GetName(), ((TFile*)fParent)->GetName());
                  return 0;
               }
            }
            else  {
               Error("ReadVersion", "Class %s not known to file %s.",
                 cl->GetName(), ((TFile*)fParent)->GetName());
               version = 0;
            }
         }
      }
   }
   return version;
}

//______________________________________________________________________________
Version_t TBufferFile::ReadVersionNoCheckSum(UInt_t *startpos, UInt_t *bcnt)
{
   // Read class version from I/O buffer, when the caller knows for sure that
   // there is no checksum written/involved.

   Version_t version;

   if (startpos) {
      // before reading object save start position
      *startpos = UInt_t(fBufCur-fBuffer);
   }

   // read byte count (older files don't have byte count)
   // byte count is packed in two individual shorts, this to be
   // backward compatible with old files that have at this location
   // only a single short (i.e. the version)
   union {
      UInt_t     cnt;
      Version_t  vers[2];
   } v;
#ifdef R__BYTESWAP
   frombuf(this->fBufCur,&v.vers[1]);
   frombuf(this->fBufCur,&v.vers[0]);
#else
   frombuf(this->fBufCur,&v.vers[0]);
   frombuf(this->fBufCur,&v.vers[1]);
#endif

   // no bytecount, backup and read version
   if (!(v.cnt & kByteCountMask)) {
      fBufCur -= sizeof(UInt_t);
      v.cnt = 0;
   }
   if (bcnt) *bcnt = (v.cnt & ~kByteCountMask);
   frombuf(this->fBufCur,&version);

   return version;
}

//______________________________________________________________________________
Version_t TBufferFile::ReadVersionForMemberWise(const TClass *cl)
{
   // Read class version from I/O buffer ; to be used when streaming out
   // memberwise streamed collection where we do not care (not save) about
   // the byte count and can safely ignore missing streamerInfo (since they
   // usually indicate empty collections).

   Version_t version;

   // not interested in byte count
   frombuf(this->fBufCur,&version);

   if (version<=1) {
      if (version <= 0)  {
         if (cl) {
            if (cl->GetClassVersion() != 0) {
               UInt_t checksum = 0;
               frombuf(this->fBufCur,&checksum);
               TStreamerInfo *vinfo = (TStreamerInfo*)cl->FindStreamerInfo(checksum);
               if (vinfo) {
                  return vinfo->TStreamerInfo::GetClassVersion(); // Try to get inlining.
               } else {
                  // There are some cases (for example when the buffer was stored outside of
                  // a ROOT file) where we do not have a TStreamerInfo.  If the checksum is
                  // the one from the current class, we can still assume that we can read
                  // the data so let use it.
                  if (checksum==cl->GetCheckSum() || cl->MatchLegacyCheckSum(checksum)) {
                     version = cl->GetClassVersion();
                  } else {
                     // If we can not find the streamerInfo this means that
                     // we do not actully need it (the collection is always empty
                     // in this file), so no need to issue a warning.
                     return 0;
                  }
               }
            }
         } else { // of if (cl) {
            UInt_t checksum = 0;
            frombuf(this->fBufCur,&checksum);
         }
      }  else if (version == 1 && fParent && ((TFile*)fParent)->GetVersion()<40000 && cl && cl->GetClassVersion() != 0) {
         // We could have a file created using a Foreign class before
         // the introduction of the CheckSum.  We need to check
         if ((!cl->IsLoaded() || cl->IsForeign()) && Class_Has_StreamerInfo(cl) ) {

            const TList *list = ((TFile*)fParent)->GetStreamerInfoCache();
            const TStreamerInfo *local = list ? (TStreamerInfo*)list->FindObject(cl->GetName()) : 0;
            if ( local )  {
               UInt_t checksum = local->GetCheckSum();
               TStreamerInfo *vinfo = (TStreamerInfo*)cl->FindStreamerInfo(checksum);
               if (vinfo) {
                  version = vinfo->GetClassVersion();
               } else {
                  // If we can not find the streamerInfo this means that
                  // we do not actully need it (the collection is always empty
                  // in this file), so no need to issue a warning.
                  return 0;
               }
            }
            else  {
               Error("ReadVersion", "Class %s not known to file %s.",
                 cl->GetName(), ((TFile*)fParent)->GetName());
               version = 0;
            }
         }
      }
   }
   return version;
}

//______________________________________________________________________________
UInt_t TBufferFile::WriteVersion(const TClass *cl, Bool_t useBcnt)
{
   // Write class version to I/O buffer.

   UInt_t cntpos = 0;
   if (useBcnt) {
      // reserve space for leading byte count
      cntpos   = UInt_t(fBufCur-fBuffer);
      fBufCur += sizeof(UInt_t);
   }

   Version_t version = cl->GetClassVersion();
   if (version<=1 && cl->IsForeign()) {
      *this << Version_t(0);
      *this << cl->GetCheckSum();
   } else {
      if (version > kMaxVersion) {
         Error("WriteVersion", "version number cannot be larger than %hd)",
               kMaxVersion);
         version = kMaxVersion;
      }
      *this <<version;
   }

   // return position where to store possible byte count
   return cntpos;
}

//______________________________________________________________________________
UInt_t TBufferFile::WriteVersionMemberWise(const TClass *cl, Bool_t useBcnt)
{
   // Write class version to I/O buffer after setting the kStreamedMemberWise
   // bit in the version number.

   UInt_t cntpos = 0;
   if (useBcnt) {
      // reserve space for leading byte count
      cntpos   = UInt_t(fBufCur-fBuffer);
      fBufCur += sizeof(UInt_t);
   }

   Version_t version = cl->GetClassVersion();
   if (version<=1 && cl->IsForeign()) {
      Error("WriteVersionMemberWise", "Member-wise streaming of foreign collection not yet implemented!");
      *this << Version_t(0);
      *this << cl->GetCheckSum();
   } else {
      if (version > kMaxVersion) {
         Error("WriteVersionMemberWise", "version number cannot be larger than %hd)",
               kMaxVersion);
         version = kMaxVersion;
      }
      version |= kStreamedMemberWise;
      *this <<version;
   }

   // return position where to store possible byte count
   return cntpos;
}

//______________________________________________________________________________
void TBufferFile::StreamObject(void *obj, const type_info &typeinfo, const TClass* onFileClass )
{
   // Stream an object given its C++ typeinfo information.

   TClass *cl = TClass::GetClass(typeinfo);
   if (cl) cl->Streamer(obj, *this, (TClass*)onFileClass );
   else Warning("StreamObject","No TClass for the type %s is available, the object was not read.", typeinfo.name());
}

//______________________________________________________________________________
void TBufferFile::StreamObject(void *obj, const char *className, const TClass* onFileClass)
{
   // Stream an object given the name of its actual class.

   TClass *cl = TClass::GetClass(className);
   if (cl) cl->Streamer(obj, *this, (TClass*)onFileClass );
   else Warning("StreamObject","No TClass for the type %s is available, the object was not read.", className);
}

//______________________________________________________________________________
void TBufferFile::StreamObject(void *obj, const TClass *cl, const TClass* onFileClass )
{
   // Stream an object given a pointer to its actual class.

   ((TClass*)cl)->Streamer(obj, *this, (TClass*)onFileClass );
}

//______________________________________________________________________________
void TBufferFile::StreamObject(TObject *obj)
{
   // Stream an object inheriting from TObject using its streamer.

   obj->Streamer(*this);
}

//______________________________________________________________________________
void TBufferFile::CheckCount(UInt_t offset)
{
   // Check if offset is not too large (< kMaxMapCount) when writing.

   if (IsWriting()) {
      if (offset >= kMaxMapCount) {
         Error("CheckCount", "buffer offset too large (larger than %d)", kMaxMapCount);
         // exception
      }
   }
}

//______________________________________________________________________________
UInt_t TBufferFile::CheckObject(UInt_t offset, const TClass *cl, Bool_t readClass)
{
   // Check for object in the read map. If the object is 0 it still has to be
   // read. Try to read it from the buffer starting at location offset. If the
   // object is -1 then it really does not exist and we return 0. If the object
   // exists just return the offset.

   // in position 0 we always have the reference to the null object
   if (!offset) return offset;

   Long_t cli;

   if (readClass) {
      if ((cli = fMap->GetValue(offset)) == 0) {
         // No class found at this location in map. It might have been skipped
         // as part of a skipped object. Try to explicitly read the class.

         // save fBufCur and set to place specified by offset (-kMapOffset-sizeof(bytecount))
         char *bufsav = fBufCur;
         fBufCur = (char *)(fBuffer + offset-kMapOffset-sizeof(UInt_t));

         TClass *c = ReadClass(cl);
         if (c == (TClass*) -1) {
            // mark class as really not available
            fMap->Remove(offset);
            fMap->Add(offset, -1);
            offset = 0;
            if (cl)
               Warning("CheckObject", "reference to unavailable class %s,"
                       " pointers of this type will be 0", cl->GetName());
            else
               Warning("CheckObject", "reference to an unavailable class,"
                       " pointers of that type will be 0");
         }

         fBufCur = bufsav;

      } else if (cli == -1) {

         // class really does not exist
         return 0;
      }

   } else {

      if ((cli = fMap->GetValue(offset)) == 0) {
         // No object found at this location in map. It might have been skipped
         // as part of a skipped object. Try to explicitly read the object.

         // save fBufCur and set to place specified by offset (-kMapOffset)
         char *bufsav = fBufCur;
         fBufCur = (char *)(fBuffer + offset-kMapOffset);

         TObject *obj = ReadObject(cl);
         if (!obj) {
            // mark object as really not available
            fMap->Remove(offset);
            fMap->Add(offset, -1);
            Warning("CheckObject", "reference to object of unavailable class %s, offset=%d"
                    " pointer will be 0", cl ? cl->GetName() : "TObject",offset);
            offset = 0;
         }

         fBufCur = bufsav;

      } else if (cli == -1) {

         // object really does not exist
         return 0;
      }

   }

   return offset;
}

//______________________________________________________________________________
Bool_t TBufferFile::CheckObject(const TObject *obj)
{
   // Check if the specified object is already in the buffer.
   // Returns kTRUE if object already in the buffer, kFALSE otherwise
   // (also if obj is 0 or TBuffer not in writing mode).

   return CheckObject(obj, TObject::Class());
}

//______________________________________________________________________________
Bool_t TBufferFile::CheckObject(const void *obj, const TClass *ptrClass)
{
   // Check if the specified object of the specified class is already in
   // the buffer. Returns kTRUE if object already in the buffer,
   // kFALSE otherwise (also if obj is 0 ).

   if (!obj || !fMap || !ptrClass) return kFALSE;

   TClass *clActual = ptrClass->GetActualClass(obj);

   ULong_t idx;

   if (clActual && (ptrClass != clActual)) {
      const char *temp = (const char*) obj;
      temp -= clActual->GetBaseClassOffset(ptrClass);
      idx = (ULong_t)fMap->GetValue(Void_Hash(temp), (Long_t)temp);
   } else {
      idx = (ULong_t)fMap->GetValue(Void_Hash(obj), (Long_t)obj);
   }

   return idx ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TBufferFile::SetPidOffset(UShort_t offset)
{
   // This offset is used when a key (or basket) is transfered from one
   // file to the other.  In this case the TRef and TObject might have stored a
   // pid index (to retrieve TProcessIDs) which refered to their order on the original
   // file, the fPidOffset is to be added to those values to correctly find the
   // TProcessID.  This fPidOffset needs to be increment if the key/basket is copied
   // and need to be zero for new key/basket.

   fPidOffset = offset;
}

//______________________________________________________________________________
void TBufferFile::GetMappedObject(UInt_t tag, void* &ptr, TClass* &ClassPtr) const
{
   // Retrieve the object stored in the buffer's object map at 'tag'
   // Set ptr and ClassPtr respectively to the address of the object and
   // a pointer to its TClass.

   if (tag > (UInt_t)fMap->GetSize()) {
      ptr = 0;
      ClassPtr = 0;
   } else {
      ptr = (void*)(Long_t)fMap->GetValue(tag);
      ClassPtr = (TClass*) (Long_t)fClassMap->GetValue(tag);
   }
}

//______________________________________________________________________________
void TBufferFile::MapObject(const TObject *obj, UInt_t offset)
{
   // Add object to the fMap container.
   // If obj is not 0 add object to the map (in read mode also add 0 objects to
   // the map). This method may only be called outside this class just before
   // calling obj->Streamer() to prevent self reference of obj, in case obj
   // contains (via via) a pointer to itself. In that case offset must be 1
   // (default value for offset).

   if (IsWriting()) {
      if (!fMap) InitMap();

      if (obj) {
         CheckCount(offset);
         ULong_t hash = Void_Hash(obj);
         fMap->Add(hash, (Long_t)obj, offset);
         // No need to keep track of the class in write mode
         // fClassMap->Add(hash, (Long_t)obj, (Long_t)((TObject*)obj)->IsA());
         fMapCount++;
      }
   } else {
      if (!fMap || !fClassMap) InitMap();

      fMap->Add(offset, (Long_t)obj);
      fClassMap->Add(offset,
             (obj && obj != (TObject*)-1) ? (Long_t)((TObject*)obj)->IsA() : 0);
      fMapCount++;
   }
}

//______________________________________________________________________________
void TBufferFile::MapObject(const void *obj, const TClass* cl, UInt_t offset)
{
   // Add object to the fMap container.
   // If obj is not 0 add object to the map (in read mode also add 0 objects to
   // the map). This method may only be called outside this class just before
   // calling obj->Streamer() to prevent self reference of obj, in case obj
   // contains (via via) a pointer to itself. In that case offset must be 1
   // (default value for offset).

   if (IsWriting()) {
      if (!fMap) InitMap();

      if (obj) {
         CheckCount(offset);
         ULong_t hash = Void_Hash(obj);
         fMap->Add(hash, (Long_t)obj, offset);
         // No need to keep track of the class in write mode
         // fClassMap->Add(hash, (Long_t)obj, (Long_t)cl);
         fMapCount++;
      }
   } else {
      if (!fMap || !fClassMap) InitMap();

      fMap->Add(offset, (Long_t)obj);
      fClassMap->Add(offset, (Long_t)cl);
      fMapCount++;
   }
}

//______________________________________________________________________________
void TBufferFile::SetReadParam(Int_t mapsize)
{
   // Set the initial size of the map used to store object and class
   // references during reading. The default size is kMapSize=503.
   // Increasing the default has the benefit that when reading many
   // small objects the map does not need to be resized too often
   // (the system is always dynamic, even with the default everything
   // will work, only the initial resizing will cost some time).
   // This method can only be called directly after the creation of
   // the TBuffer, before any reading is done. Globally this option
   // can be changed using SetGlobalReadParam().

   R__ASSERT(IsReading());
   R__ASSERT(fMap == 0);

   fMapSize = mapsize;
}

//______________________________________________________________________________
void TBufferFile::SetWriteParam(Int_t mapsize)
{
   // Set the initial size of the hashtable used to store object and class
   // references during writing. The default size is kMapSize=503.
   // Increasing the default has the benefit that when writing many
   // small objects the hashtable does not get too many collisions
   // (the system is always dynamic, even with the default everything
   // will work, only a large number of collisions will cost performance).
   // For optimal performance hashsize should always be a prime.
   // This method can only be called directly after the creation of
   // the TBuffer, before any writing is done. Globally this option
   // can be changed using SetGlobalWriteParam().

   R__ASSERT(IsWriting());
   R__ASSERT(fMap == 0);

   fMapSize = mapsize;
}

//______________________________________________________________________________
void TBufferFile::InitMap()
{
   // Create the fMap container and initialize them
   // with the null object.

   if (IsWriting()) {
      if (!fMap) {
         fMap = new TExMap(fMapSize);
         // No need to keep track of the class in write mode
         // fClassMap = new TExMap(fMapSize);
         fMapCount = 0;
      }
   } else {
      if (!fMap) {
         fMap = new TExMap(fMapSize);
         fMap->Add(0, kNullTag);      // put kNullTag in slot 0
         fMapCount = 1;
      } else if (fMapCount==0) {
         fMap->Add(0, kNullTag);      // put kNullTag in slot 0
         fMapCount = 1;
      }
      if (!fClassMap) {
         fClassMap = new TExMap(fMapSize);
         fClassMap->Add(0, kNullTag);      // put kNullTag in slot 0
      }
   }
}

//______________________________________________________________________________
void TBufferFile::ResetMap()
{
   // Delete existing fMap and reset map counter.

   if (fMap) fMap->Delete();
   if (fClassMap) fClassMap->Delete();
   fMapCount     = 0;
   fDisplacement = 0;

   // reset user bits
   ResetBit(kUser1);
   ResetBit(kUser2);
   ResetBit(kUser3);
}

//______________________________________________________________________________
Int_t TBufferFile::ReadBuf(void *buf, Int_t max)
{
   // Read max bytes from the I/O buffer into buf. The function returns
   // the actual number of bytes read.

   R__ASSERT(IsReading());

   if (max == 0) return 0;

   Int_t n = TMath::Min(max, (Int_t)(fBufMax - fBufCur));

   memcpy(buf, fBufCur, n);
   fBufCur += n;

   return n;
}

//______________________________________________________________________________
void TBufferFile::WriteBuf(const void *buf, Int_t max)
{
   // Write max bytes from buf into the I/O buffer.

   R__ASSERT(IsWriting());

   if (max == 0) return;

   if (fBufCur + max > fBufMax) AutoExpand(fBufSize+max); // a more precise request would be: fBufSize + max - (fBufMax - fBufCur)

   memcpy(fBufCur, buf, max);
   fBufCur += max;
}

//______________________________________________________________________________
char *TBufferFile::ReadString(char *s, Int_t max)
{
   // Read string from I/O buffer. String is read till 0 character is
   // found or till max-1 characters are read (i.e. string s has max
   // bytes allocated). If max = -1 no check on number of character is
   // made, reading continues till 0 character is found.

   R__ASSERT(IsReading());

   char  ch;
   Int_t nr = 0;

   if (max == -1) max = kMaxInt;

   while (nr < max-1) {

      *this >> ch;

      // stop when 0 read
      if (ch == 0) break;

      s[nr++] = ch;
   }

   s[nr] = 0;
   return s;
}

//______________________________________________________________________________
void TBufferFile::WriteString(const char *s)
{
   // Write string to I/O buffer. Writes string upto and including the
   // terminating 0.

   WriteBuf(s, (strlen(s)+1)*sizeof(char));
}

//______________________________________________________________________________
TProcessID *TBufferFile::GetLastProcessID(TRefTable *reftable) const
{
   // Return the last TProcessID in the file.

   TFile *file = (TFile*)GetParent();
   // warn if the file contains > 1 PID (i.e. if we might have ambiguity)
   if (file && !reftable->TestBit(TRefTable::kHaveWarnedReadingOld) && file->GetNProcessIDs()>1) {
      Warning("ReadBuffer", "The file was written during several processes with an "
         "older ROOT version; the TRefTable entries might be inconsistent.");
      reftable->SetBit(TRefTable::kHaveWarnedReadingOld);
   }

   // the file's last PID is the relevant one, all others might have their tables overwritten
   TProcessID *fileProcessID = TProcessID::GetProcessID(0);
   if (file && file->GetNProcessIDs() > 0) {
      // take the last loaded PID
      fileProcessID = (TProcessID *) file->GetListOfProcessIDs()->Last();
   }
   return fileProcessID;
}

//______________________________________________________________________________
TProcessID *TBufferFile::ReadProcessID(UShort_t pidf)
{
   // The TProcessID with number pidf is read from file.
   // If the object is not already entered in the gROOT list, it is added.

   TFile *file = (TFile*)GetParent();
   if (!file) {
      if (!pidf) return TProcessID::GetPID(); //may happen when cloning an object
      return 0;
   }
   return file->ReadProcessID(pidf);
}

//______________________________________________________________________________
UInt_t TBufferFile::GetTRefExecId()
{
   // Return the exec id stored in the current TStreamerInfo element.
   // The execid has been saved in the unique id of the TStreamerElement
   // being read by TStreamerElement::Streamer.
   // The current element (fgElement) is set as a static global
   // by TStreamerInfo::ReadBuffer (Clones) when reading this TRef.

   return TStreamerInfo::GetCurrentElement()->GetUniqueID();
}

//______________________________________________________________________________
UShort_t TBufferFile::WriteProcessID(TProcessID *pid)
{
   // Check if the ProcessID pid is already in the file.
   // If not, add it and return the index number in the local file list.

   TFile *file = (TFile*)GetParent();
   if (!file) return 0;
   return file->WriteProcessID(pid);
}

//---- Utilities for TStreamerInfo ----------------------------------------------

//______________________________________________________________________________
void TBufferFile::ForceWriteInfo(TVirtualStreamerInfo *info, Bool_t force)
{
   // force writing the TStreamerInfo to the file

   if (info) info->ForceWriteInfo((TFile*)GetParent(),force);
}


//______________________________________________________________________________
void TBufferFile::ForceWriteInfoClones(TClonesArray *a)
{
   // Make sure TStreamerInfo is not optimized, otherwise it will not be
   // possible to support schema evolution in read mode.
   // In case the StreamerInfo has already been computed and optimized,
   // one must disable the option BypassStreamer.

   TStreamerInfo *sinfo = (TStreamerInfo*)a->GetClass()->GetStreamerInfo();
   ForceWriteInfo(sinfo,kFALSE);
}

//______________________________________________________________________________
Int_t TBufferFile::ReadClones(TClonesArray *a, Int_t nobjects, Version_t objvers)
{
   // Interface to TStreamerInfo::ReadBufferClones.

   char **arr = (char **)a->GetObjectRef(0);
   char **end = arr + nobjects;
   //a->GetClass()->GetStreamerInfo()->ReadBufferClones(*this,a,nobjects,-1,0);
   TStreamerInfo *info = (TStreamerInfo*)a->GetClass()->GetStreamerInfo(objvers);
   //return info->ReadBuffer(*this,arr,-1,nobjects,0,1);
   return ApplySequenceVecPtr(*(info->GetReadMemberWiseActions(kTRUE)),arr,end);
}

//______________________________________________________________________________
Int_t TBufferFile::WriteClones(TClonesArray *a, Int_t nobjects)
{
   // Interface to TStreamerInfo::WriteBufferClones.

   char **arr = reinterpret_cast<char**>(a->GetObjectRef(0));
   //a->GetClass()->GetStreamerInfo()->WriteBufferClones(*this,(TClonesArray*)a,nobjects,-1,0);
   TStreamerInfo *info = (TStreamerInfo*)a->GetClass()->GetStreamerInfo();
   //return info->WriteBufferAux(*this,arr,-1,nobjects,0,1);
   char **end = arr + nobjects;
   // No need to tell call ForceWriteInfo as it by ForceWriteInfoClones.
   return ApplySequenceVecPtr(*(info->GetWriteMemberWiseActions(kTRUE)),arr,end);
}

//______________________________________________________________________________
Int_t TBufferFile::ReadClassEmulated(const TClass *cl, void *object, const TClass *onFileClass)
{
   // Read emulated class.

   UInt_t start,count;
   //We assume that the class was written with a standard streamer
   //We attempt to recover if a version count was not written
   Version_t v = ReadVersion(&start,&count);

   if (count) {
      TStreamerInfo *sinfo = 0;
      if( onFileClass ) {
         sinfo = (TStreamerInfo*)cl->GetConversionStreamerInfo( onFileClass, v );
         if( !sinfo )
            return 0;
      }

      sinfo = (TStreamerInfo*)cl->GetStreamerInfo(v);
      ApplySequence(*(sinfo->GetReadObjectWiseActions()), object);
      if (sinfo->IsRecovered()) count=0;
      CheckByteCount(start,count,cl);
   } else {
      SetBufferOffset(start);
      TStreamerInfo *sinfo = ((TStreamerInfo*)cl->GetStreamerInfo());
      ApplySequence(*(sinfo->GetReadObjectWiseActions()), object);
   }
   return 0;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadClassBuffer(const TClass *cl, void *pointer, Int_t version, UInt_t start, UInt_t count, const TClass *onFileClass)
{
   // Deserialize information from a buffer into an object.
   //
   // Note: This function is called by the xxx::Streamer()
   //       functions in rootcint-generated dictionaries.
   //   // This function assumes that the class version and the byte count
   // information have been read.
   //
   //   version  is the version number of the class
   //   start    is the starting position in the buffer b
   //   count    is the number of bytes for this object in the buffer
   //

   TObjArray* infos;
   Int_t ninfos;
   {
      R__LOCKGUARD(gCINTMutex);
      infos = cl->GetStreamerInfos();
      ninfos = infos->GetSize();
   }
   if (version < -1 || version >= ninfos) {
      Error("ReadBuffer1", "class: %s, attempting to access a wrong version: %d, object skipped at offset %d",
            cl->GetName(), version, Length() );
      CheckByteCount(start, count, cl);
      return 0;
   }

   //---------------------------------------------------------------------------
   // The ondisk class has been specified so get foreign streamer info
   //---------------------------------------------------------------------------
   TStreamerInfo *sinfo = 0;
   if( onFileClass ) {
      sinfo = (TStreamerInfo*)cl->GetConversionStreamerInfo( onFileClass, version );
      if( !sinfo ) {
         Error("ReadClassBuffer",
               "Could not find the right streamer info to convert %s version %d into a %s, object skipped at offset %d",
               onFileClass->GetName(), version, cl->GetName(), Length() );
         CheckByteCount(start, count, onFileClass);
         return 0;
      }
   }
   //---------------------------------------------------------------------------
   // Get local streamer info
   //---------------------------------------------------------------------------
   else {
      // The StreamerInfo should exist at this point.

      R__LOCKGUARD(gCINTMutex);
      sinfo = (TStreamerInfo*)infos->At(version);
      if (sinfo == 0) {
         // Unless the data is coming via a socket connection from with schema evolution
         // (tracking) was not enabled.  So let's create the StreamerInfo if it is the
         // one for the current version, otherwise let's complain ...
         // We could also get here if there old class version was '1' and the new class version is higher than 1
         // AND the checksum is the same.
         if ( version == cl->GetClassVersion() || version == 1 ) {
            const_cast<TClass*>(cl)->BuildRealData(pointer);
            sinfo = new TStreamerInfo(const_cast<TClass*>(cl));
            infos->AddAtAndExpand(sinfo, version);
            if (gDebug > 0) printf("Creating StreamerInfo for class: %s, version: %d\n", cl->GetName(), version);
            sinfo->Build();
         } else if (version==0) {
            // When the object was written the class was version zero, so
            // there is no StreamerInfo to be found.
            // Check that the buffer position corresponds to the byte count.
            CheckByteCount(start, count, cl);
            return 0;
         } else {
            Error("ReadClassBuffer", "Could not find the StreamerInfo for version %d of the class %s, object skipped at offset %d",
                  version, cl->GetName(), Length() );
            CheckByteCount(start, count, cl);
            return 0;
         }
      } else if (!sinfo->IsCompiled()) {  // Note this read is protected by the above lock.
         // Streamer info has not been compiled, but exists.
         // Therefore it was read in from a file and we have to do schema evolution.
         const_cast<TClass*>(cl)->BuildRealData(pointer);
         sinfo->BuildOld();
      }
   }

   // Deserialize the object.
   ApplySequence(*(sinfo->GetReadObjectWiseActions()), (char*)pointer);
   if (sinfo->IsRecovered()) count=0;

   // Check that the buffer position corresponds to the byte count.
   CheckByteCount(start, count, cl);
   return 0;
}

//______________________________________________________________________________
Int_t TBufferFile::ReadClassBuffer(const TClass *cl, void *pointer, const TClass *onFileClass)
{
   // Deserialize information from a buffer into an object.
   //
   // Note: This function is called by the xxx::Streamer()
   //       functions in rootcint-generated dictionaries.
   //

   // Read the class version from the buffer.
   UInt_t R__s = 0; // Start of object.
   UInt_t R__c = 0; // Count of bytes.
   Version_t version;

   if( onFileClass )
      version = ReadVersion(&R__s, &R__c, onFileClass);
   else
      version = ReadVersion(&R__s, &R__c, cl);

   Bool_t v2file = kFALSE;
   TFile *file = (TFile*)GetParent();
   if (file && file->GetVersion() < 30000) {
      version = -1; //This is old file
      v2file = kTRUE;
   }

   //---------------------------------------------------------------------------
   // The ondisk class has been specified so get foreign streamer info
   //---------------------------------------------------------------------------
   TStreamerInfo *sinfo = 0;
   if( onFileClass ) {
      sinfo = (TStreamerInfo*)cl->GetConversionStreamerInfo( onFileClass, version );
      if( !sinfo ) {
         Error("ReadClassBuffer",
               "Could not find the right streamer info to convert %s version %d into a %s, object skipped at offset %d",
               onFileClass->GetName(), version, cl->GetName(), Length() );
         CheckByteCount(R__s, R__c, onFileClass);
         return 0;
      }
   }
   //---------------------------------------------------------------------------
   // Get local streamer info
   //---------------------------------------------------------------------------
   else {
      // The StreamerInfo should exist at this point.
      TStreamerInfo *guess = (TStreamerInfo*)cl->GetLastReadInfo();
      if (guess && guess->GetClassVersion() == version) {
         sinfo = guess;
      } else {
         // The last one is not the one we are looking for.
         {
            R__LOCKGUARD(gCINTMutex);

            const TObjArray *infos = cl->GetStreamerInfos();
            Int_t infocapacity = infos->Capacity();
            if (infocapacity) {
               if (version < -1 || version >= infocapacity) {
                  Error("ReadClassBuffer","class: %s, attempting to access a wrong version: %d, object skipped at offset %d",
                        cl->GetName(), version, Length());
                  CheckByteCount(R__s, R__c, cl);
                  return 0;
               }
               sinfo = (TStreamerInfo*) infos->UncheckedAt(version);
               if (sinfo) {
                  if (!sinfo->IsCompiled())
                  {
                     // Streamer info has not been compiled, but exists.
                     // Therefore it was read in from a file and we have to do schema evolution?
                     const_cast<TClass*>(cl)->BuildRealData(pointer);
                     sinfo->BuildOld();
                  }
                  // If the compilation succeeded, remember this StreamerInfo.
                  // const_cast okay because of the lock on gInterpreterMutex.
                  if (sinfo->IsCompiled()) const_cast<TClass*>(cl)->SetLastReadInfo(sinfo);
               }
            }
         }

         if (sinfo == 0) {
            // Unless the data is coming via a socket connection from with schema evolution
            // (tracking) was not enabled.  So let's create the StreamerInfo if it is the
            // one for the current version, otherwise let's complain ...
            // We could also get here when reading a file prior to the introduction of StreamerInfo.
            // We could also get here if there old class version was '1' and the new class version is higher than 1
            // AND the checksum is the same.
            if (v2file || version == cl->GetClassVersion() || version == 1 ) {
               R__LOCKGUARD(gCINTMutex);
               TObjArray *infos = cl->GetStreamerInfos();

               const_cast<TClass*>(cl)->BuildRealData(pointer);
               sinfo = new TStreamerInfo(const_cast<TClass*>(cl));
               infos->AddAtAndExpand(sinfo,version);
               if (gDebug > 0) printf("Creating StreamerInfo for class: %s, version: %d\n", cl->GetName(), version);
               if (v2file) {
                  sinfo->Build(); // Get the elements.
                  sinfo->Clear("build"); // Undo compilation.
                  sinfo->BuildEmulated(file); // Fix the types and redo compilation.
               } else {
                  sinfo->Build();
               }
            } else if (version==0) {
               // When the object was written the class was version zero, so
               // there is no StreamerInfo to be found.
               // Check that the buffer position corresponds to the byte count.
               CheckByteCount(R__s, R__c, cl);
               return 0;
            } else {
               Error( "ReadClassBuffer", "Could not find the StreamerInfo for version %d of the class %s, object skipped at offset %d",
                     version, cl->GetName(), Length() );
               CheckByteCount(R__s, R__c, cl);
               return 0;
            }
         }
      }
   }

   //deserialize the object
   ApplySequence(*(sinfo->GetReadObjectWiseActions()), (char*)pointer );
   if (sinfo->TStreamerInfo::IsRecovered()) R__c=0; // 'TStreamerInfo::' avoids going via a virtual function.

   // Check that the buffer position corresponds to the byte count.
   CheckByteCount(R__s, R__c, cl);

   if (gDebug > 2) printf(" ReadBuffer for class: %s has read %d bytes\n", cl->GetName(), R__c);

   return 0;
}

//______________________________________________________________________________
Int_t TBufferFile::WriteClassBuffer(const TClass *cl, void *pointer)
{
   // Function called by the Streamer functions to serialize object at p
   // to buffer b. The optional argument info may be specified to give an
   // alternative StreamerInfo instead of using the default StreamerInfo
   // automatically built from the class definition.
   // For more information, see class TStreamerInfo.

   //build the StreamerInfo if first time for the class
   TStreamerInfo *sinfo = (TStreamerInfo*)const_cast<TClass*>(cl)->GetCurrentStreamerInfo();
   if (sinfo == 0) {
      //Have to be sure between the check and the taking of the lock if the current streamer has changed
      R__LOCKGUARD(gCINTMutex);
      sinfo = (TStreamerInfo*)const_cast<TClass*>(cl)->GetCurrentStreamerInfo();
      if(sinfo == 0) {
         const_cast<TClass*>(cl)->BuildRealData(pointer);
         sinfo = new TStreamerInfo(const_cast<TClass*>(cl));
         const_cast<TClass*>(cl)->SetCurrentStreamerInfo(sinfo);
         cl->GetStreamerInfos()->AddAtAndExpand(sinfo,cl->GetClassVersion());
         if (gDebug > 0) printf("Creating StreamerInfo for class: %s, version: %d\n",cl->GetName(),cl->GetClassVersion());
         sinfo->Build();
      }
   } else if (!sinfo->IsCompiled()) {
      R__LOCKGUARD(gCINTMutex);
      // Redo the test in case we have been victim of a data race on fIsCompiled.
      if (!sinfo->IsCompiled()) {
         const_cast<TClass*>(cl)->BuildRealData(pointer);
         sinfo->BuildOld();
      }
   }

   //write the class version number and reserve space for the byte count
   UInt_t R__c = WriteVersion(cl, kTRUE);

   //NOTE: In the future Philippe wants this to happen via a custom action
   TagStreamerInfo(sinfo);
   ApplySequence(*(sinfo->GetWriteObjectWiseActions()), (char*)pointer);


   //write the byte count at the start of the buffer
   SetByteCount(R__c, kTRUE);

   if (gDebug > 2) printf(" WriteBuffer for class: %s version %d has written %d bytes\n",cl->GetName(),cl->GetClassVersion(),UInt_t(fBufCur - fBuffer) - R__c - (UInt_t)sizeof(UInt_t));
   return 0;
}

//______________________________________________________________________________
Int_t TBufferFile::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *obj)
{
   // Read one collection of objects from the buffer using the StreamerInfoLoopAction.
   // The collection needs to be a split TClonesArray or a split vector of pointers.

   if (gDebug) {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         (*iter).PrintDebug(*this,obj);
         (*iter)(*this,obj);
      }

   } else {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         (*iter)(*this,obj);
      }
   }

   return 0;
}

//______________________________________________________________________________
Int_t TBufferFile::ApplySequenceVecPtr(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection)
{
   // Read one collection of objects from the buffer using the StreamerInfoLoopAction.
   // The collection needs to be a split TClonesArray or a split vector of pointers.

   if (gDebug) {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         (*iter).PrintDebug(*this,*(char**)start_collection);  // Warning: This limits us to TClonesArray and vector of pointers.
         (*iter)(*this,start_collection,end_collection);
      }

   } else {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         (*iter)(*this,start_collection,end_collection);
      }
   }

   return 0;
}

//______________________________________________________________________________
Int_t TBufferFile::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection)
{
   // Read one collection of objects from the buffer using the StreamerInfoLoopAction.

   TStreamerInfoActions::TLoopConfiguration *loopconfig = sequence.fLoopConfig;
   if (gDebug) {

      // Get the address of the first item for the PrintDebug.
      // (Performance is not essential here since we are going to print to
      // the screen anyway).
      void *arr0 = loopconfig->GetFirstAddress(start_collection,end_collection);
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         (*iter).PrintDebug(*this,arr0);
         (*iter)(*this,start_collection,end_collection,loopconfig);
      }

   } else {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         (*iter)(*this,start_collection,end_collection,loopconfig);
      }
   }

   return 0;
}

//---- Static functions --------------------------------------------------------

//______________________________________________________________________________
void TBufferFile::SetGlobalReadParam(Int_t mapsize)
{
   // Set the initial size of the map used to store object and class
   // references during reading. The default size is kMapSize=503.
   // Increasing the default has the benefit that when reading many
   // small objects the array does not need to be resized too often
   // (the system is always dynamic, even with the default everything
   // will work, only the initial resizing will cost some time).
   // Per TBuffer object this option can be changed using SetReadParam().

   fgMapSize = mapsize;
}

//______________________________________________________________________________
void TBufferFile::SetGlobalWriteParam(Int_t mapsize)
{
   // Set the initial size of the hashtable used to store object and class
   // references during writing. The default size is kMapSize=503.
   // Increasing the default has the benefit that when writing many
   // small objects the hashtable does not get too many collisions
   // (the system is always dynamic, even with the default everything
   // will work, only a large number of collisions will cost performance).
   // For optimal performance hashsize should always be a prime.
   // Per TBuffer object this option can be changed using SetWriteParam().

   fgMapSize = mapsize;
}

//______________________________________________________________________________
Int_t TBufferFile::GetGlobalReadParam()
{
   // Get default read map size.

   return fgMapSize;
}

//______________________________________________________________________________
Int_t TBufferFile::GetGlobalWriteParam()
{
   // Get default write map size.

   return fgMapSize;
}
