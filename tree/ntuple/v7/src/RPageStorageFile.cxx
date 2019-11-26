/// \file RPageStorageFile.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-11-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RRawFile.hxx>

#include <Compression.h>
#include <RVersion.h>
#include <RZip.h>
#include <TError.h>

#include <chrono>
#include <cstddef>  // for offsetof()
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <utility>

namespace {

// The following types are used to read and write the TFile binary format

/// Big-endian 16-bit unsigned integer
class RUInt16BE {
private:
   std::uint16_t fValBE = 0;
   static std::uint16_t Swap(std::uint16_t val) {
      return (val & 0x00FF) << 8 | (val & 0xFF00) >> 8;
   }
public:
   RUInt16BE() = default;
   explicit RUInt16BE(const std::uint16_t val) : fValBE(Swap(val)) {}
   operator std::uint16_t() const {
      return Swap(fValBE);
   }
   RUInt16BE& operator =(const std::uint16_t val) {
      fValBE = Swap(val);
      return *this;
   }
};

/// Big-endian 32-bit unsigned integer
class RUInt32BE {
private:
   std::uint32_t fValBE = 0;
   static std::uint32_t Swap(std::uint32_t val) {
      auto x = (val & 0x0000FFFF) << 16 | (val & 0xFFFF0000) >> 16;
      return (x & 0x00FF00FF) << 8 | (x & 0xFF00FF00) >> 8;
   }
public:
   RUInt32BE() = default;
   explicit RUInt32BE(const std::uint32_t val) : fValBE(Swap(val)) {}
   operator std::uint32_t() const {
      return Swap(fValBE);
   }
   RUInt32BE& operator =(const std::uint32_t val) {
      fValBE = Swap(val);
      return *this;
   }
};

/// Big-endian 32-bit signed integer
class RInt32BE {
private:
   std::int32_t fValBE = 0;
   static std::int32_t Swap(std::int32_t val) {
      auto x = (val & 0x0000FFFF) << 16 | (val & 0xFFFF0000) >> 16;
      return (x & 0x00FF00FF) << 8 | (x & 0xFF00FF00) >> 8;
   }
public:
   RInt32BE() = default;
   explicit RInt32BE(const std::int32_t val) : fValBE(Swap(val)) {}
   operator std::int32_t() const {
      return Swap(fValBE);
   }
   RInt32BE& operator =(const std::int32_t val) {
      fValBE = Swap(val);
      return *this;
   }
};

/// Big-endian 64-bit unsigned integer
class RUInt64BE {
private:
   std::uint64_t fValBE = 0;
   static std::uint64_t Swap(std::uint64_t val) {
      auto x = (val & 0x00000000FFFFFFFF) << 32 | (val & 0xFFFFFFFF00000000) >> 32;
      x = (x & 0x0000FFFF0000FFFF) << 16 | (x & 0xFFFF0000FFFF0000) >> 16;
      return (x & 0x00FF00FF00FF00FF) << 8  | (x & 0xFF00FF00FF00FF00) >> 8;
   }
public:
   RUInt64BE() = default;
   explicit RUInt64BE(const std::uint64_t val) : fValBE(Swap(val)) {}
   operator std::uint64_t() const {
      return Swap(fValBE);
   }
   RUInt64BE& operator =(const std::uint64_t val) {
      fValBE = Swap(val);
      return *this;
   }
};

/// TFile checksum algorithm for name and type strings
constexpr std::int32_t ChecksumString(std::int32_t id, const char *str) {
   auto len = strlen(str);
   for (unsigned i = 0; i < len; i++)
      id = id *3 + str[i];
   return id;
}

/// Composition of class RNTuple as being interpreted by TFile
constexpr std::int32_t ChecksumRNTupleClass() {
   std::int32_t id = 0;
   id = ChecksumString(id, "ROOT::Experimental::RNTuple");
   id = ChecksumString(id, "fSeekHeader");
   id = ChecksumString(id, "unsigned long");
   id = ChecksumString(id, "fNBytesHeader");
   id = ChecksumString(id, "unsigned int");
   id = ChecksumString(id, "fSeekFooter");
   id = ChecksumString(id, "unsigned long");
   id = ChecksumString(id, "fNBytesFooter");
   id = ChecksumString(id, "unsigned int");
   id = ChecksumString(id, "fReserved");
   id = ChecksumString(id, "unsigned long");
   return id;
}


#pragma pack(push, 1)
/// A name (type, identifies, ...) in the TFile binary format
struct RTFString {
   char fLName{0};
   char fData[255];
   RTFString() = default;
   RTFString(const std::string &str) {
      R__ASSERT(str.length() < 256);
      fLName = str.length();
      memcpy(fData, str.data(), fLName);
   }
   char GetSize() const { return 1 + fLName; }
};

/// The timestamp format used in TFile; the default constructor initializes with the current time
struct RTFDatetime {
   RUInt32BE fDatetime;
   RTFDatetime() {
      auto now = std::chrono::system_clock::now();
      auto tt = std::chrono::system_clock::to_time_t(now);
      auto tm = *localtime(&tt);
      fDatetime = (tm.tm_year + 1900 - 1995) << 26 | (tm.tm_mon + 1) << 22 | tm.tm_mday << 17 |
                  tm.tm_hour << 12 | tm.tm_min << 6 | tm.tm_sec;
   }
   explicit RTFDatetime(RUInt32BE val) : fDatetime(val) {}
};

/// The key part of a TFile record excluding the key, class, and title names
struct RTFKey {
   RInt32BE fNbytes{0};
   RUInt16BE fVersion{4};
   RUInt32BE fObjLen{0};
   RTFDatetime fDatetime;
   RUInt16BE fKeyLen{0};
   RUInt16BE fCycle{1};
   union {
      struct {
         RUInt32BE fSeekKey{0};
         RUInt32BE fSeekPdir{0};
      } fInfoShort;
      struct {
         RUInt64BE fSeekKey{0};
         RUInt64BE fSeekPdir{0};
      } fInfoLong;
   };

   std::uint32_t fKeyHeaderSize{18 + sizeof(fInfoShort)};  // not part of serialization

   RTFKey() : fInfoShort() {}
   RTFKey(std::uint64_t seekKey, std::uint64_t seekPdir,
          const RTFString &clName, const RTFString &objName, const RTFString &titleName,
          std::uint32_t szObjInMem, std::uint32_t szObjOnDisk = 0)
   {
      fObjLen = szObjInMem;
      if (seekKey > std::numeric_limits<std::int32_t>::max()) {
         fKeyHeaderSize = 18 + sizeof(fInfoLong);
         fKeyLen = fKeyHeaderSize + clName.GetSize() + objName.GetSize() + titleName.GetSize();
         fInfoLong.fSeekKey = seekKey;
         fInfoLong.fSeekPdir = seekPdir;
      } else {
         fKeyHeaderSize = 18 + sizeof(fInfoShort);
         fKeyLen = fKeyHeaderSize + clName.GetSize() + objName.GetSize() + titleName.GetSize();
         fInfoShort.fSeekKey = seekKey;
         fInfoShort.fSeekPdir = seekPdir;
      }
      fNbytes = fKeyLen + ((szObjOnDisk == 0) ? szObjInMem : szObjOnDisk);
   }

   std::uint32_t GetSize() const {
      if (fNbytes < 0) return -fNbytes;
      return fNbytes;
   }

   std::uint32_t GetHeaderSize() const {
      return 18 + sizeof(fInfoShort);
   }
};

/// The TFile global header
struct RTFHeader {
   char fMagic[4]{ 'r', 'o', 'o', 't' };
   RUInt32BE fVersion{(ROOT_VERSION_CODE >> 16)*10000 +
                      ((ROOT_VERSION_CODE & 0xFF00) >> 8) * 100 +
                      (ROOT_VERSION_CODE & 0xFF)};
   RUInt32BE fBEGIN{100};
   union {
      struct {
         RUInt32BE fEND{0};
         RUInt32BE fSeekFree{0};
         RUInt32BE fNbytesFree{0};
         RUInt32BE fNfree{1};
         RUInt32BE fNbytesName{0};
         unsigned char fUnits{4};
         RUInt32BE fCompress{0};
         RUInt32BE fSeekInfo{0};
         RUInt32BE fNbytesInfo{0};
      } fInfoShort;
      struct {
         RUInt64BE fEND{0};
         RUInt64BE fSeekFree{0};
         RUInt32BE fNbytesFree{0};
         RUInt32BE fNfree{1};
         RUInt32BE fNbytesName{0};
         unsigned char fUnits{8};
         RUInt32BE fCompress{0};
         RUInt64BE fSeekInfo{0};
         RUInt32BE fNbytesInfo{0};
      } fInfoLong;
   };

   RTFHeader() : fInfoShort() {}
   RTFHeader(int compression) : fInfoShort() {
      fInfoShort.fCompress = compression;
   }

   std::uint32_t GetSize() {
      std::uint32_t sizeHead = 4 + sizeof(fVersion) + sizeof(fBEGIN);
      if (fVersion >= 1000000) return sizeHead + sizeof(fInfoLong);
      return sizeHead + sizeof(fInfoShort);
   }

   std::uint64_t GetEnd() const {
      if (fVersion >= 1000000) return fInfoLong.fEND;
      return fInfoShort.fEND;
   }

   void SetEnd(std::uint64_t value) {
      if ((value > (std::uint64_t(1) << 31)) || (fVersion >= 1000000)) {
         if (fVersion < 1000000)
            fVersion = fVersion + 1000000;
         fInfoLong.fEND = value;
      } else {
         fInfoShort.fEND = value;
      }
   }

   void SetCompression(std::uint32_t value) {
      if (fVersion >= 1000000) {
         fInfoLong.fCompress = value;
      } else {
         fInfoShort.fCompress = value;
      }
   }
};

/// A reference to an unused byte-range in a TFile
struct RTFFreeEntry {
   RUInt16BE fVersion{1};
   union {
      struct {
         RUInt32BE fFirst{0};
         RUInt32BE fLast{0};
      } fInfoShort;
      struct {
         RUInt64BE fFirst{0};
         RUInt64BE fLast{0};
      } fInfoLong;
   };

   RTFFreeEntry() : fInfoShort() {}
   RTFFreeEntry(std::uint64_t first, std::uint64_t last) {
      if (last > std::numeric_limits<std::int32_t>::max()) {
         fVersion = fVersion + 1000;
         fInfoLong.fFirst = first;
         fInfoLong.fLast = last;
      } else {
         fInfoShort.fFirst = first;
         fInfoShort.fLast = last;
      }
   }
   std::uint32_t GetSize() { return (fVersion >= 1000) ? 18 : 10; }
};

/// Streamer info for TObject
struct RTFObject {
   RUInt16BE fVersion{1};
   RUInt32BE fUniqueID{0};  // unused
   RUInt32BE fBits;
   explicit RTFObject(std::uint32_t bits) : fBits(bits) {}
};

/// Streamer info for data member RNTuple::fSeekHeader
struct RTFStreamerElementSeekHeader {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerElementSeekHeader) - sizeof(RUInt32BE))};
   RUInt16BE fVersion{4};

   RUInt32BE fByteCountNamed{0x40000000 |
      (sizeof(RUInt16BE) + sizeof(RTFObject) + 13)};
   RUInt16BE fVersionNamed{1};
   RTFObject fObjectNamed{0x02000000 | 0x01000000};
   char fLName = 11;
   char fName[11]{ 'f', 'S', 'e', 'e', 'k', 'H', 'e', 'a', 'd', 'e', 'r' };
   char fLTitle = 0;

   RUInt32BE fType{14};
   RUInt32BE fSize{8};
   RUInt32BE fArrLength{0};
   RUInt32BE fArrDim{0};
   char fMaxIndex[20]{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
   char fLTypeName = 13;
   char fTypeName[13]{ 'u', 'n', 's', 'i', 'g', 'n', 'e', 'd', ' ', 'l', 'o', 'n', 'g' };
};

/// Streamer info for data member RNTuple::fNbytesHeader
struct RTFStreamerElementNBytesHeader {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerElementNBytesHeader) - sizeof(RUInt32BE))};
   RUInt16BE fVersion{4};

   RUInt32BE fByteCountNamed{0x40000000 |
      (sizeof(RUInt16BE) + sizeof(RTFObject) + 15)};
   RUInt16BE fVersionNamed{1};
   RTFObject fObjectNamed{0x02000000 | 0x01000000};
   char fLName = 13;
   char fName[13]{ 'f', 'N', 'B', 'y', 't', 'e', 's', 'H', 'e', 'a', 'd', 'e', 'r' };
   char fLTitle = 0;

   RUInt32BE fType{13};
   RUInt32BE fSize{4};
   RUInt32BE fArrLength{0};
   RUInt32BE fArrDim{0};
   char fMaxIndex[20]{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
   char fLTypeName = 12;
   char fTypeName[12]{ 'u', 'n', 's', 'i', 'g', 'n', 'e', 'd', ' ', 'i', 'n', 't' };
};

/// Streamer info for data member RNTuple::fSeekFooter
struct RTFStreamerElementSeekFooter {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerElementSeekFooter) - sizeof(RUInt32BE))};
   RUInt16BE fVersion{4};

   RUInt32BE fByteCountNamed{0x40000000 |
      (sizeof(RUInt16BE) + sizeof(RTFObject) + 13)};
   RUInt16BE fVersionNamed{1};
   RTFObject fObjectNamed{0x02000000 | 0x01000000};
   char fLName = 11;
   char fName[11]{ 'f', 'S', 'e', 'e', 'k', 'F', 'o', 'o', 't', 'e', 'r' };
   char fLTitle = 0;

   RUInt32BE fType{14};
   RUInt32BE fSize{8};
   RUInt32BE fArrLength{0};
   RUInt32BE fArrDim{0};
   char fMaxIndex[20]{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
   char fLTypeName = 13;
   char fTypeName[13]{ 'u', 'n', 's', 'i', 'g', 'n', 'e', 'd', ' ', 'l', 'o', 'n', 'g' };
};

/// Streamer info for data member RNTuple::fNbytesFooter
struct RTFStreamerElementNBytesFooter {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerElementNBytesFooter) - sizeof(RUInt32BE))};
   RUInt16BE fVersion{4};

   RUInt32BE fByteCountNamed{0x40000000 |
      (sizeof(RUInt16BE) + sizeof(RTFObject) + 15)};
   RUInt16BE fVersionNamed{1};
   RTFObject fObjectNamed{0x02000000 | 0x01000000};
   char fLName = 13;
   char fName[13]{ 'f', 'N', 'B', 'y', 't', 'e', 's', 'F', 'o', 'o', 't', 'e', 'r' };
   char fLTitle = 0;

   RUInt32BE fType{13};
   RUInt32BE fSize{4};
   RUInt32BE fArrLength{0};
   RUInt32BE fArrDim{0};
   char fMaxIndex[20]{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
   char fLTypeName = 12;
   char fTypeName[12]{ 'u', 'n', 's', 'i', 'g', 'n', 'e', 'd', ' ', 'i', 'n', 't' };
};

/// Streamer info for data member RNTuple::fReserved
struct RTFStreamerElementReserved {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerElementReserved) - sizeof(RUInt32BE))};
   RUInt16BE fVersion{4};

   RUInt32BE fByteCountNamed{0x40000000 |
      (sizeof(RUInt16BE) + sizeof(RTFObject) + 11)};
   RUInt16BE fVersionNamed{1};
   RTFObject fObjectNamed{0x02000000 | 0x01000000};
   char fLName = 9;
   char fName[9]{ 'f', 'R', 'e', 's', 'e', 'r', 'v', 'e', 'd' };
   char fLTitle = 0;

   RUInt32BE fType{14};
   RUInt32BE fSize{8};
   RUInt32BE fArrLength{0};
   RUInt32BE fArrDim{0};
   char fMaxIndex[20]{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
   char fLTypeName = 13;
   char fTypeName[13]{ 'u', 'n', 's', 'i', 'g', 'n', 'e', 'd', ' ', 'l', 'o', 'n', 'g' };
};

/// Streamer info frame for data member RNTuple::fSeekHeader
struct RTFStreamerSeekHeader {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerSeekHeader) - sizeof(RUInt32BE))};
   RUInt32BE fNewClassTag{0xffffffff};
   char fClassName[19]{'T', 'S', 't', 'r', 'e', 'a', 'm', 'e', 'r', 'B', 'a', 's', 'i', 'c', 'T', 'y', 'p', 'e', '\0'};
   RUInt32BE fByteCountRemaining{0x40000000 |
      (sizeof(RTFStreamerSeekHeader) - 2 * sizeof(RUInt32BE) - 19 /* strlen(fClassName) + 1 */ - sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementSeekHeader fStreamerElementSeekHeader;
};

/// Streamer info frame for data member RNTuple::fNbytesHeader
struct RTFStreamerNBytesHeader {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerNBytesHeader) - sizeof(RUInt32BE))};
   RUInt32BE fClassTag{0x80000000};  // Fix-up after construction, or'd with 0x80000000
   RUInt32BE fByteCountRemaining{0x40000000 | (sizeof(RTFStreamerNBytesHeader) - 3 * sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementNBytesHeader fStreamerElementNBytesHeader;
};

/// Streamer info frame for data member RNTuple::fSeekFooter
struct RTFStreamerSeekFooter {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerSeekFooter) - sizeof(RUInt32BE))};
   RUInt32BE fClassTag{0x80000000};  // Fix-up after construction, or'd with 0x80000000
   RUInt32BE fByteCountRemaining{0x40000000 | (sizeof(RTFStreamerSeekFooter) - 3 * sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementSeekFooter fStreamerElementSeekFooter;
};

/// Streamer info frame for data member RNTuple::fNBytesFooter
struct RTFStreamerNBytesFooter {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerNBytesFooter) - sizeof(RUInt32BE))};
   RUInt32BE fClassTag{0x80000000};  // Fix-up after construction, or'd with 0x80000000
   RUInt32BE fByteCountRemaining{0x40000000 | (sizeof(RTFStreamerNBytesFooter) - 3 * sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementNBytesFooter fStreamerElementNBytesFooter;
};

/// Streamer info frame for data member RNTuple::fReserved
struct RTFStreamerReserved {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerReserved) - sizeof(RUInt32BE))};
   RUInt32BE fClassTag{0x80000000};  // Fix-up after construction, or'd with 0x80000000
   RUInt32BE fByteCountRemaining{0x40000000 | (sizeof(RTFStreamerReserved) - 3 * sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementReserved fStreamerElementReserved;
};

/// Streamer info for class RNTuple
struct RTFStreamerInfoObject {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerInfoObject) - sizeof(fByteCount))};
   RUInt32BE fNewClassTag{0xffffffff};
   char fClassName[14]{ 'T', 'S', 't', 'r', 'e', 'a', 'm', 'e', 'r', 'I', 'n', 'f', 'o', '\0' };
   RUInt32BE fByteCountRemaining{0x40000000 |
      (sizeof(RTFStreamerInfoObject) - 2 * sizeof(RUInt32BE) - 14 - sizeof(RUInt32BE))};
   RUInt16BE fVersion{9};

   RUInt32BE fByteCountNamed{0x40000000 |
      (sizeof(RUInt16BE) + sizeof(RTFObject) + 29 /* strlen("ROOT::Experimental::RNTuple") + 2 */)};
   RUInt16BE fVersionNamed{1};
   RTFObject fObjectNamed{0x02000000 | 0x01000000 | 0x00010000};
   char fLName = 27;
   char fName[27]{ 'R', 'O', 'O', 'T', ':', ':',
      'E', 'x', 'p', 'e', 'r', 'i', 'm', 'e', 'n', 't', 'a', 'l', ':', ':',
      'R', 'N', 'T', 'u', 'p', 'l', 'e'};
   char fLTitle = 0;

   RInt32BE fChecksum{ChecksumRNTupleClass()};
   RUInt32BE fVersionRNTuple{1};

   RUInt32BE fByteCountObjArr{0x40000000 |
      (sizeof(RUInt32BE) + 10 /* strlen(TObjArray) + 1 */ + sizeof(RUInt32BE) +
       sizeof(RUInt16BE) + sizeof(RTFObject) + 1 + 2*sizeof(RUInt32BE) +
       sizeof(fStreamers))};
   RUInt32BE fNewClassTagObjArray{0xffffffff};
   char fClassNameObjArray[10]{'T', 'O', 'b', 'j', 'A', 'r', 'r', 'a', 'y', '\0'};
   RUInt32BE fByteCountObjArrRemaining{0x40000000 |
      (sizeof(RUInt16BE) + sizeof(RTFObject) + 1 + 2*sizeof(RUInt32BE) +
       sizeof(fStreamers))};
   RUInt16BE fVersionObjArr{3};
   RTFObject fObjectObjArr{0x02000000};
   char fNameObjArr{0};

   RUInt32BE fNObjects{5};
   RUInt32BE fLowerBound{0};

   struct {
      RTFStreamerSeekHeader fStreamerSeekHeader;
      RTFStreamerNBytesHeader fStreamerNBytesHeader;
      RTFStreamerSeekFooter fStreamerSeekFooter;
      RTFStreamerNBytesFooter fStreamerNBytesFooter;
      RTFStreamerReserved fStreamerReserved;
   } fStreamers;
};

/// The list of streamer info objects, for a new ntuple contains only the RNTuple class
struct RTFStreamerInfoList {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerInfoList) - sizeof(fByteCount))};
   RUInt16BE fVersion{5};
   RTFObject fObject{0x02000000};
   char fName{0};
   RUInt32BE fNObjects{1};
   RTFStreamerInfoObject fStreamerInfo;
   char fEnd{0};

   std::uint32_t GetSize() const { return sizeof(RTFStreamerInfoList); }
};

/// The header of the directory key index
struct RTFKeyList {
   RUInt32BE fNKeys;
   std::uint32_t GetSize() const { return sizeof(RTFKeyList); }
   explicit RTFKeyList(std::uint32_t nKeys) : fNKeys(nKeys) {}
};

/// A streamed TFile object
struct RTFFile {
   char fModified{0};
   char fWritable{1};
   RTFDatetime fDateC;
   RTFDatetime fDateM;
   RUInt32BE fNBytesKeys{0};
   RUInt32BE fNBytesName{0};
   RUInt32BE fSeekDir{100};
   RUInt32BE fSeekParent{0};
   RUInt32BE fSeekKeys{0};
   RTFFile() = default;
};

/// A streamed RNTuple class
struct RTFNTuple {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFNTuple) - sizeof(fByteCount))};
   RUInt16BE fVersion{0};
   RInt32BE fChecksum{ChecksumRNTupleClass()};
   RUInt64BE fSeekHeader{0};
   RUInt32BE fNBytesHeader{0};
   RUInt64BE fSeekFooter{0};
   RUInt32BE fNBytesFooter{0};
   RUInt64BE fReserved{0};
   std::uint32_t GetSize() const { return sizeof(RTFNTuple); }
};
#pragma pack(pop)

}

namespace ROOT {
namespace Experimental {
namespace Internal {
/// On dataset commit, the file header and the RNTuple object need to be updated
struct RTFileControlBlock {
   RTFHeader fHeader;
   RTFNTuple fNTuple;
   std::uint32_t fSeekNTuple{0};
};
}
}
}

ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, std::string_view path,
   const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options)
   , fMetrics("RPageSinkRoot")
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
   , fZipBuffer(std::make_unique<std::array<char, kMaxRecordSize>>())
{
   R__WARNING_HERE("NTuple") << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";

   std::string strPath = std::string(path);
   fFile = fopen(strPath.c_str(), "wb");
   R__ASSERT(fFile);
   size_t idxDirSep = strPath.find_last_of("\\/");
   if (idxDirSep != std::string::npos) {
      strPath.erase(0, idxDirSep + 1);
   }
   fFileName = strPath;
}

ROOT::Experimental::Detail::RPageSinkFile::~RPageSinkFile()
{
   if (fFile)
      fclose(fFile);
}

void ROOT::Experimental::Detail::RPageSinkFile::Write(void *from, size_t size, std::int64_t offset)
{
   R__ASSERT(fFile);
   size_t retval;
   if ((offset >= 0) && (static_cast<std::uint64_t>(offset) != fFilePos)) {
      retval = fseek(fFile, offset, SEEK_SET);
      R__ASSERT(retval == 0);
      fFilePos = offset;
   }
   retval = fwrite(from, 1, size, fFile);
   R__ASSERT(retval == size);
   fFilePos += size;
}

std::uint64_t ROOT::Experimental::Detail::RPageSinkFile::WriteKey(
   void *buffer, std::size_t nbytes, std::int64_t offset,
   std::uint64_t directoryOffset, int compression,
   const std::string &className,
   const std::string &objectName,
   const std::string &title)
{
   if (offset < 0)
      offset = fFilePos;
   RTFString strClass{className};
   RTFString strObject{objectName};
   RTFString strTitle{title};

   int zipBytes = nbytes;
   if (compression != 0) {
      R__ASSERT(nbytes <= kMaxRecordSize);
      auto level = compression % 100;
      auto algorithm = static_cast<ROOT::RCompressionSetting::EAlgorithm::EValues>(compression / 100);
      int szZipBuffer = kMaxRecordSize;
      int szSource = nbytes;
      char *source = reinterpret_cast<char *>(buffer);
      R__zipMultipleAlgorithm(level, &szSource, source, &szZipBuffer, fZipBuffer->data(), &zipBytes, algorithm);
      if ((zipBytes > 0) && (zipBytes < szSource)) {
         buffer = reinterpret_cast<unsigned char *>(fZipBuffer->data());
      } else {
         zipBytes = szSource;
      }
   }
   RTFKey key(offset, directoryOffset, strClass, strObject, strTitle, nbytes, zipBytes);
   Write(&key, key.fKeyHeaderSize, offset);
   Write(&strClass, strClass.GetSize());
   Write(&strObject, strObject.GetSize());
   Write(&strTitle, strTitle.GetSize());
   auto offsetData = fFilePos;
   Write(buffer, zipBytes);
   return offsetData;
}

void ROOT::Experimental::Detail::RPageSinkFile::DoCreate(const RNTupleModel & /* model */)
{
   RTFString strTFile{"TFile"};
   RTFString strFileName{fFileName};
   RTFString strTList{"TList"};
   RTFString strStreamerInfo{"StreamerInfo"};
   RTFString strStreamerTitle{"Doubly linked list"};
   RTFString strRNTupleClass{"ROOT::Experimental::RNTuple"};
   RTFString strRNTupleName{fNTupleName};
   RTFString strEmpty;

   fControlBlock = std::make_unique<ROOT::Experimental::Internal::RTFileControlBlock>();
   fControlBlock->fHeader = RTFHeader(fOptions.GetCompression());

   // First record of the file: the TFile object at offset 100
   RTFFile fileRoot;
   RTFKey keyRoot(100, 0, strTFile, strFileName, strEmpty,
                  sizeof(fileRoot) + strFileName.GetSize() + strEmpty.GetSize());
   std::uint32_t nbytesName = keyRoot.fKeyLen + strFileName.GetSize() + 1;
   fileRoot.fNBytesName = nbytesName;
   fControlBlock->fHeader.fInfoShort.fNbytesName = nbytesName;

   // Second record: the compressed StreamerInfo with the description of the RNTuple class
   fControlBlock->fHeader.fInfoShort.fSeekInfo = 100 + keyRoot.GetSize();
   RTFKey keyStreamerInfo(
      fControlBlock->fHeader.fInfoShort.fSeekInfo, 100, strTList, strStreamerInfo, strStreamerTitle, 0);
   RTFStreamerInfoList streamerInfo;
   auto classTagOffset = keyStreamerInfo.fKeyLen +
      offsetof(struct RTFStreamerInfoList, fStreamerInfo) +
      offsetof(struct RTFStreamerInfoObject, fStreamers) +
      offsetof(struct RTFStreamerSeekHeader, fNewClassTag) + 2;
   streamerInfo.fStreamerInfo.fStreamers.fStreamerNBytesHeader.fClassTag = 0x80000000 | classTagOffset;
   streamerInfo.fStreamerInfo.fStreamers.fStreamerSeekFooter.fClassTag = 0x80000000 | classTagOffset;
   streamerInfo.fStreamerInfo.fStreamers.fStreamerNBytesFooter.fClassTag = 0x80000000 | classTagOffset;
   streamerInfo.fStreamerInfo.fStreamers.fStreamerReserved.fClassTag = 0x80000000 | classTagOffset;
   WriteKey(&streamerInfo, streamerInfo.GetSize(), fControlBlock->fHeader.fInfoShort.fSeekInfo, 100, 1,
            "TList", "StreamerInfo", "Doubly linked list");
   fControlBlock->fHeader.fInfoShort.fNbytesInfo = fFilePos - fControlBlock->fHeader.fInfoShort.fSeekInfo;

   // Reserve the space for the RNTuple record, which will be written on commit
   fControlBlock->fSeekNTuple = fFilePos;
   RTFKey keyRNTuple(fControlBlock->fSeekNTuple, 100, strRNTupleClass, strRNTupleName, strEmpty,
                     fControlBlock->fNTuple.GetSize());

   // The key index of the root TFile object, containing for the time being only the RNTuple key
   fileRoot.fSeekKeys = fFilePos + keyRNTuple.GetSize();
   RTFKeyList keyList{1};
   RTFKey keyKeyList(fileRoot.fSeekKeys, 100, strEmpty, strEmpty, strEmpty, keyList.GetSize() + keyRNTuple.fKeyLen);
   Write(&keyKeyList, keyKeyList.fKeyHeaderSize, fileRoot.fSeekKeys);
   Write(&strEmpty, strEmpty.GetSize());
   Write(&strEmpty, strEmpty.GetSize());
   Write(&strEmpty, strEmpty.GetSize());
   Write(&keyList, keyList.GetSize());
   Write(&keyRNTuple, keyRNTuple.fKeyHeaderSize);
   Write(&strRNTupleClass, strRNTupleClass.GetSize());
   Write(&strRNTupleName, strRNTupleName.GetSize());
   Write(&strEmpty, strEmpty.GetSize());
   fileRoot.fNBytesKeys = fFilePos - fileRoot.fSeekKeys;

   Write(&keyRoot, keyRoot.fKeyHeaderSize, 100);
   Write(&strTFile, strTFile.GetSize());
   Write(&strFileName, strFileName.GetSize());
   Write(&strEmpty, strEmpty.GetSize());
   Write(&strFileName, strFileName.GetSize());
   Write(&strEmpty, strEmpty.GetSize());
   Write(&fileRoot, sizeof(fileRoot));

   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szHeader = descriptor.SerializeHeader(nullptr);
   auto buffer = new unsigned char[szHeader];
   descriptor.SerializeHeader(buffer);
   fControlBlock->fNTuple.fSeekHeader = fileRoot.fSeekKeys + fileRoot.fNBytesKeys;
   WriteKey(buffer, szHeader, fControlBlock->fNTuple.fSeekHeader, fControlBlock->fNTuple.fSeekHeader,
            fOptions.GetCompression());
   fControlBlock->fNTuple.fNBytesHeader = fFilePos - fControlBlock->fNTuple.fSeekHeader;
   delete[] buffer;

   fClusterStart = fFilePos;
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkFile::DoCommitPage(ColumnHandle_t columnHandle, const RPage &page)
{
   unsigned char *buffer = reinterpret_cast<unsigned char *>(page.GetBuffer());
   auto packedBytes = page.GetSize();
   auto element = columnHandle.fColumn->GetElement();
   const auto isMappable = element->IsMappable();

   if (!isMappable) {
      packedBytes = (page.GetNElements() * element->GetBitsOnStorage() + 7) / 8;
      buffer = new unsigned char[packedBytes];
      element->Pack(buffer, page.GetBuffer(), page.GetNElements());
   }

   auto offsetData = WriteKey(buffer, packedBytes, -1, fFilePos, fOptions.GetCompression());

   if (!isMappable) {
      delete[] buffer;
   }

   RClusterDescriptor::RLocator result;
   result.fPosition = offsetData;
   result.fBytesOnStorage = fFilePos - offsetData;
   return result;
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkFile::DoCommitCluster(ROOT::Experimental::NTupleSize_t /* nEntries */)
{
   RClusterDescriptor::RLocator result;
   result.fPosition = fClusterStart;
   result.fBytesOnStorage = fFilePos - fClusterStart;
   fClusterStart = fFilePos;
   return result;
}

void ROOT::Experimental::Detail::RPageSinkFile::DoCommitDataset()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szFooter = descriptor.SerializeFooter(nullptr);
   auto buffer = new unsigned char[szFooter];
   descriptor.SerializeFooter(buffer);
   fControlBlock->fNTuple.fSeekFooter = fFilePos;
   WriteKey(buffer, szFooter, -1, fFilePos, fOptions.GetCompression());
   fControlBlock->fNTuple.fNBytesFooter = fFilePos - fControlBlock->fNTuple.fSeekFooter;
   fControlBlock->fHeader.fInfoShort.fSeekFree = fFilePos;
   delete[] buffer;

   WriteKey(&fControlBlock->fNTuple, fControlBlock->fNTuple.GetSize(), fControlBlock->fSeekNTuple,
            100, 0, "ROOT::Experimental::RNTuple", fNTupleName, "");

   RTFString strEmpty;
   RTFFreeEntry freeEntry(0, 2000000000);
   RTFKey keyFreeList(fControlBlock->fHeader.fInfoShort.fSeekFree, 100,
                      strEmpty, strEmpty, strEmpty, freeEntry.GetSize());
   freeEntry.fInfoShort.fFirst = fControlBlock->fHeader.fInfoShort.fSeekFree + keyFreeList.GetSize();
   WriteKey(&freeEntry, freeEntry.GetSize(), fControlBlock->fHeader.fInfoShort.fSeekFree, 100, 0, "", "", "");
   fControlBlock->fHeader.fInfoShort.fNbytesFree = fFilePos - fControlBlock->fHeader.fInfoShort.fSeekFree;
   fControlBlock->fHeader.SetEnd(fFilePos);

   Write(&fControlBlock->fHeader, fControlBlock->fHeader.GetSize(), 0);
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkFile::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   if (nElements == 0)
      nElements = kDefaultElementsPerPage;
   auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
   return fPageAllocator->NewPage(columnHandle.fId, elementSize, nElements);
}

void ROOT::Experimental::Detail::RPageSinkFile::ReleasePage(RPage &page)
{
   fPageAllocator->DeletePage(page);
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageAllocatorFile::NewPage(
   ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements)
{
   RPage newPage(columnId, mem, elementSize * nElements, elementSize);
   newPage.TryGrow(nElements);
   return newPage;
}

void ROOT::Experimental::Detail::RPageAllocatorFile::DeletePage(const RPage& page)
{
   if (page.IsNull())
      return;
   free(page.GetBuffer());
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPageSourceFile::RPageSourceFile(std::string_view ntupleName,
   const RNTupleReadOptions &options)
   : RPageSource(ntupleName, options)
   , fMetrics("RPageSourceFile")
   , fPageAllocator(std::make_unique<RPageAllocatorKey>())
   , fPagePool(std::make_shared<RPagePool>())
   , fUnzipBuffer(std::make_unique<std::array<unsigned char, kMaxPageSize>>())
{
}

ROOT::Experimental::Detail::RPageSourceFile::RPageSourceFile(std::string_view ntupleName, std::string_view path,
   const RNTupleReadOptions &options)
   : RPageSourceFile(ntupleName, options)
{
   fFile = std::unique_ptr<RRawFile>(RRawFile::Create(path));
   R__ASSERT(fFile);
   R__ASSERT(fFile->GetFeatures() & RRawFile::kFeatureHasSize);
}


ROOT::Experimental::Detail::RPageSourceFile::~RPageSourceFile()
{
}


void ROOT::Experimental::Detail::RPageSourceFile::Read(void *buffer, std::size_t nbytes, std::uint64_t offset)
{
   auto nread = fFile->ReadAt(buffer, nbytes, offset);
   R__ASSERT(nread == nbytes);
}


ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceFile::DoAttach()
{
   RTFHeader fileHeader;
   Read(&fileHeader, sizeof(fileHeader), 0);

   RTFKey key;
   RTFString name;
   Read(&key, sizeof(key), fileHeader.fBEGIN);
   auto offset = fileHeader.fBEGIN + key.fKeyLen;
   Read(&name, 1, offset);
   offset += name.GetSize();
   Read(&name, 1, offset);
   offset += name.GetSize();
   RTFFile file;
   Read(&file, sizeof(file), offset);

   RUInt32BE nKeys;
   Read(&key, sizeof(key), file.fSeekKeys);
   offset = file.fSeekKeys + key.fKeyLen;
   Read(&nKeys, sizeof(nKeys), offset);
   offset += sizeof(nKeys);
   for (unsigned int i = 0; i < nKeys; ++i) {
      Read(&key, sizeof(key), offset);
      auto offsetNextKey = offset + key.fKeyLen;

      offset += key.GetHeaderSize();
      Read(&name, 1, offset);
      offset += name.GetSize();
      Read(&name, 1, offset);
      Read(&name, name.GetSize(), offset);
      if (std::string(name.fData, name.fLName) == fNTupleName)
         break;
      offset = offsetNextKey;
   }

   Read(&key, sizeof(key), key.fInfoShort.fSeekKey);
   offset = key.fInfoShort.fSeekKey + key.fKeyLen;
   RTFNTuple ntuple;
   Read(&ntuple, sizeof(ntuple), offset);

   unsigned char *keyHeader = new unsigned char[ntuple.fNBytesHeader];
   Read(keyHeader, ntuple.fNBytesHeader, ntuple.fSeekHeader);
   memcpy(&key, keyHeader, sizeof(key));
   unsigned char *header = new unsigned char[key.fObjLen];
   int szUnzipBuffer = key.fObjLen;
   int szSource = key.fNbytes - key.fKeyLen;
   int unzipBytes = 0;
   R__unzip(&szSource, keyHeader + key.fKeyLen, &szUnzipBuffer, header, &unzipBytes);
   delete[] keyHeader;

   unsigned char *keyFooter = new unsigned char[ntuple.fNBytesFooter];
   Read(keyFooter, ntuple.fNBytesFooter, ntuple.fSeekFooter);
   memcpy(&key, keyFooter, sizeof(key));
   unsigned char *footer = new unsigned char[key.fObjLen];
   szUnzipBuffer = key.fObjLen;
   szSource = key.fNbytes - key.fKeyLen;
   unzipBytes = 0;
   R__unzip(&szSource, keyFooter + key.fKeyLen, &szUnzipBuffer, footer, &unzipBytes);
   delete[] keyFooter;

//   unsigned char postscript[RNTupleDescriptor::kNBytesPostscript];
//   auto fileSize = fFile->GetSize();
//   R__ASSERT(fileSize != RRawFile::kUnknownFileSize);
//   R__ASSERT(fileSize >= RNTupleDescriptor::kNBytesPostscript);
//   auto offset = fileSize - RNTupleDescriptor::kNBytesPostscript;
//   Read(postscript, RNTupleDescriptor::kNBytesPostscript, offset);
//
//   std::uint32_t szHeader;
//   std::uint32_t szFooter;
//   RNTupleDescriptor::LocateMetadata(postscript, szHeader, szFooter);
//   R__ASSERT(fileSize >= szHeader + szFooter);
//
//   unsigned char *header = new unsigned char[szHeader];
//   unsigned char *footer = new unsigned char[szFooter];
//   Read(header, szHeader, 0);
//   Read(footer, szFooter, fileSize - szFooter);

   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetFromHeader(header);
   descBuilder.AddClustersFromFooter(footer);
   delete[] header;
   delete[] footer;

   return descBuilder.MoveDescriptor();
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceFile::PopulatePageFromCluster(
   ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor, ClusterSize_t::ValueType clusterIndex)
{
   auto columnId = columnHandle.fId;
   auto clusterId = clusterDescriptor.GetId();
   const auto &pageRange = clusterDescriptor.GetPageRange(columnId);

   // TODO(jblomer): binary search
   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   decltype(clusterIndex) firstInPage = 0;
   for (const auto &pi : pageRange.fPageInfos) {
      if (firstInPage + pi.fNElements > clusterIndex) {
         pageInfo = pi;
         break;
      }
      firstInPage += pi.fNElements;
   }
   R__ASSERT(firstInPage <= clusterIndex);
   R__ASSERT((firstInPage + pageInfo.fNElements) > clusterIndex);

   auto element = columnHandle.fColumn->GetElement();
   auto elementSize = element->GetSize();

   auto pageSize = pageInfo.fLocator.fBytesOnStorage;
   void *pageBuffer = malloc(std::max(pageSize, static_cast<std::uint32_t>(elementSize * pageInfo.fNElements)));
   R__ASSERT(pageBuffer);
   Read(pageBuffer, pageSize, pageInfo.fLocator.fPosition);

   auto bytesOnStorage = (element->GetBitsOnStorage() * pageInfo.fNElements + 7) / 8;
   if (pageSize != bytesOnStorage) {
      R__ASSERT(bytesOnStorage <= kMaxPageSize);
      // We do have the unzip information in the column range, but here we simply use the value from
      // the R__zip header
      int szUnzipBuffer = kMaxPageSize;
      int szSource = pageSize;
      unsigned char *source = reinterpret_cast<unsigned char *>(pageBuffer);
      int unzipBytes = 0;
      R__unzip(&szSource, source, &szUnzipBuffer, fUnzipBuffer->data(), &unzipBytes);
      R__ASSERT(unzipBytes > static_cast<int>(pageSize));
      memcpy(pageBuffer, fUnzipBuffer->data(), unzipBytes);
      pageSize = unzipBytes;
   }

   if (!element->IsMappable()) {
      pageSize = elementSize * pageInfo.fNElements;
      auto unpackedBuffer = reinterpret_cast<unsigned char *>(malloc(pageSize));
      R__ASSERT(unpackedBuffer != nullptr);
      element->Unpack(unpackedBuffer, pageBuffer, pageInfo.fNElements);
      free(pageBuffer);
      pageBuffer = unpackedBuffer;
   }

   auto indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   auto newPage = fPageAllocator->NewPage(columnId, pageBuffer, elementSize, pageInfo.fNElements);
   newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
   fPagePool->RegisterPage(newPage,
      RPageDeleter([](const RPage &page, void */*userData*/)
      {
         RPageAllocatorFile::DeletePage(page);
      }, nullptr));
   return newPage;
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceFile::PopulatePage(
   ColumnHandle_t columnHandle, NTupleSize_t globalIndex)
{
   auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, globalIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   auto clusterId = fDescriptor.FindClusterId(columnId, globalIndex);
   R__ASSERT(clusterId != kInvalidDescriptorId);
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   auto selfOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   R__ASSERT(selfOffset <= globalIndex);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, globalIndex - selfOffset);
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceFile::PopulatePage(
   ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex)
{
   auto clusterId = clusterIndex.GetClusterId();
   auto index = clusterIndex.GetIndex();
   auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, clusterIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   R__ASSERT(clusterId != kInvalidDescriptorId);
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, index);
}

void ROOT::Experimental::Detail::RPageSourceFile::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource> ROOT::Experimental::Detail::RPageSourceFile::Clone() const
{
   auto clone = new RPageSourceFile(fNTupleName, fOptions);
   clone->fFile = fFile->Clone();
   return std::unique_ptr<RPageSourceFile>(clone);
}
