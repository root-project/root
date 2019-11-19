/// \file RPageStorage.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
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
#include <ROOT/RPageStorageRoot.hxx>

#include <Compression.h>
#include <RVersion.h>
#include <TKey.h>

#include <chrono>
#include <cstddef>  // for offsetof()
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <utility>

namespace {

static constexpr const char* kKeySeparator = "_";
static constexpr const char* kKeyNTupleFooter = "NTPLF";
static constexpr const char* kKeyNTupleHeader = "NTPLH";
static constexpr const char* kKeyPagePayload = "NTPLP";

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

constexpr std::int32_t ChecksumString(std::int32_t id, const char *str) {
   auto len = strlen(str);
   for (unsigned i = 0; i < len; i++)
      id = id *3 + str[i];
   return id;
}

constexpr std::int32_t ChecksumRNTupleClass() {
   std::int32_t id = 0;
   id = ChecksumString(id, "RNTuple");
   id = ChecksumString(id, "fSeekHeader");
   id = ChecksumString(id, "unsigned long");
   id = ChecksumString(id, "fNBytesHeader");
   id = ChecksumString(id, "unsigned int");
   id = ChecksumString(id, "fSeekFooter");
   id = ChecksumString(id, "unsigned long");
   id = ChecksumString(id, "fNBytesFooter");
   id = ChecksumString(id, "unsigned int");
   return id;
}


#pragma pack(push, 1)
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
};

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
         RUInt32BE fNbytesName{56};
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
         RUInt32BE fNbytesName{56};
         unsigned char fUnits{8};
         RUInt32BE fCompress{0};
         RUInt64BE fSeekInfo{0};
         RUInt32BE fNbytesInfo{0};
      } fInfoLong;
   };

   RTFHeader() : fInfoShort() {}
   RTFHeader(int compression, const RTFKey &keyFreeList, const RTFKey &keyStreamerInfo) : fInfoShort() {
      fInfoShort.fCompress = compression;
      fInfoShort.fSeekFree = keyFreeList.fInfoShort.fSeekKey;
      fInfoShort.fNbytesFree = keyFreeList.GetSize();
      fInfoShort.fSeekInfo = keyStreamerInfo.fInfoShort.fSeekKey;
      fInfoShort.fNbytesInfo = keyStreamerInfo.GetSize();
      SetEnd(fInfoShort.fSeekFree + fInfoShort.fNbytesFree);
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

struct RTFObject {
   RUInt16BE fVersion{1};
   RUInt32BE fUniqueID{0};  // unused
   RUInt32BE fBits;
   explicit RTFObject(std::uint32_t bits) : fBits(bits) {}
};

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

struct RTFStreamerSeekHeader {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerSeekHeader) - sizeof(RUInt32BE))};
   RUInt32BE fNewClassTag{0xffffffff};
   char fClassName[19]{'T', 'S', 't', 'r', 'e', 'a', 'm', 'e', 'r', 'B', 'a', 's', 'i', 'c', 'T', 'y', 'p', 'e', '\0'};
   RUInt32BE fByteCountRemaining{0x40000000 |
      (sizeof(RTFStreamerSeekHeader) - 2 * sizeof(RUInt32BE) - 19 /* strlen(fClassName) + 1 */ - sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementSeekHeader fStreamerElementSeekHeader;
};

struct RTFStreamerNBytesHeader {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerNBytesHeader) - sizeof(RUInt32BE))};
   RUInt32BE fClassTag{0x80000000};  // Fix-up after construction, or'd with 0x80000000
   RUInt32BE fByteCountRemaining{0x40000000 | (sizeof(RTFStreamerNBytesHeader) - 3 * sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementNBytesHeader fStreamerElementNBytesHeader;
};

struct RTFStreamerSeekFooter {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerSeekFooter) - sizeof(RUInt32BE))};
   RUInt32BE fClassTag{0x80000000};  // Fix-up after construction, or'd with 0x80000000
   RUInt32BE fByteCountRemaining{0x40000000 | (sizeof(RTFStreamerSeekFooter) - 3 * sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementSeekFooter fStreamerElementSeekFooter;
};

struct RTFStreamerNBytesFooter {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerNBytesFooter) - sizeof(RUInt32BE))};
   RUInt32BE fClassTag{0x80000000};  // Fix-up after construction, or'd with 0x80000000
   RUInt32BE fByteCountRemaining{0x40000000 | (sizeof(RTFStreamerNBytesFooter) - 3 * sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementNBytesFooter fStreamerElementNBytesFooter;
};

struct RTFStreamerReserved {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFStreamerReserved) - sizeof(RUInt32BE))};
   RUInt32BE fClassTag{0x80000000};  // Fix-up after construction, or'd with 0x80000000
   RUInt32BE fByteCountRemaining{0x40000000 | (sizeof(RTFStreamerReserved) - 3 * sizeof(RUInt32BE))};
   RUInt16BE fVersion{2};
   RTFStreamerElementReserved fStreamerElementReserved;
};

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

struct RTFKeyList {
   RUInt32BE fNKeys{0};
   std::uint32_t GetSize() const { return sizeof(RTFKeyList); }
};

struct RTFFile {
   char fModified{0};
   char fWritable{1};
   RTFDatetime fDateC;
   RTFDatetime fDateM;
   RUInt32BE fNBytesKeys{0};
   RUInt32BE fNBytesName{56};
   RUInt32BE fSeekDir{100};
   RUInt32BE fSeekParent{0};
   RUInt32BE fSeekKeys{0};
   RTFFile() = default;
};

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

ROOT::Experimental::Detail::RPageSinkRoot::RPageSinkRoot(std::string_view ntupleName, std::string_view path,
   const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options)
   , fMetrics("RPageSinkRoot")
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
{
   R__WARNING_HERE("NTuple") << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";

   fBinaryFile = fopen(std::string(path).c_str(), "wb");
   R__ASSERT(fBinaryFile);
   //fFile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "RECREATE"));
   //fFile->SetCompressionSettings(fOptions.GetCompression());
}

ROOT::Experimental::Detail::RPageSinkRoot::~RPageSinkRoot()
{
   if (fBinaryFile)
      fclose(fBinaryFile);
   if (fFile)
      fFile->Close();
}

std::uint64_t ROOT::Experimental::Detail::RPageSinkRoot::Write(void *from, size_t size, std::uint64_t offset)
{
   size_t retval = fseek(fBinaryFile, offset, SEEK_SET);
   R__ASSERT(retval == 0);
   retval = fwrite(from, 1, size, fBinaryFile);
   R__ASSERT(retval == size);
   return offset + size;
}

void ROOT::Experimental::Detail::RPageSinkRoot::DoCreate(const RNTupleModel & /* model */)
{
   RTFString strTFile{"TFile"};
   RTFString strFileName{"empty.root"};
   RTFString strTList{"TList"};
   RTFString strStreamerInfo{"StreamerInfo"};
   RTFString strStreamerTitle{"Doubly linked list"};
   RTFString strRNTuple{"ROOT::Experimental::RNTuple"};
   RTFString strMyTuple{"MyTuple"};
   RTFString strEmpty;

   RTFFile fileRoot;
   RTFKey keyRoot(100, 0, strTFile, strFileName, strEmpty,
                  sizeof(fileRoot) + strFileName.GetSize() + strEmpty.GetSize());

   auto seekStreamerInfo = 100 + keyRoot.GetSize();
   RTFStreamerInfoList streamerInfo;
   RTFKey keyStreamerInfo(seekStreamerInfo, 100, strTList, strStreamerInfo, strStreamerTitle, streamerInfo.GetSize());
   auto classTagOffset = keyStreamerInfo.fKeyLen +
      offsetof(struct RTFStreamerInfoList, fStreamerInfo) +
      offsetof(struct RTFStreamerInfoObject, fStreamers) +
      offsetof(struct RTFStreamerSeekHeader, fNewClassTag) + 2;
   streamerInfo.fStreamerInfo.fStreamers.fStreamerNBytesHeader.fClassTag = 0x80000000 | classTagOffset;
   streamerInfo.fStreamerInfo.fStreamers.fStreamerSeekFooter.fClassTag = 0x80000000 | classTagOffset;
   streamerInfo.fStreamerInfo.fStreamers.fStreamerNBytesFooter.fClassTag = 0x80000000 | classTagOffset;
   streamerInfo.fStreamerInfo.fStreamers.fStreamerReserved.fClassTag = 0x80000000 | classTagOffset;
   // TODO: compress, fix key

   auto seekRNTuple = seekStreamerInfo + keyStreamerInfo.GetSize();
   RTFNTuple ntuple;
   RTFKey keyRNTuple(seekRNTuple, 100, strRNTuple, strMyTuple, strEmpty, ntuple.GetSize());

   auto seekKeyList = seekRNTuple + keyRNTuple.GetSize();
   RTFKeyList keyList;
   keyList.fNKeys = 1;
   RTFKey keyKeyList(seekKeyList, 100, strEmpty, strEmpty, strEmpty, keyList.GetSize() + keyRNTuple.fKeyLen);
   fileRoot.fSeekKeys = seekKeyList;
   fileRoot.fNBytesKeys = keyKeyList.GetSize();

   auto seekFreeList = seekKeyList + keyKeyList.GetSize();
   RTFFreeEntry freeEntry(0, 2000000000);
   RTFKey keyFreeList(seekFreeList, 100, strEmpty, strEmpty, strEmpty, freeEntry.GetSize());
   freeEntry.fInfoShort.fFirst = seekFreeList + keyFreeList.GetSize();

   RTFHeader fileHeader(fOptions.GetCompression(), keyFreeList, keyStreamerInfo);
   Write(&fileHeader, fileHeader.GetSize(), 0);

   auto pos = Write(&keyRoot, keyRoot.fKeyHeaderSize, 100);
   pos = Write(&strTFile, strTFile.GetSize(), pos);
   pos = Write(&strFileName, strFileName.GetSize(), pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);
   pos = Write(&strFileName, strFileName.GetSize(), pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);
   pos = Write(&fileRoot, sizeof(fileRoot), pos);

   pos = Write(&keyStreamerInfo, keyStreamerInfo.fKeyHeaderSize, pos);
   pos = Write(&strTList, strTList.GetSize(), pos);
   pos = Write(&strStreamerInfo, strStreamerInfo.GetSize(), pos);
   pos = Write(&strStreamerTitle, strStreamerTitle.GetSize(), pos);
   pos = Write(&streamerInfo, streamerInfo.GetSize(), pos);

   pos = Write(&keyRNTuple, keyRNTuple.fKeyHeaderSize, pos);
   pos = Write(&strRNTuple, strRNTuple.GetSize(), pos);
   pos = Write(&strMyTuple, strMyTuple.GetSize(), pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);
   pos = Write(&ntuple, ntuple.GetSize(), pos);

   pos = Write(&keyKeyList, keyKeyList.fKeyHeaderSize, pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);
   pos = Write(&keyList, keyList.GetSize(), pos);
   pos = Write(&keyRNTuple, keyRNTuple.fKeyHeaderSize, pos);
   pos = Write(&strRNTuple, strRNTuple.GetSize(), pos);
   pos = Write(&strMyTuple, strMyTuple.GetSize(), pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);

   pos = Write(&keyFreeList, keyFreeList.fKeyHeaderSize, pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);
   pos = Write(&strEmpty, strEmpty.GetSize(), pos);
   pos = Write(&freeEntry, freeEntry.GetSize(), pos);

//   fDirectory = fFile->mkdir(fNTupleName.c_str());
//   // In TBrowser, use RNTupleBrowser(TDirectory *directory) in order to show the ntuple contents
//   fDirectory->SetBit(TDirectoryFile::kCustomBrowse);
//   fDirectory->SetTitle("ROOT::Experimental::Detail::RNTupleBrowser");
//
//   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
//   auto szHeader = descriptor.SerializeHeader(nullptr);
//   auto buffer = new unsigned char[szHeader];
//   descriptor.SerializeHeader(buffer);
//   ROOT::Experimental::Internal::RNTupleBlob blob(szHeader, buffer);
//   fDirectory->WriteObject(&blob, kKeyNTupleHeader);
//   delete[] buffer;
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkRoot::DoCommitPage(ColumnHandle_t columnHandle, const RPage &page)
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

   ROOT::Experimental::Internal::RNTupleBlob pagePayload(packedBytes, buffer);
   std::string keyName = std::string(kKeyPagePayload) +
      std::to_string(fLastClusterId) + kKeySeparator +
      std::to_string(fLastPageIdx);
   //fDirectory->WriteObject(&pagePayload, keyName.c_str());

   if (!isMappable) {
      delete[] buffer;
   }

   RClusterDescriptor::RLocator result;
   result.fPosition = fLastPageIdx++;
   result.fBytesOnStorage = packedBytes;
   return result;
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkRoot::DoCommitCluster(ROOT::Experimental::NTupleSize_t /* nEntries */)
{
   fLastPageIdx = 0;
   return RClusterDescriptor::RLocator();
}

void ROOT::Experimental::Detail::RPageSinkRoot::DoCommitDataset()
{
   if (!fDirectory)
      return;

   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szFooter = descriptor.SerializeFooter(nullptr);
   auto buffer = new unsigned char[szFooter];
   descriptor.SerializeFooter(buffer);
   ROOT::Experimental::Internal::RNTupleBlob footerBlob(szFooter, buffer);
   //fDirectory->WriteObject(&footerBlob, kKeyNTupleFooter);
   delete[] buffer;
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkRoot::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   if (nElements == 0)
      nElements = kDefaultElementsPerPage;
   auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
   return fPageAllocator->NewPage(columnHandle.fId, elementSize, nElements);
}

void ROOT::Experimental::Detail::RPageSinkRoot::ReleasePage(RPage &page)
{
   fPageAllocator->DeletePage(page);
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageAllocatorKey::NewPage(
   ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements)
{
   RPage newPage(columnId, mem, elementSize * nElements, elementSize);
   newPage.TryGrow(nElements);
   return newPage;
}

void ROOT::Experimental::Detail::RPageAllocatorKey::DeletePage(
   const RPage& page, ROOT::Experimental::Internal::RNTupleBlob *payload)
{
   if (page.IsNull())
      return;
   R__ASSERT(page.GetBuffer() == payload->fContent);
   free(payload->fContent);
   delete payload;
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPageSourceRoot::RPageSourceRoot(std::string_view ntupleName, std::string_view path,
   const RNTupleReadOptions &options)
   : RPageSource(ntupleName, options)
   , fMetrics("RPageSourceRoot")
   , fPageAllocator(std::make_unique<RPageAllocatorKey>())
   , fPagePool(std::make_shared<RPagePool>())
{
   fFile = std::unique_ptr<TFile>(TFile::Open(std::string(path).c_str(), "READ"));
}


ROOT::Experimental::Detail::RPageSourceRoot::~RPageSourceRoot()
{
   if (fFile)
      fFile->Close();
}


ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceRoot::DoAttach()
{
   fDirectory = fFile->GetDirectory(fNTupleName.c_str());
   RNTupleDescriptorBuilder descBuilder;

   auto keyRawNTupleHeader = fDirectory->GetKey(kKeyNTupleHeader);
   auto ntupleRawHeader = keyRawNTupleHeader->ReadObject<ROOT::Experimental::Internal::RNTupleBlob>();
   descBuilder.SetFromHeader(ntupleRawHeader->fContent);
   free(ntupleRawHeader->fContent);
   delete ntupleRawHeader;

   auto keyRawNTupleFooter = fDirectory->GetKey(kKeyNTupleFooter);
   auto ntupleRawFooter = keyRawNTupleFooter->ReadObject<ROOT::Experimental::Internal::RNTupleBlob>();
   descBuilder.AddClustersFromFooter(ntupleRawFooter->fContent);
   free(ntupleRawFooter->fContent);
   delete ntupleRawFooter;

   return descBuilder.MoveDescriptor();
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRoot::PopulatePageFromCluster(
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

   //printf("Populating page %lu/%lu [%lu] for column %d starting at %lu\n", clusterId, pageInCluster, pageIdx, columnId, firstInPage);

   std::string keyName = std::string(kKeyPagePayload) +
      std::to_string(clusterId) + kKeySeparator +
      std::to_string(pageInfo.fLocator.fPosition);
   auto pageKey = fDirectory->GetKey(keyName.c_str());
   auto pagePayload = pageKey->ReadObject<ROOT::Experimental::Internal::RNTupleBlob>();

   unsigned char *buffer = pagePayload->fContent;
   auto element = columnHandle.fColumn->GetElement();
   auto elementSize = element->GetSize();
   if (!element->IsMappable()) {
      auto pageSize = elementSize * pageInfo.fNElements;
      buffer = reinterpret_cast<unsigned char *>(malloc(pageSize));
      R__ASSERT(buffer != nullptr);
      element->Unpack(buffer, pagePayload->fContent, pageInfo.fNElements);
      free(pagePayload->fContent);
      pagePayload->fContent = buffer;
      pagePayload->fSize = pageSize;
   }

   auto indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   auto newPage = fPageAllocator->NewPage(columnId, pagePayload->fContent, elementSize, pageInfo.fNElements);
   newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
   fPagePool->RegisterPage(newPage,
      RPageDeleter([](const RPage &page, void *userData)
      {
         RPageAllocatorKey::DeletePage(page, reinterpret_cast<ROOT::Experimental::Internal::RNTupleBlob *>(userData));
      }, pagePayload));
   return newPage;
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRoot::PopulatePage(
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


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRoot::PopulatePage(
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

void ROOT::Experimental::Detail::RPageSourceRoot::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource> ROOT::Experimental::Detail::RPageSourceRoot::Clone() const
{
   return std::make_unique<RPageSourceRoot>(fNTupleName, fFile->GetName(), fOptions);
}
