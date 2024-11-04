/// \file RMiniFile.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-12-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Rtypes.h"
#include <ROOT/RConfig.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RMiniFile.hxx>
#include <ROOT/RRawFile.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>

#include <Byteswap.h>
#include <TBufferFile.h>
#include <TError.h>
#include <TFile.h>
#include <TKey.h>
#include <TObjString.h>
#include <TVirtualStreamerInfo.h>

#include <xxhash.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <chrono>

#ifdef R__LINUX
#include <fcntl.h>
#endif

#ifndef R__LITTLE_ENDIAN
#ifdef R__BYTESWAP
// `R__BYTESWAP` is defined in RConfig.hxx for little-endian architectures; undefined otherwise
#define R__LITTLE_ENDIAN 1
#else
#define R__LITTLE_ENDIAN 0
#endif
#endif /* R__LITTLE_ENDIAN */

namespace {

// The following types are used to read and write the TFile binary format

/// Big-endian 16-bit unsigned integer
class RUInt16BE {
private:
   std::uint16_t fValBE = 0;
   static std::uint16_t Swap(std::uint16_t val)
   {
#if R__LITTLE_ENDIAN == 1
      return RByteSwap<sizeof(val)>::bswap(val);
#else
      return val;
#endif
   }

public:
   RUInt16BE() = default;
   explicit RUInt16BE(const std::uint16_t val) : fValBE(Swap(val)) {}
   operator std::uint16_t() const { return Swap(fValBE); }
   RUInt16BE &operator=(const std::uint16_t val)
   {
      fValBE = Swap(val);
      return *this;
   }
};

/// Big-endian 32-bit unsigned integer
class RUInt32BE {
private:
   std::uint32_t fValBE = 0;
   static std::uint32_t Swap(std::uint32_t val)
   {
#if R__LITTLE_ENDIAN == 1
      return RByteSwap<sizeof(val)>::bswap(val);
#else
      return val;
#endif
   }

public:
   RUInt32BE() = default;
   explicit RUInt32BE(const std::uint32_t val) : fValBE(Swap(val)) {}
   operator std::uint32_t() const { return Swap(fValBE); }
   RUInt32BE &operator=(const std::uint32_t val)
   {
      fValBE = Swap(val);
      return *this;
   }
};

/// Big-endian 32-bit signed integer
class RInt32BE {
private:
   std::int32_t fValBE = 0;
   static std::int32_t Swap(std::int32_t val)
   {
#if R__LITTLE_ENDIAN == 1
      return RByteSwap<sizeof(val)>::bswap(val);
#else
      return val;
#endif
   }

public:
   RInt32BE() = default;
   explicit RInt32BE(const std::int32_t val) : fValBE(Swap(val)) {}
   operator std::int32_t() const { return Swap(fValBE); }
   RInt32BE &operator=(const std::int32_t val)
   {
      fValBE = Swap(val);
      return *this;
   }
};

/// Big-endian 64-bit unsigned integer
class RUInt64BE {
private:
   std::uint64_t fValBE = 0;
   static std::uint64_t Swap(std::uint64_t val)
   {
#if R__LITTLE_ENDIAN == 1
      return RByteSwap<sizeof(val)>::bswap(val);
#else
      return val;
#endif
   }

public:
   RUInt64BE() = default;
   explicit RUInt64BE(const std::uint64_t val) : fValBE(Swap(val)) {}
   operator std::uint64_t() const { return Swap(fValBE); }
   RUInt64BE &operator=(const std::uint64_t val)
   {
      fValBE = Swap(val);
      return *this;
   }
};

#pragma pack(push, 1)
/// A name (type, identifies, ...) in the TFile binary format
struct RTFString {
   unsigned char fLName{0};
   char fData[255];
   RTFString() = default;
   RTFString(const std::string &str)
   {
      // The length of strings with 255 characters and longer are encoded with a 32-bit integer following the first
      // byte. This is currently not handled.
      R__ASSERT(str.length() < 255);
      fLName = str.length();
      memcpy(fData, str.data(), fLName);
   }
   std::size_t GetSize() const
   {
      // A length of 255 is special and means that the first byte is followed by a 32-bit integer with the actual
      // length.
      R__ASSERT(fLName != 255);
      return 1 + fLName;
   }
};

/// The timestamp format used in TFile; the default constructor initializes with the current time
struct RTFDatetime {
   RUInt32BE fDatetime;
   RTFDatetime()
   {
      auto now = std::chrono::system_clock::now();
      auto tt = std::chrono::system_clock::to_time_t(now);
      auto tm = *localtime(&tt);
      fDatetime = (tm.tm_year + 1900 - 1995) << 26 | (tm.tm_mon + 1) << 22 | tm.tm_mday << 17 | tm.tm_hour << 12 |
                  tm.tm_min << 6 | tm.tm_sec;
   }
   explicit RTFDatetime(RUInt32BE val) : fDatetime(val) {}
};

/// The key part of a TFile record excluding the class, object, and title names
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

   std::uint32_t fKeyHeaderSize{18 + sizeof(fInfoShort)}; // not part of serialization

   RTFKey() : fInfoShort() {}
   RTFKey(std::uint64_t seekKey, std::uint64_t seekPdir, const RTFString &clName, const RTFString &objName,
          const RTFString &titleName, std::size_t szObjInMem, std::size_t szObjOnDisk = 0)
   {
      R__ASSERT(szObjInMem <= std::numeric_limits<std::uint32_t>::max());
      R__ASSERT(szObjOnDisk <= std::numeric_limits<std::uint32_t>::max());
      fObjLen = szObjInMem;
      if ((seekKey > static_cast<unsigned int>(std::numeric_limits<std::int32_t>::max())) ||
          (seekPdir > static_cast<unsigned int>(std::numeric_limits<std::int32_t>::max()))) {
         fKeyHeaderSize = 18 + sizeof(fInfoLong);
         fKeyLen = fKeyHeaderSize + clName.GetSize() + objName.GetSize() + titleName.GetSize();
         fInfoLong.fSeekKey = seekKey;
         fInfoLong.fSeekPdir = seekPdir;
         fVersion = fVersion + 1000;
      } else {
         fKeyHeaderSize = 18 + sizeof(fInfoShort);
         fKeyLen = fKeyHeaderSize + clName.GetSize() + objName.GetSize() + titleName.GetSize();
         fInfoShort.fSeekKey = seekKey;
         fInfoShort.fSeekPdir = seekPdir;
      }
      fNbytes = fKeyLen + ((szObjOnDisk == 0) ? szObjInMem : szObjOnDisk);
   }

   void MakeBigKey()
   {
      if (fVersion >= 1000)
         return;
      std::uint32_t seekKey = fInfoShort.fSeekKey;
      std::uint32_t seekPdir = fInfoShort.fSeekPdir;
      fInfoLong.fSeekKey = seekKey;
      fInfoLong.fSeekPdir = seekPdir;
      fKeyHeaderSize = fKeyHeaderSize + sizeof(fInfoLong) - sizeof(fInfoShort);
      fKeyLen = fKeyLen + sizeof(fInfoLong) - sizeof(fInfoShort);
      fNbytes = fNbytes + sizeof(fInfoLong) - sizeof(fInfoShort);
      fVersion = fVersion + 1000;
   }

   std::uint32_t GetSize() const
   {
      // Negative size indicates a gap in the file
      if (fNbytes < 0)
         return -fNbytes;
      return fNbytes;
   }

   std::uint32_t GetHeaderSize() const
   {
      if (fVersion >= 1000)
         return 18 + sizeof(fInfoLong);
      return 18 + sizeof(fInfoShort);
   }

   std::uint64_t GetSeekKey() const
   {
      if (fVersion >= 1000)
         return fInfoLong.fSeekKey;
      return fInfoShort.fSeekKey;
   }
};

/// The TFile global header
struct RTFHeader {
   char fMagic[4]{'r', 'o', 'o', 't'};
   RUInt32BE fVersion{(ROOT_VERSION_CODE >> 16) * 10000 + ((ROOT_VERSION_CODE & 0xFF00) >> 8) * 100 +
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
   RTFHeader(int compression) : fInfoShort() { fInfoShort.fCompress = compression; }

   void SetBigFile()
   {
      if (fVersion >= 1000000)
         return;

      // clang-format off
      std::uint32_t end        = fInfoShort.fEND;
      std::uint32_t seekFree   = fInfoShort.fSeekFree;
      std::uint32_t nbytesFree = fInfoShort.fNbytesFree;
      std::uint32_t nFree      = fInfoShort.fNfree;
      std::uint32_t nbytesName = fInfoShort.fNbytesName;
      std::uint32_t compress   = fInfoShort.fCompress;
      std::uint32_t seekInfo   = fInfoShort.fSeekInfo;
      std::uint32_t nbytesInfo = fInfoShort.fNbytesInfo;
      fInfoLong.fEND        = end;
      fInfoLong.fSeekFree   = seekFree;
      fInfoLong.fNbytesFree = nbytesFree;
      fInfoLong.fNfree      = nFree;
      fInfoLong.fNbytesName = nbytesName;
      fInfoLong.fUnits      = 8;
      fInfoLong.fCompress   = compress;
      fInfoLong.fSeekInfo   = seekInfo;
      fInfoLong.fNbytesInfo = nbytesInfo;
      fVersion = fVersion + 1000000;
      // clang-format on
   }

   bool IsBigFile(std::uint64_t offset = 0) const
   {
      return (fVersion >= 1000000) || (offset > static_cast<unsigned int>(std::numeric_limits<std::int32_t>::max()));
   }

   std::uint32_t GetSize() const
   {
      std::uint32_t sizeHead = 4 + sizeof(fVersion) + sizeof(fBEGIN);
      if (IsBigFile())
         return sizeHead + sizeof(fInfoLong);
      return sizeHead + sizeof(fInfoShort);
   }

   std::uint64_t GetEnd() const
   {
      if (IsBigFile())
         return fInfoLong.fEND;
      return fInfoShort.fEND;
   }

   void SetEnd(std::uint64_t value)
   {
      if (IsBigFile(value)) {
         SetBigFile();
         fInfoLong.fEND = value;
      } else {
         fInfoShort.fEND = value;
      }
   }

   std::uint64_t GetSeekFree() const
   {
      if (IsBigFile())
         return fInfoLong.fSeekFree;
      return fInfoShort.fSeekFree;
   }

   void SetSeekFree(std::uint64_t value)
   {
      if (IsBigFile(value)) {
         SetBigFile();
         fInfoLong.fSeekFree = value;
      } else {
         fInfoShort.fSeekFree = value;
      }
   }

   void SetNbytesFree(std::uint32_t value)
   {
      if (IsBigFile()) {
         fInfoLong.fNbytesFree = value;
      } else {
         fInfoShort.fNbytesFree = value;
      }
   }

   void SetNbytesName(std::uint32_t value)
   {
      if (IsBigFile()) {
         fInfoLong.fNbytesName = value;
      } else {
         fInfoShort.fNbytesName = value;
      }
   }

   std::uint64_t GetSeekInfo() const
   {
      if (IsBigFile())
         return fInfoLong.fSeekInfo;
      return fInfoShort.fSeekInfo;
   }

   void SetSeekInfo(std::uint64_t value)
   {
      if (IsBigFile(value)) {
         SetBigFile();
         fInfoLong.fSeekInfo = value;
      } else {
         fInfoShort.fSeekInfo = value;
      }
   }

   void SetNbytesInfo(std::uint32_t value)
   {
      if (IsBigFile()) {
         fInfoLong.fNbytesInfo = value;
      } else {
         fInfoShort.fNbytesInfo = value;
      }
   }

   void SetCompression(std::uint32_t value)
   {
      if (IsBigFile()) {
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
   void Set(std::uint64_t first, std::uint64_t last)
   {
      if (last > static_cast<unsigned int>(std::numeric_limits<std::int32_t>::max())) {
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

/// The header of the directory key index
struct RTFKeyList {
   RUInt32BE fNKeys;
   std::uint32_t GetSize() const { return sizeof(RTFKeyList); }
   explicit RTFKeyList(std::uint32_t nKeys) : fNKeys(nKeys) {}
};

/// A streamed TFile object
struct RTFFile {
   RUInt16BE fClassVersion{5};
   RTFDatetime fDateC;
   RTFDatetime fDateM;
   RUInt32BE fNBytesKeys{0};
   RUInt32BE fNBytesName{0};
   // The version of the key has to tell whether offsets are 32bit or 64bit long
   union {
      struct {
         RUInt32BE fSeekDir{100};
         RUInt32BE fSeekParent{0};
         RUInt32BE fSeekKeys{0};
      } fInfoShort;
      struct {
         RUInt64BE fSeekDir{100};
         RUInt64BE fSeekParent{0};
         RUInt64BE fSeekKeys{0};
      } fInfoLong;
   };

   RTFFile() : fInfoShort() {}

   // In case of a short TFile record (<2G), 3 padding ints are written after the UUID
   std::uint32_t GetSize() const
   {
      if (fClassVersion >= 1000)
         return sizeof(RTFFile);
      return 18 + sizeof(fInfoShort);
   }

   std::uint64_t GetSeekKeys() const
   {
      if (fClassVersion >= 1000)
         return fInfoLong.fSeekKeys;
      return fInfoShort.fSeekKeys;
   }

   void SetSeekKeys(std::uint64_t seekKeys)
   {
      if (seekKeys > static_cast<unsigned int>(std::numeric_limits<std::int32_t>::max())) {
         std::uint32_t seekDir = fInfoShort.fSeekDir;
         std::uint32_t seekParent = fInfoShort.fSeekParent;
         fInfoLong.fSeekDir = seekDir;
         fInfoLong.fSeekParent = seekParent;
         fInfoLong.fSeekKeys = seekKeys;
         fClassVersion = fClassVersion + 1000;
      } else {
         fInfoShort.fSeekKeys = seekKeys;
      }
   }
};

/// A zero UUID stored at the end of the TFile record
struct RTFUUID {
   RUInt16BE fVersionClass{1};
   unsigned char fUUID[16] = {0};

   RTFUUID() = default;
   std::uint32_t GetSize() const { return sizeof(RTFUUID); }
};

/// A streamed RNTuple class
///
/// NOTE: this must be kept in sync with RNTuple.hxx.
/// Aside ensuring consistency between the two classes' members, you need to make sure
/// that fVersionClass matches the class version of RNTuple.
struct RTFNTuple {
   RUInt32BE fByteCount{0x40000000 | (sizeof(RTFNTuple) - sizeof(fByteCount))};
   RUInt16BE fVersionClass{2};
   RUInt16BE fVersionEpoch{0};
   RUInt16BE fVersionMajor{0};
   RUInt16BE fVersionMinor{0};
   RUInt16BE fVersionPatch{0};
   RUInt64BE fSeekHeader{0};
   RUInt64BE fNBytesHeader{0};
   RUInt64BE fLenHeader{0};
   RUInt64BE fSeekFooter{0};
   RUInt64BE fNBytesFooter{0};
   RUInt64BE fLenFooter{0};
   RUInt64BE fMaxKeySize{0};

   static constexpr std::uint32_t GetSizePlusChecksum() { return sizeof(RTFNTuple) + sizeof(std::uint64_t); }

   RTFNTuple() = default;
   explicit RTFNTuple(const ROOT::RNTuple &inMemoryAnchor)
   {
      fVersionEpoch = inMemoryAnchor.GetVersionEpoch();
      fVersionMajor = inMemoryAnchor.GetVersionMajor();
      fVersionMinor = inMemoryAnchor.GetVersionMinor();
      fVersionPatch = inMemoryAnchor.GetVersionPatch();
      fSeekHeader = inMemoryAnchor.GetSeekHeader();
      fNBytesHeader = inMemoryAnchor.GetNBytesHeader();
      fLenHeader = inMemoryAnchor.GetLenHeader();
      fSeekFooter = inMemoryAnchor.GetSeekFooter();
      fNBytesFooter = inMemoryAnchor.GetNBytesFooter();
      fLenFooter = inMemoryAnchor.GetLenFooter();
      fMaxKeySize = inMemoryAnchor.GetMaxKeySize();
   }
   std::uint32_t GetSize() const { return sizeof(RTFNTuple); }
   // The byte count and class version members are not checksummed
   std::uint32_t GetOffsetCkData() { return sizeof(fByteCount) + sizeof(fVersionClass); }
   std::uint32_t GetSizeCkData() { return GetSize() - GetOffsetCkData(); }
   unsigned char *GetPtrCkData() { return reinterpret_cast<unsigned char *>(this) + GetOffsetCkData(); }
};

/// The bare file global header
struct RBareFileHeader {
   char fMagic[7]{'r', 'n', 't', 'u', 'p', 'l', 'e'};
   RUInt32BE fRootVersion{(ROOT_VERSION_CODE >> 16) * 10000 + ((ROOT_VERSION_CODE & 0xFF00) >> 8) * 100 +
                          (ROOT_VERSION_CODE & 0xFF)};
   RUInt32BE fFormatVersion{1};
   RUInt32BE fCompress{0};
   RTFNTuple fNTuple;
   // followed by the ntuple name
};
#pragma pack(pop)

/// The artifical class name shown for opaque RNTuple keys (see TBasket)
constexpr char const *kBlobClassName = "RBlob";
/// The class name of the RNTuple anchor
constexpr char const *kNTupleClassName = "ROOT::RNTuple";

} // anonymous namespace

namespace ROOT {
namespace Experimental {
namespace Internal {
/// If a TFile container is written by a C stream (simple file), on dataset commit, the file header
/// and the TFile record need to be updated
struct RTFileControlBlock {
   RTFHeader fHeader;
   RTFFile fFileRecord;
   std::uint64_t fSeekNTuple{0}; // Remember the offset for the keys list
   std::uint64_t fSeekFileRecord{0};
};

/// The RKeyBlob writes an invisible key into a TFile.  That is, a key that is not indexed in the list of keys,
/// like a TBasket.
/// NOTE: out of anonymous namespace because otherwise ClassDefInline fails to compile
/// on some platforms.
class RKeyBlob : public TKey {
public:
   RKeyBlob() = default;

   explicit RKeyBlob(TFile *file) : TKey(file)
   {
      fClassName = kBlobClassName;
      fVersion += 1000;
      fKeylen = Sizeof();
   }

   /// Register a new key for a data record of size nbytes
   void Reserve(size_t nbytes, std::uint64_t *seekKey)
   {
      Create(nbytes);
      *seekKey = fSeekKey;
   }

   ClassDefInlineOverride(RKeyBlob, 0)
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

// Computes how many chunks do we need to fit `nbytes` of payload, considering that the
// first chunk also needs to house the offsets of the other chunks and no chunk can
// be bigger than `maxChunkSize`. When saved to a TFile, each chunk is part of a separate TKey.
static size_t ComputeNumChunks(size_t nbytes, size_t maxChunkSize)
{
   constexpr size_t kChunkOffsetSize = sizeof(std::uint64_t);

   assert(nbytes > maxChunkSize);
   size_t nChunks = (nbytes + maxChunkSize - 1) / maxChunkSize;
   assert(nChunks > 1);
   size_t nbytesTail = nbytes % maxChunkSize;
   size_t nbytesExtra = (nbytesTail > 0) * (maxChunkSize - nbytesTail);
   size_t nbytesChunkOffsets = (nChunks - 1) * kChunkOffsetSize;
   if (nbytesChunkOffsets > nbytesExtra) {
      ++nChunks;
      nbytesChunkOffsets += kChunkOffsetSize;
   }

   // We don't support having more chunkOffsets than what fits in one chunk.
   // For a reasonable-sized maxKeySize it looks very unlikely that we can have more chunks
   // than we can fit in the first `maxKeySize` bytes. E.g. for maxKeySize = 1GiB we can fit
   // 134217728 chunk offsets, making our multi-key blob's capacity exactly 128 PiB.
   R__ASSERT(nbytesChunkOffsets <= maxChunkSize);

   return nChunks;
}

ROOT::Experimental::Internal::RMiniFileReader::RMiniFileReader(ROOT::Internal::RRawFile *rawFile) : fRawFile(rawFile) {}

ROOT::Experimental::RResult<ROOT::RNTuple>
ROOT::Experimental::Internal::RMiniFileReader::GetNTuple(std::string_view ntupleName)
{
   char ident[4];
   ReadBuffer(ident, 4, 0);
   if (std::string(ident, 4) == "root")
      return GetNTupleProper(ntupleName);
   fIsBare = true;
   return GetNTupleBare(ntupleName);
}

ROOT::Experimental::RResult<ROOT::RNTuple>
ROOT::Experimental::Internal::RMiniFileReader::GetNTupleProper(std::string_view ntupleName)
{
   RTFHeader fileHeader;
   ReadBuffer(&fileHeader, sizeof(fileHeader), 0);

   RTFKey key;
   RTFString name;
   ReadBuffer(&key, sizeof(key), fileHeader.fBEGIN);
   // Skip over the entire key length, including the class name, object name, and title stored in it.
   std::uint64_t offset = fileHeader.fBEGIN + key.fKeyLen;
   // Skip over the name and title of the TNamed preceding the TFile entry.
   ReadBuffer(&name, 1, offset);
   offset += name.GetSize();
   ReadBuffer(&name, 1, offset);
   offset += name.GetSize();
   RTFFile file;
   ReadBuffer(&file, sizeof(file), offset);

   RUInt32BE nKeys;
   offset = file.GetSeekKeys();
   ReadBuffer(&key, sizeof(key), offset);
   offset += key.fKeyLen;
   ReadBuffer(&nKeys, sizeof(nKeys), offset);
   offset += sizeof(nKeys);
   bool found = false;
   for (unsigned int i = 0; i < nKeys; ++i) {
      ReadBuffer(&key, sizeof(key), offset);
      auto offsetNextKey = offset + key.fKeyLen;

      offset += key.GetHeaderSize();
      ReadBuffer(&name, 1, offset);
      ReadBuffer(&name, name.GetSize(), offset);
      if (std::string_view(name.fData, name.fLName) != kNTupleClassName) {
         offset = offsetNextKey;
         continue;
      }
      offset += name.GetSize();
      ReadBuffer(&name, 1, offset);
      ReadBuffer(&name, name.GetSize(), offset);
      if (std::string_view(name.fData, name.fLName) == ntupleName) {
         found = true;
         break;
      }
      offset = offsetNextKey;
   }
   if (!found) {
      return R__FAIL("no RNTuple named '" + std::string(ntupleName) + "' in file '" + fRawFile->GetUrl() + "'");
   }

   offset = key.GetSeekKey() + key.fKeyLen;

   // size of a RTFNTuple version 2 (min supported version); future anchor versions can grow.
   constexpr size_t kMinNTupleSize = 78;
   static_assert(kMinNTupleSize == RTFNTuple::GetSizePlusChecksum());
   if (key.fObjLen < kMinNTupleSize) {
      return R__FAIL("invalid anchor size: " + std::to_string(key.fObjLen) + " < " + std::to_string(sizeof(RTFNTuple)));
   }
   // The object length can be smaller than the size of RTFNTuple if it comes from a past RNTuple class version,
   // or larger than it if it comes from a future RNTuple class version.
   auto bufAnchor = std::make_unique<unsigned char[]>(std::max<size_t>(key.fObjLen, sizeof(RTFNTuple)));
   RTFNTuple *ntuple = new (bufAnchor.get()) RTFNTuple;

   auto objNbytes = key.GetSize() - key.fKeyLen;
   ReadBuffer(ntuple, objNbytes, offset);
   if (objNbytes != key.fObjLen) {
      RNTupleDecompressor decompressor;
      decompressor.Unzip(bufAnchor.get(), objNbytes, key.fObjLen);
   }

   // We require that future class versions only append members and store the checksum in the last 8 bytes
   // Checksum calculation: strip byte count, class version, fChecksum member
   auto lenCkData = key.fObjLen - ntuple->GetOffsetCkData() - sizeof(uint64_t);
   auto ckCalc = XXH3_64bits(ntuple->GetPtrCkData(), lenCkData);
   uint64_t ckOnDisk;

   RUInt64BE *ckOnDiskPtr = reinterpret_cast<RUInt64BE *>(bufAnchor.get() + key.fObjLen - sizeof(uint64_t));
   ckOnDisk = static_cast<uint64_t>(*ckOnDiskPtr);
   if (ckCalc != ckOnDisk) {
      return R__FAIL("RNTuple anchor checksum mismatch");
   }

   return CreateAnchor(ntuple->fVersionEpoch, ntuple->fVersionMajor, ntuple->fVersionMinor, ntuple->fVersionPatch,
                       ntuple->fSeekHeader, ntuple->fNBytesHeader, ntuple->fLenHeader, ntuple->fSeekFooter,
                       ntuple->fNBytesFooter, ntuple->fLenFooter, ntuple->fMaxKeySize);
}

ROOT::Experimental::RResult<ROOT::RNTuple>
ROOT::Experimental::Internal::RMiniFileReader::GetNTupleBare(std::string_view ntupleName)
{
   RBareFileHeader fileHeader;
   ReadBuffer(&fileHeader, sizeof(fileHeader), 0);
   RTFString name;
   auto offset = sizeof(fileHeader);
   ReadBuffer(&name, 1, offset);
   ReadBuffer(&name, name.GetSize(), offset);
   std::string_view foundName(name.fData, name.fLName);
   if (foundName != ntupleName) {
      return R__FAIL("expected RNTuple named '" + std::string(ntupleName) + "' but instead found '" +
                     std::string(foundName) + "' in file '" + fRawFile->GetUrl() + "'");
   }
   offset += name.GetSize();

   RTFNTuple ntuple;
   ReadBuffer(&ntuple, sizeof(ntuple), offset);
   std::uint64_t onDiskChecksum;
   ReadBuffer(&onDiskChecksum, sizeof(onDiskChecksum), offset + sizeof(ntuple));
   auto checksum = XXH3_64bits(ntuple.GetPtrCkData(), ntuple.GetSizeCkData());
   if (checksum != static_cast<uint64_t>(onDiskChecksum))
      return R__FAIL("RNTuple bare file: anchor checksum mismatch");

   return CreateAnchor(ntuple.fVersionEpoch, ntuple.fVersionMajor, ntuple.fVersionMinor, ntuple.fVersionPatch,
                       ntuple.fSeekHeader, ntuple.fNBytesHeader, ntuple.fLenHeader, ntuple.fSeekFooter,
                       ntuple.fNBytesFooter, ntuple.fLenFooter, ntuple.fMaxKeySize);
}

void ROOT::Experimental::Internal::RMiniFileReader::ReadBuffer(void *buffer, size_t nbytes, std::uint64_t offset)
{
   size_t nread;
   if (fMaxKeySize == 0 || nbytes <= fMaxKeySize) {
      // Fast path: read single blob
      nread = fRawFile->ReadAt(buffer, nbytes, offset);
   } else {
      // Read chunked blob. See RNTupleFileWriter::WriteBlob() for details.
      const size_t nChunks = ComputeNumChunks(nbytes, fMaxKeySize);
      const size_t nbytesChunkOffsets = (nChunks - 1) * sizeof(std::uint64_t);
      const size_t nbytesFirstChunk = fMaxKeySize - nbytesChunkOffsets;
      uint8_t *bufCur = reinterpret_cast<uint8_t *>(buffer);

      // Read first chunk
      nread = fRawFile->ReadAt(bufCur, fMaxKeySize, offset);
      R__ASSERT(nread == fMaxKeySize);
      // NOTE: we read the entire chunk in `bufCur`, but we only advance the pointer by `nbytesFirstChunk`,
      // since the last part of `bufCur` will later be overwritten by the next chunk's payload.
      // We do this to avoid a second ReadAt to read in the chunk offsets.
      bufCur += nbytesFirstChunk;
      nread -= nbytesChunkOffsets;

      const auto chunkOffsets = std::make_unique<std::uint64_t[]>(nChunks - 1);
      memcpy(chunkOffsets.get(), bufCur, nbytesChunkOffsets);

      size_t remainingBytes = nbytes - nbytesFirstChunk;
      std::uint64_t *curChunkOffset = &chunkOffsets[0];

      do {
         std::uint64_t chunkOffset;
         RNTupleSerializer::DeserializeUInt64(curChunkOffset, chunkOffset);
         ++curChunkOffset;

         const size_t bytesToRead = std::min<size_t>(fMaxKeySize, remainingBytes);
         // Ensure we don't read outside of the buffer
         R__ASSERT(static_cast<size_t>(bufCur - reinterpret_cast<uint8_t *>(buffer)) <= nbytes - bytesToRead);

         auto nbytesRead = fRawFile->ReadAt(bufCur, bytesToRead, chunkOffset);
         R__ASSERT(nbytesRead == bytesToRead);

         nread += bytesToRead;
         bufCur += bytesToRead;
         remainingBytes -= bytesToRead;
      } while (remainingBytes > 0);
   }
   R__ASSERT(nread == nbytes);
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Internal::RNTupleFileWriter::RFileSimple::RFileSimple() = default;

void ROOT::Experimental::Internal::RNTupleFileWriter::RFileSimple::AllocateBuffers(std::size_t bufferSize)
{
   static_assert(kHeaderBlockSize % kBlockAlign == 0, "invalid header block size");
   if (bufferSize % kBlockAlign != 0)
      throw RException(R__FAIL("Buffer size not a multiple of alignment: " + std::to_string(bufferSize)));
   fBlockSize = bufferSize;

   std::align_val_t blockAlign{kBlockAlign};
   fHeaderBlock = static_cast<unsigned char *>(::operator new[](kHeaderBlockSize, blockAlign));
   memset(fHeaderBlock, 0, kHeaderBlockSize);
   fBlock = static_cast<unsigned char *>(::operator new[](fBlockSize, blockAlign));
   memset(fBlock, 0, fBlockSize);
}

ROOT::Experimental::Internal::RNTupleFileWriter::RFileSimple::~RFileSimple()
{
   if (fFile)
      fclose(fFile);

   std::align_val_t blockAlign{kBlockAlign};
   if (fHeaderBlock)
      ::operator delete[](fHeaderBlock, blockAlign);
   if (fBlock)
      ::operator delete[](fBlock, blockAlign);
}

namespace {
int FSeek64(FILE *stream, std::int64_t offset, int origin)
{
#ifdef R__SEEK64
   return fseeko64(stream, offset, origin);
#else
   return fseek(stream, offset, origin);
#endif
}
} // namespace

void ROOT::Experimental::Internal::RNTupleFileWriter::RFileSimple::Flush()
{
   // Write the last partially filled block, which may still need appropriate alignment for Direct I/O.
   // If it is the first block, get the updated header block.
   if (fBlockOffset == 0) {
      std::size_t headerBlockSize = kHeaderBlockSize;
      if (headerBlockSize > fFilePos) {
         headerBlockSize = fFilePos;
      }
      memcpy(fBlock, fHeaderBlock, headerBlockSize);
   }

   std::size_t retval = FSeek64(fFile, fBlockOffset, SEEK_SET);
   if (retval)
      throw RException(R__FAIL(std::string("Seek failed: ") + strerror(errno)));

   std::size_t lastBlockSize = fFilePos - fBlockOffset;
   R__ASSERT(lastBlockSize <= fBlockSize);
   if (fDirectIO) {
      // Round up to a multiple of kBlockAlign.
      lastBlockSize += kBlockAlign - 1;
      lastBlockSize = (lastBlockSize / kBlockAlign) * kBlockAlign;
      R__ASSERT(lastBlockSize <= fBlockSize);
   }
   retval = fwrite(fBlock, 1, lastBlockSize, fFile);
   if (retval != lastBlockSize)
      throw RException(R__FAIL(std::string("write failed: ") + strerror(errno)));

   // Write the (updated) header block, unless it was part of the write above.
   if (fBlockOffset > 0) {
      retval = FSeek64(fFile, 0, SEEK_SET);
      if (retval)
         throw RException(R__FAIL(std::string("Seek failed: ") + strerror(errno)));

      retval = fwrite(fHeaderBlock, 1, kHeaderBlockSize, fFile);
      if (retval != RFileSimple::kHeaderBlockSize)
         throw RException(R__FAIL(std::string("write failed: ") + strerror(errno)));
   }

   retval = fflush(fFile);
   if (retval)
      throw RException(R__FAIL(std::string("Flush failed: ") + strerror(errno)));
}

void ROOT::Experimental::Internal::RNTupleFileWriter::RFileSimple::Write(const void *buffer, size_t nbytes,
                                                                         std::int64_t offset)
{
   R__ASSERT(fFile);
   size_t retval;
   if ((offset >= 0) && (static_cast<std::uint64_t>(offset) != fFilePos)) {
      fFilePos = offset;
   }

   // Keep header block to overwrite on commit.
   if (fFilePos < kHeaderBlockSize) {
      std::size_t headerBytes = nbytes;
      if (fFilePos + headerBytes > kHeaderBlockSize) {
         headerBytes = kHeaderBlockSize - fFilePos;
      }
      memcpy(fHeaderBlock + fFilePos, buffer, headerBytes);
   }

   R__ASSERT(fFilePos >= fBlockOffset);

   while (nbytes > 0) {
      std::uint64_t posInBlock = fFilePos % fBlockSize;
      std::uint64_t blockOffset = fFilePos - posInBlock;
      if (blockOffset != fBlockOffset) {
         // Write the block.
         retval = FSeek64(fFile, fBlockOffset, SEEK_SET);
         if (retval)
            throw RException(R__FAIL(std::string("Seek failed: ") + strerror(errno)));

         retval = fwrite(fBlock, 1, fBlockSize, fFile);
         if (retval != fBlockSize)
            throw RException(R__FAIL(std::string("write failed: ") + strerror(errno)));

         // Null the buffer contents for good measure.
         memset(fBlock, 0, fBlockSize);
      }

      fBlockOffset = blockOffset;
      std::size_t blockSize = nbytes;
      if (blockSize > fBlockSize - posInBlock) {
         blockSize = fBlockSize - posInBlock;
      }
      memcpy(fBlock + posInBlock, buffer, blockSize);
      buffer = static_cast<const unsigned char *>(buffer) + blockSize;
      nbytes -= blockSize;
      fFilePos += blockSize;
   }
}

std::uint64_t ROOT::Experimental::Internal::RNTupleFileWriter::RFileSimple::WriteKey(
   const void *buffer, std::size_t nbytes, std::size_t len, std::int64_t offset, std::uint64_t directoryOffset,
   const std::string &className, const std::string &objectName, const std::string &title)
{
   if (offset > 0)
      fKeyOffset = offset;
   RTFString strClass{className};
   RTFString strObject{objectName};
   RTFString strTitle{title};

   RTFKey key(fKeyOffset, directoryOffset, strClass, strObject, strTitle, len, nbytes);
   Write(&key, key.fKeyHeaderSize, fKeyOffset);
   Write(&strClass, strClass.GetSize());
   Write(&strObject, strObject.GetSize());
   Write(&strTitle, strTitle.GetSize());
   auto offsetData = fFilePos;
   // The next key starts after the data.
   fKeyOffset = offsetData + nbytes;
   if (buffer)
      Write(buffer, nbytes);

   return offsetData;
}

////////////////////////////////////////////////////////////////////////////////

void ROOT::Experimental::Internal::RNTupleFileWriter::RFileProper::Write(const void *buffer, size_t nbytes,
                                                                         std::int64_t offset)
{
   R__ASSERT(fFile);
   fFile->Seek(offset);
   bool rv = fFile->WriteBuffer((char *)(buffer), nbytes);
   if (rv)
      throw RException(R__FAIL("WriteBuffer failed."));
}

std::uint64_t
ROOT::Experimental::Internal::RNTupleFileWriter::RFileProper::WriteKey(const void *buffer, size_t nbytes, size_t len)
{
   std::uint64_t offsetKey;
   RKeyBlob keyBlob(fFile);
   // Since it is unknown beforehand if offsetKey is beyond the 2GB limit or not,
   // RKeyBlob will always reserve space for a big key (version >= 1000)
   keyBlob.Reserve(nbytes, &offsetKey);

   auto offset = offsetKey;
   RTFString strClass{kBlobClassName};
   RTFString strObject;
   RTFString strTitle;
   RTFKey keyHeader(offset, offset, strClass, strObject, strTitle, len, nbytes);
   // Follow the fact that RKeyBlob is a big key unconditionally (see above)
   keyHeader.MakeBigKey();

   Write(&keyHeader, keyHeader.fKeyHeaderSize, offset);
   offset += keyHeader.fKeyHeaderSize;
   Write(&strClass, strClass.GetSize(), offset);
   offset += strClass.GetSize();
   Write(&strObject, strObject.GetSize(), offset);
   offset += strObject.GetSize();
   Write(&strTitle, strTitle.GetSize(), offset);
   offset += strTitle.GetSize();
   auto offsetData = offset;
   if (buffer)
      Write(buffer, nbytes, offset);

   return offsetData;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Internal::RNTupleFileWriter::RNTupleFileWriter(std::string_view name, std::uint64_t maxKeySize)
   : fNTupleName(name)
{
   fFileSimple.fControlBlock = std::make_unique<ROOT::Experimental::Internal::RTFileControlBlock>();
   fNTupleAnchor.fMaxKeySize = maxKeySize;
   auto infoRNTuple = RNTuple::Class()->GetStreamerInfo();
   fStreamerInfoMap[infoRNTuple->GetNumber()] = infoRNTuple;
}

ROOT::Experimental::Internal::RNTupleFileWriter::~RNTupleFileWriter() {}

std::unique_ptr<ROOT::Experimental::Internal::RNTupleFileWriter>
ROOT::Experimental::Internal::RNTupleFileWriter::Recreate(std::string_view ntupleName, std::string_view path,
                                                          EContainerFormat containerFormat,
                                                          const RNTupleWriteOptions &options)
{
   std::string fileName(path);
   size_t idxDirSep = fileName.find_last_of("\\/");
   if (idxDirSep != std::string::npos) {
      fileName.erase(0, idxDirSep + 1);
   }
#ifdef R__LINUX
   int flags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_LARGEFILE
   // Add the equivalent flag that is passed by fopen64.
   flags |= O_LARGEFILE;
#endif
   if (options.GetUseDirectIO()) {
      flags |= O_DIRECT;
   }
   int fd = open(std::string(path).c_str(), flags, 0666);
   FILE *fileStream = fdopen(fd, "wb");
#else
#ifdef R__SEEK64
   FILE *fileStream = fopen64(std::string(path.data(), path.size()).c_str(), "wb");
#else
   FILE *fileStream = fopen(std::string(path.data(), path.size()).c_str(), "wb");
#endif
#endif
   R__ASSERT(fileStream);
   // RNTupleFileWriter::RFileSimple does its own buffering, turn off additional buffering from C stdio.
   std::setvbuf(fileStream, nullptr, _IONBF, 0);

   auto writer = std::unique_ptr<RNTupleFileWriter>(new RNTupleFileWriter(ntupleName, options.GetMaxKeySize()));
   writer->fFileSimple.fFile = fileStream;
   writer->fFileSimple.fDirectIO = options.GetUseDirectIO();
   writer->fFileSimple.AllocateBuffers(options.GetWriteBufferSize());
   writer->fFileName = fileName;

   int defaultCompression = options.GetCompression();
   switch (containerFormat) {
   case EContainerFormat::kTFile: writer->WriteTFileSkeleton(defaultCompression); break;
   case EContainerFormat::kBare:
      writer->fIsBare = true;
      writer->WriteBareFileSkeleton(defaultCompression);
      break;
   default: R__ASSERT(false && "Internal error: unhandled container format");
   }

   return writer;
}

std::unique_ptr<ROOT::Experimental::Internal::RNTupleFileWriter>
ROOT::Experimental::Internal::RNTupleFileWriter::Append(std::string_view ntupleName, TFile &file,
                                                        std::uint64_t maxKeySize)
{
   auto writer = std::unique_ptr<RNTupleFileWriter>(new RNTupleFileWriter(ntupleName, maxKeySize));
   writer->fFileProper.fFile = &file;
   return writer;
}

void ROOT::Experimental::Internal::RNTupleFileWriter::UpdateStreamerInfos(
   const RNTupleSerializer::StreamerInfoMap_t &streamerInfos)
{
   fStreamerInfoMap.insert(streamerInfos.cbegin(), streamerInfos.cend());
}

void ROOT::Experimental::Internal::RNTupleFileWriter::Commit()
{
   if (fFileProper) {
      // Easy case, the ROOT file header and the RNTuple streaming is taken care of by TFile
      fFileProper.fFile->WriteObject(&fNTupleAnchor, fNTupleName.c_str());

      // Make sure the streamer info records used in the RNTuple are written to the file
      TBufferFile buf(TBuffer::kWrite);
      buf.SetParent(fFileProper.fFile);
      for (auto [_, info] : fStreamerInfoMap)
         buf.TagStreamerInfo(info);

      fFileProper.fFile->Write();
      return;
   }

   // Writing by C file stream: prepare the container format header and stream the RNTuple anchor object
   R__ASSERT(fFileSimple);

   if (fIsBare) {
      RTFNTuple ntupleOnDisk(fNTupleAnchor);
      // Compute the checksum
      std::uint64_t checksum = XXH3_64bits(ntupleOnDisk.GetPtrCkData(), ntupleOnDisk.GetSizeCkData());
      memcpy(fFileSimple.fHeaderBlock + fFileSimple.fControlBlock->fSeekNTuple, &ntupleOnDisk, ntupleOnDisk.GetSize());
      memcpy(fFileSimple.fHeaderBlock + fFileSimple.fControlBlock->fSeekNTuple + ntupleOnDisk.GetSize(), &checksum,
             sizeof(checksum));
      fFileSimple.Flush();
      return;
   }

   WriteTFileNTupleKey();
   WriteTFileKeysList();
   WriteTFileStreamerInfo();
   WriteTFileFreeList();

   // Update header and TFile record
   memcpy(fFileSimple.fHeaderBlock, &fFileSimple.fControlBlock->fHeader, fFileSimple.fControlBlock->fHeader.GetSize());
   R__ASSERT(fFileSimple.fControlBlock->fSeekFileRecord + fFileSimple.fControlBlock->fFileRecord.GetSize() <
             RFileSimple::kHeaderBlockSize);
   memcpy(fFileSimple.fHeaderBlock + fFileSimple.fControlBlock->fSeekFileRecord,
          &fFileSimple.fControlBlock->fFileRecord, fFileSimple.fControlBlock->fFileRecord.GetSize());

   fFileSimple.Flush();
}

std::uint64_t ROOT::Experimental::Internal::RNTupleFileWriter::WriteBlob(const void *data, size_t nbytes, size_t len)
{
   auto writeKey = [this](const void *payload, size_t nBytes, size_t length) {
      std::uint64_t offset;
      if (fFileSimple) {
         if (fIsBare) {
            offset = fFileSimple.fKeyOffset;
            fFileSimple.Write(payload, nBytes);
            fFileSimple.fKeyOffset += nBytes;
         } else {
            offset = fFileSimple.WriteKey(payload, nBytes, length, -1, 100, kBlobClassName);
         }
      } else {
         offset = fFileProper.WriteKey(payload, nBytes, length);
      }
      return offset;
   };

   const std::uint64_t maxKeySize = fNTupleAnchor.fMaxKeySize;
   R__ASSERT(maxKeySize > 0);
   // We don't need the object length except for seeing compression ratios in TFile::Map()
   // Make sure that the on-disk object length fits into the TKey header.
   if (static_cast<std::uint64_t>(len) > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
      len = nbytes;

   if (nbytes <= maxKeySize) {
      // Fast path: only write 1 key.
      return writeKey(data, nbytes, len);
   }

   /**
    * Writing a key bigger than the max allowed size. In this case we split the payload
    * into multiple keys, reserving the end of the first key payload for pointers to the
    * next ones. E.g. if a key needs to be split into 3 chunks, the first chunk will have
    * the format:
    *  +--------------------+
    *  |                    |
    *  |        Data        |
    *  |--------------------|
    *  | pointer to chunk 2 |
    *  | pointer to chunk 3 |
    *  +--------------------+
    */
   const size_t nChunks = ComputeNumChunks(nbytes, maxKeySize);
   const size_t nbytesChunkOffsets = (nChunks - 1) * sizeof(std::uint64_t);
   const size_t nbytesFirstChunk = maxKeySize - nbytesChunkOffsets;
   // Skip writing the first chunk, it will be written last (in the file) below.

   const uint8_t *chunkData = reinterpret_cast<const uint8_t *>(data) + nbytesFirstChunk;
   size_t remainingBytes = nbytes - nbytesFirstChunk;

   const auto chunkOffsetsToWrite = std::make_unique<std::uint64_t[]>(nChunks - 1);
   std::uint64_t chunkOffsetIdx = 0;

   do {
      const size_t bytesNextChunk = std::min<size_t>(remainingBytes, maxKeySize);
      const std::uint64_t offset = writeKey(chunkData, bytesNextChunk, bytesNextChunk);

      RNTupleSerializer::SerializeUInt64(offset, &chunkOffsetsToWrite[chunkOffsetIdx]);
      ++chunkOffsetIdx;

      remainingBytes -= bytesNextChunk;
      chunkData += bytesNextChunk;

   } while (remainingBytes > 0);

   // Write the first key, with part of the data and the pointers to (logically) following keys appended.
   const std::uint64_t firstOffset = ReserveBlob(maxKeySize, maxKeySize);
   WriteIntoReservedBlob(data, nbytesFirstChunk, firstOffset);
   const std::uint64_t chunkOffsetsOffset = firstOffset + nbytesFirstChunk;
   WriteIntoReservedBlob(chunkOffsetsToWrite.get(), nbytesChunkOffsets, chunkOffsetsOffset);

   return firstOffset;
}

std::uint64_t ROOT::Experimental::Internal::RNTupleFileWriter::ReserveBlob(size_t nbytes, size_t len)
{
   // ReserveBlob cannot be used to reserve a multi-key blob
   R__ASSERT(nbytes <= fNTupleAnchor.GetMaxKeySize());

   std::uint64_t offset;
   if (fFileSimple) {
      if (fIsBare) {
         offset = fFileSimple.fKeyOffset;
         fFileSimple.fKeyOffset += nbytes;
      } else {
         offset = fFileSimple.WriteKey(/*buffer=*/nullptr, nbytes, len, -1, 100, kBlobClassName);
      }
   } else {
      offset = fFileProper.WriteKey(/*buffer=*/nullptr, nbytes, len);
   }
   return offset;
}

void ROOT::Experimental::Internal::RNTupleFileWriter::WriteIntoReservedBlob(const void *buffer, size_t nbytes,
                                                                            std::int64_t offset)
{
   if (fFileSimple) {
      fFileSimple.Write(buffer, nbytes, offset);
   } else {
      fFileProper.Write(buffer, nbytes, offset);
   }
}

std::uint64_t
ROOT::Experimental::Internal::RNTupleFileWriter::WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader)
{
   auto offset = WriteBlob(data, nbytes, lenHeader);
   fNTupleAnchor.fLenHeader = lenHeader;
   fNTupleAnchor.fNBytesHeader = nbytes;
   fNTupleAnchor.fSeekHeader = offset;
   return offset;
}

std::uint64_t
ROOT::Experimental::Internal::RNTupleFileWriter::WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter)
{
   auto offset = WriteBlob(data, nbytes, lenFooter);
   fNTupleAnchor.fLenFooter = lenFooter;
   fNTupleAnchor.fNBytesFooter = nbytes;
   fNTupleAnchor.fSeekFooter = offset;
   return offset;
}

void ROOT::Experimental::Internal::RNTupleFileWriter::WriteBareFileSkeleton(int defaultCompression)
{
   RBareFileHeader bareHeader;
   bareHeader.fCompress = defaultCompression;
   fFileSimple.Write(&bareHeader, sizeof(bareHeader), 0);
   RTFString ntupleName{fNTupleName};
   fFileSimple.Write(&ntupleName, ntupleName.GetSize());

   // Write zero-initialized ntuple to reserve the space; will be overwritten on commit
   RTFNTuple ntupleOnDisk;
   fFileSimple.fControlBlock->fSeekNTuple = fFileSimple.fFilePos;
   fFileSimple.Write(&ntupleOnDisk, ntupleOnDisk.GetSize());
   std::uint64_t checksum = 0;
   fFileSimple.Write(&checksum, sizeof(checksum));
   fFileSimple.fKeyOffset = fFileSimple.fFilePos;
}

void ROOT::Experimental::Internal::RNTupleFileWriter::WriteTFileStreamerInfo()
{
   // The streamer info record is a TList of TStreamerInfo object.  We cannot use
   // RNTupleSerializer::SerializeStreamerInfos because that uses TBufferIO::WriteObject.
   // This would prepend the streamed TList with self-decription information.
   // The streamer info record is just the streamed TList.

   TList streamerInfoList;
   for (auto [_, info] : fStreamerInfoMap) {
      streamerInfoList.Add(info);
   }

   // We will stream the list with a TBufferFile. When reading the streamer info records back,
   // the read buffer includes the key and the streamed list.  Therefore, we need to start streaming
   // with an offset of the key length.  Otherwise, the offset for referencing duplicate objects in the
   // buffer will point to the wrong places.

   // Figure out key length
   RTFString strTList{"TList"};
   RTFString strStreamerInfo{"StreamerInfo"};
   RTFString strStreamerTitle{"Doubly linked list"};
   fFileSimple.fControlBlock->fHeader.SetSeekInfo(fFileSimple.fKeyOffset);
   auto keyLen =
      RTFKey(fFileSimple.fControlBlock->fHeader.GetSeekInfo(), 100, strTList, strStreamerInfo, strStreamerTitle, 0)
         .fKeyLen;

   TBufferFile buffer(TBuffer::kWrite, keyLen + 1);
   buffer.SetBufferOffset(keyLen);
   streamerInfoList.Streamer(buffer);
   assert(buffer.Length() > keyLen);
   const auto bufPayload = buffer.Buffer() + keyLen;
   const auto lenPayload = buffer.Length() - keyLen;

   RNTupleCompressor compressor;
   auto zipStreamerInfos = std::make_unique<unsigned char[]>(lenPayload);
   auto szZipStreamerInfos = compressor.Zip(bufPayload, lenPayload, 1, zipStreamerInfos.get());

   fFileSimple.WriteKey(zipStreamerInfos.get(), szZipStreamerInfos, lenPayload,
                        fFileSimple.fControlBlock->fHeader.GetSeekInfo(), 100, "TList", "StreamerInfo",
                        "Doubly linked list");
   fFileSimple.fControlBlock->fHeader.SetNbytesInfo(fFileSimple.fFilePos -
                                                    fFileSimple.fControlBlock->fHeader.GetSeekInfo());
}

void ROOT::Experimental::Internal::RNTupleFileWriter::WriteTFileKeysList()
{
   RTFString strEmpty;
   RTFString strRNTupleClass{"ROOT::RNTuple"};
   RTFString strRNTupleName{fNTupleName};
   RTFString strFileName{fFileName};

   RTFKey keyRNTuple(fFileSimple.fControlBlock->fSeekNTuple, 100, strRNTupleClass, strRNTupleName, strEmpty,
                     RTFNTuple::GetSizePlusChecksum());

   fFileSimple.fControlBlock->fFileRecord.SetSeekKeys(fFileSimple.fKeyOffset);
   RTFKeyList keyList{1};
   RTFKey keyKeyList(fFileSimple.fControlBlock->fFileRecord.GetSeekKeys(), 100, strEmpty, strFileName, strEmpty,
                     keyList.GetSize() + keyRNTuple.fKeyLen);
   fFileSimple.Write(&keyKeyList, keyKeyList.fKeyHeaderSize, fFileSimple.fControlBlock->fFileRecord.GetSeekKeys());
   fFileSimple.Write(&strEmpty, strEmpty.GetSize());
   fFileSimple.Write(&strFileName, strFileName.GetSize());
   fFileSimple.Write(&strEmpty, strEmpty.GetSize());
   fFileSimple.Write(&keyList, keyList.GetSize());
   fFileSimple.Write(&keyRNTuple, keyRNTuple.fKeyHeaderSize);
   // Write class name, object name, and title for this key.
   fFileSimple.Write(&strRNTupleClass, strRNTupleClass.GetSize());
   fFileSimple.Write(&strRNTupleName, strRNTupleName.GetSize());
   fFileSimple.Write(&strEmpty, strEmpty.GetSize());
   fFileSimple.fControlBlock->fFileRecord.fNBytesKeys =
      fFileSimple.fFilePos - fFileSimple.fControlBlock->fFileRecord.GetSeekKeys();
   fFileSimple.fKeyOffset = fFileSimple.fFilePos;
}

void ROOT::Experimental::Internal::RNTupleFileWriter::WriteTFileFreeList()
{
   fFileSimple.fControlBlock->fHeader.SetSeekFree(fFileSimple.fKeyOffset);
   RTFString strEmpty;
   RTFString strFileName{fFileName};
   RTFFreeEntry freeEntry;
   RTFKey keyFreeList(fFileSimple.fControlBlock->fHeader.GetSeekFree(), 100, strEmpty, strFileName, strEmpty,
                      freeEntry.GetSize());
   std::uint64_t firstFree = fFileSimple.fControlBlock->fHeader.GetSeekFree() + keyFreeList.GetSize();
   freeEntry.Set(firstFree, std::max(2000000000ULL, ((firstFree / 1000000000ULL) + 1) * 1000000000ULL));
   fFileSimple.WriteKey(&freeEntry, freeEntry.GetSize(), freeEntry.GetSize(),
                        fFileSimple.fControlBlock->fHeader.GetSeekFree(), 100, "", fFileName, "");
   fFileSimple.fControlBlock->fHeader.SetNbytesFree(fFileSimple.fFilePos -
                                                    fFileSimple.fControlBlock->fHeader.GetSeekFree());
   fFileSimple.fControlBlock->fHeader.SetEnd(fFileSimple.fFilePos);
}

void ROOT::Experimental::Internal::RNTupleFileWriter::WriteTFileNTupleKey()
{
   RTFString strRNTupleClass{"ROOT::RNTuple"};
   RTFString strRNTupleName{fNTupleName};
   RTFString strEmpty;

   RTFNTuple ntupleOnDisk(fNTupleAnchor);
   RUInt64BE checksum{XXH3_64bits(ntupleOnDisk.GetPtrCkData(), ntupleOnDisk.GetSizeCkData())};
   fFileSimple.fControlBlock->fSeekNTuple = fFileSimple.fKeyOffset;

   char keyBuf[RTFNTuple::GetSizePlusChecksum()];

   // concatenate the RNTuple anchor with its checksum
   memcpy(keyBuf, &ntupleOnDisk, sizeof(RTFNTuple));
   memcpy(keyBuf + sizeof(RTFNTuple), &checksum, sizeof(checksum));

   fFileSimple.WriteKey(keyBuf, sizeof(keyBuf), sizeof(keyBuf), fFileSimple.fControlBlock->fSeekNTuple, 100,
                        "ROOT::RNTuple", fNTupleName, "");
}

void ROOT::Experimental::Internal::RNTupleFileWriter::WriteTFileSkeleton(int defaultCompression)
{
   RTFString strTFile{"TFile"};
   RTFString strFileName{fFileName};
   RTFString strEmpty;

   fFileSimple.fControlBlock->fHeader = RTFHeader(defaultCompression);

   RTFUUID uuid;

   // First record of the file: the TFile object at offset 100
   RTFKey keyRoot(100, 0, strTFile, strFileName, strEmpty,
                  sizeof(RTFFile) + strFileName.GetSize() + strEmpty.GetSize() + uuid.GetSize());
   std::uint32_t nbytesName = keyRoot.fKeyLen + strFileName.GetSize() + 1;
   fFileSimple.fControlBlock->fFileRecord.fNBytesName = nbytesName;
   fFileSimple.fControlBlock->fHeader.SetNbytesName(nbytesName);

   fFileSimple.Write(&keyRoot, keyRoot.fKeyHeaderSize, 100);
   // Write class name, object name, and title for the TFile key.
   fFileSimple.Write(&strTFile, strTFile.GetSize());
   fFileSimple.Write(&strFileName, strFileName.GetSize());
   fFileSimple.Write(&strEmpty, strEmpty.GetSize());
   // Write the name and title of the TNamed preceding the TFile entry.
   fFileSimple.Write(&strFileName, strFileName.GetSize());
   fFileSimple.Write(&strEmpty, strEmpty.GetSize());
   // Will be overwritten on commit
   fFileSimple.fControlBlock->fSeekFileRecord = fFileSimple.fFilePos;
   fFileSimple.Write(&fFileSimple.fControlBlock->fFileRecord, fFileSimple.fControlBlock->fFileRecord.GetSize());
   fFileSimple.Write(&uuid, uuid.GetSize());

   // Padding bytes to allow the TFile record to grow for a big file
   RUInt32BE padding{0};
   for (int i = 0; i < 3; ++i)
      fFileSimple.Write(&padding, sizeof(padding));
   fFileSimple.fKeyOffset = fFileSimple.fFilePos;
}
