// Original Author: ROOT Team

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ZipLHC4.h"

#include "lhc4codec/lhc4codec.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#else
#include <process.h>
#define getpid _getpid
#endif

#if !defined(R__unlikely)
# define R__unlikely(expr) __builtin_expect(!!(expr), 0)
#endif

static const int kHeaderSize = 9;
static const unsigned char kLHC4Version = 1;

// TODO: At some point, we need to decide on settings or make it part of the level option.
static int R__LHC4Codec = static_cast<int>(lhc4codec::Codec::Beam);
static int R__LHC4Filters = 0;
static int R__LHC4FilterFallback = 1;
static int R__LHC4FilterRle = 1;
static int R__LHC4FilterDict = 1;
static unsigned long long R__LHC4FilterMask = ~0ull;
static int R__LHC4AutoMinGainPct = 1;
static int R__LHC4AutoMaxLevel = 0;
static unsigned R__LHC4AutoCodecs = kLHC4AutoCodecsRoot;
static int R__LHC4Profile = kLHC4ProfileNone;
static unsigned R__LHC4ProfileId = 0;
static int R__LHC4ProfilePerColumn = 0;
// -1: this thread follows R__LHC4Profile. Any other value is the profile for zip calls here.
static thread_local int R__LHC4CallProfile = -1;
static thread_local unsigned R__LHC4CallProfileId = 0;
// -1 / 0: this thread follows the process-wide filter knobs.
static thread_local int R__LHC4CallFilters = -1;
static thread_local int R__LHC4CallFilterFallback = -1;
static thread_local unsigned long long R__LHC4CallFilterMask = 0;
// -1: this thread follows R__LHC4Codec.
static thread_local int R__LHC4CallCodec = -1;
static int R__LHC4LastCxLevel = -1;

namespace {

#ifdef R__LHC4_FAILURE_DUMP
std::atomic<unsigned> gFailureDumpCounter{0};

const char *CodecName(int codec)
{
   switch (codec) {
   case kLHC4CodecBwt: return "bwt";
   case kLHC4CodecZstd: return "zstd";
   case kLHC4CodecBzip3: return "bzip3";
   case kLHC4CodecLzma: return "lzma";
   case kLHC4CodecAuto: return "auto";
   case kLHC4CodecMosaic: return "mosaic";
   case kLHC4CodecOracle: return "oracle";
   case kLHC4CodecCrystal: return "crystal";
   case kLHC4CodecFlash: return "flash";
   case kLHC4CodecLaser: return "laser";
   case kLHC4CodecGrating: return "grating";
   default: return "beam";
   }
}

std::string FailureDumpDirectory()
{
   static std::string dir;
   if (dir.empty()) {
      dir = std::string("/tmp/root-lhc4-failure-") + std::to_string(getpid());
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
   }
   return dir;
}

bool WriteBinaryFile(const std::string &path, const void *data, std::size_t size)
{
   if (!data || size == 0)
      return false;
   std::ofstream out(path, std::ios::binary);
   if (!out)
      return false;
   out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
   return static_cast<bool>(out);
}

void WriteMetaFile(const std::string &path, const R__LHC4FailureDumpInfo &info)
{
   std::ofstream out(path);
   if (!out)
      return;

   out << "stage=" << (info.fStage ? info.fStage : "") << '\n';
   out << "status=" << (info.fStatusMessage ? info.fStatusMessage : "") << '\n';
   out << "codec=" << CodecName(R__LHC4Codec) << " (" << R__LHC4Codec << ")\n";
   out << "cxlevel=" << info.fCxLevel << '\n';
   out << "filters=" << R__GetLHC4Filters() << '\n';
   out << "filter_fallback=" << R__GetLHC4FilterFallback() << '\n';
   out << "filter_rle=" << R__GetLHC4FilterRle() << '\n';
   out << "filter_dict=" << R__GetLHC4FilterDict() << '\n';
   out << "auto_min_gain_pct=" << R__GetLHC4AutoMinGainPct() << '\n';
   out << "auto_max_level=" << R__GetLHC4AutoMaxLevel() << '\n';
   out << "auto_codecs=0x" << std::hex << R__GetLHC4AutoCodecs() << std::dec << '\n';
   out << "profile=" << R__GetLHC4Profile() << '\n';
   out << "profile_id=" << R__GetLHC4ProfileId() << '\n';
   out << "uncompressed_size=" << info.fUncompressedSize << '\n';
   out << "compressed_lc_size=" << info.fCompressedLcSize << '\n';

   if (info.fCompressedLc && info.fCompressedLcSize >= static_cast<size_t>(kHeaderSize)) {
      const auto *lc = static_cast<const unsigned char *>(info.fCompressedLc);
      const std::size_t payload = static_cast<std::size_t>(lc[3]) | (static_cast<std::size_t>(lc[4]) << 8) |
                                (static_cast<std::size_t>(lc[5]) << 16);
      const std::size_t expected = static_cast<std::size_t>(lc[6]) | (static_cast<std::size_t>(lc[7]) << 8) |
                                   (static_cast<std::size_t>(lc[8]) << 16);
      out << "lc_payload_size=" << payload << '\n';
      out << "lc_expected_uncompressed=" << expected << '\n';
   }
}
#endif // R__LHC4_FAILURE_DUMP

} // namespace

extern "C" void R__DumpLHC4Failure(const R__LHC4FailureDumpInfo *info)
{
#ifndef R__LHC4_FAILURE_DUMP
   (void)info;
#else
   if (!info)
      return;

   const unsigned id = gFailureDumpCounter.fetch_add(1, std::memory_order_relaxed);
   const std::string base = FailureDumpDirectory() + "/" + std::to_string(id);

   WriteMetaFile(base + ".meta", *info);
   if (info->fUncompressed && info->fUncompressedSize > 0)
      WriteBinaryFile(base + ".uncompressed.bin", info->fUncompressed, info->fUncompressedSize);
   if (info->fCompressedLc && info->fCompressedLcSize > 0)
      WriteBinaryFile(base + ".compressed_lc.bin", info->fCompressedLc, info->fCompressedLcSize);

   std::cerr << "LHC4 failure dump #" << id << " written to " << base
             << ".{meta,uncompressed.bin,compressed_lc.bin} (stage="
             << (info->fStage ? info->fStage : "?") << ", uncompressed=" << info->fUncompressedSize
             << " B, compressed_lc=" << info->fCompressedLcSize << " B, codec=" << CodecName(R__LHC4Codec)
             << ", cxlevel=" << info->fCxLevel << ")\n";
#endif
}

static lhc4codec::Codec ToLHC4Codec(int codec)
{
   switch (codec) {
   case kLHC4CodecBwt: return lhc4codec::Codec::Bwt;
   case kLHC4CodecZstd: return lhc4codec::Codec::Zstd;
   case kLHC4CodecBzip3: return lhc4codec::Codec::Bzip3;
   case kLHC4CodecLzma: return lhc4codec::Codec::Lzma;
   case kLHC4CodecAuto: return lhc4codec::Codec::Auto;
   case kLHC4CodecMosaic: return lhc4codec::Codec::Mosaic;
   case kLHC4CodecOracle: return lhc4codec::Codec::Oracle;
   case kLHC4CodecCrystal: return lhc4codec::Codec::Crystal;
   case kLHC4CodecFlash: return lhc4codec::Codec::Flash;
   case kLHC4CodecLaser: return lhc4codec::Codec::Laser;
   case kLHC4CodecGrating: return lhc4codec::Codec::Grating;
   default: return lhc4codec::Codec::Beam;
   }
}

extern "C" void R__SetLHC4Codec(int codec)
{
   R__LHC4Codec = codec;
}

extern "C" int R__GetLHC4Codec(void)
{
   return R__LHC4Codec;
}

extern "C" int R__LHC4CodecAvailable(int codec)
{
   return lhc4codec::codec_available(ToLHC4Codec(codec)) ? 1 : 0;
}

extern "C" void R__SetLHC4AutoMinGainPct(int pct)
{
   R__LHC4AutoMinGainPct = pct;
}

extern "C" int R__GetLHC4AutoMinGainPct(void)
{
   return R__LHC4AutoMinGainPct;
}

extern "C" void R__SetLHC4AutoMaxLevel(int enable)
{
   R__LHC4AutoMaxLevel = enable ? 1 : 0;
}

extern "C" int R__GetLHC4AutoMaxLevel(void)
{
   return R__LHC4AutoMaxLevel;
}

extern "C" void R__SetLHC4AutoCodecs(unsigned mask)
{
   R__LHC4AutoCodecs = mask ? mask : kLHC4AutoCodecsRoot;
}

extern "C" unsigned R__GetLHC4AutoCodecs(void)
{
   return R__LHC4AutoCodecs;
}

extern "C" void R__SetLHC4Filters(int enable)
{
   R__LHC4Filters = enable ? 1 : 0;
}

extern "C" int R__GetLHC4Filters(void)
{
   return R__LHC4Filters;
}

extern "C" void R__SetLHC4FilterFallback(int enable)
{
   R__LHC4FilterFallback = enable ? 1 : 0;
}

extern "C" int R__GetLHC4FilterFallback(void)
{
   return R__LHC4FilterFallback;
}

extern "C" void R__SetLHC4FilterRle(int enable)
{
   R__LHC4FilterRle = enable ? 1 : 0;
}

extern "C" int R__GetLHC4FilterRle(void)
{
   return R__LHC4FilterRle;
}

extern "C" void R__SetLHC4FilterDict(int enable)
{
   R__LHC4FilterDict = enable ? 1 : 0;
}

extern "C" int R__GetLHC4FilterDict(void)
{
   return R__LHC4FilterDict;
}

extern "C" void R__SetLHC4FilterMask(unsigned long long mask)
{
   R__LHC4FilterMask = mask ? mask : ~0ull;
}

extern "C" unsigned long long R__GetLHC4FilterMask(void)
{
   return R__LHC4FilterMask;
}

extern "C" void R__SetLHC4Profile(int profile)
{
   if (profile < kLHC4ProfileNone || profile > kLHC4ProfileCustom)
      profile = kLHC4ProfileNone;
   R__LHC4Profile = profile;
   lhc4codec::set_thread_profile(static_cast<lhc4codec::Profile>(R__LHC4Profile));
}

extern "C" int R__GetLHC4Profile(void)
{
   return R__LHC4Profile;
}

extern "C" void R__SetLHC4ProfilePerColumn(int enable)
{
   R__LHC4ProfilePerColumn = enable ? 1 : 0;
}

extern "C" int R__GetLHC4ProfilePerColumn(void)
{
   return R__LHC4ProfilePerColumn;
}

extern "C" void R__SetLHC4ProfileId(unsigned id)
{
   R__LHC4ProfileId = id;
}

extern "C" unsigned R__GetLHC4ProfileId(void)
{
   return R__LHC4ProfileId;
}

extern "C" void R__SetLHC4CallProfile(int profile)
{
   if (profile < -1 || profile > kLHC4ProfileCustom)
      profile = -1;
   R__LHC4CallProfile = profile;
}

extern "C" int R__GetLHC4CallProfile(void)
{
   return R__LHC4CallProfile;
}

extern "C" void R__SetLHC4CallProfileId(unsigned id)
{
   R__LHC4CallProfileId = id;
}

extern "C" unsigned R__GetLHC4CallProfileId(void)
{
   return R__LHC4CallProfileId;
}

extern "C" void R__SetLHC4CallFilterTrial(int filters, unsigned long long mask, int fallback)
{
   R__LHC4CallFilters = filters;
   R__LHC4CallFilterMask = mask;
   R__LHC4CallFilterFallback = fallback;
}

extern "C" void R__SetLHC4CallCodec(int codec)
{
   R__LHC4CallCodec = codec;
}

extern "C" int R__GetLHC4CallCodec(void)
{
   return R__LHC4CallCodec;
}

static int EffectiveLHC4Profile()
{
   return R__LHC4CallProfile >= 0 ? R__LHC4CallProfile : R__LHC4Profile;
}

extern "C" int R__LHC4TrainAndRegisterProfile(unsigned profileId, const void *sample, size_t sampleSize, int stride)
{
   if (!sample || sampleSize == 0)
      return 0;
   return lhc4codec::train_and_register_profile(
             profileId, std::span<const std::byte>(static_cast<const std::byte *>(sample), sampleSize), stride)
             ? 1
             : 0;
}

extern "C" void R__SetLHC4Bwt(int enable)
{
   if (enable)
      R__LHC4Codec = kLHC4CodecBwt;
   else if (R__LHC4Codec == kLHC4CodecBwt)
      R__LHC4Codec = kLHC4CodecBeam;
}

extern "C" int R__GetLHC4Bwt(void)
{
   return R__LHC4Codec == kLHC4CodecBwt ? 1 : 0;
}

static lhc4codec::CompressParams MakeLHC4CompressParams(int cxlevel)
{
   lhc4codec::CompressParams params;
   // lhc4codec treats level 0 as the Flash codec. Keep ROOT's LHC4 levels on the
   // selected backend (1..kMaxLevel).
   if (cxlevel < 1)
      cxlevel = 1;
   if (cxlevel > lhc4codec::kMaxLevel)
      cxlevel = lhc4codec::kMaxLevel;
   params.codec = ToLHC4Codec(R__LHC4CallCodec >= 0 ? R__LHC4CallCodec : R__LHC4Codec);
   params.level = cxlevel;
   params.filters = (R__LHC4CallFilters >= 0 ? R__LHC4CallFilters : R__LHC4Filters) != 0;
   params.filter_fallback = (R__LHC4CallFilterFallback >= 0 ? R__LHC4CallFilterFallback : R__LHC4FilterFallback) != 0;
   params.filter_rle = R__LHC4FilterRle != 0;
   params.filter_dict = R__LHC4FilterDict != 0;
   params.filter_mask = R__LHC4CallFilterMask ? R__LHC4CallFilterMask : R__LHC4FilterMask;
   params.auto_min_gain_pct = R__LHC4AutoMinGainPct;
   params.auto_max_level = R__LHC4AutoMaxLevel != 0;
   params.auto_codecs = R__LHC4AutoCodecs;
   params.profile = static_cast<lhc4codec::Profile>(EffectiveLHC4Profile());
   // A per-call profile is a builtin id. The process-wide id applies only to the forced profile.
   params.profile_id = R__LHC4CallProfile >= 0 ? 0u : R__LHC4ProfileId;
   return params;
}

static void DumpZipFailure(int cxlevel, int srcsize, const char *src, const char *status)
{
   R__LHC4FailureDumpInfo dump{};
   dump.fStage = "zip";
   dump.fUncompressed = src;
   dump.fUncompressedSize = static_cast<size_t>(srcsize);
   dump.fCxLevel = cxlevel;
   dump.fStatusMessage = status;
   R__DumpLHC4Failure(&dump);
}

void R__zipLHC4(int cxlevel, int *srcsize, const char *src, int *tgtsize, char *tgt, int *irep)
{
   *irep = 0;
   R__LHC4LastCxLevel = cxlevel;
   // CompressParams carries the profile. Also set the lhc4codec thread-local default
   // so this worker matches the column (or the process-wide override).
   lhc4codec::set_thread_profile(static_cast<lhc4codec::Profile>(EffectiveLHC4Profile()));

   const auto params = MakeLHC4CompressParams(cxlevel);
   if (R__unlikely(!lhc4codec::codec_available(params.codec))) {
      std::cerr << "Error in zip LHC4: requested codec is not available in this build" << std::endl;
      DumpZipFailure(cxlevel, *srcsize, src, "requested codec is not available in this build");
      return;
   }

   const auto result = lhc4codec::compress(
      std::span<const std::byte>(reinterpret_cast<const std::byte *>(src), static_cast<size_t>(*srcsize)),
      std::span<std::byte>(reinterpret_cast<std::byte *>(&tgt[kHeaderSize]),
                           static_cast<size_t>(*tgtsize - kHeaderSize)),
      params);

   if (R__unlikely(!result.ok())) {
      const std::string msg(lhc4codec::status_message(result.status));
      if (R__unlikely(result.status != lhc4codec::Status::BufferTooSmall)) {
         std::cerr << "Error in zip LHC4: " << msg << std::endl;
         DumpZipFailure(cxlevel, *srcsize, src, msg.c_str());
      }
      return;
   }

   const size_t written = result.value;
   *irep = static_cast<int>(written + kHeaderSize);

   const size_t deflate_size = written;
   const size_t inflate_size = static_cast<size_t>(*srcsize);
   tgt[0] = 'L';
   tgt[1] = 'C';
   tgt[2] = static_cast<char>(kLHC4Version);
   tgt[3] = deflate_size & 0xff;
   tgt[4] = (deflate_size >> 8) & 0xff;
   tgt[5] = (deflate_size >> 16) & 0xff;
   tgt[6] = inflate_size & 0xff;
   tgt[7] = (inflate_size >> 8) & 0xff;
   tgt[8] = (inflate_size >> 16) & 0xff;
}

void R__unzipLHC4(int *srcsize, const unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep)
{
   *irep = 0;

   if (R__unlikely(src[0] != 'L' || src[1] != 'C')) {
      std::cerr << "R__unzipLHC4: algorithm run against buffer with incorrect header (got " << src[0] << src[1]
                << "; expected LC)." << std::endl;
      R__LHC4FailureDumpInfo dump{};
      dump.fStage = "unzip";
      dump.fCompressedLc = src;
      dump.fCompressedLcSize = static_cast<size_t>(*srcsize);
      dump.fUncompressedSize = static_cast<size_t>(*tgtsize);
      dump.fCxLevel = R__LHC4LastCxLevel;
      dump.fStatusMessage = "incorrect LC header";
      R__DumpLHC4Failure(&dump);
      return;
   }

   if (R__unlikely(src[2] != kLHC4Version)) {
      std::cerr << "R__unzipLHC4: incompatible LHC4 on-disk version (got " << static_cast<int>(src[2]) << "; expected "
                << static_cast<int>(kLHC4Version) << ")" << std::endl;
      R__LHC4FailureDumpInfo dump{};
      dump.fStage = "unzip";
      dump.fCompressedLc = src;
      dump.fCompressedLcSize = static_cast<size_t>(*srcsize);
      dump.fUncompressedSize = static_cast<size_t>(*tgtsize);
      dump.fCxLevel = R__LHC4LastCxLevel;
      dump.fStatusMessage = "incompatible LHC4 on-disk version";
      R__DumpLHC4Failure(&dump);
      return;
   }

   const auto result = lhc4codec::decompress(
      std::span<const std::byte>(reinterpret_cast<const std::byte *>(&src[kHeaderSize]),
                                 static_cast<size_t>(*srcsize - kHeaderSize)),
      std::span<std::byte>(reinterpret_cast<std::byte *>(tgt), static_cast<size_t>(*tgtsize)));

   if (R__unlikely(!result.ok())) {
      const std::string msg(lhc4codec::status_message(result.status));
      if (R__unlikely(result.status != lhc4codec::Status::BufferTooSmall)) {
         std::cerr << "Error in unzip LHC4: " << msg << std::endl;
         R__LHC4FailureDumpInfo dump{};
         dump.fStage = "unzip";
         dump.fCompressedLc = src;
         dump.fCompressedLcSize = static_cast<size_t>(*srcsize);
         dump.fUncompressedSize = static_cast<size_t>(*tgtsize);
         dump.fCxLevel = R__LHC4LastCxLevel;
         dump.fStatusMessage = msg.c_str();
         R__DumpLHC4Failure(&dump);
      }
      return;
   }

   *irep = static_cast<int>(result.value);
}

extern "C" void R__ResetLHC4CompressStats(void)
{
   lhc4codec::reset_compress_stats();
}

extern "C" void R__PrintLHC4CompressStats(void)
{
   const auto stats = lhc4codec::get_compress_stats();
   std::cout << lhc4codec::format_compress_stats(stats);
}

extern "C" void R__GetLHC4CompressStatsSummary(R__LHC4CompressStatsSummary *out)
{
   if (!out)
      return;
   const auto stats = lhc4codec::get_compress_stats();
   out->num_pages = stats.num_pages;
   out->uncompressed_bytes = stats.uncompressed_bytes;
   out->compressed_bytes = stats.compressed_bytes;
   out->filtered_won_pages = stats.filtered_won_pages;
   out->raw_won_pages = stats.raw_won_pages;
   out->no_transform_pages = stats.no_transform_pages;
   out->stored_fallback_pages = stats.stored_fallback_pages;
   out->plain_pages = stats.plain_pages;
}

extern "C" void R__GetLHC4AutoCodecStatsSummary(R__LHC4AutoCodecStatsSummary *out)
{
   if (!out)
      return;
   const auto stats = lhc4codec::get_compress_stats();
   out->auto_selections = stats.auto_selections;
   // The library counts hits only for the codecs Auto can race; its array may be
   // shorter than ROOT's enum (Grating, for instance, is never raced).
   constexpr int kLibHits = static_cast<int>(std::size(stats.auto_codec_hits));
   for (int i = 0; i < kLHC4CodecCount; ++i)
      out->codec_hits[i] = (i < kLibHits) ? stats.auto_codec_hits[i] : 0ull;
}
