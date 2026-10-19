// Original Author: ROOT Team
/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_ZipLHC4
#define ROOT_ZipLHC4

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
   kLHC4CodecBeam = 0,     ///< Native LHC4 Beam (LZ77) frames
   kLHC4CodecLz = 0,       ///< Legacy name for Beam
   kLHC4CodecBwt = 1,
   kLHC4CodecZstd = 2,
   kLHC4CodecBzip3 = 3,
   kLHC4CodecLzma = 4,
   kLHC4CodecAuto = 5,
   kLHC4CodecMosaic = 6,
   kLHC4CodecOracle = 7,
   kLHC4CodecCrystal = 8,
   kLHC4CodecFlash = 9,    ///< Byte-aligned LZ without an entropy stage (ingestion tier)
   kLHC4CodecLaser = 10,   ///< LZMA2-class binary compressor with a decoupled range decoder
   kLHC4CodecGrating = 11, ///< Order-0 integer rANS over 1/2/4/8-byte symbols; has no levels
   kLHC4CodecCount = 12,
};

/// Bit in an Auto codec mask: `1u << kLHC4Codec*`. Auto, Oracle, Flash and Grating are not raceable.
#define kLHC4CodecBit(c) (1u << (unsigned)(c))
/// Default Auto race for ROOT / RNTuple: zstd | beam | crystal (fast decoders).
#define kLHC4AutoCodecsRoot \
   (kLHC4CodecBit(kLHC4CodecZstd) | kLHC4CodecBit(kLHC4CodecBeam) | kLHC4CodecBit(kLHC4CodecCrystal))
/// Every codec Auto can race (Oracle, Flash and Grating are never included).
#define kLHC4AutoCodecsAll                                                                 \
   (kLHC4AutoCodecsRoot | kLHC4CodecBit(kLHC4CodecMosaic) | kLHC4CodecBit(kLHC4CodecBwt) | \
    kLHC4CodecBit(kLHC4CodecLaser) | kLHC4CodecBit(kLHC4CodecLzma) | kLHC4CodecBit(kLHC4CodecBzip3))

/// Pretrained Crystal static profiles (lhc4codec::Profile). None keeps dynamic per-page models.
enum {
   kLHC4ProfileNone = 0,
   kLHC4ProfileFloatKinematics = 1, ///< IEEE-754 float columns (pT, eta, phi, mass, energy)
   kLHC4ProfilePoissonCounts = 2,   ///< multiplicities, track counts, jagged offsets
   kLHC4ProfileIntMonotonic = 3,    ///< event IDs, timestamps, entry counters
   kLHC4ProfileText = 4,            ///< text, JSON, XML
   kLHC4ProfileCustom = 5,          ///< id registered with R__LHC4TrainAndRegisterProfile
};

void R__zipLHC4(int cxlevel, int *srcsize, const char *src, int *tgtsize, char *tgt, int *irep);
void R__unzipLHC4(int *srcsize, const unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);

/// Select the lhc4codec backend (Beam/Lz, Bwt, Zstd, Bzip3, Lzma, Auto, Mosaic,
/// Oracle, Crystal, Flash, Laser, Grating). Defaults to Beam.
void R__SetLHC4Codec(int codec);
int R__GetLHC4Codec(void);
/// Returns 1 when the requested backend was linked into lhc4codec, else 0.
/// Auto is always available; it races backends that are both linked and present
/// in `R__GetLHC4AutoCodecs()`.
int R__LHC4CodecAvailable(int codec);

/// Auto only: min % size gain to prefer a slower decoder (default 1). 0 = smallest wins.
void R__SetLHC4AutoMinGainPct(int pct);
int R__GetLHC4AutoMinGainPct(void);

/// Auto only: race each codec at max effort instead of the compression level (default off).
void R__SetLHC4AutoMaxLevel(int enable);
int R__GetLHC4AutoMaxLevel(void);

/// Auto only: bitmask of codecs to race (`kLHC4CodecBit`). Default is
/// `kLHC4AutoCodecsRoot` (zstd | beam | crystal). 0 resets to that default.
/// `kLHC4AutoCodecsAll` races every available flavour except Oracle, Flash and Grating.
void R__SetLHC4AutoCodecs(unsigned mask);
unsigned R__GetLHC4AutoCodecs(void);

/// Enable lhc4codec byte filters (shuffle/delta/zigzag/dict) for columnar data.
void R__SetLHC4Filters(int enable);
int R__GetLHC4Filters(void);

/// With filters: also compress raw input and keep the smaller result (default on).
void R__SetLHC4FilterFallback(int enable);
int R__GetLHC4FilterFallback(void);

/// Try post-filter byte-RLE when it shrinks the intermediate (default on).
void R__SetLHC4FilterRle(int enable);
int R__GetLHC4FilterRle(void);

/// Allow dictionary-remap filters (dict32/64) in auto-detect (default on).
void R__SetLHC4FilterDict(int enable);
int R__GetLHC4FilterDict(void);

/// Bitmask of FilterId values detection may nominate (`1ull << id`).
/// Default is all (`~0ull`). 0 resets to all. A singleton bit forces that
/// filter (no entropy reject); use with `R__SetLHC4FilterFallback(0)` to
/// measure a transform even when it loses.
void R__SetLHC4FilterMask(unsigned long long mask);
unsigned long long R__GetLHC4FilterMask(void);

/// Crystal static profile (`kLHC4Profile*`) forced on every column.
/// `kLHC4ProfileNone` (default) keeps the dynamic Crystal coder and byte filters.
/// A non-none profile skips per-page model headers.
void R__SetLHC4Profile(int profile);
int R__GetLHC4Profile(void);

/// When enabled, and no profile is forced, RNTuple selects a pretrained profile from
/// the column type. Default is off, so Crystal behaves like the other codecs.
void R__SetLHC4ProfilePerColumn(int enable);
int R__GetLHC4ProfilePerColumn(void);

/// Profile for `R__zipLHC4` calls on this thread. `-1` (default) uses `R__GetLHC4Profile()`.
/// RNTuple sets this around each column page so workers do not share one profile.
void R__SetLHC4CallProfile(int profile);
int R__GetLHC4CallProfile(void);

/// Per-thread filter trial for `R__zipLHC4`. `filters`/`fallback` of -1 and
/// `mask` of 0 follow the process-wide settings. rntupletryzip uses this so
/// workers can race different filters without clobbering each other.
void R__SetLHC4CallFilterTrial(int filters, unsigned long long mask, int fallback);

/// Codec for `R__zipLHC4` calls on this thread. `-1` (default) uses `R__GetLHC4Codec()`.
/// rntupletryzip sets this so workers can race Crystal/Laser/… without clobbering
/// the process-wide codec.
void R__SetLHC4CallCodec(int codec);
int R__GetLHC4CallCodec(void);

/// Call profile id for `R__zipLHC4` calls on this thread. 0 uses the call profile enum value.
void R__SetLHC4CallProfileId(unsigned id);
unsigned R__GetLHC4CallProfileId(void);

/// Profile id when `R__GetLHC4Profile()` is not none. 0 uses the profile enum value.
/// Custom trained profiles use an id >= 5.
void R__SetLHC4ProfileId(unsigned id);
unsigned R__GetLHC4ProfileId(void);

/// Train a custom static profile from columnar sample bytes and register it.
/// `profileId` must be >= 5. `stride` is the element width in bytes. Returns 1 on success.
int R__LHC4TrainAndRegisterProfile(unsigned profileId, const void *sample, size_t sampleSize, int stride);

/// Legacy convenience: enable BWT block mode (sets codec to Bwt/Beam).
void R__SetLHC4Bwt(int enable);
int R__GetLHC4Bwt(void);

/// Reset aggregated lhc4codec compression statistics.
void R__ResetLHC4CompressStats(void);
/// Print aggregated lhc4codec compression statistics to stdout.
void R__PrintLHC4CompressStats(void);

/// Page-outcome counters from the most recent compress-stats window.
struct R__LHC4CompressStatsSummary {
   unsigned long long num_pages;
   unsigned long long uncompressed_bytes;
   unsigned long long compressed_bytes;
   unsigned long long filtered_won_pages;
   unsigned long long raw_won_pages;
   unsigned long long no_transform_pages;
   unsigned long long stored_fallback_pages;
   unsigned long long plain_pages;
};
void R__GetLHC4CompressStatsSummary(struct R__LHC4CompressStatsSummary *out);

/// Auto-mode winner histogram (indexed by kLHC4Codec*; only populated for lhc4_auto runs).
/// Slots for Auto and Oracle are unused: Auto is the racer, Oracle is not raced.
struct R__LHC4AutoCodecStatsSummary {
   unsigned long long auto_selections;
   unsigned long long codec_hits[kLHC4CodecCount];
};
void R__GetLHC4AutoCodecStatsSummary(struct R__LHC4AutoCodecStatsSummary *out);

/// Optional inputs for R__DumpLHC4Failure (null pointers / zero sizes are omitted from dump files).
struct R__LHC4FailureDumpInfo {
   const char *fStage;            ///< e.g. "zip", "unzip", "verify_unzip", "verify_mismatch"
   const void *fUncompressed;     ///< raw page bytes before compression (may be null)
   size_t fUncompressedSize;
   const void *fCompressedLc;       ///< full ROOT LC frame (9-byte header + lhc4codec payload)
   size_t fCompressedLcSize;
   int fCxLevel;                  ///< compression level if known, else -1
   const char *fStatusMessage;    ///< human-readable error (may be null)
};
/// Write fUncompressed / fCompressedLc payloads and a .meta sidecar under /tmp/root-lhc4-failure-<pid>/.
/// No-op unless compiled with `-DR__LHC4_FAILURE_DUMP`.
void R__DumpLHC4Failure(const struct R__LHC4FailureDumpInfo *info);

#ifdef __cplusplus
}
#endif

#endif
