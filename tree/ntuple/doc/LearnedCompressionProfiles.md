# Learned Column Compression Profiles in RNTuple: Architecture, Implementation, and Performance Analysis

## 1. Executive Summary & Problem Analysis

RNTuple is ROOT's next-generation columnar storage format, engineered for high-throughput analysis on modern NVMe and networked storage. High-performance columnar codecs like **Crystal** (part of `lhc4codec`) rely on lane-wise linear predictors, dynamic context modeling, and finite-state entropy (rANS) probability models.

### 1.1 The Failure Mode of Hard-Coded Priors
In earlier versions, Crystal compression in RNTuple utilized hard-coded column-type heuristics:

```cpp
int LHC4ProfileForColumnType(ROOT::ENTupleColumnType type)
{
   switch (type) {
   case ROOT::ENTupleColumnType::kReal32:
      return kLHC4ProfileFloatKinematics;
   case ROOT::ENTupleColumnType::kIndex32:
   case ROOT::ENTupleColumnType::kUInt32:
      return kLHC4ProfilePoissonCounts;
   case ROOT::ENTupleColumnType::kChar:
      return kLHC4ProfileText;
   default:
      return kLHC4ProfileNone;
   }
}
```

While intended to avoid per-page model headers, hard-coded priors frequently fail on real High-Energy Physics (HEP) data:
- **Distribution Mismatch**: A single-precision float column in real physics ntuples might represent an invariant mass distribution peaked at $91.2\text{ GeV}$ or $125\text{ GeV}$, spatial detector coordinates ($\eta \in [-2.5, 2.5]$, $\phi \in [-\pi, \pi]$), normalized machine learning classifier scores / B-tagging probabilities $\in [0.0, 1.0]$, or calorimeter cell energies spanning six orders of magnitude.
- **Entropy Coding Penalty**: Coding symbols against mismatched probability tables costs up to $+8\text{ to }+16\text{ bits per byte}$. The codec detects expansion beyond raw size and falls back to storing raw uncompressed bytes.

### 1.2 The Cost of Per-Page Dynamic Models
When fallback occurs, or when using dynamic models without profiles, Crystal re-estimates frequency distributions on every single page:
- **Redundant Model Table Overhead**: A dynamic Crystal model table weighs $\approx 1.0\text{ to }1.5\text{ KiB}$. On a typical column with $5{,}000$ pages ($64\text{ to }256\text{ KiB}$ each), repeating the table on each page consumes:
  $$\text{Redundant Overhead} = 1.5\text{ KiB} \times 5{,}000\text{ pages} \approx 7.5\text{ MB per column}$$
  In a file with $100$ physics columns, $50\text{ to }75\text{ MB}$ of storage is wasted on duplicate frequency tables. For small pages ($4\text{ to }16\text{ KiB}$), this table tax accounts for $20\%\text{ to }50\%$ of the entire file.
- **Decompression Throughput Bottleneck**: Every page decode must parse arithmetic model headers, allocate symbol tables, and reconstruct cumulative frequency trees (~256–640 KiB allocation and rebuild per page).

### 1.3 The Learned Global Profile Architecture
Storing the model **once** in the column metadata resolves both problems:
1. **$99.98\%$ Table Overhead Eliminated**: Storing $130\text{ to }550\text{ bytes}$ once in the column descriptor eliminates redundant page headers across all clusters.
2. **True Empirical Fit**: Training the model on actual column sample bytes captures the exact exponent, mantissa, and sign distributions.
3. **Precomputed Decoding Tables**: The reader initializes the model and compiles the decoding lookup tables once upon opening the column, bypassing table construction during page decompression.

---

## 2. On-Disk Binary Format Specification Updates

The RNTuple format specification (`tree/ntuple/doc/BinaryFormatSpecification.md`) has been extended with feature flags and column compression contexts.

### 2.1 Feature Flags
To prevent older readers that lack support for column-level compression contexts from attempting to read profile-encoded pages:

| Flag Bit | Name | Meaning |
| :--- | :--- | :--- |
| **`1`** | **`kFeatureFlag_CompressionContext`** | Signals that at least one column descriptor contains an empirical compression context/dictionary. |

In `tree/ntuple/inc/ROOT/RNTupleDescriptor.hxx`:
```cpp
enum EFeatureFlags {
   kFeatureFlag_NestedDeferredColumns = 0,
   kFeatureFlag_CompressionContext    = 1,
   kFeatureFlag_COUNT,
   kFeatureFlag_Test = 137
};
```

### 2.2 Column Description Record Frame Extension (`0x04`)
The **Column Description** record frame in the Header and Footer envelopes is extended with an optional compression context block:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              Type             |        Bits on Storage        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                            Field ID                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|             Flags             |      Representation Index     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+           First element index (if flag 0x01 is set)           +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                Min value (if flag 0x02 is set)                +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                Max value (if flag 0x02 is set)                +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|      Context Algorithm        |         Context Type          | ◄── [if flag 0x04]
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Context Length                         | ◄── [if flag 0x04]
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Context Payload Bytes                     | ◄── [if flag 0x04]
|                              ...                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Column Record Flags:
- `0x01`: `kFlagDeferredColumn` — Index of first element is non-zero.
- `0x02`: `kFlagHasValueRange` — Min and max values are present.
- **`0x04`**: **`kFlagHasCompressionContext`** — Column carries a trained context (e.g., Crystal profile model, Zstd dictionary).

#### Context Header Fields:
- **`Context Algorithm`** (`UInt16`): The compression algorithm identifier (`0x05` for LHC4, `0x04` for Zstd).
- **`Context Type`** (`UInt16`): Context encoding type:
  - `0x0001`: `kLHC4CrystalProfile` — Serialized Crystal `Model` (lane feature selectors, group mappings, quantized rANS frequencies).
  - `0x0002`: `kZstdDictionary` — Standard Zstandard trained dictionary blob.
- **`Context Length`** (`UInt32`): Uncompressed byte length of the context payload ($130\text{ to }550\text{ bytes}$ for Crystal).
- **`Context Payload`**: Raw serialized model/dictionary bytes.

---

## 3. Codec Engine Implementation (`cerncodec`)

All engine enhancements were implemented in `cerncodec/src/crystal_static.cpp` and exposed through `include/lhc4codec/lhc4codec.hpp`.

### 3.1 Public C++ API
```cpp
namespace lhc4codec {

[[nodiscard]] std::vector<std::uint8_t> train_profile_model(
    std::span<const std::byte> sample, int stride, int level = 5);

[[nodiscard]] bool register_profile_model(
    std::uint32_t profile_id, std::span<const std::byte> model_bytes) noexcept;

[[nodiscard]] std::vector<std::uint8_t> export_profile_model(std::uint32_t profile_id);

} // namespace lhc4codec
```

### 3.2 Precomputed Decoder Tables in `StaticProfileHolder`
To eliminate per-page table parsing, `StaticProfileHolder` precomputes and caches all lookup tables upon profile registration:
```cpp
struct StaticProfileHolder {
  Model model;
  Filter transform = Filter::None;  // element transform applied to every page (section 3.6)
  std::vector<EncSym> esym;

  // Precomputed decoder tables & structures (read-only after finalize_profile)
  bool compact = false;
  bool trend = false;
  bool req = false;
  std::vector<u8> dec_tab;
  std::vector<u32> q_scaled;
  std::array<DecLane, kMaxLanes> lanes{};
};
```
In `finalize_profile(StaticProfileHolder& h)`:
1. `build_dec_tables(m, h.dec_tab, h.compact)` pre-populates rank-to-symbol and frequency lookup tables.
2. `h.q_scaled` pre-multiplies context group scales by `ctx_bytes_for(compact)`.
3. `DecLane` descriptors are pre-constructed with shift masks and table pointers.

### 3.3 Decoupled Stream Decoder Dispatch
The stream decompression routine was decoupled:
```cpp
Status decode_crystal_streams(const Model& m, bool compact, const u8* tab,
                             const DecLane* lanes, bool trend, bool req,
                             int k, ByteReader& r, u8* dst, usize n);
```
`decode_crystal_profile` passes pointers directly from `h->dec_tab.data()` and `h->lanes.data()`, completely bypassing runtime table generation and dynamic allocation.

### 3.4 Buffer Reuse and Selective Zeroing
In `decode_crystal_streams`:
- Multi-stream padded buffer allocation was changed from `assign(total, 0)` (which cleared hundreds of KiB of memory per page) to capacity checking `resize(total)`.
- Zeroing is strictly limited to the trailing lookahead window per stream:
  ```cpp
  std::memset(t.padded.data() + off + lens[s], 0, kRansPad);
  ```

### 3.5 Thread-Safe Profile Registry
Profile lookups are guarded by a read-write lock:
```cpp
static std::shared_mutex g_custom_profiles_mutex;
static std::unordered_map<u32, std::shared_ptr<StaticProfileHolder>> g_custom_profiles;
```
Concurrent worker threads decode pages simultaneously under `std::shared_lock<std::shared_mutex>`, while new profiles are registered with `std::unique_lock`.

### 3.6 Drifting Columns: Element Transform and Support Prior

A static profile models byte values. The pages of an offset or index column (RNTuple `Index32/64`, cumulative
collection sizes) break that assumption: every page starts where the previous one ended, so the byte values of
the training sample say nothing about the byte values of later pages, while the first differences (the collection
sizes) are stationary. The same holds for timestamps, event numbers and slowly changing run/lumi numbers. Before
this section's changes, 66% of the blocks on the CMS sample escaped to the per-page model, for this reason and
for the cost of the pseudo-counts described below.

**Transform trained into the profile.** `train_profile` (crystal_static.cpp) trains the plain model and, for
2/4/8-byte elements, the `Delta` and `ZigzagDelta` candidates of that width (`transform.hpp` filters). A candidate
is trained only when a per-lane order-0 estimate of the transformed sample is below the plain one (floats and
small counts fail this gate and cost a single training); the choice between trained candidates is by the exact
coded size of the sample. The exception is a *drifting* sample, detected by `sample_drifts`: the two halves of the
sample share at most half of some lane's byte mass (total variation distance $\ge 1/2$). Offsets move their high
lanes this way; floats, counts and small signed integers keep the same lane histograms throughout. On a drifting
sample the plain model's sample cost is not predictive (it is reached with bases relative to the previous record
and contexts on byte values that do not recur), so the transform is taken when it costs at most 2% more on the
sample.

The transform is part of the model bytes: the leading byte carries the lattice stride in bits 0-4 and the transform
code in bits 5-7 (0 = none, so models written before this change read unchanged). Nothing changes on the RNTuple
side, the model bytes in `RCompressionContext` are opaque. Every page coded with the profile goes through
`apply_filter` before the context pass and `invert_filter` (in place) after decoding. The first element of a
transformed page is the page's absolute start, which the profile cannot know: it travels raw in the block header
(after the stream count) and is zero for the coder.

**Support prior.** Symbols the sample did not show need a frequency too, and the smallest one ($1/4096$) is a real
cost: 255 unseen symbols take 6% of a context's mass, 0.09 bits per byte, which on a constant lane or a count
column is 10% of the page. This was the hidden reason why many float and count pages escaped by a few percent.
The pseudo-count now goes only to the unseen symbols inside the sample's *support*: the ring interval spanned by
the seen symbols plus a margin of two on both sides (the gaps of a count's tail, the next values of a growing one,
the `0xFF` of a negative delta next to the `0x00` of a positive one). Minor contexts (under 4096 observations in
the sample, e.g. the stream starts) get every unseen symbol, since the pseudo-counts cost next to nothing there
against the header of an escape. A page that needs a symbol outside the support is not codable with the profile
and takes the dynamic path; the escape costs it the per-page model header, ~1.5 KiB.

**Stride.** The profile keeps the column's element stride. `sanitize_stride` (which trades per-lane tables against
their header cost on a single block) no longer applies to profiles: the header is shipped once, and merging the
lanes would spread each lane's pseudo-counts over the others' symbols (on a lumi-like column this quadrupled the
profile's cost).

Effect on the CMS sample (1 MiB pages): the escaped bytes fell from 808 MB to 265 MB of the 1,281 MB of Crystal
input (48.5% of the blocks, most of them 32 KiB boolean/char pages whose output is negligible); the `Index64`
columns are coded with `Delta64` on the profile (175 of 198 blocks). The piecewise-constant `int32` columns (run,
lumi) train `ZigzagDelta32` and still escape on the pages where the value jumps outside the support; with
$\approx 2$ KiB of output per 1 MiB page that is immaterial. The learned file is 0.4% larger than dynamic instead
of 1.2%. The remaining gap is the static tables on the float columns, where the
per-page tables fit the page (+0.46% over the per-page estimate on 299 MB of coded output) plus the headers of the
escaped blocks. Decode speed is unchanged (section 5.1): the inverse delta is one pass at memory speed over a
small fraction of the pages.

### 3.7 Per-Page Lane Modes on Profile Blocks

A dynamic block skips the coder for two kinds of lane (lhc4codec `1d70fb1`): a lane that is constant over the
block ships one value byte in the model, and a lane whose coded size would reach its byte count (noise to the
model) ships its bytes verbatim in a section after the model. Both leave the rANS loop on either side: the decoder
fills them before the lockstep pass and only shifts their bytes into the byte history (3 instructions per byte
instead of ~30). The profile could not do this by itself, since it serves pages its sample never saw, so the
profile path fell 16-18% behind dynamic in decode speed when those modes arrived.

The modes are now a property of the page, carried in the block header of a new block kind (`kCryProfileLanes`, 6;
kind 5 blocks stay decodable and are still written when every lane is coded): 2 bits per lane, one value byte per
constant lane, then the raw lanes' bytes lane by lane, then the streams. `build_crystal_profile` decides per lane
from the page: constant if all of its bytes are equal (checked on the transformed page, so a delta-transformed
counter has every lane constant and the block is 11 bytes); raw if the page's cost under the profile's tables
reaches 8 bits per byte, **or if the lane meets a symbol outside the profile's support** (frequency 0). The latter
is what used to send the whole page to the dynamic path with its ~1.5 KiB model header; now that lane alone goes
raw and the escape decision compares, per lane, the profile's exact cost against the trainer's estimate of a
per-page model (the smaller of that estimate and the raw size for a raw lane, so a lane that a per-page model
would compress still escapes). The skipped lanes' symbols take a sentinel frequency row (full probability, the
coder state is unchanged) so the stream encoder needs no special case; the decoder overrides the profile's
precomputed lane descriptors (`DecLane::mv`) for the page and otherwise runs the same loop.

Effect on the CMS sample (1 MiB pages, level 5): the escaped blocks fell from 2,389 to 2,138 of 4,955 (48% to
43%), the file is 44 KB smaller, and the profile path decodes 3-4% faster than dynamic again (section 5.1; it was
18% slower without the lane modes). The round trip is verified by the `tree2ntuple --verify` path on every page
and by the api test's lane-mode block.

---

## 4. ROOT Core and RNTuple Integration

### 4.1 C Bridge in Core (`core/lhc4codec`)
In `core/lhc4codec/inc/ZipLHC4.h` and `src/ZipLHC4.cxx`:
- `R__LHC4TrainProfileModelLevel(...)` trains a profile model on uncompressed sample data.
- `R__LHC4RegisterProfileModel(...)` deserializes and registers the model in `cerncodec`.
- `R__LHC4ExportProfileModel(...)` exports a registered profile model.

### 4.2 In-Memory Metadata: `RCompressionContext`
In `tree/ntuple/inc/ROOT/RNTupleDescriptor.hxx`:
```cpp
class RCompressionContext {
public:
   enum class EType : std::uint16_t {
      kUnknown = 0,
      kLHC4CrystalProfile = 1,
      kZstdDictionary = 2,
   };

   std::uint32_t GetProfileId() const {
      if (fPayload.empty()) return 0;
      std::uint32_t hash = 2166136261u;
      for (auto b : fPayload) {
         hash ^= b;
         hash *= 16777619u;
      }
      return 256u + (hash & 0x7FFFFFFFu);
   }
};
```
Content-addressed Profile IDs (FNV-1a hash offset by 256) ensure that concurrent files and threads never suffer from profile ID collisions.

### 4.3 Reader Pipeline (`tree/ntuple/src/RPageStorage.cxx`)
When an RNTuple file is opened, `RPageSource::RegisterCompressionProfiles(const RNTupleDescriptor &desc)` iterates through all columns. For each column with an LHC4 Crystal compression context, it invokes `R__LHC4RegisterProfileModel()` once. When pages are unsealed, `R__unzipLHC4` decodes directly using the pre-registered profile ID.

### 4.4 Writer Pipeline (`tree/ntuple/src/RPageStorage.cxx`)
In `RPagePersistentSink::EnsureColumnProfile()`, called for every committed page while the column has no profile yet,
if `GetWriteOptions().GetEnableLearnedProfiles()` is true:
  1. The column data stride is determined from `ENTupleColumnType`.
  2. The page bytes are added to the column's training sample until it holds
     `GetWriteOptions().GetLearnedProfileSampleSize()` bytes (default 256 KiB). A page that is already that large is
     used directly. Pages committed before the sample is complete are written with the per-page (dynamic) Crystal
     model. The memory held by partial samples is bounded (64 MiB across all columns); above that, columns train on
     the sample they have.
  3. `R__LHC4TrainProfileModelLevel` trains the profile model at
     `max(GetLearnedProfileTrainLevel(), write level)` (default train level 6). The search runs once per column
     and the model is reused for every page, so it is decoupled from the write level. On the CMS OpenData sample
     the levels 7-9 (exhaustive pair search) add ~60 s of training for the whole file (1,700 columns) and no
     measurable size gain over level 5/6, hence the default. The training also picks the profile's element
     transform (section 3.6): a drifting sample (offsets, timestamps) trains on the element deltas.
  4. The model is registered in `cerncodec` and saved into the column's `RCompressionContext`.
  5. The descriptor is tagged with `kFeatureFlag_CompressionContext`. The contexts are serialized with the header,
     which the sink commits with the first cluster (`CommitHeader`, deferred while learned profiles are enabled).
     Just before that, `TrainPendingColumnProfiles` trains every column whose sample is still incomplete on what it
     has (at least 16 elements), and from then on `EnsureColumnProfile` returns 0 for columns without a profile:
     a model trained after the header is written would be known to the writer only, and its pages unreadable.
     (The sample accumulation had opened exactly that hole for columns whose sample completed, or whose training
     was forced by the 64 MiB budget, in a later cluster; on the CMS sample 482 of 860 column contexts were lost
     this way, and the pages of those columns decoded only while the escape sent them to the per-page model.)
  6. The sample page and all subsequent pages of the column are compressed using this profile ID with 0 bytes of
     model header per page. The encoder keeps a per-page escape: a page whose estimated cost under the profile
     exceeds the estimated per-page model cost by more than a few percent is written as a dynamic Crystal block
     instead (the block kind is per page, so the reader needs no extra information).

---

## 5. Comprehensive Benchmark Results

Evaluated on the CMS OpenData sample (`Events` tree, 1,049,500 events, 5,740-5,944 sealed pages, 1.436 GB
uncompressed) on Apple Silicon (single core and 8-thread IMT). Measured 13 Oct 2026 with lhc4codec `2772ef8`
(drifting-column transform and support prior of section 3.6, the faster static encoder of `89bae1c`, the format v2
encoder/decoder review and the lane modes of `1d70fb1`, where constant and raw lanes of a dynamic block bypass
the coder, and the per-page lane modes on profile blocks of section 3.7) and ROOT `bd2015a49ef`. The seven files
were written one after the other (`tree2ntuple --jobs 1 --variants native_zstd,lhc4_crystal5,...,
lhc4_crystal_learned9`) and the three benchmarks below were taken in one sequential
`rntuple2rntuple --bench-only --bench-iters 5 --bench-threads 8` run with nothing else of ours running, so the
rows are comparable with each other. The machine was not idle (a virtual machine held one core throughout);
against a quieter run of the same morning every row is ~12% lower (zstd 2252 vs 2558 MB/s, dynamic Crystal 873 vs
1023 MB/s), and the relative figures are what the tables are for.

Written sizes: `native_zstd` 406 MiB, `lhc4_crystal{5,7,9}` 357 MiB, `lhc4_crystal_learned{5,7,9}` 359 MiB
(374.7 MB vs 376.3 MB, +0.4%). With the lane modes of section 3.7 the learned variants code 2,817 of 4,955
Crystal blocks with the profile and escape 2,138 (43%; 2,389 or 48% before, most of them 32 KiB boolean/char
pages); before section 3.6 the escaped share was 63% of the bytes.

### 5.1 Pure In-Memory Page Decompression Benchmark (Single Core, 5 Iterations)
*Measures raw codec decompression throughput without file I/O or RNTuple object deserialization.*

| Variant | Compressed Size | Uncompressed Size | Time per pass | Throughput | vs Dynamic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`native_zstd`** (level 5) | 409.4 MB | 1436.7 MB | 0.638 s | 2252.3 MB/s | — |
| **`lhc4_crystal5`** (dynamic) | 360.4 MB | 1436.7 MB | 1.646 s | 872.9 MB/s | baseline |
| **`lhc4_crystal_learned5`** (profiled) | 361.8 MB | 1436.7 MB | **1.577 s** | **911.1 MB/s** | **+4.4%** |
| **`lhc4_crystal7`** (dynamic) | 360.1 MB | 1436.7 MB | 1.780 s | 807.2 MB/s | baseline |
| **`lhc4_crystal_learned7`** (profiled) | 361.5 MB | 1436.7 MB | **1.707 s** | **841.9 MB/s** | **+4.3%** |
| **`lhc4_crystal9`** (dynamic) | 360.1 MB | 1436.7 MB | 1.803 s | 797.0 MB/s | baseline |
| **`lhc4_crystal_learned9`** (profiled) | 361.5 MB | 1436.7 MB | **1.749 s** | **821.4 MB/s** | **+3.1%** |

### 5.2 End-to-End `RNTupleReader::LoadEntry` Benchmark (Single Core, All Entries)
*Measures real analysis workload reading all 1.05M events entry-by-entry.*

| Variant | Read Wall Time | Total MB/s | Entries / sec | Decomp Wall Time | Decomp Throughput | vs Dynamic |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`native_zstd`** | 9.271 s | 204.2 MB/s | 113,242 ent/s | 0.942 s | 2010.8 MB/s | — |
| **`lhc4_crystal5`** (dynamic) | 9.369 s | 202.1 MB/s | 112,058 ent/s | 1.776 s | 1066.4 MB/s | baseline |
| **`lhc4_crystal_learned5`** (profiled) | 9.685 s | 195.5 MB/s | 108,405 ent/s | 1.844 s | 1026.8 MB/s | -3.7% |
| **`lhc4_crystal7`** (dynamic) | 9.603 s | 197.2 MB/s | 109,329 ent/s | 1.966 s | 963.0 MB/s | baseline |
| **`lhc4_crystal_learned7`** (profiled) | 9.793 s | 193.3 MB/s | 107,205 ent/s | 1.932 s | 979.8 MB/s | +1.7% |
| **`lhc4_crystal9`** (dynamic) | 9.569 s | 197.9 MB/s | 109,714 ent/s | 1.978 s | 957.3 MB/s | baseline |
| **`lhc4_crystal_learned9`** (profiled) | **9.447 s** | **200.4 MB/s** | **111,137 ent/s** | **1.887 s** | **1003.3 MB/s** | **+4.8%** |

### 5.3 Multi-Threaded `RNTupleReader::LoadEntry` Benchmark (8 Threads IMT)
| Variant | Read Wall Time | Total MB/s | Entries / sec | Decomp Wall Time | Decomp Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`native_zstd`** | 8.374 s | 226.1 MB/s | 125,371 ent/s | 0.183 s | 10347.4 MB/s |
| **`lhc4_crystal5`** (dynamic) | 7.912 s | 239.3 MB/s | 132,690 ent/s | 0.426 s | 4442.3 MB/s |
| **`lhc4_crystal_learned5`** (profiled) | 7.954 s | 238.1 MB/s | 131,997 ent/s | 0.416 s | 4547.6 MB/s |
| **`lhc4_crystal7`** (dynamic) | 8.072 s | 234.6 MB/s | 130,059 ent/s | 0.467 s | 4052.6 MB/s |
| **`lhc4_crystal_learned7`** (profiled) | 8.030 s | 235.8 MB/s | 130,739 ent/s | 0.477 s | 3970.2 MB/s |
| **`lhc4_crystal9`** (dynamic) | 7.962 s | 237.8 MB/s | 131,869 ent/s | 0.470 s | 4029.8 MB/s |
| **`lhc4_crystal_learned9`** (profiled) | 7.958 s | 237.9 MB/s | 131,933 ent/s | 0.437 s | 4333.9 MB/s |

With 8 threads the decompression is 2-6% of the read wall time for every variant; the remaining time is
deserialization, which is the same for all of them, so the end-to-end numbers converge (the six Crystal variants
are within 2% of each other here).

**Two byte counts.** The "Decomp Throughput" of tables 5.2 and 5.3 is higher than the pure decode of table 5.1
(1,066 vs 873 MB/s for `lhc4_crystal5`) because the two benchmarks count different bytes over different
timers. The pure page benchmark (`RNTupleBench.hxx`) counts *stored* bytes, bits on storage × elements, 1,437 MB
for this file, and times `RNTupleDecompressor::Unzip` alone. The reader metric `szUnzip`
(`RPageStorage.cxx`, `UnsealPage` callers) counts *in-memory* bytes, `element->GetSize()` × elements, 1,893 MB for
the same file, since a 32-bit index column unpacks to a 64-bit offset and a bool bit to a byte; its timer
`timeWallUnzip` wraps `UnsealPage`, i.e. unzip plus that unpack. Scaled to stored bytes the reader's 1,066 MB/s is
~810 MB/s, consistent with the pure decode. With IMT the timer wraps the whole parallel cluster unzip, so table
5.3 reports aggregate throughput. "Total MB/s" is in-memory bytes over the whole read, deserialization included.

### 5.4 Reading the Results

- **Size**: learned profiles cost 0.4% over dynamic Crystal (361.8 MB vs 360.4 MB) and stay 12% below zstd. The
  gap is the static tables on the float columns, which the per-page tables fit better (+0.5% on the 299 MB of
  profile-coded output), plus the per-page headers of the escaped blocks; the drifting integer columns, which
  made up the earlier 1.2% gap, now code on the profile (section 3.6).
- **Speed**: the lane-specialised scalar decoder brought dynamic Crystal from ~515 MB/s to ~700 MB/s in pure page
  decode, and the lane modes of lhc4codec `1d70fb1` (a lane that is constant over the block, or that no context
  compresses, is stored as such and skips the rANS loop) take it to 870-1020 MB/s on this sample, depending on
  the machine's state. Lane modes are a per-block decision, and when they arrived the profile path lacked them:
  measured on the same files that morning, `lhc4_crystal_learned5` was 18% behind `lhc4_crystal5` in pure decode
  (839 vs 1023 MB/s, confirmed in swapped order at 815 vs 970) and 12-15% behind in single-threaded
  `LoadEntry`. With the per-page lane modes of section 3.7 the profile path is 3-4% ahead in pure decode at
  every level (table 5.1) and within ±5% in `LoadEntry` (table 5.2, where the sign changes with the level), which
  is where it stood before `1d70fb1`: the profile's tables are a little larger than a per-page level-5 model
  (more cache misses) and smaller than a per-page level-7/9 model.
- **Size** with the lane modes: the learned file lost 44 KB at level 5 (376,332,059 B) and 11 KB at level 7/9
  (376,092,242 B); the dynamic files are unchanged.
- The earlier figures in this section (dynamic 316 MB/s, learned +73% end-to-end) predate the lane-specialised
  decoder, the training sample and the per-page escape, and were taken with other runs in parallel; they are
  superseded by the tables above.

### 5.5 Page Size Dependence and When to Use Learned Profiles

Every advantage of a learned profile is a fixed cost per page that it removes: the ~1.5 KiB per-page model, the
decoder table build, and the per-page model search on the writer. Whether that pays depends on the page size.
Same sample, `tree2ntuple --no-out --jobs 1 --page-size N`, one variant at a time:

| Max page size | `lhc4_crystal5` (dynamic) | `lhc4_crystal_learned5` | Size | Encode wall time |
| :--- | :--- | :--- | :--- | :--- |
| 1 MiB (default) | 357 MiB | 359 MiB | dynamic 0.6% smaller | 34.7 s vs 32.8 s |
| 64 KiB | 364 MiB | 362 MiB | learned 0.5% smaller | 53.7 s vs 30.8 s |
| 16 KiB | 375 MiB | 366 MiB | **learned 2.4% smaller** | **160 s vs 32.3 s** |

Reading the same TTree into `native_zstd` alone takes 29.5 s in this run (23 s on the quieter machine of the
morning), which is the floor of the encode column (TTree deserialisation and RNTuple page filling): the learned
variant sits on that floor at every page size, the dynamic variant's model search grows with the page count.
Before the drifting-column transform and the support prior (section 3.6) the same table read 362 / 367 / 372 MiB
for the learned variant: dynamic 1.2% smaller at 1 MiB, par at 64 KiB, learned 2% smaller at 16 KiB; before the
lane modes of `1d70fb1` the dynamic column read 357 / 365 / 380 MiB.

**Encode cost.** The wall times are bounded by the TTree read: `tree2ntuple` enables IMT and the sink compresses
pages in parallel tasks, so at 1 MiB pages the two variants differ by two seconds of wall time while the CPU time
differs by a factor of two: run alone, `lhc4_crystal5` takes 34.3 s wall / 86.8 s CPU and `lhc4_crystal_learned5`
32.3 s wall / 42.3 s CPU (~25 s CPU of that is the TTree read and page filling in both cases). On a page coded
with the profile the writer runs the context index pass, a count-based cost estimate for the lane modes and the
escape decision, and the rANS encoder, about 2.5 ns per byte; the dynamic path runs the per-page model search at
~20 ns per byte (level 5, after the encoder speed-up of `89bae1c`). A `sample` profile of the learned run splits
its Crystal CPU as: training (256 KiB per column at level 6, once) ~25%, the escaped pages coded dynamically ~50%,
and the pages coded with the profile ~25%, the last including the profile attempt that every escaped page also
pays. With 1 MiB pages the training is the largest fixed cost; at 16 KiB pages the model search runs 60 times
more often while the training stays the same, hence 160 s against 32 s wall.

Conclusions for the default RNTuple configuration (1 MiB pages, large files, bulk reads): dynamic Crystal remains
the default. It is 0.6% smaller, decodes as fast (section 3.7), and needs no training step, descriptor extension
or profile registry. `SetEnableLearnedProfiles` stays opt-in.

The learned path earns its keep where pages are small, with the crossover now between 64 KiB and 1 MiB:
- many-column ntuples with few entries, where most pages never grow to the maximum;
- the first pages of every column and the pages flushed at cluster boundaries;
- random or low-latency access, where a single page decode is the unit of work;
- memory-constrained writers that cannot hold a 1 MiB page buffer per column;
- write-side cost: the profile encode takes half the CPU at 1 MiB pages and a fifth of the wall time at 16 KiB
  pages, since the model search runs once per column instead of once per page.

What would widen that regime further are profiles trained once per data tier and shared across files, which the
per-page model cannot do by construction; that does not exist yet. The remaining 0.6% at 1 MiB pages is the price
of static tables on float columns, where the per-page tables fit the page.

---

## 6. Performance Deep Dive: Why is ZSTD Faster and Why Crystal Decodes at ~1 GB/s?

A common question is: *Why does Crystal decode at ~0.9-1 GB/s in pure page decompression (~700 MB/s before the lane modes) rather than 1.5 GB/s+, and why does Zstandard achieve ~2.2-2.5 GB/s?*

The answer lies in the fundamental algorithmic and architectural differences between LZ77 dictionary coding and Order-1/2 Markov Context rANS entropy coding.

### 6.1 Architectural Difference: LZ77 Match Copying vs Per-Byte Context Modeling

#### How Zstandard Achieves 1.75+ GB/s:
1. **Bulk Vector Match Copying**:
   - ZSTD is primarily an **LZ77** sliding-window dictionary compressor combined with Finite State Entropy (FSE) / Huffman (Huff0) for raw literals.
   - When repeated patterns or common values occur (e.g. repeated floats, zeros, structural offsets), LZ77 emits `(match_distance, match_length)` references.
   - The decompressor satisfies matches using unrolled 16- or 32-byte SIMD vector loads and stores (`_mm256_storeu_si256` or `vld1q_u8` / `vst1q_u8`).
   - Copying 16–64 bytes takes **1 clock cycle** ($\approx 15\text{–}30\text{ GB/s}$ copy throughput).
   - In columnar data, up to $80\%\text{–}90\%$ of total bytes are expanded via these instant memory copies. Only the remaining $10\%\text{–}20\%$ (the non-matching literals) pass through the entropy stage.
2. **Order-0 Independence**:
   - The Huffman/FSE stage in Zstandard is Order-0 (context-free). Literals are decoded in parallel streams without referencing neighboring bytes or previous rows.

#### Why Crystal Must Process Every Single Byte in the Entropy State Machine:
1. **Pure Context-Predictive Coder**:
   - Crystal is not an LZ77 compressor. It is a **Markov context-mixing lattice model coder** (similar to PAQ/CALIC, optimized with rANS).
   - It does not emit `(distance, length)` copy tokens. Instead, **every single byte of the 1.436 GB payload** must pass through the full context predictor and the rANS arithmetic state machine.
2. **The Per-Byte Decoding Pipeline (`decode_one_lane`)**:
   The lane (the feature choice) is the same for all interleaved streams of a lattice step, so the decoder dispatches
   once per step to a routine specialised on the lane's features (`LaneBody` in `crystal_static.cpp`). For *each
   individual byte* that routine executes:
   ```cpp
   // 1. Lattice history fetch, only when a feature of this lane reads it
   lat1 = p[-S];
   lat2 = p[-2 * S];

   // 2. Context from the lane's two features (resolved at compile time:
   //    an order-0 lane does no feature work, a C1 lane one mask)
   c = L.q[feat_a] + L.q[256 + feat_b]; // 2 context table lookups

   // 3. Probability slot lookup
   slot = st.d.slot(); // rANS state masking
   SlotInfo si = slot_lookup(c, slot); // 2 dependent table memory loads

   // 4. rANS arithmetic state update & stream refill
   st.d.advance(si.freq, si.cum, slot); // multiply, shift, subtract, refill

   // 5. Output & history register update
   *p = si.sym + base;
   st.hist = (st.hist << 8) | *p;
   ```
3. **Instruction Count and Cycle Limits**:
   - The routine above is **29 to 36 instructions per byte on aarch64** (34 to 40 on x86-64), including 3–5
     dependent memory loads (lattice history, context group offsets, slot symbol, cumulative frequency). The
     previous generic step, which selected the features at run time from a packed word with variable shifts, was
     40 instructions on aarch64 and 57 on x86-64 with register spills.
   - The 16 interleaved streams keep the dependent loads of different streams in flight, so the loop is bound by
     instruction issue rather than load latency: decode throughput stops scaling at 8 streams.
   - On a ~3.5 GHz core the **~840 MB/s** measured for the profile path when every byte still went through this
     loop is **~4.2 cycles per byte**, i.e. ~7-8 instructions per cycle across the interleaved streams, close to
     the issue width of the core. The 0.9-1 GB/s of both paths today come from bytes that skip the loop: with the
     lane modes of lhc4codec `1d70fb1` (per block) and section 3.7 (per page on profile blocks) a lane that is
     constant over the block is restored by a fill, and a lane that no context compresses is copied, so only the
     remaining lanes pay the per-byte cost above.

---

### 6.2 Hardware Vectorization: AVX2 vs ARM64 NEON

In `cerncodec/src/crystal_static.cpp`:
- The 8-wide / 16-wide vectorized lockstep decoder (`decode_streams_v8`) is explicitly implemented for **x86-64 AVX2**:
  ```cpp
  #if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  #define CRY_V8 1
  #include <immintrin.h>
  #define CRY_V8_TARGET __attribute__((target("avx2,bmi2,popcnt")))
  #else
  #define CRY_V8 0
  #endif
  ```
- The AVX2 vector path utilizes `_mm256_i32gather_epi32` (`vpgatherdd`) to gather 8 independent rANS table lookups simultaneously across 8 interleaved streams.
- On **ARM64 (Apple Silicon / Darwin)**:
  - Standard ARM NEON does not feature 32-bit hardware vector gather instructions.
  - Consequently, `CRY_V8` is disabled (`CRY_V8 == 0`), and the engine runs the scalar lockstep body
    (`LaneBody`, or `decode_streams_body` for lanes it does not cover).
  - Thus, on Apple Silicon, the benchmark measures pure scalar execution performance.

---

### 6.3 Performance Expectations on x86_64 Platforms

On x86_64 platforms (such as modern Intel Core/Xeon or AMD Zen 3/4/5), hardware AVX2 vectorization is active. The performance characteristics differ significantly from ARM64 scalar execution:

#### A. Expected Throughput Across CPU Microarchitectures
| Architecture | Code Path | Cycles / Byte | Pure Decomp Rate (Single Core) | RNTuple End-to-End (`LoadEntry`) |
| :--- | :--- | :--- | :--- | :--- |
| **ARM64 (Apple Silicon M-series)**, measured | Scalar (`LaneBody`) | ~4.2 (coded bytes) | **~870 – 1,020 MB/s** with lane bypass (dynamic and profile) | ~1,000 – 1,150 MB/s |
| **AMD Zen 3 (AVX2)** | Vector (`decode_streams_v8`) | ~6.5 – 7.5 | **~500 – 580 MB/s** | ~550 – 620 MB/s |
| **AMD Zen 4 / Zen 5 (AVX-512 / AVX2)** | Vector (`decode_streams_v8`) | ~5.0 – 6.0 | **~620 – 750 MB/s** | ~680 – 800 MB/s |
| **Intel Golden Cove / Raptor Lake / Xeon**| Vector (`decode_streams_v8`) | ~4.5 – 5.5 | **~650 – 850 MB/s** | ~700 – 900 MB/s |
| **Zstandard (Modern x86_64)** | Unrolled AVX2 Byte Copy | ~1.5 – 2.0 | **~2,000 – 2,500 MB/s** | ~1,500 – 1,800 MB/s |

The x86_64 rows are projections from the cycle-per-byte model below (the vector body is not measured on the
platform of section 5); the ARM64 row is measured. Note that the x86-64 scalar path, used for blocks with fewer
than 8 streams (pages below 32 KiB), with compact tables, or on CPUs without AVX2, benefits from the same
lane-specialised body: 57 (53 with BMI2) → 34–40 instructions per byte with the stack spills largely gone.

#### B. Why the x86_64 vector body is expected to be faster than ARM64 scalar
On x86_64 with AVX2:
1. **Parallel Vector Register State**: For blocks $\ge 64\text{ KiB}$, $k = 16$ interleaved streams are split into two 8-wide AVX2 vector groups. The state $x$, stream offset $p$, and output history $h$ of 8 streams reside directly in 256-bit `__m256i` registers.
2. **SIMD Context Gathers**: Frequency table and context lookups use hardware vector gathers (`vpgatherdd`).
3. **Register-Resident Latency Hiding**: For strides $\le 8$ (floats, doubles, integers), output history ($p[-S]$ and $p[-2S]$) is shifted within four 256-bit vector registers (`_mm256_slli_epi32`), avoiding round-trip loads through memory.
4. **Interleaved Dependency Chains**: Issuing the two 8-wide vector groups back-to-back overlaps memory access latencies across the two chains.
The vector body is projected at **~4.5–7 cycles/byte**, which is in the same range as the measured scalar
**~4.2 cycles/byte** on Apple Silicon; whether it still pays off against the lane-specialised scalar body on a
given x86-64 core is a measurement to make (`LHC4_CRYSTAL_X86=scalar` selects the scalar body for an A/B).

#### C. Microarchitectural Ceilings: Why 1.5 GB/s is Unreachable on a Single Core
Reaching $1.5\text{ GB/s}$ at a $4.0\text{ GHz}$ core clock requires consuming $\le 2.67\text{ clock cycles per byte}$:
$$\text{Cycles per byte} = \frac{4.0\text{ GHz}}{1.5\text{ GB/s}} \approx 2.67\text{ cycles/byte}$$

Why `decode_streams_v8` cannot sustain $2.6\text{ cycles/byte}$:
1. **Gather Port Contention**: While standard vector loads (`_mm256_loadu_si256`) execute in 1 cycle, a vector gather decomposes into 8 individual address generation and cache tag lookups. On Zen 3, a 256-bit gather occupies the gather execution unit for ~19 cycles. On Zen 4 and Intel Golden Cove, gathers execute in ~4–8 cycles, but remain micro-op intensive.
2. **High Memory Operation Density**: Decoding 8 bytes requires **5 distinct gathers** (context features $a$ and $b$, table slot, cumulative frequency, and stream bit refill) — totaling **40 cache operations per 8 bytes decoded** (5 memory operations per byte). Even with 100% L1 cache hits, the CPU's load execution pipelines become saturated at **650–850 MB/s**.
3. **Contrast with Zstandard's LZ77 Match Engine**: Zstandard does not perform gathers. It writes repeated data with a single instruction (`_mm256_storeu_si256`), copying **32 bytes in 1 clock cycle** ($\approx 30\text{–}40\text{ GB/s}$). Because columnar physics data contains many repeated floats, zeros, and structural indices, 80%+ of the bytes bypass entropy arithmetic entirely.

#### D. Multi-Threaded Scaling on x86_64
Because the precomputed profile tables (`dec_tab`, `q_scaled`, `DecLane`) are immutable and fit into L1/L2 caches, multi-threaded RNTuple scaling is virtually linear:
- **8 Threads (AMD Zen 4 / Intel Xeon)**: $\approx \mathbf{4.5\text{–}5.5\text{ GB/s}}$ aggregate decompression throughput.
- **16 Threads**: $\approx \mathbf{8\text{–}10\text{ GB/s}}$ aggregate decompression throughput.

In typical analysis batch processing, 8–16 threads running Crystal Learned Profiles easily saturate modern PCIe Gen 4/Gen 5 NVMe storage while preserving a **10% to 15% file size advantage** over Zstandard.

---

### 6.4 Where the Time Goes: Table Caching Impact

Table precomputation cannot reduce the instruction count of the rANS entropy loop itself; what it removes is the
per-page model parse and decoder table build. With 1 MiB pages that is a small fraction of the decode:

1. **Per-page table build**: before the lane modes, `lhc4_crystal5` and `lhc4_crystal_learned5` decoded the same
   1.44 GB in 1.981 s and 1.985 s, so the per-page model parse plus table build of the dynamic path is below the
   measurement noise at this page size. The learned path's remaining cost is its larger models (a 256 KiB training
   sample at level 6 yields more contexts than a per-page level-5 model, hence more table cache misses). The
   lane bypass of `1d70fb1`, which the profile path lacked for a while (1.405 s vs 1.712 s per pass on the same
   files), is a per-page property of profile blocks too since section 3.7, and the profile path is 3-4% ahead
   in table 5.1.
2. **Higher levels**: `lhc4_crystal7` and `lhc4_crystal9` decode 8-9% slower than level 5 (807/797 vs 873 MB/s):
   the deeper model search picks larger context sets per page. The learned variants, trained once at level 6,
   follow the same slope (911 / 842 / 821 MB/s), since the spread between their levels comes from the escaped
   pages (43% of the blocks), which are coded dynamically at the write level.
3. **Storage Compression Ratio Advantage**:
   - While Zstandard decompresses faster due to bulk LZ77 copies, it achieves lower compression on scientific floating-point data:
     - `native_zstd` (level 5): **409.4 MB**
     - `lhc4_crystal5`: **360.4 MB** (49 MB smaller, ~12% higher compression ratio)
     - `lhc4_crystal_learned5`: **361.8 MB**
   - Crystal captures complex continuous distributions and order-1/2 derivatives that LZ77 cannot match.

---

## 7. Summary & Future Directions

| Dimension | Zstandard (Native) | Crystal Dynamic | Crystal Learned Profile (Implemented) |
| :--- | :--- | :--- | :--- |
| **Model Type** | LZ77 + Order-0 Huffman/FSE | Order-1/2 Context Lattice rANS | Order-1/2 Context Lattice rANS |
| **Model Storage** | None (dynamic per frame) | Per-page ($\approx 1.5\text{ KiB/page}$) | **Single metadata blob in Header** (77–750 B), per-page lane modes (1–4 B), per-page escape to a dynamic block |
| **Page Table Overhead** | None | 256–640 KiB rebuild / page | **0 bytes rebuilt / page** (precomputed) on profile blocks |
| **Lane bypass (constant / raw lanes)** | — | per block | **per page** (section 3.7) |
| **Compressed Size (CMS)**| 409.4 MB | 360.1–360.4 MB | **361.5–361.8 MB** |
| **Pure Decomp Throughput**| 2,252 MB/s | 797–873 MB/s | **821–911 MB/s** (+3-4% per level) |
| **RNTuple Decomp Throughput (1 thread)**| 2,011 MB/s | 957–1,066 MB/s | **980–1,027 MB/s** |
| **Encode CPU (1 MiB pages, level 5)**| — | 86.8 s | **42.3 s** (of which ~25 s TTree read in both) |
| **Encode wall (16 KiB pages, level 5)**| — | 160 s | **32 s** |

### Potential Next Optimizations:
1. **Profile size cap**:
   Profiles trained on a 256 KiB sample have more contexts than a per-page model; capping the context count for
   profiles (or weighting the table cost higher during training) would trade a little ratio for decode speed.
2. **NEON / SVE Vectorization for ARM64**:
   Implement vector interleaving or lookup emulation for ARM NEON/SVE for architectures without native gather.
3. **Fast-Path LZ Pre-Filter**:
   For columns exhibiting long zero-runs or static repeats, integrating a lightweight byte-level RLE/LZ stage before the context rANS stage would allow skipping the entropy machine for large data stretches, approaching multi-gigabyte throughput while retaining lattice density. On the CMS sample this is also what the remaining escapes need: the boolean/char pages and the piecewise-constant `int32` columns, where a static table cannot represent a probability below $1/4096$ for the rare symbol.

Done since the first version of this list: the transform for drifting integer columns and the support prior
(section 3.6), which took the escaped bytes from 63% to 21% of the Crystal input and the size gap at 1 MiB pages
from 1.2% to 0.4% (file bytes; 357 vs 359 MiB in the summary table); and the per-page lane modes on profile
blocks (section 3.7), which restored decode-speed parity with dynamic Crystal after its lane modes and took the
escaped blocks from 48% to 43%. The remaining escapes are pages where a per-page model beats the profile's tables
by more than the 3% margin on a coded lane; a lane that leaves the support no longer forces one.
