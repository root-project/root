// @(#)root/net:$Id$
// Author: Jakob Blomer

#include "ROOT/RCurlConnection.hxx"
#include "ROOT/RError.hxx"
#include "ROOT/RVersion.hxx"

#include <TError.h>
#include <TSystem.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <curl/curl.h>

#if LIBCURL_VERSION_NUM >= 0x078300
#define HAS_CURL_EASY_HEADER
#endif

#if LIBCURL_VERSION_NUM >= 0x078000
#define HAS_CURL_URL_STRERROR
#endif

namespace {

static constexpr int kHttpResponseSuccessClass = 2;
static constexpr int kHttpResponsePartial = 206;
static constexpr int kHttpResponseBadRequest = 400;
static constexpr int kHttpResponseNotFound = 404;
static constexpr int kHttpResponseRangeNotSatisfiable = 416;

static constexpr int kMaxDebugDataChars = 50; ///< Maximum number of characters of debug HTTP content before snipping

/// A byte range as specified in an HTTP range request header
struct RHttpRange {
   std::uint64_t fFirstByte = std::uint64_t(-1);
   std::uint64_t fLastByte = std::uint64_t(-1);

   std::string ToString() const { return std::to_string(fFirstByte) + "-" + std::to_string(fLastByte); }
};

/// Set up before curl_easy_perform and then passed to all the write callbacks as data is streamed from the
/// web server.
struct RTransferState {
   /// The list of ranges in the order passed to SendRangesReq() and preprocessed by the range displacement,
   /// i.e. the ranges are not overlapping (but may be empty or adjacent).
   ROOT::Internal::RCurlConnection::RUserRange *fRanges = nullptr;
   const std::vector<std::size_t> &fOrder; ///< Array of indices to iterate fRanges sorted by offset
   CURL *fHandle = nullptr;

   std::size_t fCurrentRange = 0; ///< Index of the user range into which incoming data is read (to be used with fOrder)
   long fResponseCode = 0;        ///< Set to the HTTP response code before the first data buffer is processed
   std::size_t fNBytesProcessed = 0;  ///< Total number of received and processed bytes (including skipped buffers)
   bool fHasMultipartContent = false; ///< True if the server responded with HTTP 206
   std::string fExtraMsg; ///< Information to be passed to the user through RCurlConnection::RStatus::fStatusMsg

   // State for parsing the individual part headers of a multipart response
   std::string fPartHeaders;          ///< Header characters seen so far of the currently processed part
   bool fInPartHeader = false;        ///< Indicate if we are processing the content of the part of the headers
   std::uint8_t fNEohCharsFound = 0;  ///< Counts how many of the header end signature ("\r\n\r\n") we have already seen
   std::size_t fNBytesPartRemain = 0; ///< Number of remaining bytes of the content of the current part

#ifndef HAS_CURL_EASY_HEADER
   bool fHasContentRangeHeader = false; ///< On older libcurl versions, we must parse the headers ourselves
#endif

   RTransferState(ROOT::Internal::RCurlConnection::RUserRange *ranges, const std::vector<std::size_t> &order,
                  CURL *handle)
      : fRanges(ranges), fOrder(order), fHandle(handle)
   {
   }

   ROOT::Internal::RCurlConnection::RUserRange &GetCurrentRange() const { return fRanges[fOrder[fCurrentRange]]; }

   std::size_t GetNRanges() const { return fOrder.size(); }

   /// Let fCurrentRange point to the next non-empty range (if any left)
   void AdvanceRange()
   {
      do {
         fCurrentRange++;
      } while ((fCurrentRange < GetNRanges()) && (GetCurrentRange().fLength == 0));
   }

   bool IsPartial() const { return fResponseCode == kHttpResponsePartial; }
};

void EnsureCurlInitialized()
{
   static const auto kInitCode = curl_global_init(CURL_GLOBAL_DEFAULT);
   if (kInitCode != CURLE_OK) {
      // Cannot use GetCurlErrorString() because curl isn't initialized...
      throw ROOT::RException(R__FAIL("cannot initialize curl library: " + std::to_string(kInitCode)));
   }
}

const curl_version_info_data *GetCurlVersionInfo()
{
   static const curl_version_info_data *kVersionInfo = curl_version_info(CURLVERSION_NOW);
   return kVersionInfo;
}

/// Return the information of a "content-range" header in a part of a MIME multi-part message.
/// The headers input is the complete list of part headers.
/// The content-range header has the form "content-range: FIRSTBYTE-LASTBYTE/LENGTH". We ignore LENGTH; the
/// returned length is calculated from the range.
ROOT::RResult<void> ParseContentRange(const std::string &headers, std::uint64_t &offset, std::uint64_t &length)
{
   // Note that headers start with a blank line, so the leading line break is always found.
   static constexpr char kContentRangeHeader[16] = {'\r', '\n', 'c', 'o', 'n', 't', 'e', 'n',
                                                    't',  '-',  'r', 'a', 'n', 'g', 'e', ':'};

   auto it = std::search(headers.begin(), headers.end(), std::begin(kContentRangeHeader), std::end(kContentRangeHeader),
                         [](char a, char b) { return std::toupper(a) == std::toupper(b); });
   if (it == headers.end()) {
      return R__FAIL("cannot find 'content-range' header in multipart response");
   }

   std::string strBytePos[2]; // the first byte and the last byte of the range
   int idxStrBytePos = 0;
   it += sizeof(kContentRangeHeader);
   do {
      if (it == headers.end()) {
         return R__FAIL("premature end of 'content-range' header in multipart response");
      }

      if (std::isspace(*it) || std::isalpha(*it)) {
         it++;
         continue;
      }
      if (*it == '-') {
         if (idxStrBytePos == 1) {
            return R__FAIL(std::string("invalid 'content-range' header: ") + headers);
         }
         idxStrBytePos++;
         it++;
         continue;
      }
      if (*it == '/')
         break;

      strBytePos[idxStrBytePos].push_back(*it);
      it++;
   } while (true);

   if (strBytePos[0].empty() || strBytePos[1].empty())
      return R__FAIL(std::string("invalid 'content-range' header: ") + headers);

   char *end = nullptr;
   offset = std::strtoull(strBytePos[0].c_str(), &end, 10);
   if (errno == ERANGE)
      return R__FAIL(std::string("invalid 'content-range' header: ") + headers);
   auto lastBytePos = std::strtoull(strBytePos[1].c_str(), &end, 10);
   if (errno == ERANGE)
      return R__FAIL(std::string("invalid 'content-range' header: ") + headers);

   if (lastBytePos < offset)
      return R__FAIL(std::string("invalid 'content-range' header: ") + headers);

   length = lastBytePos - offset + 1;
   if (length > std::numeric_limits<std::size_t>::max()) {
      return R__FAIL(std::string("part of multipart response too big: ") + std::to_string(length) + "B");
   }

   return ROOT::RResult<void>::Success();
}

/// Process the buffer 'data' of length 'nbytes' that was received from the server as part of a MIME multipart response
std::size_t ProcessMultipartData(char *data, std::size_t nbytes, RTransferState *transfer)
{
   static constexpr char kEndOfHeaders[4] = {'\r', '\n', '\r', '\n'}; // Headers always end with a blank line

   std::size_t nbytesRemain = nbytes;

   while (nbytesRemain > 0) {
      if (transfer->fInPartHeader) {
         transfer->fPartHeaders.push_back(*data);
         if (*data == kEndOfHeaders[transfer->fNEohCharsFound]) {
            transfer->fNEohCharsFound++;
            if (transfer->fNEohCharsFound == sizeof(kEndOfHeaders)) {
               transfer->fInPartHeader = false;
               transfer->fNEohCharsFound = 0;
            }
         } else {
            transfer->fNEohCharsFound = 0;
         }
         data++;
         nbytesRemain--;

         if (transfer->fInPartHeader)
            continue;

         // Transition from part header to part content
         std::uint64_t partOffset = 0;
         std::uint64_t length = 0;
         auto result = ParseContentRange(transfer->fPartHeaders, partOffset, length);
         transfer->fPartHeaders.clear();
         if (!result) {
            transfer->fExtraMsg = result.GetError()->GetReport();
            return 0;
         }
         // By construction, at this point we have length > 0 and we know its small enough for fNBytesPartRemain
         transfer->fNBytesPartRemain = length;

         // Find matching range: the parts of a multipart response may come in any order (although unusual), so
         // we need to find the corresponing range. The ranges at this point must be non-overlapping, so we can safely
         // match on offset.
         std::size_t i = 0;
         for (; i < transfer->GetNRanges(); ++i) {
            const auto &r = transfer->GetCurrentRange();
            if ((r.fLength > 0) && (r.fOffset == partOffset))
               break;
            transfer->fCurrentRange = (transfer->fCurrentRange + 1) % transfer->GetNRanges();
         }
         if (i == transfer->GetNRanges()) {
            transfer->fExtraMsg = std::string("unexpected part with offset ") + std::to_string(partOffset);
            return 0;
         }
      } // end of part header parsing

      // Process part content
      auto &range = transfer->GetCurrentRange();
      const std::size_t nbytesCopy =
         std::min({nbytesRemain, transfer->fNBytesPartRemain, range.fLength - range.fNBytesRecv});
      if (nbytesCopy > 0) {
         memcpy(range.fDestination + range.fNBytesRecv, data, nbytesCopy);
      }
      data += nbytesCopy;
      range.fNBytesRecv += nbytesCopy;
      nbytesRemain -= nbytesCopy;
      transfer->fNBytesPartRemain -= nbytesCopy;

      if (transfer->fNBytesPartRemain == 0) {
         // End of this part, back to parsing headers of the next part
         transfer->AdvanceRange(); // not strictly necessary but speeds up the range matching if parts come in order
         transfer->fInPartHeader = true;
      } else if (range.fNBytesRecv == range.fLength) {
         // coalesced adjacent ranges, move on to the next range in the sorted array
         transfer->AdvanceRange();
         if (transfer->fCurrentRange == transfer->GetNRanges()) {
            transfer->fExtraMsg = std::string("received range too long");
            return 0;
         }
      }
   }

   transfer->fNBytesProcessed += nbytes;
   return nbytes;
}

/// Process the buffer 'data' of length 'nbytes' that was received from the server as part standard HTTP 200 response,
/// i.e. the server ignored the range request.
std::size_t ProcessRawData(char *data, std::size_t nbytes, RTransferState *transfer)
{
   std::size_t nbytesRemain = nbytes;

   while ((nbytesRemain > 0) && (transfer->fCurrentRange < transfer->GetNRanges())) {
      auto &range = transfer->GetCurrentRange();

      std::size_t nbytesSkip = 0;
      if (transfer->fNBytesProcessed < range.fOffset) {
         // Skip the first part of the data pointer that is not yet part of the requested range at hand
         nbytesSkip = std::min(static_cast<std::uint64_t>(nbytesRemain), range.fOffset - transfer->fNBytesProcessed);
      }

      const std::size_t nbytesCopy = std::min(nbytesRemain - nbytesSkip, range.fLength - range.fNBytesRecv);
      if (nbytesCopy > 0) {
         // The received buffer overlaps with the current range
         memcpy(range.fDestination + range.fNBytesRecv, data + nbytesSkip, nbytesCopy);
      }

      range.fNBytesRecv += nbytesCopy;
      if (range.fNBytesRecv == range.fLength)
         transfer->AdvanceRange();

      nbytesRemain -= nbytesSkip + nbytesCopy;
      data += nbytesSkip + nbytesCopy;
      transfer->fNBytesProcessed += nbytesSkip + nbytesCopy;
   }
   transfer->fNBytesProcessed += nbytesRemain;

   return nbytes;
}

/// Called by libcurl as data arrives from the web server. The data buffer may be empty. This callback runs
/// possibly repeatadly after the response headers are processed.
std::size_t CallbackData(char *data, std::size_t size, std::size_t nmemb, void *userdata)
{
   std::size_t nbytes = size * nmemb;
   if (nbytes == 0)
      return 0;

   RTransferState *transfer = static_cast<RTransferState *>(userdata);

   // Four possible successful responses:
   //   1) Full document, server ignores the ranges in the request
   //   2) Partial repsonse (206) with a single returned range (note that we already coalesced adjacent ranges
   //      in the request): returns the requested byte range (or shorter, if requested range goes past EOF)
   //   3) Partial repsonse (206) for multiple requested ranges: multipart MIME message;
   //      parts can come in any order (but usually they should come in the order of the requested ranges).
   //      Requested ranges that are outside the size of the remote resource are ignored.
   //   4) All ranges are outside the remote resource size: 416 (unsatisfiable request)

   if (transfer->fResponseCode == 0) {
      // Only called here the first time before any data of the reponse is processed
      auto rc = curl_easy_getinfo(transfer->fHandle, CURLINFO_RESPONSE_CODE, &transfer->fResponseCode);
      R__ASSERT(rc == CURLE_OK);

      if (transfer->IsPartial()) {
         // Check for the content-range header which will be present only if a single range was returned.
#ifdef HAS_CURL_EASY_HEADER
         curl_header *h = nullptr;
         transfer->fHasMultipartContent =
            curl_easy_header(transfer->fHandle, "content-range", 0, CURLH_HEADER, -1, &h) != CURLHE_OK;
#else
         transfer->fHasMultipartContent = !transfer->fHasContentRangeHeader;
#endif
         if (transfer->fHasMultipartContent) {
            transfer->fInPartHeader = true;
         } else {
            // A range request for a single range must return it precisely (it can only cut at the end for EOF)
            transfer->fNBytesProcessed = transfer->GetCurrentRange().fOffset;
         }
      }
   }

   if (transfer->fResponseCode / 100 != kHttpResponseSuccessClass) {
      // ignore the HTTP error message body
      return nbytes;
   }

   if (transfer->fHasMultipartContent)
      return ProcessMultipartData(data, nbytes, transfer);

   return ProcessRawData(data, nbytes, transfer);
}

#ifndef HAS_CURL_EASY_HEADER
/// TODO(jblomer): remove me when we can require libcurl >= 7.83. Used to remember if a "content-range" header was seen
std::size_t CallbackHeader(char *buffer, std::size_t size, std::size_t nitems, void *userdata)
{
   std::size_t nbytes = size * nitems;
   if (nbytes == 0)
      return 0;

   RTransferState *transfer = static_cast<RTransferState *>(userdata);

   std::string headerLine(buffer, nbytes);
   std::transform(headerLine.begin(), headerLine.end(), headerLine.begin(), ::toupper);
   if (headerLine.rfind("CONTENT-RANGE:", 0) == 0) {
      transfer->fHasContentRangeHeader = true;
   }

   return nbytes;
}
#endif

int CallbackDebug(CURL * /* handle */, curl_infotype type, char *data, size_t size, void * /* clientp */)
{
   std::string prefix = "(libcurl debug) ";
   switch (type) {
   case CURLINFO_TEXT: prefix += "{info} "; break;
   case CURLINFO_HEADER_IN: prefix += "{header/recv} "; break;
   case CURLINFO_HEADER_OUT: prefix += "{header/sent} "; break;
   case CURLINFO_DATA_IN: prefix += "{data/recv} "; break;
   case CURLINFO_DATA_OUT: prefix += "{data/sent} "; break;
   case CURLINFO_SSL_DATA_IN: prefix += "{ssldata/recv} "; break;
   case CURLINFO_SSL_DATA_OUT: prefix += "{ssldata/sent} "; break;
   default: break;
   }

   switch (type) {
   case CURLINFO_DATA_IN:
   case CURLINFO_DATA_OUT:
   case CURLINFO_SSL_DATA_IN:
   case CURLINFO_SSL_DATA_OUT:
      if (size > kMaxDebugDataChars) {
         Info("RCurlConnection", "%s <snip>", prefix.c_str());
         return 0;
      }
   default: break;
   }

   std::string msg(data, size);
   bool isPrintable = true;

   for (std::size_t i = 0; i < msg.length(); ++i) {
      if (msg[i] == '\0') {
         msg[i] = '~';
      }

      if ((msg[i] < ' ' || msg[i] > '~') && (msg[i] != 10 /*line feed*/ && msg[i] != 13 /*carriage return*/)) {
         isPrintable = false;
         break;
      }
   }

   if (!isPrintable) {
      msg = "<Non-plaintext sequence>";
   }
   Info("RCurlConnection", "%s%s", prefix.c_str(), msg.c_str());
   return 0;
}

// From the list of non-overlapping (displaced) ranges, create the HTTP request ranges, ordered by offset.
// Skip empty user ranges and coalesce adjacent user ranges.
std::vector<RHttpRange>
CreateRequestRanges(const ROOT::Internal::RCurlConnection::RUserRange *ranges, const std::vector<std::size_t> &order)
{
   const auto N = order.size();
   std::vector<RHttpRange> result;

   std::uint64_t rangeBegin = 0;
   std::uint64_t rangeEnd = 0;
   for (std::size_t i = 0; i < N; ++i) {
      const auto &r = ranges[order[i]];

      if (r.fLength == 0)
         continue;

      if (r.fOffset == rangeEnd) {
         // Merge adjacent ranges into a single request range
         rangeEnd = r.fOffset + r.fLength;
         continue;
      }

      // Emit previous range
      if (rangeEnd > 0)
         result.emplace_back(RHttpRange{rangeBegin, rangeEnd - 1});

      // Open new range
      rangeBegin = r.fOffset;
      rangeEnd = r.fOffset + r.fLength;
   }
   if (rangeEnd > 0) {
      result.emplace_back(RHttpRange{rangeBegin, rangeEnd - 1});
   }

   return result;
}

/// For overlapping ranges, the displacement moves the offset of the rear request to the end of the front one.
/// E.g., for the following situation with ranges
///
///    |---RANGE A---|
///         |--------RANGE B-------|
///            |--RANGE C--|
///               |---------RANGE D----------|
///
/// the displacement will move the offsets of ranges B, C, D so that the result is
///    |---RANGE A---|---RANGE B---|-RANGE D-|
///                        * (RANGE C, zero-sized)
///
/// Ranges fully contained in previous ranges end up zero-sized after displacement.
/// Note that the range that contains the prefix of a range at hand is not necessarily the immedate predecessor.
///
/// Returns the additions to the offsets of the passed ranges. The first range always has a displacement of zero,
/// so in principle the returned vector could be of size N - 1. But it seems simpler to keep it same sized.
/// In the retured vector, index $k$ belongs to ranges[order[k]].
std::vector<std::size_t>
CreateAndApplyDisplacements(ROOT::Internal::RCurlConnection::RUserRange *ranges, const std::vector<std::size_t> &order)
{
   const auto N = order.size();
   std::vector<std::size_t> displacements(N);
   for (std::size_t i = 1; i < N; ++i) {
      // ranges 0 .. i - 1 are already non-overlapping; check from i onwards if the ranges overlap with range i - 1.
      const auto &prevRange = ranges[order[i - 1]];
      if (prevRange.fLength == 0)
         continue;

      const auto prevLastByte = prevRange.fOffset + prevRange.fLength - 1;
      // Quadratic complexity for pathological cases only,
      // which can be easily fixed by better preprocessing of the ranges
      for (auto j = i; j < N; ++j) {
         auto &thisRange = ranges[order[j]];
         if (thisRange.fOffset > prevLastByte)
            break;
         auto displacement =
            std::min(static_cast<std::uint64_t>(thisRange.fLength), prevLastByte - thisRange.fOffset + 1);
         // As we move the offset, we may break the sorting. Violations of the sort order, however, can only
         // take place for zero-length displaced ranges, i.e. ranges fully contained within previous ranges.
         // They are ignored during the data callback, so the sort order violation doesn't matter.
         thisRange.fOffset += displacement;
         thisRange.fDestination += displacement;
         thisRange.fLength -= displacement;
         displacements[j] += displacement;
      }
   }
   return displacements;
}

/// After the HTTP transfer, all the data from the user ranges have been copied into ranges buffers once.
/// In the process of reversing the displacements, overlapping ranges are copied from the containing ranges
/// buffer into the displaced buffer sections. We always restore the originally provided information in ranges,
/// but we only actually copy data if the request was successful.
void ReverseDisplacements(std::vector<std::size_t> &displacements, ROOT::Internal::RCurlConnection::RUserRange *ranges,
                          const std::vector<std::size_t> &order, bool copyBuffers)
{
   const auto N = order.size();
   for (std::size_t i = 1; i < N; ++i) {
      if (displacements[i] == 0)
         continue;

      auto &thisRange = ranges[order[i]];
      std::size_t j = i - 1;
      do {
         // We go step by step through the previous ranges (which are already reversed) to copy the information.
         const auto &prevRange = ranges[order[j]];
         if ((prevRange.fLength == 0) || (prevRange.fOffset + prevRange.fLength < thisRange.fOffset)) {
            j--;
            continue;
         }
         const std::size_t nbytesReverse =
            std::min(static_cast<std::uint64_t>(displacements[i]), thisRange.fOffset - prevRange.fOffset);
         thisRange.fOffset -= nbytesReverse;
         thisRange.fDestination -= nbytesReverse;
         thisRange.fLength += nbytesReverse;
         displacements[i] -= nbytesReverse;

         // The previous range has not been fully filled if it goes past the size of the remote resource.
         // In this case, the current range may also only be partially filled.
         std::size_t nbytesRecvFromPrev = 0;
         if (prevRange.fOffset + prevRange.fNBytesRecv >= thisRange.fOffset) {
            nbytesRecvFromPrev = std::min(prevRange.fOffset + prevRange.fNBytesRecv - thisRange.fOffset,
                                          static_cast<std::uint64_t>(nbytesReverse));
         }

         thisRange.fNBytesRecv += nbytesRecvFromPrev;
         if (copyBuffers && (nbytesRecvFromPrev > 0)) {
            memcpy(thisRange.fDestination, prevRange.fDestination + (thisRange.fOffset - prevRange.fOffset),
                   nbytesRecvFromPrev);
         }
      } while (displacements[i]);
   }
}

std::string GetCurlErrorString(CURLcode code)
{
   return std::string(curl_easy_strerror(code)) + " (" + std::to_string(code) + ")";
}

std::string GetCurlUrlErrorString(CURLUcode code)
{
#ifdef HAS_CURL_URL_STRERROR
   return std::string(curl_url_strerror(code)) + " (" + std::to_string(code) + ")";
#else
   return std::string("libcurl too old for mapping error number to text") + " (" + std::to_string(code) + ")";
#endif
}

std::string GetUserAgentString()
{
   SysInfo_t s;
   gSystem->GetSysInfo(&s);

   auto curlVersionInfo = GetCurlVersionInfo();

   return std::string("ROOT/v") + ROOT_RELEASE + " (" + std::string(s.fOS) + ") curl/" + curlVersionInfo->version +
          " " + curlVersionInfo->ssl_version;
}

} // anonymous namespace

ROOT::Internal::RCurlConnection::RCurlConnection(const std::string &url)
{
   EnsureCurlInitialized();

   fHandle = curl_easy_init();
   if (!fHandle) {
      throw RException(R__FAIL("cannot initialize curl handle"));
   }

   SetupErrorBuffer();
   SetOptions();

   auto result = SetUrl(url);
   if (!result) {
      curl_easy_cleanup(fHandle);
      result.Throw();
   }
}

ROOT::Internal::RCurlConnection::~RCurlConnection()
{
   if (fHandle)
      curl_easy_cleanup(fHandle);
}

ROOT::Internal::RCurlConnection::RCurlConnection(RCurlConnection &&other)
{
   std::swap(fHandle, other.fHandle);
   SetupErrorBuffer();
}

ROOT::Internal::RCurlConnection &ROOT::Internal::RCurlConnection::RCurlConnection::operator=(RCurlConnection &&other)
{
   if (this == &other)
      return *this;
   fHandle = other.fHandle;
   other.fHandle = nullptr;
   SetupErrorBuffer();
   return *this;
}

void ROOT::Internal::RCurlConnection::SetupErrorBuffer()
{
   if (!fErrorBuffer)
      fErrorBuffer = std::make_unique<char[]>(CURL_ERROR_SIZE);
   auto rc = curl_easy_setopt(fHandle, CURLOPT_ERRORBUFFER, fErrorBuffer.get());
   R__ASSERT(rc == CURLE_OK);
}

void ROOT::Internal::RCurlConnection::SetOptions()
{
   int rc;

   if (gDebug) {
      rc = curl_easy_setopt(fHandle, CURLOPT_VERBOSE, 1);
      R__ASSERT(rc == CURLE_OK);
      rc = curl_easy_setopt(fHandle, CURLOPT_DEBUGFUNCTION, CallbackDebug);
      R__ASSERT(rc == CURLE_OK);
   } else {
      rc = curl_easy_setopt(fHandle, CURLOPT_VERBOSE, 0);
      R__ASSERT(rc == CURLE_OK);
   }

   static const std::string kUserAgent = GetUserAgentString();
   rc = curl_easy_setopt(fHandle, CURLOPT_USERAGENT, kUserAgent.c_str());
   R__ASSERT(rc == CURLE_OK);

   rc = curl_easy_setopt(fHandle, CURLOPT_FOLLOWLOCATION, 1);
   R__ASSERT(rc == CURLE_OK);

   rc = curl_easy_setopt(fHandle, CURLOPT_WRITEFUNCTION, CallbackData);
   R__ASSERT(rc == CURLE_OK);
}

ROOT::RResult<void> ROOT::Internal::RCurlConnection::SetUrl(const std::string &url)
{
   CURLU *cu = curl_url();
   R__ASSERT(cu);
   auto rc = curl_url_set(cu, CURLUPART_URL, url.c_str(), CURLU_URLENCODE);
   if (rc != CURLUE_OK) {
      curl_url_cleanup(cu);
      return R__FAIL(std::string("invalid URL: ") + std::string(url) + " [" + GetCurlUrlErrorString(rc) + "]");
   }

   char *escaped_url = nullptr;
   rc = curl_url_get(cu, CURLUPART_URL, &escaped_url, CURLU_NO_DEFAULT_PORT);
   curl_url_cleanup(cu);
   if (rc != CURLUE_OK) {
      return R__FAIL(std::string("URL escape error: ") + std::string(url) + " [" + GetCurlUrlErrorString(rc) + "]");
   }

   fEscapedUrl = escaped_url;

   auto rcOpt = curl_easy_setopt(fHandle, CURLOPT_URL, escaped_url);
   curl_free(escaped_url);
   if (rcOpt != CURLE_OK) {
      return R__FAIL("cannot set URL: " + GetCurlErrorString(rcOpt));
   }

   return RResult<void>::Success();
}

void ROOT::Internal::RCurlConnection::Perform(RStatus &status)
{
   auto rc = curl_easy_perform(fHandle);

// CURLE_TOO_LARGE is available as of curl version 8.6.0
#ifdef CURLE_TOO_LARGE
   if (rc == CURLE_TOO_LARGE) {
#else
   if (rc == CURLE_OUT_OF_MEMORY) {
#endif
      // The ranges don't even fit in the request header
      status.fStatusCode = RStatus::kTooManyRanges;
   } else if (rc != CURLE_OK) {
      status.fStatusMsg = fErrorBuffer.get();
      status.fStatusMsg += " [" + GetCurlErrorString(rc) + "]";

      long osErrNo = 0;
      rc = curl_easy_getinfo(fHandle, CURLINFO_OS_ERRNO, &osErrNo);
      if (rc == CURLE_OK)
         status.fStatusMsg += " (OS errno: " + std::to_string(osErrNo) + ")";
   } else {
      long responseCode = 0;
      rc = curl_easy_getinfo(fHandle, CURLINFO_RESPONSE_CODE, &responseCode);
      R__ASSERT(rc == CURLE_OK);
      if ((responseCode / 100 == kHttpResponseSuccessClass) || (responseCode == kHttpResponseRangeNotSatisfiable)) {
         // Requests past the size of the remote resource are considered valid. They simply receive zero bytes.
         status.fStatusCode = RStatus::kSuccess;
      } else if (responseCode == kHttpResponseNotFound) {
         status.fStatusCode = RStatus::kNotFound;
      } else if (responseCode == kHttpResponseBadRequest) {
         status.fStatusCode = RStatus::kTooManyRanges;
      } else {
         status.fStatusCode = RStatus::kIOError;
      }
   }
}

ROOT::Internal::RCurlConnection::RStatus ROOT::Internal::RCurlConnection::SendHeadReq(std::uint64_t &remoteSize)
{
   remoteSize = kUnknownSize;

   auto rc = curl_easy_setopt(fHandle, CURLOPT_NOBODY, 1);
   R__ASSERT(rc == CURLE_OK);
   rc = curl_easy_setopt(fHandle, CURLOPT_RANGE, NULL); // may have been set by a previous SendRangesReq() on the object
   R__ASSERT(rc == CURLE_OK);

#ifndef HAS_CURL_EASY_HEADER
   rc = curl_easy_setopt(fHandle, CURLOPT_HEADERFUNCTION, NULL);
   R__ASSERT(rc == CURLE_OK);
   rc = curl_easy_setopt(fHandle, CURLOPT_HEADERDATA, NULL);
   R__ASSERT(rc == CURLE_OK);
#endif

   RStatus status;
   Perform(status);
   if (status) {
      curl_off_t length = -1;
      rc = curl_easy_getinfo(fHandle, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length);
      if (rc == CURLE_OK && length >= 0)
         remoteSize = length;
   }

   return status;
}

ROOT::Internal::RCurlConnection::RStatus
ROOT::Internal::RCurlConnection::SendRangesReq(std::size_t N, RUserRange *ranges)
{
   if (N == 0) {
      // Pretend that we submitted a successful request
      return RStatus(RStatus::kSuccess);
   }

   // Construct an array of indices that allows to iterate the ranges in order sorted by offset
   std::vector<std::size_t> order(N);
   std::iota(order.begin(), order.end(), 0);
   std::sort(order.begin(), order.end(), [ranges](std::size_t a, std::size_t b) { return ranges[a] < ranges[b]; });

   // Fixup overlapping ranges
   auto displacements = CreateAndApplyDisplacements(ranges, order);

   // Construct the consolidated HTTP ranges from the ordered and non-overlapping user ranges
   const auto requestRanges = CreateRequestRanges(ranges, order);
   if (requestRanges.empty()) {
      // In this case, we know that we did not apply any displacements
      return RStatus(RStatus::kSuccess);
   }

   auto rc = curl_easy_setopt(fHandle, CURLOPT_HTTPGET, 1);
   R__ASSERT(rc == CURLE_OK);

   RTransferState transfer(ranges, order, fHandle);
   rc = curl_easy_setopt(fHandle, CURLOPT_WRITEDATA, &transfer);
   R__ASSERT(rc == CURLE_OK);

#ifndef HAS_CURL_EASY_HEADER
   rc = curl_easy_setopt(fHandle, CURLOPT_HEADERFUNCTION, CallbackHeader);
   R__ASSERT(rc == CURLE_OK);
   rc = curl_easy_setopt(fHandle, CURLOPT_HEADERDATA, &transfer);
   R__ASSERT(rc == CURLE_OK);
#endif

   RStatus status;
   // There is no HTTP request to determine the maximum number of ranges that the web server can serve.
   // Therefore, we try with all the ranges (or fMaxNRangesPerReqest, if explicitly set), and half that number
   // as long as needed.
   // If we need to reduce the number of ranges per requests and no limit was set,
   // we will remember the working number for the next requests.
   std::size_t batchSize = fMaxNRangesPerReqest ? fMaxNRangesPerReqest : requestRanges.size();
   bool tryAgain;
   do {
      tryAgain = false;
      // If we have multiple batches, we could in principle submit them concurrently using multiple connections
      // (and CURL easy handles) that get pooled in a CURL multi handle.
      // This is a potential future optimization.
      for (std::size_t b = 0; b < requestRanges.size(); b += batchSize) {
         const std::size_t nRanges = std::min(batchSize, requestRanges.size() - b);
         std::string rangeHeader = requestRanges[b].ToString();
         for (std::size_t i = 1; i < nRanges; ++i) {
            rangeHeader += "," + requestRanges[b + i].ToString();
         }
         rc = curl_easy_setopt(fHandle, CURLOPT_RANGE, rangeHeader.c_str());
         R__ASSERT(rc == CURLE_OK);

         if (b > 0) {
            const std::uint64_t lastByteRequested = requestRanges[b - 1].fLastByte;
            // Advance all ranges that are already out of scope
            // Note that we have to start at zero because the previous multi-part request may have visited the
            // ranges in arbirary order.
            for (transfer.fCurrentRange = 0; transfer.fCurrentRange < N; transfer.fCurrentRange++) {
               if (transfer.GetCurrentRange().fOffset > lastByteRequested)
                  break;
            }
         }

         transfer.fResponseCode = 0; // reset HTTP response code for the next request
         Perform(status);
         if ((status.fStatusCode == RStatus::kTooManyRanges) && (batchSize > 1)) {
            batchSize /= 2;
            tryAgain = true;
            break;
         }
         if (!status)
            break;
      }
   } while (tryAgain);

   if (status && (fMaxNRangesPerReqest == 0) && (batchSize < requestRanges.size()))
      fMaxNRangesPerReqest = batchSize;

   if (!transfer.fExtraMsg.empty()) {
      status.fStatusMsg += "; extra information: " + transfer.fExtraMsg;
   }

   ReverseDisplacements(displacements, ranges, order, static_cast<bool>(status));

   return status;
}
