// @(#)root/net:$Id$
// Author: Jakob Blomer

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RCurlConnection
#define ROOT_RCurlConnection

#include <ROOT/RError.hxx>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Internal {

/// Encapsulates a curl easy handle and provides an interface to send HTTP HEAD and (multi-)range queries.
class RCurlConnection {
public:
   /// Return value for both HEAD and GET requests. In case of errors, provides the reason for the failure as code
   /// and as message.
   struct RStatus {
      enum EStatusCode {
         kSuccess = 0,
         kTooManyRanges, ///< should not get to the user; number of request ranges is automatically reduced as needed
         kNotFound,
         kIOError,
         kUnknown
      };

      EStatusCode fStatusCode = kUnknown;
      std::string fStatusMsg;

      RStatus() = default;
      explicit RStatus(EStatusCode code) : fStatusCode(code) {}

      explicit operator bool() const { return fStatusCode == kSuccess; }
   };

private:
   void *fHandle = nullptr; ///< the CURL easy handle corresponding to this connection
   /// If set to zero, automatically adjust: try with all given ranges and as long as the number of ranges is too large,
   /// half it. If set to zero and automatic reduction of the number of requests is necessary, the number of requests
   /// that works will be saved for further requests with this object.
   std::size_t fMaxNRangesPerReqest = 0;
   std::string fEscapedUrl;              ///< The URL provided in the constructor escaped according to standard rules
   std::unique_ptr<char[]> fErrorBuffer; ///< For use by libcurl

   void SetupErrorBuffer();
   void SetOptions();
   RResult<void> SetUrl(const std::string &url);
   void Perform(RStatus &status);

public:
   /// Returned by SendHeadReq() if the HTTP response contains no content-length header
   static constexpr std::uint64_t kUnknownSize = static_cast<std::uint64_t>(-1);

   /// Caller-provided byte-range of the remote resource together with a pointer to a buffer.
   struct RUserRange {
      unsigned char *fDestination = nullptr;
      std::uint64_t fOffset = 0;
      std::size_t fLength = 0;
      /// Usually equal to fLength for a successful call unless range goes out of the size of the remote resource
      std::size_t fNBytesRecv = 0;

      bool operator<(const RUserRange &other) const { return fOffset < other.fOffset; }
   };

   explicit RCurlConnection(const std::string &url);
   ~RCurlConnection();
   RCurlConnection(const RCurlConnection &other) = delete;
   RCurlConnection &operator=(const RCurlConnection &other) = delete;
   RCurlConnection(RCurlConnection &&other);
   RCurlConnection &operator=(RCurlConnection &&other);

   /// Checks if the resource exists and if it does, return the value of the content-length header as size
   RStatus SendHeadReq(std::uint64_t &remoteSize);
   /// Reads the given ranges from the remote resource. The ranges can be in any order and also overlapping. They
   /// will be transformed in optimized HTTP ranges for a multi-range request. Ranges past the resource size are
   /// valid (but won't receive any data). No limit on the number of ranges; if fMaxNRangesPerReqest is zero,
   /// a valid batching of requests into multiple multi-range requests takes place automatically.
   /// The fNBytesRecv member of the ranges is only well-defined on success.
   RStatus SendRangesReq(std::size_t N, RUserRange *ranges);

   const std::string &GetEscapedUrl() const { return fEscapedUrl; }

   void SetMaxNRangesPerRequest(std::size_t val) { fMaxNRangesPerReqest = val; }
   std::size_t GetMaxNRangesPerRequest() const { return fMaxNRangesPerReqest; }
};

} // namespace Internal
} // namespace ROOT

#endif
