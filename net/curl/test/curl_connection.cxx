#include "gtest/gtest.h"

#include "ROOT/RCurlConnection.hxx"

#include "TServerSocket.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

/// Return a lower-cased copy of the input string.
static std::string ToLower(std::string s)
{
   std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
   return s;
}

/// Accept a PUT request: read headers + body, optionally respond to Expect: 100-continue, send 200 OK.
static void TaskRecvPut(TServerSocket *serverSocket, std::string *requestHeaders, std::string *requestBody)
{
   requestHeaders->clear();
   requestBody->clear();
   auto sock = serverSocket->Accept();

   const char *eof = "\r\n\r\n";
   const std::size_t eofLen = strlen(eof);
   std::size_t nextInEof = 0;
   char c;
   while (sock->RecvRaw(&c, 1)) {
      requestHeaders->push_back(c);
      if (c == eof[nextInEof]) {
         if (++nextInEof == eofLen)
            break;
      } else {
         nextInEof = 0;
      }
   }

   // If the client sent Expect: 100-continue, respond with HTTP 100 before reading the body
   std::string headersLower = ToLower(*requestHeaders);
   if (headersLower.find("expect: 100-continue") != std::string::npos) {
      const char *continueResponse = "HTTP/1.1 100 Continue\r\n\r\n";
      sock->SendRaw(continueResponse, strlen(continueResponse));
   }

   // Parse content-length (case-insensitive)
   std::size_t contentLength = 0;
   auto pos = headersLower.find("content-length: ");
   if (pos != std::string::npos) {
      auto valStart = pos + strlen("content-length: ");
      auto valEnd = headersLower.find("\r\n", valStart);
      contentLength = std::stoul(headersLower.substr(valStart, valEnd - valStart));
   }

   if (contentLength > 0) {
      requestBody->resize(contentLength);
      sock->RecvRaw(&(*requestBody)[0], contentLength);
   }

   const char *response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
   sock->SendRaw(response, strlen(response));
   sock->Close();
}

static void TaskRecv(TServerSocket *serverSocket, std::string *request)
{
   request->clear();
   auto sock = serverSocket->Accept();

   const char *eof = "\r\n\r\n";
   const std::size_t eofLen = strlen(eof);
   std::size_t nextInEof = 0;
   char c;
   while (sock->RecvRaw(&c, 1)) {
      request->push_back(c);
      if (c == eof[nextInEof]) {
         if (++nextInEof == eofLen)
            break;
      } else {
         nextInEof = 0;
      }
   }

   sock->Close();
}

TEST(RCurlConnection, Cred)
{
   TServerSocket sock(0, false, TServerSocket::kDefaultBacklog, -1, ESocketBindOption::kInaddrLoopback);
   const std::string url =
      std::string("http://") + sock.GetLocalInetAddress().GetHostAddress() + ":" + std::to_string(sock.GetLocalPort());

   std::string request;
   std::thread threadRecv(TaskRecv, &sock, &request);

   ROOT::Internal::RCurlConnection conn(url);
   EXPECT_EQ(ROOT::Internal::EHTTPCredentialsType::kNone, conn.GetCredentialsType());

   std::uint64_t remoteSize;
   conn.SendHeadReq(remoteSize);

   threadRecv.join();
   EXPECT_EQ(std::string::npos, request.find("\r\nAuthorization: "));

   conn.SetCredentials(ROOT::Internal::RS3Credentials{"a", "b", ""});
   EXPECT_EQ(ROOT::Internal::EHTTPCredentialsType::kS3, conn.GetCredentialsType());
   threadRecv = std::thread(TaskRecv, &sock, &request);
   conn.SendHeadReq(remoteSize);
   threadRecv.join();
   EXPECT_NE(std::string::npos, request.find("\r\nAuthorization: "));

   conn.ClearCredentials();
   EXPECT_EQ(ROOT::Internal::EHTTPCredentialsType::kNone, conn.GetCredentialsType());
   threadRecv = std::thread(TaskRecv, &sock, &request);
   conn.SendHeadReq(remoteSize);
   threadRecv.join();
   EXPECT_EQ(std::string::npos, request.find("\r\nAuthorization: "));
}

TEST(RCurlConnection, Put)
{
   TServerSocket sock(0, false, TServerSocket::kDefaultBacklog, -1, ESocketBindOption::kInaddrLoopback);
   const std::string url =
      std::string("http://") + sock.GetLocalInetAddress().GetHostAddress() + ":" + std::to_string(sock.GetLocalPort());

   const unsigned char payload[] = "Hello, S3!";
   const std::size_t payloadLen = sizeof(payload) - 1; // exclude null terminator

   std::string headers;
   std::string body;
   std::thread threadRecv(TaskRecvPut, &sock, &headers, &body);

   ROOT::Internal::RCurlConnection conn(url);
   auto status = conn.SendPutReq(payload, payloadLen);

   threadRecv.join();
   EXPECT_TRUE(static_cast<bool>(status));
   EXPECT_EQ(0u, headers.find("PUT "));

   // Normalize headers to lower-case for case-insensitive matching
   std::string headersLower = ToLower(headers);
   auto clPos = headersLower.find("content-length: " + std::to_string(payloadLen));
   ASSERT_NE(std::string::npos, clPos) << "content-length header not found in request";

   EXPECT_EQ(std::string(reinterpret_cast<const char *>(payload), payloadLen), body);
}

/// GET (range read) after PUT on the same handle — verifies that WRITEFUNCTION is set correctly
/// in SendRangesReq after a PUT cleared it.
TEST(RCurlConnection, GetAfterPut)
{
   TServerSocket sock(0, false, TServerSocket::kDefaultBacklog, -1, ESocketBindOption::kInaddrLoopback);
   const std::string url =
      std::string("http://") + sock.GetLocalInetAddress().GetHostAddress() + ":" + std::to_string(sock.GetLocalPort());

   // First: do a PUT
   const unsigned char putPayload[] = "put-data";
   const std::size_t putPayloadLen = sizeof(putPayload) - 1;

   std::string putHeaders;
   std::string putBody;
   std::thread threadRecvPut(TaskRecvPut, &sock, &putHeaders, &putBody);

   ROOT::Internal::RCurlConnection conn(url);
   auto putStatus = conn.SendPutReq(putPayload, putPayloadLen);

   threadRecvPut.join();
   EXPECT_TRUE(static_cast<bool>(putStatus));
   EXPECT_EQ(0u, putHeaders.find("PUT "));

   // Second: do a GET (SendRangesReq) on the same handle.
   // The server sends a plain 200 response with the body "response-from-get".
   const std::string expectedBody = "response-from-get";
   std::string getHeaders;
   auto taskRecvGet = [&](TServerSocket *serverSocket) {
      getHeaders.clear();
      auto s = serverSocket->Accept();

      const char *eof = "\r\n\r\n";
      const std::size_t eofLen = strlen(eof);
      std::size_t nextInEof = 0;
      char c;
      while (s->RecvRaw(&c, 1)) {
         getHeaders.push_back(c);
         if (c == eof[nextInEof]) {
            if (++nextInEof == eofLen)
               break;
         } else {
            nextInEof = 0;
         }
      }

      std::string response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(expectedBody.size()) +
                             "\r\n\r\n" + expectedBody;
      s->SendRaw(response.data(), response.size());
      s->Close();
   };
   std::thread threadRecvGet(taskRecvGet, &sock);

   std::vector<unsigned char> readBuf(expectedBody.size(), 0);
   ROOT::Internal::RCurlConnection::RUserRange range;
   range.fDestination = readBuf.data();
   range.fOffset = 0;
   range.fLength = expectedBody.size();
   auto getStatus = conn.SendRangesReq(1, &range);

   threadRecvGet.join();
   EXPECT_TRUE(static_cast<bool>(getStatus));
   EXPECT_EQ(0u, getHeaders.find("GET "));
   EXPECT_EQ(expectedBody.size(), range.fNBytesRecv);
   std::string received(reinterpret_cast<char *>(readBuf.data()), range.fNBytesRecv);
   EXPECT_EQ(expectedBody, received);
}

/// PUT with a payload larger than libcurl's internal Expect: 100-continue threshold (1 MB since curl 7.69).
/// Verifies that the server-side 100 Continue handshake works and all bytes arrive correctly.
TEST(RCurlConnection, PutLargeExpect100)
{
   TServerSocket sock(0, false, TServerSocket::kDefaultBacklog, -1, ESocketBindOption::kInaddrLoopback);
   const std::string url =
      std::string("http://") + sock.GetLocalInetAddress().GetHostAddress() + ":" + std::to_string(sock.GetLocalPort());

   // 2 MB payload with a known repeating pattern
   const std::size_t payloadLen = 2 * 1024 * 1024;
   std::vector<unsigned char> payload(payloadLen);
   for (std::size_t i = 0; i < payloadLen; ++i)
      payload[i] = static_cast<unsigned char>(i & 0xFF);

   std::string headers;
   std::string body;
   std::thread threadRecv(TaskRecvPut, &sock, &headers, &body);

   ROOT::Internal::RCurlConnection conn(url);
   auto status = conn.SendPutReq(payload.data(), payloadLen);

   threadRecv.join();
   EXPECT_TRUE(static_cast<bool>(status));
   EXPECT_EQ(0u, headers.find("PUT "));

   std::string headersLower = ToLower(headers);
   EXPECT_NE(std::string::npos, headersLower.find("expect: 100-continue"))
      << "large upload should include Expect: 100-continue header";
   EXPECT_NE(std::string::npos, headersLower.find("content-length: " + std::to_string(payloadLen)));
   ASSERT_EQ(payloadLen, body.size());
   EXPECT_EQ(0, memcmp(body.data(), payload.data(), payloadLen));
}
