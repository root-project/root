#include "gtest/gtest.h"

#include "ROOT/RCurlConnection.hxx"

#include "TServerSocket.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

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
