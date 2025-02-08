#include "TServerSocket.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

TEST(TServerSocket, SocketBinding)
{
   // The socket is 0 to let ROOT find a free port for this test
   TServerSocket theSocket(0, false, TServerSocket::kDefaultBacklog, -1, ESocketBindOption::kInaddrLoopback);
   const auto addr = theSocket.GetLocalInetAddress().GetHostAddress();
   const auto expectedAddr = "0.0.0.0";
   ASSERT_THAT(addr, ::testing::StrNe(expectedAddr))
      << "The address is " << addr << " while the expected one must be different from " << expectedAddr;
}
