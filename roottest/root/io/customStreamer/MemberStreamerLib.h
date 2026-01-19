#ifndef TEST_MEMBER_STREAMER_LIB
#define TEST_MEMBER_STREAMER_LIB

#include <memory>

struct MemberStreamer {
   int fData = 0;
};

struct MemberStreamerContainer {
   std::unique_ptr<MemberStreamer> fMember;

   MemberStreamerContainer() : fMember(new MemberStreamer()) {}
};

#endif
