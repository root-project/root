#ifndef StreamerThrowClass_h
#define StreamerThrowClass_h

#include <stdexcept>

class TBuffer;

struct StreamerThrowClass final {
   void Streamer(TBuffer &) { throw std::runtime_error("streaming not supported"); }
};

#endif
