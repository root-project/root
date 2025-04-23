#include <Rtypes.h>
#include <TBuffer.h>
#include <iostream>

class FooCustomStreamer {
public:
  FooCustomStreamer() {}

private:
  int fData = 3;
  ClassDefNV(FooCustomStreamer, 1);
};

inline void FooCustomStreamer::Streamer(TBuffer &R__b) {
  if (R__b.IsReading()) {
    std::cout << "Custom streamer reading" << std::endl;
  } else {
    std::cout << "Custom streamer writing" << std::endl;
  }
}
