#ifndef TEST_AUTO_PTR_V2_H
#define TEST_AUTO_PTR_V2_H

#include <RtypesCore.h>

#include <memory>

// Since std::auto_ptr is deprecated and may not exist anymore, we emulate it.
template <typename T>
struct EmulatedAutoPtr {
   T *fRawPtr = nullptr;

   EmulatedAutoPtr() = default;
   explicit EmulatedAutoPtr(T *rawPtr) : fRawPtr(rawPtr) {}
   EmulatedAutoPtr &operator=(EmulatedAutoPtr &other)
   {
      fRawPtr = other.fRawPtr;
      other.fRawPtr = nullptr;
      return *this;
   }
   EmulatedAutoPtr &operator=(T *rawPtr)
   {
      fRawPtr = rawPtr;
      return *this;
   }
};

struct Track {
   int fFoo;

   ClassDefNV(Track, 2)
};

struct TestAutoPtr {
   EmulatedAutoPtr<Track> fTrack;
   float fBar = 137.0;

   ClassDefNV(TestAutoPtr, 2)
};

#endif // TEST_AUTO_PTR_V2_H
