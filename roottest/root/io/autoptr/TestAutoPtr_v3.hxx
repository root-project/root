#ifndef TEST_AUTO_PTR_V3_H
#define TEST_AUTO_PTR_V3_H

#include <RtypesCore.h>

#include <memory>

namespace Compat {

template <typename T>
struct DeprecatedAutoPtr {
   // We use Compat::DeprecatedAutoPtr only to assign the wrapped raw pointer to a unique pointer
   // in an I/O customization rule.
   // However, since the DeprecatedAutoPtr object can be reused, it is essential to always reset the
   // value after using it, so we can safely delete it (it should always be nullptr)
   ~DeprecatedAutoPtr() { delete fRawPtr; }

   T *fRawPtr = nullptr;
};

} // namespace Compat

struct Track {
   int fFoo;

   ClassDefNV(Track, 2)
};

struct TestAutoPtr {
   std::unique_ptr<Track> fTrack;
   float fBar = 137.0;

   ClassDefNV(TestAutoPtr, 3)
};

#endif // TEST_AUTO_PTR_V3_H
