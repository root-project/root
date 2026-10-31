/// \file IoReadTargetOrder.hxx
/// \brief Class definitions for the IoReadTargetOrder regression test (ROOT #23686).

#ifndef IOREADTARGETORDER_HXX
#define IOREADTARGETORDER_HXX

#include <Rtypes.h>
#include <cstdint>
#include <vector>

struct ClusterIoReadV1 {
   std::vector<uint8_t> v;
   uint16_t f = 0;

   ClassDefNV(ClusterIoReadV1, 1)
};

struct ClusterIoRead {
   std::vector<uint8_t> v;
   uint16_t f = 0;
   uint8_t q = 0;
   int ran = 0; //! transient - incremented by the read rule

   ClassDefNV(ClusterIoRead, 2)
};

template <class T>
struct HolderIoRead {
   std::vector<T> v;
   ClassDefNV(HolderIoRead, 1)
};

template <class T>
struct WrapperIoRead {
   T obj;
   ClassDefNV(WrapperIoRead, 1)
};

#endif // IOREADTARGETORDER_HXX
