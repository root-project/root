#ifndef ATLASLIKEDATAVECTOR
#define ATLASLIKEDATAVECTOR

// Mimicks the DataVector hierarchy as found in athena

#include <RootMetaSelection.h>

#include <cstdint>
#include <string>
#include <vector>
#include <cstddef>

// https://gitlab.cern.ch/atlas/athena/-/blob/26885cbcf82f069cddff6cd30c123572e362d8fa/Control/AthContainers/AthContainers/tools/DVLNoBase.h#L32
namespace DataModel_detail {
struct NoBase {};
} // namespace DataModel_detail

// https://gitlab.cern.ch/atlas/athena/-/blob/26885cbcf82f069cddff6cd30c123572e362d8fa/Control/AthContainers/AthContainers/DataVector.h#L632
template <class T>
struct DataVectorBase {
   typedef DataModel_detail::NoBase Base;
};

// https://gitlab.cern.ch/atlas/athena/-/blob/26885cbcf82f069cddff6cd30c123572e362d8fa/Control/AthContainers/AthContainers/DataVector.h#L792
template <class T, class BASE = typename DataVectorBase<T>::Base>
class AtlasLikeDataVector : public BASE {};

namespace SG {
// https://gitlab.cern.ch/atlas/athena/-/blob/26885cbcf82f069cddff6cd30c123572e362d8fa/Control/AthContainers/AthContainers/AuxVectorData.h#L164
class AuxVectorData {
public:
   AuxVectorData() = default;
};
// https://gitlab.cern.ch/atlas/athena/-/blob/26885cbcf82f069cddff6cd30c123572e362d8fa/Control/AthContainers/AthContainers/AuxVectorBase.h#L96
class AuxVectorBase : public AuxVectorData {
public:
   AuxVectorBase() = default;
};
} // namespace SG

// https://gitlab.cern.ch/atlas/athena/-/blob/26885cbcf82f069cddff6cd30c123572e362d8fa/Control/AthContainers/AthContainers/DataVector.h#L2058
template <class T>
class AtlasLikeDataVector<T, DataModel_detail::NoBase> : public SG::AuxVectorBase {};

namespace ROOT::Meta::Selection {
// https://gitlab.cern.ch/atlas/athena/-/blob/26885cbcf82f069cddff6cd30c123572e362d8fa/Control/AthContainers/AthContainers/DataVector.h#L3422
template <class T, class BASE>
class AtlasLikeDataVector : KeepFirstTemplateArguments<1>, SelectNoInstance {};
} // namespace ROOT::Meta::Selection

struct CustomStruct {
   float a{};
   std::vector<float> v1{};
   std::vector<std::vector<float>> v2{};
   std::string s{};
   std::byte b{};
};

namespace {
// https://gitlab.cern.ch/atlas/athena/-/blob/26885cbcf82f069cddff6cd30c123572e362d8fa/Control/AthContainers/AthContainers/AthContainersDict.h#L29
struct DUMMY_INSTANTIATION {
   AtlasLikeDataVector<CustomStruct, DataModel_detail::NoBase> dummy1;
   ROOT::Meta::Selection::AtlasLikeDataVector<CustomStruct, DataModel_detail::NoBase> dummy2;
};
} // namespace

#endif // ATLASLIKEDATAVECTOR
