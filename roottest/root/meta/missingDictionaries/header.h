#if !defined(ROOTTEST_ROOT_META_MISSINGDICTIONARIES_HEADER_H)
#define ROOTTEST_ROOT_META_MISSINGDICTIONARIES_HEADER_H

#include <array>
#include <vector>

namespace edm {
template <typename T>
struct Ptr {
   T fValue;
};
template <typename T, int N>
struct Ref {
   T fValue;
};
} // namespace edm

namespace l1t::io_v1 {
struct L1Candidate {};
struct PFCandidate {};
struct SAMuon {};
} // namespace l1t::io_v1

namespace l1ct::io_v1 {
struct JetTagClass {};
} // namespace l1ct::io_v1

namespace reco::io_v1 {
struct GsfTrack {};
} // namespace reco::io_v1

namespace ticl::io_v1 {
struct Trackster {};
} // namespace ticl::io_v1

struct EcalEBTriggerPrimitiveSample {};

struct TriggerPrimitive {
   std::vector<EcalEBTriggerPrimitiveSample> fSamples;
};

struct Decoder {
   std::vector<std::pair<edm::Ptr<l1t::io_v1::L1Candidate>, float>> fCandidatesWeight;
};

struct Puppi {
   std::vector<edm::Ptr<l1t::io_v1::PFCandidate>> fCandidates;
   std::vector<l1ct::io_v1::JetTagClass> fJetTag;
};

struct Muons {
   std::vector<edm::Ref<std::vector<l1t::io_v1::SAMuon>, 2>> fMuon;
};

struct Candidate {
   std::vector<edm::Ptr<reco::io_v1::GsfTrack>> fTracks;
   std::vector<edm::Ptr<ticl::io_v1::Trackster>> fTrackster;
   std::vector<std::array<unsigned int, 2>> fEdges;
};

#endif // ROOTTEST_ROOT_META_MISSINGDICTIONARIES_HEADER_H
