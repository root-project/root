#include "atlasFlushedProjectHeaders.h"

#include "atlasFlushedLinkDef.h"

struct DeleteObjectFunctor {
   template <typename T>
   void operator()(const T *ptr) const {
      delete ptr;
   }
   template <typename T, typename Q>
   void operator()(const std::pair<T,Q> &) const {
      // Do nothing
   }
   template <typename T, typename Q>
   void operator()(const std::pair<T,Q*> &ptr) const {
      delete ptr.second;
   }
   template <typename T, typename Q>
   void operator()(const std::pair<T*,Q> &ptr) const {
      delete ptr.first;
   }
   template <typename T, typename Q>
   void operator()(const std::pair<T*,Q*> &ptr) const {
      delete ptr.first;
      delete ptr.second;
   }
};

#ifndef TPCnvTokenList_p1_cxx
#define TPCnvTokenList_p1_cxx
TPCnvTokenList_p1::TPCnvTokenList_p1() {
}
TPCnvTokenList_p1::TPCnvTokenList_p1(const TPCnvTokenList_p1 & rhs)
   : vector<TPCnvToken_p1>(const_cast<TPCnvTokenList_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   TPCnvTokenList_p1 &modrhs = const_cast<TPCnvTokenList_p1 &>( rhs );
   modrhs.clear();
}
TPCnvTokenList_p1::~TPCnvTokenList_p1() {
}
#endif // TPCnvTokenList_p1_cxx

#ifndef Trk__TrackCollection_p1_cxx
#define Trk__TrackCollection_p1_cxx
Trk::TrackCollection_p1::TrackCollection_p1() {
}
Trk::TrackCollection_p1::TrackCollection_p1(const TrackCollection_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrackCollection_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   TrackCollection_p1 &modrhs = const_cast<TrackCollection_p1 &>( rhs );
   modrhs.clear();
}
Trk::TrackCollection_p1::~TrackCollection_p1() {
}
#endif // Trk__TrackCollection_p1_cxx

#ifndef ElectronContainer_p1_cxx
#define ElectronContainer_p1_cxx
ElectronContainer_p1::ElectronContainer_p1() {
}
ElectronContainer_p1::ElectronContainer_p1(const ElectronContainer_p1 & rhs)
   : vector<Electron_p1>(const_cast<ElectronContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   ElectronContainer_p1 &modrhs = const_cast<ElectronContainer_p1 &>( rhs );
   modrhs.clear();
}
ElectronContainer_p1::~ElectronContainer_p1() {
}
#endif // ElectronContainer_p1_cxx

#ifndef Rec__TrackParticleContainer_p1_cxx
#define Rec__TrackParticleContainer_p1_cxx
Rec::TrackParticleContainer_p1::TrackParticleContainer_p1() {
}
Rec::TrackParticleContainer_p1::TrackParticleContainer_p1(const TrackParticleContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrackParticleContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   TrackParticleContainer_p1 &modrhs = const_cast<TrackParticleContainer_p1 &>( rhs );
   modrhs.clear();
}
Rec::TrackParticleContainer_p1::~TrackParticleContainer_p1() {
}
#endif // Rec__TrackParticleContainer_p1_cxx

#ifndef MuonContainer_p4_cxx
#define MuonContainer_p4_cxx
MuonContainer_p4::MuonContainer_p4() {
}
MuonContainer_p4::MuonContainer_p4(const MuonContainer_p4 & rhs)
   : vector<Muon_p4>(const_cast<MuonContainer_p4 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   MuonContainer_p4 &modrhs = const_cast<MuonContainer_p4 &>( rhs );
   modrhs.clear();
}
MuonContainer_p4::~MuonContainer_p4() {
}
#endif // MuonContainer_p4_cxx

#ifndef MuonCaloEnergyContainer_p1_cxx
#define MuonCaloEnergyContainer_p1_cxx
MuonCaloEnergyContainer_p1::MuonCaloEnergyContainer_p1() {
}
MuonCaloEnergyContainer_p1::MuonCaloEnergyContainer_p1(const MuonCaloEnergyContainer_p1 & rhs)
   : vector<CaloEnergy_p2>(const_cast<MuonCaloEnergyContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   MuonCaloEnergyContainer_p1 &modrhs = const_cast<MuonCaloEnergyContainer_p1 &>( rhs );
   modrhs.clear();
}
MuonCaloEnergyContainer_p1::~MuonCaloEnergyContainer_p1() {
}
#endif // MuonCaloEnergyContainer_p1_cxx

#ifndef TauJetContainer_p3_cxx
#define TauJetContainer_p3_cxx
TauJetContainer_p3::TauJetContainer_p3() {
}
TauJetContainer_p3::TauJetContainer_p3(const TauJetContainer_p3 & rhs)
   : vector<TauJet_p3>(const_cast<TauJetContainer_p3 &>( rhs ))
   , m_ROIauthor(const_cast<TauJetContainer_p3 &>( rhs ).m_ROIauthor)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   TauJetContainer_p3 &modrhs = const_cast<TauJetContainer_p3 &>( rhs );
   modrhs.clear();
   modrhs.m_ROIauthor.clear();
}
TauJetContainer_p3::~TauJetContainer_p3() {
}
#endif // TauJetContainer_p3_cxx

#ifndef TauDetailsContainer_p1_cxx
#define TauDetailsContainer_p1_cxx
TauDetailsContainer_p1::TauDetailsContainer_p1() {
}
TauDetailsContainer_p1::TauDetailsContainer_p1(const TauDetailsContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TauDetailsContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   TauDetailsContainer_p1 &modrhs = const_cast<TauDetailsContainer_p1 &>( rhs );
   modrhs.clear();
}
TauDetailsContainer_p1::~TauDetailsContainer_p1() {
}
#endif // TauDetailsContainer_p1_cxx

#ifndef TrigRoiDescriptorCollection_p1_cxx
#define TrigRoiDescriptorCollection_p1_cxx
TrigRoiDescriptorCollection_p1::TrigRoiDescriptorCollection_p1() {
}
TrigRoiDescriptorCollection_p1::TrigRoiDescriptorCollection_p1(const TrigRoiDescriptorCollection_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigRoiDescriptorCollection_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   TrigRoiDescriptorCollection_p1 &modrhs = const_cast<TrigRoiDescriptorCollection_p1 &>( rhs );
   modrhs.clear();
}
TrigRoiDescriptorCollection_p1::~TrigRoiDescriptorCollection_p1() {
}
#endif // TrigRoiDescriptorCollection_p1_cxx

#ifndef TrigTauClusterContainer_p1_cxx
#define TrigTauClusterContainer_p1_cxx
TrigTauClusterContainer_p1::TrigTauClusterContainer_p1() {
}
TrigTauClusterContainer_p1::TrigTauClusterContainer_p1(const TrigTauClusterContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigTauClusterContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   TrigTauClusterContainer_p1 &modrhs = const_cast<TrigTauClusterContainer_p1 &>( rhs );
   modrhs.clear();
}
TrigTauClusterContainer_p1::~TrigTauClusterContainer_p1() {
}
#endif // TrigTauClusterContainer_p1_cxx

#ifndef TrigT2JetContainer_p1_cxx
#define TrigT2JetContainer_p1_cxx
TrigT2JetContainer_p1::TrigT2JetContainer_p1() {
}
TrigT2JetContainer_p1::TrigT2JetContainer_p1(const TrigT2JetContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigT2JetContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   TrigT2JetContainer_p1 &modrhs = const_cast<TrigT2JetContainer_p1 &>( rhs );
   modrhs.clear();
}
TrigT2JetContainer_p1::~TrigT2JetContainer_p1() {
}
#endif // TrigT2JetContainer_p1_cxx

#ifndef TrigVertexCollection_p1_cxx
#define TrigVertexCollection_p1_cxx
TrigVertexCollection_p1::TrigVertexCollection_p1() {
}
TrigVertexCollection_p1::TrigVertexCollection_p1(const TrigVertexCollection_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigVertexCollection_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!
   TrigVertexCollection_p1 &modrhs = const_cast<TrigVertexCollection_p1 &>( rhs );
   modrhs.clear();
}
TrigVertexCollection_p1::~TrigVertexCollection_p1() {
}
#endif // TrigVertexCollection_p1_cxx

#ifndef TileMuFeatureContainer_p1_cxx
#define TileMuFeatureContainer_p1_cxx
TileMuFeatureContainer_p1::TileMuFeatureContainer_p1() {
}
TileMuFeatureContainer_p1::TileMuFeatureContainer_p1(const TileMuFeatureContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TileMuFeatureContainer_p1 &>( rhs ))
   , m_TileMuFeature(const_cast<TileMuFeatureContainer_p1 &>( rhs ).m_TileMuFeature)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TileMuFeatureContainer_p1 &modrhs = const_cast<TileMuFeatureContainer_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_TileMuFeature.clear();
}
TileMuFeatureContainer_p1::~TileMuFeatureContainer_p1() {
}
#endif // TileMuFeatureContainer_p1_cxx

#ifndef Trk__V0Container_p1_cxx
#define Trk__V0Container_p1_cxx
Trk::V0Container_p1::V0Container_p1() {
}
Trk::V0Container_p1::V0Container_p1(const V0Container_p1 & rhs)
   : vector<TPObjRef>(const_cast<V0Container_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   V0Container_p1 &modrhs = const_cast<V0Container_p1 &>( rhs );
   modrhs.clear();
}
Trk::V0Container_p1::~V0Container_p1() {
}
#endif // Trk__V0Container_p1_cxx

#ifndef TrigEFBphysContainer_p1_cxx
#define TrigEFBphysContainer_p1_cxx
TrigEFBphysContainer_p1::TrigEFBphysContainer_p1() {
}
TrigEFBphysContainer_p1::TrigEFBphysContainer_p1(const TrigEFBphysContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigEFBphysContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigEFBphysContainer_p1 &modrhs = const_cast<TrigEFBphysContainer_p1 &>( rhs );
   modrhs.clear();
}
TrigEFBphysContainer_p1::~TrigEFBphysContainer_p1() {
}
#endif // TrigEFBphysContainer_p1_cxx

#ifndef TrigElectronContainer_p2_cxx
#define TrigElectronContainer_p2_cxx
TrigElectronContainer_p2::TrigElectronContainer_p2() {
}
TrigElectronContainer_p2::TrigElectronContainer_p2(const TrigElectronContainer_p2 & rhs)
   : vector<TPObjRef>(const_cast<TrigElectronContainer_p2 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigElectronContainer_p2 &modrhs = const_cast<TrigElectronContainer_p2 &>( rhs );
   modrhs.clear();
}
TrigElectronContainer_p2::~TrigElectronContainer_p2() {
}
#endif // TrigElectronContainer_p2_cxx

#ifndef IsoMuonFeatureContainer_p1_cxx
#define IsoMuonFeatureContainer_p1_cxx
IsoMuonFeatureContainer_p1::IsoMuonFeatureContainer_p1() {
}
IsoMuonFeatureContainer_p1::IsoMuonFeatureContainer_p1(const IsoMuonFeatureContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<IsoMuonFeatureContainer_p1 &>( rhs ))
   , m_isoMuonFeature(const_cast<IsoMuonFeatureContainer_p1 &>( rhs ).m_isoMuonFeature)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   IsoMuonFeatureContainer_p1 &modrhs = const_cast<IsoMuonFeatureContainer_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_isoMuonFeature.clear();
}
IsoMuonFeatureContainer_p1::~IsoMuonFeatureContainer_p1() {
}
#endif // IsoMuonFeatureContainer_p1_cxx

#ifndef JetCollection_p5_cxx
#define JetCollection_p5_cxx
JetCollection_p5::JetCollection_p5() {
}
JetCollection_p5::JetCollection_p5(const JetCollection_p5 & rhs)
   : vector<TPObjRef>(const_cast<JetCollection_p5 &>( rhs ))
   , m_ordered(const_cast<JetCollection_p5 &>( rhs ).m_ordered)
   , m_keyStore(const_cast<JetCollection_p5 &>( rhs ).m_keyStore)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   JetCollection_p5 &modrhs = const_cast<JetCollection_p5 &>( rhs );
   modrhs.clear();
}
JetCollection_p5::~JetCollection_p5() {
}
#endif // JetCollection_p5_cxx

#ifndef TrigL2BphysContainer_p1_cxx
#define TrigL2BphysContainer_p1_cxx
TrigL2BphysContainer_p1::TrigL2BphysContainer_p1() {
}
TrigL2BphysContainer_p1::TrigL2BphysContainer_p1(const TrigL2BphysContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigL2BphysContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigL2BphysContainer_p1 &modrhs = const_cast<TrigL2BphysContainer_p1 &>( rhs );
   modrhs.clear();
}
TrigL2BphysContainer_p1::~TrigL2BphysContainer_p1() {
}
#endif // TrigL2BphysContainer_p1_cxx

#ifndef TrigPhotonContainer_p2_cxx
#define TrigPhotonContainer_p2_cxx
TrigPhotonContainer_p2::TrigPhotonContainer_p2() {
}
TrigPhotonContainer_p2::TrigPhotonContainer_p2(const TrigPhotonContainer_p2 & rhs)
   : vector<TPObjRef>(const_cast<TrigPhotonContainer_p2 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigPhotonContainer_p2 &modrhs = const_cast<TrigPhotonContainer_p2 &>( rhs );
   modrhs.clear();
}
TrigPhotonContainer_p2::~TrigPhotonContainer_p2() {
}
#endif // TrigPhotonContainer_p2_cxx

#ifndef PhotonContainer_p1_cxx
#define PhotonContainer_p1_cxx
PhotonContainer_p1::PhotonContainer_p1() {
}
PhotonContainer_p1::PhotonContainer_p1(const PhotonContainer_p1 & rhs)
   : vector<Photon_p1>(const_cast<PhotonContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   PhotonContainer_p1 &modrhs = const_cast<PhotonContainer_p1 &>( rhs );
   modrhs.clear();
}
PhotonContainer_p1::~PhotonContainer_p1() {
}
#endif // PhotonContainer_p1_cxx

#ifndef TrigTrackCountsCollection_p1_cxx
#define TrigTrackCountsCollection_p1_cxx
TrigTrackCountsCollection_p1::TrigTrackCountsCollection_p1() {
}
TrigTrackCountsCollection_p1::TrigTrackCountsCollection_p1(const TrigTrackCountsCollection_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigTrackCountsCollection_p1 &>( rhs ))
   , m_trigTrackCounts(const_cast<TrigTrackCountsCollection_p1 &>( rhs ).m_trigTrackCounts)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTrackCountsCollection_p1 &modrhs = const_cast<TrigTrackCountsCollection_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_trigTrackCounts.clear();
}
TrigTrackCountsCollection_p1::~TrigTrackCountsCollection_p1() {
}
#endif // TrigTrackCountsCollection_p1_cxx

#ifndef Trk__VxContainer_p1_cxx
#define Trk__VxContainer_p1_cxx
Trk::VxContainer_p1::VxContainer_p1() {
}
Trk::VxContainer_p1::VxContainer_p1(const VxContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<VxContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   VxContainer_p1 &modrhs = const_cast<VxContainer_p1 &>( rhs );
   modrhs.clear();
}
Trk::VxContainer_p1::~VxContainer_p1() {
}
#endif // Trk__VxContainer_p1_cxx

#ifndef TrigEMClusterContainer_p1_cxx
#define TrigEMClusterContainer_p1_cxx
TrigEMClusterContainer_p1::TrigEMClusterContainer_p1() {
}
TrigEMClusterContainer_p1::TrigEMClusterContainer_p1(const TrigEMClusterContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigEMClusterContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigEMClusterContainer_p1 &modrhs = const_cast<TrigEMClusterContainer_p1 &>( rhs );
   modrhs.clear();
}
TrigEMClusterContainer_p1::~TrigEMClusterContainer_p1() {
}
#endif // TrigEMClusterContainer_p1_cxx

#ifndef egammaContainer_p1_cxx
#define egammaContainer_p1_cxx
egammaContainer_p1::egammaContainer_p1() {
}
egammaContainer_p1::egammaContainer_p1(const egammaContainer_p1 & rhs)
   : vector<egamma_p1>(const_cast<egammaContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   egammaContainer_p1 &modrhs = const_cast<egammaContainer_p1 &>( rhs );
   modrhs.clear();
}
egammaContainer_p1::~egammaContainer_p1() {
}
#endif // egammaContainer_p1_cxx

#ifndef TrigSpacePointCountsCollection_p1_cxx
#define TrigSpacePointCountsCollection_p1_cxx
TrigSpacePointCountsCollection_p1::TrigSpacePointCountsCollection_p1() {
}
TrigSpacePointCountsCollection_p1::TrigSpacePointCountsCollection_p1(const TrigSpacePointCountsCollection_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigSpacePointCountsCollection_p1 &>( rhs ))
   , m_trigSpacePointCounts(const_cast<TrigSpacePointCountsCollection_p1 &>( rhs ).m_trigSpacePointCounts)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigSpacePointCountsCollection_p1 &modrhs = const_cast<TrigSpacePointCountsCollection_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_trigSpacePointCounts.clear();
}
TrigSpacePointCountsCollection_p1::~TrigSpacePointCountsCollection_p1() {
}
#endif // TrigSpacePointCountsCollection_p1_cxx

#ifndef TrigTauTracksInfoCollection_p1_cxx
#define TrigTauTracksInfoCollection_p1_cxx
TrigTauTracksInfoCollection_p1::TrigTauTracksInfoCollection_p1() {
}
TrigTauTracksInfoCollection_p1::TrigTauTracksInfoCollection_p1(const TrigTauTracksInfoCollection_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigTauTracksInfoCollection_p1 &>( rhs ))
   , m_trigTauTracksInfo(const_cast<TrigTauTracksInfoCollection_p1 &>( rhs ).m_trigTauTracksInfo)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTauTracksInfoCollection_p1 &modrhs = const_cast<TrigTauTracksInfoCollection_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_trigTauTracksInfo.clear();
}
TrigTauTracksInfoCollection_p1::~TrigTauTracksInfoCollection_p1() {
}
#endif // TrigTauTracksInfoCollection_p1_cxx

#ifndef egDetailContainer_p1_cxx
#define egDetailContainer_p1_cxx
egDetailContainer_p1::egDetailContainer_p1() {
}
egDetailContainer_p1::egDetailContainer_p1(const egDetailContainer_p1 & rhs)
   : vector<egDetail_p1>(const_cast<egDetailContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   egDetailContainer_p1 &modrhs = const_cast<egDetailContainer_p1 &>( rhs );
   modrhs.clear();
}
egDetailContainer_p1::~egDetailContainer_p1() {
}
#endif // egDetailContainer_p1_cxx

#ifndef TrigTauClusterDetailsContainer_p1_cxx
#define TrigTauClusterDetailsContainer_p1_cxx
TrigTauClusterDetailsContainer_p1::TrigTauClusterDetailsContainer_p1() {
}
TrigTauClusterDetailsContainer_p1::TrigTauClusterDetailsContainer_p1(const TrigTauClusterDetailsContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigTauClusterDetailsContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTauClusterDetailsContainer_p1 &modrhs = const_cast<TrigTauClusterDetailsContainer_p1 &>( rhs );
   modrhs.clear();
}
TrigTauClusterDetailsContainer_p1::~TrigTauClusterDetailsContainer_p1() {
}
#endif // TrigTauClusterDetailsContainer_p1_cxx

#ifndef TrigEFBjetContainer_p2_cxx
#define TrigEFBjetContainer_p2_cxx
TrigEFBjetContainer_p2::TrigEFBjetContainer_p2() {
}
TrigEFBjetContainer_p2::TrigEFBjetContainer_p2(const TrigEFBjetContainer_p2 & rhs)
   : vector<TPObjRef>(const_cast<TrigEFBjetContainer_p2 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigEFBjetContainer_p2 &modrhs = const_cast<TrigEFBjetContainer_p2 &>( rhs );
   modrhs.clear();
}
TrigEFBjetContainer_p2::~TrigEFBjetContainer_p2() {
}
#endif // TrigEFBjetContainer_p2_cxx

#ifndef TrigL2BjetContainer_p2_cxx
#define TrigL2BjetContainer_p2_cxx
TrigL2BjetContainer_p2::TrigL2BjetContainer_p2() {
}
TrigL2BjetContainer_p2::TrigL2BjetContainer_p2(const TrigL2BjetContainer_p2 & rhs)
   : vector<TPObjRef>(const_cast<TrigL2BjetContainer_p2 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigL2BjetContainer_p2 &modrhs = const_cast<TrigL2BjetContainer_p2 &>( rhs );
   modrhs.clear();
}
TrigL2BjetContainer_p2::~TrigL2BjetContainer_p2() {
}
#endif // TrigL2BjetContainer_p2_cxx

#ifndef TrigT2MbtsBitsContainer_p1_cxx
#define TrigT2MbtsBitsContainer_p1_cxx
TrigT2MbtsBitsContainer_p1::TrigT2MbtsBitsContainer_p1() {
}
TrigT2MbtsBitsContainer_p1::TrigT2MbtsBitsContainer_p1(const TrigT2MbtsBitsContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigT2MbtsBitsContainer_p1 &>( rhs ))
   , m_t2MbtsBits(const_cast<TrigT2MbtsBitsContainer_p1 &>( rhs ).m_t2MbtsBits)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigT2MbtsBitsContainer_p1 &modrhs = const_cast<TrigT2MbtsBitsContainer_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_t2MbtsBits.clear();
}
TrigT2MbtsBitsContainer_p1::~TrigT2MbtsBitsContainer_p1() {
}
#endif // TrigT2MbtsBitsContainer_p1_cxx

#ifndef CosmicMuonCollection_p1_cxx
#define CosmicMuonCollection_p1_cxx
CosmicMuonCollection_p1::CosmicMuonCollection_p1() {
}
CosmicMuonCollection_p1::CosmicMuonCollection_p1(const CosmicMuonCollection_p1 & rhs)
   : vector<TPObjRef>(const_cast<CosmicMuonCollection_p1 &>( rhs ))
   , m_cosmicMuon(const_cast<CosmicMuonCollection_p1 &>( rhs ).m_cosmicMuon)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CosmicMuonCollection_p1 &modrhs = const_cast<CosmicMuonCollection_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_cosmicMuon.clear();
}
CosmicMuonCollection_p1::~CosmicMuonCollection_p1() {
}
#endif // CosmicMuonCollection_p1_cxx

#ifndef Trk__SegmentCollection_p1_cxx
#define Trk__SegmentCollection_p1_cxx
Trk::SegmentCollection_p1::SegmentCollection_p1() {
}
Trk::SegmentCollection_p1::SegmentCollection_p1(const SegmentCollection_p1 & rhs)
   : vector<TPObjRef>(const_cast<SegmentCollection_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   SegmentCollection_p1 &modrhs = const_cast<SegmentCollection_p1 &>( rhs );
   modrhs.clear();
}
Trk::SegmentCollection_p1::~SegmentCollection_p1() {
}
#endif // Trk__SegmentCollection_p1_cxx

#ifndef RingerRingsContainer_p1_cxx
#define RingerRingsContainer_p1_cxx
RingerRingsContainer_p1::RingerRingsContainer_p1() {
}
RingerRingsContainer_p1::RingerRingsContainer_p1(const RingerRingsContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<RingerRingsContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   RingerRingsContainer_p1 &modrhs = const_cast<RingerRingsContainer_p1 &>( rhs );
   modrhs.clear();
}
RingerRingsContainer_p1::~RingerRingsContainer_p1() {
}
#endif // RingerRingsContainer_p1_cxx

#ifndef CombinedMuonFeatureContainer_p1_cxx
#define CombinedMuonFeatureContainer_p1_cxx
CombinedMuonFeatureContainer_p1::CombinedMuonFeatureContainer_p1() {
}
CombinedMuonFeatureContainer_p1::CombinedMuonFeatureContainer_p1(const CombinedMuonFeatureContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<CombinedMuonFeatureContainer_p1 &>( rhs ))
   , m_combinedMuonFeature(const_cast<CombinedMuonFeatureContainer_p1 &>( rhs ).m_combinedMuonFeature)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CombinedMuonFeatureContainer_p1 &modrhs = const_cast<CombinedMuonFeatureContainer_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_combinedMuonFeature.clear();
}
CombinedMuonFeatureContainer_p1::~CombinedMuonFeatureContainer_p1() {
}
#endif // CombinedMuonFeatureContainer_p1_cxx

#ifndef TileTrackMuFeatureContainer_p1_cxx
#define TileTrackMuFeatureContainer_p1_cxx
TileTrackMuFeatureContainer_p1::TileTrackMuFeatureContainer_p1() {
}
TileTrackMuFeatureContainer_p1::TileTrackMuFeatureContainer_p1(const TileTrackMuFeatureContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TileTrackMuFeatureContainer_p1 &>( rhs ))
   , m_TileTrackMuFeature(const_cast<TileTrackMuFeatureContainer_p1 &>( rhs ).m_TileTrackMuFeature)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TileTrackMuFeatureContainer_p1 &modrhs = const_cast<TileTrackMuFeatureContainer_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_TileTrackMuFeature.clear();
}
TileTrackMuFeatureContainer_p1::~TileTrackMuFeatureContainer_p1() {
}
#endif // TileTrackMuFeatureContainer_p1_cxx

#ifndef MdtTrackSegmentCollection_p1_cxx
#define MdtTrackSegmentCollection_p1_cxx
MdtTrackSegmentCollection_p1::MdtTrackSegmentCollection_p1() {
}
MdtTrackSegmentCollection_p1::MdtTrackSegmentCollection_p1(const MdtTrackSegmentCollection_p1 & rhs)
   : vector<TPObjRef>(const_cast<MdtTrackSegmentCollection_p1 &>( rhs ))
   , m_mdtTrackSegment(const_cast<MdtTrackSegmentCollection_p1 &>( rhs ).m_mdtTrackSegment)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MdtTrackSegmentCollection_p1 &modrhs = const_cast<MdtTrackSegmentCollection_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_mdtTrackSegment.clear();
}
MdtTrackSegmentCollection_p1::~MdtTrackSegmentCollection_p1() {
}
#endif // MdtTrackSegmentCollection_p1_cxx

#ifndef MuonFeatureContainer_p2_cxx
#define MuonFeatureContainer_p2_cxx
MuonFeatureContainer_p2::MuonFeatureContainer_p2() {
}
MuonFeatureContainer_p2::MuonFeatureContainer_p2(const MuonFeatureContainer_p2 & rhs)
   : vector<TPObjRef>(const_cast<MuonFeatureContainer_p2 &>( rhs ))
   , m_muonFeature(const_cast<MuonFeatureContainer_p2 &>( rhs ).m_muonFeature)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MuonFeatureContainer_p2 &modrhs = const_cast<MuonFeatureContainer_p2 &>( rhs );
   modrhs.clear();
   modrhs.m_muonFeature.clear();
}
MuonFeatureContainer_p2::~MuonFeatureContainer_p2() {
}
#endif // MuonFeatureContainer_p2_cxx

#ifndef TrigMuonEFContainer_p1_cxx
#define TrigMuonEFContainer_p1_cxx
TrigMuonEFContainer_p1::TrigMuonEFContainer_p1() {
}
TrigMuonEFContainer_p1::TrigMuonEFContainer_p1(const TrigMuonEFContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigMuonEFContainer_p1 &>( rhs ))
   , m_trigMuonEF(const_cast<TrigMuonEFContainer_p1 &>( rhs ).m_trigMuonEF)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigMuonEFContainer_p1 &modrhs = const_cast<TrigMuonEFContainer_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_trigMuonEF.clear();
}
TrigMuonEFContainer_p1::~TrigMuonEFContainer_p1() {
}
#endif // TrigMuonEFContainer_p1_cxx

#ifndef TrigTauContainer_p1_cxx
#define TrigTauContainer_p1_cxx
TrigTauContainer_p1::TrigTauContainer_p1() {
}
TrigTauContainer_p1::TrigTauContainer_p1(const TrigTauContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigTauContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTauContainer_p1 &modrhs = const_cast<TrigTauContainer_p1 &>( rhs );
   modrhs.clear();
}
TrigTauContainer_p1::~TrigTauContainer_p1() {
}
#endif // TrigTauContainer_p1_cxx

#ifndef TrigMuonEFInfoContainer_p1_cxx
#define TrigMuonEFInfoContainer_p1_cxx
TrigMuonEFInfoContainer_p1::TrigMuonEFInfoContainer_p1() {
}
TrigMuonEFInfoContainer_p1::TrigMuonEFInfoContainer_p1(const TrigMuonEFInfoContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigMuonEFInfoContainer_p1 &>( rhs ))
   , m_trigMuonEFInfo(const_cast<TrigMuonEFInfoContainer_p1 &>( rhs ).m_trigMuonEFInfo)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigMuonEFInfoContainer_p1 &modrhs = const_cast<TrigMuonEFInfoContainer_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_trigMuonEFInfo.clear();
}
TrigMuonEFInfoContainer_p1::~TrigMuonEFInfoContainer_p1() {
}
#endif // TrigMuonEFInfoContainer_p1_cxx

#ifndef TrigMuonEFInfoTrackContainer_p1_cxx
#define TrigMuonEFInfoTrackContainer_p1_cxx
TrigMuonEFInfoTrackContainer_p1::TrigMuonEFInfoTrackContainer_p1() {
}
TrigMuonEFInfoTrackContainer_p1::TrigMuonEFInfoTrackContainer_p1(const TrigMuonEFInfoTrackContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigMuonEFInfoTrackContainer_p1 &>( rhs ))
   , m_trigMuonEFInfoTrack(const_cast<TrigMuonEFInfoTrackContainer_p1 &>( rhs ).m_trigMuonEFInfoTrack)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigMuonEFInfoTrackContainer_p1 &modrhs = const_cast<TrigMuonEFInfoTrackContainer_p1 &>( rhs );
   modrhs.clear();
   modrhs.m_trigMuonEFInfoTrack.clear();
}
TrigMuonEFInfoTrackContainer_p1::~TrigMuonEFInfoTrackContainer_p1() {
}
#endif // TrigMuonEFInfoTrackContainer_p1_cxx

#ifndef TrigMissingETContainer_p1_cxx
#define TrigMissingETContainer_p1_cxx
TrigMissingETContainer_p1::TrigMissingETContainer_p1() {
}
TrigMissingETContainer_p1::TrigMissingETContainer_p1(const TrigMissingETContainer_p1 & rhs)
   : vector<TPObjRef>(const_cast<TrigMissingETContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigMissingETContainer_p1 &modrhs = const_cast<TrigMissingETContainer_p1 &>( rhs );
   modrhs.clear();
}
TrigMissingETContainer_p1::~TrigMissingETContainer_p1() {
}
#endif // TrigMissingETContainer_p1_cxx

#ifndef TileMuContainer_p1_cxx
#define TileMuContainer_p1_cxx
TileMuContainer_p1::TileMuContainer_p1() {
}
TileMuContainer_p1::TileMuContainer_p1(const TileMuContainer_p1 & rhs)
   : vector<TileMu_p1>(const_cast<TileMuContainer_p1 &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TileMuContainer_p1 &modrhs = const_cast<TileMuContainer_p1 &>( rhs );
   modrhs.clear();
}
TileMuContainer_p1::~TileMuContainer_p1() {
}
#endif // TileMuContainer_p1_cxx

#ifndef EventID_p1_cxx
#define EventID_p1_cxx
EventID_p1::EventID_p1() {
}
EventID_p1::EventID_p1(const EventID_p1 & rhs)
   : m_run_number(const_cast<EventID_p1 &>( rhs ).m_run_number)
   , m_event_number(const_cast<EventID_p1 &>( rhs ).m_event_number)
   , m_time_stamp(const_cast<EventID_p1 &>( rhs ).m_time_stamp)
   , m_time_stamp_ns_offset(const_cast<EventID_p1 &>( rhs ).m_time_stamp_ns_offset)
   , m_lumiBlock(const_cast<EventID_p1 &>( rhs ).m_lumiBlock)
   , m_bunch_crossing_id(const_cast<EventID_p1 &>( rhs ).m_bunch_crossing_id)
   , m_detector_mask0(const_cast<EventID_p1 &>( rhs ).m_detector_mask0)
   , m_detector_mask1(const_cast<EventID_p1 &>( rhs ).m_detector_mask1)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
EventID_p1::~EventID_p1() {
}
#endif // EventID_p1_cxx

#ifndef EventInfo_p2_cxx
#define EventInfo_p2_cxx
EventInfo_p2::EventInfo_p2() {
}
EventInfo_p2::EventInfo_p2(const EventInfo_p2 & rhs)
   : m_event_ID(const_cast<EventInfo_p2 &>( rhs ).m_event_ID)
   , m_event_type(const_cast<EventInfo_p2 &>( rhs ).m_event_type)
   , m_trigger_info(const_cast<EventInfo_p2 &>( rhs ).m_trigger_info)
   , m_event_flags(const_cast<EventInfo_p2 &>( rhs ).m_event_flags)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   EventInfo_p2 &modrhs = const_cast<EventInfo_p2 &>( rhs );
   modrhs.m_event_flags.clear();
}
EventInfo_p2::~EventInfo_p2() {
}
#endif // EventInfo_p2_cxx

#ifndef EventType_p1_cxx
#define EventType_p1_cxx
EventType_p1::EventType_p1() {
}
EventType_p1::EventType_p1(const EventType_p1 & rhs)
   : m_bit_mask(const_cast<EventType_p1 &>( rhs ).m_bit_mask)
   , m_user_type(const_cast<EventType_p1 &>( rhs ).m_user_type)
   , m_mc_event_weight(const_cast<EventType_p1 &>( rhs ).m_mc_event_weight)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   EventType_p1 &modrhs = const_cast<EventType_p1 &>( rhs );
   modrhs.m_bit_mask.clear();
   modrhs.m_user_type.clear();
}
EventType_p1::~EventType_p1() {
}
#endif // EventType_p1_cxx

#ifndef TriggerInfo_p2__StreamTag_p2_cxx
#define TriggerInfo_p2__StreamTag_p2_cxx
TriggerInfo_p2::StreamTag_p2::StreamTag_p2() {
}
TriggerInfo_p2::StreamTag_p2::StreamTag_p2(const StreamTag_p2 & rhs)
   : m_name(const_cast<StreamTag_p2 &>( rhs ).m_name)
   , m_type(const_cast<StreamTag_p2 &>( rhs ).m_type)
   , m_obeysLumiblock(const_cast<StreamTag_p2 &>( rhs ).m_obeysLumiblock)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   StreamTag_p2 &modrhs = const_cast<StreamTag_p2 &>( rhs );
   modrhs.m_name.clear();
   modrhs.m_type.clear();
}
TriggerInfo_p2::StreamTag_p2::~StreamTag_p2() {
}
#endif // TriggerInfo_p2__StreamTag_p2_cxx

#ifndef TriggerInfo_p2_cxx
#define TriggerInfo_p2_cxx
TriggerInfo_p2::TriggerInfo_p2() {
}
TriggerInfo_p2::TriggerInfo_p2(const TriggerInfo_p2 & rhs)
   : m_statusElement(const_cast<TriggerInfo_p2 &>( rhs ).m_statusElement)
   , m_extendedLevel1ID(const_cast<TriggerInfo_p2 &>( rhs ).m_extendedLevel1ID)
   , m_level1TriggerType(const_cast<TriggerInfo_p2 &>( rhs ).m_level1TriggerType)
   , m_level1TriggerInfo(const_cast<TriggerInfo_p2 &>( rhs ).m_level1TriggerInfo)
   , m_level2TriggerInfo(const_cast<TriggerInfo_p2 &>( rhs ).m_level2TriggerInfo)
   , m_eventFilterInfo(const_cast<TriggerInfo_p2 &>( rhs ).m_eventFilterInfo)
   , m_streamTags(const_cast<TriggerInfo_p2 &>( rhs ).m_streamTags)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TriggerInfo_p2 &modrhs = const_cast<TriggerInfo_p2 &>( rhs );
   modrhs.m_level1TriggerInfo.clear();
   modrhs.m_level2TriggerInfo.clear();
   modrhs.m_eventFilterInfo.clear();
   modrhs.m_streamTags.clear();
}
TriggerInfo_p2::~TriggerInfo_p2() {
}
#endif // TriggerInfo_p2_cxx

#ifndef CaloCompactCellContainer_cxx
#define CaloCompactCellContainer_cxx
CaloCompactCellContainer::CaloCompactCellContainer() {
}
CaloCompactCellContainer::CaloCompactCellContainer(const CaloCompactCellContainer & rhs)
   : m_compactData(const_cast<CaloCompactCellContainer &>( rhs ).m_compactData)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CaloCompactCellContainer &modrhs = const_cast<CaloCompactCellContainer &>( rhs );
   modrhs.m_compactData.clear();
}
CaloCompactCellContainer::~CaloCompactCellContainer() {
}
#endif // CaloCompactCellContainer_cxx

#ifndef P4EEtaPhiMFloat_p2_cxx
#define P4EEtaPhiMFloat_p2_cxx
P4EEtaPhiMFloat_p2::P4EEtaPhiMFloat_p2() {
}
P4EEtaPhiMFloat_p2::P4EEtaPhiMFloat_p2(const P4EEtaPhiMFloat_p2 & rhs)
   : m_e(const_cast<P4EEtaPhiMFloat_p2 &>( rhs ).m_e)
   , m_eta(const_cast<P4EEtaPhiMFloat_p2 &>( rhs ).m_eta)
   , m_phi(const_cast<P4EEtaPhiMFloat_p2 &>( rhs ).m_phi)
   , m_m(const_cast<P4EEtaPhiMFloat_p2 &>( rhs ).m_m)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
P4EEtaPhiMFloat_p2::~P4EEtaPhiMFloat_p2() {
}
#endif // P4EEtaPhiMFloat_p2_cxx

#ifndef ElementLink_p2_unsigned_int__cxx
#define ElementLink_p2_unsigned_int__cxx
ElementLink_p2<unsigned int>::ElementLink_p2() {
}
ElementLink_p2<unsigned int>::ElementLink_p2(const ElementLink_p2 & rhs)
   : m_contIndex(const_cast<ElementLink_p2 &>( rhs ).m_contIndex)
   , m_elementIndex(const_cast<ElementLink_p2 &>( rhs ).m_elementIndex)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
ElementLink_p2<unsigned int>::~ElementLink_p2() {
}
#endif // ElementLink_p2_unsigned_int__cxx

#ifndef CaloClusterContainer_p5__CaloCluster_p_cxx
#define CaloClusterContainer_p5__CaloCluster_p_cxx
CaloClusterContainer_p5::CaloCluster_p::CaloCluster_p() {
}
CaloClusterContainer_p5::CaloCluster_p::CaloCluster_p(const CaloCluster_p & rhs)
   : m_basicSignal(const_cast<CaloCluster_p &>( rhs ).m_basicSignal)
   , m_time(const_cast<CaloCluster_p &>( rhs ).m_time)
   , m_eta0(const_cast<CaloCluster_p &>( rhs ).m_eta0)
   , m_phi0(const_cast<CaloCluster_p &>( rhs ).m_phi0)
   , m_samplingPattern(const_cast<CaloCluster_p &>( rhs ).m_samplingPattern)
   , m_caloRecoStatus(const_cast<CaloCluster_p &>( rhs ).m_caloRecoStatus)
   , m_clusterSize(const_cast<CaloCluster_p &>( rhs ).m_clusterSize)
   , m_P4EEtaPhiM(const_cast<CaloCluster_p &>( rhs ).m_P4EEtaPhiM)
   , m_dataLink(const_cast<CaloCluster_p &>( rhs ).m_dataLink)
   , m_cellLink(const_cast<CaloCluster_p &>( rhs ).m_cellLink)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
CaloClusterContainer_p5::CaloCluster_p::~CaloCluster_p() {
}
#endif // CaloClusterContainer_p5__CaloCluster_p_cxx

#ifndef CaloClusterContainer_p5_cxx
#define CaloClusterContainer_p5_cxx
CaloClusterContainer_p5::CaloClusterContainer_p5() {
}
CaloClusterContainer_p5::CaloClusterContainer_p5(const CaloClusterContainer_p5 & rhs)
   : m_vec(const_cast<CaloClusterContainer_p5 &>( rhs ).m_vec)
   , m_varTypePattern(const_cast<CaloClusterContainer_p5 &>( rhs ).m_varTypePattern)
   , m_dataStore(const_cast<CaloClusterContainer_p5 &>( rhs ).m_dataStore)
   , m_momentContainer(const_cast<CaloClusterContainer_p5 &>( rhs ).m_momentContainer)
   , m_towerSeg(const_cast<CaloClusterContainer_p5 &>( rhs ).m_towerSeg)
   , m_linkNames(const_cast<CaloClusterContainer_p5 &>( rhs ).m_linkNames)
   , m_badClusIndexList(const_cast<CaloClusterContainer_p5 &>( rhs ).m_badClusIndexList)
   , m_badLayerStatusList(const_cast<CaloClusterContainer_p5 &>( rhs ).m_badLayerStatusList)
   , m_badEtaList(const_cast<CaloClusterContainer_p5 &>( rhs ).m_badEtaList)
   , m_badPhiList(const_cast<CaloClusterContainer_p5 &>( rhs ).m_badPhiList)
   , m_rawE(const_cast<CaloClusterContainer_p5 &>( rhs ).m_rawE)
   , m_rawEtaPhiM(const_cast<CaloClusterContainer_p5 &>( rhs ).m_rawEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CaloClusterContainer_p5 &modrhs = const_cast<CaloClusterContainer_p5 &>( rhs );
   modrhs.m_vec.clear();
   modrhs.m_dataStore.clear();
   modrhs.m_badClusIndexList.clear();
   modrhs.m_badLayerStatusList.clear();
   modrhs.m_badEtaList.clear();
   modrhs.m_badPhiList.clear();
   modrhs.m_rawE.clear();
   modrhs.m_rawEtaPhiM.clear();
}
CaloClusterContainer_p5::~CaloClusterContainer_p5() {
}
#endif // CaloClusterContainer_p5_cxx

#ifndef CaloClusterMomentContainer_p2_cxx
#define CaloClusterMomentContainer_p2_cxx
CaloClusterMomentContainer_p2::CaloClusterMomentContainer_p2() {
}
CaloClusterMomentContainer_p2::CaloClusterMomentContainer_p2(const CaloClusterMomentContainer_p2 & rhs)
   : m_nMoments(const_cast<CaloClusterMomentContainer_p2 &>( rhs ).m_nMoments)
   , m_Mvalue(const_cast<CaloClusterMomentContainer_p2 &>( rhs ).m_Mvalue)
   , m_Mkey(const_cast<CaloClusterMomentContainer_p2 &>( rhs ).m_Mkey)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CaloClusterMomentContainer_p2 &modrhs = const_cast<CaloClusterMomentContainer_p2 &>( rhs );
   modrhs.m_Mvalue.clear();
   modrhs.m_Mkey.clear();
}
CaloClusterMomentContainer_p2::~CaloClusterMomentContainer_p2() {
}
#endif // CaloClusterMomentContainer_p2_cxx

#ifndef CaloTowerSeg_p1_cxx
#define CaloTowerSeg_p1_cxx
CaloTowerSeg_p1::CaloTowerSeg_p1() {
}
CaloTowerSeg_p1::CaloTowerSeg_p1(const CaloTowerSeg_p1 & rhs)
   : m_neta(const_cast<CaloTowerSeg_p1 &>( rhs ).m_neta)
   , m_nphi(const_cast<CaloTowerSeg_p1 &>( rhs ).m_nphi)
   , m_etamin(const_cast<CaloTowerSeg_p1 &>( rhs ).m_etamin)
   , m_etamax(const_cast<CaloTowerSeg_p1 &>( rhs ).m_etamax)
   , m_phimin(const_cast<CaloTowerSeg_p1 &>( rhs ).m_phimin)
   , m_phimax(const_cast<CaloTowerSeg_p1 &>( rhs ).m_phimax)
   , m_deta(const_cast<CaloTowerSeg_p1 &>( rhs ).m_deta)
   , m_dphi(const_cast<CaloTowerSeg_p1 &>( rhs ).m_dphi)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
CaloTowerSeg_p1::~CaloTowerSeg_p1() {
}
#endif // CaloTowerSeg_p1_cxx

#ifndef ElementLinkContNames_p2_cxx
#define ElementLinkContNames_p2_cxx
ElementLinkContNames_p2::ElementLinkContNames_p2() {
}
ElementLinkContNames_p2::ElementLinkContNames_p2(const ElementLinkContNames_p2 & rhs)
   : m_names(const_cast<ElementLinkContNames_p2 &>( rhs ).m_names)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   ElementLinkContNames_p2 &modrhs = const_cast<ElementLinkContNames_p2 &>( rhs );
   modrhs.m_names.clear();
}
ElementLinkContNames_p2::~ElementLinkContNames_p2() {
}
#endif // ElementLinkContNames_p2_cxx

#ifndef TileCellVec_cxx
#define TileCellVec_cxx
TileCellVec::TileCellVec() {
}
TileCellVec::TileCellVec(const TileCellVec & rhs)
   : vector<unsigned int>(const_cast<TileCellVec &>( rhs ))
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TileCellVec &modrhs = const_cast<TileCellVec &>( rhs );
   modrhs.clear();
}
TileCellVec::~TileCellVec() {
}
#endif // TileCellVec_cxx

#ifndef TileMu_p1_cxx
#define TileMu_p1_cxx
TileMu_p1::TileMu_p1() {
}
TileMu_p1::TileMu_p1(const TileMu_p1 & rhs)
   : m_eta(const_cast<TileMu_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<TileMu_p1 &>( rhs ).m_phi)
   , m_energy_deposited(const_cast<TileMu_p1 &>( rhs ).m_energy_deposited)
   , m_quality_factor(const_cast<TileMu_p1 &>( rhs ).m_quality_factor)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TileMu_p1 &modrhs = const_cast<TileMu_p1 &>( rhs );
   modrhs.m_energy_deposited.clear();
}
TileMu_p1::~TileMu_p1() {
}
#endif // TileMu_p1_cxx

#ifndef P4EEtaPhiM_p1_cxx
#define P4EEtaPhiM_p1_cxx
P4EEtaPhiM_p1::P4EEtaPhiM_p1() {
}
P4EEtaPhiM_p1::P4EEtaPhiM_p1(const P4EEtaPhiM_p1 & rhs)
   : m_e(const_cast<P4EEtaPhiM_p1 &>( rhs ).m_e)
   , m_eta(const_cast<P4EEtaPhiM_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<P4EEtaPhiM_p1 &>( rhs ).m_phi)
   , m_m(const_cast<P4EEtaPhiM_p1 &>( rhs ).m_m)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
P4EEtaPhiM_p1::~P4EEtaPhiM_p1() {
}
#endif // P4EEtaPhiM_p1_cxx

#ifndef ElementLink_p1_unsigned_int__cxx
#define ElementLink_p1_unsigned_int__cxx
ElementLink_p1<unsigned int>::ElementLink_p1() {
}
ElementLink_p1<unsigned int>::ElementLink_p1(const ElementLink_p1 & rhs)
   : m_contName(const_cast<ElementLink_p1 &>( rhs ).m_contName)
   , m_elementIndex(const_cast<ElementLink_p1 &>( rhs ).m_elementIndex)
   , m_SGKeyHash(const_cast<ElementLink_p1 &>( rhs ).m_SGKeyHash)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   ElementLink_p1 &modrhs = const_cast<ElementLink_p1 &>( rhs );
   modrhs.m_contName.clear();
}
ElementLink_p1<unsigned int>::~ElementLink_p1() {
}
#endif // ElementLink_p1_unsigned_int__cxx

#ifndef ParticleBase_p1_cxx
#define ParticleBase_p1_cxx
ParticleBase_p1::ParticleBase_p1() {
}
ParticleBase_p1::ParticleBase_p1(const ParticleBase_p1 & rhs)
   : m_origin(const_cast<ParticleBase_p1 &>( rhs ).m_origin)
   , m_charge(const_cast<ParticleBase_p1 &>( rhs ).m_charge)
   , m_hasCharge(const_cast<ParticleBase_p1 &>( rhs ).m_hasCharge)
   , m_pdgId(const_cast<ParticleBase_p1 &>( rhs ).m_pdgId)
   , m_hasPdgId(const_cast<ParticleBase_p1 &>( rhs ).m_hasPdgId)
   , m_dataType(const_cast<ParticleBase_p1 &>( rhs ).m_dataType)
   , m_athenabarcode(const_cast<ParticleBase_p1 &>( rhs ).m_athenabarcode)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
ParticleBase_p1::~ParticleBase_p1() {
}
#endif // ParticleBase_p1_cxx

#ifndef AthenaBarCode_p1_cxx
#define AthenaBarCode_p1_cxx
AthenaBarCode_p1::AthenaBarCode_p1() {
}
AthenaBarCode_p1::AthenaBarCode_p1(const AthenaBarCode_p1 & rhs)
   : m_athenabarcode(const_cast<AthenaBarCode_p1 &>( rhs ).m_athenabarcode)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
AthenaBarCode_p1::~AthenaBarCode_p1() {
}
#endif // AthenaBarCode_p1_cxx

#ifndef ElementLinkVector_p1_unsigned_int___ElementRef_cxx
#define ElementLinkVector_p1_unsigned_int___ElementRef_cxx
ElementLinkVector_p1<unsigned int>::ElementRef::ElementRef() {
}
ElementLinkVector_p1<unsigned int>::ElementRef::ElementRef(const ElementRef & rhs)
   : m_elementIndex(const_cast<ElementRef &>( rhs ).m_elementIndex)
   , m_nameIndex(const_cast<ElementRef &>( rhs ).m_nameIndex)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
ElementLinkVector_p1<unsigned int>::ElementRef::~ElementRef() {
}
#endif // ElementLinkVector_p1_unsigned_int___ElementRef_cxx

#ifndef ElementLinkVector_p1_unsigned_int__cxx
#define ElementLinkVector_p1_unsigned_int__cxx
ElementLinkVector_p1<unsigned int>::ElementLinkVector_p1() {
}
ElementLinkVector_p1<unsigned int>::ElementLinkVector_p1(const ElementLinkVector_p1 & rhs)
   : m_elementRefs(const_cast<ElementLinkVector_p1 &>( rhs ).m_elementRefs)
   , m_links(const_cast<ElementLinkVector_p1 &>( rhs ).m_links)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   ElementLinkVector_p1 &modrhs = const_cast<ElementLinkVector_p1 &>( rhs );
   modrhs.m_elementRefs.clear();
   modrhs.m_links.clear();
}
ElementLinkVector_p1<unsigned int>::~ElementLinkVector_p1() {
}
#endif // ElementLinkVector_p1_unsigned_int__cxx

#ifndef egamma_p1_cxx
#define egamma_p1_cxx
egamma_p1::egamma_p1() {
}
egamma_p1::egamma_p1(const egamma_p1 & rhs)
   : m_momentum(const_cast<egamma_p1 &>( rhs ).m_momentum)
   , m_particleBase(const_cast<egamma_p1 &>( rhs ).m_particleBase)
   , m_cluster(const_cast<egamma_p1 &>( rhs ).m_cluster)
   , m_trackParticle(const_cast<egamma_p1 &>( rhs ).m_trackParticle)
   , m_conversion(const_cast<egamma_p1 &>( rhs ).m_conversion)
   , m_egDetails(const_cast<egamma_p1 &>( rhs ).m_egDetails)
   , m_author(const_cast<egamma_p1 &>( rhs ).m_author)
   , m_egammaEnumPIDs(const_cast<egamma_p1 &>( rhs ).m_egammaEnumPIDs)
   , m_egammaDblPIDs(const_cast<egamma_p1 &>( rhs ).m_egammaDblPIDs)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   egamma_p1 &modrhs = const_cast<egamma_p1 &>( rhs );
   modrhs.m_egammaEnumPIDs.clear();
   modrhs.m_egammaDblPIDs.clear();
}
egamma_p1::~egamma_p1() {
}
#endif // egamma_p1_cxx

#ifndef Electron_p1_cxx
#define Electron_p1_cxx
Electron_p1::Electron_p1() {
}
Electron_p1::Electron_p1(const Electron_p1 & rhs)
   : m_egamma(const_cast<Electron_p1 &>( rhs ).m_egamma)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Electron_p1::~Electron_p1() {
}
#endif // Electron_p1_cxx

#ifndef Photon_p1_cxx
#define Photon_p1_cxx
Photon_p1::Photon_p1() {
}
Photon_p1::Photon_p1(const Photon_p1 & rhs)
   : m_egamma(const_cast<Photon_p1 &>( rhs ).m_egamma)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Photon_p1::~Photon_p1() {
}
#endif // Photon_p1_cxx

#ifndef GenEvent_p4_cxx
#define GenEvent_p4_cxx
GenEvent_p4::GenEvent_p4() {
}
GenEvent_p4::GenEvent_p4(const GenEvent_p4 & rhs)
   : m_signalProcessId(const_cast<GenEvent_p4 &>( rhs ).m_signalProcessId)
   , m_eventNbr(const_cast<GenEvent_p4 &>( rhs ).m_eventNbr)
   , m_eventScale(const_cast<GenEvent_p4 &>( rhs ).m_eventScale)
   , m_alphaQCD(const_cast<GenEvent_p4 &>( rhs ).m_alphaQCD)
   , m_alphaQED(const_cast<GenEvent_p4 &>( rhs ).m_alphaQED)
   , m_signalProcessVtx(const_cast<GenEvent_p4 &>( rhs ).m_signalProcessVtx)
   , m_weights(const_cast<GenEvent_p4 &>( rhs ).m_weights)
   , m_pdfinfo(const_cast<GenEvent_p4 &>( rhs ).m_pdfinfo)
   , m_randomStates(const_cast<GenEvent_p4 &>( rhs ).m_randomStates)
   , m_verticesBegin(const_cast<GenEvent_p4 &>( rhs ).m_verticesBegin)
   , m_verticesEnd(const_cast<GenEvent_p4 &>( rhs ).m_verticesEnd)
   , m_particlesBegin(const_cast<GenEvent_p4 &>( rhs ).m_particlesBegin)
   , m_particlesEnd(const_cast<GenEvent_p4 &>( rhs ).m_particlesEnd)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   GenEvent_p4 &modrhs = const_cast<GenEvent_p4 &>( rhs );
   modrhs.m_weights.clear();
   modrhs.m_pdfinfo.clear();
   modrhs.m_randomStates.clear();
}
GenEvent_p4::~GenEvent_p4() {
}
#endif // GenEvent_p4_cxx

#ifndef McEventCollection_p4_cxx
#define McEventCollection_p4_cxx
McEventCollection_p4::McEventCollection_p4() {
}
McEventCollection_p4::McEventCollection_p4(const McEventCollection_p4 & rhs)
   : m_genEvents(const_cast<McEventCollection_p4 &>( rhs ).m_genEvents)
   , m_genVertices(const_cast<McEventCollection_p4 &>( rhs ).m_genVertices)
   , m_genParticles(const_cast<McEventCollection_p4 &>( rhs ).m_genParticles)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   McEventCollection_p4 &modrhs = const_cast<McEventCollection_p4 &>( rhs );
   modrhs.m_genEvents.clear();
   modrhs.m_genVertices.clear();
   modrhs.m_genParticles.clear();
}
McEventCollection_p4::~McEventCollection_p4() {
}
#endif // McEventCollection_p4_cxx

#ifndef GenVertex_p4_cxx
#define GenVertex_p4_cxx
GenVertex_p4::GenVertex_p4() {
}
GenVertex_p4::GenVertex_p4(const GenVertex_p4 & rhs)
   : m_x(const_cast<GenVertex_p4 &>( rhs ).m_x)
   , m_y(const_cast<GenVertex_p4 &>( rhs ).m_y)
   , m_z(const_cast<GenVertex_p4 &>( rhs ).m_z)
   , m_t(const_cast<GenVertex_p4 &>( rhs ).m_t)
   , m_particlesIn(const_cast<GenVertex_p4 &>( rhs ).m_particlesIn)
   , m_particlesOut(const_cast<GenVertex_p4 &>( rhs ).m_particlesOut)
   , m_id(const_cast<GenVertex_p4 &>( rhs ).m_id)
   , m_weights(const_cast<GenVertex_p4 &>( rhs ).m_weights)
   , m_barcode(const_cast<GenVertex_p4 &>( rhs ).m_barcode)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   GenVertex_p4 &modrhs = const_cast<GenVertex_p4 &>( rhs );
   modrhs.m_particlesIn.clear();
   modrhs.m_particlesOut.clear();
   modrhs.m_weights.clear();
}
GenVertex_p4::~GenVertex_p4() {
}
#endif // GenVertex_p4_cxx

#ifndef GenParticle_p4_cxx
#define GenParticle_p4_cxx
GenParticle_p4::GenParticle_p4() {
}
GenParticle_p4::GenParticle_p4(const GenParticle_p4 & rhs)
   : m_px(const_cast<GenParticle_p4 &>( rhs ).m_px)
   , m_py(const_cast<GenParticle_p4 &>( rhs ).m_py)
   , m_pz(const_cast<GenParticle_p4 &>( rhs ).m_pz)
   , m_m(const_cast<GenParticle_p4 &>( rhs ).m_m)
   , m_pdgId(const_cast<GenParticle_p4 &>( rhs ).m_pdgId)
   , m_status(const_cast<GenParticle_p4 &>( rhs ).m_status)
   , m_flow(const_cast<GenParticle_p4 &>( rhs ).m_flow)
   , m_thetaPolarization(const_cast<GenParticle_p4 &>( rhs ).m_thetaPolarization)
   , m_phiPolarization(const_cast<GenParticle_p4 &>( rhs ).m_phiPolarization)
   , m_prodVtx(const_cast<GenParticle_p4 &>( rhs ).m_prodVtx)
   , m_endVtx(const_cast<GenParticle_p4 &>( rhs ).m_endVtx)
   , m_barcode(const_cast<GenParticle_p4 &>( rhs ).m_barcode)
   , m_recoMethod(const_cast<GenParticle_p4 &>( rhs ).m_recoMethod)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   GenParticle_p4 &modrhs = const_cast<GenParticle_p4 &>( rhs );
   modrhs.m_flow.clear();
}
GenParticle_p4::~GenParticle_p4() {
}
#endif // GenParticle_p4_cxx

#ifndef TrigDec__TrigDecision_p4_cxx
#define TrigDec__TrigDecision_p4_cxx
TrigDec::TrigDecision_p4::TrigDecision_p4() {
}
TrigDec::TrigDecision_p4::TrigDecision_p4(const TrigDecision_p4 & rhs)
   : m_configMasterKey(const_cast<TrigDecision_p4 &>( rhs ).m_configMasterKey)
   , m_bgCode(const_cast<TrigDecision_p4 &>( rhs ).m_bgCode)
   , m_l1_result(const_cast<TrigDecision_p4 &>( rhs ).m_l1_result)
   , m_l2_result(const_cast<TrigDecision_p4 &>( rhs ).m_l2_result)
   , m_ef_result(const_cast<TrigDecision_p4 &>( rhs ).m_ef_result)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigDec::TrigDecision_p4::~TrigDecision_p4() {
}
#endif // TrigDec__TrigDecision_p4_cxx

#ifndef LVL1CTP__Lvl1Result_p2_cxx
#define LVL1CTP__Lvl1Result_p2_cxx
LVL1CTP::Lvl1Result_p2::Lvl1Result_p2() {
}
LVL1CTP::Lvl1Result_p2::Lvl1Result_p2(const Lvl1Result_p2 & rhs)
   : m_configured(const_cast<Lvl1Result_p2 &>( rhs ).m_configured)
   , m_l1_itemsTBP(const_cast<Lvl1Result_p2 &>( rhs ).m_l1_itemsTBP)
   , m_l1_itemsTAP(const_cast<Lvl1Result_p2 &>( rhs ).m_l1_itemsTAP)
   , m_l1_itemsTAV(const_cast<Lvl1Result_p2 &>( rhs ).m_l1_itemsTAV)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Lvl1Result_p2 &modrhs = const_cast<Lvl1Result_p2 &>( rhs );
   modrhs.m_l1_itemsTBP.clear();
   modrhs.m_l1_itemsTAP.clear();
   modrhs.m_l1_itemsTAV.clear();
}
LVL1CTP::Lvl1Result_p2::~Lvl1Result_p2() {
}
#endif // LVL1CTP__Lvl1Result_p2_cxx

#ifndef DataLink_p1_cxx
#define DataLink_p1_cxx
DataLink_p1::DataLink_p1() {
}
DataLink_p1::DataLink_p1(const DataLink_p1 & rhs)
   : m_link(const_cast<DataLink_p1 &>( rhs ).m_link)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   DataLink_p1 &modrhs = const_cast<DataLink_p1 &>( rhs );
   modrhs.m_link.clear();
}
DataLink_p1::~DataLink_p1() {
}
#endif // DataLink_p1_cxx

#ifndef Muon_ROI_p1_cxx
#define Muon_ROI_p1_cxx
Muon_ROI_p1::Muon_ROI_p1() {
}
Muon_ROI_p1::Muon_ROI_p1(const Muon_ROI_p1 & rhs)
   : m_roiWord(const_cast<Muon_ROI_p1 &>( rhs ).m_roiWord)
   , m_eta(const_cast<Muon_ROI_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<Muon_ROI_p1 &>( rhs ).m_phi)
   , m_thrValue(const_cast<Muon_ROI_p1 &>( rhs ).m_thrValue)
   , m_thrName(const_cast<Muon_ROI_p1 &>( rhs ).m_thrName)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Muon_ROI_p1 &modrhs = const_cast<Muon_ROI_p1 &>( rhs );
   modrhs.m_thrName.clear();
}
Muon_ROI_p1::~Muon_ROI_p1() {
}
#endif // Muon_ROI_p1_cxx

#ifndef LVL1_ROI_p1_cxx
#define LVL1_ROI_p1_cxx
LVL1_ROI_p1::LVL1_ROI_p1() {
}
LVL1_ROI_p1::LVL1_ROI_p1(const LVL1_ROI_p1 & rhs)
   : m_muonROIs(const_cast<LVL1_ROI_p1 &>( rhs ).m_muonROIs)
   , m_jetROIs(const_cast<LVL1_ROI_p1 &>( rhs ).m_jetROIs)
   , m_jetetROIs(const_cast<LVL1_ROI_p1 &>( rhs ).m_jetetROIs)
   , m_emtauROIs(const_cast<LVL1_ROI_p1 &>( rhs ).m_emtauROIs)
   , m_energysumROIs(const_cast<LVL1_ROI_p1 &>( rhs ).m_energysumROIs)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   LVL1_ROI_p1 &modrhs = const_cast<LVL1_ROI_p1 &>( rhs );
   modrhs.m_muonROIs.clear();
   modrhs.m_jetROIs.clear();
   modrhs.m_jetetROIs.clear();
   modrhs.m_emtauROIs.clear();
   modrhs.m_energysumROIs.clear();
}
LVL1_ROI_p1::~LVL1_ROI_p1() {
}
#endif // LVL1_ROI_p1_cxx

#ifndef Jet_ROI_p1_cxx
#define Jet_ROI_p1_cxx
Jet_ROI_p1::Jet_ROI_p1() {
}
Jet_ROI_p1::Jet_ROI_p1(const Jet_ROI_p1 & rhs)
   : m_roiWord(const_cast<Jet_ROI_p1 &>( rhs ).m_roiWord)
   , m_eta(const_cast<Jet_ROI_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<Jet_ROI_p1 &>( rhs ).m_phi)
   , m_ET4x4(const_cast<Jet_ROI_p1 &>( rhs ).m_ET4x4)
   , m_ET6x6(const_cast<Jet_ROI_p1 &>( rhs ).m_ET6x6)
   , m_ET8x8(const_cast<Jet_ROI_p1 &>( rhs ).m_ET8x8)
   , m_thresholdNames(const_cast<Jet_ROI_p1 &>( rhs ).m_thresholdNames)
   , m_thresholdValues(const_cast<Jet_ROI_p1 &>( rhs ).m_thresholdValues)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Jet_ROI_p1 &modrhs = const_cast<Jet_ROI_p1 &>( rhs );
   modrhs.m_thresholdNames.clear();
   modrhs.m_thresholdValues.clear();
}
Jet_ROI_p1::~Jet_ROI_p1() {
}
#endif // Jet_ROI_p1_cxx

#ifndef JetET_ROI_p1_cxx
#define JetET_ROI_p1_cxx
JetET_ROI_p1::JetET_ROI_p1() {
}
JetET_ROI_p1::JetET_ROI_p1(const JetET_ROI_p1 & rhs)
   : m_roiWord(const_cast<JetET_ROI_p1 &>( rhs ).m_roiWord)
   , m_thresholds(const_cast<JetET_ROI_p1 &>( rhs ).m_thresholds)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   JetET_ROI_p1 &modrhs = const_cast<JetET_ROI_p1 &>( rhs );
   modrhs.m_thresholds.clear();
}
JetET_ROI_p1::~JetET_ROI_p1() {
}
#endif // JetET_ROI_p1_cxx

#ifndef EmTau_ROI_p1_cxx
#define EmTau_ROI_p1_cxx
EmTau_ROI_p1::EmTau_ROI_p1() {
}
EmTau_ROI_p1::EmTau_ROI_p1(const EmTau_ROI_p1 & rhs)
   : m_roiWord(const_cast<EmTau_ROI_p1 &>( rhs ).m_roiWord)
   , m_eta(const_cast<EmTau_ROI_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<EmTau_ROI_p1 &>( rhs ).m_phi)
   , m_Core(const_cast<EmTau_ROI_p1 &>( rhs ).m_Core)
   , m_EMClus(const_cast<EmTau_ROI_p1 &>( rhs ).m_EMClus)
   , m_TauClus(const_cast<EmTau_ROI_p1 &>( rhs ).m_TauClus)
   , m_EMIsol(const_cast<EmTau_ROI_p1 &>( rhs ).m_EMIsol)
   , m_HadIsol(const_cast<EmTau_ROI_p1 &>( rhs ).m_HadIsol)
   , m_HadCore(const_cast<EmTau_ROI_p1 &>( rhs ).m_HadCore)
   , m_thresholdNames(const_cast<EmTau_ROI_p1 &>( rhs ).m_thresholdNames)
   , m_thresholdValues(const_cast<EmTau_ROI_p1 &>( rhs ).m_thresholdValues)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   EmTau_ROI_p1 &modrhs = const_cast<EmTau_ROI_p1 &>( rhs );
   modrhs.m_thresholdNames.clear();
   modrhs.m_thresholdValues.clear();
}
EmTau_ROI_p1::~EmTau_ROI_p1() {
}
#endif // EmTau_ROI_p1_cxx

#ifndef EnergySum_ROI_p1_cxx
#define EnergySum_ROI_p1_cxx
EnergySum_ROI_p1::EnergySum_ROI_p1() {
}
EnergySum_ROI_p1::EnergySum_ROI_p1(const EnergySum_ROI_p1 & rhs)
   : m_roiWord0(const_cast<EnergySum_ROI_p1 &>( rhs ).m_roiWord0)
   , m_roiWord1(const_cast<EnergySum_ROI_p1 &>( rhs ).m_roiWord1)
   , m_roiWord2(const_cast<EnergySum_ROI_p1 &>( rhs ).m_roiWord2)
   , m_energyX(const_cast<EnergySum_ROI_p1 &>( rhs ).m_energyX)
   , m_energyY(const_cast<EnergySum_ROI_p1 &>( rhs ).m_energyY)
   , m_energyT(const_cast<EnergySum_ROI_p1 &>( rhs ).m_energyT)
   , m_thresholds(const_cast<EnergySum_ROI_p1 &>( rhs ).m_thresholds)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   EnergySum_ROI_p1 &modrhs = const_cast<EnergySum_ROI_p1 &>( rhs );
   modrhs.m_thresholds.clear();
}
EnergySum_ROI_p1::~EnergySum_ROI_p1() {
}
#endif // EnergySum_ROI_p1_cxx

#ifndef TPCnvToken_p1_cxx
#define TPCnvToken_p1_cxx
TPCnvToken_p1::TPCnvToken_p1() {
}
TPCnvToken_p1::TPCnvToken_p1(const TPCnvToken_p1 & rhs)
   : m_converterID(const_cast<TPCnvToken_p1 &>( rhs ).m_converterID)
   , m_token(const_cast<TPCnvToken_p1 &>( rhs ).m_token)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TPCnvToken_p1 &modrhs = const_cast<TPCnvToken_p1 &>( rhs );
   modrhs.m_token.clear();
}
TPCnvToken_p1::~TPCnvToken_p1() {
}
#endif // TPCnvToken_p1_cxx

#ifndef Trk__VxContainer_tlp1_cxx
#define Trk__VxContainer_tlp1_cxx
Trk::VxContainer_tlp1::VxContainer_tlp1() {
}
Trk::VxContainer_tlp1::VxContainer_tlp1(const VxContainer_tlp1 & rhs)
   : m_tokenList(const_cast<VxContainer_tlp1 &>( rhs ).m_tokenList)
   , m_vxContainers(const_cast<VxContainer_tlp1 &>( rhs ).m_vxContainers)
   , m_vxCandidates(const_cast<VxContainer_tlp1 &>( rhs ).m_vxCandidates)
   , m_extendedVxCandidates(const_cast<VxContainer_tlp1 &>( rhs ).m_extendedVxCandidates)
   , m_vxTrackAtVertices(const_cast<VxContainer_tlp1 &>( rhs ).m_vxTrackAtVertices)
   , m_recVertices(const_cast<VxContainer_tlp1 &>( rhs ).m_recVertices)
   , m_vertices(const_cast<VxContainer_tlp1 &>( rhs ).m_vertices)
   , m_tracks(const_cast<VxContainer_tlp1 &>( rhs ).m_tracks)
   , m_trackParameters(const_cast<VxContainer_tlp1 &>( rhs ).m_trackParameters)
   , m_perigees(const_cast<VxContainer_tlp1 &>( rhs ).m_perigees)
   , m_measPerigees(const_cast<VxContainer_tlp1 &>( rhs ).m_measPerigees)
   , m_surfaces(const_cast<VxContainer_tlp1 &>( rhs ).m_surfaces)
   , m_fitQualities(const_cast<VxContainer_tlp1 &>( rhs ).m_fitQualities)
   , m_hepSymMatrices(const_cast<VxContainer_tlp1 &>( rhs ).m_hepSymMatrices)
   , m_localPositions(const_cast<VxContainer_tlp1 &>( rhs ).m_localPositions)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   VxContainer_tlp1 &modrhs = const_cast<VxContainer_tlp1 &>( rhs );
   modrhs.m_vxContainers.clear();
   modrhs.m_vxCandidates.clear();
   modrhs.m_extendedVxCandidates.clear();
   modrhs.m_vxTrackAtVertices.clear();
   modrhs.m_recVertices.clear();
   modrhs.m_vertices.clear();
   modrhs.m_tracks.clear();
   modrhs.m_trackParameters.clear();
   modrhs.m_perigees.clear();
   modrhs.m_measPerigees.clear();
   modrhs.m_surfaces.clear();
   modrhs.m_fitQualities.clear();
   modrhs.m_hepSymMatrices.clear();
   modrhs.m_localPositions.clear();
}
Trk::VxContainer_tlp1::~VxContainer_tlp1() {
}
#endif // Trk__VxContainer_tlp1_cxx

#ifndef TPObjRef__typeID_t_cxx
#define TPObjRef__typeID_t_cxx
TPObjRef::typeID_t::typeID_t() {
}
TPObjRef::typeID_t::typeID_t(const typeID_t & rhs)
   : m_TLCnvID(const_cast<typeID_t &>( rhs ).m_TLCnvID)
   , m_cnvID(const_cast<typeID_t &>( rhs ).m_cnvID)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TPObjRef::typeID_t::~typeID_t() {
}
#endif // TPObjRef__typeID_t_cxx

#ifndef TPObjRef_cxx
#define TPObjRef_cxx
TPObjRef::TPObjRef() {
}
TPObjRef::TPObjRef(const TPObjRef & rhs)
   : m_typeID(const_cast<TPObjRef &>( rhs ).m_typeID)
   , m_index(const_cast<TPObjRef &>( rhs ).m_index)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TPObjRef::~TPObjRef() {
}
#endif // TPObjRef_cxx

#ifndef Trk__VxCandidate_p1_cxx
#define Trk__VxCandidate_p1_cxx
Trk::VxCandidate_p1::VxCandidate_p1() {
}
Trk::VxCandidate_p1::VxCandidate_p1(const VxCandidate_p1 & rhs)
   : m_recVertex(const_cast<VxCandidate_p1 &>( rhs ).m_recVertex)
   , m_vxTrackAtVertex(const_cast<VxCandidate_p1 &>( rhs ).m_vxTrackAtVertex)
   , m_vertexType(const_cast<VxCandidate_p1 &>( rhs ).m_vertexType)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   VxCandidate_p1 &modrhs = const_cast<VxCandidate_p1 &>( rhs );
   modrhs.m_vxTrackAtVertex.clear();
}
Trk::VxCandidate_p1::~VxCandidate_p1() {
}
#endif // Trk__VxCandidate_p1_cxx

#ifndef Trk__ExtendedVxCandidate_p1_cxx
#define Trk__ExtendedVxCandidate_p1_cxx
Trk::ExtendedVxCandidate_p1::ExtendedVxCandidate_p1() {
}
Trk::ExtendedVxCandidate_p1::ExtendedVxCandidate_p1(const ExtendedVxCandidate_p1 & rhs)
   : m_vxCandidate(const_cast<ExtendedVxCandidate_p1 &>( rhs ).m_vxCandidate)
   , m_fullCovariance(const_cast<ExtendedVxCandidate_p1 &>( rhs ).m_fullCovariance)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::ExtendedVxCandidate_p1::~ExtendedVxCandidate_p1() {
}
#endif // Trk__ExtendedVxCandidate_p1_cxx

#ifndef Trk__VxTrackAtVertex_p1_cxx
#define Trk__VxTrackAtVertex_p1_cxx
Trk::VxTrackAtVertex_p1::VxTrackAtVertex_p1() {
}
Trk::VxTrackAtVertex_p1::VxTrackAtVertex_p1(const VxTrackAtVertex_p1 & rhs)
   : m_trkWeight(const_cast<VxTrackAtVertex_p1 &>( rhs ).m_trkWeight)
   , m_VertexCompatibility(const_cast<VxTrackAtVertex_p1 &>( rhs ).m_VertexCompatibility)
   , m_perigeeAtVertex(const_cast<VxTrackAtVertex_p1 &>( rhs ).m_perigeeAtVertex)
   , m_typeOfLink(const_cast<VxTrackAtVertex_p1 &>( rhs ).m_typeOfLink)
   , m_origTrack(const_cast<VxTrackAtVertex_p1 &>( rhs ).m_origTrack)
   , m_origTrackNames(const_cast<VxTrackAtVertex_p1 &>( rhs ).m_origTrackNames)
   , m_fitQuality(const_cast<VxTrackAtVertex_p1 &>( rhs ).m_fitQuality)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::VxTrackAtVertex_p1::~VxTrackAtVertex_p1() {
}
#endif // Trk__VxTrackAtVertex_p1_cxx

#ifndef Trk__RecVertex_p1_cxx
#define Trk__RecVertex_p1_cxx
Trk::RecVertex_p1::RecVertex_p1() {
}
Trk::RecVertex_p1::RecVertex_p1(const RecVertex_p1 & rhs)
   : vtx(const_cast<RecVertex_p1 &>( rhs ).vtx)
   , m_positionError(const_cast<RecVertex_p1 &>( rhs ).m_positionError)
   , m_fitQuality(const_cast<RecVertex_p1 &>( rhs ).m_fitQuality)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::RecVertex_p1::~RecVertex_p1() {
}
#endif // Trk__RecVertex_p1_cxx

#ifndef Trk__Vertex_p1_cxx
#define Trk__Vertex_p1_cxx
Trk::Vertex_p1::Vertex_p1() {
}
Trk::Vertex_p1::Vertex_p1(const Vertex_p1 & rhs)
   : m_position(const_cast<Vertex_p1 &>( rhs ).m_position)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Vertex_p1 &modrhs = const_cast<Vertex_p1 &>( rhs );
   modrhs.m_position.clear();
}
Trk::Vertex_p1::~Vertex_p1() {
}
#endif // Trk__Vertex_p1_cxx

#ifndef Trk__Track_p1_cxx
#define Trk__Track_p1_cxx
Trk::Track_p1::Track_p1() {
}
Trk::Track_p1::Track_p1(const Track_p1 & rhs)
   : m_author(const_cast<Track_p1 &>( rhs ).m_author)
   , m_particleHypo(const_cast<Track_p1 &>( rhs ).m_particleHypo)
   , m_fitQuality(const_cast<Track_p1 &>( rhs ).m_fitQuality)
   , m_trackState(const_cast<Track_p1 &>( rhs ).m_trackState)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Track_p1 &modrhs = const_cast<Track_p1 &>( rhs );
   modrhs.m_trackState.clear();
}
Trk::Track_p1::~Track_p1() {
}
#endif // Trk__Track_p1_cxx

#ifndef Trk__TrackParameters_p1_cxx
#define Trk__TrackParameters_p1_cxx
Trk::TrackParameters_p1::TrackParameters_p1() {
}
Trk::TrackParameters_p1::TrackParameters_p1(const TrackParameters_p1 & rhs)
   : m_parameters(const_cast<TrackParameters_p1 &>( rhs ).m_parameters)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrackParameters_p1 &modrhs = const_cast<TrackParameters_p1 &>( rhs );
   modrhs.m_parameters.clear();
}
Trk::TrackParameters_p1::~TrackParameters_p1() {
}
#endif // Trk__TrackParameters_p1_cxx

#ifndef Trk__Perigee_p1_cxx
#define Trk__Perigee_p1_cxx
Trk::Perigee_p1::Perigee_p1() {
}
Trk::Perigee_p1::Perigee_p1(const Perigee_p1 & rhs)
   : m_parameters(const_cast<Perigee_p1 &>( rhs ).m_parameters)
   , m_assocSurface(const_cast<Perigee_p1 &>( rhs ).m_assocSurface)
   , m_localPos(const_cast<Perigee_p1 &>( rhs ).m_localPos)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::Perigee_p1::~Perigee_p1() {
}
#endif // Trk__Perigee_p1_cxx

#ifndef Trk__MeasuredPerigee_p1_cxx
#define Trk__MeasuredPerigee_p1_cxx
Trk::MeasuredPerigee_p1::MeasuredPerigee_p1() {
}
Trk::MeasuredPerigee_p1::MeasuredPerigee_p1(const MeasuredPerigee_p1 & rhs)
   : m_perigee(const_cast<MeasuredPerigee_p1 &>( rhs ).m_perigee)
   , m_errorMatrix(const_cast<MeasuredPerigee_p1 &>( rhs ).m_errorMatrix)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::MeasuredPerigee_p1::~MeasuredPerigee_p1() {
}
#endif // Trk__MeasuredPerigee_p1_cxx

#ifndef Trk__Surface_p1_cxx
#define Trk__Surface_p1_cxx
Trk::Surface_p1::Surface_p1() {
}
Trk::Surface_p1::Surface_p1(const Surface_p1 & rhs)
   : m_associatedDetElementId(const_cast<Surface_p1 &>( rhs ).m_associatedDetElementId)
   , m_transform(const_cast<Surface_p1 &>( rhs ).m_transform)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Surface_p1 &modrhs = const_cast<Surface_p1 &>( rhs );
   modrhs.m_transform.clear();
}
Trk::Surface_p1::~Surface_p1() {
}
#endif // Trk__Surface_p1_cxx

#ifndef Trk__FitQuality_p1_cxx
#define Trk__FitQuality_p1_cxx
Trk::FitQuality_p1::FitQuality_p1() {
}
Trk::FitQuality_p1::FitQuality_p1(const FitQuality_p1 & rhs)
   : m_chiSquared(const_cast<FitQuality_p1 &>( rhs ).m_chiSquared)
   , m_numberDoF(const_cast<FitQuality_p1 &>( rhs ).m_numberDoF)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::FitQuality_p1::~FitQuality_p1() {
}
#endif // Trk__FitQuality_p1_cxx

#ifndef Trk__HepSymMatrix_p1_cxx
#define Trk__HepSymMatrix_p1_cxx
Trk::HepSymMatrix_p1::HepSymMatrix_p1() {
}
Trk::HepSymMatrix_p1::HepSymMatrix_p1(const HepSymMatrix_p1 & rhs)
   : m_matrix_val(const_cast<HepSymMatrix_p1 &>( rhs ).m_matrix_val)
   , m_nrow(const_cast<HepSymMatrix_p1 &>( rhs ).m_nrow)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   HepSymMatrix_p1 &modrhs = const_cast<HepSymMatrix_p1 &>( rhs );
   modrhs.m_matrix_val.clear();
}
Trk::HepSymMatrix_p1::~HepSymMatrix_p1() {
}
#endif // Trk__HepSymMatrix_p1_cxx

#ifndef Trk__LocalPosition_p1_cxx
#define Trk__LocalPosition_p1_cxx
Trk::LocalPosition_p1::LocalPosition_p1() {
}
Trk::LocalPosition_p1::LocalPosition_p1(const LocalPosition_p1 & rhs)
   : m_x(const_cast<LocalPosition_p1 &>( rhs ).m_x)
   , m_y(const_cast<LocalPosition_p1 &>( rhs ).m_y)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::LocalPosition_p1::~LocalPosition_p1() {
}
#endif // Trk__LocalPosition_p1_cxx

#ifndef TauJet_p3_cxx
#define TauJet_p3_cxx
TauJet_p3::TauJet_p3() {
}
TauJet_p3::TauJet_p3(const TauJet_p3 & rhs)
   : m_momentum(const_cast<TauJet_p3 &>( rhs ).m_momentum)
   , m_particleBase(const_cast<TauJet_p3 &>( rhs ).m_particleBase)
   , m_cluster(const_cast<TauJet_p3 &>( rhs ).m_cluster)
   , m_cellCluster(const_cast<TauJet_p3 &>( rhs ).m_cellCluster)
   , m_jet(const_cast<TauJet_p3 &>( rhs ).m_jet)
   , m_tracks(const_cast<TauJet_p3 &>( rhs ).m_tracks)
   , m_tauDetails(const_cast<TauJet_p3 &>( rhs ).m_tauDetails)
   , m_flags(const_cast<TauJet_p3 &>( rhs ).m_flags)
   , m_vetoFlags(const_cast<TauJet_p3 &>( rhs ).m_vetoFlags)
   , m_isTauFlags(const_cast<TauJet_p3 &>( rhs ).m_isTauFlags)
   , m_numberOfTracks(const_cast<TauJet_p3 &>( rhs ).m_numberOfTracks)
   , m_roiWord(const_cast<TauJet_p3 &>( rhs ).m_roiWord)
   , m_params(const_cast<TauJet_p3 &>( rhs ).m_params)
   , m_conversionTracks(const_cast<TauJet_p3 &>( rhs ).m_conversionTracks)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TauJet_p3 &modrhs = const_cast<TauJet_p3 &>( rhs );
   modrhs.m_params.clear();
}
TauJet_p3::~TauJet_p3() {
}
#endif // TauJet_p3_cxx

#ifndef JetKeyDescriptor_p1_cxx
#define JetKeyDescriptor_p1_cxx
JetKeyDescriptor_p1::JetKeyDescriptor_p1() {
}
JetKeyDescriptor_p1::JetKeyDescriptor_p1(const JetKeyDescriptor_p1 & rhs)
   : m_keyStoreLength(const_cast<JetKeyDescriptor_p1 &>( rhs ).m_keyStoreLength)
   , m_keyStore(const_cast<JetKeyDescriptor_p1 &>( rhs ).m_keyStore)
   , m_catStore(const_cast<JetKeyDescriptor_p1 &>( rhs ).m_catStore)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   JetKeyDescriptor_p1 &modrhs = const_cast<JetKeyDescriptor_p1 &>( rhs );
   modrhs.m_keyStoreLength.clear();
   modrhs.m_keyStore.clear();
   modrhs.m_catStore.clear();
}
JetKeyDescriptor_p1::~JetKeyDescriptor_p1() {
}
#endif // JetKeyDescriptor_p1_cxx

#ifndef MissingEtCalo_p2_cxx
#define MissingEtCalo_p2_cxx
MissingEtCalo_p2::MissingEtCalo_p2() {
}
MissingEtCalo_p2::MissingEtCalo_p2(const MissingEtCalo_p2 & rhs)
   : m_AllTheData(const_cast<MissingEtCalo_p2 &>( rhs ).m_AllTheData)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MissingEtCalo_p2 &modrhs = const_cast<MissingEtCalo_p2 &>( rhs );
   modrhs.m_AllTheData.clear();
}
MissingEtCalo_p2::~MissingEtCalo_p2() {
}
#endif // MissingEtCalo_p2_cxx

#ifndef MissingEtRegions_p1_cxx
#define MissingEtRegions_p1_cxx
MissingEtRegions_p1::MissingEtRegions_p1() {
}
MissingEtRegions_p1::MissingEtRegions_p1(const MissingEtRegions_p1 & rhs)
   : m_exReg(const_cast<MissingEtRegions_p1 &>( rhs ).m_exReg)
   , m_eyReg(const_cast<MissingEtRegions_p1 &>( rhs ).m_eyReg)
   , m_etReg(const_cast<MissingEtRegions_p1 &>( rhs ).m_etReg)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MissingEtRegions_p1 &modrhs = const_cast<MissingEtRegions_p1 &>( rhs );
   modrhs.m_exReg.clear();
   modrhs.m_eyReg.clear();
   modrhs.m_etReg.clear();
}
MissingEtRegions_p1::~MissingEtRegions_p1() {
}
#endif // MissingEtRegions_p1_cxx

#ifndef MissingET_p1_cxx
#define MissingET_p1_cxx
MissingET_p1::MissingET_p1() {
}
MissingET_p1::MissingET_p1(const MissingET_p1 & rhs)
   : m_regions(const_cast<MissingET_p1 &>( rhs ).m_regions)
   , m_source(const_cast<MissingET_p1 &>( rhs ).m_source)
   , m_ex(const_cast<MissingET_p1 &>( rhs ).m_ex)
   , m_ey(const_cast<MissingET_p1 &>( rhs ).m_ey)
   , m_etSum(const_cast<MissingET_p1 &>( rhs ).m_etSum)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
MissingET_p1::~MissingET_p1() {
}
#endif // MissingET_p1_cxx

#ifndef MissingEtTruth_p1_cxx
#define MissingEtTruth_p1_cxx
MissingEtTruth_p1::MissingEtTruth_p1() {
}
MissingEtTruth_p1::MissingEtTruth_p1(const MissingEtTruth_p1 & rhs)
   : m_met(const_cast<MissingEtTruth_p1 &>( rhs ).m_met)
   , m_exTruth(const_cast<MissingEtTruth_p1 &>( rhs ).m_exTruth)
   , m_eyTruth(const_cast<MissingEtTruth_p1 &>( rhs ).m_eyTruth)
   , m_etSumTruth(const_cast<MissingEtTruth_p1 &>( rhs ).m_etSumTruth)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MissingEtTruth_p1 &modrhs = const_cast<MissingEtTruth_p1 &>( rhs );
   modrhs.m_exTruth.clear();
   modrhs.m_eyTruth.clear();
   modrhs.m_etSumTruth.clear();
}
MissingEtTruth_p1::~MissingEtTruth_p1() {
}
#endif // MissingEtTruth_p1_cxx

#ifndef MissingET_p2_cxx
#define MissingET_p2_cxx
MissingET_p2::MissingET_p2() {
}
MissingET_p2::MissingET_p2(const MissingET_p2 & rhs)
   : m_AllTheData(const_cast<MissingET_p2 &>( rhs ).m_AllTheData)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MissingET_p2 &modrhs = const_cast<MissingET_p2 &>( rhs );
   modrhs.m_AllTheData.clear();
}
MissingET_p2::~MissingET_p2() {
}
#endif // MissingET_p2_cxx

#ifndef TruthParticleContainer_p5_cxx
#define TruthParticleContainer_p5_cxx
TruthParticleContainer_p5::TruthParticleContainer_p5() {
}
TruthParticleContainer_p5::TruthParticleContainer_p5(const TruthParticleContainer_p5 & rhs)
   : m_genEvent(const_cast<TruthParticleContainer_p5 &>( rhs ).m_genEvent)
   , m_etIsolations(const_cast<TruthParticleContainer_p5 &>( rhs ).m_etIsolations)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TruthParticleContainer_p5::~TruthParticleContainer_p5() {
}
#endif // TruthParticleContainer_p5_cxx

#ifndef TrigInDetTrackTruthMap_p1_cxx
#define TrigInDetTrackTruthMap_p1_cxx
TrigInDetTrackTruthMap_p1::TrigInDetTrackTruthMap_p1() {
}
TrigInDetTrackTruthMap_p1::TrigInDetTrackTruthMap_p1(const TrigInDetTrackTruthMap_p1 & rhs)
   : m_elink_vec(const_cast<TrigInDetTrackTruthMap_p1 &>( rhs ).m_elink_vec)
   , m_truth_vec(const_cast<TrigInDetTrackTruthMap_p1 &>( rhs ).m_truth_vec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigInDetTrackTruthMap_p1 &modrhs = const_cast<TrigInDetTrackTruthMap_p1 &>( rhs );
   modrhs.m_truth_vec.clear();
}
TrigInDetTrackTruthMap_p1::~TrigInDetTrackTruthMap_p1() {
}
#endif // TrigInDetTrackTruthMap_p1_cxx

#ifndef TrigInDetTrackTruthMap_tlp1_cxx
#define TrigInDetTrackTruthMap_tlp1_cxx
TrigInDetTrackTruthMap_tlp1::TrigInDetTrackTruthMap_tlp1() {
}
TrigInDetTrackTruthMap_tlp1::TrigInDetTrackTruthMap_tlp1(const TrigInDetTrackTruthMap_tlp1 & rhs)
   : m_trigInDetTrackTruthMap_p1(const_cast<TrigInDetTrackTruthMap_tlp1 &>( rhs ).m_trigInDetTrackTruthMap_p1)
   , m_trigInDetTrackTruth_p1(const_cast<TrigInDetTrackTruthMap_tlp1 &>( rhs ).m_trigInDetTrackTruth_p1)
   , m_trigIDHitStats_p1(const_cast<TrigInDetTrackTruthMap_tlp1 &>( rhs ).m_trigIDHitStats_p1)
   , m_hepMcParticleLink_p1(const_cast<TrigInDetTrackTruthMap_tlp1 &>( rhs ).m_hepMcParticleLink_p1)
   , m_trigInDetTrackCollection_p1(const_cast<TrigInDetTrackTruthMap_tlp1 &>( rhs ).m_trigInDetTrackCollection_p1)
   , m_trigInDetTrack_p1(const_cast<TrigInDetTrackTruthMap_tlp1 &>( rhs ).m_trigInDetTrack_p1)
   , m_trigInDetTrackFitPar_p1(const_cast<TrigInDetTrackTruthMap_tlp1 &>( rhs ).m_trigInDetTrackFitPar_p1)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigInDetTrackTruthMap_tlp1 &modrhs = const_cast<TrigInDetTrackTruthMap_tlp1 &>( rhs );
   modrhs.m_trigInDetTrackTruthMap_p1.clear();
   modrhs.m_trigInDetTrackTruth_p1.clear();
   modrhs.m_trigIDHitStats_p1.clear();
   modrhs.m_hepMcParticleLink_p1.clear();
   modrhs.m_trigInDetTrackCollection_p1.clear();
   modrhs.m_trigInDetTrack_p1.clear();
   modrhs.m_trigInDetTrackFitPar_p1.clear();
}
TrigInDetTrackTruthMap_tlp1::~TrigInDetTrackTruthMap_tlp1() {
}
#endif // TrigInDetTrackTruthMap_tlp1_cxx

#ifndef TrigInDetTrackTruth_p1_cxx
#define TrigInDetTrackTruth_p1_cxx
TrigInDetTrackTruth_p1::TrigInDetTrackTruth_p1() {
}
TrigInDetTrackTruth_p1::TrigInDetTrackTruth_p1(const TrigInDetTrackTruth_p1 & rhs)
   : best_match_hits(const_cast<TrigInDetTrackTruth_p1 &>( rhs ).best_match_hits)
   , best_Si_match_hits(const_cast<TrigInDetTrackTruth_p1 &>( rhs ).best_Si_match_hits)
   , best_TRT_match_hits(const_cast<TrigInDetTrackTruth_p1 &>( rhs ).best_TRT_match_hits)
   , m_true_part_vec(const_cast<TrigInDetTrackTruth_p1 &>( rhs ).m_true_part_vec)
   , m_nr_common_hits(const_cast<TrigInDetTrackTruth_p1 &>( rhs ).m_nr_common_hits)
   , m_family_tree(const_cast<TrigInDetTrackTruth_p1 &>( rhs ).m_family_tree)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigInDetTrackTruth_p1 &modrhs = const_cast<TrigInDetTrackTruth_p1 &>( rhs );
   modrhs.m_true_part_vec.clear();
   modrhs.m_nr_common_hits.clear();
   modrhs.m_family_tree.clear();
}
TrigInDetTrackTruth_p1::~TrigInDetTrackTruth_p1() {
}
#endif // TrigInDetTrackTruth_p1_cxx

#ifndef TrigIDHitStats_p1_cxx
#define TrigIDHitStats_p1_cxx
TrigIDHitStats_p1::TrigIDHitStats_p1() {
}
TrigIDHitStats_p1::TrigIDHitStats_p1(const TrigIDHitStats_p1 & rhs)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<3;i++) numHits[i] = rhs.numHits[i];
}
TrigIDHitStats_p1::~TrigIDHitStats_p1() {
}
#endif // TrigIDHitStats_p1_cxx

#ifndef HepMcParticleLink_p1_cxx
#define HepMcParticleLink_p1_cxx
HepMcParticleLink_p1::HepMcParticleLink_p1() {
}
HepMcParticleLink_p1::HepMcParticleLink_p1(const HepMcParticleLink_p1 & rhs)
   : m_mcEvtIndex(const_cast<HepMcParticleLink_p1 &>( rhs ).m_mcEvtIndex)
   , m_barcode(const_cast<HepMcParticleLink_p1 &>( rhs ).m_barcode)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
HepMcParticleLink_p1::~HepMcParticleLink_p1() {
}
#endif // HepMcParticleLink_p1_cxx

#ifndef TrigInDetTrackCollection_p1_cxx
#define TrigInDetTrackCollection_p1_cxx
TrigInDetTrackCollection_p1::TrigInDetTrackCollection_p1() {
}
TrigInDetTrackCollection_p1::TrigInDetTrackCollection_p1(const TrigInDetTrackCollection_p1 & rhs)
   : m_RoI_ID(const_cast<TrigInDetTrackCollection_p1 &>( rhs ).m_RoI_ID)
   , m_trigInDetTrackVector(const_cast<TrigInDetTrackCollection_p1 &>( rhs ).m_trigInDetTrackVector)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigInDetTrackCollection_p1 &modrhs = const_cast<TrigInDetTrackCollection_p1 &>( rhs );
   modrhs.m_trigInDetTrackVector.clear();
}
TrigInDetTrackCollection_p1::~TrigInDetTrackCollection_p1() {
}
#endif // TrigInDetTrackCollection_p1_cxx

#ifndef TrigInDetTrack_p1_cxx
#define TrigInDetTrack_p1_cxx
TrigInDetTrack_p1::TrigInDetTrack_p1() {
}
TrigInDetTrack_p1::TrigInDetTrack_p1(const TrigInDetTrack_p1 & rhs)
   : m_algId(const_cast<TrigInDetTrack_p1 &>( rhs ).m_algId)
   , m_param(const_cast<TrigInDetTrack_p1 &>( rhs ).m_param)
   , m_endParam(const_cast<TrigInDetTrack_p1 &>( rhs ).m_endParam)
   , m_chi2(const_cast<TrigInDetTrack_p1 &>( rhs ).m_chi2)
   , m_NStrawHits(const_cast<TrigInDetTrack_p1 &>( rhs ).m_NStrawHits)
   , m_NStraw(const_cast<TrigInDetTrack_p1 &>( rhs ).m_NStraw)
   , m_NStrawTime(const_cast<TrigInDetTrack_p1 &>( rhs ).m_NStrawTime)
   , m_NTRHits(const_cast<TrigInDetTrack_p1 &>( rhs ).m_NTRHits)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigInDetTrack_p1::~TrigInDetTrack_p1() {
}
#endif // TrigInDetTrack_p1_cxx

#ifndef TrigInDetTrackFitPar_p1_cxx
#define TrigInDetTrackFitPar_p1_cxx
TrigInDetTrackFitPar_p1::TrigInDetTrackFitPar_p1() {
   m_cov = 0;
}
TrigInDetTrackFitPar_p1::TrigInDetTrackFitPar_p1(const TrigInDetTrackFitPar_p1 & rhs)
   : m_a0(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_a0)
   , m_phi0(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_phi0)
   , m_z0(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_z0)
   , m_eta(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_eta)
   , m_pT(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_pT)
   , m_ea0(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_ea0)
   , m_ephi0(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_ephi0)
   , m_ez0(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_ez0)
   , m_eeta(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_eeta)
   , m_epT(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_epT)
   , m_cov(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_cov)
   , m_surfaceType(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_surfaceType)
   , m_surfaceCoordinate(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_surfaceCoordinate)
   , m_covtmp(const_cast<TrigInDetTrackFitPar_p1 &>( rhs ).m_covtmp)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigInDetTrackFitPar_p1 &modrhs = const_cast<TrigInDetTrackFitPar_p1 &>( rhs );
   modrhs.m_cov = 0;
   modrhs.m_covtmp.clear();
}
TrigInDetTrackFitPar_p1::~TrigInDetTrackFitPar_p1() {
   delete m_cov;   m_cov = 0;
}
#endif // TrigInDetTrackFitPar_p1_cxx

#ifndef INav4MomAssocs_p2_cxx
#define INav4MomAssocs_p2_cxx
INav4MomAssocs_p2::INav4MomAssocs_p2() {
}
INav4MomAssocs_p2::INav4MomAssocs_p2(const INav4MomAssocs_p2 & rhs)
   : m_contNames(const_cast<INav4MomAssocs_p2 &>( rhs ).m_contNames)
   , m_assocs(const_cast<INav4MomAssocs_p2 &>( rhs ).m_assocs)
   , m_assocStores(const_cast<INav4MomAssocs_p2 &>( rhs ).m_assocStores)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   INav4MomAssocs_p2 &modrhs = const_cast<INav4MomAssocs_p2 &>( rhs );
   modrhs.m_assocs.clear();
   modrhs.m_assocStores.clear();
}
INav4MomAssocs_p2::~INav4MomAssocs_p2() {
}
#endif // INav4MomAssocs_p2_cxx

#ifndef Trk__SegmentCollection_tlp1_cxx
#define Trk__SegmentCollection_tlp1_cxx
Trk::SegmentCollection_tlp1::SegmentCollection_tlp1() {
}
Trk::SegmentCollection_tlp1::SegmentCollection_tlp1(const SegmentCollection_tlp1 & rhs)
   : m_tokenList(const_cast<SegmentCollection_tlp1 &>( rhs ).m_tokenList)
   , m_segmentCollections(const_cast<SegmentCollection_tlp1 &>( rhs ).m_segmentCollections)
   , m_segments(const_cast<SegmentCollection_tlp1 &>( rhs ).m_segments)
   , m_tracksegments(const_cast<SegmentCollection_tlp1 &>( rhs ).m_tracksegments)
   , m_boundSurfaces(const_cast<SegmentCollection_tlp1 &>( rhs ).m_boundSurfaces)
   , m_surfaces(const_cast<SegmentCollection_tlp1 &>( rhs ).m_surfaces)
   , m_cylinderBounds(const_cast<SegmentCollection_tlp1 &>( rhs ).m_cylinderBounds)
   , m_diamondBounds(const_cast<SegmentCollection_tlp1 &>( rhs ).m_diamondBounds)
   , m_discBounds(const_cast<SegmentCollection_tlp1 &>( rhs ).m_discBounds)
   , m_rectangleBounds(const_cast<SegmentCollection_tlp1 &>( rhs ).m_rectangleBounds)
   , m_trapesoidBounds(const_cast<SegmentCollection_tlp1 &>( rhs ).m_trapesoidBounds)
   , m_rotatedTrapesoidBounds(const_cast<SegmentCollection_tlp1 &>( rhs ).m_rotatedTrapesoidBounds)
   , m_fitQualities(const_cast<SegmentCollection_tlp1 &>( rhs ).m_fitQualities)
   , m_localParameters(const_cast<SegmentCollection_tlp1 &>( rhs ).m_localParameters)
   , m_hepSymMatrices(const_cast<SegmentCollection_tlp1 &>( rhs ).m_hepSymMatrices)
   , m_RIO_OnTrack(const_cast<SegmentCollection_tlp1 &>( rhs ).m_RIO_OnTrack)
   , m_pseudoMeasurementOnTrack(const_cast<SegmentCollection_tlp1 &>( rhs ).m_pseudoMeasurementOnTrack)
   , m_competingRotsOnTrack(const_cast<SegmentCollection_tlp1 &>( rhs ).m_competingRotsOnTrack)
   , m_detElementSurfaces(const_cast<SegmentCollection_tlp1 &>( rhs ).m_detElementSurfaces)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   SegmentCollection_tlp1 &modrhs = const_cast<SegmentCollection_tlp1 &>( rhs );
   modrhs.m_segmentCollections.clear();
   modrhs.m_segments.clear();
   modrhs.m_tracksegments.clear();
   modrhs.m_boundSurfaces.clear();
   modrhs.m_surfaces.clear();
   modrhs.m_cylinderBounds.clear();
   modrhs.m_diamondBounds.clear();
   modrhs.m_discBounds.clear();
   modrhs.m_rectangleBounds.clear();
   modrhs.m_trapesoidBounds.clear();
   modrhs.m_rotatedTrapesoidBounds.clear();
   modrhs.m_fitQualities.clear();
   modrhs.m_localParameters.clear();
   modrhs.m_hepSymMatrices.clear();
   modrhs.m_RIO_OnTrack.clear();
   modrhs.m_pseudoMeasurementOnTrack.clear();
   modrhs.m_competingRotsOnTrack.clear();
   modrhs.m_detElementSurfaces.clear();
}
Trk::SegmentCollection_tlp1::~SegmentCollection_tlp1() {
}
#endif // Trk__SegmentCollection_tlp1_cxx

#ifndef Trk__Segment_p1_cxx
#define Trk__Segment_p1_cxx
Trk::Segment_p1::Segment_p1() {
}
Trk::Segment_p1::Segment_p1(const Segment_p1 & rhs)
   : m_localParameters(const_cast<Segment_p1 &>( rhs ).m_localParameters)
   , m_localErrorMatrix(const_cast<Segment_p1 &>( rhs ).m_localErrorMatrix)
   , m_fitQuality(const_cast<Segment_p1 &>( rhs ).m_fitQuality)
   , m_containedMeasBases(const_cast<Segment_p1 &>( rhs ).m_containedMeasBases)
   , m_author(const_cast<Segment_p1 &>( rhs ).m_author)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Segment_p1 &modrhs = const_cast<Segment_p1 &>( rhs );
   modrhs.m_containedMeasBases.clear();
}
Trk::Segment_p1::~Segment_p1() {
}
#endif // Trk__Segment_p1_cxx

#ifndef Trk__TrackSegment_p1_cxx
#define Trk__TrackSegment_p1_cxx
Trk::TrackSegment_p1::TrackSegment_p1() {
}
Trk::TrackSegment_p1::TrackSegment_p1(const TrackSegment_p1 & rhs)
   : m_segment(const_cast<TrackSegment_p1 &>( rhs ).m_segment)
   , m_associatedSurface(const_cast<TrackSegment_p1 &>( rhs ).m_associatedSurface)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::TrackSegment_p1::~TrackSegment_p1() {
}
#endif // Trk__TrackSegment_p1_cxx

#ifndef Trk__BoundSurface_p1_cxx
#define Trk__BoundSurface_p1_cxx
Trk::BoundSurface_p1::BoundSurface_p1() {
}
Trk::BoundSurface_p1::BoundSurface_p1(const BoundSurface_p1 & rhs)
   : Trk::Surface_p1(const_cast<BoundSurface_p1 &>( rhs ))
   , m_bounds(const_cast<BoundSurface_p1 &>( rhs ).m_bounds)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::BoundSurface_p1::~BoundSurface_p1() {
}
#endif // Trk__BoundSurface_p1_cxx

#ifndef Trk__CylinderBounds_p1_cxx
#define Trk__CylinderBounds_p1_cxx
Trk::CylinderBounds_p1::CylinderBounds_p1() {
}
Trk::CylinderBounds_p1::CylinderBounds_p1(const CylinderBounds_p1 & rhs)
   : m_radius(const_cast<CylinderBounds_p1 &>( rhs ).m_radius)
   , m_averagePhi(const_cast<CylinderBounds_p1 &>( rhs ).m_averagePhi)
   , m_halfPhiSector(const_cast<CylinderBounds_p1 &>( rhs ).m_halfPhiSector)
   , m_halfZ(const_cast<CylinderBounds_p1 &>( rhs ).m_halfZ)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::CylinderBounds_p1::~CylinderBounds_p1() {
}
#endif // Trk__CylinderBounds_p1_cxx

#ifndef Trk__DiamondBounds_p1_cxx
#define Trk__DiamondBounds_p1_cxx
Trk::DiamondBounds_p1::DiamondBounds_p1() {
}
Trk::DiamondBounds_p1::DiamondBounds_p1(const DiamondBounds_p1 & rhs)
   : m_minHalfX(const_cast<DiamondBounds_p1 &>( rhs ).m_minHalfX)
   , m_medHalfX(const_cast<DiamondBounds_p1 &>( rhs ).m_medHalfX)
   , m_maxHalfX(const_cast<DiamondBounds_p1 &>( rhs ).m_maxHalfX)
   , m_halfY1(const_cast<DiamondBounds_p1 &>( rhs ).m_halfY1)
   , m_halfY2(const_cast<DiamondBounds_p1 &>( rhs ).m_halfY2)
   , m_alpha1(const_cast<DiamondBounds_p1 &>( rhs ).m_alpha1)
   , m_alpha2(const_cast<DiamondBounds_p1 &>( rhs ).m_alpha2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::DiamondBounds_p1::~DiamondBounds_p1() {
}
#endif // Trk__DiamondBounds_p1_cxx

#ifndef Trk__DiscBounds_p1_cxx
#define Trk__DiscBounds_p1_cxx
Trk::DiscBounds_p1::DiscBounds_p1() {
}
Trk::DiscBounds_p1::DiscBounds_p1(const DiscBounds_p1 & rhs)
   : m_rMin(const_cast<DiscBounds_p1 &>( rhs ).m_rMin)
   , m_rMax(const_cast<DiscBounds_p1 &>( rhs ).m_rMax)
   , m_avePhi(const_cast<DiscBounds_p1 &>( rhs ).m_avePhi)
   , m_hPhiSec(const_cast<DiscBounds_p1 &>( rhs ).m_hPhiSec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::DiscBounds_p1::~DiscBounds_p1() {
}
#endif // Trk__DiscBounds_p1_cxx

#ifndef Trk__RectangleBounds_p1_cxx
#define Trk__RectangleBounds_p1_cxx
Trk::RectangleBounds_p1::RectangleBounds_p1() {
}
Trk::RectangleBounds_p1::RectangleBounds_p1(const RectangleBounds_p1 & rhs)
   : m_halfX(const_cast<RectangleBounds_p1 &>( rhs ).m_halfX)
   , m_halfY(const_cast<RectangleBounds_p1 &>( rhs ).m_halfY)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::RectangleBounds_p1::~RectangleBounds_p1() {
}
#endif // Trk__RectangleBounds_p1_cxx

#ifndef Trk__TrapezoidBounds_p1_cxx
#define Trk__TrapezoidBounds_p1_cxx
Trk::TrapezoidBounds_p1::TrapezoidBounds_p1() {
}
Trk::TrapezoidBounds_p1::TrapezoidBounds_p1(const TrapezoidBounds_p1 & rhs)
   : m_minHalfX(const_cast<TrapezoidBounds_p1 &>( rhs ).m_minHalfX)
   , m_maxHalfX(const_cast<TrapezoidBounds_p1 &>( rhs ).m_maxHalfX)
   , m_halfY(const_cast<TrapezoidBounds_p1 &>( rhs ).m_halfY)
   , m_alpha(const_cast<TrapezoidBounds_p1 &>( rhs ).m_alpha)
   , m_beta(const_cast<TrapezoidBounds_p1 &>( rhs ).m_beta)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::TrapezoidBounds_p1::~TrapezoidBounds_p1() {
}
#endif // Trk__TrapezoidBounds_p1_cxx

#ifndef Trk__RotatedTrapezoidBounds_p1_cxx
#define Trk__RotatedTrapezoidBounds_p1_cxx
Trk::RotatedTrapezoidBounds_p1::RotatedTrapezoidBounds_p1() {
}
Trk::RotatedTrapezoidBounds_p1::RotatedTrapezoidBounds_p1(const RotatedTrapezoidBounds_p1 & rhs)
   : m_halfX(const_cast<RotatedTrapezoidBounds_p1 &>( rhs ).m_halfX)
   , m_minHalfY(const_cast<RotatedTrapezoidBounds_p1 &>( rhs ).m_minHalfY)
   , m_maxHalfY(const_cast<RotatedTrapezoidBounds_p1 &>( rhs ).m_maxHalfY)
   , m_kappa(const_cast<RotatedTrapezoidBounds_p1 &>( rhs ).m_kappa)
   , m_delta(const_cast<RotatedTrapezoidBounds_p1 &>( rhs ).m_delta)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::RotatedTrapezoidBounds_p1::~RotatedTrapezoidBounds_p1() {
}
#endif // Trk__RotatedTrapezoidBounds_p1_cxx

#ifndef Trk__LocalParameters_p1_cxx
#define Trk__LocalParameters_p1_cxx
Trk::LocalParameters_p1::LocalParameters_p1() {
}
Trk::LocalParameters_p1::LocalParameters_p1(const LocalParameters_p1 & rhs)
   : m_parameterKey(const_cast<LocalParameters_p1 &>( rhs ).m_parameterKey)
   , m_vec(const_cast<LocalParameters_p1 &>( rhs ).m_vec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   LocalParameters_p1 &modrhs = const_cast<LocalParameters_p1 &>( rhs );
   modrhs.m_vec.clear();
}
Trk::LocalParameters_p1::~LocalParameters_p1() {
}
#endif // Trk__LocalParameters_p1_cxx

#ifndef Trk__RIO_OnTrack_p1_cxx
#define Trk__RIO_OnTrack_p1_cxx
Trk::RIO_OnTrack_p1::RIO_OnTrack_p1() {
}
Trk::RIO_OnTrack_p1::RIO_OnTrack_p1(const RIO_OnTrack_p1 & rhs)
   : m_id(const_cast<RIO_OnTrack_p1 &>( rhs ).m_id)
   , m_localParams(const_cast<RIO_OnTrack_p1 &>( rhs ).m_localParams)
   , m_localErrMat(const_cast<RIO_OnTrack_p1 &>( rhs ).m_localErrMat)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::RIO_OnTrack_p1::~RIO_OnTrack_p1() {
}
#endif // Trk__RIO_OnTrack_p1_cxx

#ifndef Trk__PseudoMeasurementOnTrack_p1_cxx
#define Trk__PseudoMeasurementOnTrack_p1_cxx
Trk::PseudoMeasurementOnTrack_p1::PseudoMeasurementOnTrack_p1() {
}
Trk::PseudoMeasurementOnTrack_p1::PseudoMeasurementOnTrack_p1(const PseudoMeasurementOnTrack_p1 & rhs)
   : m_localParams(const_cast<PseudoMeasurementOnTrack_p1 &>( rhs ).m_localParams)
   , m_localErrMat(const_cast<PseudoMeasurementOnTrack_p1 &>( rhs ).m_localErrMat)
   , m_associatedSurface(const_cast<PseudoMeasurementOnTrack_p1 &>( rhs ).m_associatedSurface)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::PseudoMeasurementOnTrack_p1::~PseudoMeasurementOnTrack_p1() {
}
#endif // Trk__PseudoMeasurementOnTrack_p1_cxx

#ifndef Trk__CompetingRIOsOnTrack_p1_cxx
#define Trk__CompetingRIOsOnTrack_p1_cxx
Trk::CompetingRIOsOnTrack_p1::CompetingRIOsOnTrack_p1() {
}
Trk::CompetingRIOsOnTrack_p1::CompetingRIOsOnTrack_p1(const CompetingRIOsOnTrack_p1 & rhs)
   : m_localParameters(const_cast<CompetingRIOsOnTrack_p1 &>( rhs ).m_localParameters)
   , m_localErrorMatrix(const_cast<CompetingRIOsOnTrack_p1 &>( rhs ).m_localErrorMatrix)
   , m_assignProb(const_cast<CompetingRIOsOnTrack_p1 &>( rhs ).m_assignProb)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CompetingRIOsOnTrack_p1 &modrhs = const_cast<CompetingRIOsOnTrack_p1 &>( rhs );
   modrhs.m_assignProb.clear();
}
Trk::CompetingRIOsOnTrack_p1::~CompetingRIOsOnTrack_p1() {
}
#endif // Trk__CompetingRIOsOnTrack_p1_cxx

#ifndef Trk__DetElementSurface_p1_cxx
#define Trk__DetElementSurface_p1_cxx
Trk::DetElementSurface_p1::DetElementSurface_p1() {
}
Trk::DetElementSurface_p1::DetElementSurface_p1(const DetElementSurface_p1 & rhs)
   : m_id(const_cast<DetElementSurface_p1 &>( rhs ).m_id)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::DetElementSurface_p1::~DetElementSurface_p1() {
}
#endif // Trk__DetElementSurface_p1_cxx

#ifndef HLT__HLTResult_p1_cxx
#define HLT__HLTResult_p1_cxx
HLT::HLTResult_p1::HLTResult_p1() {
}
HLT::HLTResult_p1::HLTResult_p1(const HLTResult_p1 & rhs)
   : m_headerResult(const_cast<HLTResult_p1 &>( rhs ).m_headerResult)
   , m_chainsResult(const_cast<HLTResult_p1 &>( rhs ).m_chainsResult)
   , m_navigationResult(const_cast<HLTResult_p1 &>( rhs ).m_navigationResult)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   HLTResult_p1 &modrhs = const_cast<HLTResult_p1 &>( rhs );
   modrhs.m_headerResult.clear();
   modrhs.m_chainsResult.clear();
   modrhs.m_navigationResult.clear();
}
HLT::HLTResult_p1::~HLTResult_p1() {
}
#endif // HLT__HLTResult_p1_cxx

#ifndef Trk__TrackCollection_tlp3_cxx
#define Trk__TrackCollection_tlp3_cxx
Trk::TrackCollection_tlp3::TrackCollection_tlp3() {
}
Trk::TrackCollection_tlp3::TrackCollection_tlp3(const TrackCollection_tlp3 & rhs)
   : m_tokenList(const_cast<TrackCollection_tlp3 &>( rhs ).m_tokenList)
   , m_trackCollections(const_cast<TrackCollection_tlp3 &>( rhs ).m_trackCollections)
   , m_tracks(const_cast<TrackCollection_tlp3 &>( rhs ).m_tracks)
   , m_trackStates(const_cast<TrackCollection_tlp3 &>( rhs ).m_trackStates)
   , m_competingRotsOnTrack(const_cast<TrackCollection_tlp3 &>( rhs ).m_competingRotsOnTrack)
   , m_RIOs(const_cast<TrackCollection_tlp3 &>( rhs ).m_RIOs)
   , m_pseudoMeasurementOnTrack(const_cast<TrackCollection_tlp3 &>( rhs ).m_pseudoMeasurementOnTrack)
   , m_parameters(const_cast<TrackCollection_tlp3 &>( rhs ).m_parameters)
   , m_ataSurfaces(const_cast<TrackCollection_tlp3 &>( rhs ).m_ataSurfaces)
   , m_measuredAtaSurfaces(const_cast<TrackCollection_tlp3 &>( rhs ).m_measuredAtaSurfaces)
   , m_perigees(const_cast<TrackCollection_tlp3 &>( rhs ).m_perigees)
   , m_measuredPerigees(const_cast<TrackCollection_tlp3 &>( rhs ).m_measuredPerigees)
   , m_boundSurfaces(const_cast<TrackCollection_tlp3 &>( rhs ).m_boundSurfaces)
   , m_surfaces(const_cast<TrackCollection_tlp3 &>( rhs ).m_surfaces)
   , m_cylinderBounds(const_cast<TrackCollection_tlp3 &>( rhs ).m_cylinderBounds)
   , m_diamondBounds(const_cast<TrackCollection_tlp3 &>( rhs ).m_diamondBounds)
   , m_discBounds(const_cast<TrackCollection_tlp3 &>( rhs ).m_discBounds)
   , m_rectangleBounds(const_cast<TrackCollection_tlp3 &>( rhs ).m_rectangleBounds)
   , m_trapesoidBounds(const_cast<TrackCollection_tlp3 &>( rhs ).m_trapesoidBounds)
   , m_rotatedTrapesoidBounds(const_cast<TrackCollection_tlp3 &>( rhs ).m_rotatedTrapesoidBounds)
   , m_detElementSurfaces(const_cast<TrackCollection_tlp3 &>( rhs ).m_detElementSurfaces)
   , m_fitQualities(const_cast<TrackCollection_tlp3 &>( rhs ).m_fitQualities)
   , m_hepSymMatrices(const_cast<TrackCollection_tlp3 &>( rhs ).m_hepSymMatrices)
   , m_matEffectsBases(const_cast<TrackCollection_tlp3 &>( rhs ).m_matEffectsBases)
   , m_energyLosses(const_cast<TrackCollection_tlp3 &>( rhs ).m_energyLosses)
   , m_materialEffects(const_cast<TrackCollection_tlp3 &>( rhs ).m_materialEffects)
   , m_estimatedBrems(const_cast<TrackCollection_tlp3 &>( rhs ).m_estimatedBrems)
   , m_localDirections(const_cast<TrackCollection_tlp3 &>( rhs ).m_localDirections)
   , m_localPositions(const_cast<TrackCollection_tlp3 &>( rhs ).m_localPositions)
   , m_localParameters(const_cast<TrackCollection_tlp3 &>( rhs ).m_localParameters)
   , m_trackInfos(const_cast<TrackCollection_tlp3 &>( rhs ).m_trackInfos)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrackCollection_tlp3 &modrhs = const_cast<TrackCollection_tlp3 &>( rhs );
   modrhs.m_trackCollections.clear();
   modrhs.m_tracks.clear();
   modrhs.m_trackStates.clear();
   modrhs.m_competingRotsOnTrack.clear();
   modrhs.m_RIOs.clear();
   modrhs.m_pseudoMeasurementOnTrack.clear();
   modrhs.m_parameters.clear();
   modrhs.m_ataSurfaces.clear();
   modrhs.m_measuredAtaSurfaces.clear();
   modrhs.m_perigees.clear();
   modrhs.m_measuredPerigees.clear();
   modrhs.m_boundSurfaces.clear();
   modrhs.m_surfaces.clear();
   modrhs.m_cylinderBounds.clear();
   modrhs.m_diamondBounds.clear();
   modrhs.m_discBounds.clear();
   modrhs.m_rectangleBounds.clear();
   modrhs.m_trapesoidBounds.clear();
   modrhs.m_rotatedTrapesoidBounds.clear();
   modrhs.m_detElementSurfaces.clear();
   modrhs.m_fitQualities.clear();
   modrhs.m_hepSymMatrices.clear();
   modrhs.m_matEffectsBases.clear();
   modrhs.m_energyLosses.clear();
   modrhs.m_materialEffects.clear();
   modrhs.m_estimatedBrems.clear();
   modrhs.m_localDirections.clear();
   modrhs.m_localPositions.clear();
   modrhs.m_localParameters.clear();
   modrhs.m_trackInfos.clear();
}
Trk::TrackCollection_tlp3::~TrackCollection_tlp3() {
}
#endif // Trk__TrackCollection_tlp3_cxx

#ifndef Trk__Track_p2_cxx
#define Trk__Track_p2_cxx
Trk::Track_p2::Track_p2() {
}
Trk::Track_p2::Track_p2(const Track_p2 & rhs)
   : m_trackInfo(const_cast<Track_p2 &>( rhs ).m_trackInfo)
   , m_fitQuality(const_cast<Track_p2 &>( rhs ).m_fitQuality)
   , m_trackState(const_cast<Track_p2 &>( rhs ).m_trackState)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Track_p2 &modrhs = const_cast<Track_p2 &>( rhs );
   modrhs.m_trackState.clear();
}
Trk::Track_p2::~Track_p2() {
}
#endif // Trk__Track_p2_cxx

#ifndef Trk__TrackStateOnSurface_p2_cxx
#define Trk__TrackStateOnSurface_p2_cxx
Trk::TrackStateOnSurface_p2::TrackStateOnSurface_p2() {
}
Trk::TrackStateOnSurface_p2::TrackStateOnSurface_p2(const TrackStateOnSurface_p2 & rhs)
   : m_trackParameters(const_cast<TrackStateOnSurface_p2 &>( rhs ).m_trackParameters)
   , m_fitQualityOnSurface(const_cast<TrackStateOnSurface_p2 &>( rhs ).m_fitQualityOnSurface)
   , m_materialEffects(const_cast<TrackStateOnSurface_p2 &>( rhs ).m_materialEffects)
   , m_measurementOnTrack(const_cast<TrackStateOnSurface_p2 &>( rhs ).m_measurementOnTrack)
   , m_typeFlags(const_cast<TrackStateOnSurface_p2 &>( rhs ).m_typeFlags)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::TrackStateOnSurface_p2::~TrackStateOnSurface_p2() {
}
#endif // Trk__TrackStateOnSurface_p2_cxx

#ifndef Trk__AtaSurface_p1_cxx
#define Trk__AtaSurface_p1_cxx
Trk::AtaSurface_p1::AtaSurface_p1() {
}
Trk::AtaSurface_p1::AtaSurface_p1(const AtaSurface_p1 & rhs)
   : m_parameters(const_cast<AtaSurface_p1 &>( rhs ).m_parameters)
   , m_assocSurface(const_cast<AtaSurface_p1 &>( rhs ).m_assocSurface)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::AtaSurface_p1::~AtaSurface_p1() {
}
#endif // Trk__AtaSurface_p1_cxx

#ifndef Trk__MeasuredAtaSurface_p1_cxx
#define Trk__MeasuredAtaSurface_p1_cxx
Trk::MeasuredAtaSurface_p1::MeasuredAtaSurface_p1() {
}
Trk::MeasuredAtaSurface_p1::MeasuredAtaSurface_p1(const MeasuredAtaSurface_p1 & rhs)
   : Trk::AtaSurface_p1(const_cast<MeasuredAtaSurface_p1 &>( rhs ))
   , m_errorMatrix(const_cast<MeasuredAtaSurface_p1 &>( rhs ).m_errorMatrix)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::MeasuredAtaSurface_p1::~MeasuredAtaSurface_p1() {
}
#endif // Trk__MeasuredAtaSurface_p1_cxx

#ifndef Trk__MaterialEffectsBase_p1_cxx
#define Trk__MaterialEffectsBase_p1_cxx
Trk::MaterialEffectsBase_p1::MaterialEffectsBase_p1() {
}
Trk::MaterialEffectsBase_p1::MaterialEffectsBase_p1(const MaterialEffectsBase_p1 & rhs)
   : m_tInX0(const_cast<MaterialEffectsBase_p1 &>( rhs ).m_tInX0)
   , m_typeFlags(const_cast<MaterialEffectsBase_p1 &>( rhs ).m_typeFlags)
   , m_associatedSurface(const_cast<MaterialEffectsBase_p1 &>( rhs ).m_associatedSurface)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::MaterialEffectsBase_p1::~MaterialEffectsBase_p1() {
}
#endif // Trk__MaterialEffectsBase_p1_cxx

#ifndef Trk__EnergyLoss_p1_cxx
#define Trk__EnergyLoss_p1_cxx
Trk::EnergyLoss_p1::EnergyLoss_p1() {
}
Trk::EnergyLoss_p1::EnergyLoss_p1(const EnergyLoss_p1 & rhs)
   : m_deltaE(const_cast<EnergyLoss_p1 &>( rhs ).m_deltaE)
   , m_sigmaDeltaE(const_cast<EnergyLoss_p1 &>( rhs ).m_sigmaDeltaE)
   , m_sigmaMinusDeltaE(const_cast<EnergyLoss_p1 &>( rhs ).m_sigmaMinusDeltaE)
   , m_sigmaPlusDeltaE(const_cast<EnergyLoss_p1 &>( rhs ).m_sigmaPlusDeltaE)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::EnergyLoss_p1::~EnergyLoss_p1() {
}
#endif // Trk__EnergyLoss_p1_cxx

#ifndef Trk__MaterialEffectsOnTrack_p2_cxx
#define Trk__MaterialEffectsOnTrack_p2_cxx
Trk::MaterialEffectsOnTrack_p2::MaterialEffectsOnTrack_p2() {
}
Trk::MaterialEffectsOnTrack_p2::MaterialEffectsOnTrack_p2(const MaterialEffectsOnTrack_p2 & rhs)
   : m_mefBase(const_cast<MaterialEffectsOnTrack_p2 &>( rhs ).m_mefBase)
   , m_deltaPhi(const_cast<MaterialEffectsOnTrack_p2 &>( rhs ).m_deltaPhi)
   , m_deltaTheta(const_cast<MaterialEffectsOnTrack_p2 &>( rhs ).m_deltaTheta)
   , m_sigmaDeltaPhi(const_cast<MaterialEffectsOnTrack_p2 &>( rhs ).m_sigmaDeltaPhi)
   , m_sigmaDeltaTheta(const_cast<MaterialEffectsOnTrack_p2 &>( rhs ).m_sigmaDeltaTheta)
   , m_energyLoss(const_cast<MaterialEffectsOnTrack_p2 &>( rhs ).m_energyLoss)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::MaterialEffectsOnTrack_p2::~MaterialEffectsOnTrack_p2() {
}
#endif // Trk__MaterialEffectsOnTrack_p2_cxx

#ifndef Trk__EstimatedBremOnTrack_p1_cxx
#define Trk__EstimatedBremOnTrack_p1_cxx
Trk::EstimatedBremOnTrack_p1::EstimatedBremOnTrack_p1() {
}
Trk::EstimatedBremOnTrack_p1::EstimatedBremOnTrack_p1(const EstimatedBremOnTrack_p1 & rhs)
   : m_mefBase(const_cast<EstimatedBremOnTrack_p1 &>( rhs ).m_mefBase)
   , m_retainedEnFraction(const_cast<EstimatedBremOnTrack_p1 &>( rhs ).m_retainedEnFraction)
   , m_sigmaRetEnFraction(const_cast<EstimatedBremOnTrack_p1 &>( rhs ).m_sigmaRetEnFraction)
   , m_sigmaQoverPsquared(const_cast<EstimatedBremOnTrack_p1 &>( rhs ).m_sigmaQoverPsquared)
   , m_direction(const_cast<EstimatedBremOnTrack_p1 &>( rhs ).m_direction)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::EstimatedBremOnTrack_p1::~EstimatedBremOnTrack_p1() {
}
#endif // Trk__EstimatedBremOnTrack_p1_cxx

#ifndef Trk__LocalDirection_p1_cxx
#define Trk__LocalDirection_p1_cxx
Trk::LocalDirection_p1::LocalDirection_p1() {
}
Trk::LocalDirection_p1::LocalDirection_p1(const LocalDirection_p1 & rhs)
   : m_angleXZ(const_cast<LocalDirection_p1 &>( rhs ).m_angleXZ)
   , m_angleYZ(const_cast<LocalDirection_p1 &>( rhs ).m_angleYZ)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::LocalDirection_p1::~LocalDirection_p1() {
}
#endif // Trk__LocalDirection_p1_cxx

#ifndef Trk__TrackInfo_p1_cxx
#define Trk__TrackInfo_p1_cxx
Trk::TrackInfo_p1::TrackInfo_p1() {
}
Trk::TrackInfo_p1::TrackInfo_p1(const TrackInfo_p1 & rhs)
   : m_fitter(const_cast<TrackInfo_p1 &>( rhs ).m_fitter)
   , m_particleHypo(const_cast<TrackInfo_p1 &>( rhs ).m_particleHypo)
   , m_properties(const_cast<TrackInfo_p1 &>( rhs ).m_properties)
   , m_patternRecognition(const_cast<TrackInfo_p1 &>( rhs ).m_patternRecognition)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::TrackInfo_p1::~TrackInfo_p1() {
}
#endif // Trk__TrackInfo_p1_cxx

#ifndef CTP_Decision_p2_cxx
#define CTP_Decision_p2_cxx
CTP_Decision_p2::CTP_Decision_p2() {
}
CTP_Decision_p2::CTP_Decision_p2(const CTP_Decision_p2 & rhs)
   : m_CTPResultWord(const_cast<CTP_Decision_p2 &>( rhs ).m_CTPResultWord)
   , m_triggerTypeWord(const_cast<CTP_Decision_p2 &>( rhs ).m_triggerTypeWord)
   , m_items(const_cast<CTP_Decision_p2 &>( rhs ).m_items)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CTP_Decision_p2 &modrhs = const_cast<CTP_Decision_p2 &>( rhs );
   modrhs.m_CTPResultWord.clear();
   modrhs.m_items.clear();
}
CTP_Decision_p2::~CTP_Decision_p2() {
}
#endif // CTP_Decision_p2_cxx

#ifndef TrigElectronContainer_tlp2_cxx
#define TrigElectronContainer_tlp2_cxx
TrigElectronContainer_tlp2::TrigElectronContainer_tlp2() {
}
TrigElectronContainer_tlp2::TrigElectronContainer_tlp2(const TrigElectronContainer_tlp2 & rhs)
   : m_trigElectronContainerVec(const_cast<TrigElectronContainer_tlp2 &>( rhs ).m_trigElectronContainerVec)
   , m_trigElectronVec(const_cast<TrigElectronContainer_tlp2 &>( rhs ).m_trigElectronVec)
   , m_p4PtEtaPhiMVec(const_cast<TrigElectronContainer_tlp2 &>( rhs ).m_p4PtEtaPhiMVec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigElectronContainer_tlp2 &modrhs = const_cast<TrigElectronContainer_tlp2 &>( rhs );
   modrhs.m_trigElectronContainerVec.clear();
   modrhs.m_trigElectronVec.clear();
   modrhs.m_p4PtEtaPhiMVec.clear();
}
TrigElectronContainer_tlp2::~TrigElectronContainer_tlp2() {
}
#endif // TrigElectronContainer_tlp2_cxx

#ifndef TrigElectron_p2_cxx
#define TrigElectron_p2_cxx
TrigElectron_p2::TrigElectron_p2() {
}
TrigElectron_p2::TrigElectron_p2(const TrigElectron_p2 & rhs)
   : m_roiWord(const_cast<TrigElectron_p2 &>( rhs ).m_roiWord)
   , m_valid(const_cast<TrigElectron_p2 &>( rhs ).m_valid)
   , m_tr_Algo(const_cast<TrigElectron_p2 &>( rhs ).m_tr_Algo)
   , m_tr_Zvtx(const_cast<TrigElectron_p2 &>( rhs ).m_tr_Zvtx)
   , m_tr_eta_at_calo(const_cast<TrigElectron_p2 &>( rhs ).m_tr_eta_at_calo)
   , m_tr_phi_at_calo(const_cast<TrigElectron_p2 &>( rhs ).m_tr_phi_at_calo)
   , m_etoverpt(const_cast<TrigElectron_p2 &>( rhs ).m_etoverpt)
   , m_cl_eta(const_cast<TrigElectron_p2 &>( rhs ).m_cl_eta)
   , m_cl_phi(const_cast<TrigElectron_p2 &>( rhs ).m_cl_phi)
   , m_cl_Rcore(const_cast<TrigElectron_p2 &>( rhs ).m_cl_Rcore)
   , m_cl_Eratio(const_cast<TrigElectron_p2 &>( rhs ).m_cl_Eratio)
   , m_cl_EThad(const_cast<TrigElectron_p2 &>( rhs ).m_cl_EThad)
   , m_cluster(const_cast<TrigElectron_p2 &>( rhs ).m_cluster)
   , m_track(const_cast<TrigElectron_p2 &>( rhs ).m_track)
   , m_p4PtEtaPhiM(const_cast<TrigElectron_p2 &>( rhs ).m_p4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigElectron_p2::~TrigElectron_p2() {
}
#endif // TrigElectron_p2_cxx

#ifndef P4PtEtaPhiM_p1_cxx
#define P4PtEtaPhiM_p1_cxx
P4PtEtaPhiM_p1::P4PtEtaPhiM_p1() {
}
P4PtEtaPhiM_p1::P4PtEtaPhiM_p1(const P4PtEtaPhiM_p1 & rhs)
   : m_pt(const_cast<P4PtEtaPhiM_p1 &>( rhs ).m_pt)
   , m_eta(const_cast<P4PtEtaPhiM_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<P4PtEtaPhiM_p1 &>( rhs ).m_phi)
   , m_mass(const_cast<P4PtEtaPhiM_p1 &>( rhs ).m_mass)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
P4PtEtaPhiM_p1::~P4PtEtaPhiM_p1() {
}
#endif // P4PtEtaPhiM_p1_cxx

#ifndef TauDetailsContainer_tlp1_cxx
#define TauDetailsContainer_tlp1_cxx
TauDetailsContainer_tlp1::TauDetailsContainer_tlp1() {
}
TauDetailsContainer_tlp1::TauDetailsContainer_tlp1(const TauDetailsContainer_tlp1 & rhs)
   : m_tauDetailsContainers(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_tauDetailsContainers)
   , m_tauCommonDetails(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_tauCommonDetails)
   , m_tauCommonExtraDetails(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_tauCommonExtraDetails)
   , m_tau1P3PDetails(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_tau1P3PDetails)
   , m_tau1P3PExtraDetails(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_tau1P3PExtraDetails)
   , m_tauRecDetails(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_tauRecDetails)
   , m_tauRecExtraDetails(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_tauRecExtraDetails)
   , m_tauAnalysisHelperObjects(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_tauAnalysisHelperObjects)
   , m_recVertices(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_recVertices)
   , m_vertices(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_vertices)
   , m_fitQualities(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_fitQualities)
   , m_errorMatrices(const_cast<TauDetailsContainer_tlp1 &>( rhs ).m_errorMatrices)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TauDetailsContainer_tlp1 &modrhs = const_cast<TauDetailsContainer_tlp1 &>( rhs );
   modrhs.m_tauDetailsContainers.clear();
   modrhs.m_tauCommonDetails.clear();
   modrhs.m_tauCommonExtraDetails.clear();
   modrhs.m_tau1P3PDetails.clear();
   modrhs.m_tau1P3PExtraDetails.clear();
   modrhs.m_tauRecDetails.clear();
   modrhs.m_tauRecExtraDetails.clear();
   modrhs.m_tauAnalysisHelperObjects.clear();
   modrhs.m_recVertices.clear();
   modrhs.m_vertices.clear();
   modrhs.m_fitQualities.clear();
   modrhs.m_errorMatrices.clear();
}
TauDetailsContainer_tlp1::~TauDetailsContainer_tlp1() {
}
#endif // TauDetailsContainer_tlp1_cxx

#ifndef TauCommonDetails_p1_cxx
#define TauCommonDetails_p1_cxx
TauCommonDetails_p1::TauCommonDetails_p1() {
}
TauCommonDetails_p1::TauCommonDetails_p1(const TauCommonDetails_p1 & rhs)
   : m_ipZ0SinThetaSigLeadTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_ipZ0SinThetaSigLeadTrk)
   , m_etOverPtLeadTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_etOverPtLeadTrk)
   , m_etOverPtLeadLooseTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_etOverPtLeadLooseTrk)
   , m_leadTrkPt(const_cast<TauCommonDetails_p1 &>( rhs ).m_leadTrkPt)
   , m_leadLooseTrkPt(const_cast<TauCommonDetails_p1 &>( rhs ).m_leadLooseTrkPt)
   , m_ipSigLeadTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_ipSigLeadTrk)
   , m_ipSigLeadLooseTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_ipSigLeadLooseTrk)
   , m_looseTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_looseTrk)
   , m_looseConvTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_looseConvTrk)
   , m_chrgLooseTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_chrgLooseTrk)
   , m_cellEM012Cluster(const_cast<TauCommonDetails_p1 &>( rhs ).m_cellEM012Cluster)
   , m_sumPi0Vec(const_cast<TauCommonDetails_p1 &>( rhs ).m_sumPi0Vec)
   , m_massTrkSys(const_cast<TauCommonDetails_p1 &>( rhs ).m_massTrkSys)
   , m_trkWidth2(const_cast<TauCommonDetails_p1 &>( rhs ).m_trkWidth2)
   , m_trFlightPathSig(const_cast<TauCommonDetails_p1 &>( rhs ).m_trFlightPathSig)
   , m_secVtx(const_cast<TauCommonDetails_p1 &>( rhs ).m_secVtx)
   , m_etEflow(const_cast<TauCommonDetails_p1 &>( rhs ).m_etEflow)
   , m_pi0(const_cast<TauCommonDetails_p1 &>( rhs ).m_pi0)
   , m_seedCalo_nIsolLooseTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_nIsolLooseTrk)
   , m_seedCalo_EMRadius(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_EMRadius)
   , m_seedCalo_hadRadius(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_hadRadius)
   , m_seedCalo_etEMAtEMScale(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_etEMAtEMScale)
   , m_seedCalo_etHadAtEMScale(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_etHadAtEMScale)
   , m_seedCalo_isolFrac(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_isolFrac)
   , m_seedCalo_centFrac(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_centFrac)
   , m_seedCalo_stripWidth2(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_stripWidth2)
   , m_seedCalo_nStrip(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_nStrip)
   , m_seedCalo_etEMCalib(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_etEMCalib)
   , m_seedCalo_etHadCalib(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_etHadCalib)
   , m_seedCalo_eta(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_eta)
   , m_seedCalo_phi(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_phi)
   , m_seedCalo_trkAvgDist(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_trkAvgDist)
   , m_seedCalo_trkRmsDist(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedCalo_trkRmsDist)
   , m_seedTrk_EMRadius(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_EMRadius)
   , m_seedTrk_isolFrac(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_isolFrac)
   , m_seedTrk_etChrgHadOverSumTrkPt(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etChrgHadOverSumTrkPt)
   , m_seedTrk_isolFracWide(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_isolFracWide)
   , m_seedTrk_etHadAtEMScale(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etHadAtEMScale)
   , m_seedTrk_etEMAtEMScale(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etEMAtEMScale)
   , m_seedTrk_etEMCL(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etEMCL)
   , m_seedTrk_etChrgEM(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etChrgEM)
   , m_seedTrk_etNeuEM(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etNeuEM)
   , m_seedTrk_etResNeuEM(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etResNeuEM)
   , m_seedTrk_hadLeakEt(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_hadLeakEt)
   , m_seedTrk_etChrgEM01Trk(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etChrgEM01Trk)
   , m_seedTrk_etResChrgEMTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etResChrgEMTrk)
   , m_seedTrk_sumEMCellEtOverLeadTrkPt(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_sumEMCellEtOverLeadTrkPt)
   , m_seedTrk_secMaxStripEt(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_secMaxStripEt)
   , m_seedTrk_stripWidth2(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_stripWidth2)
   , m_seedTrk_nStrip(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_nStrip)
   , m_seedTrk_etChrgHad(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etChrgHad)
   , m_seedTrk_nOtherCoreTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_nOtherCoreTrk)
   , m_seedTrk_nIsolTrk(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_nIsolTrk)
   , m_seedTrk_etIsolEM(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etIsolEM)
   , m_seedTrk_etIsolHad(const_cast<TauCommonDetails_p1 &>( rhs ).m_seedTrk_etIsolHad)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TauCommonDetails_p1 &modrhs = const_cast<TauCommonDetails_p1 &>( rhs );
   modrhs.m_seedTrk_etChrgEM01Trk.clear();
   modrhs.m_seedTrk_etResChrgEMTrk.clear();
}
TauCommonDetails_p1::~TauCommonDetails_p1() {
}
#endif // TauCommonDetails_p1_cxx

#ifndef HepLorentzVector_p1_cxx
#define HepLorentzVector_p1_cxx
HepLorentzVector_p1::HepLorentzVector_p1() {
}
HepLorentzVector_p1::HepLorentzVector_p1(const HepLorentzVector_p1 & rhs)
   : m_px(const_cast<HepLorentzVector_p1 &>( rhs ).m_px)
   , m_py(const_cast<HepLorentzVector_p1 &>( rhs ).m_py)
   , m_pz(const_cast<HepLorentzVector_p1 &>( rhs ).m_pz)
   , m_ene(const_cast<HepLorentzVector_p1 &>( rhs ).m_ene)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
HepLorentzVector_p1::~HepLorentzVector_p1() {
}
#endif // HepLorentzVector_p1_cxx

#ifndef TauCommonExtraDetails_p1_cxx
#define TauCommonExtraDetails_p1_cxx
TauCommonExtraDetails_p1::TauCommonExtraDetails_p1() {
}
TauCommonExtraDetails_p1::TauCommonExtraDetails_p1(const TauCommonExtraDetails_p1 & rhs)
   : m_sumPtLooseTrk(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_sumPtLooseTrk)
   , m_sumPtTrk(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_sumPtTrk)
   , m_closestEtaTrkVertCell(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_closestEtaTrkVertCell)
   , m_closestPhiTrkVertCell(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_closestPhiTrkVertCell)
   , m_closestEtaTrkCell(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_closestEtaTrkCell)
   , m_closestPhiTrkCell(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_closestPhiTrkCell)
   , m_etaTrkCaloSamp(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_etaTrkCaloSamp)
   , m_phiTrkCaloSamp(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_phiTrkCaloSamp)
   , m_etaLooseTrkCaloSamp(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_etaLooseTrkCaloSamp)
   , m_phiLooseTrkCaloSamp(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_phiLooseTrkCaloSamp)
   , m_seedCalo_nEMCell(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_seedCalo_nEMCell)
   , m_seedCalo_stripEt(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_seedCalo_stripEt)
   , m_seedCalo_EMCentFrac(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_seedCalo_EMCentFrac)
   , m_seedCalo_sumCellEnergy(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_seedCalo_sumCellEnergy)
   , m_seedCalo_sumEMCellEnergy(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_seedCalo_sumEMCellEnergy)
   , m_linkNames(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_linkNames)
   , m_tracks(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_tracks)
   , m_looseTracks(const_cast<TauCommonExtraDetails_p1 &>( rhs ).m_looseTracks)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TauCommonExtraDetails_p1 &modrhs = const_cast<TauCommonExtraDetails_p1 &>( rhs );
   modrhs.m_closestEtaTrkVertCell.clear();
   modrhs.m_closestPhiTrkVertCell.clear();
   modrhs.m_closestEtaTrkCell.clear();
   modrhs.m_closestPhiTrkCell.clear();
   modrhs.m_etaTrkCaloSamp.clear();
   modrhs.m_phiTrkCaloSamp.clear();
   modrhs.m_etaLooseTrkCaloSamp.clear();
   modrhs.m_phiLooseTrkCaloSamp.clear();
}
TauCommonExtraDetails_p1::~TauCommonExtraDetails_p1() {
}
#endif // TauCommonExtraDetails_p1_cxx

#ifndef Tau1P3PDetails_p1_cxx
#define Tau1P3PDetails_p1_cxx
Tau1P3PDetails_p1::Tau1P3PDetails_p1() {
}
Tau1P3PDetails_p1::Tau1P3PDetails_p1(const Tau1P3PDetails_p1 & rhs)
   : m_numStripCells(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_numStripCells)
   , m_stripWidth2(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_stripWidth2)
   , m_emRadius(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_emRadius)
   , m_ET12Frac(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_ET12Frac)
   , m_etIsolHAD(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etIsolHAD)
   , m_etIsolEM(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etIsolEM)
   , m_etChrgHAD(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etChrgHAD)
   , m_nAssocTracksCore(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_nAssocTracksCore)
   , m_nAssocTracksIsol(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_nAssocTracksIsol)
   , m_signD0Trk3P(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_signD0Trk3P)
   , m_massTrk3P(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_massTrk3P)
   , m_rWidth2Trk3P(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_rWidth2Trk3P)
   , m_etHadAtEMScale(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etHadAtEMScale)
   , m_etEMAtEMScale(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etEMAtEMScale)
   , m_etEMCL(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etEMCL)
   , m_etChrgEM(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etChrgEM)
   , m_etNeuEM(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etNeuEM)
   , m_etResNeuEM(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etResNeuEM)
   , m_trFlightPathSig(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_trFlightPathSig)
   , m_z0SinThetaSig(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_z0SinThetaSig)
   , m_etChrgHADoverPttot(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etChrgHADoverPttot)
   , m_etIsolFrac(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etIsolFrac)
   , m_sumEtCellsLArOverLeadTrackPt(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_sumEtCellsLArOverLeadTrackPt)
   , m_hadronicLeak(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_hadronicLeak)
   , m_secondaryMax(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_secondaryMax)
   , m_etChrgEM01Trk(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etChrgEM01Trk)
   , m_etResChrgEMTrk(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etResChrgEMTrk)
   , m_sumEM(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_sumEM)
   , m_secVertex(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_secVertex)
   , m_pi0(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_pi0)
   , m_cellEM012Cluster(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_cellEM012Cluster)
   , m_etEflow(const_cast<Tau1P3PDetails_p1 &>( rhs ).m_etEflow)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Tau1P3PDetails_p1 &modrhs = const_cast<Tau1P3PDetails_p1 &>( rhs );
   modrhs.m_etChrgEM01Trk.clear();
   modrhs.m_etResChrgEMTrk.clear();
}
Tau1P3PDetails_p1::~Tau1P3PDetails_p1() {
}
#endif // Tau1P3PDetails_p1_cxx

#ifndef Tau1P3PExtraDetails_p1_cxx
#define Tau1P3PExtraDetails_p1_cxx
Tau1P3PExtraDetails_p1::Tau1P3PExtraDetails_p1() {
}
Tau1P3PExtraDetails_p1::Tau1P3PExtraDetails_p1(const Tau1P3PExtraDetails_p1 & rhs)
   : m_closestEtaTrkVertCell(const_cast<Tau1P3PExtraDetails_p1 &>( rhs ).m_closestEtaTrkVertCell)
   , m_closestEtaTrkCell(const_cast<Tau1P3PExtraDetails_p1 &>( rhs ).m_closestEtaTrkCell)
   , m_closestPhiTrkVertCell(const_cast<Tau1P3PExtraDetails_p1 &>( rhs ).m_closestPhiTrkVertCell)
   , m_closestPhiTrkCell(const_cast<Tau1P3PExtraDetails_p1 &>( rhs ).m_closestPhiTrkCell)
   , m_etaTrackCaloSamp(const_cast<Tau1P3PExtraDetails_p1 &>( rhs ).m_etaTrackCaloSamp)
   , m_phiTrackCaloSamp(const_cast<Tau1P3PExtraDetails_p1 &>( rhs ).m_phiTrackCaloSamp)
   , m_sumPTTracks(const_cast<Tau1P3PExtraDetails_p1 &>( rhs ).m_sumPTTracks)
   , m_tracks(const_cast<Tau1P3PExtraDetails_p1 &>( rhs ).m_tracks)
   , m_linkNames(const_cast<Tau1P3PExtraDetails_p1 &>( rhs ).m_linkNames)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Tau1P3PExtraDetails_p1 &modrhs = const_cast<Tau1P3PExtraDetails_p1 &>( rhs );
   modrhs.m_closestEtaTrkVertCell.clear();
   modrhs.m_closestEtaTrkCell.clear();
   modrhs.m_closestPhiTrkVertCell.clear();
   modrhs.m_closestPhiTrkCell.clear();
   modrhs.m_etaTrackCaloSamp.clear();
   modrhs.m_phiTrackCaloSamp.clear();
}
Tau1P3PExtraDetails_p1::~Tau1P3PExtraDetails_p1() {
}
#endif // Tau1P3PExtraDetails_p1_cxx

#ifndef TauRecDetails_p1_cxx
#define TauRecDetails_p1_cxx
TauRecDetails_p1::TauRecDetails_p1() {
}
TauRecDetails_p1::TauRecDetails_p1(const TauRecDetails_p1 & rhs)
   : m_looseTracks(const_cast<TauRecDetails_p1 &>( rhs ).m_looseTracks)
   , m_emRadius(const_cast<TauRecDetails_p1 &>( rhs ).m_emRadius)
   , m_hadRadius(const_cast<TauRecDetails_p1 &>( rhs ).m_hadRadius)
   , m_sumEmCellEt(const_cast<TauRecDetails_p1 &>( rhs ).m_sumEmCellEt)
   , m_sumHadCellEt(const_cast<TauRecDetails_p1 &>( rhs ).m_sumHadCellEt)
   , m_ET12Frac(const_cast<TauRecDetails_p1 &>( rhs ).m_ET12Frac)
   , m_centralityFraction(const_cast<TauRecDetails_p1 &>( rhs ).m_centralityFraction)
   , m_stripWidth2(const_cast<TauRecDetails_p1 &>( rhs ).m_stripWidth2)
   , m_numStripCells(const_cast<TauRecDetails_p1 &>( rhs ).m_numStripCells)
   , m_sumEM(const_cast<TauRecDetails_p1 &>( rhs ).m_sumEM)
   , m_etEMCalib(const_cast<TauRecDetails_p1 &>( rhs ).m_etEMCalib)
   , m_etHadCalib(const_cast<TauRecDetails_p1 &>( rhs ).m_etHadCalib)
   , m_secVertex(const_cast<TauRecDetails_p1 &>( rhs ).m_secVertex)
   , m_trackCaloEta(const_cast<TauRecDetails_p1 &>( rhs ).m_trackCaloEta)
   , m_trackCaloPhi(const_cast<TauRecDetails_p1 &>( rhs ).m_trackCaloPhi)
   , m_leadingTrackPT(const_cast<TauRecDetails_p1 &>( rhs ).m_leadingTrackPT)
   , m_trFlightPathSig(const_cast<TauRecDetails_p1 &>( rhs ).m_trFlightPathSig)
   , m_etaCalo(const_cast<TauRecDetails_p1 &>( rhs ).m_etaCalo)
   , m_phiCalo(const_cast<TauRecDetails_p1 &>( rhs ).m_phiCalo)
   , m_ipSigLeadTrack(const_cast<TauRecDetails_p1 &>( rhs ).m_ipSigLeadTrack)
   , m_etOverPtLeadTrack(const_cast<TauRecDetails_p1 &>( rhs ).m_etOverPtLeadTrack)
   , m_nTracksdrdR(const_cast<TauRecDetails_p1 &>( rhs ).m_nTracksdrdR)
   , m_chargeLooseTracks(const_cast<TauRecDetails_p1 &>( rhs ).m_chargeLooseTracks)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TauRecDetails_p1 &modrhs = const_cast<TauRecDetails_p1 &>( rhs );
   modrhs.m_trackCaloEta.clear();
   modrhs.m_trackCaloPhi.clear();
}
TauRecDetails_p1::~TauRecDetails_p1() {
}
#endif // TauRecDetails_p1_cxx

#ifndef TauRecExtraDetails_p1_cxx
#define TauRecExtraDetails_p1_cxx
TauRecExtraDetails_p1::TauRecExtraDetails_p1() {
}
TauRecExtraDetails_p1::TauRecExtraDetails_p1(const TauRecExtraDetails_p1 & rhs)
   : m_analysisHelper(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_analysisHelper)
   , m_seedType(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_seedType)
   , m_numEMCells(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_numEMCells)
   , m_stripET(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_stripET)
   , m_emCentralityFraction(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_emCentralityFraction)
   , m_etHadAtEMScale(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_etHadAtEMScale)
   , m_etEMAtEMScale(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_etEMAtEMScale)
   , m_sumCellE(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_sumCellE)
   , m_sumEMCellE(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_sumEMCellE)
   , m_sumPTTracks(const_cast<TauRecExtraDetails_p1 &>( rhs ).m_sumPTTracks)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TauRecExtraDetails_p1::~TauRecExtraDetails_p1() {
}
#endif // TauRecExtraDetails_p1_cxx

#ifndef tauAnalysisHelperObject_p1_cxx
#define tauAnalysisHelperObject_p1_cxx
tauAnalysisHelperObject_p1::tauAnalysisHelperObject_p1() {
}
tauAnalysisHelperObject_p1::tauAnalysisHelperObject_p1(const tauAnalysisHelperObject_p1 & rhs)
   : m_decmode(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_decmode)
   , m_jettype(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_jettype)
   , m_TowEMRadius(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_TowEMRadius)
   , m_TowET12Frac(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_TowET12Frac)
   , m_d0prf(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0prf)
   , m_d0iso(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0iso)
   , m_d0isoet(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0isoet)
   , m_d0ettr(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0ettr)
   , m_d0etem(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0etem)
   , m_d0etem2(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0etem2)
   , m_d0emclet(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0emclet)
   , m_d0emcleta(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0emcleta)
   , m_d0emclphi(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0emclphi)
   , m_d0et05(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0et05)
   , m_d0eta05(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0eta05)
   , m_d0phi05(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0phi05)
   , m_d0hadet(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0hadet)
   , m_d0hadeta(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0hadeta)
   , m_d0hadphi(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0hadphi)
   , m_d0type(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0type)
   , m_d0deltaR1(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0deltaR1)
   , m_d0eTosumpT(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0eTosumpT)
   , m_d0deltaR1had(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0deltaR1had)
   , m_d0em3iso(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0em3iso)
   , m_d0mtrem3(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0mtrem3)
   , m_d0deltaR2(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0deltaR2)
   , m_d0ntr1030(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0ntr1030)
   , m_d0EM12isof(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0EM12isof)
   , m_d0e1e2otaupT(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0e1e2otaupT)
   , m_d0ettro123(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0ettro123)
   , m_d0ett1oEtiso(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0ett1oEtiso)
   , m_d0ett1oEtisoet(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0ett1oEtisoet)
   , m_d0dalpha(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0dalpha)
   , m_d0e1e2(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0e1e2)
   , m_d0mtr1tr2(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0mtr1tr2)
   , m_d0mtr1tr2tr3(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0mtr1tr2tr3)
   , m_d0sumtaupt(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0sumtaupt)
   , m_d0sumnontaupt(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0sumnontaupt)
   , m_d0sumpt(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0sumpt)
   , m_towere(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_towere)
   , m_towereta(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_towereta)
   , m_towerphi(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_towerphi)
   , m_d0_emcluster(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0_emcluster)
   , m_d0_05_Tracks(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0_05_Tracks)
   , m_d0_tau_Tracks(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0_tau_Tracks)
   , m_d0_nontau_Tracks(const_cast<tauAnalysisHelperObject_p1 &>( rhs ).m_d0_nontau_Tracks)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<6;i++) m_emradii[i] = rhs.m_emradii[i];
   for (Int_t i=0;i<6;i++) m_hadradii[i] = rhs.m_hadradii[i];
   for (Int_t i=0;i<10;i++) m_ImpactParameter[i] = rhs.m_ImpactParameter[i];
   for (Int_t i=0;i<10;i++) m_RawImpactParameter[i] = rhs.m_RawImpactParameter[i];
   for (Int_t i=0;i<10;i++) m_SignedImpactParameter[i] = rhs.m_SignedImpactParameter[i];
   for (Int_t i=0;i<10;i++) m_ImpactParameterSignificance[i] = rhs.m_ImpactParameterSignificance[i];
   for (Int_t i=0;i<10;i++) m_SignedImpactParameterSignificance[i] = rhs.m_SignedImpactParameterSignificance[i];
   tauAnalysisHelperObject_p1 &modrhs = const_cast<tauAnalysisHelperObject_p1 &>( rhs );
   modrhs.m_towere.clear();
   modrhs.m_towereta.clear();
   modrhs.m_towerphi.clear();
   for (Int_t i=0;i<25;i++) m_d0uncaletlayers[i] = rhs.m_d0uncaletlayers[i];
}
tauAnalysisHelperObject_p1::~tauAnalysisHelperObject_p1() {
}
#endif // tauAnalysisHelperObject_p1_cxx

#ifndef P4IPtCotThPhiM_p1_cxx
#define P4IPtCotThPhiM_p1_cxx
P4IPtCotThPhiM_p1::P4IPtCotThPhiM_p1() {
}
P4IPtCotThPhiM_p1::P4IPtCotThPhiM_p1(const P4IPtCotThPhiM_p1 & rhs)
   : m_iPt(const_cast<P4IPtCotThPhiM_p1 &>( rhs ).m_iPt)
   , m_cotTh(const_cast<P4IPtCotThPhiM_p1 &>( rhs ).m_cotTh)
   , m_phi(const_cast<P4IPtCotThPhiM_p1 &>( rhs ).m_phi)
   , m_mass(const_cast<P4IPtCotThPhiM_p1 &>( rhs ).m_mass)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
P4IPtCotThPhiM_p1::~P4IPtCotThPhiM_p1() {
}
#endif // P4IPtCotThPhiM_p1_cxx

#ifndef Muon_p4_cxx
#define Muon_p4_cxx
Muon_p4::Muon_p4() {
}
Muon_p4::Muon_p4(const Muon_p4 & rhs)
   : m_momentum(const_cast<Muon_p4 &>( rhs ).m_momentum)
   , m_particleBase(const_cast<Muon_p4 &>( rhs ).m_particleBase)
   , m_author(const_cast<Muon_p4 &>( rhs ).m_author)
   , m_hasCombinedMuon(const_cast<Muon_p4 &>( rhs ).m_hasCombinedMuon)
   , m_hasInDetTrackParticle(const_cast<Muon_p4 &>( rhs ).m_hasInDetTrackParticle)
   , m_hasMuonExtrapolatedTrackParticle(const_cast<Muon_p4 &>( rhs ).m_hasMuonExtrapolatedTrackParticle)
   , m_hasInnerExtrapolatedTrackParticle(const_cast<Muon_p4 &>( rhs ).m_hasInnerExtrapolatedTrackParticle)
   , m_hasCombinedMuonTrackParticle(const_cast<Muon_p4 &>( rhs ).m_hasCombinedMuonTrackParticle)
   , m_hasCluster(const_cast<Muon_p4 &>( rhs ).m_hasCluster)
   , m_matchChi2(const_cast<Muon_p4 &>( rhs ).m_matchChi2)
   , m_associatedEtaDigits(const_cast<Muon_p4 &>( rhs ).m_associatedEtaDigits)
   , m_associatedPhiDigits(const_cast<Muon_p4 &>( rhs ).m_associatedPhiDigits)
   , m_inDetTrackParticle(const_cast<Muon_p4 &>( rhs ).m_inDetTrackParticle)
   , m_muonSegments(const_cast<Muon_p4 &>( rhs ).m_muonSegments)
   , m_muonSpectrometerTrackParticle(const_cast<Muon_p4 &>( rhs ).m_muonSpectrometerTrackParticle)
   , m_muonExtrapolatedTrackParticle(const_cast<Muon_p4 &>( rhs ).m_muonExtrapolatedTrackParticle)
   , m_innerExtrapolatedTrackParticle(const_cast<Muon_p4 &>( rhs ).m_innerExtrapolatedTrackParticle)
   , m_combinedMuonTrackParticle(const_cast<Muon_p4 &>( rhs ).m_combinedMuonTrackParticle)
   , m_cluster(const_cast<Muon_p4 &>( rhs ).m_cluster)
   , m_parameters(const_cast<Muon_p4 &>( rhs ).m_parameters)
   , m_bestMatch(const_cast<Muon_p4 &>( rhs ).m_bestMatch)
   , m_matchNumberDoF(const_cast<Muon_p4 &>( rhs ).m_matchNumberDoF)
   , m_isAlsoFoundByLowPt(const_cast<Muon_p4 &>( rhs ).m_isAlsoFoundByLowPt)
   , m_isAlsoFoundByCaloMuonId(const_cast<Muon_p4 &>( rhs ).m_isAlsoFoundByCaloMuonId)
   , m_caloEnergyLoss(const_cast<Muon_p4 &>( rhs ).m_caloEnergyLoss)
   , m_caloMuonAlsoFoundByMuonReco(const_cast<Muon_p4 &>( rhs ).m_caloMuonAlsoFoundByMuonReco)
   , m_isCorrected(const_cast<Muon_p4 &>( rhs ).m_isCorrected)
   , m_allAuthors(const_cast<Muon_p4 &>( rhs ).m_allAuthors)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Muon_p4 &modrhs = const_cast<Muon_p4 &>( rhs );
   modrhs.m_associatedEtaDigits.clear();
   modrhs.m_associatedPhiDigits.clear();
   modrhs.m_parameters.clear();
}
Muon_p4::~Muon_p4() {
}
#endif // Muon_p4_cxx

#ifndef CombinedMuonFeatureContainer_tlp1_cxx
#define CombinedMuonFeatureContainer_tlp1_cxx
CombinedMuonFeatureContainer_tlp1::CombinedMuonFeatureContainer_tlp1() {
}
CombinedMuonFeatureContainer_tlp1::CombinedMuonFeatureContainer_tlp1(const CombinedMuonFeatureContainer_tlp1 & rhs)
   : m_combinedMuonFeatureContainerVec(const_cast<CombinedMuonFeatureContainer_tlp1 &>( rhs ).m_combinedMuonFeatureContainerVec)
   , m_combinedMuonFeatureVec(const_cast<CombinedMuonFeatureContainer_tlp1 &>( rhs ).m_combinedMuonFeatureVec)
   , m_combinedMuonFeatureVec_p2(const_cast<CombinedMuonFeatureContainer_tlp1 &>( rhs ).m_combinedMuonFeatureVec_p2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CombinedMuonFeatureContainer_tlp1 &modrhs = const_cast<CombinedMuonFeatureContainer_tlp1 &>( rhs );
   modrhs.m_combinedMuonFeatureContainerVec.clear();
   modrhs.m_combinedMuonFeatureVec.clear();
   modrhs.m_combinedMuonFeatureVec_p2.clear();
}
CombinedMuonFeatureContainer_tlp1::~CombinedMuonFeatureContainer_tlp1() {
}
#endif // CombinedMuonFeatureContainer_tlp1_cxx

#ifndef CombinedMuonFeature_p1_cxx
#define CombinedMuonFeature_p1_cxx
CombinedMuonFeature_p1::CombinedMuonFeature_p1() {
}
CombinedMuonFeature_p1::CombinedMuonFeature_p1(const CombinedMuonFeature_p1 & rhs)
   : m_pt(const_cast<CombinedMuonFeature_p1 &>( rhs ).m_pt)
   , m_sigma_pt(const_cast<CombinedMuonFeature_p1 &>( rhs ).m_sigma_pt)
   , m_muFastTrack(const_cast<CombinedMuonFeature_p1 &>( rhs ).m_muFastTrack)
   , m_IDTrack(const_cast<CombinedMuonFeature_p1 &>( rhs ).m_IDTrack)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
CombinedMuonFeature_p1::~CombinedMuonFeature_p1() {
}
#endif // CombinedMuonFeature_p1_cxx

#ifndef CombinedMuonFeature_p2_cxx
#define CombinedMuonFeature_p2_cxx
CombinedMuonFeature_p2::CombinedMuonFeature_p2() {
}
CombinedMuonFeature_p2::CombinedMuonFeature_p2(const CombinedMuonFeature_p2 & rhs)
   : m_pt(const_cast<CombinedMuonFeature_p2 &>( rhs ).m_pt)
   , m_sigma_pt(const_cast<CombinedMuonFeature_p2 &>( rhs ).m_sigma_pt)
   , m_muFastTrack(const_cast<CombinedMuonFeature_p2 &>( rhs ).m_muFastTrack)
   , m_IDTrack(const_cast<CombinedMuonFeature_p2 &>( rhs ).m_IDTrack)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
CombinedMuonFeature_p2::~CombinedMuonFeature_p2() {
}
#endif // CombinedMuonFeature_p2_cxx

#ifndef TrigTauTracksInfoCollection_tlp1_cxx
#define TrigTauTracksInfoCollection_tlp1_cxx
TrigTauTracksInfoCollection_tlp1::TrigTauTracksInfoCollection_tlp1() {
}
TrigTauTracksInfoCollection_tlp1::TrigTauTracksInfoCollection_tlp1(const TrigTauTracksInfoCollection_tlp1 & rhs)
   : m_trigTauTracksInfoCollectionVec(const_cast<TrigTauTracksInfoCollection_tlp1 &>( rhs ).m_trigTauTracksInfoCollectionVec)
   , m_trigTauTracksInfoVec(const_cast<TrigTauTracksInfoCollection_tlp1 &>( rhs ).m_trigTauTracksInfoVec)
   , m_p4PtEtaPhiM(const_cast<TrigTauTracksInfoCollection_tlp1 &>( rhs ).m_p4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTauTracksInfoCollection_tlp1 &modrhs = const_cast<TrigTauTracksInfoCollection_tlp1 &>( rhs );
   modrhs.m_trigTauTracksInfoCollectionVec.clear();
   modrhs.m_trigTauTracksInfoVec.clear();
   modrhs.m_p4PtEtaPhiM.clear();
}
TrigTauTracksInfoCollection_tlp1::~TrigTauTracksInfoCollection_tlp1() {
}
#endif // TrigTauTracksInfoCollection_tlp1_cxx

#ifndef TrigTauTracksInfo_p1_cxx
#define TrigTauTracksInfo_p1_cxx
TrigTauTracksInfo_p1::TrigTauTracksInfo_p1() {
}
TrigTauTracksInfo_p1::TrigTauTracksInfo_p1(const TrigTauTracksInfo_p1 & rhs)
   : m_roiID(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_roiID)
   , m_nCoreTracks(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_nCoreTracks)
   , m_nSlowTracks(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_nSlowTracks)
   , m_nIsoTracks(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_nIsoTracks)
   , m_charge(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_charge)
   , m_leadingTrackPt(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_leadingTrackPt)
   , m_scalarPtSumCore(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_scalarPtSumCore)
   , m_scalarPtSumIso(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_scalarPtSumIso)
   , m_ptBalance(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_ptBalance)
   , m_3fastest(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_3fastest)
   , m_P4PtEtaPhiM(const_cast<TrigTauTracksInfo_p1 &>( rhs ).m_P4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigTauTracksInfo_p1::~TrigTauTracksInfo_p1() {
}
#endif // TrigTauTracksInfo_p1_cxx

#ifndef TrigRoiDescriptorCollection_tlp1_cxx
#define TrigRoiDescriptorCollection_tlp1_cxx
TrigRoiDescriptorCollection_tlp1::TrigRoiDescriptorCollection_tlp1() {
}
TrigRoiDescriptorCollection_tlp1::TrigRoiDescriptorCollection_tlp1(const TrigRoiDescriptorCollection_tlp1 & rhs)
   : m_TrigRoiDescriptorCollection(const_cast<TrigRoiDescriptorCollection_tlp1 &>( rhs ).m_TrigRoiDescriptorCollection)
   , m_TrigRoiDescriptor(const_cast<TrigRoiDescriptorCollection_tlp1 &>( rhs ).m_TrigRoiDescriptor)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigRoiDescriptorCollection_tlp1 &modrhs = const_cast<TrigRoiDescriptorCollection_tlp1 &>( rhs );
   modrhs.m_TrigRoiDescriptorCollection.clear();
   modrhs.m_TrigRoiDescriptor.clear();
}
TrigRoiDescriptorCollection_tlp1::~TrigRoiDescriptorCollection_tlp1() {
}
#endif // TrigRoiDescriptorCollection_tlp1_cxx

#ifndef TrigRoiDescriptor_p1_cxx
#define TrigRoiDescriptor_p1_cxx
TrigRoiDescriptor_p1::TrigRoiDescriptor_p1() {
}
TrigRoiDescriptor_p1::TrigRoiDescriptor_p1(const TrigRoiDescriptor_p1 & rhs)
   : m_phi0(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_phi0)
   , m_eta0(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_eta0)
   , m_zed0(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_zed0)
   , m_phiHalfWidth(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_phiHalfWidth)
   , m_etaHalfWidth(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_etaHalfWidth)
   , m_zedHalfWidth(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_zedHalfWidth)
   , m_etaPlus(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_etaPlus)
   , m_etaMinus(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_etaMinus)
   , m_l1Id(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_l1Id)
   , m_roiId(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_roiId)
   , m_roiWord(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_roiWord)
   , m_serialized(const_cast<TrigRoiDescriptor_p1 &>( rhs ).m_serialized)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigRoiDescriptor_p1 &modrhs = const_cast<TrigRoiDescriptor_p1 &>( rhs );
   modrhs.m_serialized.clear();
}
TrigRoiDescriptor_p1::~TrigRoiDescriptor_p1() {
}
#endif // TrigRoiDescriptor_p1_cxx

#ifndef TrigTrackCountsCollection_tlp1_cxx
#define TrigTrackCountsCollection_tlp1_cxx
TrigTrackCountsCollection_tlp1::TrigTrackCountsCollection_tlp1() {
}
TrigTrackCountsCollection_tlp1::TrigTrackCountsCollection_tlp1(const TrigTrackCountsCollection_tlp1 & rhs)
   : m_trigTrackCountsCollectionVec(const_cast<TrigTrackCountsCollection_tlp1 &>( rhs ).m_trigTrackCountsCollectionVec)
   , m_trigTrackCountsVec(const_cast<TrigTrackCountsCollection_tlp1 &>( rhs ).m_trigTrackCountsVec)
   , m_trigTrackCountsVec_p2(const_cast<TrigTrackCountsCollection_tlp1 &>( rhs ).m_trigTrackCountsVec_p2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTrackCountsCollection_tlp1 &modrhs = const_cast<TrigTrackCountsCollection_tlp1 &>( rhs );
   modrhs.m_trigTrackCountsCollectionVec.clear();
   modrhs.m_trigTrackCountsVec.clear();
   modrhs.m_trigTrackCountsVec_p2.clear();
}
TrigTrackCountsCollection_tlp1::~TrigTrackCountsCollection_tlp1() {
}
#endif // TrigTrackCountsCollection_tlp1_cxx

#ifndef TrigTrackCounts_p1_cxx
#define TrigTrackCounts_p1_cxx
TrigTrackCounts_p1::TrigTrackCounts_p1() {
}
TrigTrackCounts_p1::TrigTrackCounts_p1(const TrigTrackCounts_p1 & rhs)
   : m_z0pcnt(const_cast<TrigTrackCounts_p1 &>( rhs ).m_z0pcnt)
   , m_phi0cnt(const_cast<TrigTrackCounts_p1 &>( rhs ).m_phi0cnt)
   , m_etacnt(const_cast<TrigTrackCounts_p1 &>( rhs ).m_etacnt)
   , m_ptcnt(const_cast<TrigTrackCounts_p1 &>( rhs ).m_ptcnt)
   , m_trkcnt(const_cast<TrigTrackCounts_p1 &>( rhs ).m_trkcnt)
   , m_pixcnt(const_cast<TrigTrackCounts_p1 &>( rhs ).m_pixcnt)
   , m_sctcnt(const_cast<TrigTrackCounts_p1 &>( rhs ).m_sctcnt)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTrackCounts_p1 &modrhs = const_cast<TrigTrackCounts_p1 &>( rhs );
   modrhs.m_z0pcnt.clear();
   modrhs.m_phi0cnt.clear();
   modrhs.m_etacnt.clear();
   modrhs.m_ptcnt.clear();
}
TrigTrackCounts_p1::~TrigTrackCounts_p1() {
}
#endif // TrigTrackCounts_p1_cxx

#ifndef TrigHisto2D_p1_cxx
#define TrigHisto2D_p1_cxx
TrigHisto2D_p1::TrigHisto2D_p1() {
}
TrigHisto2D_p1::TrigHisto2D_p1(const TrigHisto2D_p1 & rhs)
   : m_contents(const_cast<TrigHisto2D_p1 &>( rhs ).m_contents)
   , m_nbins_x(const_cast<TrigHisto2D_p1 &>( rhs ).m_nbins_x)
   , m_min_x(const_cast<TrigHisto2D_p1 &>( rhs ).m_min_x)
   , m_max_x(const_cast<TrigHisto2D_p1 &>( rhs ).m_max_x)
   , m_nbins_y(const_cast<TrigHisto2D_p1 &>( rhs ).m_nbins_y)
   , m_min_y(const_cast<TrigHisto2D_p1 &>( rhs ).m_min_y)
   , m_max_y(const_cast<TrigHisto2D_p1 &>( rhs ).m_max_y)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigHisto2D_p1 &modrhs = const_cast<TrigHisto2D_p1 &>( rhs );
   modrhs.m_contents.clear();
}
TrigHisto2D_p1::~TrigHisto2D_p1() {
}
#endif // TrigHisto2D_p1_cxx

#ifndef TrigTrackCounts_p2_cxx
#define TrigTrackCounts_p2_cxx
TrigTrackCounts_p2::TrigTrackCounts_p2() {
}
TrigTrackCounts_p2::TrigTrackCounts_p2(const TrigTrackCounts_p2 & rhs)
   : m_z0_pt(const_cast<TrigTrackCounts_p2 &>( rhs ).m_z0_pt)
   , m_eta_phi(const_cast<TrigTrackCounts_p2 &>( rhs ).m_eta_phi)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigTrackCounts_p2::~TrigTrackCounts_p2() {
}
#endif // TrigTrackCounts_p2_cxx

#ifndef TrigSpacePointCountsCollection_tlp1_cxx
#define TrigSpacePointCountsCollection_tlp1_cxx
TrigSpacePointCountsCollection_tlp1::TrigSpacePointCountsCollection_tlp1() {
}
TrigSpacePointCountsCollection_tlp1::TrigSpacePointCountsCollection_tlp1(const TrigSpacePointCountsCollection_tlp1 & rhs)
   : m_trigSpacePointCountsCollectionVec(const_cast<TrigSpacePointCountsCollection_tlp1 &>( rhs ).m_trigSpacePointCountsCollectionVec)
   , m_trigSpacePointCountsVec(const_cast<TrigSpacePointCountsCollection_tlp1 &>( rhs ).m_trigSpacePointCountsVec)
   , m_trigSpacePointCountsVec_p2(const_cast<TrigSpacePointCountsCollection_tlp1 &>( rhs ).m_trigSpacePointCountsVec_p2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigSpacePointCountsCollection_tlp1 &modrhs = const_cast<TrigSpacePointCountsCollection_tlp1 &>( rhs );
   modrhs.m_trigSpacePointCountsCollectionVec.clear();
   modrhs.m_trigSpacePointCountsVec.clear();
   modrhs.m_trigSpacePointCountsVec_p2.clear();
}
TrigSpacePointCountsCollection_tlp1::~TrigSpacePointCountsCollection_tlp1() {
}
#endif // TrigSpacePointCountsCollection_tlp1_cxx

#ifndef TrigSpacePointCounts_p1_cxx
#define TrigSpacePointCounts_p1_cxx
TrigSpacePointCounts_p1::TrigSpacePointCounts_p1() {
}
TrigSpacePointCounts_p1::TrigSpacePointCounts_p1(const TrigSpacePointCounts_p1 & rhs)
   : m_pixSPcnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_pixSPcnt)
   , m_pixCL1cnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_pixCL1cnt)
   , m_pixCL2cnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_pixCL2cnt)
   , m_pixCLmin3cnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_pixCLmin3cnt)
   , m_SPpixBarr_cnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_SPpixBarr_cnt)
   , m_SPpixECAcnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_SPpixECAcnt)
   , m_SPpixECCcnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_SPpixECCcnt)
   , m_sctSPcnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_sctSPcnt)
   , m_SPsctBarr_cnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_SPsctBarr_cnt)
   , m_SPsctECAcnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_SPsctECAcnt)
   , m_SPsctECCcnt(const_cast<TrigSpacePointCounts_p1 &>( rhs ).m_SPsctECCcnt)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigSpacePointCounts_p1::~TrigSpacePointCounts_p1() {
}
#endif // TrigSpacePointCounts_p1_cxx

#ifndef TrigSpacePointCounts_p2_cxx
#define TrigSpacePointCounts_p2_cxx
TrigSpacePointCounts_p2::TrigSpacePointCounts_p2() {
}
TrigSpacePointCounts_p2::TrigSpacePointCounts_p2(const TrigSpacePointCounts_p2 & rhs)
   : m_pixelClusEndcapC(const_cast<TrigSpacePointCounts_p2 &>( rhs ).m_pixelClusEndcapC)
   , m_pixelClusBarrel(const_cast<TrigSpacePointCounts_p2 &>( rhs ).m_pixelClusBarrel)
   , m_pixelClusEndcapA(const_cast<TrigSpacePointCounts_p2 &>( rhs ).m_pixelClusEndcapA)
   , m_sctSpEndcapC(const_cast<TrigSpacePointCounts_p2 &>( rhs ).m_sctSpEndcapC)
   , m_sctSpBarrel(const_cast<TrigSpacePointCounts_p2 &>( rhs ).m_sctSpBarrel)
   , m_sctSpEndcapA(const_cast<TrigSpacePointCounts_p2 &>( rhs ).m_sctSpEndcapA)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigSpacePointCounts_p2::~TrigSpacePointCounts_p2() {
}
#endif // TrigSpacePointCounts_p2_cxx

#ifndef TrigEFBjetContainer_tlp2_cxx
#define TrigEFBjetContainer_tlp2_cxx
TrigEFBjetContainer_tlp2::TrigEFBjetContainer_tlp2() {
}
TrigEFBjetContainer_tlp2::TrigEFBjetContainer_tlp2(const TrigEFBjetContainer_tlp2 & rhs)
   : m_TrigEFBjetContainers(const_cast<TrigEFBjetContainer_tlp2 &>( rhs ).m_TrigEFBjetContainers)
   , m_EFBjet(const_cast<TrigEFBjetContainer_tlp2 &>( rhs ).m_EFBjet)
   , m_p4PtEtaPhiM(const_cast<TrigEFBjetContainer_tlp2 &>( rhs ).m_p4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigEFBjetContainer_tlp2 &modrhs = const_cast<TrigEFBjetContainer_tlp2 &>( rhs );
   modrhs.m_TrigEFBjetContainers.clear();
   modrhs.m_EFBjet.clear();
   modrhs.m_p4PtEtaPhiM.clear();
}
TrigEFBjetContainer_tlp2::~TrigEFBjetContainer_tlp2() {
}
#endif // TrigEFBjetContainer_tlp2_cxx

#ifndef TrigEFBjet_p2_cxx
#define TrigEFBjet_p2_cxx
TrigEFBjet_p2::TrigEFBjet_p2() {
}
TrigEFBjet_p2::TrigEFBjet_p2(const TrigEFBjet_p2 & rhs)
   : m_valid(const_cast<TrigEFBjet_p2 &>( rhs ).m_valid)
   , m_roiID(const_cast<TrigEFBjet_p2 &>( rhs ).m_roiID)
   , m_prmVtx(const_cast<TrigEFBjet_p2 &>( rhs ).m_prmVtx)
   , m_xcomb(const_cast<TrigEFBjet_p2 &>( rhs ).m_xcomb)
   , m_xIP1d(const_cast<TrigEFBjet_p2 &>( rhs ).m_xIP1d)
   , m_xIP2d(const_cast<TrigEFBjet_p2 &>( rhs ).m_xIP2d)
   , m_xIP3d(const_cast<TrigEFBjet_p2 &>( rhs ).m_xIP3d)
   , m_xChi2(const_cast<TrigEFBjet_p2 &>( rhs ).m_xChi2)
   , m_xSv(const_cast<TrigEFBjet_p2 &>( rhs ).m_xSv)
   , m_xmvtx(const_cast<TrigEFBjet_p2 &>( rhs ).m_xmvtx)
   , m_xevtx(const_cast<TrigEFBjet_p2 &>( rhs ).m_xevtx)
   , m_xnvtx(const_cast<TrigEFBjet_p2 &>( rhs ).m_xnvtx)
   , m_p4PtEtaPhiM(const_cast<TrigEFBjet_p2 &>( rhs ).m_p4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigEFBjet_p2::~TrigEFBjet_p2() {
}
#endif // TrigEFBjet_p2_cxx

#ifndef TrigTauClusterDetailsContainer_tlp1_cxx
#define TrigTauClusterDetailsContainer_tlp1_cxx
TrigTauClusterDetailsContainer_tlp1::TrigTauClusterDetailsContainer_tlp1() {
}
TrigTauClusterDetailsContainer_tlp1::TrigTauClusterDetailsContainer_tlp1(const TrigTauClusterDetailsContainer_tlp1 & rhs)
   : m_TrigTauClusterDetailsContainers(const_cast<TrigTauClusterDetailsContainer_tlp1 &>( rhs ).m_TrigTauClusterDetailsContainers)
   , m_TauClusterDetails(const_cast<TrigTauClusterDetailsContainer_tlp1 &>( rhs ).m_TauClusterDetails)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTauClusterDetailsContainer_tlp1 &modrhs = const_cast<TrigTauClusterDetailsContainer_tlp1 &>( rhs );
   modrhs.m_TrigTauClusterDetailsContainers.clear();
   modrhs.m_TauClusterDetails.clear();
}
TrigTauClusterDetailsContainer_tlp1::~TrigTauClusterDetailsContainer_tlp1() {
}
#endif // TrigTauClusterDetailsContainer_tlp1_cxx

#ifndef TrigTauClusterDetails_p1_cxx
#define TrigTauClusterDetails_p1_cxx
TrigTauClusterDetails_p1::TrigTauClusterDetails_p1() {
}
TrigTauClusterDetails_p1::TrigTauClusterDetails_p1(const TrigTauClusterDetails_p1 & rhs)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<4;i++) m_EMRadius[i] = rhs.m_EMRadius[i];
   for (Int_t i=0;i<3;i++) m_HADRadius[i] = rhs.m_HADRadius[i];
   for (Int_t i=0;i<4;i++) m_EMenergyWidth[i] = rhs.m_EMenergyWidth[i];
   for (Int_t i=0;i<3;i++) m_HADenergyWidth[i] = rhs.m_HADenergyWidth[i];
   for (Int_t i=0;i<4;i++) m_EMenergyWide[i] = rhs.m_EMenergyWide[i];
   for (Int_t i=0;i<4;i++) m_EMenergyMedium[i] = rhs.m_EMenergyMedium[i];
   for (Int_t i=0;i<4;i++) m_EMenergyNarrow[i] = rhs.m_EMenergyNarrow[i];
   for (Int_t i=0;i<3;i++) m_HADenergyWide[i] = rhs.m_HADenergyWide[i];
   for (Int_t i=0;i<3;i++) m_HADenergyMedium[i] = rhs.m_HADenergyMedium[i];
   for (Int_t i=0;i<3;i++) m_HADenergyNarrow[i] = rhs.m_HADenergyNarrow[i];
}
TrigTauClusterDetails_p1::~TrigTauClusterDetails_p1() {
}
#endif // TrigTauClusterDetails_p1_cxx

#ifndef JetCollection_tlp5_cxx
#define JetCollection_tlp5_cxx
JetCollection_tlp5::JetCollection_tlp5() {
}
JetCollection_tlp5::JetCollection_tlp5(const JetCollection_tlp5 & rhs)
   : m_jetCollectionContainers(const_cast<JetCollection_tlp5 &>( rhs ).m_jetCollectionContainers)
   , m_jet_p5(const_cast<JetCollection_tlp5 &>( rhs ).m_jet_p5)
   , m_jetAssociationBase_p1(const_cast<JetCollection_tlp5 &>( rhs ).m_jetAssociationBase_p1)
   , m_tokenList(const_cast<JetCollection_tlp5 &>( rhs ).m_tokenList)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   JetCollection_tlp5 &modrhs = const_cast<JetCollection_tlp5 &>( rhs );
   modrhs.m_jetCollectionContainers.clear();
   modrhs.m_jet_p5.clear();
   modrhs.m_jetAssociationBase_p1.clear();
}
JetCollection_tlp5::~JetCollection_tlp5() {
}
#endif // JetCollection_tlp5_cxx

#ifndef Navigable_p1_unsigned_int_double__cxx
#define Navigable_p1_unsigned_int_double__cxx
Navigable_p1<unsigned int,double>::Navigable_p1() {
}
Navigable_p1<unsigned int,double>::Navigable_p1(const Navigable_p1 & rhs)
   : m_links(const_cast<Navigable_p1 &>( rhs ).m_links)
   , m_parameters(const_cast<Navigable_p1 &>( rhs ).m_parameters)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Navigable_p1 &modrhs = const_cast<Navigable_p1 &>( rhs );
   modrhs.m_parameters.clear();
}
Navigable_p1<unsigned int,double>::~Navigable_p1() {
}
#endif // Navigable_p1_unsigned_int_double__cxx

#ifndef JetConverterTypes__momentum_cxx
#define JetConverterTypes__momentum_cxx
JetConverterTypes::momentum::momentum() {
}
JetConverterTypes::momentum::momentum(const momentum & rhs)
   : m_px(const_cast<momentum &>( rhs ).m_px)
   , m_py(const_cast<momentum &>( rhs ).m_py)
   , m_pz(const_cast<momentum &>( rhs ).m_pz)
   , m_m(const_cast<momentum &>( rhs ).m_m)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
JetConverterTypes::momentum::~momentum() {
}
#endif // JetConverterTypes__momentum_cxx

#ifndef Jet_p5_cxx
#define Jet_p5_cxx
Jet_p5::Jet_p5() {
}
Jet_p5::Jet_p5(const Jet_p5 & rhs)
   : m_nav(const_cast<Jet_p5 &>( rhs ).m_nav)
   , m_momentum(const_cast<Jet_p5 &>( rhs ).m_momentum)
   , m_rawSignal(const_cast<Jet_p5 &>( rhs ).m_rawSignal)
   , m_partBase(const_cast<Jet_p5 &>( rhs ).m_partBase)
   , m_author(const_cast<Jet_p5 &>( rhs ).m_author)
   , m_num_combinedLikelihood(const_cast<Jet_p5 &>( rhs ).m_num_combinedLikelihood)
   , m_shapeStore(const_cast<Jet_p5 &>( rhs ).m_shapeStore)
   , m_tagJetInfo(const_cast<Jet_p5 &>( rhs ).m_tagJetInfo)
   , m_associations(const_cast<Jet_p5 &>( rhs ).m_associations)
   , m_recoStatus(const_cast<Jet_p5 &>( rhs ).m_recoStatus)
   , m_usedForTrigger(const_cast<Jet_p5 &>( rhs ).m_usedForTrigger)
   , m_constituentsN(const_cast<Jet_p5 &>( rhs ).m_constituentsN)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   Jet_p5 &modrhs = const_cast<Jet_p5 &>( rhs );
   modrhs.m_rawSignal.clear();
   modrhs.m_shapeStore.clear();
   modrhs.m_tagJetInfo.clear();
   modrhs.m_associations.clear();
}
Jet_p5::~Jet_p5() {
}
#endif // Jet_p5_cxx

#ifndef JetAssociationBase_p1_cxx
#define JetAssociationBase_p1_cxx
JetAssociationBase_p1::JetAssociationBase_p1() {
}
JetAssociationBase_p1::JetAssociationBase_p1(const JetAssociationBase_p1 & rhs)
   : m_keyIndex(const_cast<JetAssociationBase_p1 &>( rhs ).m_keyIndex)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
JetAssociationBase_p1::~JetAssociationBase_p1() {
}
#endif // JetAssociationBase_p1_cxx

#ifndef TrigMuonEFContainer_tlp1_cxx
#define TrigMuonEFContainer_tlp1_cxx
TrigMuonEFContainer_tlp1::TrigMuonEFContainer_tlp1() {
}
TrigMuonEFContainer_tlp1::TrigMuonEFContainer_tlp1(const TrigMuonEFContainer_tlp1 & rhs)
   : m_TrigMuonEFContainers(const_cast<TrigMuonEFContainer_tlp1 &>( rhs ).m_TrigMuonEFContainers)
   , m_MuonEF(const_cast<TrigMuonEFContainer_tlp1 &>( rhs ).m_MuonEF)
   , m_P4IPtCotThPhiM(const_cast<TrigMuonEFContainer_tlp1 &>( rhs ).m_P4IPtCotThPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigMuonEFContainer_tlp1 &modrhs = const_cast<TrigMuonEFContainer_tlp1 &>( rhs );
   modrhs.m_TrigMuonEFContainers.clear();
   modrhs.m_MuonEF.clear();
   modrhs.m_P4IPtCotThPhiM.clear();
}
TrigMuonEFContainer_tlp1::~TrigMuonEFContainer_tlp1() {
}
#endif // TrigMuonEFContainer_tlp1_cxx

#ifndef TrigMuonEF_p1_cxx
#define TrigMuonEF_p1_cxx
TrigMuonEF_p1::TrigMuonEF_p1() {
}
TrigMuonEF_p1::TrigMuonEF_p1(const TrigMuonEF_p1 & rhs)
   : m_muonCode(const_cast<TrigMuonEF_p1 &>( rhs ).m_muonCode)
   , m_roi(const_cast<TrigMuonEF_p1 &>( rhs ).m_roi)
   , m_charge(const_cast<TrigMuonEF_p1 &>( rhs ).m_charge)
   , m_P4IPtCotThPhiM(const_cast<TrigMuonEF_p1 &>( rhs ).m_P4IPtCotThPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigMuonEF_p1 &modrhs = const_cast<TrigMuonEF_p1 &>( rhs );
   modrhs.m_roi.clear();
}
TrigMuonEF_p1::~TrigMuonEF_p1() {
}
#endif // TrigMuonEF_p1_cxx

#ifndef TrigMuonEFInfoContainer_tlp1_cxx
#define TrigMuonEFInfoContainer_tlp1_cxx
TrigMuonEFInfoContainer_tlp1::TrigMuonEFInfoContainer_tlp1() {
}
TrigMuonEFInfoContainer_tlp1::TrigMuonEFInfoContainer_tlp1(const TrigMuonEFInfoContainer_tlp1 & rhs)
   : m_TrigMuonEFInfoContainers(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_TrigMuonEFInfoContainers)
   , m_MuonEFInfo(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_MuonEFInfo)
   , m_MuonEFInfo_p2(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_MuonEFInfo_p2)
   , m_MuonEFInfo_p3(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_MuonEFInfo_p3)
   , m_MuonEFTrack(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_MuonEFTrack)
   , m_MuonEFTrack_p2(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_MuonEFTrack_p2)
   , m_MuonEFCbTrack(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_MuonEFCbTrack)
   , m_MuonEFCbTrack_p2(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_MuonEFCbTrack_p2)
   , m_P4IPtCotThPhiM(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_P4IPtCotThPhiM)
   , m_MuonEFInfoTrackContainer(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_MuonEFInfoTrackContainer)
   , m_MuonEFInfoTrack(const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs ).m_MuonEFInfoTrack)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigMuonEFInfoContainer_tlp1 &modrhs = const_cast<TrigMuonEFInfoContainer_tlp1 &>( rhs );
   modrhs.m_TrigMuonEFInfoContainers.clear();
   modrhs.m_MuonEFInfo.clear();
   modrhs.m_MuonEFInfo_p2.clear();
   modrhs.m_MuonEFInfo_p3.clear();
   modrhs.m_MuonEFTrack.clear();
   modrhs.m_MuonEFTrack_p2.clear();
   modrhs.m_MuonEFCbTrack.clear();
   modrhs.m_MuonEFCbTrack_p2.clear();
   modrhs.m_P4IPtCotThPhiM.clear();
   modrhs.m_MuonEFInfoTrackContainer.clear();
   modrhs.m_MuonEFInfoTrack.clear();
}
TrigMuonEFInfoContainer_tlp1::~TrigMuonEFInfoContainer_tlp1() {
}
#endif // TrigMuonEFInfoContainer_tlp1_cxx

#ifndef TrigMuonEFInfo_p1_cxx
#define TrigMuonEFInfo_p1_cxx
TrigMuonEFInfo_p1::TrigMuonEFInfo_p1() {
}
TrigMuonEFInfo_p1::TrigMuonEFInfo_p1(const TrigMuonEFInfo_p1 & rhs)
   : m_roi(const_cast<TrigMuonEFInfo_p1 &>( rhs ).m_roi)
   , m_spectrometerTrack(const_cast<TrigMuonEFInfo_p1 &>( rhs ).m_spectrometerTrack)
   , m_extrapolatedTrack(const_cast<TrigMuonEFInfo_p1 &>( rhs ).m_extrapolatedTrack)
   , m_combinedTrack(const_cast<TrigMuonEFInfo_p1 &>( rhs ).m_combinedTrack)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigMuonEFInfo_p1::~TrigMuonEFInfo_p1() {
}
#endif // TrigMuonEFInfo_p1_cxx

#ifndef TrigMuonEFInfo_p2_cxx
#define TrigMuonEFInfo_p2_cxx
TrigMuonEFInfo_p2::TrigMuonEFInfo_p2() {
}
TrigMuonEFInfo_p2::TrigMuonEFInfo_p2(const TrigMuonEFInfo_p2 & rhs)
   : m_roi(const_cast<TrigMuonEFInfo_p2 &>( rhs ).m_roi)
   , m_spectrometerTrack(const_cast<TrigMuonEFInfo_p2 &>( rhs ).m_spectrometerTrack)
   , m_extrapolatedTrack(const_cast<TrigMuonEFInfo_p2 &>( rhs ).m_extrapolatedTrack)
   , m_combinedTrack(const_cast<TrigMuonEFInfo_p2 &>( rhs ).m_combinedTrack)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigMuonEFInfo_p2::~TrigMuonEFInfo_p2() {
}
#endif // TrigMuonEFInfo_p2_cxx

#ifndef TrigMuonEFInfo_p3_cxx
#define TrigMuonEFInfo_p3_cxx
TrigMuonEFInfo_p3::TrigMuonEFInfo_p3() {
}
TrigMuonEFInfo_p3::TrigMuonEFInfo_p3(const TrigMuonEFInfo_p3 & rhs)
   : m_roi(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_roi)
   , m_nSegments(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_nSegments)
   , m_nMdtHits(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_nMdtHits)
   , m_nRpcHits(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_nRpcHits)
   , m_nTgcHits(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_nTgcHits)
   , m_nCscHits(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_nCscHits)
   , m_etaPreviousLevel(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_etaPreviousLevel)
   , m_phiPreviousLevel(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_phiPreviousLevel)
   , m_spectrometerTrack(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_spectrometerTrack)
   , m_extrapolatedTrack(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_extrapolatedTrack)
   , m_combinedTrack(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_combinedTrack)
   , m_trackContainer(const_cast<TrigMuonEFInfo_p3 &>( rhs ).m_trackContainer)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigMuonEFInfo_p3::~TrigMuonEFInfo_p3() {
}
#endif // TrigMuonEFInfo_p3_cxx

#ifndef TrigMuonEFTrack_p1_cxx
#define TrigMuonEFTrack_p1_cxx
TrigMuonEFTrack_p1::TrigMuonEFTrack_p1() {
}
TrigMuonEFTrack_p1::TrigMuonEFTrack_p1(const TrigMuonEFTrack_p1 & rhs)
   : m_charge(const_cast<TrigMuonEFTrack_p1 &>( rhs ).m_charge)
   , m_P4IPtCotThPhiM(const_cast<TrigMuonEFTrack_p1 &>( rhs ).m_P4IPtCotThPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigMuonEFTrack_p1::~TrigMuonEFTrack_p1() {
}
#endif // TrigMuonEFTrack_p1_cxx

#ifndef TrigMuonEFTrack_p2_cxx
#define TrigMuonEFTrack_p2_cxx
TrigMuonEFTrack_p2::TrigMuonEFTrack_p2() {
}
TrigMuonEFTrack_p2::TrigMuonEFTrack_p2(const TrigMuonEFTrack_p2 & rhs)
   : m_charge(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_charge)
   , m_d0(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_d0)
   , m_z0(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_z0)
   , m_chi2(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_chi2)
   , m_chi2prob(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_chi2prob)
   , m_posx(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_posx)
   , m_posy(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_posy)
   , m_posz(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_posz)
   , m_nMdtHitsPhi(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_nMdtHitsPhi)
   , m_nRpcHitsPhi(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_nRpcHitsPhi)
   , m_nTgcHitsPhi(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_nTgcHitsPhi)
   , m_nCscHitsPhi(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_nCscHitsPhi)
   , m_nMdtHitsEta(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_nMdtHitsEta)
   , m_nRpcHitsEta(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_nRpcHitsEta)
   , m_nTgcHitsEta(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_nTgcHitsEta)
   , m_nCscHitsEta(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_nCscHitsEta)
   , m_P4IPtCotThPhiM(const_cast<TrigMuonEFTrack_p2 &>( rhs ).m_P4IPtCotThPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigMuonEFTrack_p2::~TrigMuonEFTrack_p2() {
}
#endif // TrigMuonEFTrack_p2_cxx

#ifndef TrigMuonEFCbTrack_p1_cxx
#define TrigMuonEFCbTrack_p1_cxx
TrigMuonEFCbTrack_p1::TrigMuonEFCbTrack_p1() {
}
TrigMuonEFCbTrack_p1::TrigMuonEFCbTrack_p1(const TrigMuonEFCbTrack_p1 & rhs)
   : m_matchChi2(const_cast<TrigMuonEFCbTrack_p1 &>( rhs ).m_matchChi2)
   , m_TrigMuonEFTrack(const_cast<TrigMuonEFCbTrack_p1 &>( rhs ).m_TrigMuonEFTrack)
   , m_P4IPtCotThPhiM(const_cast<TrigMuonEFCbTrack_p1 &>( rhs ).m_P4IPtCotThPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigMuonEFCbTrack_p1::~TrigMuonEFCbTrack_p1() {
}
#endif // TrigMuonEFCbTrack_p1_cxx

#ifndef TrigMuonEFCbTrack_p2_cxx
#define TrigMuonEFCbTrack_p2_cxx
TrigMuonEFCbTrack_p2::TrigMuonEFCbTrack_p2() {
}
TrigMuonEFCbTrack_p2::TrigMuonEFCbTrack_p2(const TrigMuonEFCbTrack_p2 & rhs)
   : m_matchChi2(const_cast<TrigMuonEFCbTrack_p2 &>( rhs ).m_matchChi2)
   , m_nIdSctHits(const_cast<TrigMuonEFCbTrack_p2 &>( rhs ).m_nIdSctHits)
   , m_nIdPixelHits(const_cast<TrigMuonEFCbTrack_p2 &>( rhs ).m_nIdPixelHits)
   , m_nTrtHits(const_cast<TrigMuonEFCbTrack_p2 &>( rhs ).m_nTrtHits)
   , m_TrigMuonEFTrack(const_cast<TrigMuonEFCbTrack_p2 &>( rhs ).m_TrigMuonEFTrack)
   , m_P4IPtCotThPhiM(const_cast<TrigMuonEFCbTrack_p2 &>( rhs ).m_P4IPtCotThPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigMuonEFCbTrack_p2::~TrigMuonEFCbTrack_p2() {
}
#endif // TrigMuonEFCbTrack_p2_cxx

#ifndef TrigMuonEFInfoTrack_p1_cxx
#define TrigMuonEFInfoTrack_p1_cxx
TrigMuonEFInfoTrack_p1::TrigMuonEFInfoTrack_p1() {
}
TrigMuonEFInfoTrack_p1::TrigMuonEFInfoTrack_p1(const TrigMuonEFInfoTrack_p1 & rhs)
   : m_muonType(const_cast<TrigMuonEFInfoTrack_p1 &>( rhs ).m_muonType)
   , m_spectrometerTrack(const_cast<TrigMuonEFInfoTrack_p1 &>( rhs ).m_spectrometerTrack)
   , m_extrapolatedTrack(const_cast<TrigMuonEFInfoTrack_p1 &>( rhs ).m_extrapolatedTrack)
   , m_combinedTrack(const_cast<TrigMuonEFInfoTrack_p1 &>( rhs ).m_combinedTrack)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigMuonEFInfoTrack_p1::~TrigMuonEFInfoTrack_p1() {
}
#endif // TrigMuonEFInfoTrack_p1_cxx

#ifndef TrigT2JetContainer_tlp1_cxx
#define TrigT2JetContainer_tlp1_cxx
TrigT2JetContainer_tlp1::TrigT2JetContainer_tlp1() {
}
TrigT2JetContainer_tlp1::TrigT2JetContainer_tlp1(const TrigT2JetContainer_tlp1 & rhs)
   : m_TrigT2JetContainers(const_cast<TrigT2JetContainer_tlp1 &>( rhs ).m_TrigT2JetContainers)
   , m_T2Jet(const_cast<TrigT2JetContainer_tlp1 &>( rhs ).m_T2Jet)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigT2JetContainer_tlp1 &modrhs = const_cast<TrigT2JetContainer_tlp1 &>( rhs );
   modrhs.m_TrigT2JetContainers.clear();
   modrhs.m_T2Jet.clear();
}
TrigT2JetContainer_tlp1::~TrigT2JetContainer_tlp1() {
}
#endif // TrigT2JetContainer_tlp1_cxx

#ifndef TrigT2Jet_p1_cxx
#define TrigT2Jet_p1_cxx
TrigT2Jet_p1::TrigT2Jet_p1() {
}
TrigT2Jet_p1::TrigT2Jet_p1(const TrigT2Jet_p1 & rhs)
   : m_e(const_cast<TrigT2Jet_p1 &>( rhs ).m_e)
   , m_ehad0(const_cast<TrigT2Jet_p1 &>( rhs ).m_ehad0)
   , m_eem0(const_cast<TrigT2Jet_p1 &>( rhs ).m_eem0)
   , m_eta(const_cast<TrigT2Jet_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<TrigT2Jet_p1 &>( rhs ).m_phi)
   , m_roiWord(const_cast<TrigT2Jet_p1 &>( rhs ).m_roiWord)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigT2Jet_p1::~TrigT2Jet_p1() {
}
#endif // TrigT2Jet_p1_cxx

#ifndef egDetail_p1_cxx
#define egDetail_p1_cxx
egDetail_p1::egDetail_p1() {
}
egDetail_p1::egDetail_p1(const egDetail_p1 & rhs)
   : m_className(const_cast<egDetail_p1 &>( rhs ).m_className)
   , m_egDetailEnumParams(const_cast<egDetail_p1 &>( rhs ).m_egDetailEnumParams)
   , m_egDetailFloatParams(const_cast<egDetail_p1 &>( rhs ).m_egDetailFloatParams)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   egDetail_p1 &modrhs = const_cast<egDetail_p1 &>( rhs );
   modrhs.m_className.clear();
   modrhs.m_egDetailEnumParams.clear();
   modrhs.m_egDetailFloatParams.clear();
}
egDetail_p1::~egDetail_p1() {
}
#endif // egDetail_p1_cxx

#ifndef TrigT2MbtsBitsContainer_tlp1_cxx
#define TrigT2MbtsBitsContainer_tlp1_cxx
TrigT2MbtsBitsContainer_tlp1::TrigT2MbtsBitsContainer_tlp1() {
}
TrigT2MbtsBitsContainer_tlp1::TrigT2MbtsBitsContainer_tlp1(const TrigT2MbtsBitsContainer_tlp1 & rhs)
   : m_trigT2MbtsBitsContainerVec(const_cast<TrigT2MbtsBitsContainer_tlp1 &>( rhs ).m_trigT2MbtsBitsContainerVec)
   , m_trigT2MbtsBitsVec(const_cast<TrigT2MbtsBitsContainer_tlp1 &>( rhs ).m_trigT2MbtsBitsVec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigT2MbtsBitsContainer_tlp1 &modrhs = const_cast<TrigT2MbtsBitsContainer_tlp1 &>( rhs );
   modrhs.m_trigT2MbtsBitsContainerVec.clear();
   modrhs.m_trigT2MbtsBitsVec.clear();
}
TrigT2MbtsBitsContainer_tlp1::~TrigT2MbtsBitsContainer_tlp1() {
}
#endif // TrigT2MbtsBitsContainer_tlp1_cxx

#ifndef TrigT2MbtsBits_p1_cxx
#define TrigT2MbtsBits_p1_cxx
TrigT2MbtsBits_p1::TrigT2MbtsBits_p1() {
}
TrigT2MbtsBits_p1::TrigT2MbtsBits_p1(const TrigT2MbtsBits_p1 & rhs)
   : m_mbtsWord(const_cast<TrigT2MbtsBits_p1 &>( rhs ).m_mbtsWord)
   , m_triggerTimes(const_cast<TrigT2MbtsBits_p1 &>( rhs ).m_triggerTimes)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigT2MbtsBits_p1 &modrhs = const_cast<TrigT2MbtsBits_p1 &>( rhs );
   modrhs.m_triggerTimes.clear();
}
TrigT2MbtsBits_p1::~TrigT2MbtsBits_p1() {
}
#endif // TrigT2MbtsBits_p1_cxx

#ifndef IsoMuonFeatureContainer_tlp1_cxx
#define IsoMuonFeatureContainer_tlp1_cxx
IsoMuonFeatureContainer_tlp1::IsoMuonFeatureContainer_tlp1() {
}
IsoMuonFeatureContainer_tlp1::IsoMuonFeatureContainer_tlp1(const IsoMuonFeatureContainer_tlp1 & rhs)
   : m_isoMuonFeatureContainerVec(const_cast<IsoMuonFeatureContainer_tlp1 &>( rhs ).m_isoMuonFeatureContainerVec)
   , m_isoMuonFeatureVec(const_cast<IsoMuonFeatureContainer_tlp1 &>( rhs ).m_isoMuonFeatureVec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   IsoMuonFeatureContainer_tlp1 &modrhs = const_cast<IsoMuonFeatureContainer_tlp1 &>( rhs );
   modrhs.m_isoMuonFeatureContainerVec.clear();
   modrhs.m_isoMuonFeatureVec.clear();
}
IsoMuonFeatureContainer_tlp1::~IsoMuonFeatureContainer_tlp1() {
}
#endif // IsoMuonFeatureContainer_tlp1_cxx

#ifndef IsoMuonFeature_p1_cxx
#define IsoMuonFeature_p1_cxx
IsoMuonFeature_p1::IsoMuonFeature_p1() {
}
IsoMuonFeature_p1::IsoMuonFeature_p1(const IsoMuonFeature_p1 & rhs)
   : m_EtInnerConeEC(const_cast<IsoMuonFeature_p1 &>( rhs ).m_EtInnerConeEC)
   , m_EtOuterConeEC(const_cast<IsoMuonFeature_p1 &>( rhs ).m_EtOuterConeEC)
   , m_EtInnerConeHC(const_cast<IsoMuonFeature_p1 &>( rhs ).m_EtInnerConeHC)
   , m_EtOuterConeHC(const_cast<IsoMuonFeature_p1 &>( rhs ).m_EtOuterConeHC)
   , m_NTracksCone(const_cast<IsoMuonFeature_p1 &>( rhs ).m_NTracksCone)
   , m_SumPtTracksCone(const_cast<IsoMuonFeature_p1 &>( rhs ).m_SumPtTracksCone)
   , m_PtMuTracksCone(const_cast<IsoMuonFeature_p1 &>( rhs ).m_PtMuTracksCone)
   , m_LAr_w(const_cast<IsoMuonFeature_p1 &>( rhs ).m_LAr_w)
   , m_Tile_w(const_cast<IsoMuonFeature_p1 &>( rhs ).m_Tile_w)
   , m_RoiIdMu(const_cast<IsoMuonFeature_p1 &>( rhs ).m_RoiIdMu)
   , m_PtMu(const_cast<IsoMuonFeature_p1 &>( rhs ).m_PtMu)
   , m_EtaMu(const_cast<IsoMuonFeature_p1 &>( rhs ).m_EtaMu)
   , m_PhiMu(const_cast<IsoMuonFeature_p1 &>( rhs ).m_PhiMu)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
IsoMuonFeature_p1::~IsoMuonFeature_p1() {
}
#endif // IsoMuonFeature_p1_cxx

#ifndef MuonFeatureContainer_tlp2_cxx
#define MuonFeatureContainer_tlp2_cxx
MuonFeatureContainer_tlp2::MuonFeatureContainer_tlp2() {
}
MuonFeatureContainer_tlp2::MuonFeatureContainer_tlp2(const MuonFeatureContainer_tlp2 & rhs)
   : m_muonFeatureContainerVec(const_cast<MuonFeatureContainer_tlp2 &>( rhs ).m_muonFeatureContainerVec)
   , m_muonFeatureVec(const_cast<MuonFeatureContainer_tlp2 &>( rhs ).m_muonFeatureVec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MuonFeatureContainer_tlp2 &modrhs = const_cast<MuonFeatureContainer_tlp2 &>( rhs );
   modrhs.m_muonFeatureContainerVec.clear();
   modrhs.m_muonFeatureVec.clear();
}
MuonFeatureContainer_tlp2::~MuonFeatureContainer_tlp2() {
}
#endif // MuonFeatureContainer_tlp2_cxx

#ifndef MuonFeature_p2_cxx
#define MuonFeature_p2_cxx
MuonFeature_p2::MuonFeature_p2() {
}
MuonFeature_p2::MuonFeature_p2(const MuonFeature_p2 & rhs)
   : m_algoId(const_cast<MuonFeature_p2 &>( rhs ).m_algoId)
   , m_RoIId(const_cast<MuonFeature_p2 &>( rhs ).m_RoIId)
   , m_saddress(const_cast<MuonFeature_p2 &>( rhs ).m_saddress)
   , m_pt(const_cast<MuonFeature_p2 &>( rhs ).m_pt)
   , m_radius(const_cast<MuonFeature_p2 &>( rhs ).m_radius)
   , m_eta(const_cast<MuonFeature_p2 &>( rhs ).m_eta)
   , m_phi(const_cast<MuonFeature_p2 &>( rhs ).m_phi)
   , m_dir_phi(const_cast<MuonFeature_p2 &>( rhs ).m_dir_phi)
   , m_zeta(const_cast<MuonFeature_p2 &>( rhs ).m_zeta)
   , m_dir_zeta(const_cast<MuonFeature_p2 &>( rhs ).m_dir_zeta)
   , m_beta(const_cast<MuonFeature_p2 &>( rhs ).m_beta)
   , m_sp1_r(const_cast<MuonFeature_p2 &>( rhs ).m_sp1_r)
   , m_sp1_z(const_cast<MuonFeature_p2 &>( rhs ).m_sp1_z)
   , m_sp1_slope(const_cast<MuonFeature_p2 &>( rhs ).m_sp1_slope)
   , m_sp2_r(const_cast<MuonFeature_p2 &>( rhs ).m_sp2_r)
   , m_sp2_z(const_cast<MuonFeature_p2 &>( rhs ).m_sp2_z)
   , m_sp2_slope(const_cast<MuonFeature_p2 &>( rhs ).m_sp2_slope)
   , m_sp3_r(const_cast<MuonFeature_p2 &>( rhs ).m_sp3_r)
   , m_sp3_z(const_cast<MuonFeature_p2 &>( rhs ).m_sp3_z)
   , m_sp3_slope(const_cast<MuonFeature_p2 &>( rhs ).m_sp3_slope)
   , m_br_radius(const_cast<MuonFeature_p2 &>( rhs ).m_br_radius)
   , m_br_sagitta(const_cast<MuonFeature_p2 &>( rhs ).m_br_sagitta)
   , m_ec_alpha(const_cast<MuonFeature_p2 &>( rhs ).m_ec_alpha)
   , m_ec_beta(const_cast<MuonFeature_p2 &>( rhs ).m_ec_beta)
   , m_dq_var1(const_cast<MuonFeature_p2 &>( rhs ).m_dq_var1)
   , m_dq_var2(const_cast<MuonFeature_p2 &>( rhs ).m_dq_var2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
MuonFeature_p2::~MuonFeature_p2() {
}
#endif // MuonFeature_p2_cxx

#ifndef TrigEMClusterContainer_tlp1_cxx
#define TrigEMClusterContainer_tlp1_cxx
TrigEMClusterContainer_tlp1::TrigEMClusterContainer_tlp1() {
}
TrigEMClusterContainer_tlp1::TrigEMClusterContainer_tlp1(const TrigEMClusterContainer_tlp1 & rhs)
   : m_TrigEMClusterContainers(const_cast<TrigEMClusterContainer_tlp1 &>( rhs ).m_TrigEMClusterContainers)
   , m_EMCluster(const_cast<TrigEMClusterContainer_tlp1 &>( rhs ).m_EMCluster)
   , m_CaloCluster(const_cast<TrigEMClusterContainer_tlp1 &>( rhs ).m_CaloCluster)
   , m_EMCluster_p2(const_cast<TrigEMClusterContainer_tlp1 &>( rhs ).m_EMCluster_p2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigEMClusterContainer_tlp1 &modrhs = const_cast<TrigEMClusterContainer_tlp1 &>( rhs );
   modrhs.m_TrigEMClusterContainers.clear();
   modrhs.m_EMCluster.clear();
   modrhs.m_CaloCluster.clear();
   modrhs.m_EMCluster_p2.clear();
}
TrigEMClusterContainer_tlp1::~TrigEMClusterContainer_tlp1() {
}
#endif // TrigEMClusterContainer_tlp1_cxx

#ifndef TrigEMCluster_p1_cxx
#define TrigEMCluster_p1_cxx
TrigEMCluster_p1::TrigEMCluster_p1() {
}
TrigEMCluster_p1::TrigEMCluster_p1(const TrigEMCluster_p1 & rhs)
   : m_Energy(const_cast<TrigEMCluster_p1 &>( rhs ).m_Energy)
   , m_Et(const_cast<TrigEMCluster_p1 &>( rhs ).m_Et)
   , m_Eta(const_cast<TrigEMCluster_p1 &>( rhs ).m_Eta)
   , m_Phi(const_cast<TrigEMCluster_p1 &>( rhs ).m_Phi)
   , m_e237(const_cast<TrigEMCluster_p1 &>( rhs ).m_e237)
   , m_e277(const_cast<TrigEMCluster_p1 &>( rhs ).m_e277)
   , m_fracs1(const_cast<TrigEMCluster_p1 &>( rhs ).m_fracs1)
   , m_weta2(const_cast<TrigEMCluster_p1 &>( rhs ).m_weta2)
   , m_ehad1(const_cast<TrigEMCluster_p1 &>( rhs ).m_ehad1)
   , m_Eta1(const_cast<TrigEMCluster_p1 &>( rhs ).m_Eta1)
   , m_emaxs1(const_cast<TrigEMCluster_p1 &>( rhs ).m_emaxs1)
   , m_e2tsts1(const_cast<TrigEMCluster_p1 &>( rhs ).m_e2tsts1)
   , m_trigCaloCluster(const_cast<TrigEMCluster_p1 &>( rhs ).m_trigCaloCluster)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<25;i++) m_EnergyS[i] = rhs.m_EnergyS[i];
}
TrigEMCluster_p1::~TrigEMCluster_p1() {
}
#endif // TrigEMCluster_p1_cxx

#ifndef TrigCaloCluster_p1_cxx
#define TrigCaloCluster_p1_cxx
TrigCaloCluster_p1::TrigCaloCluster_p1() {
}
TrigCaloCluster_p1::TrigCaloCluster_p1(const TrigCaloCluster_p1 & rhs)
   : m_rawEnergy(const_cast<TrigCaloCluster_p1 &>( rhs ).m_rawEnergy)
   , m_rawEt(const_cast<TrigCaloCluster_p1 &>( rhs ).m_rawEt)
   , m_rawEta(const_cast<TrigCaloCluster_p1 &>( rhs ).m_rawEta)
   , m_rawPhi(const_cast<TrigCaloCluster_p1 &>( rhs ).m_rawPhi)
   , m_roiWord(const_cast<TrigCaloCluster_p1 &>( rhs ).m_roiWord)
   , m_numberUsedCells(const_cast<TrigCaloCluster_p1 &>( rhs ).m_numberUsedCells)
   , m_quality(const_cast<TrigCaloCluster_p1 &>( rhs ).m_quality)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<25;i++) m_rawEnergyS[i] = rhs.m_rawEnergyS[i];
}
TrigCaloCluster_p1::~TrigCaloCluster_p1() {
}
#endif // TrigCaloCluster_p1_cxx

#ifndef TrigEMCluster_p2_cxx
#define TrigEMCluster_p2_cxx
TrigEMCluster_p2::TrigEMCluster_p2() {
}
TrigEMCluster_p2::TrigEMCluster_p2(const TrigEMCluster_p2 & rhs)
   : m_Energy(const_cast<TrigEMCluster_p2 &>( rhs ).m_Energy)
   , m_Et(const_cast<TrigEMCluster_p2 &>( rhs ).m_Et)
   , m_Eta(const_cast<TrigEMCluster_p2 &>( rhs ).m_Eta)
   , m_Phi(const_cast<TrigEMCluster_p2 &>( rhs ).m_Phi)
   , m_e237(const_cast<TrigEMCluster_p2 &>( rhs ).m_e237)
   , m_e277(const_cast<TrigEMCluster_p2 &>( rhs ).m_e277)
   , m_fracs1(const_cast<TrigEMCluster_p2 &>( rhs ).m_fracs1)
   , m_weta2(const_cast<TrigEMCluster_p2 &>( rhs ).m_weta2)
   , m_ehad1(const_cast<TrigEMCluster_p2 &>( rhs ).m_ehad1)
   , m_Eta1(const_cast<TrigEMCluster_p2 &>( rhs ).m_Eta1)
   , m_emaxs1(const_cast<TrigEMCluster_p2 &>( rhs ).m_emaxs1)
   , m_e2tsts1(const_cast<TrigEMCluster_p2 &>( rhs ).m_e2tsts1)
   , m_trigCaloCluster(const_cast<TrigEMCluster_p2 &>( rhs ).m_trigCaloCluster)
   , m_rings(const_cast<TrigEMCluster_p2 &>( rhs ).m_rings)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<25;i++) m_EnergyS[i] = rhs.m_EnergyS[i];
}
TrigEMCluster_p2::~TrigEMCluster_p2() {
}
#endif // TrigEMCluster_p2_cxx

#ifndef TrackRecord_p1_cxx
#define TrackRecord_p1_cxx
TrackRecord_p1::TrackRecord_p1() {
}
TrackRecord_p1::TrackRecord_p1(const TrackRecord_p1 & rhs)
   : m_PDG_code(const_cast<TrackRecord_p1 &>( rhs ).m_PDG_code)
   , m_energy(const_cast<TrackRecord_p1 &>( rhs ).m_energy)
   , m_momentumX(const_cast<TrackRecord_p1 &>( rhs ).m_momentumX)
   , m_momentumY(const_cast<TrackRecord_p1 &>( rhs ).m_momentumY)
   , m_momentumZ(const_cast<TrackRecord_p1 &>( rhs ).m_momentumZ)
   , m_positionX(const_cast<TrackRecord_p1 &>( rhs ).m_positionX)
   , m_positionY(const_cast<TrackRecord_p1 &>( rhs ).m_positionY)
   , m_positionZ(const_cast<TrackRecord_p1 &>( rhs ).m_positionZ)
   , m_time(const_cast<TrackRecord_p1 &>( rhs ).m_time)
   , m_barCode(const_cast<TrackRecord_p1 &>( rhs ).m_barCode)
   , m_volName(const_cast<TrackRecord_p1 &>( rhs ).m_volName)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrackRecord_p1 &modrhs = const_cast<TrackRecord_p1 &>( rhs );
   modrhs.m_volName.clear();
}
TrackRecord_p1::~TrackRecord_p1() {
}
#endif // TrackRecord_p1_cxx

#ifndef TrackRecordCollection_p2_cxx
#define TrackRecordCollection_p2_cxx
TrackRecordCollection_p2::TrackRecordCollection_p2() {
}
TrackRecordCollection_p2::TrackRecordCollection_p2(const TrackRecordCollection_p2 & rhs)
   : m_cont(const_cast<TrackRecordCollection_p2 &>( rhs ).m_cont)
   , m_name(const_cast<TrackRecordCollection_p2 &>( rhs ).m_name)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrackRecordCollection_p2 &modrhs = const_cast<TrackRecordCollection_p2 &>( rhs );
   modrhs.m_cont.clear();
   modrhs.m_name.clear();
}
TrackRecordCollection_p2::~TrackRecordCollection_p2() {
}
#endif // TrackRecordCollection_p2_cxx

#ifndef TrigTauClusterContainer_tlp1_cxx
#define TrigTauClusterContainer_tlp1_cxx
TrigTauClusterContainer_tlp1::TrigTauClusterContainer_tlp1() {
}
TrigTauClusterContainer_tlp1::TrigTauClusterContainer_tlp1(const TrigTauClusterContainer_tlp1 & rhs)
   : m_TrigTauClusterContainers(const_cast<TrigTauClusterContainer_tlp1 &>( rhs ).m_TrigTauClusterContainers)
   , m_TauCluster(const_cast<TrigTauClusterContainer_tlp1 &>( rhs ).m_TauCluster)
   , m_CaloCluster(const_cast<TrigTauClusterContainer_tlp1 &>( rhs ).m_CaloCluster)
   , m_TauCluster_p2(const_cast<TrigTauClusterContainer_tlp1 &>( rhs ).m_TauCluster_p2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTauClusterContainer_tlp1 &modrhs = const_cast<TrigTauClusterContainer_tlp1 &>( rhs );
   modrhs.m_TrigTauClusterContainers.clear();
   modrhs.m_TauCluster.clear();
   modrhs.m_CaloCluster.clear();
   modrhs.m_TauCluster_p2.clear();
}
TrigTauClusterContainer_tlp1::~TrigTauClusterContainer_tlp1() {
}
#endif // TrigTauClusterContainer_tlp1_cxx

#ifndef TrigTauCluster_p1_cxx
#define TrigTauCluster_p1_cxx
TrigTauCluster_p1::TrigTauCluster_p1() {
}
TrigTauCluster_p1::TrigTauCluster_p1(const TrigTauCluster_p1 & rhs)
   : m_EMenergy(const_cast<TrigTauCluster_p1 &>( rhs ).m_EMenergy)
   , m_HADenergy(const_cast<TrigTauCluster_p1 &>( rhs ).m_HADenergy)
   , m_eEMCalib(const_cast<TrigTauCluster_p1 &>( rhs ).m_eEMCalib)
   , m_eCalib(const_cast<TrigTauCluster_p1 &>( rhs ).m_eCalib)
   , m_Eta(const_cast<TrigTauCluster_p1 &>( rhs ).m_Eta)
   , m_Phi(const_cast<TrigTauCluster_p1 &>( rhs ).m_Phi)
   , m_IsoFrac(const_cast<TrigTauCluster_p1 &>( rhs ).m_IsoFrac)
   , m_numStripCells(const_cast<TrigTauCluster_p1 &>( rhs ).m_numStripCells)
   , m_stripWidth(const_cast<TrigTauCluster_p1 &>( rhs ).m_stripWidth)
   , m_trigCaloCluster(const_cast<TrigTauCluster_p1 &>( rhs ).m_trigCaloCluster)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<25;i++) m_EnergyS[i] = rhs.m_EnergyS[i];
   for (Int_t i=0;i<4;i++) m_EMRadius[i] = rhs.m_EMRadius[i];
   for (Int_t i=0;i<4;i++) m_EMenergyWidth[i] = rhs.m_EMenergyWidth[i];
   for (Int_t i=0;i<3;i++) m_HADenergyWidth[i] = rhs.m_HADenergyWidth[i];
   for (Int_t i=0;i<4;i++) m_EMenergyNor[i] = rhs.m_EMenergyNor[i];
   for (Int_t i=0;i<4;i++) m_EMenergyWid[i] = rhs.m_EMenergyWid[i];
   for (Int_t i=0;i<4;i++) m_EMenergyNar[i] = rhs.m_EMenergyNar[i];
   for (Int_t i=0;i<3;i++) m_HADenergyNor[i] = rhs.m_HADenergyNor[i];
   for (Int_t i=0;i<3;i++) m_HADenergyWid[i] = rhs.m_HADenergyWid[i];
   for (Int_t i=0;i<3;i++) m_HADenergyNar[i] = rhs.m_HADenergyNar[i];
}
TrigTauCluster_p1::~TrigTauCluster_p1() {
}
#endif // TrigTauCluster_p1_cxx

#ifndef TrigTauCluster_p2_cxx
#define TrigTauCluster_p2_cxx
TrigTauCluster_p2::TrigTauCluster_p2() {
}
TrigTauCluster_p2::TrigTauCluster_p2(const TrigTauCluster_p2 & rhs)
   : m_EMenergy(const_cast<TrigTauCluster_p2 &>( rhs ).m_EMenergy)
   , m_HADenergy(const_cast<TrigTauCluster_p2 &>( rhs ).m_HADenergy)
   , m_eCalib(const_cast<TrigTauCluster_p2 &>( rhs ).m_eCalib)
   , m_EMRadius2(const_cast<TrigTauCluster_p2 &>( rhs ).m_EMRadius2)
   , m_IsoFrac(const_cast<TrigTauCluster_p2 &>( rhs ).m_IsoFrac)
   , m_numStripCells(const_cast<TrigTauCluster_p2 &>( rhs ).m_numStripCells)
   , m_stripWidth(const_cast<TrigTauCluster_p2 &>( rhs ).m_stripWidth)
   , m_stripWidthOffline(const_cast<TrigTauCluster_p2 &>( rhs ).m_stripWidthOffline)
   , m_valid(const_cast<TrigTauCluster_p2 &>( rhs ).m_valid)
   , m_details(const_cast<TrigTauCluster_p2 &>( rhs ).m_details)
   , m_trigCaloCluster(const_cast<TrigTauCluster_p2 &>( rhs ).m_trigCaloCluster)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigTauCluster_p2::~TrigTauCluster_p2() {
}
#endif // TrigTauCluster_p2_cxx

#ifndef TrigMissingETContainer_tlp1_cxx
#define TrigMissingETContainer_tlp1_cxx
TrigMissingETContainer_tlp1::TrigMissingETContainer_tlp1() {
}
TrigMissingETContainer_tlp1::TrigMissingETContainer_tlp1(const TrigMissingETContainer_tlp1 & rhs)
   : m_TrigMissingETContainer_p1(const_cast<TrigMissingETContainer_tlp1 &>( rhs ).m_TrigMissingETContainer_p1)
   , m_TrigMissingET_p1(const_cast<TrigMissingETContainer_tlp1 &>( rhs ).m_TrigMissingET_p1)
   , m_TrigMissingET_p2(const_cast<TrigMissingETContainer_tlp1 &>( rhs ).m_TrigMissingET_p2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigMissingETContainer_tlp1 &modrhs = const_cast<TrigMissingETContainer_tlp1 &>( rhs );
   modrhs.m_TrigMissingETContainer_p1.clear();
   modrhs.m_TrigMissingET_p1.clear();
   modrhs.m_TrigMissingET_p2.clear();
}
TrigMissingETContainer_tlp1::~TrigMissingETContainer_tlp1() {
}
#endif // TrigMissingETContainer_tlp1_cxx

#ifndef TrigMissingET_p1_cxx
#define TrigMissingET_p1_cxx
TrigMissingET_p1::TrigMissingET_p1() {
}
TrigMissingET_p1::TrigMissingET_p1(const TrigMissingET_p1 & rhs)
   : m_ex(const_cast<TrigMissingET_p1 &>( rhs ).m_ex)
   , m_ey(const_cast<TrigMissingET_p1 &>( rhs ).m_ey)
   , m_sum_et(const_cast<TrigMissingET_p1 &>( rhs ).m_sum_et)
   , m_roiWord(const_cast<TrigMissingET_p1 &>( rhs ).m_roiWord)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigMissingET_p1::~TrigMissingET_p1() {
}
#endif // TrigMissingET_p1_cxx

#ifndef TrigMissingET_p2_cxx
#define TrigMissingET_p2_cxx
TrigMissingET_p2::TrigMissingET_p2() {
}
TrigMissingET_p2::TrigMissingET_p2(const TrigMissingET_p2 & rhs)
   : m_ex(const_cast<TrigMissingET_p2 &>( rhs ).m_ex)
   , m_ey(const_cast<TrigMissingET_p2 &>( rhs ).m_ey)
   , m_ez(const_cast<TrigMissingET_p2 &>( rhs ).m_ez)
   , m_sum_et(const_cast<TrigMissingET_p2 &>( rhs ).m_sum_et)
   , m_sum_e(const_cast<TrigMissingET_p2 &>( rhs ).m_sum_e)
   , m_flag(const_cast<TrigMissingET_p2 &>( rhs ).m_flag)
   , m_roiWord(const_cast<TrigMissingET_p2 &>( rhs ).m_roiWord)
   , m_comp_number(const_cast<TrigMissingET_p2 &>( rhs ).m_comp_number)
   , m_c_name(const_cast<TrigMissingET_p2 &>( rhs ).m_c_name)
   , m_c_status(const_cast<TrigMissingET_p2 &>( rhs ).m_c_status)
   , m_c_ex(const_cast<TrigMissingET_p2 &>( rhs ).m_c_ex)
   , m_c_ey(const_cast<TrigMissingET_p2 &>( rhs ).m_c_ey)
   , m_c_ez(const_cast<TrigMissingET_p2 &>( rhs ).m_c_ez)
   , m_c_sumEt(const_cast<TrigMissingET_p2 &>( rhs ).m_c_sumEt)
   , m_c_sumE(const_cast<TrigMissingET_p2 &>( rhs ).m_c_sumE)
   , m_c_calib0(const_cast<TrigMissingET_p2 &>( rhs ).m_c_calib0)
   , m_c_calib1(const_cast<TrigMissingET_p2 &>( rhs ).m_c_calib1)
   , m_c_sumOfSigns(const_cast<TrigMissingET_p2 &>( rhs ).m_c_sumOfSigns)
   , m_c_usedChannels(const_cast<TrigMissingET_p2 &>( rhs ).m_c_usedChannels)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigMissingET_p2 &modrhs = const_cast<TrigMissingET_p2 &>( rhs );
   modrhs.m_c_name.clear();
   modrhs.m_c_status.clear();
   modrhs.m_c_ex.clear();
   modrhs.m_c_ey.clear();
   modrhs.m_c_ez.clear();
   modrhs.m_c_sumEt.clear();
   modrhs.m_c_sumE.clear();
   modrhs.m_c_calib0.clear();
   modrhs.m_c_calib1.clear();
   modrhs.m_c_sumOfSigns.clear();
   modrhs.m_c_usedChannels.clear();
}
TrigMissingET_p2::~TrigMissingET_p2() {
}
#endif // TrigMissingET_p2_cxx

#ifndef TrigL2BjetContainer_tlp2_cxx
#define TrigL2BjetContainer_tlp2_cxx
TrigL2BjetContainer_tlp2::TrigL2BjetContainer_tlp2() {
}
TrigL2BjetContainer_tlp2::TrigL2BjetContainer_tlp2(const TrigL2BjetContainer_tlp2 & rhs)
   : m_TrigL2BjetContainers(const_cast<TrigL2BjetContainer_tlp2 &>( rhs ).m_TrigL2BjetContainers)
   , m_L2Bjet(const_cast<TrigL2BjetContainer_tlp2 &>( rhs ).m_L2Bjet)
   , m_p4PtEtaPhiM(const_cast<TrigL2BjetContainer_tlp2 &>( rhs ).m_p4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigL2BjetContainer_tlp2 &modrhs = const_cast<TrigL2BjetContainer_tlp2 &>( rhs );
   modrhs.m_TrigL2BjetContainers.clear();
   modrhs.m_L2Bjet.clear();
   modrhs.m_p4PtEtaPhiM.clear();
}
TrigL2BjetContainer_tlp2::~TrigL2BjetContainer_tlp2() {
}
#endif // TrigL2BjetContainer_tlp2_cxx

#ifndef TrigL2Bjet_p2_cxx
#define TrigL2Bjet_p2_cxx
TrigL2Bjet_p2::TrigL2Bjet_p2() {
}
TrigL2Bjet_p2::TrigL2Bjet_p2(const TrigL2Bjet_p2 & rhs)
   : m_valid(const_cast<TrigL2Bjet_p2 &>( rhs ).m_valid)
   , m_roiID(const_cast<TrigL2Bjet_p2 &>( rhs ).m_roiID)
   , m_prmVtx(const_cast<TrigL2Bjet_p2 &>( rhs ).m_prmVtx)
   , m_xcomb(const_cast<TrigL2Bjet_p2 &>( rhs ).m_xcomb)
   , m_xIP1d(const_cast<TrigL2Bjet_p2 &>( rhs ).m_xIP1d)
   , m_xIP2d(const_cast<TrigL2Bjet_p2 &>( rhs ).m_xIP2d)
   , m_xIP3d(const_cast<TrigL2Bjet_p2 &>( rhs ).m_xIP3d)
   , m_xChi2(const_cast<TrigL2Bjet_p2 &>( rhs ).m_xChi2)
   , m_xSv(const_cast<TrigL2Bjet_p2 &>( rhs ).m_xSv)
   , m_xmvtx(const_cast<TrigL2Bjet_p2 &>( rhs ).m_xmvtx)
   , m_xevtx(const_cast<TrigL2Bjet_p2 &>( rhs ).m_xevtx)
   , m_xnvtx(const_cast<TrigL2Bjet_p2 &>( rhs ).m_xnvtx)
   , m_p4PtEtaPhiM(const_cast<TrigL2Bjet_p2 &>( rhs ).m_p4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigL2Bjet_p2::~TrigL2Bjet_p2() {
}
#endif // TrigL2Bjet_p2_cxx

#ifndef CaloClusterMomentContainer_p1__ClusterMoment_p_cxx
#define CaloClusterMomentContainer_p1__ClusterMoment_p_cxx
CaloClusterMomentContainer_p1::ClusterMoment_p::ClusterMoment_p() {
}
CaloClusterMomentContainer_p1::ClusterMoment_p::ClusterMoment_p(const ClusterMoment_p & rhs)
   : key(const_cast<ClusterMoment_p &>( rhs ).key)
   , value(const_cast<ClusterMoment_p &>( rhs ).value)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
CaloClusterMomentContainer_p1::ClusterMoment_p::~ClusterMoment_p() {
}
#endif // CaloClusterMomentContainer_p1__ClusterMoment_p_cxx

#ifndef CaloClusterMomentContainer_p1_cxx
#define CaloClusterMomentContainer_p1_cxx
CaloClusterMomentContainer_p1::CaloClusterMomentContainer_p1() {
}
CaloClusterMomentContainer_p1::CaloClusterMomentContainer_p1(const CaloClusterMomentContainer_p1 & rhs)
   : m_store(const_cast<CaloClusterMomentContainer_p1 &>( rhs ).m_store)
   , m_nMoments(const_cast<CaloClusterMomentContainer_p1 &>( rhs ).m_nMoments)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CaloClusterMomentContainer_p1 &modrhs = const_cast<CaloClusterMomentContainer_p1 &>( rhs );
   modrhs.m_store.clear();
}
CaloClusterMomentContainer_p1::~CaloClusterMomentContainer_p1() {
}
#endif // CaloClusterMomentContainer_p1_cxx

#ifndef CaloShowerContainer_p2_cxx
#define CaloShowerContainer_p2_cxx
CaloShowerContainer_p2::CaloShowerContainer_p2() {
}
CaloShowerContainer_p2::CaloShowerContainer_p2(const CaloShowerContainer_p2 & rhs)
   : m_momentContainer(const_cast<CaloShowerContainer_p2 &>( rhs ).m_momentContainer)
   , m_samplingDataContainer(const_cast<CaloShowerContainer_p2 &>( rhs ).m_samplingDataContainer)
   , m_nClusters(const_cast<CaloShowerContainer_p2 &>( rhs ).m_nClusters)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
CaloShowerContainer_p2::~CaloShowerContainer_p2() {
}
#endif // CaloShowerContainer_p2_cxx

#ifndef CaloSamplingDataContainer_p1_cxx
#define CaloSamplingDataContainer_p1_cxx
CaloSamplingDataContainer_p1::CaloSamplingDataContainer_p1() {
}
CaloSamplingDataContainer_p1::CaloSamplingDataContainer_p1(const CaloSamplingDataContainer_p1 & rhs)
   : m_varTypePatterns(const_cast<CaloSamplingDataContainer_p1 &>( rhs ).m_varTypePatterns)
   , m_dataStore(const_cast<CaloSamplingDataContainer_p1 &>( rhs ).m_dataStore)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CaloSamplingDataContainer_p1 &modrhs = const_cast<CaloSamplingDataContainer_p1 &>( rhs );
   modrhs.m_varTypePatterns.clear();
   modrhs.m_dataStore.clear();
}
CaloSamplingDataContainer_p1::~CaloSamplingDataContainer_p1() {
}
#endif // CaloSamplingDataContainer_p1_cxx

#ifndef TileTrackMuFeatureContainer_tlp1_cxx
#define TileTrackMuFeatureContainer_tlp1_cxx
TileTrackMuFeatureContainer_tlp1::TileTrackMuFeatureContainer_tlp1() {
}
TileTrackMuFeatureContainer_tlp1::TileTrackMuFeatureContainer_tlp1(const TileTrackMuFeatureContainer_tlp1 & rhs)
   : m_TileTrackMuFeatureContainerVec(const_cast<TileTrackMuFeatureContainer_tlp1 &>( rhs ).m_TileTrackMuFeatureContainerVec)
   , m_TileTrackMuFeatureVec(const_cast<TileTrackMuFeatureContainer_tlp1 &>( rhs ).m_TileTrackMuFeatureVec)
   , m_TileTrackMuFeatureVec_p2(const_cast<TileTrackMuFeatureContainer_tlp1 &>( rhs ).m_TileTrackMuFeatureVec_p2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TileTrackMuFeatureContainer_tlp1 &modrhs = const_cast<TileTrackMuFeatureContainer_tlp1 &>( rhs );
   modrhs.m_TileTrackMuFeatureContainerVec.clear();
   modrhs.m_TileTrackMuFeatureVec.clear();
   modrhs.m_TileTrackMuFeatureVec_p2.clear();
}
TileTrackMuFeatureContainer_tlp1::~TileTrackMuFeatureContainer_tlp1() {
}
#endif // TileTrackMuFeatureContainer_tlp1_cxx

#ifndef TileTrackMuFeature_p1_cxx
#define TileTrackMuFeature_p1_cxx
TileTrackMuFeature_p1::TileTrackMuFeature_p1() {
}
TileTrackMuFeature_p1::TileTrackMuFeature_p1(const TileTrackMuFeature_p1 & rhs)
   : m_PtTR_Trk(const_cast<TileTrackMuFeature_p1 &>( rhs ).m_PtTR_Trk)
   , m_EtaTR_Trk(const_cast<TileTrackMuFeature_p1 &>( rhs ).m_EtaTR_Trk)
   , m_PhiTR_Trk(const_cast<TileTrackMuFeature_p1 &>( rhs ).m_PhiTR_Trk)
   , m_Typ_IDTrk(const_cast<TileTrackMuFeature_p1 &>( rhs ).m_Typ_IDTrk)
   , m_TileMu(const_cast<TileTrackMuFeature_p1 &>( rhs ).m_TileMu)
   , m_Track(const_cast<TileTrackMuFeature_p1 &>( rhs ).m_Track)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TileTrackMuFeature_p1::~TileTrackMuFeature_p1() {
}
#endif // TileTrackMuFeature_p1_cxx

#ifndef TileTrackMuFeature_p2_cxx
#define TileTrackMuFeature_p2_cxx
TileTrackMuFeature_p2::TileTrackMuFeature_p2() {
}
TileTrackMuFeature_p2::TileTrackMuFeature_p2(const TileTrackMuFeature_p2 & rhs)
   : m_PtTR_Trk(const_cast<TileTrackMuFeature_p2 &>( rhs ).m_PtTR_Trk)
   , m_EtaTR_Trk(const_cast<TileTrackMuFeature_p2 &>( rhs ).m_EtaTR_Trk)
   , m_PhiTR_Trk(const_cast<TileTrackMuFeature_p2 &>( rhs ).m_PhiTR_Trk)
   , m_Typ_IDTrk(const_cast<TileTrackMuFeature_p2 &>( rhs ).m_Typ_IDTrk)
   , m_TileMu(const_cast<TileTrackMuFeature_p2 &>( rhs ).m_TileMu)
   , m_Track(const_cast<TileTrackMuFeature_p2 &>( rhs ).m_Track)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TileTrackMuFeature_p2::~TileTrackMuFeature_p2() {
}
#endif // TileTrackMuFeature_p2_cxx

#ifndef TrigPhotonContainer_tlp2_cxx
#define TrigPhotonContainer_tlp2_cxx
TrigPhotonContainer_tlp2::TrigPhotonContainer_tlp2() {
}
TrigPhotonContainer_tlp2::TrigPhotonContainer_tlp2(const TrigPhotonContainer_tlp2 & rhs)
   : m_TrigPhotonContainers(const_cast<TrigPhotonContainer_tlp2 &>( rhs ).m_TrigPhotonContainers)
   , m_Photon(const_cast<TrigPhotonContainer_tlp2 &>( rhs ).m_Photon)
   , m_P4PtEtaPhiM(const_cast<TrigPhotonContainer_tlp2 &>( rhs ).m_P4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigPhotonContainer_tlp2 &modrhs = const_cast<TrigPhotonContainer_tlp2 &>( rhs );
   modrhs.m_TrigPhotonContainers.clear();
   modrhs.m_Photon.clear();
   modrhs.m_P4PtEtaPhiM.clear();
}
TrigPhotonContainer_tlp2::~TrigPhotonContainer_tlp2() {
}
#endif // TrigPhotonContainer_tlp2_cxx

#ifndef TrigPhoton_p2_cxx
#define TrigPhoton_p2_cxx
TrigPhoton_p2::TrigPhoton_p2() {
}
TrigPhoton_p2::TrigPhoton_p2(const TrigPhoton_p2 & rhs)
   : m_roiID(const_cast<TrigPhoton_p2 &>( rhs ).m_roiID)
   , m_HadEt(const_cast<TrigPhoton_p2 &>( rhs ).m_HadEt)
   , m_energyRatio(const_cast<TrigPhoton_p2 &>( rhs ).m_energyRatio)
   , m_rCore(const_cast<TrigPhoton_p2 &>( rhs ).m_rCore)
   , m_dPhi(const_cast<TrigPhoton_p2 &>( rhs ).m_dPhi)
   , m_dEta(const_cast<TrigPhoton_p2 &>( rhs ).m_dEta)
   , m_valid(const_cast<TrigPhoton_p2 &>( rhs ).m_valid)
   , m_cluster(const_cast<TrigPhoton_p2 &>( rhs ).m_cluster)
   , m_p4PtEtaPhiM(const_cast<TrigPhoton_p2 &>( rhs ).m_p4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigPhoton_p2::~TrigPhoton_p2() {
}
#endif // TrigPhoton_p2_cxx

#ifndef TileMuFeatureContainer_tlp1_cxx
#define TileMuFeatureContainer_tlp1_cxx
TileMuFeatureContainer_tlp1::TileMuFeatureContainer_tlp1() {
}
TileMuFeatureContainer_tlp1::TileMuFeatureContainer_tlp1(const TileMuFeatureContainer_tlp1 & rhs)
   : m_TileMuFeatureContainerVec(const_cast<TileMuFeatureContainer_tlp1 &>( rhs ).m_TileMuFeatureContainerVec)
   , m_TileMuFeatureVec(const_cast<TileMuFeatureContainer_tlp1 &>( rhs ).m_TileMuFeatureVec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TileMuFeatureContainer_tlp1 &modrhs = const_cast<TileMuFeatureContainer_tlp1 &>( rhs );
   modrhs.m_TileMuFeatureContainerVec.clear();
   modrhs.m_TileMuFeatureVec.clear();
}
TileMuFeatureContainer_tlp1::~TileMuFeatureContainer_tlp1() {
}
#endif // TileMuFeatureContainer_tlp1_cxx

#ifndef TileMuFeature_p1_cxx
#define TileMuFeature_p1_cxx
TileMuFeature_p1::TileMuFeature_p1() {
}
TileMuFeature_p1::TileMuFeature_p1(const TileMuFeature_p1 & rhs)
   : m_eta(const_cast<TileMuFeature_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<TileMuFeature_p1 &>( rhs ).m_phi)
   , m_energy_deposited(const_cast<TileMuFeature_p1 &>( rhs ).m_energy_deposited)
   , m_quality_factor(const_cast<TileMuFeature_p1 &>( rhs ).m_quality_factor)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TileMuFeature_p1 &modrhs = const_cast<TileMuFeature_p1 &>( rhs );
   modrhs.m_energy_deposited.clear();
}
TileMuFeature_p1::~TileMuFeature_p1() {
}
#endif // TileMuFeature_p1_cxx

#ifndef TrigVertexCollection_tlp1_cxx
#define TrigVertexCollection_tlp1_cxx
TrigVertexCollection_tlp1::TrigVertexCollection_tlp1() {
}
TrigVertexCollection_tlp1::TrigVertexCollection_tlp1(const TrigVertexCollection_tlp1 & rhs)
   : m_TrigVertexCollection(const_cast<TrigVertexCollection_tlp1 &>( rhs ).m_TrigVertexCollection)
   , m_Vertex(const_cast<TrigVertexCollection_tlp1 &>( rhs ).m_Vertex)
   , m_Track(const_cast<TrigVertexCollection_tlp1 &>( rhs ).m_Track)
   , m_TrackFitPar(const_cast<TrigVertexCollection_tlp1 &>( rhs ).m_TrackFitPar)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigVertexCollection_tlp1 &modrhs = const_cast<TrigVertexCollection_tlp1 &>( rhs );
   modrhs.m_TrigVertexCollection.clear();
   modrhs.m_Vertex.clear();
   modrhs.m_Track.clear();
   modrhs.m_TrackFitPar.clear();
}
TrigVertexCollection_tlp1::~TrigVertexCollection_tlp1() {
}
#endif // TrigVertexCollection_tlp1_cxx

#ifndef TrigVertex_p1_cxx
#define TrigVertex_p1_cxx
TrigVertex_p1::TrigVertex_p1() {
}
TrigVertex_p1::TrigVertex_p1(const TrigVertex_p1 & rhs)
   : m_x(const_cast<TrigVertex_p1 &>( rhs ).m_x)
   , m_y(const_cast<TrigVertex_p1 &>( rhs ).m_y)
   , m_z(const_cast<TrigVertex_p1 &>( rhs ).m_z)
   , m_mass(const_cast<TrigVertex_p1 &>( rhs ).m_mass)
   , m_massVar(const_cast<TrigVertex_p1 &>( rhs ).m_massVar)
   , m_energyFraction(const_cast<TrigVertex_p1 &>( rhs ).m_energyFraction)
   , m_nTwoTracksSecVtx(const_cast<TrigVertex_p1 &>( rhs ).m_nTwoTracksSecVtx)
   , m_chiSquared(const_cast<TrigVertex_p1 &>( rhs ).m_chiSquared)
   , m_nDOF(const_cast<TrigVertex_p1 &>( rhs ).m_nDOF)
   , m_tracks(const_cast<TrigVertex_p1 &>( rhs ).m_tracks)
   , m_algId(const_cast<TrigVertex_p1 &>( rhs ).m_algId)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<6;i++) m_cov[i] = rhs.m_cov[i];
   TrigVertex_p1 &modrhs = const_cast<TrigVertex_p1 &>( rhs );
   modrhs.m_tracks.clear();
}
TrigVertex_p1::~TrigVertex_p1() {
}
#endif // TrigVertex_p1_cxx

#ifndef TrigInDetTrackCollection_tlp1_cxx
#define TrigInDetTrackCollection_tlp1_cxx
TrigInDetTrackCollection_tlp1::TrigInDetTrackCollection_tlp1() {
}
TrigInDetTrackCollection_tlp1::TrigInDetTrackCollection_tlp1(const TrigInDetTrackCollection_tlp1 & rhs)
   : m_trigInDetTrackCollections(const_cast<TrigInDetTrackCollection_tlp1 &>( rhs ).m_trigInDetTrackCollections)
   , m_trigInDetTracks(const_cast<TrigInDetTrackCollection_tlp1 &>( rhs ).m_trigInDetTracks)
   , m_trigInDetTrackFitPars(const_cast<TrigInDetTrackCollection_tlp1 &>( rhs ).m_trigInDetTrackFitPars)
   , m_trigInDetTrackFitPars_p2(const_cast<TrigInDetTrackCollection_tlp1 &>( rhs ).m_trigInDetTrackFitPars_p2)
   , m_trigInDetTracks_p2(const_cast<TrigInDetTrackCollection_tlp1 &>( rhs ).m_trigInDetTracks_p2)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigInDetTrackCollection_tlp1 &modrhs = const_cast<TrigInDetTrackCollection_tlp1 &>( rhs );
   modrhs.m_trigInDetTrackCollections.clear();
   modrhs.m_trigInDetTracks.clear();
   modrhs.m_trigInDetTrackFitPars.clear();
   modrhs.m_trigInDetTrackFitPars_p2.clear();
   modrhs.m_trigInDetTracks_p2.clear();
}
TrigInDetTrackCollection_tlp1::~TrigInDetTrackCollection_tlp1() {
}
#endif // TrigInDetTrackCollection_tlp1_cxx

#ifndef TrigInDetTrackFitPar_p2_cxx
#define TrigInDetTrackFitPar_p2_cxx
TrigInDetTrackFitPar_p2::TrigInDetTrackFitPar_p2() {
}
TrigInDetTrackFitPar_p2::TrigInDetTrackFitPar_p2(const TrigInDetTrackFitPar_p2 & rhs)
   : m_a0(const_cast<TrigInDetTrackFitPar_p2 &>( rhs ).m_a0)
   , m_phi0(const_cast<TrigInDetTrackFitPar_p2 &>( rhs ).m_phi0)
   , m_z0(const_cast<TrigInDetTrackFitPar_p2 &>( rhs ).m_z0)
   , m_eta(const_cast<TrigInDetTrackFitPar_p2 &>( rhs ).m_eta)
   , m_pT(const_cast<TrigInDetTrackFitPar_p2 &>( rhs ).m_pT)
   , m_surfaceType(const_cast<TrigInDetTrackFitPar_p2 &>( rhs ).m_surfaceType)
   , m_surfaceCoordinate(const_cast<TrigInDetTrackFitPar_p2 &>( rhs ).m_surfaceCoordinate)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<15;i++) m_cov[i] = rhs.m_cov[i];
}
TrigInDetTrackFitPar_p2::~TrigInDetTrackFitPar_p2() {
}
#endif // TrigInDetTrackFitPar_p2_cxx

#ifndef TrigInDetTrack_p2_cxx
#define TrigInDetTrack_p2_cxx
TrigInDetTrack_p2::TrigInDetTrack_p2() {
}
TrigInDetTrack_p2::TrigInDetTrack_p2(const TrigInDetTrack_p2 & rhs)
   : m_algId(const_cast<TrigInDetTrack_p2 &>( rhs ).m_algId)
   , m_param(const_cast<TrigInDetTrack_p2 &>( rhs ).m_param)
   , m_endParam(const_cast<TrigInDetTrack_p2 &>( rhs ).m_endParam)
   , m_chi2(const_cast<TrigInDetTrack_p2 &>( rhs ).m_chi2)
   , m_NStrawHits(const_cast<TrigInDetTrack_p2 &>( rhs ).m_NStrawHits)
   , m_NStraw(const_cast<TrigInDetTrack_p2 &>( rhs ).m_NStraw)
   , m_NStrawTime(const_cast<TrigInDetTrack_p2 &>( rhs ).m_NStrawTime)
   , m_NTRHits(const_cast<TrigInDetTrack_p2 &>( rhs ).m_NTRHits)
   , m_NPixelSpacePoints(const_cast<TrigInDetTrack_p2 &>( rhs ).m_NPixelSpacePoints)
   , m_NSCT_SpacePoints(const_cast<TrigInDetTrack_p2 &>( rhs ).m_NSCT_SpacePoints)
   , m_HitPattern(const_cast<TrigInDetTrack_p2 &>( rhs ).m_HitPattern)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigInDetTrack_p2::~TrigInDetTrack_p2() {
}
#endif // TrigInDetTrack_p2_cxx

#ifndef TrigL2BphysContainer_tlp1_cxx
#define TrigL2BphysContainer_tlp1_cxx
TrigL2BphysContainer_tlp1::TrigL2BphysContainer_tlp1() {
}
TrigL2BphysContainer_tlp1::TrigL2BphysContainer_tlp1(const TrigL2BphysContainer_tlp1 & rhs)
   : m_TrigL2BphysContainers(const_cast<TrigL2BphysContainer_tlp1 &>( rhs ).m_TrigL2BphysContainers)
   , m_L2Bphys(const_cast<TrigL2BphysContainer_tlp1 &>( rhs ).m_L2Bphys)
   , m_L2Bphys_p2(const_cast<TrigL2BphysContainer_tlp1 &>( rhs ).m_L2Bphys_p2)
   , m_trigVertex(const_cast<TrigL2BphysContainer_tlp1 &>( rhs ).m_trigVertex)
   , m_trigInDetTrack(const_cast<TrigL2BphysContainer_tlp1 &>( rhs ).m_trigInDetTrack)
   , m_trigInDetTrackFitPar(const_cast<TrigL2BphysContainer_tlp1 &>( rhs ).m_trigInDetTrackFitPar)
   , m_trigInDetTrackCollection(const_cast<TrigL2BphysContainer_tlp1 &>( rhs ).m_trigInDetTrackCollection)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigL2BphysContainer_tlp1 &modrhs = const_cast<TrigL2BphysContainer_tlp1 &>( rhs );
   modrhs.m_TrigL2BphysContainers.clear();
   modrhs.m_L2Bphys.clear();
   modrhs.m_L2Bphys_p2.clear();
   modrhs.m_trigVertex.clear();
   modrhs.m_trigInDetTrack.clear();
   modrhs.m_trigInDetTrackFitPar.clear();
   modrhs.m_trigInDetTrackCollection.clear();
}
TrigL2BphysContainer_tlp1::~TrigL2BphysContainer_tlp1() {
}
#endif // TrigL2BphysContainer_tlp1_cxx

#ifndef TrigL2Bphys_p1_cxx
#define TrigL2Bphys_p1_cxx
TrigL2Bphys_p1::TrigL2Bphys_p1() {
}
TrigL2Bphys_p1::TrigL2Bphys_p1(const TrigL2Bphys_p1 & rhs)
   : m_roiID(const_cast<TrigL2Bphys_p1 &>( rhs ).m_roiID)
   , m_particleType(const_cast<TrigL2Bphys_p1 &>( rhs ).m_particleType)
   , m_eta(const_cast<TrigL2Bphys_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<TrigL2Bphys_p1 &>( rhs ).m_phi)
   , m_mass(const_cast<TrigL2Bphys_p1 &>( rhs ).m_mass)
   , m_dist(const_cast<TrigL2Bphys_p1 &>( rhs ).m_dist)
   , m_valid(const_cast<TrigL2Bphys_p1 &>( rhs ).m_valid)
   , m_pVertex(const_cast<TrigL2Bphys_p1 &>( rhs ).m_pVertex)
   , m_secondaryDecay(const_cast<TrigL2Bphys_p1 &>( rhs ).m_secondaryDecay)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigL2Bphys_p1::~TrigL2Bphys_p1() {
}
#endif // TrigL2Bphys_p1_cxx

#ifndef TrigL2Bphys_p2_cxx
#define TrigL2Bphys_p2_cxx
TrigL2Bphys_p2::TrigL2Bphys_p2() {
}
TrigL2Bphys_p2::TrigL2Bphys_p2(const TrigL2Bphys_p2 & rhs)
   : m_roiID(const_cast<TrigL2Bphys_p2 &>( rhs ).m_roiID)
   , m_particleType(const_cast<TrigL2Bphys_p2 &>( rhs ).m_particleType)
   , m_eta(const_cast<TrigL2Bphys_p2 &>( rhs ).m_eta)
   , m_phi(const_cast<TrigL2Bphys_p2 &>( rhs ).m_phi)
   , m_mass(const_cast<TrigL2Bphys_p2 &>( rhs ).m_mass)
   , m_fitmass(const_cast<TrigL2Bphys_p2 &>( rhs ).m_fitmass)
   , m_fitchi2(const_cast<TrigL2Bphys_p2 &>( rhs ).m_fitchi2)
   , m_fitndof(const_cast<TrigL2Bphys_p2 &>( rhs ).m_fitndof)
   , m_fitx(const_cast<TrigL2Bphys_p2 &>( rhs ).m_fitx)
   , m_fity(const_cast<TrigL2Bphys_p2 &>( rhs ).m_fity)
   , m_fitz(const_cast<TrigL2Bphys_p2 &>( rhs ).m_fitz)
   , m_trackVector(const_cast<TrigL2Bphys_p2 &>( rhs ).m_trackVector)
   , m_secondaryDecay(const_cast<TrigL2Bphys_p2 &>( rhs ).m_secondaryDecay)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigL2Bphys_p2::~TrigL2Bphys_p2() {
}
#endif // TrigL2Bphys_p2_cxx

#ifndef TrigEFBphysContainer_tlp1_cxx
#define TrigEFBphysContainer_tlp1_cxx
TrigEFBphysContainer_tlp1::TrigEFBphysContainer_tlp1() {
}
TrigEFBphysContainer_tlp1::TrigEFBphysContainer_tlp1(const TrigEFBphysContainer_tlp1 & rhs)
   : m_TrigEFBphysContainers(const_cast<TrigEFBphysContainer_tlp1 &>( rhs ).m_TrigEFBphysContainers)
   , m_EFBphys(const_cast<TrigEFBphysContainer_tlp1 &>( rhs ).m_EFBphys)
   , m_EFBphys_p2(const_cast<TrigEFBphysContainer_tlp1 &>( rhs ).m_EFBphys_p2)
   , m_TrackParticleContainer(const_cast<TrigEFBphysContainer_tlp1 &>( rhs ).m_TrackParticleContainer)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigEFBphysContainer_tlp1 &modrhs = const_cast<TrigEFBphysContainer_tlp1 &>( rhs );
   modrhs.m_TrigEFBphysContainers.clear();
   modrhs.m_EFBphys.clear();
   modrhs.m_EFBphys_p2.clear();
   modrhs.m_TrackParticleContainer.clear();
}
TrigEFBphysContainer_tlp1::~TrigEFBphysContainer_tlp1() {
}
#endif // TrigEFBphysContainer_tlp1_cxx

#ifndef TrigEFBphys_p1_cxx
#define TrigEFBphys_p1_cxx
TrigEFBphys_p1::TrigEFBphys_p1() {
}
TrigEFBphys_p1::TrigEFBphys_p1(const TrigEFBphys_p1 & rhs)
   : m_roiID(const_cast<TrigEFBphys_p1 &>( rhs ).m_roiID)
   , m_particleType(const_cast<TrigEFBphys_p1 &>( rhs ).m_particleType)
   , m_eta(const_cast<TrigEFBphys_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<TrigEFBphys_p1 &>( rhs ).m_phi)
   , m_mass(const_cast<TrigEFBphys_p1 &>( rhs ).m_mass)
   , m_valid(const_cast<TrigEFBphys_p1 &>( rhs ).m_valid)
   , m_secondaryDecay(const_cast<TrigEFBphys_p1 &>( rhs ).m_secondaryDecay)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigEFBphys_p1::~TrigEFBphys_p1() {
}
#endif // TrigEFBphys_p1_cxx

#ifndef TrigEFBphys_p2_cxx
#define TrigEFBphys_p2_cxx
TrigEFBphys_p2::TrigEFBphys_p2() {
}
TrigEFBphys_p2::TrigEFBphys_p2(const TrigEFBphys_p2 & rhs)
   : m_roiID(const_cast<TrigEFBphys_p2 &>( rhs ).m_roiID)
   , m_particleType(const_cast<TrigEFBphys_p2 &>( rhs ).m_particleType)
   , m_eta(const_cast<TrigEFBphys_p2 &>( rhs ).m_eta)
   , m_phi(const_cast<TrigEFBphys_p2 &>( rhs ).m_phi)
   , m_mass(const_cast<TrigEFBphys_p2 &>( rhs ).m_mass)
   , m_fitmass(const_cast<TrigEFBphys_p2 &>( rhs ).m_fitmass)
   , m_fitchi2(const_cast<TrigEFBphys_p2 &>( rhs ).m_fitchi2)
   , m_fitndof(const_cast<TrigEFBphys_p2 &>( rhs ).m_fitndof)
   , m_fitx(const_cast<TrigEFBphys_p2 &>( rhs ).m_fitx)
   , m_fity(const_cast<TrigEFBphys_p2 &>( rhs ).m_fity)
   , m_fitz(const_cast<TrigEFBphys_p2 &>( rhs ).m_fitz)
   , m_secondaryDecay(const_cast<TrigEFBphys_p2 &>( rhs ).m_secondaryDecay)
   , m_trackVector(const_cast<TrigEFBphys_p2 &>( rhs ).m_trackVector)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigEFBphys_p2::~TrigEFBphys_p2() {
}
#endif // TrigEFBphys_p2_cxx

#ifndef RingerRingsContainer_tlp1_cxx
#define RingerRingsContainer_tlp1_cxx
RingerRingsContainer_tlp1::RingerRingsContainer_tlp1() {
}
RingerRingsContainer_tlp1::RingerRingsContainer_tlp1(const RingerRingsContainer_tlp1 & rhs)
   : m_RingerRingsContainers(const_cast<RingerRingsContainer_tlp1 &>( rhs ).m_RingerRingsContainers)
   , m_RingerRings(const_cast<RingerRingsContainer_tlp1 &>( rhs ).m_RingerRings)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   RingerRingsContainer_tlp1 &modrhs = const_cast<RingerRingsContainer_tlp1 &>( rhs );
   modrhs.m_RingerRingsContainers.clear();
   modrhs.m_RingerRings.clear();
}
RingerRingsContainer_tlp1::~RingerRingsContainer_tlp1() {
}
#endif // RingerRingsContainer_tlp1_cxx

#ifndef RingerRings_p1_cxx
#define RingerRings_p1_cxx
RingerRings_p1::RingerRings_p1() {
}
RingerRings_p1::RingerRings_p1(const RingerRings_p1 & rhs)
   : m_numberOfRings(const_cast<RingerRings_p1 &>( rhs ).m_numberOfRings)
   , m_rings(const_cast<RingerRings_p1 &>( rhs ).m_rings)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   RingerRings_p1 &modrhs = const_cast<RingerRings_p1 &>( rhs );
   modrhs.m_rings.clear();
}
RingerRings_p1::~RingerRings_p1() {
}
#endif // RingerRings_p1_cxx

#ifndef Rec__TrackParticleTruthCollection_p1__Entry_cxx
#define Rec__TrackParticleTruthCollection_p1__Entry_cxx
Rec::TrackParticleTruthCollection_p1::Entry::Entry() {
}
Rec::TrackParticleTruthCollection_p1::Entry::Entry(const Entry & rhs)
   : index(const_cast<Entry &>( rhs ).index)
   , probability(const_cast<Entry &>( rhs ).probability)
   , particle(const_cast<Entry &>( rhs ).particle)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Rec::TrackParticleTruthCollection_p1::Entry::~Entry() {
}
#endif // Rec__TrackParticleTruthCollection_p1__Entry_cxx

#ifndef Rec__TrackParticleTruthCollection_p1_cxx
#define Rec__TrackParticleTruthCollection_p1_cxx
Rec::TrackParticleTruthCollection_p1::TrackParticleTruthCollection_p1() {
}
Rec::TrackParticleTruthCollection_p1::TrackParticleTruthCollection_p1(const TrackParticleTruthCollection_p1 & rhs)
   : m_trackCollectionLink(const_cast<TrackParticleTruthCollection_p1 &>( rhs ).m_trackCollectionLink)
   , m_entries(const_cast<TrackParticleTruthCollection_p1 &>( rhs ).m_entries)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrackParticleTruthCollection_p1 &modrhs = const_cast<TrackParticleTruthCollection_p1 &>( rhs );
   modrhs.m_entries.clear();
}
Rec::TrackParticleTruthCollection_p1::~TrackParticleTruthCollection_p1() {
}
#endif // Rec__TrackParticleTruthCollection_p1_cxx

#ifndef Trk__V0Container_tlp1_cxx
#define Trk__V0Container_tlp1_cxx
Trk::V0Container_tlp1::V0Container_tlp1() {
}
Trk::V0Container_tlp1::V0Container_tlp1(const V0Container_tlp1 & rhs)
   : m_v0Containers(const_cast<V0Container_tlp1 &>( rhs ).m_v0Containers)
   , m_v0Candidates(const_cast<V0Container_tlp1 &>( rhs ).m_v0Candidates)
   , m_v0Hypothesises(const_cast<V0Container_tlp1 &>( rhs ).m_v0Hypothesises)
   , m_extendedVxCandidates(const_cast<V0Container_tlp1 &>( rhs ).m_extendedVxCandidates)
   , m_vxCandidates(const_cast<V0Container_tlp1 &>( rhs ).m_vxCandidates)
   , m_vxTracksAtVertex(const_cast<V0Container_tlp1 &>( rhs ).m_vxTracksAtVertex)
   , m_recVertices(const_cast<V0Container_tlp1 &>( rhs ).m_recVertices)
   , m_vertices(const_cast<V0Container_tlp1 &>( rhs ).m_vertices)
   , m_tracks(const_cast<V0Container_tlp1 &>( rhs ).m_tracks)
   , m_trackParameters(const_cast<V0Container_tlp1 &>( rhs ).m_trackParameters)
   , m_perigees(const_cast<V0Container_tlp1 &>( rhs ).m_perigees)
   , m_measPerigees(const_cast<V0Container_tlp1 &>( rhs ).m_measPerigees)
   , m_surfaces(const_cast<V0Container_tlp1 &>( rhs ).m_surfaces)
   , m_fitQualities(const_cast<V0Container_tlp1 &>( rhs ).m_fitQualities)
   , m_hepSymMatrices(const_cast<V0Container_tlp1 &>( rhs ).m_hepSymMatrices)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   V0Container_tlp1 &modrhs = const_cast<V0Container_tlp1 &>( rhs );
   modrhs.m_v0Containers.clear();
   modrhs.m_v0Candidates.clear();
   modrhs.m_v0Hypothesises.clear();
   modrhs.m_extendedVxCandidates.clear();
   modrhs.m_vxCandidates.clear();
   modrhs.m_vxTracksAtVertex.clear();
   modrhs.m_recVertices.clear();
   modrhs.m_vertices.clear();
   modrhs.m_tracks.clear();
   modrhs.m_trackParameters.clear();
   modrhs.m_perigees.clear();
   modrhs.m_measPerigees.clear();
   modrhs.m_surfaces.clear();
   modrhs.m_fitQualities.clear();
   modrhs.m_hepSymMatrices.clear();
}
Trk::V0Container_tlp1::~V0Container_tlp1() {
}
#endif // Trk__V0Container_tlp1_cxx

#ifndef Trk__V0Candidate_p1_cxx
#define Trk__V0Candidate_p1_cxx
Trk::V0Candidate_p1::V0Candidate_p1() {
}
Trk::V0Candidate_p1::V0Candidate_p1(const V0Candidate_p1 & rhs)
   : m_v0Hyp(const_cast<V0Candidate_p1 &>( rhs ).m_v0Hyp)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   V0Candidate_p1 &modrhs = const_cast<V0Candidate_p1 &>( rhs );
   modrhs.m_v0Hyp.clear();
}
Trk::V0Candidate_p1::~V0Candidate_p1() {
}
#endif // Trk__V0Candidate_p1_cxx

#ifndef Trk__V0Hypothesis_p1_cxx
#define Trk__V0Hypothesis_p1_cxx
Trk::V0Hypothesis_p1::V0Hypothesis_p1() {
}
Trk::V0Hypothesis_p1::V0Hypothesis_p1(const V0Hypothesis_p1 & rhs)
   : m_extendedVxCandidate(const_cast<V0Hypothesis_p1 &>( rhs ).m_extendedVxCandidate)
   , m_positiveTrackID(const_cast<V0Hypothesis_p1 &>( rhs ).m_positiveTrackID)
   , m_negativeTrackID(const_cast<V0Hypothesis_p1 &>( rhs ).m_negativeTrackID)
   , m_constraintID(const_cast<V0Hypothesis_p1 &>( rhs ).m_constraintID)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Trk::V0Hypothesis_p1::~V0Hypothesis_p1() {
}
#endif // Trk__V0Hypothesis_p1_cxx

#ifndef CosmicMuonCollection_tlp1_cxx
#define CosmicMuonCollection_tlp1_cxx
CosmicMuonCollection_tlp1::CosmicMuonCollection_tlp1() {
}
CosmicMuonCollection_tlp1::CosmicMuonCollection_tlp1(const CosmicMuonCollection_tlp1 & rhs)
   : m_cosmicMuonCollectionVec(const_cast<CosmicMuonCollection_tlp1 &>( rhs ).m_cosmicMuonCollectionVec)
   , m_cosmicMuonVec(const_cast<CosmicMuonCollection_tlp1 &>( rhs ).m_cosmicMuonVec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CosmicMuonCollection_tlp1 &modrhs = const_cast<CosmicMuonCollection_tlp1 &>( rhs );
   modrhs.m_cosmicMuonCollectionVec.clear();
   modrhs.m_cosmicMuonVec.clear();
}
CosmicMuonCollection_tlp1::~CosmicMuonCollection_tlp1() {
}
#endif // CosmicMuonCollection_tlp1_cxx

#ifndef CosmicMuon_p1_cxx
#define CosmicMuon_p1_cxx
CosmicMuon_p1::CosmicMuon_p1() {
}
CosmicMuon_p1::CosmicMuon_p1(const CosmicMuon_p1 & rhs)
   : mP(const_cast<CosmicMuon_p1 &>( rhs ).mP)
   , mRadius(const_cast<CosmicMuon_p1 &>( rhs ).mRadius)
   , mTheta(const_cast<CosmicMuon_p1 &>( rhs ).mTheta)
   , mPhi(const_cast<CosmicMuon_p1 &>( rhs ).mPhi)
   , mT(const_cast<CosmicMuon_p1 &>( rhs ).mT)
   , mIsIngoing(const_cast<CosmicMuon_p1 &>( rhs ).mIsIngoing)
   , mNRpcPairs(const_cast<CosmicMuon_p1 &>( rhs ).mNRpcPairs)
   , mNTgcPairs(const_cast<CosmicMuon_p1 &>( rhs ).mNTgcPairs)
   , mNMdtHits(const_cast<CosmicMuon_p1 &>( rhs ).mNMdtHits)
   , mNMdtSegs(const_cast<CosmicMuon_p1 &>( rhs ).mNMdtSegs)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   for (Int_t i=0;i<3;i++) mPoint[i] = rhs.mPoint[i];
}
CosmicMuon_p1::~CosmicMuon_p1() {
}
#endif // CosmicMuon_p1_cxx

#ifndef CaloCellLinkContainer_p2_cxx
#define CaloCellLinkContainer_p2_cxx
CaloCellLinkContainer_p2::CaloCellLinkContainer_p2() {
}
CaloCellLinkContainer_p2::CaloCellLinkContainer_p2(const CaloCellLinkContainer_p2 & rhs)
   : m_linkI(const_cast<CaloCellLinkContainer_p2 &>( rhs ).m_linkI)
   , m_linkW(const_cast<CaloCellLinkContainer_p2 &>( rhs ).m_linkW)
   , m_vISizes(const_cast<CaloCellLinkContainer_p2 &>( rhs ).m_vISizes)
   , m_vWSizes(const_cast<CaloCellLinkContainer_p2 &>( rhs ).m_vWSizes)
   , m_nClusters(const_cast<CaloCellLinkContainer_p2 &>( rhs ).m_nClusters)
   , m_contName(const_cast<CaloCellLinkContainer_p2 &>( rhs ).m_contName)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CaloCellLinkContainer_p2 &modrhs = const_cast<CaloCellLinkContainer_p2 &>( rhs );
   modrhs.m_linkI.clear();
   modrhs.m_linkW.clear();
   modrhs.m_vISizes.clear();
   modrhs.m_vWSizes.clear();
   modrhs.m_contName.clear();
}
CaloCellLinkContainer_p2::~CaloCellLinkContainer_p2() {
}
#endif // CaloCellLinkContainer_p2_cxx

#ifndef TrigTauContainer_tlp1_cxx
#define TrigTauContainer_tlp1_cxx
TrigTauContainer_tlp1::TrigTauContainer_tlp1() {
}
TrigTauContainer_tlp1::TrigTauContainer_tlp1(const TrigTauContainer_tlp1 & rhs)
   : m_TrigTauContainers(const_cast<TrigTauContainer_tlp1 &>( rhs ).m_TrigTauContainers)
   , m_Tau(const_cast<TrigTauContainer_tlp1 &>( rhs ).m_Tau)
   , m_P4PtEtaPhiM(const_cast<TrigTauContainer_tlp1 &>( rhs ).m_P4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrigTauContainer_tlp1 &modrhs = const_cast<TrigTauContainer_tlp1 &>( rhs );
   modrhs.m_TrigTauContainers.clear();
   modrhs.m_Tau.clear();
   modrhs.m_P4PtEtaPhiM.clear();
}
TrigTauContainer_tlp1::~TrigTauContainer_tlp1() {
}
#endif // TrigTauContainer_tlp1_cxx

#ifndef TrigTau_p2_cxx
#define TrigTau_p2_cxx
TrigTau_p2::TrigTau_p2() {
}
TrigTau_p2::TrigTau_p2(const TrigTau_p2 & rhs)
   : m_roiID(const_cast<TrigTau_p2 &>( rhs ).m_roiID)
   , m_Zvtx(const_cast<TrigTau_p2 &>( rhs ).m_Zvtx)
   , m_err_Zvtx(const_cast<TrigTau_p2 &>( rhs ).m_err_Zvtx)
   , m_etCalibCluster(const_cast<TrigTau_p2 &>( rhs ).m_etCalibCluster)
   , m_simpleEtFlow(const_cast<TrigTau_p2 &>( rhs ).m_simpleEtFlow)
   , m_nMatchedTracks(const_cast<TrigTau_p2 &>( rhs ).m_nMatchedTracks)
   , m_p4PtEtaPhiM(const_cast<TrigTau_p2 &>( rhs ).m_p4PtEtaPhiM)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
TrigTau_p2::~TrigTau_p2() {
}
#endif // TrigTau_p2_cxx

#ifndef MuonSpShowerContainer_p1_cxx
#define MuonSpShowerContainer_p1_cxx
MuonSpShowerContainer_p1::MuonSpShowerContainer_p1() {
}
MuonSpShowerContainer_p1::MuonSpShowerContainer_p1(const MuonSpShowerContainer_p1 & rhs)
   : m_showers(const_cast<MuonSpShowerContainer_p1 &>( rhs ).m_showers)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MuonSpShowerContainer_p1 &modrhs = const_cast<MuonSpShowerContainer_p1 &>( rhs );
   modrhs.m_showers.clear();
}
MuonSpShowerContainer_p1::~MuonSpShowerContainer_p1() {
}
#endif // MuonSpShowerContainer_p1_cxx

#ifndef MdtTrackSegmentCollection_tlp1_cxx
#define MdtTrackSegmentCollection_tlp1_cxx
MdtTrackSegmentCollection_tlp1::MdtTrackSegmentCollection_tlp1() {
}
MdtTrackSegmentCollection_tlp1::MdtTrackSegmentCollection_tlp1(const MdtTrackSegmentCollection_tlp1 & rhs)
   : m_mdtTrackSegmentCollectionVec(const_cast<MdtTrackSegmentCollection_tlp1 &>( rhs ).m_mdtTrackSegmentCollectionVec)
   , m_mdtTrackSegmentVec(const_cast<MdtTrackSegmentCollection_tlp1 &>( rhs ).m_mdtTrackSegmentVec)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MdtTrackSegmentCollection_tlp1 &modrhs = const_cast<MdtTrackSegmentCollection_tlp1 &>( rhs );
   modrhs.m_mdtTrackSegmentCollectionVec.clear();
   modrhs.m_mdtTrackSegmentVec.clear();
}
MdtTrackSegmentCollection_tlp1::~MdtTrackSegmentCollection_tlp1() {
}
#endif // MdtTrackSegmentCollection_tlp1_cxx

#ifndef MdtTrackSegment_p1_cxx
#define MdtTrackSegment_p1_cxx
MdtTrackSegment_p1::MdtTrackSegment_p1() {
}
MdtTrackSegment_p1::MdtTrackSegment_p1(const MdtTrackSegment_p1 & rhs)
   : mStationId(const_cast<MdtTrackSegment_p1 &>( rhs ).mStationId)
   , mTrackId(const_cast<MdtTrackSegment_p1 &>( rhs ).mTrackId)
   , mAlpha(const_cast<MdtTrackSegment_p1 &>( rhs ).mAlpha)
   , mB(const_cast<MdtTrackSegment_p1 &>( rhs ).mB)
   , mSwap(const_cast<MdtTrackSegment_p1 &>( rhs ).mSwap)
   , mT0(const_cast<MdtTrackSegment_p1 &>( rhs ).mT0)
   , mChi2(const_cast<MdtTrackSegment_p1 &>( rhs ).mChi2)
   , mNHits(const_cast<MdtTrackSegment_p1 &>( rhs ).mNHits)
   , mR(const_cast<MdtTrackSegment_p1 &>( rhs ).mR)
   , mZ(const_cast<MdtTrackSegment_p1 &>( rhs ).mZ)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
MdtTrackSegment_p1::~MdtTrackSegment_p1() {
}
#endif // MdtTrackSegment_p1_cxx

#ifndef Rec__TrackParticleContainer_tlp1_cxx
#define Rec__TrackParticleContainer_tlp1_cxx
Rec::TrackParticleContainer_tlp1::TrackParticleContainer_tlp1() {
}
Rec::TrackParticleContainer_tlp1::TrackParticleContainer_tlp1(const TrackParticleContainer_tlp1 & rhs)
   : m_tokenList(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_tokenList)
   , m_trackParticleContainer(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_trackParticleContainer)
   , m_trackParticle(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_trackParticle)
   , m_trackParticleBase(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_trackParticleBase)
   , m_tracks(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_tracks)
   , m_vxCandidates(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_vxCandidates)
   , m_trackParameters(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_trackParameters)
   , m_ataSurfaces(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_ataSurfaces)
   , m_measuredAtaSurfaces(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_measuredAtaSurfaces)
   , m_perigees(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_perigees)
   , m_measPerigees(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_measPerigees)
   , m_trackSummaries(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_trackSummaries)
   , m_boundSurfaces(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_boundSurfaces)
   , m_surfaces(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_surfaces)
   , m_cylinderBounds(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_cylinderBounds)
   , m_diamondBounds(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_diamondBounds)
   , m_discBounds(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_discBounds)
   , m_rectangleBounds(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_rectangleBounds)
   , m_trapesoidBounds(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_trapesoidBounds)
   , m_rotatedTrapesoidBounds(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_rotatedTrapesoidBounds)
   , m_detElementSurfaces(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_detElementSurfaces)
   , m_fitQualities(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_fitQualities)
   , m_hepSymMatrices(const_cast<TrackParticleContainer_tlp1 &>( rhs ).m_hepSymMatrices)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrackParticleContainer_tlp1 &modrhs = const_cast<TrackParticleContainer_tlp1 &>( rhs );
   modrhs.m_trackParticleContainer.clear();
   modrhs.m_trackParticle.clear();
   modrhs.m_trackParticleBase.clear();
   modrhs.m_tracks.clear();
   modrhs.m_vxCandidates.clear();
   modrhs.m_trackParameters.clear();
   modrhs.m_ataSurfaces.clear();
   modrhs.m_measuredAtaSurfaces.clear();
   modrhs.m_perigees.clear();
   modrhs.m_measPerigees.clear();
   modrhs.m_trackSummaries.clear();
   modrhs.m_boundSurfaces.clear();
   modrhs.m_surfaces.clear();
   modrhs.m_cylinderBounds.clear();
   modrhs.m_diamondBounds.clear();
   modrhs.m_discBounds.clear();
   modrhs.m_rectangleBounds.clear();
   modrhs.m_trapesoidBounds.clear();
   modrhs.m_rotatedTrapesoidBounds.clear();
   modrhs.m_detElementSurfaces.clear();
   modrhs.m_fitQualities.clear();
   modrhs.m_hepSymMatrices.clear();
}
Rec::TrackParticleContainer_tlp1::~TrackParticleContainer_tlp1() {
}
#endif // Rec__TrackParticleContainer_tlp1_cxx

#ifndef Rec__TrackParticle_p1_cxx
#define Rec__TrackParticle_p1_cxx
Rec::TrackParticle_p1::TrackParticle_p1() {
}
Rec::TrackParticle_p1::TrackParticle_p1(const TrackParticle_p1 & rhs)
   : m_trackParticleBase(const_cast<TrackParticle_p1 &>( rhs ).m_trackParticleBase)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
Rec::TrackParticle_p1::~TrackParticle_p1() {
}
#endif // Rec__TrackParticle_p1_cxx

#ifndef Trk__TrackParticleBase_p1_cxx
#define Trk__TrackParticleBase_p1_cxx
Trk::TrackParticleBase_p1::TrackParticleBase_p1() {
}
Trk::TrackParticleBase_p1::TrackParticleBase_p1(const TrackParticleBase_p1 & rhs)
   : m_originalTrack(const_cast<TrackParticleBase_p1 &>( rhs ).m_originalTrack)
   , m_originalTrackNames(const_cast<TrackParticleBase_p1 &>( rhs ).m_originalTrackNames)
   , m_elVxCandidate(const_cast<TrackParticleBase_p1 &>( rhs ).m_elVxCandidate)
   , m_elVxCandidateNames(const_cast<TrackParticleBase_p1 &>( rhs ).m_elVxCandidateNames)
   , m_trackParameters(const_cast<TrackParticleBase_p1 &>( rhs ).m_trackParameters)
   , m_trackParticleOrigin(const_cast<TrackParticleBase_p1 &>( rhs ).m_trackParticleOrigin)
   , m_trackSummary(const_cast<TrackParticleBase_p1 &>( rhs ).m_trackSummary)
   , m_fitQuality(const_cast<TrackParticleBase_p1 &>( rhs ).m_fitQuality)
   , m_fitter(const_cast<TrackParticleBase_p1 &>( rhs ).m_fitter)
   , m_particleHypo(const_cast<TrackParticleBase_p1 &>( rhs ).m_particleHypo)
   , m_properties(const_cast<TrackParticleBase_p1 &>( rhs ).m_properties)
   , m_patternRecognition(const_cast<TrackParticleBase_p1 &>( rhs ).m_patternRecognition)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrackParticleBase_p1 &modrhs = const_cast<TrackParticleBase_p1 &>( rhs );
   modrhs.m_trackParameters.clear();
}
Trk::TrackParticleBase_p1::~TrackParticleBase_p1() {
}
#endif // Trk__TrackParticleBase_p1_cxx

#ifndef Trk__TrackSummary_p1_cxx
#define Trk__TrackSummary_p1_cxx
Trk::TrackSummary_p1::TrackSummary_p1() {
}
Trk::TrackSummary_p1::TrackSummary_p1(const TrackSummary_p1 & rhs)
   : m_information(const_cast<TrackSummary_p1 &>( rhs ).m_information)
   , m_idHitPattern(const_cast<TrackSummary_p1 &>( rhs ).m_idHitPattern)
   , m_eProbability(const_cast<TrackSummary_p1 &>( rhs ).m_eProbability)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   TrackSummary_p1 &modrhs = const_cast<TrackSummary_p1 &>( rhs );
   modrhs.m_information.clear();
   modrhs.m_eProbability.clear();
}
Trk::TrackSummary_p1::~TrackSummary_p1() {
}
#endif // Trk__TrackSummary_p1_cxx

#ifndef MuonCaloEnergyContainer_tlp1_cxx
#define MuonCaloEnergyContainer_tlp1_cxx
MuonCaloEnergyContainer_tlp1::MuonCaloEnergyContainer_tlp1() {
}
MuonCaloEnergyContainer_tlp1::MuonCaloEnergyContainer_tlp1(const MuonCaloEnergyContainer_tlp1 & rhs)
   : m_muonCaloEnergyContainer(const_cast<MuonCaloEnergyContainer_tlp1 &>( rhs ).m_muonCaloEnergyContainer)
   , m_caloEnergies(const_cast<MuonCaloEnergyContainer_tlp1 &>( rhs ).m_caloEnergies)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   MuonCaloEnergyContainer_tlp1 &modrhs = const_cast<MuonCaloEnergyContainer_tlp1 &>( rhs );
   modrhs.m_muonCaloEnergyContainer.clear();
   modrhs.m_caloEnergies.clear();
}
MuonCaloEnergyContainer_tlp1::~MuonCaloEnergyContainer_tlp1() {
}
#endif // MuonCaloEnergyContainer_tlp1_cxx

#ifndef CaloEnergy_p2_cxx
#define CaloEnergy_p2_cxx
CaloEnergy_p2::CaloEnergy_p2() {
}
CaloEnergy_p2::CaloEnergy_p2(const CaloEnergy_p2 & rhs)
   : m_energyLoss(const_cast<CaloEnergy_p2 &>( rhs ).m_energyLoss)
   , m_energyLossType(const_cast<CaloEnergy_p2 &>( rhs ).m_energyLossType)
   , m_caloLRLikelihood(const_cast<CaloEnergy_p2 &>( rhs ).m_caloLRLikelihood)
   , m_caloMuonIdTag(const_cast<CaloEnergy_p2 &>( rhs ).m_caloMuonIdTag)
   , m_fsrCandidateEnergy(const_cast<CaloEnergy_p2 &>( rhs ).m_fsrCandidateEnergy)
   , m_deposits(const_cast<CaloEnergy_p2 &>( rhs ).m_deposits)
   , m_etCore(const_cast<CaloEnergy_p2 &>( rhs ).m_etCore)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

   CaloEnergy_p2 &modrhs = const_cast<CaloEnergy_p2 &>( rhs );
   modrhs.m_deposits.clear();
}
CaloEnergy_p2::~CaloEnergy_p2() {
}
#endif // CaloEnergy_p2_cxx

#ifndef MuonSpShower_p1_cxx
#define MuonSpShower_p1_cxx
MuonSpShower_p1::MuonSpShower_p1() {
}
MuonSpShower_p1::MuonSpShower_p1(const MuonSpShower_p1 & rhs)
   : m_eta(const_cast<MuonSpShower_p1 &>( rhs ).m_eta)
   , m_phi(const_cast<MuonSpShower_p1 &>( rhs ).m_phi)
   , m_numberOfTriggerHits(const_cast<MuonSpShower_p1 &>( rhs ).m_numberOfTriggerHits)
   , m_numberOfInnerHits(const_cast<MuonSpShower_p1 &>( rhs ).m_numberOfInnerHits)
   , m_numberOfInnerSegments(const_cast<MuonSpShower_p1 &>( rhs ).m_numberOfInnerSegments)
   , m_numberOfMiddleHits(const_cast<MuonSpShower_p1 &>( rhs ).m_numberOfMiddleHits)
   , m_numberOfMiddleSegments(const_cast<MuonSpShower_p1 &>( rhs ).m_numberOfMiddleSegments)
   , m_numberOfOuterHits(const_cast<MuonSpShower_p1 &>( rhs ).m_numberOfOuterHits)
   , m_numberOfOuterSegments(const_cast<MuonSpShower_p1 &>( rhs ).m_numberOfOuterSegments)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
MuonSpShower_p1::~MuonSpShower_p1() {
}
#endif // MuonSpShower_p1_cxx

#ifndef DepositInCalo_p2_cxx
#define DepositInCalo_p2_cxx
DepositInCalo_p2::DepositInCalo_p2() {
}
DepositInCalo_p2::DepositInCalo_p2(const DepositInCalo_p2 & rhs)
   : m_subCaloId(const_cast<DepositInCalo_p2 &>( rhs ).m_subCaloId)
   , m_energyDeposited(const_cast<DepositInCalo_p2 &>( rhs ).m_energyDeposited)
   , m_muonEnergyLoss(const_cast<DepositInCalo_p2 &>( rhs ).m_muonEnergyLoss)
   , m_etDeposited(const_cast<DepositInCalo_p2 &>( rhs ).m_etDeposited)
{
   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).
   // Use at your own risk!

}
DepositInCalo_p2::~DepositInCalo_p2() {
}
#endif // DepositInCalo_p2_cxx

