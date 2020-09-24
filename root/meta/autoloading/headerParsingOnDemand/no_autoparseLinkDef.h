#ifdef __ROOTCLING__
#pragma link C++ namespace edm;
#pragma link C++ class edm::Value+;
#pragma link C++ class pair<edm::Value, int>+;
#pragma link C++ namespace reco;
#pragma link C++ class reco::Muon+;
// # pragma link C++ enum reco::Muon::MuonTrackType; // not needed?
#pragma link C++ class map<reco::Muon::MuonTrackType, double>;
#pragma link C++ class vector<reco::Muon>+;
#pragma link C++ namespace edmNew;
#pragma link C++ namespace edmNew::dstvdetails;
#pragma link C++ class edmNew::dstvdetails::DetSetVectorTrans+;
#pragma link C++ class edmNew::dstvdetails::DetSetVectorTrans::Item+;
#ifndef ITEM_V10
#pragma read sourceClass="edmNew::dstvdetails::DetSetVectorTrans::Item" version="[10]" targetClass="edmNew::dstvdetails::DetSetVectorTrans::Item" source="int fOldValue;" target="fNewValue"
#endif // ITEM_V10
#pragma link C++ class vector<edmNew::dstvdetails::DetSetVectorTrans::Item>+;
#endif // __ROOTCLING__
