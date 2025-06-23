#ifdef __ROOTCLING__

#pragma link C++ class Compat::DeprecatedAutoPtr<Track>+;
#pragma link C++ class Track+;
#pragma link C++ class TestAutoPtr+;

// In practice, we would have:
// #pragma read sourceClass="std::auto_ptr<Track>" targetClass="Compat::DeprecatedAutoPtr<Track>"
// but in this example the writing code had to use EmulatedAutoPtr, so we need this:
#pragma read sourceClass="EmulatedAutoPtr<Track>" targetClass="Compat::DeprecatedAutoPtr<Track>"

#pragma read sourceClass="TestAutoPtr" targetClass="TestAutoPtr" version="[2]" \
  source="Compat::DeprecatedAutoPtr<Track> fTrack" target="fTrack" \
  code="{ fTrack.reset(onfile.fTrack.fRawPtr); onfile.fTrack.fRawPtr = nullptr; }"

#endif
