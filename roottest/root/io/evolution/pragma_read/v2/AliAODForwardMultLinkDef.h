#if !defined( __MAKECINT__ ) && !defined(__CLING__)
# error Not for compilation
#else

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
// #pragma link off all methods;

#pragma read \
  sourceClass="AliAODForwardMult" \
  targetClass="AliAODForwardMult" \
  source="UInt_t fTriggers; Float_t fIpZ; Float_t fCentrality; UShort_t fNClusters;" \
  target="fHeader" \
  version="[-5]" \
  code="{newObj->CreateHeader(onfile.fTriggers,onfile.fIpZ,onfile.fCentrality,onfile.fNClusters);}"
#if CUSTOM_STREAMER
# pragma link C++ class AliAODForwardMult-;
#else
# pragma link C++ class AliAODForwardMult+;
#endif
#pragma link C++ class AliAODForwardHeader+;

#endif
//
// EOF
//
