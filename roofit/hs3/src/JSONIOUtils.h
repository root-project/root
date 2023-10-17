#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <RooFit/Detail/JSONInterface.h>

std::unique_ptr<RooFit::Detail::JSONTree> varJSONString(const RooFit::Detail::JSONNode &treeRoot);

#endif
