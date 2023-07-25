#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <ROOT/RStringView.hxx>
#include <RooFit/Detail/JSONInterface.h>

bool startsWith(std::string_view str, std::string_view prefix);
bool endsWith(std::string_view str, std::string_view suffix);
std::unique_ptr<RooFit::Detail::JSONTree> varJSONString(const RooFit::Detail::JSONNode &treeRoot);

#endif
