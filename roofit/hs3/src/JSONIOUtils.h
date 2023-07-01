#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <ROOT/RStringView.hxx>

bool startsWith(std::string_view str, std::string_view prefix);
bool endsWith(std::string_view str, std::string_view suffix);

#endif
