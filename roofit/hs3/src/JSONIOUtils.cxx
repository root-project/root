#include "JSONIOUtils.h"

bool endsWith(std::string_view str, std::string_view suffix) {
   return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
} 