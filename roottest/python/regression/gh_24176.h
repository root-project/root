#ifndef GH_24176_H
#define GH_24176_H

// Rootmaps don't forward declare enums and variables.
namespace gh_24176 {
namespace ns {

enum EnumInNamespace {
   kFirst = 1,
   kSecond = 2
};

const int kConstant = 3;

} // namespace ns
} // namespace gh_24176

#endif
