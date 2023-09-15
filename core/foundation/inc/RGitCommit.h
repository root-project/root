#ifndef ROOT_RGITCOMMIT_H
#define ROOT_RGITCOMMIT_H
#include "ROOT/RConfig.hxx"
namespace ROOT {
namespace Internal {
  static constexpr const char R__DEPRECATED(6,32, "Not updated anymore; please use etc/gitinfo.txt") gGitBranch[] = "";
  static constexpr const char R__DEPRECATED(6,32, "Not updated anymore; please use etc/gitinfo.txt") gGitCommit[] = "";
}
}
#define ROOT_GIT_BRANCH (ROOT::Internal::gGitBranch)
#define ROOT_GIT_COMMIT (ROOT::Internal::gGitCommit)
#endif
