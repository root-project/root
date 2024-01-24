#ifndef DF_TEST
#define DF_TEST
#include <string>

namespace {

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the
 * guarded file when the wrapper object goes out of scope.
 */

class FileRaii {
private:
  std::string fPath;

public:
  explicit FileRaii(const std::string &path) : fPath(path) {}
  FileRaii(const FileRaii &) = delete;
  FileRaii &operator=(const FileRaii &) = delete;
  ~FileRaii() { std::remove(fPath.c_str()); }
  std::string GetPath() const { return fPath; }
};

} // anonymous namespace
#endif
