#include <zlib.h>
#include <string.h>

bool test_zlib_version()
{
  return strcmp(ZLIB_VERSION, zlibVersion()) == 0;
}

int main(int argc, char** argv)
{
   return test_zlib_version() ? 0 : 1;
}
