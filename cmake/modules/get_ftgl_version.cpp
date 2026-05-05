#include <iostream>
#include <FTGL/ftgl.h>

int main()
{
   std::cout << FTGL::GetString(FTGL::CONFIG_VERSION) << std::flush;
   return 0;
}
