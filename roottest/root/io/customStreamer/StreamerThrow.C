#include "StreamerThrowClass.h"

#include <TMemFile.h>

#include <iostream>
#include <stdexcept>

void StreamerThrow()
{
   StreamerThrowClass c;
   TMemFile f("mem.root", "RECREATE");
   try {
      f.WriteObject(&c, "c");
   } catch (const std::runtime_error &e) {
      std::cerr << "std::runtime_error: " << e.what() << "\n";
   }
   f.ls();
   f.Close();
}
