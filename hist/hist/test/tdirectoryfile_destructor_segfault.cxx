#include "TDirectoryFile.h"

int main()
{
   // Test against https://github.com/root-project/root/issues/13691
   // At destruction time TDirectoryFile called the destructor of
   // TDirectory, thus:
   // - inadvertently triggered initialization of gROOT
   // - called TDirectory::RecursiveRemove which didn't check for the validity
   //   of the `fList` data member, which had already been deleted in the
   //   TDirectoryFile destructor
   //
   // NOTE: In order for the segfault to actually be triggered, this test needs
   // to link against some library that is not in the list of globally ignored
   // PCMs (gIgnoredPCMNames in TCling.cxx). The loading of a PCM is what
   // actually triggers the call to TDirectory::RecursiveRemove in the end.
   TDirectoryFile f{"f", "f"};
}
