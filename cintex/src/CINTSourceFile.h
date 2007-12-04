#ifndef INCLUDE_CINTSOURCEFILE_H
#define INCLUDE_CINTSOURCEFILE_H

#include "G__ci.h"

namespace ROOT {
   namespace Cintex {
      class ArtificialSourceFile {
      public:
         ArtificialSourceFile();
         ~ArtificialSourceFile();

      private:
         G__input_file fOldIFile;
      };
   }
}

inline
ROOT::Cintex::ArtificialSourceFile::ArtificialSourceFile()
{
   // Set CINT's source file to libCintex.

   G__setfilecontext("{CINTEX dictionary translator}", &fOldIFile);
}

inline
ROOT::Cintex::ArtificialSourceFile::~ArtificialSourceFile()
{
   // Reset CINT's source file to original one.

   G__input_file* ifile = G__get_ifile();
   if (ifile) {
      *ifile = fOldIFile;
   }
}

#endif // ifdef INCLUDE_CINTSOURCEFILE_H
