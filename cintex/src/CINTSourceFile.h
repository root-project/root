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

         class CintexSrcFileInit {
         public:
            CintexSrcFileInit();
            operator G__input_file() { return fSrcFile; }
            operator bool() { return fSrcFile.filenum != -1; }
         private:
            G__input_file fSrcFile;
         };
         
         G__input_file fOldIFile;
         static CintexSrcFileInit fgCintexSrcFile;
      };
   }
}

ROOT::Cintex::ArtificialSourceFile::CintexSrcFileInit
  ROOT::Cintex::ArtificialSourceFile::fgCintexSrcFile;

inline
ROOT::Cintex::ArtificialSourceFile::ArtificialSourceFile()
{
   // Set CINT's source file to libCintex.

   G__input_file* ifile = G__get_ifile();
   if (ifile && fgCintexSrcFile) {
      fOldIFile = *ifile;
      *ifile = fgCintexSrcFile;
   }
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

ROOT::Cintex::ArtificialSourceFile::CintexSrcFileInit::CintexSrcFileInit()
{
   // Set fSrcFile to current G__ifile - probably libCintex.

   G__input_file* ifile = G__get_ifile();
   if (ifile)
      fSrcFile = *ifile;
   else {
      memset(&fSrcFile, 0 ,sizeof(fSrcFile));
      fSrcFile.filenum = -1;
   }
}


#endif // ifdef INCLUDE_CINTSOURCEFILE_H
