/*
 * Project: xRooFit
 * Author:
 *   Will Buttinger, RAL 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "TSystem.h"
#include "TUUID.h"
#include <fstream>

struct cout_redirect {
   cout_redirect(std::string &_out, size_t bufSize = 102 * 1024)
      : out(_out), filename{"xRooFit-logging-"}, fBufSize(bufSize)
   {
      old = std::cout.rdbuf(buffer.rdbuf());
      old2 = std::cerr.rdbuf(buffer.rdbuf());
      // buffer2 = (char *)calloc(sizeof(char), bufSize);fp = fmemopen(buffer2, bufSize, "w");
      fp = gSystem->TempFileName(filename);
      if (fp) {
         stdout = fp;
         stderr = fp;
      }
   }
   ~cout_redirect()
   {
      std::cout.rdbuf(old);
      std::cerr.rdbuf(old2);
      stdout = old3;
      stderr = old4;
      if (fp) {
         std::fclose(fp);
         {
            std::ifstream t(filename);
            buffer << t.rdbuf();
         }
         gSystem->Unlink(filename); // delete the temp file
      }
      out = buffer.str();
      if (buffer2) {
         out += buffer2;
         free(buffer2);
      }
      if (out.length() > fBufSize)
         out.resize(fBufSize);
   }

private:
   std::streambuf *old;
   std::streambuf *old2;
   std::stringstream buffer;
   char *buffer2 = nullptr;
   FILE *fp = nullptr;
   FILE *old3 = stdout;
   FILE *old4 = stdout;
   std::string &out;
   TString filename;
   size_t fBufSize;
};