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

struct cout_redirect {
   cout_redirect(std::string &_out, size_t bufSize = 102 * 1024) : out(_out)
   {
      old = std::cout.rdbuf(buffer.rdbuf());
      old2 = std::cerr.rdbuf(buffer.rdbuf());
      old3 = stdout;
      buffer2 = (char *)calloc(sizeof(char), bufSize);
      fp = fmemopen(buffer2, bufSize, "w");
      stdout = fp;
   }
   ~cout_redirect()
   {
      std::cout.rdbuf(old);
      std::cerr.rdbuf(old2);
      std::fclose(fp);
      stdout = old3;
      out = buffer.str();
      out += buffer2;
      free(buffer2);
   }

private:
   std::streambuf *old, *old2;
   std::stringstream buffer;
   char *buffer2;
   FILE *fp;
   FILE *old3;
   std::string &out;
};