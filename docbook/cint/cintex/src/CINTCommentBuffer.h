// @(#)root/cintex:$Name:  $:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Cintex_CintexCommentBuffer
#define ROOT_Cintex_CintexCommentBuffer

#include <vector>

namespace ROOT { namespace Cintex {
   class CommentBuffer  {
   private:
      typedef std::vector<char*> VecC;
      VecC fC;
      CommentBuffer() {}
      ~CommentBuffer()  {
         for(VecC::iterator i=fC.begin(); i != fC.end(); ++i)
            delete [] *i;
         fC.clear();
      }
   public:
      static CommentBuffer& Instance()  {     
         static CommentBuffer inst;
         return inst;
      }
      void Add(char* cm)  {
         fC.push_back(cm);
      }
   };
} }

#endif // ROOT_Cintex_CintexCommentBuffer
