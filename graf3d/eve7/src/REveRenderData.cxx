// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/REveRenderData.hxx"

#include <cstdio>
#include <cstring>

using namespace ROOT::Experimental;

REveRenderData::REveRenderData(const std::string &func, int size_vert, int size_norm, int size_idx) : fRnrFunc(func)
{
   Reserve(size_vert, size_norm, size_idx);
}

REveRenderData::~REveRenderData() {}


void REveRenderData::Reserve(int size_vert, int size_norm, int size_idx)
{
   if (size_vert > 0)
      fVertexBuffer.reserve(size_vert);
   if (size_norm > 0)
      fNormalBuffer.reserve(size_norm);
   if (size_idx > 0)
      fIndexBuffer.reserve(size_idx);
}

int REveRenderData::Write(char *msg)
{
   // XXXX Where do we make sure the buffer is large enough?
   //std::string fh = fHeader.dump();
   //memcpy(msg, fh.c_str(), fh.size());
   //int off = int(ceil(fh.size()/4.0))*4;

   int off = 0;

   if (!fVertexBuffer.empty()) {
      int binsize = fVertexBuffer.size() * sizeof(float);
      memcpy(msg + off, &fVertexBuffer[0], binsize);
      off += binsize;
   }
   if (!fNormalBuffer.empty()) {
      int binsize = fNormalBuffer.size() * sizeof(float);
      memcpy(msg + off, &fNormalBuffer[0], binsize);
      off += binsize;
   }
   if (!fIndexBuffer.empty()) {
      int binsize = fIndexBuffer.size() * sizeof(float);
      memcpy(msg + off, &fIndexBuffer[0], binsize);
      off += binsize;
   }
   return off;
}

void REveRenderData::Dump()
{
   printf("RederData dump %d\n", (int)fVertexBuffer.size());
   int cnt = 0;
   for (auto it = fVertexBuffer.begin(); it !=fVertexBuffer.end(); ++it )
   {
      printf("%d %f", cnt++, *it);
   }
}
