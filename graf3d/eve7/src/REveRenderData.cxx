// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveRenderData.hxx>
#include <ROOT/REveUtil.hxx>

#include <cstdio>
#include <cstring>

using namespace ROOT::Experimental;

/////////////////////////////////////////////////////////////////////////////////////////
/// Constructor

REveRenderData::REveRenderData(const std::string &func, int size_vert, int size_norm, int size_idx) :
   fRnrFunc(func)
{
   Reserve(size_vert, size_norm, size_idx);
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Reserve place for render data

void REveRenderData::Reserve(int size_vert, int size_norm, int size_idx)
{
   if (size_vert > 0)
      fVertexBuffer.reserve(size_vert);
   if (size_norm > 0)
      fNormalBuffer.reserve(size_norm);
   if (size_idx > 0)
      fIndexBuffer.reserve(size_idx);
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Write render data to binary buffer

int REveRenderData::Write(char *msg, int maxlen)
{
   static const REveException eh("REveRenderData::Write ");

   int off{0};

   auto append = [&](void *buf, int len) {
      if (off + len > maxlen)
         throw eh + "output buffer does not have enough memory";
      memcpy(msg + off, buf, len);
      off += len;
   };

   if (!fMatrix.empty())
      append(&fMatrix[0], fMatrix.size() * sizeof(float));

   if (!fVertexBuffer.empty())
      append(&fVertexBuffer[0], fVertexBuffer.size() * sizeof(float));

   if (!fNormalBuffer.empty())
      append(&fNormalBuffer[0], fNormalBuffer.size() * sizeof(float));

   if (!fIndexBuffer.empty())
      append(&fIndexBuffer[0], fIndexBuffer.size() * sizeof(int));

   return off;
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Set transformation matrix

void REveRenderData::SetMatrix(const double *arr)
{
   fMatrix.reserve(16);
   for (int i = 0; i < 16; ++i) {
      fMatrix.push_back(arr[i]);
   }
}
