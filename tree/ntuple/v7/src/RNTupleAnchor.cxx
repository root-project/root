/// \file RNTupleAnchor.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2023-09-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleAnchor.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>

#include <TBuffer.h>
#include <TError.h>
#include <TFile.h>

void ROOT::Experimental::RNTuple::Streamer(TBuffer &buf)
{
   if (buf.IsReading()) {
      RNTuple::Class()->ReadBuffer(buf, this);
      R__ASSERT(buf.GetParent() && buf.GetParent()->InheritsFrom("TFile"));
      fFile = static_cast<TFile *>(buf.GetParent());
   } else {
      RNTuple::Class()->WriteBuffer(buf, this);
   }
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource>
ROOT::Experimental::RNTuple::MakePageSource(const RNTupleReadOptions &options)
{
   if (!fFile)
      throw RException(R__FAIL("This RNTuple object was not streamed from a file"));

   // TODO(jblomer): Add RRawFile factory that create a raw file from a TFile. This may then duplicate the file
   // descriptor (to avoid re-open).  There could also be a raw file that uses a TFile as a "backend" for TFile cases
   // that are unsupported by raw file.
   auto path = fFile->GetEndpointUrl()->GetFile();
   return Detail::RPageSourceFile::CreateFromAnchor(*this, path, options);
}
