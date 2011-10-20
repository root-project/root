// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 17/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_IOSFileContainer
#define ROOT_IOSFileContainer

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// FileContainer                                                        //
//                                                                      //
// Class which owns objects read from root file.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <memory>
#include <string>

#ifndef ROOT_TString
#include "TString.h"
#endif

//
//TODO: These classes should be removed from graf2d/ios as an application-specific
//code which can be placed where it's used - in a RootBrowser application for iPad.
//

class TObject;
class TFile;

namespace ROOT {
namespace iOS {

class Pad;

enum EHistogramErrorOption {
   hetNoError,
   hetE,
   hetE1,
   hetE2,
   hetE3,
   hetE4
};


class FileContainer {
public:
   typedef std::vector<TObject *>::size_type size_type;

   FileContainer(const std::string &fileName);
   ~FileContainer();

   size_type GetNumberOfObjects()const;
   TObject *GetObject(size_type ind)const;
   const char *GetDrawOption(size_type ind)const;
   Pad *GetPadAttached(size_type ind)const;

   void SetErrorDrawOption(size_type ind, EHistogramErrorOption opt);
   EHistogramErrorOption GetErrorDrawOption(size_type ind)const;
   
   void SetMarkerDrawOption(size_type ind, bool on);
   bool GetMarkerDrawOption(size_type ind)const;

   const char *GetFileName()const;

private:
   std::string fFileName;

   std::auto_ptr<TFile> fFileHandler;
   std::vector<TObject *> fFileContents;
   std::vector<TString> fOptions;
   
   std::vector<Pad *> fAttachedPads;
};

//This is the function to be called from Obj-C++ code.
//Return: non-null pointer in case file was
//opened and its content read.
FileContainer *CreateFileContainer(const char *fileName);
//Just for symmetry.
void DeleteFileContainer(FileContainer *container);

}//namespace iOS
}//namespace ROOT

#endif
