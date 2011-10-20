// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 17/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
#include <stdexcept>
#include <utility>
#include <memory>

#include "IOSFileScanner.h"
#include "TMultiGraph.h"
#include "TGraphPolar.h"
#include "TIterator.h"
#include "TFile.h"
#include "TList.h"
#include "TKey.h"
#include "TH1.h"
#include "TF2.h"

namespace ROOT {
namespace iOS {
namespace FileUtils {

//__________________________________________________________________________________________________________________________
TObject *ReadObjectForKey(TFile *inputFile, const TKey *key, TString &option)
{
   option = "";

   TObject *objPtr = inputFile->Get(key->GetName());
   if (!objPtr)
      throw std::runtime_error("bad key in ReadObjectForKey");
   //Objects of some types are onwed by the file. So I have to make
   //them free from such ownership to make
   //their processing later more uniform.
   if (TH1 *hist = dynamic_cast<TH1 *>(objPtr))
      hist->SetDirectory(0);

   //The code below can throw, so I use auto_ptr.
   std::auto_ptr<TObject> obj(objPtr);

   //This is the trick, since ROOT seems not to preserve
   //Draw's option in a file.
   if (dynamic_cast<TF2 *>(obj.get()))
      option = "surf1";
   if (dynamic_cast<TMultiGraph *>(obj.get()))
      option = "acp";
   
   //All this "home-made memory management" is an ugly and broken thing.
   obj->SetBit(kCanDelete, kFALSE);
   obj->SetBit(kMustCleanup, kFALSE);

   return obj.release();
}


//__________________________________________________________________________________________________________________________
void ScanFileForVisibleObjects(TFile *inputFile, const std::set<TString> &visibleTypes, std::vector<TObject *> &objects, std::vector<TString> &options)
{
   //Find objects of visible types in a root file.
   const TList *keys = inputFile->GetListOfKeys();
   TIter next(keys);
   std::vector<TObject *>tmp;
   std::vector<TString> opts;
   TString option;
   
   try {
      std::auto_ptr<TObject> newObject;
      
      while (const TKey *k = static_cast<TKey *>(next())) {
         //Check, if object, pointed by the key, is supported.
         if (visibleTypes.find(k->GetClassName()) != visibleTypes.end()) {
            newObject.reset(ReadObjectForKey(inputFile, k, option));//can throw std::runtimer_error (me) || std::bad_alloc (ROOT)
            tmp.push_back(newObject.get());//bad_alloc.
            opts.push_back(option);
            newObject.release();
         }
      }
   } catch (const std::exception &) {
      for (std::vector<TObject*>::size_type i = 0; i < tmp.size(); ++i)
         delete tmp[i];
      throw;
   }
   
   objects.swap(tmp);
   options.swap(opts);
}

}//namespace FileUtils
}//namespace iOS
}//namespace ROOT
