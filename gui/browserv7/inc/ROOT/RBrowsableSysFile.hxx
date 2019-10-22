/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowsableSysFile
#define ROOT7_RBrowsableSysFile

#include <ROOT/RBrowsable.hxx>

#include "TSystem.h"
#include <string>

namespace ROOT {
namespace Experimental {

/** Representation of single item in the file browser */
class RBrowserFileItem : public RBrowserItem {
public:
   // internal data, used for generate directory list
   int type{0};             ///<! file type
   int uid{0};              ///<! file uid
   int gid{0};              ///<! file gid
   bool islink{false};      ///<! true if symbolic link
   bool isdir{false};       ///<! true if directory
   long modtime{0};         ///<! modification time
   int64_t size{0};         ///<! file size

   // this is part for browser, visible for I/O
   std::string fsize;    ///< file size
   std::string mtime;    ///< modification time
   std::string ftype;    ///< file attributes
   std::string fuid;     ///< user id
   std::string fgid;     ///< group id

   RBrowserFileItem() = default;

   RBrowserFileItem(const std::string &_name, int _nchilds) : RBrowserItem(_name, _nchilds) {}

   // should be here, one needs virtual table for correct streaming of RRootBrowserReply
   virtual ~RBrowserFileItem() = default;
};



namespace Browsable {

// ========================================================================================

class SysFileElement : public RElement {
   FileStat_t fStat;       ///<! file stat object
   std::string fDirName;   ///<! fully-qualified directory name
   std::string fFileName;  ///<! file name in current dir

   std::string GetFullName() const;

public:
   SysFileElement(const std::string &filename);

   SysFileElement(const FileStat_t& stat, const std::string &dirname, const std::string &filename) : fStat(stat), fDirName(dirname), fFileName(filename)
   {
   }

   virtual ~SysFileElement() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   std::string GetName() const override { return fFileName; }

   /** Title of RBrowsable (optional) */
   std::string GetTitle() const override { return GetFullName(); }

   std::unique_ptr<RLevelIter> GetChildsIter() override;

   std::string GetContent(const std::string &kind) override;
};

} // namespace Browsable
} // namespace Experimental
} // namespace ROOT


#endif
