/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsabl_RSysFileItem
#define ROOT7_Browsabl_RSysFileItem

#include <ROOT/Browsable/RItem.hxx>

namespace ROOT {
namespace Browsable {


class RSysDirLevelIter;

/** \class RSysFileItem
\ingroup rbrowser
\brief Representation of single item in the file browser
*/

class RSysFileItem : public RItem {

friend class RSysDirLevelIter;

private:
   // internal data, used for generate directory list
   int type{0};             ///<! file type
   int uid{0};              ///<! file uid
   int gid{0};              ///<! file gid
   bool islink{false};      ///<! true if symbolic link
   bool isdir{false};       ///<! true if directory
   long modtime{0};         ///<! modification time
   int64_t size{0};         ///<! file size

protected:
   // this is part for browser, visible for I/O
   std::string ftype;    ///< file attributes
   std::string fuid;     ///< user id
   std::string fgid;     ///< group id

public:

   /** Default constructor */
   RSysFileItem() = default;

   RSysFileItem(const std::string &_name, int _nchilds) : RItem(_name, _nchilds) {}

   // should be here, one needs virtual table for correct streaming of RRootBrowserReply
   virtual ~RSysFileItem() = default;

   void SetType(const std::string &_type) { ftype  = _type; }
   void SetUid(const std::string &_uid) { fuid  = _uid; }
   void SetGid(const std::string &_gid) { fgid  = _gid; }

   const std::string &GetType() const { return ftype; }
   const std::string &GetUid() const { return fuid; }
   const std::string &GetGid() const { return fgid; }


   // only subdir is folder for files items
   bool IsFolder() const override { return isdir; }

   // return true for hidden files
   bool IsHidden() const override
   {
      auto &n = GetName();
      if ((n.length() == 0) || (n[0] != '.')) return false;
      return (n != ".") && (n != "..");
   }

   bool Compare(const RItem *b, const std::string &method) const override
   {
      if (IsFolder() != b->IsFolder())
         return IsFolder();

      if (method == "size") {
         auto fb = dynamic_cast<const RSysFileItem *> (b);
         if (fb)
            return size > fb->size;
      }

      return GetName() < b->GetName();
   }
};

} // namespace Browsable
} // namespace ROOT


#endif
