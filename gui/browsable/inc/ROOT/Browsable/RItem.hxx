/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RItem
#define ROOT7_Browsable_RItem

#include <string>
#include "RtypesCore.h"

namespace ROOT {
namespace Experimental {
namespace Browsable {

/** \class RItem
\ingroup rbrowser
\brief Representation of single item in the browser
*/

class RItem {
protected:
   std::string name;     ///< item name
   int nchilds{0};       ///< number of childs
   std::string icon;     ///< icon associated with item
   std::string title;    ///< item title
   std::string fsize;    ///< item size
   bool checked{false};  ///< is checked, not yet used
   bool expanded{false}; ///< is expanded
public:

   RItem() = default;
   RItem(const std::string &_name, int _nchilds = 0, const std::string &_icon = "") : name(_name), nchilds(_nchilds), icon(_icon) {}
   // must be here, one needs virtual table for correct streaming of sub-classes
   virtual ~RItem() = default;

   const std::string &GetName() const { return name; }
   const std::string &GetIcon() const { return icon; }
   const std::string &GetTitle() const { return title; }
   const std::string &GetSize() const { return fsize; }
   virtual bool IsFolder() const { return nchilds != 0; }
   virtual bool IsHidden() const { return false; }

   void SetChecked(bool on = true) { checked = on; }
   void SetExpanded(bool on = true) { expanded = on; }

   void SetName(const std::string &_name) { name = _name; }
   void SetTitle(const std::string &_title) { title = _title; }
   void SetIcon(const std::string &_icon) { icon = _icon; }
   void SetSize(const std::string &_size) { fsize = _size; }

   void SetSize(Long64_t _size)
   {
      if (_size > 1024) {
         Long64_t _ksize = _size / 1024;
         if (_ksize > 1024) {
            // 3.7MB is more informative than just 3MB
            fsize = std::to_string(_ksize/1024) + "." + std::to_string((_ksize%1024)/103) + "M";
         } else {
            fsize = std::to_string(_ksize) + "." + std::to_string((_size%1024)/103) + "K";
         }
      } else {
         fsize = std::to_string(_size);
      }
   }

   virtual bool Compare(const RItem *b, const std::string &) const
   {
      if (IsFolder() != b->IsFolder())
         return IsFolder();
      return GetName() < b->GetName();
   }
};


} // namespace Browsable
} // namespace Experimental
} // namespace ROOT

#endif


