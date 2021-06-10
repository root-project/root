/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TObjectDisplayItem
#define ROOT7_TObjectDisplayItem

#include <ROOT/RDisplayItem.hxx>
#include <ROOT/RDrawable.hxx>

#include <string>
#include <vector>
#include "TObject.h"

namespace ROOT {
namespace Experimental {


/** \class TObjectDisplayItem
\ingroup GpadROOT7
\brief Display item for TObject with drawing options
\author Sergey Linev <s.linev@gsi.de>
\date 2017-05-31
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class TObjectDisplayItem : public RIndirectDisplayItem {
protected:

   int fKind{0};                           ///< object kind
   const TObject *fObject{nullptr};        ///< ROOT6 object
   std::string fCssType;                   ///< CSS type
   std::string fOption;                    ///< drawing options
   bool fOwner{false};                     ///<! if object must be deleted
   std::vector<int> fColIndex;             ///< stored color index
   std::vector<std::string> fColValue;     ///< stored color value

public:

   /// normal constructor, also copies drawable id and csstype
   TObjectDisplayItem(const RDrawable &dr, int kind, const TObject *obj, const std::string &opt) : RIndirectDisplayItem(dr)
   {
      fCssType = dr.GetCssType();
      fKind = kind;
      fObject = obj;
      fOption = opt;
   }

   /// constructor for special objects like palette, takes ownership!!
   TObjectDisplayItem(int kind, const TObject *obj, const std::string &opt) : RIndirectDisplayItem()
   {
      fKind = kind;
      fObject = obj;
      fOption = opt;
      fOwner = true;
   }

   virtual ~TObjectDisplayItem()
   {
      if (fOwner) delete fObject;
   }

   void AddTColor(int color_indx, const std::string &color_value)
   {
      fColIndex.emplace_back(color_indx);
      fColValue.emplace_back(color_value);
   }

};

} // namespace Experimental
} // namespace ROOT

#endif
