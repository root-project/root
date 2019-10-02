/// \file ROOT/RMenuItem.hxx
/// \ingroup Base ROOT7
/// \author Sergey Linev
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RMenuItem
#define ROOT7_RMenuItem

#include <string>
#include <vector>
#include <memory>

class TClass;

namespace ROOT {
namespace Experimental {
namespace Detail {

/** \class RMenuItem
  Class contains info for producing menu item on the JS side.
  */

class RMenuItem {
protected:
   std::string fName;  ///<  name of the menu item
   std::string fTitle; ///<  title of menu item
   std::string fExec;  ///< execute when item is activated
public:
   /** Default constructor */
   RMenuItem() = default;

   /** Create menu item with the name and title
    *  name used to display item in the object context menu,
    *  title shown as hint info for that item  */
   RMenuItem(const std::string &name, const std::string &title) : fName(name), fTitle(title), fExec() {}

   /** virtual destructor need for vtable, used when vector of RMenuItem* is stored */
   virtual ~RMenuItem() {}

   /** Set execution string with all required arguments,
    * which will be executed when menu item is selected  */
   void SetExec(const std::string &exec) { fExec = exec; }

   /** Returns menu item name */
   const std::string &GetName() const { return fName; }

   /** Returns execution string for the menu item */
   const std::string &GetExec() const { return fExec; }
};

class RCheckedMenuItem : public RMenuItem {
protected:
   bool fChecked = false; ///< -1 not exists, 0 - off, 1 - on
public:
   /** Default constructor */
   RCheckedMenuItem() = default;

   /** Create checked menu item  */
   RCheckedMenuItem(const std::string &name, const std::string &title, bool checked = false)
      : RMenuItem(name, title), fChecked(checked)
   {
   }

   /** virtual destructor need for vtable, used when vector of RMenuItem* is stored */
   virtual ~RCheckedMenuItem() {}

   /** Set checked state for the item, default is none */
   void SetChecked(bool on = true) { fChecked = on; }

   bool IsChecked() const { return fChecked; }
};

class RMenuArgument {
protected:
   std::string fName;     ///<  name of call argument
   std::string fTitle;    ///<  title of call argument
   std::string fTypeName; ///<  typename
   std::string fDefault;  ///<  default value
public:
   /** Default constructor */
   RMenuArgument() = default;

   RMenuArgument(const std::string &name, const std::string &title, const std::string &typname,
                 const std::string &dflt = "")
      : fName(name), fTitle(title), fTypeName(typname), fDefault(dflt)
   {
   }

   void SetDefault(const std::string &dflt) { fDefault = dflt; }
};

class RArgsMenuItem : public RMenuItem {
protected:
   std::vector<RMenuArgument> fArgs;

public:
   /** Default constructor */
   RArgsMenuItem() = default;

   RArgsMenuItem(const std::string &name, const std::string &title) : RMenuItem(name, title) {}

   /** virtual destructor need for vtable, used when vector of RMenuItem* is stored */
   virtual ~RArgsMenuItem() {}

   void AddArg(const RMenuArgument &arg) { fArgs.emplace_back(arg); }
};

} // namespace Detail

///////////////////////////////////////////////////////////////////////

class RMenuItems {
protected:
   std::string fId;                                        ///< object identifier
   std::vector<std::unique_ptr<Detail::RMenuItem>> fItems; ///< list of items in the menu
public:
   void SetId(const std::string &id) { fId = id; }

   auto Size() const { return fItems.size(); }

   void Add(std::unique_ptr<Detail::RMenuItem> &&item) { fItems.emplace_back(std::move(item)); }

   void AddMenuItem(const std::string &name, const std::string &title, const std::string &exec)
   {
      auto item = std::make_unique<Detail::RMenuItem>(name, title);
      item->SetExec(exec);
      Add(std::move(item));
   }

   void AddChkMenuItem(const std::string &name, const std::string &title, bool checked, const std::string &toggle)
   {
      auto item = std::make_unique<Detail::RCheckedMenuItem>(name, title, checked);
      item->SetExec(toggle);
      Add(std::move(item));
   }

   void PopulateObjectMenu(void *obj, TClass *cl);
};

} // namespace Experimental
} // namespace ROOT

#endif
