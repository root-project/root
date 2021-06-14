/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TObjectDrawable
#define ROOT7_TObjectDrawable

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RAttrMarker.hxx>
#include <ROOT/RAttrFill.hxx>

class TObject;
class TColor;
class TClass;

namespace ROOT {
namespace Experimental {

class RPadBase;
class TObjectDisplayItem;

/** \class TObjectDrawable
\ingroup GpadROOT7
\brief Provides v7 drawing facilities for TObject types (TGraph, TH1, TH2, etc).
\author Sergey Linev
\date 2017-05-31
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class TObjectDrawable final : public RDrawable {
private:

   enum {
      kNone = 0,     ///< empty container
      kObject = 1,   ///< plain object
   };

   int fKind{kNone};                           ///< object kind
   Internal::RIOShared<TObject> fObj;          ///< The object to be painted
   RAttrValue<std::string> fOpt{this, "opt"};  ///<! object draw options
   RAttrLine fAttrLine{this, "line"};          ///<! object line attributes
   RAttrFill fAttrFill{this, "fill"};          ///<! object fill attributes
   RAttrText fAttrText{this, "text"};          ///<! object text attributes
   RAttrMarker fMarkerAttr{this, "marker"};    ///<! object marker attributes

   const char *GetColorCode(TColor *col);

   std::unique_ptr<TObject> CreateSpecials(int kind);

protected:

   void CollectShared(Internal::RIOSharedVector_t &vect) final { vect.emplace_back(&fObj); }

   std::unique_ptr<RDisplayItem> Display(const RDisplayContext &) override;

   void PopulateMenu(RMenuItems &) final;

   void Execute(const std::string &) final;

   static void ExtractTColor(std::unique_ptr<TObjectDisplayItem> &item, TObject *obj, const char *class_name, const char *class_member);

   static void ExtractObjectColors(std::unique_ptr<TObjectDisplayItem> &item, TObject *obj);

   static std::string DetectCssType(const TObject *obj);

public:
   // special kinds, see TWebSnapshot enums
   enum EKind {
      kColors = 4,   ///< list of ROOT colors
      kStyle = 5,    ///< instance of TStyle object
      kPalette = 6   ///< list of colors from palette
   };

   TObjectDrawable(const std::shared_ptr<TObject> &obj) : RDrawable(DetectCssType(obj.get()))
   {
      fKind = kObject;
      fObj = obj;
   }

   TObjectDrawable(const std::shared_ptr<TObject> &obj, const std::string &opt) : RDrawable(DetectCssType(obj.get()))
   {
      fKind = kObject;
      fObj = obj;
      SetOpt(opt);
   }

   /// Constructor takes ownership
   TObjectDrawable(TObject *obj) : RDrawable(DetectCssType(obj))
   {
      fKind = kObject;
      fObj = std::shared_ptr<TObject>(obj);
   }

   /// Constructor takes ownership
   TObjectDrawable(TObject *obj, const std::string &opt) : RDrawable(DetectCssType(obj))
   {
      fKind = kObject;
      fObj = std::shared_ptr<TObject>(obj);
      SetOpt(opt);
   }

   TObjectDrawable(EKind kind, bool persistent = false) : RDrawable("tobject")
   {
      fKind = kind;

      if (persistent)
         fObj = CreateSpecials(kind);
   }

   virtual ~TObjectDrawable() = default;

   std::shared_ptr<TObject> GetObject() const { return fObj.get_shared(); }

   void SetOpt(const std::string &opt) { fOpt = opt; }
   std::string GetOpt() const { return fOpt; }

   const RAttrLine &AttrLine() const { return fAttrLine; }
   RAttrLine &AttrLine() { return fAttrLine; }

   const RAttrFill &AttrFill() const { return fAttrFill; }
   RAttrFill &AttrFill() { return fAttrFill; }

   const RAttrText &AttrText() const { return fAttrText; }
   RAttrText &AttrText() { return fAttrText; }

   const RAttrMarker &AttrMarker() const { return fMarkerAttr; }
   RAttrMarker &AttrMarker() { return fMarkerAttr; }
};

} // namespace Experimental
} // namespace ROOT


#endif
