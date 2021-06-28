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
   Internal::RIOShared<TObject> fObj;          ///< The object to be painted, owned by the drawable
   const TObject *fExtObj{nullptr};            ///<! external object, managed outside of the drawable, not persistent
   RAttrValue<std::string> fOpt{this, "opt"};  ///<! object draw options

   static std::string GetColorCode(TColor *col);

   std::unique_ptr<TObject> CreateSpecials(int kind);

protected:

   void CollectShared(Internal::RIOSharedVector_t &vect) final { vect.emplace_back(&fObj); }

   std::unique_ptr<RDisplayItem> Display(const RDisplayContext &) override;

   void PopulateMenu(RMenuItems &) final;

   void Execute(const std::string &) final;

   static void ExtractObjectColors(std::unique_ptr<TObjectDisplayItem> &item, const TObject *obj);

   static void CheckOwnership(TObject *obj);

   static std::string DetectCssType(const TObject *obj);

public:
   // special kinds, see TWebSnapshot enums
   enum EKind {
      kColors = 4,   ///< list of ROOT colors
      kStyle = 5,    ///< instance of TStyle object
      kPalette = 6   ///< list of colors from palette
   };

   RAttrLine line{this, "line"};          ///<! object line attributes
   RAttrFill fill{this, "fill"};          ///<! object fill attributes
   RAttrMarker marker{this, "marker"};    ///<! object marker attributes
   RAttrText text{this, "text"};          ///<! object text attributes

   TObjectDrawable();
   TObjectDrawable(TObject *obj, bool isowner = false);
   TObjectDrawable(TObject *obj, const std::string &opt, bool isowner = false);
   TObjectDrawable(const std::shared_ptr<TObject> &obj);
   TObjectDrawable(const std::shared_ptr<TObject> &obj, const std::string &opt);
   TObjectDrawable(EKind kind, bool persistent = false);
   virtual ~TObjectDrawable();

   void Reset();

   void Set(TObject *obj, bool isowner = false);
   void Set(TObject *obj, const std::string &opt, bool isowner = false);

   const TObject *Get();

   void SetOpt(const std::string &opt) { fOpt = opt; }
   std::string GetOpt() const { return fOpt; }

};

} // namespace Experimental
} // namespace ROOT


#endif
