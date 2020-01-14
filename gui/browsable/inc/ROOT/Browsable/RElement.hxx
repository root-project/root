/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RElement
#define ROOT7_Browsable_RElement

#include <ROOT/Browsable/RHolder.hxx>

#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

using RElementPath_t = std::vector<std::string>;

namespace Browsable {


class RLevelIter;


/** \class RElement
\ingroup rbrowser
\brief Basic element of RBrowsable hierarchy. Provides access to data, creates iterator if any
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RElement {
public:

   enum EContentKind {
      kNone,      ///< not recognized
      kText,      ///< "text" - plain text for code editor
      kImage,     ///< "image64" - base64 for supported image formats (png/gif/gpeg)
      kPng,       ///< "png" - plain png binary code, returned inside std::string
      kJpeg,      ///< "jpg" or "jpeg" - plain jpg binary code, returned inside std::string
      kFileName   ///< "filename" - file name if applicable
   };

   static EContentKind GetContentKind(const std::string &kind);

   virtual ~RElement() = default;

   /** Name of RBrowsable, must be provided in derived classes */
   virtual std::string GetName() const = 0;

   /** Checks if element name match to provided value */
   virtual bool MatchName(const std::string &name) const { return name == GetName(); }

   /** Title of RBrowsable (optional) */
   virtual std::string GetTitle() const { return ""; }

   /** Create iterator for childs elements if any */
   virtual std::unique_ptr<RLevelIter> GetChildsIter() { return nullptr; }

   /** Returns element content, depends from kind. Can be "text" or "image64" */
   virtual std::string GetContent(const std::string & = "text") { return ""; }

   /** Access object */
   virtual std::unique_ptr<RHolder> GetObject() { return nullptr; }

   static std::shared_ptr<RElement> GetSubElement(std::shared_ptr<RElement> &elem, const RElementPath_t &path);
};

} // namespace Browsable
} // namespace Experimental
} // namespace ROOT

#endif
