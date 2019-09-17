/// \file RPadPainter.cxx
/// \ingroup Gpad ROOT7
/// \author Sergey Linev
/// \date 2018-03-12
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include <ROOT/RPadPainter.hxx>

#include <ROOT/RPadDisplayItem.hxx>
#include <ROOT/RPad.hxx>


/// destructor
ROOT::Experimental::Internal::RPadPainter::~RPadPainter()
{
   // defined here, while TPadDisplayItem only included here
}


void ROOT::Experimental::Internal::RPadPainter::AddDisplayItem(std::unique_ptr<RDisplayItem> &&item)
{
   item->SetObjectID(fCurrentDrawableId);
   fPadDisplayItem->Add(std::move(item));
}

void ROOT::Experimental::Internal::RPadPainter::PaintDrawables(const RPadBase &pad)
{
   fPadDisplayItem = std::make_unique<RPadDisplayItem>();

   fPadDisplayItem->SetFrame(pad.GetFrame());

   auto primitives = pad.GetPrimitives();

   for (auto &drawable : primitives) {

      fCurrentDrawableId = drawable->GetId();

      drawable->Paint(*this);
   }

}
