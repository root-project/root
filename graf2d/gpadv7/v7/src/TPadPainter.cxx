/// \file TPadPainter.cxx
/// \ingroup Gpad ROOT7
/// \author Sergey Linev
/// \date 2018-03-12
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include <ROOT/TPadPainter.hxx>

#include <ROOT/TPadDisplayItem.hxx>
#include <ROOT/TPad.hxx>


/// destructor
ROOT::Experimental::Internal::TPadPainter::~TPadPainter()
{
   // defined here, while TPadDisplayItem only included here
}


void ROOT::Experimental::Internal::TPadPainter::AddDisplayItem(std::unique_ptr<TDisplayItem> &&item)
{
   item->SetObjectID(fCurrentDrawableId);
   fPadDisplayItem->Add(std::move(item));
}

void ROOT::Experimental::Internal::TPadPainter::PaintDrawables(const TPadBase &pad)
{
   fPadDisplayItem = std::make_unique<TPadDisplayItem>();

   fPadDisplayItem->SetObjectIDAsPtr((void *) (&pad));

   fPadDisplayItem->SetFrame(pad.GetFrame());

   for (auto &&drawable : pad.GetPrimitives()) {

      fCurrentDrawableId = TDisplayItem::MakeIDFromPtr(drawable.get());

      drawable->Paint(*this);
   }

}
