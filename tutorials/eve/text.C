/// \file
/// \ingroup tutorial_eve
/// Demonstrates usage of class TEveText - 2D & 3D text in GL.
///
/// \image html eve_text.png
/// \macro_code
///
/// \author Alja Mrak-Tadel

TEveText* text()
{
   gSystem->IgnoreSignal(kSigSegmentationViolation, true);

   TEveManager::Create();

   auto marker = new TEvePointSet(8);
   marker->SetName("Origin marker");
   marker->SetMarkerColor(6);
   marker->SetMarkerStyle(3);
   Float_t a = 10;
   marker->SetPoint(0, a,  +a, +a);
   marker->SetPoint(1, a,  -a, +a);
   marker->SetPoint(2, -a, -a, +a);
   marker->SetPoint(3, -a, +a, +a);
   marker->SetPoint(4, +a, +a, -a);
   marker->SetPoint(5, +a, -a, -a);
   marker->SetPoint(6, -a, +a, -a);
   marker->SetPoint(7, -a, -a, -a);
   gEve->AddElement(marker);

   auto t = new TEveText("DADA");
   t->PtrMainTrans()->RotateLF(1, 3, TMath::PiOver2());
   t->SetMainColor(kOrange-2);
   t->SetFontSize(64);
   t->SetFontMode(TGLFont::kExtrude);
   t->SetLighting(kTRUE);
   gEve->AddElement(t);

   // TEveText does not know its bounding box before first rendering.
   gEve->FullRedraw3D(kTRUE);
   gEve->GetDefaultGLViewer()->ResetCurrentCamera();
   gEve->GetDefaultGLViewer()->RequestDraw(TGLRnrCtx::kLODHigh);

   return t;
}
