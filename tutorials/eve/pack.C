// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstrates usage of class TGPack.

TGPack *hp = 0;
TGPack *vp = 0;

TGTextButton* b = 0;

void pack()
{
   TGMainFrame* mf = new TGMainFrame(0, 400, 300);
   mf->SetWindowName("Foo");

   hp = new TGPack(mf, mf->GetWidth(), mf->GetHeight());
   hp->SetVertical(kFALSE);

   b = new TGTextButton(hp, "Ailaaha");  hp->AddFrame(b);

   vp = new TGPack(hp, hp->GetWidth(), hp->GetHeight());
   b = new TGTextButton(vp, "Blaaaaa");  vp->AddFrameWithWeight(b, 0, 5);
   b = new TGTextButton(vp, "Blooooo");  vp->AddFrameWithWeight(b, 0, 3);
   b = new TGTextButton(vp, "Bleeeee");  vp->AddFrameWithWeight(b, 0, 5);
   hp->AddFrame(vp, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   b = new TGTextButton(hp, "Cilnouk");  hp->AddFrame(b);

   mf->AddFrame(hp, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   mf->Layout();
   mf->MapSubwindows();
   mf->MapWindow();
}
