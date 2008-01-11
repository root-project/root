// Author: Matevz Tadel

TEvePointSet* pointset_test(Int_t npoints = 512) 
{
   TEveManager::Create();

   TRandom r(0);
   Float_t s = 100;

   TEvePointSet* ps = new TEvePointSet();
   ps->SetOwnIds(kTRUE);

   for(Int_t i = 0; i<npoints; i++)
   {
      ps->SetNextPoint(r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
      ps->SetPointId(new TNamed(Form("Point %d", i), ""));
   }

   ps->SetMarkerColor(5);
   ps->SetMarkerSize(1.5);
   ps->SetMarkerStyle(4);

   gEve->AddElement(ps);
   gEve->Redraw3D();

   return ps;
}
