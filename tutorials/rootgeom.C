void rootgeom()
{
//--- Definition of a simple geometry
   gSystem->Load("libGeom");
   TGeoManager *geom = new TGeoManager("simple1", "Simple geometry");
   
   //--- define some materials
   TGeoMaterial *mat;
   mat = new TGeoMaterial("mat1", "Vacuum", 0,0,0);
   mat = new TGeoMaterial("mat2", "Al", 26.98,13,2.7);

   //--- define the transformations
   TGeoTranslation *tr1 = new TGeoTranslation(20., 0, 0.);
   TGeoTranslation *tr2 = new TGeoTranslation(10., 0., 0.);
   TGeoTranslation *tr3 = new TGeoTranslation(10., 20., 0.);
   TGeoTranslation *tr4 = new TGeoTranslation(5., 10., 0.);
   TGeoTranslation *tr5 = new TGeoTranslation(20., 0., 0.);
   TGeoTranslation *tr6 = new TGeoTranslation(-5., 0., 0.);
   TGeoTranslation *tr7 = new TGeoTranslation(7.5, 7.5, 0.);
   TGeoRotation   *rot1 = new TGeoRotation("rot1", 90., 0., 90., 270., 0., 0.);
   TGeoCombiTrans *combi1 = new TGeoCombiTrans(7.5, -7.5, 0., rot1);
   TGeoTranslation *tr8 = new TGeoTranslation(7.5, -5., 0.);
   TGeoTranslation *tr9 = new TGeoTranslation(7.5, 20., 0.);
   TGeoTranslation *tr10 = new TGeoTranslation(85., 0., 0.);
   TGeoTranslation *tr11 = new TGeoTranslation(35., 0., 0.);
   TGeoTranslation *tr12 = new TGeoTranslation(-15., 0., 0.);
   TGeoTranslation *tr13 = new TGeoTranslation(-65., 0., 0.);
   
   TGeoTranslation  *tr14 = new TGeoTranslation(0,0,-100);
   TGeoCombiTrans *combi2 = new TGeoCombiTrans(0,0,100, 
                                   new TGeoRotation("rot2",90,180,90,90,180,0));
   TGeoCombiTrans *combi3 = new TGeoCombiTrans(100,0,0, 
                                   new TGeoRotation("rot3",90,270,0,0,90,180));
   TGeoCombiTrans *combi4 = new TGeoCombiTrans(-100,0,0, 
                                   new TGeoRotation("rot4",90,90,0,0,90,0));
   TGeoCombiTrans *combi5 = new TGeoCombiTrans(0,100,0, 
                                   new TGeoRotation("rot5",0,0,90,180,90,270));
   TGeoCombiTrans *combi6 = new TGeoCombiTrans(0,-100,0, 
                                   new TGeoRotation("rot6",180,0,90,180,90,90));
   
   //--- make the top container volume
   Double_t worldx = 110.;
   Double_t worldy = 50.;
   Double_t worldz = 5.;
   TGeoVolume *top = geom->MakeBox("TOP", "mat1", 270., 270., 120.);
   geom->SetTopVolume(top);
   TGeoVolume *replica = geom->MakeBox("REPLICA", "mat1",120,120,120);
   replica->SetVisibility(kFALSE);
   TGeoVolume *rootbox = geom->MakeBox("ROOT", "mat1", 110., 50., 5.);
   rootbox->SetVisibility(kFALSE);
   
   //--- make letter 'R'
   TGeoVolume *R = geom->MakeBox("R", "mat1", 25., 25., 5.);
   R->SetVisibility(kFALSE);
   TGeoVolume *bar1 = geom->MakeBox("bar1", "mat2", 5., 25, 5.);
   bar1->SetLineColor(kRed);
   R->AddNode(bar1, 1, tr1);
   TGeoVolume *bar2 = geom->MakeBox("bar2", "mat2", 5., 5., 5.);
   bar2->SetLineColor(kRed);
   R->AddNode(bar2, 1, tr2);
   R->AddNode(bar2, 2, tr3);
   TGeoVolume *tub1 = geom->MakeTubs("tub1", "mat2", 5., 15., 5., 90., 270.);
   tub1->SetLineColor(kRed);
   R->AddNode(tub1, 1, tr4);
   TGeoVolume *bar3 = geom->MakeArb8("bar3", "mat2", 5.);
   bar3->SetLineColor(kRed);
   TGeoArb8 *arb = (TGeoArb8*)bar3->GetShape();
   arb->SetVertex(0, 15., -5.);
   arb->SetVertex(1, 5., -5.);
   arb->SetVertex(2, -10., -25.);
   arb->SetVertex(3, 0., -25.);
   arb->SetVertex(4, 15., -5.);
   arb->SetVertex(5, 5., -5.);
   arb->SetVertex(6, -10., -25.);
   arb->SetVertex(7, 0., -25.);
   R->AddNode(bar3, 1, gGeoIdentity);
   
   //--- make letter 'O'
   TGeoVolume *O = geom->MakeBox("O", "mat1", 25., 25., 5.);
   O->SetVisibility(kFALSE);
   TGeoVolume *bar4 = geom->MakeBox("bar4", "mat2", 5., 7.5, 5.);
   bar4->SetLineColor(kYellow);
   O->AddNode(bar4, 1, tr5);
   O->AddNode(bar4, 2, tr6);
   TGeoVolume *tub2 = geom->MakeTubs("tub1", "mat2", 7.5, 17.5, 5., 0., 180.);
   tub2->SetLineColor(kYellow);
   O->AddNode(tub2, 1, tr7);
   O->AddNode(tub2, 2, combi1);
   
   //--- make letter 'T'
   TGeoVolume *T = geom->MakeBox("T", "mat1", 25., 25., 5.);
   T->SetVisibility(kFALSE);
   TGeoVolume *bar5 = geom->MakeBox("bar5", "mat2", 5., 20., 5.);
   bar5->SetLineColor(kBlue);
   T->AddNode(bar5, 1, tr8);
   TGeoVolume *bar6 = geom->MakeBox("bar6", "mat2", 17.5, 5., 5.);
   bar6->SetLineColor(kBlue);
   T->AddNode(bar6, 1, tr9);
      
   

   rootbox->AddNode(R, 1, tr10);
   rootbox->AddNode(O, 1, tr11);
   rootbox->AddNode(O, 2, tr12);
   rootbox->AddNode(T, 1, tr13);
   
   replica->AddNode(rootbox, 1, tr14);
   replica->AddNode(rootbox, 2, combi2);
   replica->AddNode(rootbox, 3, combi3);
   replica->AddNode(rootbox, 4, combi4);
   replica->AddNode(rootbox, 5, combi5);
   replica->AddNode(rootbox, 6, combi6);
   
   top->AddNode(replica, 1, new TGeoTranslation(-150, -150, 0));
   top->AddNode(replica, 2, new TGeoTranslation(150, -150, 0));
   top->AddNode(replica, 3, new TGeoTranslation(150, 150, 0));
   top->AddNode(replica, 4, new TGeoTranslation(-150, 150, 0));

   //--- close the geometry
   geom->CloseGeometry();
   
   //--- draw the ROOT box
   geom->SetVisLevel(4);
   top->Draw();
   if (gPad) gPad->x3d();
}   
   