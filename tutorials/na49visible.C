{
//   Set visibility attributes for the NA49 geometry
//        Set Shape attributes
YK01->SetVisibility(0);
YK03->SetLineColor(2);
YK04->SetLineColor(5);
SEC1->SetLineColor(6);
SEC2->SetLineColor(6);
SEC3->SetLineColor(3);
SEC4->SetLineColor(3);
TOFR->SetLineColor(5);
COI1->SetLineColor(4);
COI2->SetLineColor(4);
COI3->SetLineColor(4);
COI4->SetLineColor(4);
CS38->SetLineColor(5);
CS28->SetLineColor(5);
CS18->SetLineColor(5);
TF4D->SetLineColor(3);
OGB4->SetLineColor(3);
TF3D->SetLineColor(3);
OGB3->SetLineColor(3);
TF4A->SetLineColor(3);
OGB4->SetLineColor(3);
TF3A->SetLineColor(3);
OGB3->SetLineColor(3);

//   Copy shape attributes (colors,etc) in nodes referencing the shapse
CAVE1->ImportShapeAttributes();

//  Set Node attributes
CAVE1->SetVisibility(2);   //node is not drawn but its sons are drawn
VT1_1->SetVisibility(-4);  //Node is not drawn. Its immediate sons are drawn
VT2_1->SetVisibility(-4);
MTL_1->SetVisibility(-4);
MTR_1->SetVisibility(-4);
TOFR1->SetVisibility(-4);
}
