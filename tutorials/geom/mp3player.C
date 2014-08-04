#include "TCanvas.h"
#include "TPaveText.h"
#include "TImage.h"
#include "TLine.h"
#include "TLatex.h"
#include "TButton.h"
#include "TGeoManager.h"

void mp3player()
{
  // Drawing a mp3 type music player, using ROOT geometry class.
  // Name: mp3player.C
  // Author: Eun Young Kim, Dept. of Physics, Univ. of Seoul
  // Reviewed by Sunman Kim (sunman98@hanmail.net)
  // Supervisor: Prof. Inkyu Park (icpark@physics.uos.ac.kr)
  //
  // How to run: .x mp3player.C in ROOT terminal, then use OpenGL
  //
  // This macro was created for the evaluation of Computational Physics course in 2006.
  // We thank to Prof. Inkyu Park for his special lecture on ROOT and to all of ROOT team
  //

  TGeoManager *geom=new TGeoManager("geom","My first 3D geometry");



  //materials
  TGeoMaterial *vacuum=new TGeoMaterial("vacuum",0,0,0);
  TGeoMaterial *Fe=new TGeoMaterial("Fe",55.845,26,7.87);

  //create media

  TGeoMedium *Iron=new TGeoMedium("Iron",1,Fe);
  TGeoMedium *Air=new TGeoMedium("Vacuum",0,vacuum);


 //create volume

  TGeoVolume *top=geom->MakeBox("top",Air,800,800,800);
  geom->SetTopVolume(top);
  geom->SetTopVisible(0);
  // If you want to see the boundary, please input the number, 1 instead of 0.
  // Like this, geom->SetTopVisible(1);



  TGeoVolume *b1=geom->MakeBox("b1",Iron,100,200,600);
  b1->SetLineColor(2);


  TGeoVolume *b2=geom->MakeTubs("b2",Iron,0,50,200,0,90);
  b2->SetLineColor(10);


  TGeoVolume *b3=geom->MakeTubs("b3",Iron,0,50,200,90,180);
  b3->SetLineColor(10);


  TGeoVolume *b4=geom->MakeTubs("b4",Iron,0,50,200,180,270);
  b4->SetLineColor(10);

  TGeoVolume *b5=geom->MakeTubs("b5",Iron,0,50,200,270,360);
  b5->SetLineColor(10);


  TGeoVolume *b6=geom->MakeTubs("b6",Iron,0,50,600,0,90);
  b6->SetLineColor(10);

  TGeoVolume *b7=geom->MakeTubs("b7",Iron,0,50,600,90,180);
  b7->SetLineColor(10);

  TGeoVolume *b8=geom->MakeTubs("b8",Iron,0,50,600,180,270);
  b8->SetLineColor(10);

  TGeoVolume *b9=geom->MakeTubs("b9",Iron,0,50,600,270,360);
  b9->SetLineColor(10);



  TGeoVolume *b10=geom->MakeTubs("b10",Iron,0,50,100,0,90);
  b10->SetLineColor(10);

  TGeoVolume *b11=geom->MakeTubs("b11",Iron,0,50,100,90,180);
  b11->SetLineColor(10);

  TGeoVolume *b12=geom->MakeTubs("b12",Iron,0,50,100,180,270);
  b12->SetLineColor(10);

  TGeoVolume *b13=geom->MakeTubs("b13",Iron,0,50,100,270,360);
  b13->SetLineColor(10);


  TGeoVolume *b14=geom->MakeBox("b14",Iron,100,50,450);
  b14->SetLineColor(10);
  TGeoVolume *b15=geom->MakeBox("b15",Iron,50,200,600);
  b15->SetLineColor(10);



  TGeoVolume *b16=geom->MakeSphere("b16",Iron,0,50,0,90,0,90);
  b16->SetLineColor(10);

  TGeoVolume *b17=geom->MakeSphere("b17",Iron,0,50,0,90,270,360);
  b17->SetLineColor(10);

  TGeoVolume *b18=geom->MakeSphere("b18",Iron,0,50,0,90,180,270);
  b18->SetLineColor(10);

  TGeoVolume *b19=geom->MakeSphere("b19",Iron,0,50,0,90,90,180);
  b19->SetLineColor(10);


  TGeoVolume *b20=geom->MakeTube("b20",Iron,50,150,150);
  b20->SetLineColor(17);



  TGeoVolume *b21=geom->MakeSphere("b21",Iron,0,50,90,180,0,90);
  b21->SetLineColor(10);

  TGeoVolume *b22=geom->MakeSphere("b22",Iron,0,50,90,180,270,360);
  b22->SetLineColor(10);

  TGeoVolume *b23=geom->MakeSphere("b23",Iron,0,50,90,180,180,270);
  b23->SetLineColor(10);

  TGeoVolume *b24=geom->MakeSphere("b24",Iron,0,50,90,180,90,180);
  b24->SetLineColor(10);



  TGeoVolume *b25=geom->MakeTube("b25",Iron,51,54,150);
  b25->SetLineColor(17);
  TGeoVolume *b26=geom->MakeTube("b26",Iron,56,59,150);
  b26->SetLineColor(17);
  TGeoVolume *b27=geom->MakeTube("b27",Iron,61,64,150);
  b27->SetLineColor(17);
  TGeoVolume *b28=geom->MakeTube("b28",Iron,66,69,150);
  b28->SetLineColor(17);
  TGeoVolume *b29=geom->MakeTube("b29",Iron,71,74,150);
  b29->SetLineColor(17);

  TGeoVolume *b30=geom->MakeTube("b30",Iron,76,79,150);
  b30->SetLineColor(17);
  TGeoVolume *b31=geom->MakeTube("b31",Iron,81,84,150);
  b31->SetLineColor(17);
  TGeoVolume *b32=geom->MakeTube("b32",Iron,86,89,150);
  b32->SetLineColor(17);
  TGeoVolume *b33=geom->MakeTube("b33",Iron,91,94,150);
  b33->SetLineColor(17);
  TGeoVolume *b34=geom->MakeTube("b34",Iron,96,99,150);
  b34->SetLineColor(17);
  TGeoVolume *b35=geom->MakeTube("b35",Iron,101,104,150);
  b35->SetLineColor(17);
  TGeoVolume *b36=geom->MakeTube("b36",Iron,106,109,150);
  b36->SetLineColor(17);
  TGeoVolume *b37=geom->MakeTube("b37",Iron,111,114,150);
  b37->SetLineColor(17);
  TGeoVolume *b38=geom->MakeTube("b38",Iron,116,119,150);
  b38->SetLineColor(17);
  TGeoVolume *b39=geom->MakeTube("b39",Iron,121,124,150);
  b39->SetLineColor(17);
  TGeoVolume *b40=geom->MakeTube("b40",Iron,126,129,150);
  b40->SetLineColor(17);
  TGeoVolume *b41=geom->MakeTube("b41",Iron,131,134,150);
  b41->SetLineColor(17);
  TGeoVolume *b42=geom->MakeTube("b42",Iron,136,139,150);
  b42->SetLineColor(17);
  TGeoVolume *b43=geom->MakeTube("b43",Iron,141,144,150);
  b43->SetLineColor(17);
  TGeoVolume *b44=geom->MakeTube("b44",Iron,146,149,150);
  b44->SetLineColor(17);


  TGeoVolume *b45=geom->MakeTube("b45",Iron,0,25,150);
  b45->SetLineColor(10);

  TGeoVolume *b46=geom->MakeTube("b46",Iron,25,30,150);
  b46->SetLineColor(17);



  TGeoVolume *b47=geom->MakeBox("b47",Iron,140,194,504);
  b47->SetLineColor(32);


  TGeoVolume *b48=geom->MakeBox("b48",Iron,150,176,236);
  b48->SetLineColor(37);


  TGeoVolume *b49=geom->MakeBox("b49",Iron,150,2,236);
  b49->SetLineColor(20);
  top->AddNodeOverlap(b49,49,new TGeoTranslation(-2,179,-150));

  TGeoVolume *b50=geom->MakeBox("b50",Iron,150,2,236);
  b50->SetLineColor(20);
  top->AddNodeOverlap(b50,50,new TGeoTranslation(-2,-179,-150));

  TGeoVolume *b51=geom->MakeBox("b51",Iron,150,176,2);
  b51->SetLineColor(20);
  top->AddNodeOverlap(b51,51,new TGeoTranslation(-2,0,89));

  TGeoVolume *b52=geom->MakeBox("b52",Iron,150,176,2);
  b52->SetLineColor(20);
  top->AddNodeOverlap(b52,52,new TGeoTranslation(-2,0,-389));


  TGeoVolume *b53=geom->MakeBox("b53",Iron,150,200,90);
  b53->SetLineColor(10);
  top->AddNodeOverlap(b53,53,new TGeoTranslation(0,0,-510));





  TGeoVolume *b54=geom->MakeBox("b54",Iron,15,254,600);
  b54->SetLineColor(37);
  top->AddNodeOverlap(b54,54,new TGeoTranslation(25,0,0));

  TGeoVolume *b55=geom->MakeTubs("b55",Iron,0,54,15,270,360);
  b55->SetLineColor(37);
  top->AddNodeOverlap(b55,55,new TGeoCombiTrans(25,200,-600,new TGeoRotation("r1",90,90,0)));


  TGeoVolume *b56=geom->MakeTubs("b56",Iron,0,54,15,180,270);
  b56->SetLineColor(37);
  top->AddNodeOverlap(b56,56,new TGeoCombiTrans(25,-200,-600,new TGeoRotation("r1",90,90,0)));


  TGeoVolume *b57=geom->MakeTubs("b57",Iron,0,54,15,0,90);
  b57->SetLineColor(37);
  top->AddNodeOverlap(b57,57,new TGeoCombiTrans(25,200,600,new TGeoRotation("r1",90,90,0)));

  TGeoVolume *b58=geom->MakeTubs("b58",Iron,0,54,15,90,180);
  b58->SetLineColor(37);
  top->AddNodeOverlap(b58,58,new TGeoCombiTrans(25,-200,600,new TGeoRotation("r1",90,90,0)));

  //TGeoVolume *b59=geom->MakePgon("b59",Iron,100,100,100,100);
  //b59->SetLineColor(37);
  //top->AddNodeOverlap(b59,59,new TGeoCombiTrans(200,200,100,new TGeoRotation("r1",90,90,0)));



//IAudid



  TGeoVolume *b61=geom->MakeBox("b61",Iron,5,19,150);
  b61->SetLineColor(38);
  top->AddNodeOverlap(b61,61,new TGeoCombiTrans(-4,-87,-495,new TGeoRotation("r1",90,90,30)));

  TGeoVolume *b62=geom->MakeBox("b62",Iron,5,19,150);
  b62->SetLineColor(38);
  top->AddNodeOverlap(b62,62,new TGeoCombiTrans(-4,-65,-495,new TGeoRotation("r1",90,90,330)));
//u
  TGeoVolume *b63=geom->MakeBox("b63",Iron,5,15,150);
  b63->SetLineColor(38);
  top->AddNodeOverlap(b63,63,new TGeoCombiTrans(-4,-40,-497,new TGeoRotation("r1",90,90,0)));

  TGeoVolume *b64=geom->MakeBox("b64",Iron,5,15,150);
  b64->SetLineColor(38);
  top->AddNodeOverlap(b64,64,new TGeoCombiTrans(-4,-10,-497,new TGeoRotation("r1",90,90,0)));

  TGeoVolume *b65=geom->MakeTubs("b65",Iron,7,17,150,0,180);
  b65->SetLineColor(38);
  top->AddNodeOverlap(b65,65,new TGeoCombiTrans(-4,-25,-490,new TGeoRotation("r1",90,90,0)));


//D

  TGeoVolume *b66=geom->MakeBox("b66",Iron,5,19,150);
  b66->SetLineColor(38);
  top->AddNodeOverlap(b66,66,new TGeoCombiTrans(-4,10,-495,new TGeoRotation("r1",90,90,0)));


  TGeoVolume *b67=geom->MakeTubs("b67",Iron,10,20,150,230,480);
  b67->SetLineColor(38);
  top->AddNodeOverlap(b67,67,new TGeoCombiTrans(-4,23,-495,new TGeoRotation("r1",90,90,0)));

//I

  TGeoVolume *b68=geom->MakeBox("b68",Iron,5,20,150);
  b68->SetLineColor(38);
  top->AddNodeOverlap(b68,68,new TGeoCombiTrans(-4,53,-495,new TGeoRotation("r1",90,90,0)));

//O

  TGeoVolume *b69=geom->MakeTubs("b69",Iron,10,22,150,0,360);
  b69->SetLineColor(38);
  top->AddNodeOverlap(b69,69,new TGeoCombiTrans(-4,85,-495,new TGeoRotation("r1",90,90,0)));


// I
  TGeoVolume *b60=geom->MakeTube("b60",Iron,0,10,150);
  b60->SetLineColor(38);
  top->AddNodeOverlap(b60,60,new TGeoCombiTrans(-4,-120,-550,new TGeoRotation("r1",90,90,0)));


  TGeoVolume *b70=geom->MakeBox("b70",Iron,2,19,150);
  b70->SetLineColor(38);
  top->AddNodeOverlap(b70,70,new TGeoCombiTrans(-4,-114,-495,new TGeoRotation("r1",90,90,0)));

  TGeoVolume *b71=geom->MakeBox("b71",Iron,2,19,150);
  b71->SetLineColor(38);
  top->AddNodeOverlap(b71,71,new TGeoCombiTrans(-4,-126,-495,new TGeoRotation("r1",90,90,0)));


  TGeoVolume *b72=geom->MakeBox("b72",Iron,8,2,150);
  b72->SetLineColor(38);
  top->AddNodeOverlap(b72,72,new TGeoCombiTrans(-4,-120,-515,new TGeoRotation("r1",90,90,0)));


  TGeoVolume *b73=geom->MakeBox("b73",Iron,8,2,150);
  b73->SetLineColor(38);
  top->AddNodeOverlap(b73,73,new TGeoCombiTrans(-4,-120,-475,new TGeoRotation("r1",90,90,0)));


// button


  TGeoVolume *b74=geom->MakeBox("b74",Iron,35,250,70);
  b74->SetLineColor(38);
  top->AddNodeOverlap(b74,74,new TGeoCombiTrans(-25,10,-60,new TGeoRotation("r1",0,0,0)));

  TGeoVolume *b75=geom->MakeBox("b75",Iron,35,250,35);
  b75->SetLineColor(38);
  top->AddNodeOverlap(b75,75,new TGeoCombiTrans(-25,10,-175,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b76=geom->MakeBox("b76",Iron,35,250,35);
  b76->SetLineColor(38);
  top->AddNodeOverlap(b76,76,new TGeoCombiTrans(-25,10,55,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b77=geom->MakeTubs("b77",Iron,0,70,250,180,270);
  b77->SetLineColor(38);
  top->AddNodeOverlap(b77,77,new TGeoCombiTrans(10,10,-210,new TGeoRotation("r1",0,90,0)));


  TGeoVolume *b78=geom->MakeTubs("b78",Iron,0,70,250,90,180);
  b78->SetLineColor(38);
  top->AddNodeOverlap(b78,78,new TGeoCombiTrans(10,10,90,new TGeoRotation("r1",0,90,0)));



//Hold

  TGeoVolume *b79=geom->MakeBox("b79",Iron,40,250,150);
  b79->SetLineColor(10);
  top->AddNodeOverlap(b79,79,new TGeoCombiTrans(60,0,450,new TGeoRotation("r1",0,0,0)));

  TGeoVolume *b80=geom->MakeTubs("b80",Iron,50,100,250,180,270);
  b80->SetLineColor(10);
  top->AddNodeOverlap(b80,80,new TGeoCombiTrans(10,0,350,new TGeoRotation("r1",0,90,0)));


  TGeoVolume *b81=geom->MakeTubs("b81",Iron,50,100,250,90,180);
  b81->SetLineColor(10);
  top->AddNodeOverlap(b81,81,new TGeoCombiTrans(10,0,400,new TGeoRotation("r1",0,90,0)));


  TGeoVolume *b82=geom->MakeBox("b82",Iron,30,250,150);
  b82->SetLineColor(10);
  top->AddNodeOverlap(b82,82,new TGeoCombiTrans(-70,0,450,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b83=geom->MakeBox("b83",Iron,30,250,60);
  b83->SetLineColor(10);
  top->AddNodeOverlap(b83,83,new TGeoCombiTrans(-20,0,540,new TGeoRotation("r1",0,0,0)));




  TGeoVolume *b85=geom->MakeTubs("b85",Iron,0,40,240,180,270);
  b85->SetLineColor(38);
  top->AddNodeOverlap(b85,85,new TGeoCombiTrans(10,10,370,new TGeoRotation("r1",0,90,0)));




  TGeoVolume *b84=geom->MakeTubs("b84",Iron,0,40,240,90,180);
  b84->SetLineColor(38);
  top->AddNodeOverlap(b84,84,new TGeoCombiTrans(10,10,400,new TGeoRotation("r1",0,90,0)));


  TGeoVolume *b86=geom->MakeBox("b86",Iron,20,240,20);
  b86->SetLineColor(38);
  top->AddNodeOverlap(b86,86,new TGeoCombiTrans(-10,10,380,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b87=geom->MakeBox("b87",Iron,20,250,10);
  b87->SetLineColor(35);
  top->AddNodeOverlap(b87,87,new TGeoCombiTrans(-10,20,385,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b88=geom->MakeBox("b88",Iron,100,220,600);
  b88->SetLineColor(10);
  top->AddNodeOverlap(b88,88,new TGeoCombiTrans(0,-30,0,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b89=geom->MakeTube("b89",Iron,25,95,650);
  b89->SetLineColor(10);
  top->AddNodeOverlap(b89,89,new TGeoCombiTrans(0,-60,0,new TGeoRotation("r1",0,0,0)));

  TGeoVolume *b90=geom->MakeTube("b90",Iron,25,95,650);
  b90->SetLineColor(10);
  top->AddNodeOverlap(b90,90,new TGeoCombiTrans(0,60,0,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b91=geom->MakeBox("b91",Iron,40,200,650);
  b91->SetLineColor(10);
  top->AddNodeOverlap(b91,91,new TGeoCombiTrans(70,0,0,new TGeoRotation("r1",0,0,0)));

  TGeoVolume *b92=geom->MakeBox("b92",Iron,100,50,650);
  b92->SetLineColor(10);
  top->AddNodeOverlap(b92,92,new TGeoCombiTrans(0,150,0,new TGeoRotation("r1",0,0,0)));

  TGeoVolume *b93=geom->MakeBox("b93",Iron,100,50,650);
  b93->SetLineColor(10);
  top->AddNodeOverlap(b93,93,new TGeoCombiTrans(0,-150,0,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b94=geom->MakeBox("b94",Iron,40,200,650);
  b94->SetLineColor(10);
  top->AddNodeOverlap(b94,94,new TGeoCombiTrans(-70,0,0,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b95=geom->MakeTube("b95",Iron,25,35,650);
  b95->SetLineColor(1);
  top->AddNodeOverlap(b95,95,new TGeoCombiTrans(0,-60,-10,new TGeoRotation("r1",0,0,0)));

  TGeoVolume *b96=geom->MakeTube("b96",Iron,25,35,650);
  b96->SetLineColor(1);
  top->AddNodeOverlap(b96,96,new TGeoCombiTrans(0,60,-10,new TGeoRotation("r1",0,0,0)));
//usb

  TGeoVolume *b97=geom->MakeBox("b97",Iron,70,70,600);
  b97->SetLineColor(17);
  top->AddNodeOverlap(b97,97,new TGeoCombiTrans(0,0,57,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b98=geom->MakeTubs("b98",Iron,0,50,600,0,90);
  b98->SetLineColor(17);
  top->AddNodeOverlap(b98,98,new TGeoCombiTrans(20,60,57,new TGeoRotation("r1",0,0,0)));

  TGeoVolume *b99=geom->MakeTubs("b99",Iron,0,50,600,180,270);
  b99->SetLineColor(17);
  top->AddNodeOverlap(b99,99,new TGeoCombiTrans(-20,-60,57,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b100=geom->MakeTubs("b100",Iron,0,50,600,90,180);
  b100->SetLineColor(17);
  top->AddNodeOverlap(b100,100,new TGeoCombiTrans(-20,60,57,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b101=geom->MakeTubs("b101",Iron,0,50,600,270,360);
  b101->SetLineColor(17);
  top->AddNodeOverlap(b101,101,new TGeoCombiTrans(20,-60,57,new TGeoRotation("r1",0,0,0)));

  TGeoVolume *b102=geom->MakeBox("b102",Iron,20,110,600);
  b102->SetLineColor(17);
  top->AddNodeOverlap(b102,102,new TGeoCombiTrans(0,0,57,new TGeoRotation("r1",0,0,0)));


  TGeoVolume *b103=geom->MakeBox("b103",Iron,15,200,600);
  b103->SetLineColor(37);
  top->AddNodeOverlap(b103,103,new TGeoCombiTrans(25,0,57,new TGeoRotation("r1",0,0,0)));
//AddNode
  top->AddNodeOverlap(b1,1,new TGeoTranslation(0,0,0));
  top->AddNodeOverlap(b2,2,new TGeoCombiTrans(100,0,600,new TGeoRotation("r1",0,90,0)));
  top->AddNodeOverlap(b3,3,new TGeoCombiTrans(-100,0,600,new TGeoRotation("r1",0,90,0)));
  top->AddNodeOverlap(b4,4,new TGeoCombiTrans(-100,0,-600,new TGeoRotation("r1",0,90,0)));
  top->AddNodeOverlap(b5,5,new TGeoCombiTrans(100,0,-600,new TGeoRotation("r1",0,90,0)));
  top->AddNodeOverlap(b6,6,new TGeoCombiTrans(100,200,0,new TGeoRotation("r1",0,0,0)));
  top->AddNodeOverlap(b7,7,new TGeoCombiTrans(-100,200,0,new TGeoRotation("r1",0,0,0)));
  top->AddNodeOverlap(b8,8,new TGeoCombiTrans(-100,-200,0,new TGeoRotation("r1",0,0,0)));
  top->AddNodeOverlap(b9,9,new TGeoCombiTrans(100,-200,0,new TGeoRotation("r1",0,0,0)));

  top->AddNodeOverlap(b10,10,new TGeoCombiTrans(0,200,600,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b11,11,new TGeoCombiTrans(0,-200,600,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b12,12,new TGeoCombiTrans(0,-200,-600, new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b13,13,new TGeoCombiTrans(0,200,-600,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b14,14,new TGeoTranslation(0,200,-150));
  top->AddNodeOverlap(b15,15,new TGeoTranslation(100,0,0));

  top->AddNodeOverlap(b16,16,new TGeoCombiTrans(100,200,600,new TGeoRotation("r2",0,0,0)));
  top->AddNodeOverlap(b17,17,new TGeoCombiTrans(100,-200,600,new TGeoRotation("r2",0,0,0)));
  top->AddNodeOverlap(b18,18,new TGeoCombiTrans(-100,-200,600,new TGeoRotation("r2",0,0,0)));
  top->AddNodeOverlap(b19,19,new TGeoCombiTrans(-100,200,600,new TGeoRotation("r2",0,0,0)));
  top->AddNodeOverlap(b20,20,new TGeoCombiTrans(-3,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b21,21,new TGeoCombiTrans(100,200,-600,new TGeoRotation("r2",0,0,0)));
  top->AddNodeOverlap(b22,22,new TGeoCombiTrans(100,-200,-600,new TGeoRotation("r2",0,0,0)));
  top->AddNodeOverlap(b23,23,new TGeoCombiTrans(-100,-200,-600,new TGeoRotation("r2",0,0,0)));
  top->AddNodeOverlap(b24,24,new TGeoCombiTrans(-100,200,-600,new TGeoRotation("r2",0,0,0)));



  top->AddNodeOverlap(b25,25,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b26,26,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b27,27,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b28,28,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b29,29,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b30,30,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b31,31,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b32,32,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b33,33,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b34,34,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b35,35,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b36,36,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b37,37,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b38,38,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b39,39,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b40,40,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b41,41,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b42,42,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b43,43,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b44,44,new TGeoCombiTrans(-9,0,350,new TGeoRotation("r2",90,90,0)));


  top->AddNodeOverlap(b45,45,new TGeoCombiTrans(-20,0,350,new TGeoRotation("r2",90,90,0)));
  top->AddNodeOverlap(b46,46,new TGeoCombiTrans(-25,0,350,new TGeoRotation("r2",90,90,0)));

  top->AddNodeOverlap(b47,47,new TGeoTranslation(5,0,85));
  top->AddNodeOverlap(b48,48,new TGeoTranslation(-2,0,-150));
  geom->CloseGeometry();



  TCanvas *can=new TCanvas("can","My virtual laboratory",800,800);


//Mp3
   TPad *pad=new TPad("pad","Pad",0,0.5,0.5,1);
   pad->SetFillColor(1);
   pad->Draw();
   pad->cd();
  top->Draw();
//Sound
   can->cd();
   TPad *pad2=new TPad("pad2","Pad2",0.5,0.5,1,1);
   pad2->SetFillColor(10);
   pad2->Draw();
   pad2->cd();


   TPaveText *pt = new TPaveText(0.4,0.90,0.6,0.95,"br");
   pt->SetFillColor(30);
   pt->AddText(0.5,0.5,"Musics");
   pt->Draw();

   TLatex Tex;

   Tex.SetTextSize(0.04);
   Tex.SetTextColor(31);
   Tex.DrawLatex(0.3,0.81,"Mariah Carey - Shake it off");

   Tex.SetTextSize(0.04);
   Tex.SetTextColor(31);
   Tex.DrawLatex(0.3,0.71,"Alicia keys - If I ain't got you");

   Tex.SetTextSize(0.04);
   Tex.SetTextColor(31);
   Tex.DrawLatex(0.3,0.61,"Michael Jackson - Billie Jean");

   Tex.SetTextSize(0.04);
   Tex.SetTextColor(31);
   Tex.DrawLatex(0.3,0.51,"Christina Milian - Am to Pm");

   Tex.SetTextSize(0.04);
   Tex.SetTextColor(31);
   Tex.DrawLatex(0.3,0.41,"Zapp&Roger - Slow and Easy");

   Tex.SetTextSize(0.04);
   Tex.SetTextColor(31);
   Tex.DrawLatex(0.3,0.31,"Black Eyes Peas - Let's get retarded");

   Tex.SetTextSize(0.04);
   Tex.SetTextColor(31);
   Tex.DrawLatex(0.3,0.21,"Bosson - One in a Millin");

   Tex.SetTextSize(0.04);
   Tex.SetTextColor(15);
   Tex.DrawLatex(0.2,0.11,"Click Button!! You Can Listen to Musics");
   TButton *but1=new TButton("","Sound(1)",0.2,0.8,0.25,0.85);
   but1->Draw();
   but1->SetFillColor(29);
   TButton *but2=new TButton("","Sound(2)",0.2,0.7,0.25,.75);
   but2->Draw();
   but2->SetFillColor(29);
   TButton *but3=new TButton("","Sound(3)",0.2,0.6,0.25,0.65);
   but3->Draw();
   but3->SetFillColor(29);
   TButton *but4=new TButton("","Sound(4)",0.2,0.5,0.25,0.55);
   but4->Draw();
   but4->SetFillColor(29);
   TButton *but5=new TButton("","Sound(5)",0.2,0.4,0.25,0.45);
   but5->Draw();
   but5->SetFillColor(29);
   TButton *but6=new TButton("","Sound(6)",0.2,0.3,0.25,0.35);
   but6->Draw();
   but6->SetFillColor(29);
   TButton *but7=new TButton("","Sound(7)",0.2,0.2,0.25,0.25);
   but7->Draw();
   but7->SetFillColor(29);

   pad->cd();

//introduction
   can->cd();

   TPad *pad3=new TPad("pad3","Pad3",0,0,1,0.5);
   pad3->SetFillColor(10);
   pad3->Draw();
   pad3->cd();

   TImage *image=TImage::Open("mp3.jpg");
   image->Draw();



   TPad *pad4=new TPad("pad4","Pad4",0.6,0.1,0.9,0.9);
   pad4->SetFillColor(1);
   pad4->Draw();
   pad4->cd();


   TLine L;

   Tex.SetTextSize(0.08);
   Tex.SetTextColor(10);
   Tex.DrawLatex(0.06,0.85,"IAudio U3 Mp3 Player");


   L.SetLineColor(10);
   L.SetLineWidth(3);
   L.DrawLine(0.05, 0.83,0.90, 0.83);

   Tex.SetTextSize(0.06);
   Tex.SetTextColor(10);
   Tex.DrawLatex(0.06,0.75,"+ Color LCD");

   Tex.SetTextSize(0.06);
   Tex.SetTextColor(10);
   Tex.DrawLatex(0.06,0.65,"+ 60mW High Generating Power");

   Tex.SetTextSize(0.06);
   Tex.SetTextColor(10);
   Tex.DrawLatex(0.06,0.55,"+ GUI Theme Skin");

   Tex.SetTextSize(0.06);
   Tex.SetTextColor(10);
   Tex.DrawLatex(0.06,0.45,"+ Noble White&Black");

   Tex.SetTextSize(0.06);
   Tex.SetTextColor(10);
   Tex.DrawLatex(0.06,0.35,"+ Text Viewer+Image Viewer");

   Tex.SetTextSize(0.06);
   Tex.SetTextColor(10);
   Tex.DrawLatex(0.06,0.25,"+ 20 Hours Playing");

   Tex.SetTextSize(0.06);
   Tex.SetTextColor(10);
   Tex.DrawLatex(0.06,0.15,"+ The Best Quality of Sound");


   pad->cd();



}

void Sound(int i)
{
   char sound[128];
   sprintf(sound,"cat sound%d.wav > /dev/audio",i);
   gSystem->Exec(sound);
}
