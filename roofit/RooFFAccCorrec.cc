// Current calib: 4ac   4.16.2002
/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooFFAccCorrec.cc,v 1.1 2002/05/07 18:47:13 msgill Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   MSG, Mandeep Gill, using RooGaussian framework,  extra Form Factor additions
 *
 * History:
 *   05-Jan-2000 DK Created initial version from RooGaussianProb
 *   02-May-2001 WV Port to RooFitModels/RooFitCore
 *   20-Aug-2001 MSG Begin adding FF stuff
 *   07-May-2002 MSG Version with acc correction in the actual pdf
 *               definition vs. in the normzn
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --

#include "BaBar/BaBar.hh"
#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooFFAccCorrec.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"

// Need these next includes if i want to do any couts inside this code
// (but not for standalone macros in Root, because Root autoloads
// these things)

#include <stdio.h>
#include <iostream.h>

ClassImp(RooFFAccCorrec)
  ;

static const char rcsid[] =
"$Id: RooFFAccCorrec.cc,v 1.1 2002/05/07 18:47:13 msgill Exp $";

RooFFAccCorrec::RooFFAccCorrec(const char *name, const char *title,
			     RooAbsReal& _w, RooAbsReal& _ctl, RooAbsReal& _ctv, 
			     RooAbsReal& _chi,
			     RooAbsReal& _R1, RooAbsReal& _R2, RooAbsReal& _rho2) :

  // The two addresses refer to our first dependent variable and
  // parameter, respectively, as declared in the rdl file
  RooAbsPdf(name, title),

  // Declare our dependent variable(s) in the order they are listed
  // in the rdl file
  w("w"," w",this,_w),
  ctl("ctl"," ctl",this,_ctl),
  ctv("ctv"," ctv",this,_ctv),
  chi("chi"," chi",this,_chi),
  // Declare our parameter(s) in the order they are listed in the rdl file
  R1("R1"," R1",this,_R1),
  R2("R2"," R2",this,_R2),
  rho2("rho2"," rho2",this,_rho2)
  

{
}

// Copy ctor
RooFFAccCorrec::RooFFAccCorrec(const RooFFAccCorrec& other,const  char *name):
			   RooAbsPdf(other,name)  ,
			   w("w",this,other.w),
			   ctl("ctl",this,other.ctl),
			   ctv("ctv",this,other.ctv),
			   chi("chi",this,other.chi),
			   R1("R1",this,other.R1),
			   R2("R2",this,other.R2),
			   rho2("rho2",this,other.rho2)
{
}


Int_t RooFFAccCorrec::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,w,ctl,ctv,chi)) return 1 ;
  return 0 ;
}


Double_t RooFFAccCorrec::analyticalIntegral(Int_t code) const 
{
  switch(code) {
  case 1: 
    {
      return 1.0 ; 
    }
  }
}


Double_t RooFFAccCorrec::evaluate() const {
  
  // This is the 4-dim PDF depending on the wlvc vars, and with
  // R1,R2,rho2 as the independent params we'll want to fit for
  // (Neubert HQET FFAccCorrec form)

  Double_t Pi=4*atan(1);
  
  Double_t mb=5.28;
  Double_t mdstr=2.01   ;
  
  Double_t mb2 = mb*mb;
  Double_t mdstr2 = mdstr*mdstr;

  Double_t stl = sqrt(1-ctl*ctl);
  Double_t stv = sqrt(1-ctv*ctv);
  Double_t cchi = cos(chi);
  Double_t c2chi = cos(2*chi);
  
  Double_t ctl2 = ctl*ctl;
  Double_t ctv2 = ctv*ctv;
  Double_t stl2 = stl*stl;
  Double_t stv2 = stv*stv;
 
  Double_t omctl2 = (1-ctl)*(1-ctl);
  Double_t opctl2 = (1+ctl)*(1+ctl);
    
  Double_t pdstr = mdstr*sqrt(w*w-1);
  
  Double_t opw2=(w+1)*(w+1);
  
  Double_t r=mdstr/mb ;
  Double_t rsq=r*r;
  Double_t omr2 = (1-r)*(1-r);
  
  
  Double_t ha1=(1-rho2*(w-1));
  
  Double_t hpfac = (1-sqrt( (w-1)/(w+1) ) * R1 );
  Double_t hmfac = (1+sqrt( (w-1)/(w+1) ) * R1 );
  Double_t hzfac = (1+((w-1)/(1-r))*(1-R2));
  
  Double_t hp= sqrt( (1-2*w*r+rsq)/omr2 )* hpfac;
  Double_t hm= sqrt( (1-2*w*r+rsq)/omr2 )* hmfac;
  Double_t hz= hzfac;  
	
  Double_t hphmterm = -2*hp*hm* stl*stl* stv*stv* c2chi;
  Double_t hphzterm = -4*hp*hz* stl*(1-ctl)*stv*ctv*cchi;
  Double_t hmhzterm = 4*hm*hz*stl*(1+ctl)*stv*ctv*cchi;
    
  Double_t hp2term = hp*hp* stv*stv * omctl2 ;
  Double_t hm2term = hm*hm* stv*stv* opctl2;
  Double_t hz2term = hz*hz* 4* stl2* ctv2;
    
  Double_t dgd4numr= (ha1*ha1)* opw2 * pdstr* 
    (hp2term + hm2term + hz2term + hphmterm + hphzterm + hmhzterm);


  // ** All calib files --
  // filt.sig
  // filt.reco
  // fgr.sig
  // fgr.reco

  /// *********************** Evtgen calibs ***************************
  // reff Evtgen/kv.allcuts.Rset0.5.dat 1087 5   -- 5.13.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(4.771413  )+
//  R1*R1*rho2*(-27.832760       )+
//  R1*R1*( 45.245800          )+
// R1*rho2*rho2*( 15.372794     )+
//  R1*rho2*(-85.164833           )+
//  R1*(125.301178                )+
// R2*R2*rho2*rho2*(37.513245    )+
//  R2*R2*rho2*(-178.635391        )+
//  R2*R2*(221.282028            )+
// R2*rho2*rho2*(-186.357529      )+
//  R2*rho2*(909.122192           )+
//  R2*( -1168.731689              )+
// R1*R2*rho2*rho2*(-0.536427   )+
//  R1*R2*rho2*(2.507541        )+
//  R1*R2*(  -3.130799          )+
// rho2*rho2*(  267.852356      )+
// rho2*(-1395.131958               )+
// (       2011.738647            ); 

 //  reff Evtgen/kv.allcuts.Rset0.75.dat 1106 5  --- 5.12.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(5.730415  )+
//  R1*R1*rho2*(-34.621120       )+
//  R1*R1*( 57.626850          )+
// R1*rho2*rho2*( 19.253941     )+
//  R1*rho2*(-108.375267           )+
//  R1*(162.499069                )+
// R2*R2*rho2*rho2*(50.025120    )+
//  R2*R2*rho2*(-238.587555        )+
//  R2*R2*(295.767395            )+
// R2*rho2*rho2*(-250.112518      )+
//  R2*rho2*(1221.135498           )+
//  R2*( -1571.475830              )+
// R1*R2*rho2*rho2*(-1.306156   )+
//  R1*R2*rho2*(5.460313        )+
//  R1*R2*(  -5.241727          )+
// rho2*rho2*(  359.986816      )+
// rho2*(-1885.439575               )+
// (       2745.675781            ); 

  // reff Evtgen/kv.allcuts.Rset1.0.dat 1054 5    5.12.2002   [ remember to adjust the R params inside effcalc also ]
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(6.584343  )+
//  R1*R1*rho2*(-40.659534       )+
//  R1*R1*( 70.083641          )+
// R1*rho2*rho2*( 23.395493     )+
//  R1*rho2*(-131.555649           )+
//  R1*(197.747543                )+
// R2*R2*rho2*rho2*(62.840446    )+
//  R2*R2*rho2*(-300.672638        )+
//  R2*R2*(374.839508            )+
// R2*rho2*rho2*(-313.046295      )+
//  R2*rho2*(1535.712769           )+
//  R2*( -1989.590820              )+
// R1*R2*rho2*rho2*(-1.548047   )+
//  R1*R2*rho2*(6.408904        )+
//  R1*R2*(  -5.475188          )+
// rho2*rho2*(  451.114380      )+
// rho2*(-2379.422852               )+
// (       3487.246094            ); 


//  reff Evtgen/kv.allcuts.Rset1.25.dat 1070 5 -- 5.12.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(9.460588  )+
//  R1*R1*rho2*(-57.844025       )+
//  R1*R1*( 98.031357          )+
// R1*rho2*rho2*( 29.717583     )+
//  R1*rho2*(-167.035538           )+
//  R1*(255.534698                )+
// R2*R2*rho2*rho2*(91.265739    )+
//  R2*R2*rho2*(-431.537170        )+
//  R2*R2*(531.876221            )+
// R2*rho2*rho2*(-450.277588      )+
//  R2*rho2*(2180.864014           )+
//  R2*( -2789.162109              )+
// R1*R2*rho2*rho2*(-0.736193   )+
//  R1*R2*rho2*(1.023754        )+
//  R1*R2*(  2.321096          )+
// rho2*rho2*(  636.483459      )+
// rho2*(-3307.562500               )+
// (       4773.637695            ); 

 // reff Evtgen/kv.allcuts.Rset1.50.dat 1017 5   --- 5.12.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(13.632824  )+
//  R1*R1*rho2*(-82.201599       )+
//  R1*R1*( 136.019211          )+
// R1*rho2*rho2*( 28.561733     )+
//  R1*rho2*(-160.001724           )+
//  R1*(245.930084                )+
// R2*R2*rho2*rho2*(151.553894    )+
//  R2*R2*rho2*(-689.784607        )+
//  R2*R2*(814.887756            )+
// R2*rho2*rho2*(-726.020630      )+
//  R2*rho2*(3369.775635           )+
//  R2*( -4111.264160              )+
// R1*R2*rho2*rho2*(1.953723   )+
//  R1*R2*rho2*(-15.392529        )+
//  R1*R2*(  27.403812          )+
// rho2*rho2*(  969.508728      )+
// rho2*(-4802.021973               )+
// (       6578.625977            ); 


 // reff Evtgen/kv.allcuts.Rset2.0.dat 948 5   --  5.12.2002
//   Double_t ffnorm =  
//  R1*R1*rho2*rho2*(18.897270  )+
//  R1*R1*rho2*(-113.223175       )+
//  R1*R1*( 187.627686          )+
// R1*rho2*rho2*( 27.294411     )+
//  R1*rho2*(-170.379425           )+
//  R1*(284.988922                )+
// R2*R2*rho2*rho2*(64.062675    )+
//  R2*R2*rho2*(-353.861328        )+
//  R2*R2*(508.689087            )+
// R2*rho2*rho2*(-355.619476      )+
//  R2*rho2*(2003.252930           )+
//  R2*( -2978.923340              )+
// R1*R2*rho2*rho2*(4.247729   )+
//  R1*R2*rho2*(-23.138624        )+
//  R1*R2*(  31.843088          )+
// rho2*rho2*(  611.176208      )+
// rho2*(-3679.436279               )+
// (       6158.966309            ); 


// ****************************** SP4 calibs *******************************

  //sp4:  reff DatFiles/totnew.dat 75953 8   bg0: about 49K
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(588.503967  )+
//  R1*R1*rho2*(-3312.455322       )+
//  R1*R1*( 5170.446289          )+
// R1*rho2*rho2*( -299.437592     )+
//  R1*rho2*(519.470276           )+
//  R1*(1268.974854                )+
// R2*R2*rho2*rho2*(2613.126465    )+
//  R2*R2*rho2*(-12722.883789        )+
//  R2*R2*(16076.250977            )+
// R2*rho2*rho2*(-11424.409180      )+
//  R2*rho2*(57090.109375           )+
//  R2*( -75288.093750              )+
// R1*R2*rho2*rho2*(862.752441   )+
//  R1*R2*rho2*(-4259.356445        )+
//  R1*R2*(  5443.169922          )+
// rho2*rho2*(  15540.171875      )+
// rho2*(-83726.945312               )+
// (       126376.500000            ); 


  // reff DatFiles/new.neutsp4.bg0nocuts 59155 8  -- 5.2.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(678.933289  )+
//  R1*R1*rho2*(-3868.915039       )+
//  R1*R1*( 6127.645020          )+
// R1*rho2*rho2*( -265.929321     )+
//  R1*rho2*(207.643372           )+
//  R1*(2019.915527                )+
// R2*R2*rho2*rho2*(2899.513672    )+
//  R2*R2*rho2*(-14175.350586        )+
//  R2*R2*(18016.789062            )+
// R2*rho2*rho2*(-12782.414062      )+
//  R2*rho2*(64174.703125           )+
//  R2*( -85217.507812              )+
// R1*R2*rho2*rho2*(917.272034   )+
//  R1*R2*rho2*(-4570.840332        )+
//  R1*R2*(  5912.227539          )+
// rho2*rho2*(  17648.478516      )+
// rho2*(-95809.335938               )+
// (       146317.984375            ); 



  // 4ac -- 
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(36.916618  )+
//  R1*R1*rho2*(-248.872559       )+
//  R1*R1*( 499.578308          )+
// R1*rho2*rho2*( 29.469492     )+
//  R1*rho2*(-46.457188           )+
//  R1*(-284.321869                )+
// R2*R2*rho2*rho2*(185.472916    )+
//  R2*R2*rho2*(-887.902344        )+
//  R2*R2*(1116.221436            )+
// R2*rho2*rho2*(-880.320435      )+
//  R2*rho2*(4320.746582           )+
//  R2*( -5647.561035              )+
// R1*R2*rho2*rho2*(23.164299   )+
//  R1*R2*rho2*(-158.652008        )+
//  R1*R2*(  332.346649          )+
// rho2*rho2*(  1240.862183      )+
// rho2*(-6410.861816               )+
// (       9057.004883            ); 


// 4a -- reff DatFiles/sp4.a.dat 13507 8  -- 2083 bg0.  -- 4.18.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(28.753843  )+
//  R1*R1*rho2*(-204.192230       )+
//  R1*R1*( 433.964386          )+
// R1*rho2*rho2*( 15.390920     )+
//  R1*rho2*(35.278915           )+
//  R1*(-412.002563                )+
// R2*R2*rho2*rho2*(133.140594    )+
//  R2*R2*rho2*(-637.333862        )+
//  R2*R2*(805.276001            )+
// R2*rho2*rho2*(-629.475647      )+
//  R2*rho2*(3095.051270           )+
//  R2*( -4079.416748              )+
// R1*R2*rho2*rho2*(18.936563   )+
//  R1*R2*rho2*(-136.367188        )+
//  R1*R2*(  301.937805          )+
// rho2*rho2*(  884.327271      )+
// rho2*(-4584.147461               )+
// (       6543.009766            ); 

 //  4b  -- reff DatFiles/sp4.b.dat 10839 8   -- 1660 bg0  -- 4.24.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(20.620123  )+
//  R1*R1*rho2*(-112.327332       )+
//  R1*R1*( 164.271957          )+
// R1*rho2*rho2*( 16.468382     )+
//  R1*rho2*(-102.978157           )+
//  R1*(172.472382                )+
// R2*R2*rho2*rho2*(110.447403    )+
//  R2*R2*rho2*(-526.095825        )+
//  R2*R2*(646.572021            )+
// R2*rho2*rho2*(-516.034607      )+
//  R2*rho2*(2504.132324           )+
//  R2*( -3167.174805              )+
// R1*R2*rho2*rho2*(16.629858   )+
//  R1*R2*rho2*(-85.451385        )+
//  R1*R2*(  113.372658          )+
// rho2*rho2*(  715.611694      )+
// rho2*(-3636.820801               )+
// (       4944.748535            ); 

 //  4c  -- reff DatFiles/sp4.c.dat 11879  8   -- 1845 bg0  -- 4.24.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(23.509275  )+
//  R1*R1*rho2*(-125.528114       )+
//  R1*R1*( 179.260010          )+
// R1*rho2*rho2*( 0.801834     )+
//  R1*rho2*(-38.699661           )+
//  R1*(110.470879                )+
// R2*R2*rho2*rho2*(130.724930    )+
//  R2*R2*rho2*(-621.026367        )+
//  R2*R2*(761.621399            )+
// R2*rho2*rho2*(-599.030029      )+
//  R2*rho2*(2900.348877           )+
//  R2*( -3661.360352              )+
// R1*R2*rho2*rho2*(28.100069   )+
//  R1*R2*rho2*(-137.304962        )+
//  R1*R2*(  173.126755          )+
// rho2*rho2*(  815.118530      )+
// rho2*(-4136.221191               )+
// (       5618.828613            ); 



  // 3c  -- reff DatFiles/881b.2cut.dslbg.dat 10986 8  -- 4.15.2002
 Double_t ffnorm =  
 R1*R1*rho2*rho2*(78.021629  )+
 R1*R1*rho2*(-446.695282       )+
 R1*R1*( 694.050598          )+
R1*rho2*rho2*( -29.294533     )+
 R1*rho2*(71.266090           )+
 R1*(111.152092                )+
R2*R2*rho2*rho2*(293.745026    )+
 R2*R2*rho2*(-1446.829834        )+
 R2*R2*(1853.594360            )+
R2*rho2*rho2*(-1377.838135      )+
 R2*rho2*(6945.558105           )+
 R2*( -9247.802734              )+
R1*R2*rho2*rho2*(86.380432   )+
 R1*R2*rho2*(-454.513336        )+
 R1*R2*(  609.810181          )+
rho2*rho2*(  1951.352417      )+
rho2*(-10581.006836               )+
(       16071.130859            ); 


  // reff DatFiles/fgr.2cut.withcosdsl.dat 20594 7 0.2  (831 fail) 4.4.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(341.698456  )+
//  R1*R1*rho2*(-2082.634033       )+
//  R1*R1*( 3773.983643          )+
// R1*rho2*rho2*( -574.944336     )+
//  R1*rho2*(3181.444580           )+
//  R1*(-5124.202637                )+
// R2*R2*rho2*rho2*(860.191528    )+
//  R2*R2*rho2*(-4302.309082        )+
//  R2*R2*(5673.384277            )+
// R2*rho2*rho2*(-3845.044922      )+
//  R2*rho2*(19734.664062           )+
//  R2*( -27231.425781              )+
// R1*R2*rho2*rho2*(447.048859   )+
//  R1*R2*rho2*(-2418.315430        )+
//  R1*R2*(  3574.058838          )+
// rho2*rho2*(  5418.664551      )+
// rho2*(-30354.626953               )+
// (       49423.507812            ); 


  // reff DatFiles/filt.sig 14357 [2.23.2002]
//   Double_t ffnorm =  
//     R1*R1*rho2*rho2*(168.320099  )+
//     R1*R1*rho2*(-959.445374       )+
//     R1*R1*( 1529.763428          )+
//     R1*rho2*rho2*( -113.655762     )+
//     R1*rho2*(304.976501           )+
//     R1*(158.023453                )+
//     R2*R2*rho2*rho2*(719.973511    )+
//     R2*R2*rho2*(-3504.680176        )+
//     R2*R2*(4445.942383            )+
//     R2*rho2*rho2*(-3156.972168      )+
//     R2*rho2*(15809.172852           )+
//     R2*( -20955.710938              )+
//     R1*R2*rho2*rho2*(231.866043   )+
//     R1*R2*rho2*(-1160.084351        )+
//     R1*R2*(  1521.978149          )+
//     rho2*rho2*(  4352.562500      )+
//     rho2*(-23566.873047               )+
//     (       35889.062500            ); 
  

 // reff DatFiles/filt.ctail.sigreco.dat 15094 [ filt.reco ] [ 2.21.2002 ]
 /*   Double_t _normNeu =  
 R1*R1*rho2*rho2*(202.471710  )+
 R1*R1*rho2*(-1188.211304       )+
 R1*R1*( 1962.311890          )+
R1*rho2*rho2*( -210.493195     )+
 R1*rho2*(946.162048           )+
 R1*(-940.249023                )+
R2*R2*rho2*rho2*(778.660583    )+
 R2*R2*rho2*(-3833.177490        )+
 R2*R2*(4917.463867            )+
R2*rho2*rho2*(-3424.062256      )+
 R2*rho2*(17280.679688           )+
 R2*( -23084.304688              )+
R1*R2*rho2*rho2*(292.067291   )+
 R1*R2*rho2*(-1539.189331        )+
 R1*R2*(  2149.176025          )+
rho2*rho2*(  4693.681641      )+
rho2*(-25484.900391               )+
(       38776.246094            ); 
  */

// reff DatFiles/fgr.bg0.cosbycut.truelep.dat 9912 1 -- 3.30.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(81.885185  )+
//  R1*R1*rho2*(-483.701355       )+
//  R1*R1*( 812.964355          )+
// R1*rho2*rho2*( 110.773941     )+
//  R1*rho2*(-722.752319           )+
//  R1*(1272.864136                )+
// R2*R2*rho2*rho2*(408.000946    )+
//  R2*R2*rho2*(-1999.689087        )+
//  R2*R2*(2559.445801            )+
// R2*rho2*rho2*(-1949.667847      )+
//  R2*rho2*(9818.930664           )+
//  R2*( -13107.150391              )+
// R1*R2*rho2*rho2*(44.691734   )+
//  R1*R2*rho2*(-236.843079        )+
//  R1*R2*(  334.708038          )+
// rho2*rho2*(  2857.416992      )+
// rho2*(-15538.020508               )+
// (       23789.275391            ); 


  // reff DatFiles/filt.bg0.cosbycut.truelep.dat 11529 1 -- 3.30.2002
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(127.395905  )+
//  R1*R1*rho2*(-718.254395       )+
//  R1*R1*( 1129.789673          )+
// R1*rho2*rho2*( -57.303513     )+
//  R1*rho2*(53.286354           )+
//  R1*(413.902618                )+
// R2*R2*rho2*rho2*(605.669312    )+
//  R2*R2*rho2*(-2927.456055        )+
//  R2*R2*(3671.454590            )+
// R2*rho2*rho2*(-2659.999268      )+
//  R2*rho2*(13227.144531           )+
//  R2*( -17349.404297              )+
// R1*R2*rho2*rho2*(183.312653   )+
//  R1*R2*rho2*(-893.318298        )+
//  R1*R2*(  1125.560181          )+
// rho2*rho2*(  3630.322998      )+
// rho2*(-19479.710938               )+
// (       29273.728516            ); 

//  reff DatFiles/fgr.bg0.reco.dat 9931 1
//  Double_t ffnorm =  
//  R1*R1*rho2*rho2*(84.482582  )+
//  R1*R1*rho2*(-495.114197       )+
//  R1*R1*( 814.912109          )+
// R1*rho2*rho2*( 117.622459     )+
//  R1*rho2*(-785.810059           )+
//  R1*(1476.199097                )+
// R2*R2*rho2*rho2*(420.097443    )+
//  R2*R2*rho2*(-2059.502686        )+
//  R2*R2*(2634.571533            )+
// R2*rho2*rho2*(-2008.430420      )+
//  R2*rho2*(10113.311523           )+
//  R2*( -13473.730469              )+
// R1*R2*rho2*rho2*(45.445953   )+
//  R1*R2*rho2*(-241.394379        )+
//  R1*R2*(  343.233795          )+
// rho2*rho2*(  2941.290771      )+
// rho2*(-15927.472656               )+
// (       24010.902344            ); 



  // reff DatFiles/fgr.sig 11770 [12.30.2001, 2.22.2002]
//  Double_t ffnorm =  
//    R1*R1*rho2*rho2*(102.050156  )+
//    R1*R1*rho2*(-617.278992       )+
//    R1*R1*( 1059.125854          )+
//    R1*rho2*rho2*( 114.101089     )+
//    R1*rho2*(-731.938660           )+
//    R1*(1288.837036                )+
//    R2*R2*rho2*rho2*(465.127808    )+
//    R2*R2*rho2*(-2306.364746        )+
//    R2*R2*(3003.485596            )+
//    R2*rho2*rho2*(-2212.800293      )+
//    R2*rho2*(11253.737305           )+
//    R2*( -15231.733398              )+
//    R1*R2*rho2*rho2*(62.042931   )+
//    R1*R2*rho2*(-362.582458        )+
//    R1*R2*(  574.839111          )+
//    rho2*rho2*(  3260.495850      )+
//    rho2*(-17912.267578               )+
//    (       27795.294922            ); 
 


  // reff DatFiles/fgr.ctail.sigreco.dat 12682 [ fgr.reco ] [ 2.21.2002 ]
/*   Double_t ffnorm =  
   R1*R1*rho2*rho2*(164.836426  )+
   R1*R1*rho2*(-1026.806885       )+
   R1*R1*( 1804.127441          )+
   R1*rho2*rho2*( -73.566635     )+
   R1*rho2*(450.141327           )+
   R1*(-711.780701                )+
   R2*R2*rho2*rho2*(580.725769    )+
   R2*R2*rho2*(-2888.881104        )+
   R2*R2*(3773.854980            )+
   R2*rho2*rho2*(-2672.507812      )+
   R2*rho2*(13613.079102           )+
   R2*( -18460.658203              )+
   R1*R2*rho2*rho2*(160.714005   )+
   R1*R2*rho2*(-921.567810        )+
   R1*R2*(  1440.283813          )+
   rho2*rho2*(  3826.777832      )+
   rho2*(-20960.349609               )+
   (       32211.052734            ); 
*/

 
  Double_t dgd4= dgd4numr / ffnorm;

  //assert(0) ; // ?
  return dgd4;
}
