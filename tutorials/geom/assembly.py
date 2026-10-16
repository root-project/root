## \file
## \ingroup tutorial_geom
## Geometry detector assembly example
##
## \macro_image
## \macro_code
##
## \author Andrei Gheata
## \translator P. P.


import ROOT

TMath = ROOT.TMath

TGeoManager = ROOT.TGeoManager
TGeoMaterial = ROOT.TGeoMaterial
TGeoMedium = ROOT.TGeoMedium
TGeoVolumeAssembly = ROOT.TGeoVolumeAssembly
TGeoTranslation = ROOT.TGeoTranslation
TGeoRotation = ROOT.TGeoRotation
TGeoCombiTrans = ROOT.TGeoCombiTrans

kBlue = ROOT.kBlue

Int_t = ROOT.Int_t
Double_t = ROOT.Double_t



class assembly :
   #--- Definition of a simple geometry
   geom = TGeoManager("Assemblies",
                      "Geometry using assemblies")
   i = Int_t()
   #--- define some materials
   matVacuum = TGeoMaterial("Vacuum", 0,0,0)
   matAl = TGeoMaterial("Al", 26.98,13,2.7)
   #   #--- define some media
   Vacuum = TGeoMedium("Vacuum",1, matVacuum)
   Al = TGeoMedium("Aluminium",2, matAl)
   
   #--- make the top container volume
   top = geom.MakeBox("TOP", Vacuum, 1000., 1000., 100.)
   geom.SetTopVolume(top)
   
   # Make the elementary assembly of the whole structure
   tplate = TGeoVolumeAssembly("TOOTHPLATE")
   
   ntooth = 5
   xplate = 25
   yplate = 50
   xtooth = 10
   ytooth = 0.5*yplate/ntooth
   dshift = 2.*xplate + xtooth
   xt = yt = Double_t()
   
   plate = geom.MakeBox("PLATE", Al, xplate,yplate,1)
   plate.SetLineColor(kBlue)
   tooth = geom.MakeBox("TOOTH", Al, xtooth,ytooth,1)
   tooth.SetLineColor(kBlue)
   tplate.AddNode(plate,1)

   for  i in range(0,ntooth):
      xt = xplate+xtooth
      yt = -yplate + (4*i+1)*ytooth
      tplate.AddNode(tooth, i+1, TGeoTranslation(xt,yt,0))
      xt = -xplate-xtooth
      yt = -yplate + (4*i+3)*ytooth
      tplate.AddNode(tooth, ntooth+i+1, TGeoTranslation(xt,yt,0))
      
   
   rot1 = TGeoRotation()
   rot1.RotateX(90)
   rot = TGeoRotation ()
   # Make a hexagone cell out of 6 tooth plates. These can zip together
   # without generating overlaps (they are self-contained)
   cell = TGeoVolumeAssembly("CELL")

   for  i in range(0,6):
      phi = 60.*i
      phirad = phi * TMath.DegToRad()
      xp = dshift * TMath.Sin(phirad)
      yp = -dshift * TMath.Cos(phirad)
      rot = TGeoRotation(rot1)
      rot.RotateZ(phi)
      cell.AddNode(tplate,i+1,TGeoCombiTrans(xp,yp,0,rot))
      
   
   # Make a row as an assembly of cells, then combine rows in a honeycomb
   # structure. This again works without any need to define rows as
   # "overlapping"
   row = TGeoVolumeAssembly("ROW")
   ncells = 5

   for  i in range(0,ncells):
      ycell = (2*i+1)*(dshift+10)
      row.AddNode(cell, ncells+i+1, TGeoTranslation(0,ycell,0))
      row.AddNode(cell,ncells-i,TGeoTranslation(0,-ycell,0))
      
   
   dxrow = 3.*(dshift+10.)*TMath.Tan(30.*TMath.DegToRad())
   dyrow = dshift+10.
   nrows = 5

   for  i in range(0,nrows):
      xrow = 0.5*(2*i+1)*dxrow
      yrow = 0.5*dyrow
      if (i%2)==0: yrow = -yrow
      top.AddNode(row, nrows+i+1, TGeoTranslation(xrow,yrow,0))
      top.AddNode(row, nrows-i, TGeoTranslation(-xrow,-yrow,0))
      
   
   #--- close the geometry
   geom.CloseGeometry()
   
   geom.SetVisLevel(4)
   geom.SetVisOption(0)
   top.Draw()
   
   #DelROOTObjs(self) 
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   #print("Deleting objs from gROOT")
   myvars = [x for x in dir() if not x.startswith("__")]
   #myvars = [x for x in vars(self) ] 
   for var in myvars: 
      try:
         exec(f"ROOT.gROOT.Remove({var})")
         #exec(f"ROOT.gROOT.Remove(self.{var})")
         print("deleting", var)
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!
   
   


if __name__ == "__main__":
   myassembly = assembly()
   #del myassembly
