## \file
## \ingroup tutorial_geom
## Tests importing/exporting optical surfaces from GDML
##
## Optical surfaces, skin surfaces and border surfaces are imported in object arrays
## stored by TGeoManager class. Optical surfaces do not store property arrays but point
## to GDML matrices describing such properties. One can get the data for such property
## like:
##   surf = geom.GetOpticalSurface("surf1")   # TGeoOpticalSurface
##   property = surf.GetPropertyRef("REFLECTIVITY")   # char
##   m = geom.GetGDMLMatrix(property)   # TGeoGDMLMatrix
## Surfaces and border surfaces can be retrieved from the TGeoManager object by using:   # Skin
##   skin_array = geom.GetListOfSkinSurfaces()   # TObjArray
##   border_array = geom.GetListOfBorderSurfaces()   # TObjArray
## Alternatively accessors by name can also be used: GetSkinSurface(name)/GetBorderSurface(name)
##
## \author Andrei Gheata
## \translator P. P.


import ROOT

TObjArray = ROOT.TObjArray 
TROOT = ROOT.TROOT 
TGeoOpticalSurface = ROOT.TGeoOpticalSurface 
TGeoManager = ROOT.TGeoManager 
TIter = ROOT.TIter
TString = ROOT.TString


# float 
def Checksum(geom = TGeoManager) :
   Sum = 0. # float
   Next = TIter(geom.GetListOfOpticalSurfaces())
   surfaces = (geom.GetListOfOpticalSurfaces())
   surf = TGeoOpticalSurface() 
   #while ((surf := (TGeoOpticalSurface )Next())) {
   #while True: #((surf := (TGeoOpticalSurface )Next())) 
   for surf in surfaces : #((surf := (TGeoOpticalSurface )Next())) 
      Sum += surf.GetType() + surf.GetModel() + surf.GetFinish() + surf.GetValue()
      name = TString(surf.GetName())
      Sum += name.Hash()
      name = TString(surf.GetTitle())
      Sum += name.Hash()
      
   return Sum
   

# int
def testoptical() :
   geofile = str(ROOT.gROOT.GetTutorialDir()) + "/geom/gdml/opticalsurfaces.gdml"
   geofile = TString(geofile)
   geofile.ReplaceAll("\\", "/")
   TGeoManager.SetExportPrecision(8)
   TGeoManager.SetVerboseLevel(0)
   print("=== Importing ", geofile.Data(), "...")
   geom = TGeoManager.Import(str(geofile))
   print("=== List of GDML matrices:\n")
   geom.GetListOfGDMLMatrices().Print()
   print("=== List of optical surfaces:\n")
   geom.GetListOfOpticalSurfaces().Print()
   print("=== List of skin surfaces:\n")
   geom.GetListOfSkinSurfaces().Print()
   print("=== List of border surfaces:\n")
   geom.GetListOfBorderSurfaces().Print()
   # Compute some checksum for optical surfaces
   checksum1 = Checksum(geom)
   print("=== Exporting as .gdml, then importing back\n")
   geom.Export("tmp.gdml")
   geom = TGeoManager.Import("tmp.gdml")
   checksum2 = Checksum(geom)
   assert((checksum2 == checksum1) and "Exporting/importing as .gdml not OK")
   print("=== Exporting as .root, then importing back\n")
   geom.Export("tmp.root")
   geom = TGeoManager.Import("tmp.root")
   checksum3 = Checksum(geom)
   assert((checksum3 == checksum1) and "Exporting/importing as .root not OK")
   print("all OK\n")
  

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
         print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!

 
   return 0
   


if __name__ == "__main__":
   testoptical()
