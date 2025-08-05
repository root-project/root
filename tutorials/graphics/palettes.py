## \file
## \ingroup tutorial_graphics
## \notebook
##
## This script draws all of the high definition palettes available in ROOT.
## It generates a png-file for each palette and one pdf-file --with a table of
## contents-- containing all the palettes.
##
## In ROOT, [more than 60 high quality palettes are predefined with 255 colors each].
## Ref: < https:#root.cern/doc/master/classTColor.html#C06 >.
##
## These palettes can be accessed "by name" with `gStyle->SetPalette(num)`. num
## can be taken within the enum given in the previous link. As an example
## `gStyle->SetPalette(kCividis)` will select the following palette.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TF2 = ROOT.TF2
TPaveText = ROOT.TPaveText
TLatex = ROOT.TLatex
TString = ROOT.TString

#types
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr

#constants
kDeepSea = ROOT.kDeepSea
kGreyScale = ROOT.kGreyScale
kDarkBodyRadiator = ROOT.kDarkBodyRadiator
kBlack = ROOT.kBlack

#globals
gStyle = ROOT.gStyle

#TODO = ROOT.TODO: Improve tab-format to equally spaced columns.
#constants
kDeepSea        =        ROOT.kDeepSea  #        Deap Sea
kGreyScale      =        ROOT.kGreyScale        #        Grey Scale
kDarkBodyRadiator       =        ROOT.kDarkBodyRadiator         #        Dark Body Radiator
kBlueYellow     =        ROOT.kBlueYellow       #        Blue Yellow
kRainBow        =        ROOT.kRainBow  #        Rain Bow
kInvertedDarkBodyRadiator       =        ROOT.kInvertedDarkBodyRadiator         #        Inverted Dark Body Radiator
kBird   =        ROOT.kBird     #        Bird
kCubehelix      =        ROOT.kCubehelix        #        Cube helix
kGreenRedViolet         =        ROOT.kGreenRedViolet   #        Green Red Violet
kBlueRedYellow  =        ROOT.kBlueRedYellow    #        Blue Red Yellow
kOcean  =        ROOT.kOcean    #        Ocean
kColorPrintableOnGrey   =        ROOT.kColorPrintableOnGrey     #        Color Printable On Grey
kAlpine         =        ROOT.kAlpine   #        Alpine
kAquamarine     =        ROOT.kAquamarine       #        Aquamarine
kArmy   =        ROOT.kArmy     #        Army
kAtlantic       =        ROOT.kAtlantic         #        Atlantic
kAurora         =        ROOT.kAurora   #        Aurora
kAvocado        =        ROOT.kAvocado  #        Avocado
kBeach  =        ROOT.kBeach    #        Beach
kBlackBody      =        ROOT.kBlackBody        #        Black Body
kBlueGreenYellow        =        ROOT.kBlueGreenYellow  #        Blue Green Yellow
kBrownCyan      =        ROOT.kBrownCyan        #        Brown Cyan
kCMYK   =        ROOT.kCMYK     #        CMYK
kCandy  =        ROOT.kCandy    #        Candy
kCherry         =        ROOT.kCherry   #        Cherry
kCoffee         =        ROOT.kCoffee   #        Coffee
kDarkRainBow    =        ROOT.kDarkRainBow      #        Dark Rain Bow
kDarkTerrain    =        ROOT.kDarkTerrain      #        Dark Terrain
kFall   =        ROOT.kFall     #        Fall
kFruitPunch     =        ROOT.kFruitPunch       #        Fruit Punch
kFuchsia        =        ROOT.kFuchsia  #        Fuchsia
kGreyYellow     =        ROOT.kGreyYellow       #        Grey Yellow
kGreenBrownTerrain      =        ROOT.kGreenBrownTerrain        #        Green Brown Terrain
kGreenPink      =        ROOT.kGreenPink        #        Green Pink
kIsland         =        ROOT.kIsland   #        Island
kLake   =        ROOT.kLake     #        Lake
kLightTemperature       =        ROOT.kLightTemperature         #        Light Temperature
kLightTerrain   =        ROOT.kLightTerrain     #        Light Terrain
kMint   =        ROOT.kMint     #        Mint
kNeon   =        ROOT.kNeon     #        Neon
kPastel         =        ROOT.kPastel   #        Pastel
kPearl  =        ROOT.kPearl    #        Pearl
kPigeon         =        ROOT.kPigeon   #        Pigeon
kPlum   =        ROOT.kPlum     #        Plum
kRedBlue        =        ROOT.kRedBlue  #        Red Blue
kRose   =        ROOT.kRose     #        Rose
kRust   =        ROOT.kRust     #        Rust
kSandyTerrain   =        ROOT.kSandyTerrain     #        Sandy Terrain
kSienna         =        ROOT.kSienna   #        Sienna
kSolar  =        ROOT.kSolar    #        Solar
kSouthWest      =        ROOT.kSouthWest        #        South West
kStarryNight    =        ROOT.kStarryNight      #        Starry Night
kSunset         =        ROOT.kSunset   #        Sunset
kTemperatureMap         =        ROOT.kTemperatureMap   #        Temperature Map
kThermometer    =        ROOT.kThermometer      #        Thermometer
kValentine      =        ROOT.kValentine        #        Valentine
kVisibleSpectrum        =        ROOT.kVisibleSpectrum  #        Visible Spectrum
kWaterMelon     =        ROOT.kWaterMelon       #        Water Melon
kCool   =        ROOT.kCool     #        Cool
kCopper         =        ROOT.kCopper   #        Copper
kGistEarth      =        ROOT.kGistEarth        #        Gist Earth
kViridis        =        ROOT.kViridis  #        Viridis
kCividis        =        ROOT.kCividis  #        Cividis

#types

#globals

c = nullptr

# void
def draw_palette(p : Int_t, n : TString) :
   #del c
   global c
   if c: del c
   c = TCanvas("c","Contours",0,0,500,500)

   global f2
   f2 = TF2("f2","0.1+(1-(x-2)*(x-2))*(1-(y-2)*(y-2))",0.999,3.002,0.999,3.002)
   f2.SetContour(99)
   gStyle.SetPalette(p)
   f2.SetLineWidth(1)
   f2.SetLineColor(kBlack)
   f2.Draw("surf1z")
   
   # Title
   global pt
   pt = TPaveText(10,11,10,11,"blNDC")
   pt.SetName("title")
   pt.Draw()
   n = TString(n)
   num = TString(n) # TString
   num.ReplaceAll(" ","")

   global l
   l = TLatex(-0.8704441,0.9779387, "Palette #%d: %s #scale[0.7]{(#font[82]{k%s})}"%(p,n.Data(),num.Data()))
   l.SetTextFont(42)
   l.SetTextSize(0.035)
   l.Draw()
   c.Draw()
   c.Update()
   c.Print("palette_%d.png"%(p))
   
   global opt
   opt = TString("Title:") + n
   if (p == kDeepSea):
      c.Print("palettes.pdf(", opt.Data())
   elif (p == kCividis):
      c.Print("palettes.pdf)", opt.Data())
   else:
      c.Print("palettes.pdf", opt.Data())

   

# void
def palettes() :

   ROOT.gROOT.SetBatch(1)

   draw_palette(kDeepSea, "Deap Sea")
   draw_palette(kGreyScale, "Grey Scale")
   draw_palette(kDarkBodyRadiator, "Dark Body Radiator")
   draw_palette(kBlueYellow, "Blue Yellow")
   draw_palette(kRainBow, "Rain Bow")
   draw_palette(kInvertedDarkBodyRadiator, "Inverted Dark Body Radiator")
   draw_palette(kBird, "Bird")
   draw_palette(kCubehelix, "Cube helix")
   draw_palette(kGreenRedViolet, "Green Red Violet")
   draw_palette(kBlueRedYellow, "Blue Red Yellow")
   draw_palette(kOcean, "Ocean")
   draw_palette(kColorPrintableOnGrey, "Color Printable On Grey")
   draw_palette(kAlpine, "Alpine")
   draw_palette(kAquamarine, "Aquamarine")
   draw_palette(kArmy, "Army")
   draw_palette(kAtlantic, "Atlantic")
   draw_palette(kAurora, "Aurora")
   draw_palette(kAvocado, "Avocado")
   draw_palette(kBeach, "Beach")
   draw_palette(kBlackBody, "Black Body")
   draw_palette(kBlueGreenYellow, "Blue Green Yellow")
   draw_palette(kBrownCyan, "Brown Cyan")
   draw_palette(kCMYK, "CMYK")
   draw_palette(kCandy, "Candy")
   draw_palette(kCherry, "Cherry")
   draw_palette(kCoffee, "Coffee")
   draw_palette(kDarkRainBow, "Dark Rain Bow")
   draw_palette(kDarkTerrain, "Dark Terrain")
   draw_palette(kFall, "Fall")
   draw_palette(kFruitPunch, "Fruit Punch")
   draw_palette(kFuchsia, "Fuchsia")
   draw_palette(kGreyYellow, "Grey Yellow")
   draw_palette(kGreenBrownTerrain, "Green Brown Terrain")
   draw_palette(kGreenPink, "Green Pink")
   draw_palette(kIsland, "Island")
   draw_palette(kLake, "Lake")
   draw_palette(kLightTemperature, "Light Temperature")
   draw_palette(kLightTerrain, "Light Terrain")
   draw_palette(kMint, "Mint")
   draw_palette(kNeon, "Neon")
   draw_palette(kPastel, "Pastel")
   draw_palette(kPearl, "Pearl")
   draw_palette(kPigeon, "Pigeon")
   draw_palette(kPlum, "Plum")
   draw_palette(kRedBlue, "Red Blue")
   draw_palette(kRose, "Rose")
   draw_palette(kRust, "Rust")
   draw_palette(kSandyTerrain, "Sandy Terrain")
   draw_palette(kSienna, "Sienna")
   draw_palette(kSolar, "Solar")
   draw_palette(kSouthWest, "South West")
   draw_palette(kStarryNight, "Starry Night")
   draw_palette(kSunset, "Sunset")
   draw_palette(kTemperatureMap, "Temperature Map")
   draw_palette(kThermometer, "Thermometer")
   draw_palette(kValentine, "Valentine")
   draw_palette(kVisibleSpectrum, "Visible Spectrum")
   draw_palette(kWaterMelon, "Water Melon")
   draw_palette(kCool, "Cool")
   draw_palette(kCopper, "Copper")
   draw_palette(kGistEarth, "Gist Earth")
   draw_palette(kViridis, "Viridis")
   draw_palette(kCividis, "Cividis")
   
   print( "\n" ) 
   print( 20*">> " )
   print( "Many palette_*.png files have been created. See your current directory." )



if __name__ == "__main__":
   palettes()
