/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// This macro draws all the high definition palettes available in ROOT.
/// It generates a png file for each palette and one pdf file, with a table of
/// content, containing all the palettes.
///
/// In ROOT, [more than 60 high quality palettes are predefined with 255 colors each](https://root.cern/doc/master/classTColor.html#C06).
///
/// These palettes can be accessed "by name" with `gStyle->SetPalette(num)`. num
/// can be taken within the enum given in the previous link. As an example
/// `gStyle->SetPalette(kCividis)` will select the following palette.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

TCanvas *c;

void draw_palette(int p, TString n){
   delete c;
   c  = new TCanvas("c","Contours",0,0,500,500);
   TF2 *f2 = new TF2("f2","0.1+(1-(x-2)*(x-2))*(1-(y-2)*(y-2))",0.999,3.002,0.999,3.002);
   f2->SetContour(99);
   gStyle->SetPalette(p);
   f2->SetLineWidth(1);
   f2->SetLineColor(kBlack);
   f2->Draw("surf1z");

   // Title
   TPaveText *pt = new TPaveText(10,11,10,11,"blNDC");
   pt->SetName("title");
   pt->Draw();
   TString num = n;
   num.ReplaceAll(" ","");
   TLatex *l = new TLatex(-0.8704441,0.9779387,Form("Palette #%d: %s #scale[0.7]{(#font[82]{k%s})}",p,n.Data(),num.Data()));
   l->SetTextFont(42);
   l->SetTextSize(0.035);
   l->Draw();
   c->Update();
   c->Print(Form("palette_%d.png",p));
   if (p==51)  {c->Print("palettes.pdf(", Form("Title:%s",n.Data())); return;}
   if (p==111) {c->Print("palettes.pdf)", Form("Title:%s",n.Data())); return;}
   c->Print("palettes.pdf", Form("Title:%s",n.Data()));
}

void palettes() {
   gROOT->SetBatch(1);
   c  = new TCanvas("c","Contours",0,0,500,500);
   draw_palette(kDeepSea, "Deap Sea");
   draw_palette(kGreyScale, "Grey Scale");
   draw_palette(kDarkBodyRadiator, "Dark Body Radiator");
   draw_palette(kBlueYellow, "Blue Yellow");
   draw_palette(kRainBow, "Rain Bow");
   draw_palette(kInvertedDarkBodyRadiator, "Inverted Dark Body Radiator");
   draw_palette(kBird, "Bird");
   draw_palette(kCubehelix, "Cube helix");
   draw_palette(kGreenRedViolet, "Green Red Violet");
   draw_palette(kBlueRedYellow, "Blue Red Yellow");
   draw_palette(kOcean, "Ocean");
   draw_palette(kColorPrintableOnGrey, "Color Printable On Grey");
   draw_palette(kAlpine, "Alpine");
   draw_palette(kAquamarine, "Aquamarine");
   draw_palette(kArmy, "Army");
   draw_palette(kAtlantic, "Atlantic");
   draw_palette(kAurora, "Aurora");
   draw_palette(kAvocado, "Avocado");
   draw_palette(kBeach, "Beach");
   draw_palette(kBlackBody, "Black Body");
   draw_palette(kBlueGreenYellow, "Blue Green Yellow");
   draw_palette(kBrownCyan, "Brown Cyan");
   draw_palette(kCMYK, "CMYK");
   draw_palette(kCandy, "Candy");
   draw_palette(kCherry, "Cherry");
   draw_palette(kCoffee, "Coffee");
   draw_palette(kDarkRainBow, "Dark Rain Bow");
   draw_palette(kDarkTerrain, "Dark Terrain");
   draw_palette(kFall, "Fall");
   draw_palette(kFruitPunch, "Fruit Punch");
   draw_palette(kFuchsia, "Fuchsia");
   draw_palette(kGreyYellow, "Grey Yellow");
   draw_palette(kGreenBrownTerrain, "Green Brown Terrain");
   draw_palette(kGreenPink, "Green Pink");
   draw_palette(kIsland, "Island");
   draw_palette(kLake, "Lake");
   draw_palette(kLightTemperature, "Light Temperature");
   draw_palette(kLightTerrain, "Light Terrain");
   draw_palette(kMint, "Mint");
   draw_palette(kNeon, "Neon");
   draw_palette(kPastel, "Pastel");
   draw_palette(kPearl, "Pearl");
   draw_palette(kPigeon, "Pigeon");
   draw_palette(kPlum, "Plum");
   draw_palette(kRedBlue, "Red Blue");
   draw_palette(kRose, "Rose");
   draw_palette(kRust, "Rust");
   draw_palette(kSandyTerrain, "Sandy Terrain");
   draw_palette(kSienna, "Sienna");
   draw_palette(kSolar, "Solar");
   draw_palette(kSouthWest, "South West");
   draw_palette(kStarryNight, "Starry Night");
   draw_palette(kSunset, "Sunset");
   draw_palette(kTemperatureMap, "Temperature Map");
   draw_palette(kThermometer, "Thermometer");
   draw_palette(kValentine, "Valentine");
   draw_palette(kVisibleSpectrum, "Visible Spectrum");
   draw_palette(kWaterMelon, "Water Melon");
   draw_palette(kCool, "Cool");
   draw_palette(kCopper, "Copper");
   draw_palette(kGistEarth, "Gist Earth");
   draw_palette(kViridis, "Viridis");
   draw_palette(kCividis, "Cividis");
}

