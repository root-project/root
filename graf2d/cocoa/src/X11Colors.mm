// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   28/11/2011

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#define NDEBUG

#include "X11Colors.h"
#include "TError.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {

namespace {

//______________________________________________________________________________
bool HexCharToInt(char symbol, unsigned &val)
{
   if (symbol >= '0' && symbol <= '9') {
      val = symbol - '0';
      return true;
   }

   switch (symbol) {
   case 'a': case 'A':
      val = 10;
      return true;
   case 'b': case 'B':
      val = 11;
      return true;
   case 'c': case 'C':
      val = 12;
      return true;
   case 'd': case 'D':
      val = 13;
      return true;
   case 'e': case 'E':
      val = 14;
      return true;
   case 'f': case 'F':
      val = 15;
      return true;
   default:
      return false;
   }
}

//______________________________________________________________________________
bool GetHex(const TString &rgb, Ssiz_t first, Ssiz_t len, unsigned &component)
{
   const Ssiz_t last = first + len;
   for (; first < last; ++first) {
      unsigned val = 0;
      if (!HexCharToInt(rgb[first], val)) {
         ::Error("ROOT::MacOSX::X11::GetGex",
                  "Bad symbol in color component, hex digit expected, got %c", rgb[first]);
         return false;
      } else
         component = (component << 4) | val;
   }

   return true;
}

}

//______________________________________________________________________________
bool ColorParser::ParseColor(const TString &colorName, ColorStruct_t &color)const
{
   if (colorName.Length()) {
      if (colorName[0] == '#')
         return ParseRGBTriplet(colorName, color);
      else
         return LookupColorByName(colorName, color);
   }

   return false;
}

//______________________________________________________________________________
bool ColorParser::ParseRGBTriplet(const TString &rgb, ColorStruct_t &color)const
{
   //Minimal triplet is #rgb, max. is #rrrrggggbbbb
   const Ssiz_t len = rgb.Length();
   if (len < 4 || len > 13 || (len - 1) % 3) {
      //Bad format.
      ::Error("ROOT::MacOSX::X11::ParseRGBTriplet",
              "Bad color name or rgb triplet %s", rgb.Data());
      return false;
   }

   //TGX11 and TGWin32 sets this member to zero:
   color.fPixel = 0;

   const Ssiz_t compLen = (len - 1) / 3;
   unsigned r = 0, g = 0, b = 0;
   if (GetHex(rgb, 1, compLen, r) && GetHex(rgb, 1 + compLen, compLen, g) &&
       GetHex(rgb, 1 + compLen * 2, compLen, b))
   {
      //Problem with bitPad: ROOT/X11's "pixel" uses 0xXX for component
      //(after all, pixel is 4 bytes in most cases),
      //X11's color component may be 0xXXXX.

      const unsigned bitPad = 16 - compLen * 4;
      color.fRed   = r << bitPad;
      color.fGreen = g << bitPad;
      color.fBlue  = b << bitPad;

      return true;
   }

   return false;
}

//______________________________________________________________________________
bool ColorParser::LookupColorByName(const TString &colorName, ColorStruct_t &color)const
{
   TString lowerCaseName(colorName);
   lowerCaseName.ToLower();
   const_rgb_iterator it = fX11RGB.find(lowerCaseName);

   if (it != fX11RGB.end()) {
      color.fPixel = 0;

      color.fRed = it->second.fRed * 0xFFFF / 0xFF;
      color.fGreen = it->second.fGreen * 0xFFFF / 0xFF;
      color.fBlue = it->second.fBlue * 0xFFFF / 0xFF;

      return true;
   } else {
      ::Error("ROOT::MacOSX::X11::ColorParser::LookupColorByName",
              "Could not find color with name %s", colorName.Data());
      return false;
   }
}

//______________________________________________________________________________
ColorParser::ColorParser()
{
   //Ugly map. May be, I'll do somehting better later.
   //Names are sorted here, I can place them in some sequence
   //and use binary search algorithms, for example.
   fX11RGB["alice blue"] = RGB_t(240, 248, 255);
   fX11RGB["AliceBlue"] = RGB_t(240, 248, 255);
   fX11RGB["antique white"] = RGB_t(250, 235, 215);
   fX11RGB["AntiqueWhite"] = RGB_t(250, 235, 215);
   fX11RGB["AntiqueWhite1"] = RGB_t(255, 239, 219);
   fX11RGB["AntiqueWhite2"] = RGB_t(238, 223, 204);
   fX11RGB["AntiqueWhite3"] = RGB_t(205, 192, 176);
   fX11RGB["AntiqueWhite4"] = RGB_t(139, 131, 120);
   fX11RGB["aquamarine"] = RGB_t(127, 255, 212);
   fX11RGB["aquamarine1"] = RGB_t(127, 255, 212);
   fX11RGB["aquamarine2"] = RGB_t(118, 238, 198);
   fX11RGB["aquamarine3"] = RGB_t(102, 205, 170);
   fX11RGB["aquamarine4"] = RGB_t(69, 139, 116);
   fX11RGB["azure"] = RGB_t(240, 255, 255);
   fX11RGB["azure1"] = RGB_t(240, 255, 255);
   fX11RGB["azure2"] = RGB_t(224, 238, 238);
   fX11RGB["azure3"] = RGB_t(193, 205, 205);
   fX11RGB["azure4"] = RGB_t(131, 139, 139);
   fX11RGB["beige"] = RGB_t(245, 245, 220);
   fX11RGB["bisque"] = RGB_t(255, 228, 196);
   fX11RGB["bisque1"] = RGB_t(255, 228, 196);
   fX11RGB["bisque2"] = RGB_t(238, 213, 183);
   fX11RGB["bisque3"] = RGB_t(205, 183, 158);
   fX11RGB["bisque4"] = RGB_t(139, 125, 107);
   fX11RGB["black"] = RGB_t(0, 0, 0);
   fX11RGB["blanched almond"] = RGB_t(255, 235, 205);
   fX11RGB["BlanchedAlmond"] = RGB_t(255, 235, 205);
   fX11RGB["blue"] = RGB_t(0, 0, 255);
   fX11RGB["blue violet"] = RGB_t(138, 43, 226);
   fX11RGB["blue1"] = RGB_t(0, 0, 255);
   fX11RGB["blue2"] = RGB_t(0, 0, 238);
   fX11RGB["blue3"] = RGB_t(0, 0, 205);
   fX11RGB["blue4"] = RGB_t(0, 0, 139);
   fX11RGB["BlueViolet"] = RGB_t(138, 43, 226);
   fX11RGB["brown"] = RGB_t(165, 42, 42);
   fX11RGB["brown1"] = RGB_t(255, 64, 64);
   fX11RGB["brown2"] = RGB_t(238, 59, 59);
   fX11RGB["brown3"] = RGB_t(205, 51, 51);
   fX11RGB["brown4"] = RGB_t(139, 35, 35);
   fX11RGB["burlywood"] = RGB_t(222, 184, 135);
   fX11RGB["burlywood1"] = RGB_t(255, 211, 155);
   fX11RGB["burlywood2"] = RGB_t(238, 197, 145);
   fX11RGB["burlywood3"] = RGB_t(205, 170, 125);
   fX11RGB["burlywood4"] = RGB_t(139, 115, 85);
   fX11RGB["cadet blue"] = RGB_t(95, 158, 160);
   fX11RGB["CadetBlue"] = RGB_t(95, 158, 160);
   fX11RGB["CadetBlue1"] = RGB_t(152, 245, 255);
   fX11RGB["CadetBlue2"] = RGB_t(142, 229, 238);
   fX11RGB["CadetBlue3"] = RGB_t(122, 197, 205);
   fX11RGB["CadetBlue4"] = RGB_t(83, 134, 139);
   fX11RGB["chartreuse"] = RGB_t(127, 255, 0);
   fX11RGB["chartreuse1"] = RGB_t(127, 255, 0);
   fX11RGB["chartreuse2"] = RGB_t(118, 238, 0);
   fX11RGB["chartreuse3"] = RGB_t(102, 205, 0);
   fX11RGB["chartreuse4"] = RGB_t(69, 139, 0);
   fX11RGB["chocolate"] = RGB_t(210, 105, 30);
   fX11RGB["chocolate1"] = RGB_t(255, 127, 36);
   fX11RGB["chocolate2"] = RGB_t(238, 118, 33);
   fX11RGB["chocolate3"] = RGB_t(205, 102, 29);
   fX11RGB["chocolate4"] = RGB_t(139, 69, 19);
   fX11RGB["coral"] = RGB_t(255, 127, 80);
   fX11RGB["coral1"] = RGB_t(255, 114, 86);
   fX11RGB["coral2"] = RGB_t(238, 106, 80);
   fX11RGB["coral3"] = RGB_t(205, 91, 69);
   fX11RGB["coral4"] = RGB_t(139, 62, 47);
   fX11RGB["cornflower blue"] = RGB_t(100, 149, 237);
   fX11RGB["CornflowerBlue"] = RGB_t(100, 149, 237);
   fX11RGB["cornsilk"] = RGB_t(255, 248, 220);
   fX11RGB["cornsilk1"] = RGB_t(255, 248, 220);
   fX11RGB["cornsilk2"] = RGB_t(238, 232, 205);
   fX11RGB["cornsilk3"] = RGB_t(205, 200, 177);
   fX11RGB["cornsilk4"] = RGB_t(139, 136, 120);
   fX11RGB["cyan"] = RGB_t(0, 255, 255);
   fX11RGB["cyan1"] = RGB_t(0, 255, 255);
   fX11RGB["cyan2"] = RGB_t(0, 238, 238);
   fX11RGB["cyan3"] = RGB_t(0, 205, 205);
   fX11RGB["cyan4"] = RGB_t(0, 139, 139);
   fX11RGB["dark blue"] = RGB_t(0, 0, 139);
   fX11RGB["dark cyan"] = RGB_t(0, 139, 139);
   fX11RGB["dark goldenrod"] = RGB_t(184, 134, 11);
   fX11RGB["dark gray"] = RGB_t(169, 169, 169);
   fX11RGB["dark green"] = RGB_t(0, 100, 0);
   fX11RGB["dark grey"] = RGB_t(169, 169, 169);
   fX11RGB["dark khaki"] = RGB_t(189, 183, 107);
   fX11RGB["dark magenta"] = RGB_t(139, 0, 139);
   fX11RGB["dark olive green"] = RGB_t(85, 107, 47);
   fX11RGB["dark orange"] = RGB_t(255, 140, 0);
   fX11RGB["dark orchid"] = RGB_t(153, 50, 204);
   fX11RGB["dark red"] = RGB_t(139, 0, 0);
   fX11RGB["dark salmon"] = RGB_t(233, 150, 122);
   fX11RGB["dark sea green"] = RGB_t(143, 188, 143);
   fX11RGB["dark slate blue"] = RGB_t(72, 61, 139);
   fX11RGB["dark slate gray"] = RGB_t(47, 79, 79);
   fX11RGB["dark slate grey"] = RGB_t(47, 79, 79);
   fX11RGB["dark turquoise"] = RGB_t(0, 206, 209);
   fX11RGB["dark violet"] = RGB_t(148, 0, 211);
   fX11RGB["DarkBlue"] = RGB_t(0, 0, 139);
   fX11RGB["DarkCyan"] = RGB_t(0, 139, 139);
   fX11RGB["DarkGoldenrod"] = RGB_t(184, 134, 11);
   fX11RGB["DarkGoldenrod1"] = RGB_t(255, 185, 15);
   fX11RGB["DarkGoldenrod2"] = RGB_t(238, 173, 14);
   fX11RGB["DarkGoldenrod3"] = RGB_t(205, 149, 12);
   fX11RGB["DarkGoldenrod4"] = RGB_t(139, 101, 8);
   fX11RGB["DarkGray"] = RGB_t(169, 169, 169);
   fX11RGB["DarkGreen"] = RGB_t(0, 100, 0);
   fX11RGB["DarkGrey"] = RGB_t(169, 169, 169);
   fX11RGB["DarkKhaki"] = RGB_t(189, 183, 107);
   fX11RGB["DarkMagenta"] = RGB_t(139, 0, 139);
   fX11RGB["DarkOliveGreen"] = RGB_t(85, 107, 47);
   fX11RGB["DarkOliveGreen1"] = RGB_t(202, 255, 112);
   fX11RGB["DarkOliveGreen2"] = RGB_t(188, 238, 104);
   fX11RGB["DarkOliveGreen3"] = RGB_t(162, 205, 90);
   fX11RGB["DarkOliveGreen4"] = RGB_t(110, 139, 61);
   fX11RGB["DarkOrange"] = RGB_t(255, 140, 0);
   fX11RGB["DarkOrange1"] = RGB_t(255, 127, 0);
   fX11RGB["DarkOrange2"] = RGB_t(238, 118, 0);
   fX11RGB["DarkOrange3"] = RGB_t(205, 102, 0);
   fX11RGB["DarkOrange4"] = RGB_t(139, 69, 0);
   fX11RGB["DarkOrchid"] = RGB_t(153, 50, 204);
   fX11RGB["DarkOrchid1"] = RGB_t(191, 62, 255);
   fX11RGB["DarkOrchid2"] = RGB_t(178, 58, 238);
   fX11RGB["DarkOrchid3"] = RGB_t(154, 50, 205);
   fX11RGB["DarkOrchid4"] = RGB_t(104, 34, 139);
   fX11RGB["DarkRed"] = RGB_t(139, 0, 0);
   fX11RGB["DarkSalmon"] = RGB_t(233, 150, 122);
   fX11RGB["DarkSeaGreen"] = RGB_t(143, 188, 143);
   fX11RGB["DarkSeaGreen1"] = RGB_t(193, 255, 193);
   fX11RGB["DarkSeaGreen2"] = RGB_t(180, 238, 180);
   fX11RGB["DarkSeaGreen3"] = RGB_t(155, 205, 155);
   fX11RGB["DarkSeaGreen4"] = RGB_t(105, 139, 105);
   fX11RGB["DarkSlateBlue"] = RGB_t(72, 61, 139);
   fX11RGB["DarkSlateGray"] = RGB_t(47, 79, 79);
   fX11RGB["DarkSlateGray1"] = RGB_t(151, 255, 255);
   fX11RGB["DarkSlateGray2"] = RGB_t(141, 238, 238);
   fX11RGB["DarkSlateGray3"] = RGB_t(121, 205, 205);
   fX11RGB["DarkSlateGray4"] = RGB_t(82, 139, 139);
   fX11RGB["DarkSlateGrey"] = RGB_t(47, 79, 79);
   fX11RGB["DarkTurquoise"] = RGB_t(0, 206, 209);
   fX11RGB["DarkViolet"] = RGB_t(148, 0, 211);
   fX11RGB["deep pink"] = RGB_t(255, 20, 147);
   fX11RGB["deep sky blue"] = RGB_t(0, 191, 255);
   fX11RGB["DeepPink"] = RGB_t(255, 20, 147);
   fX11RGB["DeepPink1"] = RGB_t(255, 20, 147);
   fX11RGB["DeepPink2"] = RGB_t(238, 18, 137);
   fX11RGB["DeepPink3"] = RGB_t(205, 16, 118);
   fX11RGB["DeepPink4"] = RGB_t(139, 10, 80);
   fX11RGB["DeepSkyBlue"] = RGB_t(0, 191, 255);
   fX11RGB["DeepSkyBlue1"] = RGB_t(0, 191, 255);
   fX11RGB["DeepSkyBlue2"] = RGB_t(0, 178, 238);
   fX11RGB["DeepSkyBlue3"] = RGB_t(0, 154, 205);
   fX11RGB["DeepSkyBlue4"] = RGB_t(0, 104, 139);
   fX11RGB["dim gray"] = RGB_t(105, 105, 105);
   fX11RGB["dim grey"] = RGB_t(105, 105, 105);
   fX11RGB["DimGray"] = RGB_t(105, 105, 105);
   fX11RGB["DimGrey"] = RGB_t(105, 105, 105);
   fX11RGB["dodger blue"] = RGB_t(30, 144, 255);
   fX11RGB["DodgerBlue"] = RGB_t(30, 144, 255);
   fX11RGB["DodgerBlue1"] = RGB_t(30, 144, 255);
   fX11RGB["DodgerBlue2"] = RGB_t(28, 134, 238);
   fX11RGB["DodgerBlue3"] = RGB_t(24, 116, 205);
   fX11RGB["DodgerBlue4"] = RGB_t(16, 78, 139);
   fX11RGB["firebrick"] = RGB_t(178, 34, 34);
   fX11RGB["firebrick1"] = RGB_t(255, 48, 48);
   fX11RGB["firebrick2"] = RGB_t(238, 44, 44);
   fX11RGB["firebrick3"] = RGB_t(205, 38, 38);
   fX11RGB["firebrick4"] = RGB_t(139, 26, 26);
   fX11RGB["floral white"] = RGB_t(255, 250, 240);
   fX11RGB["FloralWhite"] = RGB_t(255, 250, 240);
   fX11RGB["forest green"] = RGB_t(34, 139, 34);
   fX11RGB["ForestGreen"] = RGB_t(34, 139, 34);
   fX11RGB["gainsboro"] = RGB_t(220, 220, 220);
   fX11RGB["ghost white"] = RGB_t(248, 248, 255);
   fX11RGB["GhostWhite"] = RGB_t(248, 248, 255);
   fX11RGB["gold"] = RGB_t(255, 215, 0);
   fX11RGB["gold1"] = RGB_t(255, 215, 0);
   fX11RGB["gold2"] = RGB_t(238, 201, 0);
   fX11RGB["gold3"] = RGB_t(205, 173, 0);
   fX11RGB["gold4"] = RGB_t(139, 117, 0);
   fX11RGB["goldenrod"] = RGB_t(218, 165, 32);
   fX11RGB["goldenrod1"] = RGB_t(255, 193, 37);
   fX11RGB["goldenrod2"] = RGB_t(238, 180, 34);
   fX11RGB["goldenrod3"] = RGB_t(205, 155, 29);
   fX11RGB["goldenrod4"] = RGB_t(139, 105, 20);
   fX11RGB["gray"] = RGB_t(190, 190, 190);
   fX11RGB["gray0"] = RGB_t(0, 0, 0);
   fX11RGB["gray1"] = RGB_t(3, 3, 3);
   fX11RGB["gray10"] = RGB_t(26, 26, 26);
   fX11RGB["gray100"] = RGB_t(255, 255, 255);
   fX11RGB["gray11"] = RGB_t(28, 28, 28);
   fX11RGB["gray12"] = RGB_t(31, 31, 31);
   fX11RGB["gray13"] = RGB_t(33, 33, 33);
   fX11RGB["gray14"] = RGB_t(36, 36, 36);
   fX11RGB["gray15"] = RGB_t(38, 38, 38);
   fX11RGB["gray16"] = RGB_t(41, 41, 41);
   fX11RGB["gray17"] = RGB_t(43, 43, 43);
   fX11RGB["gray18"] = RGB_t(46, 46, 46);
   fX11RGB["gray19"] = RGB_t(48, 48, 48);
   fX11RGB["gray2"] = RGB_t(5, 5, 5);
   fX11RGB["gray20"] = RGB_t(51, 51, 51);
   fX11RGB["gray21"] = RGB_t(54, 54, 54);
   fX11RGB["gray22"] = RGB_t(56, 56, 56);
   fX11RGB["gray23"] = RGB_t(59, 59, 59);
   fX11RGB["gray24"] = RGB_t(61, 61, 61);
   fX11RGB["gray25"] = RGB_t(64, 64, 64);
   fX11RGB["gray26"] = RGB_t(66, 66, 66);
   fX11RGB["gray27"] = RGB_t(69, 69, 69);
   fX11RGB["gray28"] = RGB_t(71, 71, 71);
   fX11RGB["gray29"] = RGB_t(74, 74, 74);
   fX11RGB["gray3"] = RGB_t(8, 8, 8);
   fX11RGB["gray30"] = RGB_t(77, 77, 77);
   fX11RGB["gray31"] = RGB_t(79, 79, 79);
   fX11RGB["gray32"] = RGB_t(82, 82, 82);
   fX11RGB["gray33"] = RGB_t(84, 84, 84);
   fX11RGB["gray34"] = RGB_t(87, 87, 87);
   fX11RGB["gray35"] = RGB_t(89, 89, 89);
   fX11RGB["gray36"] = RGB_t(92, 92, 92);
   fX11RGB["gray37"] = RGB_t(94, 94, 94);
   fX11RGB["gray38"] = RGB_t(97, 97, 97);
   fX11RGB["gray39"] = RGB_t(99, 99, 99);
   fX11RGB["gray4"] = RGB_t(10, 10, 10);
   fX11RGB["gray40"] = RGB_t(102, 102, 102);
   fX11RGB["gray41"] = RGB_t(105, 105, 105);
   fX11RGB["gray42"] = RGB_t(107, 107, 107);
   fX11RGB["gray43"] = RGB_t(110, 110, 110);
   fX11RGB["gray44"] = RGB_t(112, 112, 112);
   fX11RGB["gray45"] = RGB_t(115, 115, 115);
   fX11RGB["gray46"] = RGB_t(117, 117, 117);
   fX11RGB["gray47"] = RGB_t(120, 120, 120);
   fX11RGB["gray48"] = RGB_t(122, 122, 122);
   fX11RGB["gray49"] = RGB_t(125, 125, 125);
   fX11RGB["gray5"] = RGB_t(13, 13, 13);
   fX11RGB["gray50"] = RGB_t(127, 127, 127);
   fX11RGB["gray51"] = RGB_t(130, 130, 130);
   fX11RGB["gray52"] = RGB_t(133, 133, 133);
   fX11RGB["gray53"] = RGB_t(135, 135, 135);
   fX11RGB["gray54"] = RGB_t(138, 138, 138);
   fX11RGB["gray55"] = RGB_t(140, 140, 140);
   fX11RGB["gray56"] = RGB_t(143, 143, 143);
   fX11RGB["gray57"] = RGB_t(145, 145, 145);
   fX11RGB["gray58"] = RGB_t(148, 148, 148);
   fX11RGB["gray59"] = RGB_t(150, 150, 150);
   fX11RGB["gray6"] = RGB_t(15, 15, 15);
   fX11RGB["gray60"] = RGB_t(153, 153, 153);
   fX11RGB["gray61"] = RGB_t(156, 156, 156);
   fX11RGB["gray62"] = RGB_t(158, 158, 158);
   fX11RGB["gray63"] = RGB_t(161, 161, 161);
   fX11RGB["gray64"] = RGB_t(163, 163, 163);
   fX11RGB["gray65"] = RGB_t(166, 166, 166);
   fX11RGB["gray66"] = RGB_t(168, 168, 168);
   fX11RGB["gray67"] = RGB_t(171, 171, 171);
   fX11RGB["gray68"] = RGB_t(173, 173, 173);
   fX11RGB["gray69"] = RGB_t(176, 176, 176);
   fX11RGB["gray7"] = RGB_t(18, 18, 18);
   fX11RGB["gray70"] = RGB_t(179, 179, 179);
   fX11RGB["gray71"] = RGB_t(181, 181, 181);
   fX11RGB["gray72"] = RGB_t(184, 184, 184);
   fX11RGB["gray73"] = RGB_t(186, 186, 186);
   fX11RGB["gray74"] = RGB_t(189, 189, 189);
   fX11RGB["gray75"] = RGB_t(191, 191, 191);
   fX11RGB["gray76"] = RGB_t(194, 194, 194);
   fX11RGB["gray77"] = RGB_t(196, 196, 196);
   fX11RGB["gray78"] = RGB_t(199, 199, 199);
   fX11RGB["gray79"] = RGB_t(201, 201, 201);
   fX11RGB["gray8"] = RGB_t(20, 20, 20);
   fX11RGB["gray80"] = RGB_t(204, 204, 204);
   fX11RGB["gray81"] = RGB_t(207, 207, 207);
   fX11RGB["gray82"] = RGB_t(209, 209, 209);
   fX11RGB["gray83"] = RGB_t(212, 212, 212);
   fX11RGB["gray84"] = RGB_t(214, 214, 214);
   fX11RGB["gray85"] = RGB_t(217, 217, 217);
   fX11RGB["gray86"] = RGB_t(219, 219, 219);
   fX11RGB["gray87"] = RGB_t(222, 222, 222);
   fX11RGB["gray88"] = RGB_t(224, 224, 224);
   fX11RGB["gray89"] = RGB_t(227, 227, 227);
   fX11RGB["gray9"] = RGB_t(23, 23, 23);
   fX11RGB["gray90"] = RGB_t(229, 229, 229);
   fX11RGB["gray91"] = RGB_t(232, 232, 232);
   fX11RGB["gray92"] = RGB_t(235, 235, 235);
   fX11RGB["gray93"] = RGB_t(237, 237, 237);
   fX11RGB["gray94"] = RGB_t(240, 240, 240);
   fX11RGB["gray95"] = RGB_t(242, 242, 242);
   fX11RGB["gray96"] = RGB_t(245, 245, 245);
   fX11RGB["gray97"] = RGB_t(247, 247, 247);
   fX11RGB["gray98"] = RGB_t(250, 250, 250);
   fX11RGB["gray99"] = RGB_t(252, 252, 252);
   fX11RGB["green"] = RGB_t(0, 255, 0);
   fX11RGB["green yellow"] = RGB_t(173, 255, 47);
   fX11RGB["green1"] = RGB_t(0, 255, 0);
   fX11RGB["green2"] = RGB_t(0, 238, 0);
   fX11RGB["green3"] = RGB_t(0, 205, 0);
   fX11RGB["green4"] = RGB_t(0, 139, 0);
   fX11RGB["GreenYellow"] = RGB_t(173, 255, 47);
   fX11RGB["grey"] = RGB_t(190, 190, 190);
   fX11RGB["grey0"] = RGB_t(0, 0, 0);
   fX11RGB["grey1"] = RGB_t(3, 3, 3);
   fX11RGB["grey10"] = RGB_t(26, 26, 26);
   fX11RGB["grey100"] = RGB_t(255, 255, 255);
   fX11RGB["grey11"] = RGB_t(28, 28, 28);
   fX11RGB["grey12"] = RGB_t(31, 31, 31);
   fX11RGB["grey13"] = RGB_t(33, 33, 33);
   fX11RGB["grey14"] = RGB_t(36, 36, 36);
   fX11RGB["grey15"] = RGB_t(38, 38, 38);
   fX11RGB["grey16"] = RGB_t(41, 41, 41);
   fX11RGB["grey17"] = RGB_t(43, 43, 43);
   fX11RGB["grey18"] = RGB_t(46, 46, 46);
   fX11RGB["grey19"] = RGB_t(48, 48, 48);
   fX11RGB["grey2"] = RGB_t(5, 5, 5);
   fX11RGB["grey20"] = RGB_t(51, 51, 51);
   fX11RGB["grey21"] = RGB_t(54, 54, 54);
   fX11RGB["grey22"] = RGB_t(56, 56, 56);
   fX11RGB["grey23"] = RGB_t(59, 59, 59);
   fX11RGB["grey24"] = RGB_t(61, 61, 61);
   fX11RGB["grey25"] = RGB_t(64, 64, 64);
   fX11RGB["grey26"] = RGB_t(66, 66, 66);
   fX11RGB["grey27"] = RGB_t(69, 69, 69);
   fX11RGB["grey28"] = RGB_t(71, 71, 71);
   fX11RGB["grey29"] = RGB_t(74, 74, 74);
   fX11RGB["grey3"] = RGB_t(8, 8, 8);
   fX11RGB["grey30"] = RGB_t(77, 77, 77);
   fX11RGB["grey31"] = RGB_t(79, 79, 79);
   fX11RGB["grey32"] = RGB_t(82, 82, 82);
   fX11RGB["grey33"] = RGB_t(84, 84, 84);
   fX11RGB["grey34"] = RGB_t(87, 87, 87);
   fX11RGB["grey35"] = RGB_t(89, 89, 89);
   fX11RGB["grey36"] = RGB_t(92, 92, 92);
   fX11RGB["grey37"] = RGB_t(94, 94, 94);
   fX11RGB["grey38"] = RGB_t(97, 97, 97);
   fX11RGB["grey39"] = RGB_t(99, 99, 99);
   fX11RGB["grey4"] = RGB_t(10, 10, 10);
   fX11RGB["grey40"] = RGB_t(102, 102, 102);
   fX11RGB["grey41"] = RGB_t(105, 105, 105);
   fX11RGB["grey42"] = RGB_t(107, 107, 107);
   fX11RGB["grey43"] = RGB_t(110, 110, 110);
   fX11RGB["grey44"] = RGB_t(112, 112, 112);
   fX11RGB["grey45"] = RGB_t(115, 115, 115);
   fX11RGB["grey46"] = RGB_t(117, 117, 117);
   fX11RGB["grey47"] = RGB_t(120, 120, 120);
   fX11RGB["grey48"] = RGB_t(122, 122, 122);
   fX11RGB["grey49"] = RGB_t(125, 125, 125);
   fX11RGB["grey5"] = RGB_t(13, 13, 13);
   fX11RGB["grey50"] = RGB_t(127, 127, 127);
   fX11RGB["grey51"] = RGB_t(130, 130, 130);
   fX11RGB["grey52"] = RGB_t(133, 133, 133);
   fX11RGB["grey53"] = RGB_t(135, 135, 135);
   fX11RGB["grey54"] = RGB_t(138, 138, 138);
   fX11RGB["grey55"] = RGB_t(140, 140, 140);
   fX11RGB["grey56"] = RGB_t(143, 143, 143);
   fX11RGB["grey57"] = RGB_t(145, 145, 145);
   fX11RGB["grey58"] = RGB_t(148, 148, 148);
   fX11RGB["grey59"] = RGB_t(150, 150, 150);
   fX11RGB["grey6"] = RGB_t(15, 15, 15);
   fX11RGB["grey60"] = RGB_t(153, 153, 153);
   fX11RGB["grey61"] = RGB_t(156, 156, 156);
   fX11RGB["grey62"] = RGB_t(158, 158, 158);
   fX11RGB["grey63"] = RGB_t(161, 161, 161);
   fX11RGB["grey64"] = RGB_t(163, 163, 163);
   fX11RGB["grey65"] = RGB_t(166, 166, 166);
   fX11RGB["grey66"] = RGB_t(168, 168, 168);
   fX11RGB["grey67"] = RGB_t(171, 171, 171);
   fX11RGB["grey68"] = RGB_t(173, 173, 173);
   fX11RGB["grey69"] = RGB_t(176, 176, 176);
   fX11RGB["grey7"] = RGB_t(18, 18, 18);
   fX11RGB["grey70"] = RGB_t(179, 179, 179);
   fX11RGB["grey71"] = RGB_t(181, 181, 181);
   fX11RGB["grey72"] = RGB_t(184, 184, 184);
   fX11RGB["grey73"] = RGB_t(186, 186, 186);
   fX11RGB["grey74"] = RGB_t(189, 189, 189);
   fX11RGB["grey75"] = RGB_t(191, 191, 191);
   fX11RGB["grey76"] = RGB_t(194, 194, 194);
   fX11RGB["grey77"] = RGB_t(196, 196, 196);
   fX11RGB["grey78"] = RGB_t(199, 199, 199);
   fX11RGB["grey79"] = RGB_t(201, 201, 201);
   fX11RGB["grey8"] = RGB_t(20, 20, 20);
   fX11RGB["grey80"] = RGB_t(204, 204, 204);
   fX11RGB["grey81"] = RGB_t(207, 207, 207);
   fX11RGB["grey82"] = RGB_t(209, 209, 209);
   fX11RGB["grey83"] = RGB_t(212, 212, 212);
   fX11RGB["grey84"] = RGB_t(214, 214, 214);
   fX11RGB["grey85"] = RGB_t(217, 217, 217);
   fX11RGB["grey86"] = RGB_t(219, 219, 219);
   fX11RGB["grey87"] = RGB_t(222, 222, 222);
   fX11RGB["grey88"] = RGB_t(224, 224, 224);
   fX11RGB["grey89"] = RGB_t(227, 227, 227);
   fX11RGB["grey9"] = RGB_t(23, 23, 23);
   fX11RGB["grey90"] = RGB_t(229, 229, 229);
   fX11RGB["grey91"] = RGB_t(232, 232, 232);
   fX11RGB["grey92"] = RGB_t(235, 235, 235);
   fX11RGB["grey93"] = RGB_t(237, 237, 237);
   fX11RGB["grey94"] = RGB_t(240, 240, 240);
   fX11RGB["grey95"] = RGB_t(242, 242, 242);
   fX11RGB["grey96"] = RGB_t(245, 245, 245);
   fX11RGB["grey97"] = RGB_t(247, 247, 247);
   fX11RGB["grey98"] = RGB_t(250, 250, 250);
   fX11RGB["grey99"] = RGB_t(252, 252, 252);
   fX11RGB["honeydew"] = RGB_t(240, 255, 240);
   fX11RGB["honeydew1"] = RGB_t(240, 255, 240);
   fX11RGB["honeydew2"] = RGB_t(224, 238, 224);
   fX11RGB["honeydew3"] = RGB_t(193, 205, 193);
   fX11RGB["honeydew4"] = RGB_t(131, 139, 131);
   fX11RGB["hot pink"] = RGB_t(255, 105, 180);
   fX11RGB["HotPink"] = RGB_t(255, 105, 180);
   fX11RGB["HotPink1"] = RGB_t(255, 110, 180);
   fX11RGB["HotPink2"] = RGB_t(238, 106, 167);
   fX11RGB["HotPink3"] = RGB_t(205, 96, 144);
   fX11RGB["HotPink4"] = RGB_t(139, 58, 98);
   fX11RGB["indian red"] = RGB_t(205, 92, 92);
   fX11RGB["IndianRed"] = RGB_t(205, 92, 92);
   fX11RGB["IndianRed1"] = RGB_t(255, 106, 106);
   fX11RGB["IndianRed2"] = RGB_t(238, 99, 99);
   fX11RGB["IndianRed3"] = RGB_t(205, 85, 85);
   fX11RGB["IndianRed4"] = RGB_t(139, 58, 58);
   fX11RGB["ivory"] = RGB_t(255, 255, 240);
   fX11RGB["ivory1"] = RGB_t(255, 255, 240);
   fX11RGB["ivory2"] = RGB_t(238, 238, 224);
   fX11RGB["ivory3"] = RGB_t(205, 205, 193);
   fX11RGB["ivory4"] = RGB_t(139, 139, 131);
   fX11RGB["khaki"] = RGB_t(240, 230, 140);
   fX11RGB["khaki1"] = RGB_t(255, 246, 143);
   fX11RGB["khaki2"] = RGB_t(238, 230, 133);
   fX11RGB["khaki3"] = RGB_t(205, 198, 115);
   fX11RGB["khaki4"] = RGB_t(139, 134, 78);
   fX11RGB["lavender"] = RGB_t(230, 230, 250);
   fX11RGB["lavender blush"] = RGB_t(255, 240, 245);
   fX11RGB["LavenderBlush"] = RGB_t(255, 240, 245);
   fX11RGB["LavenderBlush1"] = RGB_t(255, 240, 245);
   fX11RGB["LavenderBlush2"] = RGB_t(238, 224, 229);
   fX11RGB["LavenderBlush3"] = RGB_t(205, 193, 197);
   fX11RGB["LavenderBlush4"] = RGB_t(139, 131, 134);
   fX11RGB["lawn green"] = RGB_t(124, 252, 0);
   fX11RGB["LawnGreen"] = RGB_t(124, 252, 0);
   fX11RGB["lemon chiffon"] = RGB_t(255, 250, 205);
   fX11RGB["LemonChiffon"] = RGB_t(255, 250, 205);
   fX11RGB["LemonChiffon1"] = RGB_t(255, 250, 205);
   fX11RGB["LemonChiffon2"] = RGB_t(238, 233, 191);
   fX11RGB["LemonChiffon3"] = RGB_t(205, 201, 165);
   fX11RGB["LemonChiffon4"] = RGB_t(139, 137, 112);
   fX11RGB["light blue"] = RGB_t(173, 216, 230);
   fX11RGB["light coral"] = RGB_t(240, 128, 128);
   fX11RGB["light cyan"] = RGB_t(224, 255, 255);
   fX11RGB["light goldenrod"] = RGB_t(238, 221, 130);
   fX11RGB["light goldenrod yellow"] = RGB_t(250, 250, 210);
   fX11RGB["light gray"] = RGB_t(211, 211, 211);
   fX11RGB["light green"] = RGB_t(144, 238, 144);
   fX11RGB["light grey"] = RGB_t(211, 211, 211);
   fX11RGB["light pink"] = RGB_t(255, 182, 193);
   fX11RGB["light salmon"] = RGB_t(255, 160, 122);
   fX11RGB["light sea green"] = RGB_t(32, 178, 170);
   fX11RGB["light sky blue"] = RGB_t(135, 206, 250);
   fX11RGB["light slate blue"] = RGB_t(132, 112, 255);
   fX11RGB["light slate gray"] = RGB_t(119, 136, 153);
   fX11RGB["light slate grey"] = RGB_t(119, 136, 153);
   fX11RGB["light steel blue"] = RGB_t(176, 196, 222);
   fX11RGB["light yellow"] = RGB_t(255, 255, 224);
   fX11RGB["LightBlue"] = RGB_t(173, 216, 230);
   fX11RGB["LightBlue1"] = RGB_t(191, 239, 255);
   fX11RGB["LightBlue2"] = RGB_t(178, 223, 238);
   fX11RGB["LightBlue3"] = RGB_t(154, 192, 205);
   fX11RGB["LightBlue4"] = RGB_t(104, 131, 139);
   fX11RGB["LightCoral"] = RGB_t(240, 128, 128);
   fX11RGB["LightCyan"] = RGB_t(224, 255, 255);
   fX11RGB["LightCyan1"] = RGB_t(224, 255, 255);
   fX11RGB["LightCyan2"] = RGB_t(209, 238, 238);
   fX11RGB["LightCyan3"] = RGB_t(180, 205, 205);
   fX11RGB["LightCyan4"] = RGB_t(122, 139, 139);
   fX11RGB["LightGoldenrod"] = RGB_t(238, 221, 130);
   fX11RGB["LightGoldenrod1"] = RGB_t(255, 236, 139);
   fX11RGB["LightGoldenrod2"] = RGB_t(238, 220, 130);
   fX11RGB["LightGoldenrod3"] = RGB_t(205, 190, 112);
   fX11RGB["LightGoldenrod4"] = RGB_t(139, 129, 76);
   fX11RGB["LightGoldenrodYellow"] = RGB_t(250, 250, 210);
   fX11RGB["LightGray"] = RGB_t(211, 211, 211);
   fX11RGB["LightGreen"] = RGB_t(144, 238, 144);
   fX11RGB["LightGrey"] = RGB_t(211, 211, 211);
   fX11RGB["LightPink"] = RGB_t(255, 182, 193);
   fX11RGB["LightPink1"] = RGB_t(255, 174, 185);
   fX11RGB["LightPink2"] = RGB_t(238, 162, 173);
   fX11RGB["LightPink3"] = RGB_t(205, 140, 149);
   fX11RGB["LightPink4"] = RGB_t(139, 95, 101);
   fX11RGB["LightSalmon"] = RGB_t(255, 160, 122);
   fX11RGB["LightSalmon1"] = RGB_t(255, 160, 122);
   fX11RGB["LightSalmon2"] = RGB_t(238, 149, 114);
   fX11RGB["LightSalmon3"] = RGB_t(205, 129, 98);
   fX11RGB["LightSalmon4"] = RGB_t(139, 87, 66);
   fX11RGB["LightSeaGreen"] = RGB_t(32, 178, 170);
   fX11RGB["LightSkyBlue"] = RGB_t(135, 206, 250);
   fX11RGB["LightSkyBlue1"] = RGB_t(176, 226, 255);
   fX11RGB["LightSkyBlue2"] = RGB_t(164, 211, 238);
   fX11RGB["LightSkyBlue3"] = RGB_t(141, 182, 205);
   fX11RGB["LightSkyBlue4"] = RGB_t(96, 123, 139);
   fX11RGB["LightSlateBlue"] = RGB_t(132, 112, 255);
   fX11RGB["LightSlateGray"] = RGB_t(119, 136, 153);
   fX11RGB["LightSlateGrey"] = RGB_t(119, 136, 153);
   fX11RGB["LightSteelBlue"] = RGB_t(176, 196, 222);
   fX11RGB["LightSteelBlue1"] = RGB_t(202, 225, 255);
   fX11RGB["LightSteelBlue2"] = RGB_t(188, 210, 238);
   fX11RGB["LightSteelBlue3"] = RGB_t(162, 181, 205);
   fX11RGB["LightSteelBlue4"] = RGB_t(110, 123, 139);
   fX11RGB["LightYellow"] = RGB_t(255, 255, 224);
   fX11RGB["LightYellow1"] = RGB_t(255, 255, 224);
   fX11RGB["LightYellow2"] = RGB_t(238, 238, 209);
   fX11RGB["LightYellow3"] = RGB_t(205, 205, 180);
   fX11RGB["LightYellow4"] = RGB_t(139, 139, 122);
   fX11RGB["lime green"] = RGB_t(50, 205, 50);
   fX11RGB["LimeGreen"] = RGB_t(50, 205, 50);
   fX11RGB["linen"] = RGB_t(250, 240, 230);
   fX11RGB["magenta"] = RGB_t(255, 0, 255);
   fX11RGB["magenta1"] = RGB_t(255, 0, 255);
   fX11RGB["magenta2"] = RGB_t(238, 0, 238);
   fX11RGB["magenta3"] = RGB_t(205, 0, 205);
   fX11RGB["magenta4"] = RGB_t(139, 0, 139);
   fX11RGB["maroon"] = RGB_t(176, 48, 96);
   fX11RGB["maroon1"] = RGB_t(255, 52, 179);
   fX11RGB["maroon2"] = RGB_t(238, 48, 167);
   fX11RGB["maroon3"] = RGB_t(205, 41, 144);
   fX11RGB["maroon4"] = RGB_t(139, 28, 98);
   fX11RGB["medium aquamarine"] = RGB_t(102, 205, 170);
   fX11RGB["medium blue"] = RGB_t(0, 0, 205);
   fX11RGB["medium orchid"] = RGB_t(186, 85, 211);
   fX11RGB["medium purple"] = RGB_t(147, 112, 219);
   fX11RGB["medium sea green"] = RGB_t(60, 179, 113);
   fX11RGB["medium slate blue"] = RGB_t(123, 104, 238);
   fX11RGB["medium spring green"] = RGB_t(0, 250, 154);
   fX11RGB["medium turquoise"] = RGB_t(72, 209, 204);
   fX11RGB["medium violet red"] = RGB_t(199, 21, 133);
   fX11RGB["MediumAquamarine"] = RGB_t(102, 205, 170);
   fX11RGB["MediumBlue"] = RGB_t(0, 0, 205);
   fX11RGB["MediumOrchid"] = RGB_t(186, 85, 211);
   fX11RGB["MediumOrchid1"] = RGB_t(224, 102, 255);
   fX11RGB["MediumOrchid2"] = RGB_t(209, 95, 238);
   fX11RGB["MediumOrchid3"] = RGB_t(180, 82, 205);
   fX11RGB["MediumOrchid4"] = RGB_t(122, 55, 139);
   fX11RGB["MediumPurple"] = RGB_t(147, 112, 219);
   fX11RGB["MediumPurple1"] = RGB_t(171, 130, 255);
   fX11RGB["MediumPurple2"] = RGB_t(159, 121, 238);
   fX11RGB["MediumPurple3"] = RGB_t(137, 104, 205);
   fX11RGB["MediumPurple4"] = RGB_t(93, 71, 139);
   fX11RGB["MediumSeaGreen"] = RGB_t(60, 179, 113);
   fX11RGB["MediumSlateBlue"] = RGB_t(123, 104, 238);
   fX11RGB["MediumSpringGreen"] = RGB_t(0, 250, 154);
   fX11RGB["MediumTurquoise"] = RGB_t(72, 209, 204);
   fX11RGB["MediumVioletRed"] = RGB_t(199, 21, 133);
   fX11RGB["midnight blue"] = RGB_t(25, 25, 112);
   fX11RGB["MidnightBlue"] = RGB_t(25, 25, 112);
   fX11RGB["mint cream"] = RGB_t(245, 255, 250);
   fX11RGB["MintCream"] = RGB_t(245, 255, 250);
   fX11RGB["misty rose"] = RGB_t(255, 228, 225);
   fX11RGB["MistyRose"] = RGB_t(255, 228, 225);
   fX11RGB["MistyRose1"] = RGB_t(255, 228, 225);
   fX11RGB["MistyRose2"] = RGB_t(238, 213, 210);
   fX11RGB["MistyRose3"] = RGB_t(205, 183, 181);
   fX11RGB["MistyRose4"] = RGB_t(139, 125, 123);
   fX11RGB["moccasin"] = RGB_t(255, 228, 181);
   fX11RGB["navajo white"] = RGB_t(255, 222, 173);
   fX11RGB["NavajoWhite"] = RGB_t(255, 222, 173);
   fX11RGB["NavajoWhite1"] = RGB_t(255, 222, 173);
   fX11RGB["NavajoWhite2"] = RGB_t(238, 207, 161);
   fX11RGB["NavajoWhite3"] = RGB_t(205, 179, 139);
   fX11RGB["NavajoWhite4"] = RGB_t(139, 121, 94);
   fX11RGB["navy"] = RGB_t(0, 0, 128);
   fX11RGB["navy blue"] = RGB_t(0, 0, 128);
   fX11RGB["NavyBlue"] = RGB_t(0, 0, 128);
   fX11RGB["old lace"] = RGB_t(253, 245, 230);
   fX11RGB["OldLace"] = RGB_t(253, 245, 230);
   fX11RGB["olive drab"] = RGB_t(107, 142, 35);
   fX11RGB["OliveDrab"] = RGB_t(107, 142, 35);
   fX11RGB["OliveDrab1"] = RGB_t(192, 255, 62);
   fX11RGB["OliveDrab2"] = RGB_t(179, 238, 58);
   fX11RGB["OliveDrab3"] = RGB_t(154, 205, 50);
   fX11RGB["OliveDrab4"] = RGB_t(105, 139, 34);
   fX11RGB["orange"] = RGB_t(255, 165, 0);
   fX11RGB["orange red"] = RGB_t(255, 69, 0);
   fX11RGB["orange1"] = RGB_t(255, 165, 0);
   fX11RGB["orange2"] = RGB_t(238, 154, 0);
   fX11RGB["orange3"] = RGB_t(205, 133, 0);
   fX11RGB["orange4"] = RGB_t(139, 90, 0);
   fX11RGB["OrangeRed"] = RGB_t(255, 69, 0);
   fX11RGB["OrangeRed1"] = RGB_t(255, 69, 0);
   fX11RGB["OrangeRed2"] = RGB_t(238, 64, 0);
   fX11RGB["OrangeRed3"] = RGB_t(205, 55, 0);
   fX11RGB["OrangeRed4"] = RGB_t(139, 37, 0);
   fX11RGB["orchid"] = RGB_t(218, 112, 214);
   fX11RGB["orchid1"] = RGB_t(255, 131, 250);
   fX11RGB["orchid2"] = RGB_t(238, 122, 233);
   fX11RGB["orchid3"] = RGB_t(205, 105, 201);
   fX11RGB["orchid4"] = RGB_t(139, 71, 137);
   fX11RGB["pale goldenrod"] = RGB_t(238, 232, 170);
   fX11RGB["pale green"] = RGB_t(152, 251, 152);
   fX11RGB["pale turquoise"] = RGB_t(175, 238, 238);
   fX11RGB["pale violet red"] = RGB_t(219, 112, 147);
   fX11RGB["PaleGoldenrod"] = RGB_t(238, 232, 170);
   fX11RGB["PaleGreen"] = RGB_t(152, 251, 152);
   fX11RGB["PaleGreen1"] = RGB_t(154, 255, 154);
   fX11RGB["PaleGreen2"] = RGB_t(144, 238, 144);
   fX11RGB["PaleGreen3"] = RGB_t(124, 205, 124);
   fX11RGB["PaleGreen4"] = RGB_t(84, 139, 84);
   fX11RGB["PaleTurquoise"] = RGB_t(175, 238, 238);
   fX11RGB["PaleTurquoise1"] = RGB_t(187, 255, 255);
   fX11RGB["PaleTurquoise2"] = RGB_t(174, 238, 238);
   fX11RGB["PaleTurquoise3"] = RGB_t(150, 205, 205);
   fX11RGB["PaleTurquoise4"] = RGB_t(102, 139, 139);
   fX11RGB["PaleVioletRed"] = RGB_t(219, 112, 147);
   fX11RGB["PaleVioletRed1"] = RGB_t(255, 130, 171);
   fX11RGB["PaleVioletRed2"] = RGB_t(238, 121, 159);
   fX11RGB["PaleVioletRed3"] = RGB_t(205, 104, 137);
   fX11RGB["PaleVioletRed4"] = RGB_t(139, 71, 93);
   fX11RGB["papaya whip"] = RGB_t(255, 239, 213);
   fX11RGB["PapayaWhip"] = RGB_t(255, 239, 213);
   fX11RGB["peach puff"] = RGB_t(255, 218, 185);
   fX11RGB["PeachPuff"] = RGB_t(255, 218, 185);
   fX11RGB["PeachPuff1"] = RGB_t(255, 218, 185);
   fX11RGB["PeachPuff2"] = RGB_t(238, 203, 173);
   fX11RGB["PeachPuff3"] = RGB_t(205, 175, 149);
   fX11RGB["PeachPuff4"] = RGB_t(139, 119, 101);
   fX11RGB["peru"] = RGB_t(205, 133, 63);
   fX11RGB["pink"] = RGB_t(255, 192, 203);
   fX11RGB["pink1"] = RGB_t(255, 181, 197);
   fX11RGB["pink2"] = RGB_t(238, 169, 184);
   fX11RGB["pink3"] = RGB_t(205, 145, 158);
   fX11RGB["pink4"] = RGB_t(139, 99, 108);
   fX11RGB["plum"] = RGB_t(221, 160, 221);
   fX11RGB["plum1"] = RGB_t(255, 187, 255);
   fX11RGB["plum2"] = RGB_t(238, 174, 238);
   fX11RGB["plum3"] = RGB_t(205, 150, 205);
   fX11RGB["plum4"] = RGB_t(139, 102, 139);
   fX11RGB["powder blue"] = RGB_t(176, 224, 230);
   fX11RGB["PowderBlue"] = RGB_t(176, 224, 230);
   fX11RGB["purple"] = RGB_t(160, 32, 240);
   fX11RGB["purple1"] = RGB_t(155, 48, 255);
   fX11RGB["purple2"] = RGB_t(145, 44, 238);
   fX11RGB["purple3"] = RGB_t(125, 38, 205);
   fX11RGB["purple4"] = RGB_t(85, 26, 139);
   fX11RGB["red"] = RGB_t(255, 0, 0);
   fX11RGB["red1"] = RGB_t(255, 0, 0);
   fX11RGB["red2"] = RGB_t(238, 0, 0);
   fX11RGB["red3"] = RGB_t(205, 0, 0);
   fX11RGB["red4"] = RGB_t(139, 0, 0);
   fX11RGB["rosy brown"] = RGB_t(188, 143, 143);
   fX11RGB["RosyBrown"] = RGB_t(188, 143, 143);
   fX11RGB["RosyBrown1"] = RGB_t(255, 193, 193);
   fX11RGB["RosyBrown2"] = RGB_t(238, 180, 180);
   fX11RGB["RosyBrown3"] = RGB_t(205, 155, 155);
   fX11RGB["RosyBrown4"] = RGB_t(139, 105, 105);
   fX11RGB["royal blue"] = RGB_t(65, 105, 225);
   fX11RGB["RoyalBlue"] = RGB_t(65, 105, 225);
   fX11RGB["RoyalBlue1"] = RGB_t(72, 118, 255);
   fX11RGB["RoyalBlue2"] = RGB_t(67, 110, 238);
   fX11RGB["RoyalBlue3"] = RGB_t(58, 95, 205);
   fX11RGB["RoyalBlue4"] = RGB_t(39, 64, 139);
   fX11RGB["saddle brown"] = RGB_t(139, 69, 19);
   fX11RGB["SaddleBrown"] = RGB_t(139, 69, 19);
   fX11RGB["salmon"] = RGB_t(250, 128, 114);
   fX11RGB["salmon1"] = RGB_t(255, 140, 105);
   fX11RGB["salmon2"] = RGB_t(238, 130, 98);
   fX11RGB["salmon3"] = RGB_t(205, 112, 84);
   fX11RGB["salmon4"] = RGB_t(139, 76, 57);
   fX11RGB["sandy brown"] = RGB_t(244, 164, 96);
   fX11RGB["SandyBrown"] = RGB_t(244, 164, 96);
   fX11RGB["sea green"] = RGB_t(46, 139, 87);
   fX11RGB["SeaGreen"] = RGB_t(46, 139, 87);
   fX11RGB["SeaGreen1"] = RGB_t(84, 255, 159);
   fX11RGB["SeaGreen2"] = RGB_t(78, 238, 148);
   fX11RGB["SeaGreen3"] = RGB_t(67, 205, 128);
   fX11RGB["SeaGreen4"] = RGB_t(46, 139, 87);
   fX11RGB["seashell"] = RGB_t(255, 245, 238);
   fX11RGB["seashell1"] = RGB_t(255, 245, 238);
   fX11RGB["seashell2"] = RGB_t(238, 229, 222);
   fX11RGB["seashell3"] = RGB_t(205, 197, 191);
   fX11RGB["seashell4"] = RGB_t(139, 134, 130);
   fX11RGB["sienna"] = RGB_t(160, 82, 45);
   fX11RGB["sienna1"] = RGB_t(255, 130, 71);
   fX11RGB["sienna2"] = RGB_t(238, 121, 66);
   fX11RGB["sienna3"] = RGB_t(205, 104, 57);
   fX11RGB["sienna4"] = RGB_t(139, 71, 38);
   fX11RGB["sky blue"] = RGB_t(135, 206, 235);
   fX11RGB["SkyBlue"] = RGB_t(135, 206, 235);
   fX11RGB["SkyBlue1"] = RGB_t(135, 206, 255);
   fX11RGB["SkyBlue2"] = RGB_t(126, 192, 238);
   fX11RGB["SkyBlue3"] = RGB_t(108, 166, 205);
   fX11RGB["SkyBlue4"] = RGB_t(74, 112, 139);
   fX11RGB["slate blue"] = RGB_t(106, 90, 205);
   fX11RGB["slate gray"] = RGB_t(112, 128, 144);
   fX11RGB["slate grey"] = RGB_t(112, 128, 144);
   fX11RGB["SlateBlue"] = RGB_t(106, 90, 205);
   fX11RGB["SlateBlue1"] = RGB_t(131, 111, 255);
   fX11RGB["SlateBlue2"] = RGB_t(122, 103, 238);
   fX11RGB["SlateBlue3"] = RGB_t(105, 89, 205);
   fX11RGB["SlateBlue4"] = RGB_t(71, 60, 139);
   fX11RGB["SlateGray"] = RGB_t(112, 128, 144);
   fX11RGB["SlateGray1"] = RGB_t(198, 226, 255);
   fX11RGB["SlateGray2"] = RGB_t(185, 211, 238);
   fX11RGB["SlateGray3"] = RGB_t(159, 182, 205);
   fX11RGB["SlateGray4"] = RGB_t(108, 123, 139);
   fX11RGB["SlateGrey"] = RGB_t(112, 128, 144);
   fX11RGB["snow"] = RGB_t(255, 250, 250);
   fX11RGB["snow1"] = RGB_t(255, 250, 250);
   fX11RGB["snow2"] = RGB_t(238, 233, 233);
   fX11RGB["snow3"] = RGB_t(205, 201, 201);
   fX11RGB["snow4"] = RGB_t(139, 137, 137);
   fX11RGB["spring green"] = RGB_t(0, 255, 127);
   fX11RGB["SpringGreen"] = RGB_t(0, 255, 127);
   fX11RGB["SpringGreen1"] = RGB_t(0, 255, 127);
   fX11RGB["SpringGreen2"] = RGB_t(0, 238, 118);
   fX11RGB["SpringGreen3"] = RGB_t(0, 205, 102);
   fX11RGB["SpringGreen4"] = RGB_t(0, 139, 69);
   fX11RGB["steel blue"] = RGB_t(70, 130, 180);
   fX11RGB["SteelBlue"] = RGB_t(70, 130, 180);
   fX11RGB["SteelBlue1"] = RGB_t(99, 184, 255);
   fX11RGB["SteelBlue2"] = RGB_t(92, 172, 238);
   fX11RGB["SteelBlue3"] = RGB_t(79, 148, 205);
   fX11RGB["SteelBlue4"] = RGB_t(54, 100, 139);
   fX11RGB["tan"] = RGB_t(210, 180, 140);
   fX11RGB["tan1"] = RGB_t(255, 165, 79);
   fX11RGB["tan2"] = RGB_t(238, 154, 73);
   fX11RGB["tan3"] = RGB_t(205, 133, 63);
   fX11RGB["tan4"] = RGB_t(139, 90, 43);
   fX11RGB["thistle"] = RGB_t(216, 191, 216);
   fX11RGB["thistle1"] = RGB_t(255, 225, 255);
   fX11RGB["thistle2"] = RGB_t(238, 210, 238);
   fX11RGB["thistle3"] = RGB_t(205, 181, 205);
   fX11RGB["thistle4"] = RGB_t(139, 123, 139);
   fX11RGB["tomato"] = RGB_t(255, 99, 71);
   fX11RGB["tomato1"] = RGB_t(255, 99, 71);
   fX11RGB["tomato2"] = RGB_t(238, 92, 66);
   fX11RGB["tomato3"] = RGB_t(205, 79, 57);
   fX11RGB["tomato4"] = RGB_t(139, 54, 38);
   fX11RGB["turquoise"] = RGB_t(64, 224, 208);
   fX11RGB["turquoise1"] = RGB_t(0, 245, 255);
   fX11RGB["turquoise2"] = RGB_t(0, 229, 238);
   fX11RGB["turquoise3"] = RGB_t(0, 197, 205);
   fX11RGB["turquoise4"] = RGB_t(0, 134, 139);
   fX11RGB["violet"] = RGB_t(238, 130, 238);
   fX11RGB["violet red"] = RGB_t(208, 32, 144);
   fX11RGB["VioletRed"] = RGB_t(208, 32, 144);
   fX11RGB["VioletRed1"] = RGB_t(255, 62, 150);
   fX11RGB["VioletRed2"] = RGB_t(238, 58, 140);
   fX11RGB["VioletRed3"] = RGB_t(205, 50, 120);
   fX11RGB["VioletRed4"] = RGB_t(139, 34, 82);
   fX11RGB["wheat"] = RGB_t(245, 222, 179);
   fX11RGB["wheat1"] = RGB_t(255, 231, 186);
   fX11RGB["wheat2"] = RGB_t(238, 216, 174);
   fX11RGB["wheat3"] = RGB_t(205, 186, 150);
   fX11RGB["wheat4"] = RGB_t(139, 126, 102);
   fX11RGB["white"] = RGB_t(255, 255, 255);
   fX11RGB["white smoke"] = RGB_t(245, 245, 245);
   fX11RGB["WhiteSmoke"] = RGB_t(245, 245, 245);
   fX11RGB["yellow"] = RGB_t(255, 255, 0);
   fX11RGB["yellow green"] = RGB_t(154, 205, 50);
   fX11RGB["yellow1"] = RGB_t(255, 255, 0);
   fX11RGB["yellow2"] = RGB_t(238, 238, 0);
   fX11RGB["yellow3"] = RGB_t(205, 205, 0);
   fX11RGB["yellow4"] = RGB_t(139, 139, 0);
   fX11RGB["YellowGreen"] = RGB_t(154, 205, 50);

   //These were colors names from X11's rgb.txt.
   //But X11 also understand lower case names.
   //And ROOT uses this. Convert all keys into lower case.

   rgb_map tmpMap;
   TString key;
   for (const_rgb_iterator iter = fX11RGB.begin(), endIter = fX11RGB.end(); iter != endIter; ++iter) {
      key = iter->first;
      key.ToLower();
      //Insert fails, if we have such case already - we do not care about such a fail.
      tmpMap.insert(std::make_pair(key, iter->second));
   }

   fX11RGB.swap(tmpMap);
}

//______________________________________________________________________________
void PixelToRGB(Pixel_t pixelColor, CGFloat *rgb)
{
   rgb[0] = (pixelColor >> 16 & 0xff) / 255.;
   rgb[1] = (pixelColor >> 8 & 0xff) / 255.;
   rgb[2] = (pixelColor & 0xff) / 255.;
}


//______________________________________________________________________________
void PixelToRGB(Pixel_t pixelColor, unsigned char *rgb)
{
   rgb[0] = pixelColor >> 16 & 0xff;
   rgb[1] = pixelColor >> 8 & 0xff;
   rgb[2] = pixelColor & 0xff;
}

}
}
}
