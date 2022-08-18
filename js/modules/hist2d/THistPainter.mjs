import { gStyle, BIT, settings, constants, internals, create } from '../core.mjs';
import { ColorPalette, toHex, getColor } from '../base/colors.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { EAxisBits } from '../gpad/TAxisPainter.mjs';
import { TPavePainter } from '../hist/TPavePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';


const CoordSystem = { kCARTESIAN: 1, kPOLAR: 2, kCYLINDRICAL: 3, kSPHERICAL: 4, kRAPIDITY: 5 };

function createDefaultPalette() {
   const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2/3 - t) * 6;
      return p;
   }, HLStoRGB = (h, l, s) => {
      const q = (l < 0.5) ? l * (1 + s) : l + s - l * s,
            p = 2 * l - q,
            r = hue2rgb(p, q, h + 1/3),
            g = hue2rgb(p, q, h),
            b = hue2rgb(p, q, h - 1/3);
      return '#' + toHex(r) + toHex(g) + toHex(b);
   }, minHue = 0, maxHue = 280, maxPretty = 50, palette = [];
   for (let i = 0; i < maxPretty; ++i) {
      const hue = (maxHue - (i + 1) * ((maxHue - minHue) / maxPretty)) / 360;
      palette.push(HLStoRGB(hue, 0.5, 1));
   }
   return new ColorPalette(palette);
}

function createGrayPalette() {
   let palette = [];
   for (let i = 0; i < 50; ++i) {
      const code = toHex((i+2)/60);
      palette.push('#'+code+code+code);
   }
   return new ColorPalette(palette);
}

/** @summary Create color palette
  * @private */
function getColorPalette(id) {
   id = id || settings.Palette;
   if ((id > 0) && (id < 10)) return createGrayPalette();
   if (id < 51) return createDefaultPalette();
   if (id > 113) id = 57;
   let rgb, stops = [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1];
   switch(id) {
      // Deep Sea
      case 51: rgb = [[0,9,13,17,24,32,27,25,29],[0,0,0,2,37,74,113,160,221],[28,42,59,78,98,129,154,184,221]]; break;
      // Grey Scale
      case 52: rgb = [[0,32,64,96,128,160,192,224,255],[0,32,64,96,128,160,192,224,255],[0,32,64,96,128,160,192,224,255]]; break;
      // Dark Body Radiator
      case 53: rgb = [[0,45,99,156,212,230,237,234,242],[0,0,0,45,101,168,238,238,243],[0,1,1,3,9,8,11,95,230]]; break;
      // Two-color hue (dark blue through neutral gray to bright yellow)
      case 54: rgb = [[0,22,44,68,93,124,160,192,237],[0,16,41,67,93,125,162,194,241],[97,100,99,99,93,68,44,26,74]]; break;
      // Rain Bow
      case 55: rgb = [[0,5,15,35,102,196,208,199,110],[0,48,124,192,206,226,97,16,0],[99,142,198,201,90,22,13,8,2]]; break;
      // Inverted Dark Body Radiator
      case 56: rgb = [[242,234,237,230,212,156,99,45,0],[243,238,238,168,101,45,0,0,0],[230,95,11,8,9,3,1,1,0]]; break;
      // Bird (default, keep float for backward compat)
      case 57: rgb = [[ 53.091,15.096,19.89,5.916,45.951,135.1755,208.743,253.878,248.982],[42.432,91.7745,128.5455,163.6845,183.039,191.046,186.864,200.481,250.716],[134.9715,221.442,213.8175,201.807,163.8375,118.881,89.2245,50.184,13.7445]]; break;
      // Cubehelix
      case 58: rgb = [[0,24,2,54,176,236,202,194,255],[0,29,92,129,117,120,176,236,255],[0,68,80,34,57,172,252,245,255]]; break;
      // Green Red Violet
      case 59: rgb = [[13,23,25,63,76,104,137,161,206],[95,67,37,21,0,12,35,52,79],[4,3,2,6,11,22,49,98,208]]; break;
      // Blue Red Yellow
      case 60: rgb = [[0,61,89,122,143,160,185,204,231],[0,0,0,0,14,37,72,132,235],[0,140,224,144,4,5,6,9,13]]; break;
      // Ocean
      case 61: rgb = [[14,7,2,0,5,11,55,131,229],[105,56,26,1,42,74,131,171,229],[2,21,35,60,92,113,160,185,229]]; break;
      // Color Printable On Grey
      case 62: rgb = [[0,0,0,70,148,231,235,237,244],[0,0,0,0,0,69,67,216,244],[0,102,228,231,177,124,137,20,244]]; break;
      // Alpine
      case 63: rgb = [[50,56,63,68,93,121,165,192,241],[66,81,91,96,111,128,155,189,241],[97,91,75,65,77,103,143,167,217]]; break;
      // Aquamarine
      case 64: rgb = [[145,166,167,156,131,114,101,112,132],[158,178,179,181,163,154,144,152,159],[190,199,201,192,176,169,160,166,190]]; break;
      // Army
      case 65: rgb = [[93,91,99,108,130,125,132,155,174],[126,124,128,129,131,121,119,153,173],[103,94,87,85,80,85,107,120,146]]; break;
      // Atlantic
      case 66: rgb = [[24,40,69,90,104,114,120,132,103],[29,52,94,127,150,162,159,151,101],[29,52,96,132,162,181,184,186,131]]; break;
      // Aurora
      case 67: rgb = [[46,38,61,92,113,121,132,150,191],[46,36,40,69,110,135,131,92,34],[46,80,74,70,81,105,165,211,225]]; break;
      // Avocado
      case 68: rgb = [[0,4,12,30,52,101,142,190,237],[0,40,86,121,140,172,187,213,240],[0,9,14,18,21,23,27,35,101]]; break;
      // Beach
      case 69: rgb = [[198,206,206,211,198,181,161,171,244],[103,133,150,172,178,174,163,175,244],[49,54,55,66,91,130,184,224,244]]; break;
      // Black Body
      case 70: rgb = [[243,243,240,240,241,239,186,151,129],[0,46,99,149,194,220,183,166,147],[6,8,36,91,169,235,246,240,233]]; break;
      // Blue Green Yellow
      case 71: rgb = [[22,19,19,25,35,53,88,139,210],[0,32,69,108,135,159,183,198,215],[77,96,110,116,110,100,90,78,70]]; break;
      // Brown Cyan
      case 72: rgb = [[68,116,165,182,189,180,145,111,71],[37,82,135,178,204,225,221,202,147],[16,55,105,147,196,226,232,224,178]]; break;
      // CMYK
      case 73: rgb = [[61,99,136,181,213,225,198,136,24],[149,140,96,83,132,178,190,135,22],[214,203,168,135,110,100,111,113,22]]; break;
      // Candy
      case 74: rgb = [[76,120,156,183,197,180,162,154,140],[34,35,42,69,102,137,164,188,197],[ 64,69,78,105,142,177,205,217,198]]; break;
      // Cherry
      case 75: rgb = [[37,102,157,188,196,214,223,235,251],[37,29,25,37,67,91,132,185,251],[37,32,33,45,66,98,137,187,251]]; break;
      // Coffee
      case 76: rgb = [[79,100,119,137,153,172,192,205,250],[63,79,93,103,115,135,167,196,250],[51,59,66,61,62,70,110,160,250]]; break;
      // Dark Rain Bow
      case 77: rgb = [[43,44,50,66,125,172,178,155,157],[63,63,85,101,138,163,122,51,39],[121,101,58,44,47,55,57,44,43]]; break;
      // Dark Terrain
      case 78: rgb = [[0,41,62,79,90,87,99,140,228],[0,57,81,93,85,70,71,125,228],[95,91,91,82,60,43,44,112,228]]; break;
      // Fall
      case 79: rgb = [[49,59,72,88,114,141,176,205,222],[78,72,66,57,59,75,106,142,173],[ 78,55,46,40,39,39,40,41,47]]; break;
      // Fruit Punch
      case 80: rgb = [[243,222,201,185,165,158,166,187,219],[94,108,132,135,125,96,68,51,61],[7,9,12,19,45,89,118,146,118]]; break;
      // Fuchsia
      case 81: rgb = [[19,44,74,105,137,166,194,206,220],[19,28,40,55,82,110,159,181,220],[19,42,68,96,129,157,188,203,220]]; break;
      // Grey Yellow
      case 82: rgb = [[33,44,70,99,140,165,199,211,216],[ 38,50,76,105,140,165,191,189,167],[ 55,67,97,124,140,166,163,129,52]]; break;
      // Green Brown Terrain
      case 83: rgb = [[0,33,73,124,136,152,159,171,223],[0,43,92,124,134,126,121,144,223],[0,43,68,76,73,64,72,114,223]]; break;
      // Green Pink
      case 84: rgb = [[5,18,45,124,193,223,205,128,49],[48,134,207,230,193,113,28,0,7],[6,15,41,121,193,226,208,130,49]]; break;
      // Island
      case 85: rgb = [[180,106,104,135,164,188,189,165,144],[72,126,154,184,198,207,205,190,179],[41,120,158,188,194,181,145,100,62]]; break;
      // Lake
      case 86: rgb = [[57,72,94,117,136,154,174,192,215],[0,33,68,109,140,171,192,196,209],[116,137,173,201,200,201,203,190,187]]; break;
      // Light Temperature
      case 87: rgb = [[31,71,123,160,210,222,214,199,183],[40,117,171,211,231,220,190,132,65],[234,214,228,222,210,160,105,60,34]]; break;
      // Light Terrain
      case 88: rgb = [[123,108,109,126,154,172,188,196,218],[184,138,130,133,154,175,188,196,218],[208,130,109,99,110,122,150,171,218]]; break;
      // Mint
      case 89: rgb = [[105,106,122,143,159,172,176,181,207],[252,197,194,187,174,162,153,136,125],[146,133,144,155,163,167,166,162,174]]; break;
      // Neon
      case 90: rgb = [[171,141,145,152,154,159,163,158,177],[236,143,100,63,53,55,44,31,6],[59,48,46,44,42,54,82,112,179]]; break;
      // Pastel
      case 91: rgb = [[180,190,209,223,204,228,205,152,91],[93,125,147,172,181,224,233,198,158],[236,218,160,133,114,132,162,220,218]]; break;
      // Pearl
      case 92: rgb = [[225,183,162,135,115,111,119,145,211],[205,177,166,135,124,117,117,132,172],[186,165,155,135,126,130,150,178,226]]; break;
      // Pigeon
      case 93: rgb = [[39,43,59,63,80,116,153,177,223],[39,43,59,74,91,114,139,165,223],[ 39,50,59,70,85,115,151,176,223]]; break;
      // Plum
      case 94: rgb = [[0,38,60,76,84,89,101,128,204],[0,10,15,23,35,57,83,123,199],[0,11,22,40,63,86,97,94,85]]; break;
      // Red Blue
      case 95: rgb = [[94,112,141,165,167,140,91,49,27],[27,46,88,135,166,161,135,97,58],[42,52,81,106,139,158,155,137,116]]; break;
      // Rose
      case 96: rgb = [[30,49,79,117,135,151,146,138,147],[63,60,72,90,94,94,68,46,16],[18,28,41,56,62,63,50,36,21]]; break;
      // Rust
      case 97: rgb = [[0,30,63,101,143,152,169,187,230],[0,14,28,42,58,61,67,74,91],[39,26,21,18,15,14,14,13,13]]; break;
      // Sandy Terrain
      case 98: rgb = [[149,140,164,179,182,181,131,87,61],[62,70,107,136,144,138,117,87,74],[40,38,45,49,49,49,38,32,34]]; break;
      // Sienna
      case 99: rgb = [[99,112,148,165,179,182,183,183,208],[39,40,57,79,104,127,148,161,198],[15,16,18,33,51,79,103,129,177]]; break;
      // Solar
      case 100: rgb = [[99,116,154,174,200,196,201,201,230],[0,0,8,32,58,83,119,136,173],[5,6,7,9,9,14,17,19,24]]; break;
      // South West
      case 101: rgb = [[82,106,126,141,155,163,142,107,66],[ 62,44,69,107,135,152,149,132,119],[39,25,31,60,73,68,49,72,188]]; break;
      // Starry Night
      case 102: rgb = [[18,29,44,72,116,158,184,208,221],[27,46,71,105,146,177,189,190,183],[39,55,80,108,130,133,124,100,76]]; break;
      // Sunset
      case 103: rgb = [[0,48,119,173,212,224,228,228,245],[0,13,30,47,79,127,167,205,245],[0,68,75,43,16,22,55,128,245]]; break;
      // Temperature Map
      case 104: rgb = [[34,70,129,187,225,226,216,193,179],[48,91,147,194,226,229,196,110,12],[234,212,216,224,206,110,53,40,29]]; break;
      // Thermometer
      case 105: rgb = [[30,55,103,147,174,203,188,151,105],[0,65,138,182,187,175,121,53,9],[191,202,212,208,171,140,97,57,30]]; break;
      // Valentine
      case 106: rgb = [[112,97,113,125,138,159,178,188,225],[16,17,24,37,56,81,110,136,189],[38,35,46,59,78,103,130,152,201]]; break;
      // Visible Spectrum
      case 107: rgb = [[18,72,5,23,29,201,200,98,29],[0,0,43,167,211,117,0,0,0],[51,203,177,26,10,9,8,3,0]]; break;
      // Water Melon
      case 108: rgb = [[19,42,64,88,118,147,175,187,205],[19,55,89,125,154,169,161,129,70],[19,32,47,70,100,128,145,130,75]]; break;
      // Cool
      case 109: rgb = [[33,31,42,68,86,111,141,172,227],[255,175,145,106,88,55,15,0,0],[255,205,202,203,208,205,203,206,231]]; break;
      // Copper
      case 110: rgb = [[0,25,50,79,110,145,181,201,254],[0,16,30,46,63,82,101,124,179],[0,12,21,29,39,49,61,74,103]]; break;
      // Gist Earth
      case 111: rgb = [[0,13,30,44,72,120,156,200,247],[0,36,84,117,141,153,151,158,247],[0,94,100,82,56,66,76,131,247]]; break;
      // Viridis
      case 112: rgb = [[26,51,43,33,28,35,74,144,246],[9,24,55,87,118,150,180,200,222],[30,96,112,114,112,101,72,35,0]]; break;
      // Cividis
      case 113: rgb = [[0,5,65,97,124,156,189,224,255],[32,54,77,100,123,148,175,203,234],[77,110,107,111,120,119,111,94,70]]; break;
      default: return createDefaultPalette();
   }

   const NColors = 255, Red = rgb[0], Green = rgb[1], Blue = rgb[2], palette = [];

   for (let g = 1; g < stops.length; g++) {
       // create the colors...
       const nColorsGradient = Math.round(Math.floor(NColors*stops[g]) - Math.floor(NColors*stops[g-1]));
       for (let c = 0; c < nColorsGradient; c++) {
          const col = '#' + toHex(Red[g-1] + c * (Red[g] - Red[g-1]) / nColorsGradient, 1)
                          + toHex(Green[g-1] + c * (Green[g] - Green[g-1]) / nColorsGradient, 1)
                          + toHex(Blue[g-1] + c * (Blue[g] - Blue[g-1]) / nColorsGradient, 1);
          palette.push(col);
       }
    }

    return new ColorPalette(palette);
}


/**
 * @summary Class to decode histograms draw options
 *
 * @private
 */

class THistDrawOptions {
   constructor() { this.reset(); }

   /** @summary Reset hist draw options */
   reset() {
      Object.assign(this,
            { Axis: 0, RevX: false, RevY: false, SymlogX: 0, SymlogY: 0,
              Bar: false, BarStyle: 0, Curve: false,
              Hist: true, Line: false, Fill: false,
              Error: false, ErrorKind: -1, errorX: gStyle.fErrorX,
              Mark: false, Same: false, Scat: false, ScatCoef: 1., Func: true,
              Arrow: false, Box: false, BoxStyle: 0,
              Text: false, TextAngle: 0, TextKind: "", Char: 0, Color: false, Contour: 0, Cjust: false,
              Lego: 0, Surf: 0, Off: 0, Tri: 0, Proj: 0, AxisPos: 0,
              Spec: false, Pie: false, List: false, Zscale: false, Zvert: true, PadPalette: false,
              Candle: "", Violin: "", Scaled: null, Circular: 0,
              GLBox: 0, GLColor: false, Project: "",
              System: CoordSystem.kCARTESIAN,
              AutoColor: false, NoStat: false, ForceStat: false, PadStats: false, PadTitle: false, AutoZoom: false,
              HighRes: 0, Zero: true, Palette: 0, BaseLine: false,
              Optimize: settings.OptimizeDraw, adjustFrame: false,
              Mode3D: false, x3dscale: 1, y3dscale: 1,
              Render3D: constants.Render3D.Default,
              FrontBox: true, BackBox: true,
              _pmc: false, _plc: false, _pfc: false, need_fillcol: false,
              minimum: -1111, maximum: -1111, ymin: 0, ymax: 0 });
   }

   /** @summary Decode histogram draw options */
   decode(opt, hdim, histo, pad, painter) {
      this.orginal = opt; // will be overwritten by storeDrawOpt call

      const d = new DrawOptions(opt);

      if ((hdim === 1) && (histo.fSumw2.length > 0))
         for (let n = 0; n < histo.fSumw2.length; ++n)
            if (histo.fSumw2[n] > 0) { this.Error = true; this.Hist = false; this.Zero = false; break; }

      this.ndim = hdim || 1; // keep dimensions, used for now in GED

      this.PadStats = d.check("USE_PAD_STATS");
      this.PadPalette = d.check("USE_PAD_PALETTE");
      this.PadTitle = d.check("USE_PAD_TITLE");

      if (d.check('PAL', true)) this.Palette = d.partAsInt();
      // this is zooming of histo content
      if (d.check('MINIMUM:', true)) { this.ominimum = true; this.minimum = parseFloat(d.part); }
                                else { this.ominimum = false; this.minimum = histo.fMinimum; }
      if (d.check('MAXIMUM:', true)) { this.omaximum = true; this.maximum = parseFloat(d.part); }
                                else { this.omaximum = false; this.maximum = histo.fMaximum; }

      // let configure histogram titles - only for debug purposes
      if (d.check('HTITLE:', true)) histo.fTitle = decodeURIComponent(d.part.toLowerCase());
      if (d.check('XTITLE:', true)) histo.fXaxis.fTitle = decodeURIComponent(d.part.toLowerCase());
      if (d.check('YTITLE:', true)) histo.fYaxis.fTitle = decodeURIComponent(d.part.toLowerCase());
      if (d.check('ZTITLE:', true)) histo.fZaxis.fTitle = decodeURIComponent(d.part.toLowerCase());

      if (d.check('_ADJUST_FRAME_')) this.adjustFrame = true;

      if (d.check('NOOPTIMIZE')) this.Optimize = 0;
      if (d.check('OPTIMIZE')) this.Optimize = 2;

      if (d.check('AUTOCOL')) this.AutoColor = true;
      if (d.check('AUTOZOOM')) this.AutoZoom = true;

      if (d.check('OPTSTAT',true)) this.optstat = d.partAsInt();
      if (d.check('OPTFIT',true)) this.optfit = d.partAsInt();

      if (d.check('NOSTAT')) this.NoStat = true;
      if (d.check('STAT')) this.ForceStat = true;

      if (d.check('NOTOOLTIP') && painter) painter.setTooltipAllowed(false);
      if (d.check('TOOLTIP') && painter) painter.setTooltipAllowed(true);

      if (d.check("SYMLOGX", true)) this.SymlogX = d.partAsInt(0, 3);
      if (d.check("SYMLOGY", true)) this.SymlogY = d.partAsInt(0, 3);

      if (d.check('X3DSC', true)) this.x3dscale = d.partAsInt(0, 100) / 100;
      if (d.check('Y3DSC', true)) this.y3dscale = d.partAsInt(0, 100) / 100;

      let lx = false, ly = false, check3dbox = "", check3d = (hdim == 3);
      if (d.check('LOGXY')) lx = ly = true;
      if (d.check('LOGX')) lx = true;
      if (d.check('LOGY')) ly = true;
      if (lx && pad) { pad.fLogx = 1; pad.fUxmin = 0; pad.fUxmax = 1; pad.fX1 = 0; pad.fX2 = 1; }
      if (ly && pad) { pad.fLogy = 1; pad.fUymin = 0; pad.fUymax = 1; pad.fY1 = 0; pad.fY2 = 1; }
      if (d.check('LOGZ') && pad) pad.fLogz = 1;
      if (d.check('GRIDXY') && pad) pad.fGridx = pad.fGridy = 1;
      if (d.check('GRIDX') && pad) pad.fGridx = 1;
      if (d.check('GRIDY') && pad) pad.fGridy = 1;
      if (d.check('TICKXY') && pad) pad.fTickx = pad.fTicky = 1;
      if (d.check('TICKX') && pad) pad.fTickx = 1;
      if (d.check('TICKY') && pad) pad.fTicky = 1;

      d.getColor = function() {
         this.color = this.partAsInt(1) - 1;
         if (this.color >= 0) return true;
         for (let col = 0; col < 8; ++col)
            if (getColor(col).toUpperCase() === this.part)
               { this.color = col; return true; }
         return false;
      };

      if (d.check('FILL_', true) && d.getColor())
         this.histoFillColor = d.color;

      if (d.check('LINE_', true) && d.getColor())
         this.histoLineColor = getColor(d.color);

      if (d.check('XAXIS_', true) && d.getColor())
         histo.fXaxis.fAxisColor = histo.fXaxis.fLabelColor = histo.fXaxis.fTitleColor = d.color;

      if (d.check('YAXIS_', true) && d.getColor())
         histo.fYaxis.fAxisColor = histo.fYaxis.fLabelColor = histo.fYaxis.fTitleColor = d.color;

      let has_main = painter ? !!painter.getMainPainter() : false;

      if (d.check('X+')) { this.AxisPos = 10; this.second_x = has_main; }
      if (d.check('Y+')) { this.AxisPos += 1; this.second_y = has_main; }

      if (d.check('SAMES')) { this.Same = true; this.ForceStat = true; }
      if (d.check('SAME')) { this.Same = true; this.Func = true; }

      if (d.check('SPEC')) this.Spec = true; // not used

      if (d.check('BASE0') || d.check('MIN0'))
         this.BaseLine = 0;
      else if (gStyle.fHistMinimumZero)
         this.BaseLine = 0;

      if (d.check('PIE')) this.Pie = true; // not used

      if (d.check('CANDLE', true)) this.Candle = d.part || "1";
      if (d.check('VIOLIN', true)) { this.Violin = d.part || "1"; delete this.Candle; }
      if (d.check('NOSCALED')) this.Scaled = false;
      if (d.check('SCALED')) this.Scaled = true;

      if (d.check('GLBOX',true)) this.GLBox = 10 + d.partAsInt();
      if (d.check('GLCOL')) this.GLColor = true;

      d.check('GL'); // suppress GL

      if (d.check('CIRCULAR', true) || d.check('CIRC', true)) {
         this.Circular = 11;
         if (d.part.indexOf('0') >= 0) this.Circular = 10; // black and white
         if (d.part.indexOf('1') >= 0) this.Circular = 11; // color
         if (d.part.indexOf('2') >= 0) this.Circular = 12; // color and width
      }

      this.Chord = d.check('CHORD');

      if (d.check('LEGO', true)) {
         this.Lego = 1;
         if (d.part.indexOf('0') >= 0) this.Zero = false;
         if (d.part.indexOf('1') >= 0) this.Lego = 11;
         if (d.part.indexOf('2') >= 0) this.Lego = 12;
         if (d.part.indexOf('3') >= 0) this.Lego = 13;
         if (d.part.indexOf('4') >= 0) this.Lego = 14;
         check3dbox = d.part;
         if (d.part.indexOf('Z') >= 0) this.Zscale = true;
      }

      if (d.check('R3D_', true))
         this.Render3D = constants.Render3D.fromString(d.part.toLowerCase());

      if (d.check('SURF', true)) {
         this.Surf = d.partAsInt(10, 1);
         check3dbox = d.part;
         if (d.part.indexOf('Z')>=0) this.Zscale = true;
      }

      if (d.check('TF3', true)) check3dbox = d.part;

      if (d.check('ISO', true)) check3dbox = d.part;

      if (d.check('LIST')) this.List = true; // not used

      if (d.check('CONT', true) && (hdim>1)) {
         this.Contour = 1;
         if (d.part.indexOf('Z') >= 0) this.Zscale = true;
         if (d.part.indexOf('1') >= 0) this.Contour = 11; else
         if (d.part.indexOf('2') >= 0) this.Contour = 12; else
         if (d.part.indexOf('3') >= 0) this.Contour = 13; else
         if (d.part.indexOf('4') >= 0) this.Contour = 14;
      }

      // decode bar/hbar option
      if (d.check('HBAR', true))
         this.BarStyle = 20;
      else if (d.check('BAR', true))
         this.BarStyle = 10;
      if (this.BarStyle > 0) {
         this.Hist = false;
         this.need_fillcol = true;
         this.BarStyle += d.partAsInt();
      }

      if (d.check('ARR'))
         this.Arrow = true;

      if (d.check('BOX',true))
         this.BoxStyle = 10 + d.partAsInt();

      this.Box = this.BoxStyle > 0;

      if (d.check('CJUST')) this.Cjust = true;
      if (d.check('COL')) this.Color = true;
      if (d.check('CHAR')) this.Char = 1;
      if (d.check('FUNC')) { this.Func = true; this.Hist = false; }
      if (d.check('AXIS3D')) { this.Axis = 1; this.Lego = 1; check3d = true; }
      if (d.check('AXIS')) this.Axis = 1;
      if (d.check('AXIG')) this.Axis = 2;

      if (d.check('TEXT', true)) {
         this.Text = true;
         this.Hist = false;
         this.TextAngle = Math.min(d.partAsInt(), 90);
         if (d.part.indexOf('N') >= 0) this.TextKind = "N";
         if (d.part.indexOf('E0') >= 0) this.TextLine = true;
         if (d.part.indexOf('E') >= 0) this.TextKind = "E";
      }

      if (d.check('SCAT=', true)) {
         this.Scat = true;
         this.ScatCoef = parseFloat(d.part);
         if (!Number.isFinite(this.ScatCoef) || (this.ScatCoef<=0)) this.ScatCoef = 1.;
      }

      if (d.check('SCAT')) this.Scat = true;
      if (d.check('POL')) this.System = CoordSystem.kPOLAR;
      if (d.check('CYL')) this.System = CoordSystem.kCYLINDRICAL;
      if (d.check('SPH')) this.System = CoordSystem.kSPHERICAL;
      if (d.check('PSR')) this.System = CoordSystem.kRAPIDITY;

      if (d.check('TRI', true)) {
         this.Color = false;
         this.Tri = 1;
         check3dbox = d.part;
         if (d.part.indexOf('ERR') >= 0) this.Error = true;
      }

      if (d.check('AITOFF')) this.Proj = 1;
      if (d.check('MERCATOR')) this.Proj = 2;
      if (d.check('SINUSOIDAL')) this.Proj = 3;
      if (d.check('PARABOLIC')) this.Proj = 4;
      if (this.Proj > 0) this.Contour = 14;

      if (d.check('PROJX',true)) this.Project = "X" + d.partAsInt(0,1);
      if (d.check('PROJY',true)) this.Project = "Y" + d.partAsInt(0,1);
      if (d.check('PROJ')) this.Project = "Y1";

      if (check3dbox) {
         if (check3dbox.indexOf('FB') >= 0) this.FrontBox = false;
         if (check3dbox.indexOf('BB') >= 0) this.BackBox = false;
      }

      if (check3d && d.check('FB')) this.FrontBox = false;
      if (check3d && d.check('BB')) this.BackBox = false;

      this._pfc = d.check("PFC");
      this._plc = d.check("PLC") || this.AutoColor;
      this._pmc = d.check("PMC");

      if (d.check('L')) { this.Line = true; this.Hist = false; this.Error = false; }
      if (d.check('F')) { this.Fill = true; this.need_fillcol = true; }

      if (d.check('A')) this.Axis = -1;

      if (d.check("RX") || (pad && pad.$RX)) this.RevX = true;
      if (d.check("RY") || (pad && pad.$RY)) this.RevY = true;
      const check_axis_bit = (opt, axis, bit) => {
         let flag = d.check(opt);
         if (pad && pad['$'+opt]) { flag = true; pad['$'+opt] = undefined; }
         if (flag && histo)
             if (!histo[axis].TestBit(bit))
                histo[axis].InvertBit(bit);
      };
      check_axis_bit("OTX", "fXaxis", EAxisBits.kOppositeTitle);
      check_axis_bit("OTY", "fYaxis", EAxisBits.kOppositeTitle);
      check_axis_bit("CTX", "fXaxis", EAxisBits.kCenterTitle);
      check_axis_bit("CTY", "fYaxis", EAxisBits.kCenterTitle);

      if (d.check('B1')) { this.BarStyle = 1; this.BaseLine = 0; this.Hist = false; this.need_fillcol = true; }
      if (d.check('B')) { this.BarStyle = 1; this.Hist = false; this.need_fillcol = true; }
      if (d.check('C')) { this.Curve = true; this.Hist = false; }
      if (d.check('][')) { this.Off = 1; this.Hist = true; }

      if (d.check('HIST')) { this.Hist = true; this.Func = true; this.Error = false; }

      this.Bar = (this.BarStyle > 0);

      delete this.MarkStyle; // remove mark style if any

      if (d.check('P0')) { this.Mark = true; this.Hist = false; this.Zero = true; }
      if (d.check('P')) { this.Mark = true; this.Hist = false; this.Zero = false; }
      if (d.check('HZ')) { this.Zscale = true; this.Zvert = false; }
      if (d.check('Z')) this.Zscale = true;
      if (d.check('*')) { this.Mark = true; this.MarkStyle = 3; this.Hist = false; }
      if (d.check('H')) this.Hist = true;

      if (d.check('E', true)) {
         this.Error = true;
         if (hdim == 1) {
            this.Zero = false; // do not draw empty bins with errors
            this.Hist = false;
            if (Number.isInteger(parseInt(d.part[0]))) this.ErrorKind = parseInt(d.part[0]);
            if ((this.ErrorKind === 3) || (this.ErrorKind === 4)) this.need_fillcol = true;
            if (this.ErrorKind === 0) this.Zero = true; // enable drawing of empty bins
            if (d.part.indexOf('X0')>=0) this.errorX = 0;
         }
      }
      if (d.check('9')) this.HighRes = 1;
      if (d.check('0')) this.Zero = false;
      if (this.Color && d.check('1')) this.Zero = false;

      // flag identifies 3D drawing mode for histogram
      if ((this.Lego > 0) || (hdim == 3) ||
          ((this.Surf > 0) || this.Error && (hdim == 2))) this.Mode3D = true;

      //if (this.Surf == 15)
      //   if (this.System == CoordSystem.kPOLAR || this.System == CoordSystem.kCARTESIAN)
      //      this.Surf = 13;
   }

   /** @summary Tries to reconstruct string with hist draw options */
   asString(is_main_hist, pad) {
      let res = "";
      if (this.Mode3D) {

         if (this.Lego) {
            res = "LEGO";
            if (!this.Zero) res += "0";
            if (this.Lego > 10) res += (this.Lego-10);
            if (this.Zscale) res+="Z";
         } else if (this.Surf) {
            res = "SURF" + (this.Surf-10);
            if (this.Zscale) res+="Z";
         }
         if (!this.FrontBox) res+="FB";
         if (!this.BackBox) res+="BB";

         if (this.x3dscale !== 1) res += "_X3DSC" + Math.round(this.x3dscale * 100);
         if (this.y3dscale !== 1) res += "_Y3DSC" + Math.round(this.y3dscale * 100);

      } else {
         if (this.Scat) {
            res = "SCAT";
         } else if (this.Color) {
            res = "COL";
            if (!this.Zero) res+="0";
            if (this.Zscale) res += (!this.Zvert ? "HZ" : "Z");
            if (this.Axis < 0) res+="A";
         } else if (this.Contour) {
            res = "CONT";
            if (this.Contour > 10) res += (this.Contour-10);
            if (this.Zscale) res+="Z";
         } else if (this.Bar) {
            res = (this.BaseLine === false) ? "B" : "B1";
         } else if (this.Mark) {
            res = this.Zero ? "P0" : "P"; // here invert logic with 0
         } else if (this.Error) {
            res = "E";
            if (this.ErrorKind>=0) res += this.ErrorKind;
         } else if (this.Line) {
            res += "L";
            if (this.Fill) res += "F";
         }

         if (this.Cjust) res += " CJUST";

         if (this.Text) {
            res += "TEXT";
            if (this.TextAngle) res += this.TextAngle;
            res += this.TextKind;
         }
      }

      if (is_main_hist && res) {

         if (this.ForceStat || (this.StatEnabled === true))
            res += "_STAT";
         else if (this.NoStat || (this.StatEnabled === false))
            res += "_NOSTAT";
      }

      if (is_main_hist && pad && res) {
         if (pad.fLogx) res += "_LOGX";
         if (pad.fLogy) res += "_LOGY";
         if (pad.fLogz) res += "_LOGZ";
         if (pad.fGridx) res += "_GRIDX";
         if (pad.fGridy) res += "_GRIDY";
         if (pad.fTickx) res += "_TICKX";
         if (pad.fTicky) res += "_TICKY";
      }

      return res;
   }

} // class THistDrawOptions


/**
 * @summary Handle for histogram contour
 *
 * @private
 */

class HistContour {

   constructor(zmin, zmax) {
      this.arr = [];
      this.colzmin = zmin;
      this.colzmax = zmax;
      this.below_min_indx = -1;
      this.exact_min_indx = 0;
   }

   /** @summary Returns contour levels */
   getLevels() { return this.arr; }

   /** @summary Create normal contour levels */
   createNormal(nlevels, log_scale, zminpositive) {
      if (log_scale) {
         if (this.colzmax <= 0)
            this.colzmax = 1.;
         if (this.colzmin <= 0)
            if ((zminpositive===undefined) || (zminpositive <= 0))
               this.colzmin = 0.0001*this.colzmax;
            else
               this.colzmin = ((zminpositive < 3) || (zminpositive > 100)) ? 0.3*zminpositive : 1;
         if (this.colzmin >= this.colzmax) this.colzmin = 0.0001*this.colzmax;

         let logmin = Math.log(this.colzmin)/Math.log(10),
             logmax = Math.log(this.colzmax)/Math.log(10),
             dz = (logmax-logmin)/nlevels;
         this.arr.push(this.colzmin);
         for (let level=1; level<nlevels; level++)
            this.arr.push(Math.exp((logmin + dz*level)*Math.log(10)));
         this.arr.push(this.colzmax);
         this.custom = true;
      } else {
         if ((this.colzmin === this.colzmax) && (this.colzmin !== 0)) {
            this.colzmax += 0.01*Math.abs(this.colzmax);
            this.colzmin -= 0.01*Math.abs(this.colzmin);
         }
         let dz = (this.colzmax-this.colzmin)/nlevels;
         for (let level = 0; level <= nlevels; level++)
            this.arr.push(this.colzmin + dz*level);
      }
   }

   /** @summary Create custom contour levels */
   createCustom(levels) {
      this.custom = true;
      for (let n = 0; n < levels.length; ++n)
         this.arr.push(levels[n]);

      if (this.colzmax > this.arr[this.arr.length-1])
         this.arr.push(this.colzmax);
   }

   /** @summary Configure indicies */
   configIndicies(below_min, exact_min) {
      this.below_min_indx = below_min;
      this.exact_min_indx = exact_min;
   }

   /** @summary Get index based on z value */
   getContourIndex(zc) {
      // bins less than zmin not drawn
      if (zc < this.colzmin) return this.below_min_indx;

      // if bin content exactly zmin, draw it when col0 specified or when content is positive
      if (zc === this.colzmin) return this.exact_min_indx;

      if (!this.custom)
         return Math.floor(0.01+(zc-this.colzmin)*(this.arr.length-1)/(this.colzmax-this.colzmin));

      let l = 0, r = this.arr.length-1;
      if (zc < this.arr[0]) return -1;
      if (zc >= this.arr[r]) return r;
      while (l < r-1) {
         let mid = Math.round((l+r)/2);
         if (this.arr[mid] > zc) r = mid; else l = mid;
      }
      return l;
   }

   /** @summary Get palette color */
   getPaletteColor(palette, zc) {
      let zindx = this.getContourIndex(zc);
      if (zindx < 0) return null;

      let pindx = palette.calcColorIndex(zindx, this.arr.length);

      return palette.getColor(pindx);
   }

   /** @summary Get palette index */
   getPaletteIndex(palette, zc) {
      let zindx = this.getContourIndex(zc);

      return (zindx < 0) ? null : palette.calcColorIndex(zindx, this.arr.length);
   }

} // class HistContour

/** @summary histogram status bits
  * @private */
const TH1StatusBits = {
   kNoStats       : BIT(9),  // don't draw stats box
   kUserContour   : BIT(10), // user specified contour levels
   kCanRebin      : BIT(11), // can rebin axis
   kLogX          : BIT(15), // X-axis in log scale
   kIsZoomed      : BIT(16), // bit set when zooming on Y axis
   kNoTitle       : BIT(17), // don't draw the histogram title
   kIsAverage     : BIT(18)  // Bin contents are average (used by Add)
};


/**
 * @summary Basic painter for histogram classes
 * @private
 */

class THistPainter extends ObjectPainter {

   /** @summary Constructor
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} histo - TH1 derived histogram object */
   constructor(dom, histo) {
      super(dom, histo);
      this.draw_content = true;
      this.nbinsx = 0;
      this.nbinsy = 0;
      this.accept_drops = true; // indicate that one can drop other objects like doing Draw("same")
      this.mode3d = false;
      this.hist_painter_id = internals.id_counter++; // assign unique identifier for hist painter
   }

   /** @summary Returns histogram object */
   getHisto() {
      return this.getObject();
   }

   /** @summary Returns histogram axis */
   getAxis(name) {
      let histo = this.getObject();
      switch(name) {
         case "x": return histo?.fXaxis;
         case "y": return histo?.fYaxis;
         case "z": return histo?.fZaxis;
      }
      return null;
   }

   /** @summary Returns true if TProfile */
   isTProfile() {
      return this.matchObjectType('TProfile');
   }

   /** @summary Returns true if TH1K */
   isTH1K() {
      return this.matchObjectType('TH1K');
   }

   /** @summary Returns true if TH2Poly */
   isTH2Poly() {
      return this.matchObjectType(/^TH2Poly/) || this.matchObjectType(/^TProfile2Poly/);
   }

   /** @summary Clear 3d drawings - if any */
   clear3DScene() {
      let fp = this.getFramePainter();
      if (typeof fp?.create3DScene === 'function')
         fp.create3DScene(-1);
      this.mode3d = false;
   }

   /** @summary Cleanup histogram painter */
   cleanup() {

      this.clear3DScene();

      delete this.fPalette;
      delete this.fContour;
      delete this.options;

      super.cleanup();
   }

   /** @summary Returns number of histogram dimensions */
   getDimension() {
      let histo = this.getHisto();
      if (!histo) return 0;
      if (histo._typename.match(/^TH2/)) return 2;
      if (histo._typename.match(/^TProfile2D/)) return 2;
      if (histo._typename.match(/^TH3/)) return 3;
      if (this.isTH2Poly()) return 2;
      return 1;
   }

   /** @summary Decode options string opt and fill the option structure */
   decodeOptions(opt) {
      let histo = this.getHisto(),
          hdim = this.getDimension(),
          pp = this.getPadPainter(),
          pad = pp?.getRootPad(true);

      if (!this.options)
         this.options = new THistDrawOptions;
      else
         this.options.reset();

      this.options.decode(opt || histo.fOption, hdim, histo, pad, this);

      this.storeDrawOpt(opt); // opt will be return as default draw option, used in webcanvas
   }

   /** @summary Copy draw options from other painter */
   copyOptionsFrom(src) {
      if (src === this) return;
      let o = this.options, o0 = src.options;

      o.Mode3D = o0.Mode3D;
      o.Zero = o0.Zero;
      if (o0.Mode3D) {
         o.Lego = o0.Lego;
         o.Surf = o0.Surf;
      } else {
         o.Color = o0.Color;
         o.Contour = o0.Contour;
      }
   }

   /** @summary copy draw options to all other histograms in the pad */
   copyOptionsToOthers() {
      this.forEachPainter(painter => {
         if ((painter !== this) && (typeof painter.copyOptionsFrom == 'function'))
            painter.copyOptionsFrom(this);
      }, "objects");
   }

   /** @summary Scan histogram content
     * @abstract */
   scanContent(/*when_axis_changed*/) {
      // function will be called once new histogram or
      // new histogram content is assigned
      // one should find min,max,nbins, maxcontent values
      // if when_axis_changed === true specified, content will be scanned after axis zoom changed
   }

   /** @summary Check pad ranges when drawing of frame axes will be performed */
   checkPadRange(use_pad) {
      if (this.isMainPainter())
         this.check_pad_range = use_pad ? "pad_range" : true;
   }

   /** @summary Generates automatic color for some objects painters */
   createAutoColor(numprimitives) {
      if (!numprimitives) {
         let pad = this.getPadPainter().getRootPad(true);
         numprimitives = pad && pad.fPrimitves ? pad.fPrimitves.arr.length : 5;
      }

      let indx = this._auto_color || 0;
      this._auto_color = indx+1;

      let pal = this.getHistPalette();

      if (pal) {
         if (numprimitives < 2) numprimitives = 2;
         if (indx >= numprimitives) indx = numprimitives - 1;
         let palindx = Math.round(indx * (pal.getLength()-3) / (numprimitives-1));
         let colvalue = pal.getColor(palindx);
         let colindx = this.addColor(colvalue);
         return colindx;
      }

      this._auto_color = this._auto_color % 8;
      return indx+2;
   }

   /** @summary Create necessary histogram draw attributes */
   createHistDrawAttributes() {

      let histo = this.getHisto();

      if (this.options._pfc || this.options._plc || this.options._pmc) {
         let mp = this.getMainPainter();
         if (typeof mp?.createAutoColor == 'function') {
            let icolor = mp.createAutoColor();
            if (this.options._pfc) { histo.fFillColor = icolor; delete this.fillatt; }
            if (this.options._plc) { histo.fLineColor = icolor; delete this.lineatt; }
            if (this.options._pmc) { histo.fMarkerColor = icolor; delete this.markeratt; }
            this.options._pfc = this.options._plc = this.options._pmc = false;
         }
      }

      this.createAttFill({ attr: histo, color: this.options.histoFillColor, kind: 1 });

      this.createAttLine({ attr: histo, color0: this.options.histoLineColor });
   }

   /** @summary Assign snapid for histo painter
     * @desc Used to assign snapid also for functions painters */
   setSnapId(snapid) {
      this.snapid = snapid;

      this.getPadPainter().forEachPainterInPad(objp => {
         if (objp.child_painter_id === this.hist_painter_id) {
            let obj = objp.getObject();
            if (obj && obj.fName)
               objp.snapid = snapid + "#func_" + obj.fName;
         }
       }, "objects");
   }

   /** @summary Update histogram object
     * @param obj - new histogram instance
     * @param opt - new drawing option (optional)
     * @returns {Boolean} - true if histogram was successfully updated */
   updateObject(obj, opt) {

      let histo = this.getHisto(),
          fp = this.getFramePainter(),
          pp = this.getPadPainter();

      if (obj !== histo) {

         if (!this.matchObjectType(obj)) return false;

         // simple replace of object does not help - one can have different
         // complex relations between histo and stat box, histo and colz axis,
         // one could have THStack or TMultiGraph object
         // The only that could be done is update of content

         // check only stats bit, later other settings can be monitored
         let statpainter = pp?.findPainterFor(this.findStat());
         if (histo.TestBit(TH1StatusBits.kNoStats) != obj.TestBit(TH1StatusBits.kNoStats)) {
            histo.fBits = obj.fBits;
            if (statpainter) statpainter.Enabled = !histo.TestBit(TH1StatusBits.kNoStats);
         }

         // special treatment for webcanvas - also name can be changed
         if (this.snapid !== undefined)
            histo.fName = obj.fName;

         histo.fFillColor = obj.fFillColor;
         histo.fFillStyle = obj.fFillStyle;
         histo.fLineColor = obj.fLineColor;
         histo.fLineStyle = obj.fLineStyle;
         histo.fLineWidth = obj.fLineWidth;

         histo.fEntries = obj.fEntries;
         histo.fTsumw = obj.fTsumw;
         histo.fTsumwx = obj.fTsumwx;
         histo.fTsumwx2 = obj.fTsumwx2;
         histo.fXaxis.fNbins = obj.fXaxis.fNbins;
         if (this.getDimension() > 1) {
            histo.fTsumwy = obj.fTsumwy;
            histo.fTsumwy2 = obj.fTsumwy2;
            histo.fTsumwxy = obj.fTsumwxy;
            histo.fYaxis.fNbins = obj.fYaxis.fNbins;
            if (this.getDimension() > 2) {
               histo.fTsumwz = obj.fTsumwz;
               histo.fTsumwz2 = obj.fTsumwz2;
               histo.fTsumwxz = obj.fTsumwxz;
               histo.fTsumwyz = obj.fTsumwyz;
               histo.fZaxis.fNbins = obj.fZaxis.fNbins;
            }
         }

         const copyAxisMembers = (name, tgt, src) => {
            tgt.fTitle = src.fTitle;
            tgt.fLabels = src.fLabels;
            tgt.fXmin = src.fXmin;
            tgt.fXmax = src.fXmax;
            tgt.fTimeDisplay = src.fTimeDisplay;
            tgt.fTimeFormat = src.fTimeFormat;
            tgt.fAxisColor = src.fAxisColor;
            tgt.fLabelColor = src.fLabelColor;
            tgt.fLabelFont = src.fLabelFont;
            tgt.fLabelOffset = src.fLabelOffset;
            tgt.fLabelSize = src.fLabelSize;
            tgt.fNdivisions = src.fNdivisions;
            tgt.fTickLength = src.fTickLength;
            tgt.fTitleColor = src.fTitleColor;
            tgt.fTitleFont = src.fTitleFont;
            tgt.fTitleOffset = src.fTitleOffset;
            tgt.fTitleSize = src.fTitleSize;
            if (this.snapid && (!fp || !fp.zoomChangedInteractive(name))) {
               tgt.fFirst = src.fFirst;
               tgt.fLast = src.fLast;
               tgt.fBits = src.fBits;
            }
         };

         copyAxisMembers("x", histo.fXaxis, obj.fXaxis);
         copyAxisMembers("y", histo.fYaxis, obj.fYaxis);
         copyAxisMembers("z", histo.fZaxis, obj.fZaxis);

         histo.fArray = obj.fArray;
         histo.fNcells = obj.fNcells;
         histo.fTitle = obj.fTitle;
         histo.fMinimum = obj.fMinimum;
         histo.fMaximum = obj.fMaximum;
         histo.fSumw2 = obj.fSumw2;

         if (this.isTProfile()) {
            histo.fBinEntries = obj.fBinEntries;
         } else if (this.isTH1K()) {
            histo.fNIn = obj.fNIn;
            histo.fReady = false;
         } else if (this.isTH2Poly()) {
            histo.fBins = obj.fBins;
         }

         if (this.options.Func) {

            let painters = [], newfuncs = [], pid = this.hist_painter_id;

            // find painters associated with histogram
            if (pp)
               pp.forEachPainterInPad(objp => {
                  if (objp.child_painter_id === pid)
                     painters.push(objp);
               }, "objects");

            if (obj.fFunctions)
               for (let n=0;n<obj.fFunctions.arr.length;++n) {
                  let func = obj.fFunctions.arr[n];
                  if (!func || !func._typename) continue;

                  if (!this.needDrawFunc(histo, func)) continue;

                  let funcpainter = null, func_indx = -1;

                  // try to find matching object in associated list of painters
                  for (let i=0;i<painters.length;++i)
                     if (painters[i].matchObjectType(func._typename) && (painters[i].getObject().fName === func.fName)) {
                        funcpainter = painters[i];
                        func_indx = i;
                        break;
                     }
                  // or just in generic list of painted objects
                  if (!funcpainter && func.fName)
                     funcpainter = pp ? pp.findPainterFor(null, func.fName, func._typename) : null;

                  if (funcpainter) {
                     funcpainter.updateObject(func);
                     if (func_indx >= 0) painters.splice(func_indx, 1);
                  } else {
                     newfuncs.push(func);
                  }
               }

            // stat painter has to be kept even when no object exists in the list
            if (statpainter) {
               let indx = painters.indexOf(statpainter);
               if (indx >= 0) painters.splice(indx, 1);
            }

            // remove all function which are not found in new list of primitives
            if (pp && (painters.length > 0))
               pp.cleanPrimitives(p => painters.indexOf(p) >= 0);

            // plot new objects on the same pad - will works only for simple drawings already loaded
            if (pp && (newfuncs.length > 0)) {
               let arr = [], prev_name = pp.has_canvas ? pp.selectCurrentPad(pp.this_pad_name) : undefined;
               for (let k = 0; k < newfuncs.length; ++k)
                  arr.push(pp.drawObject(this.getDom(), newfuncs[k]));
               Promise.all(arr).then(parr => {
                  for (let k = 0; k < parr.length; ++k)
                     if (parr[k]) parr[k].child_painter_id = pid;
                  pp.selectCurrentPad(prev_name);
               });
            }
         }

         let changed_opt = (histo.fOption != obj.fOption);
         histo.fOption = obj.fOption;

         if (((opt !== undefined) && (this.options.original !== opt)) || changed_opt)
            this.decodeOptions(opt || histo.fOption);
      }

      if (!this.options.ominimum) this.options.minimum = histo.fMinimum;
      if (!this.options.omaximum) this.options.maximum = histo.fMaximum;

      if (this.snapid || !fp || !fp.zoomChangedInteractive())
         this.checkPadRange();

      this.scanContent();

      this.histogram_updated = true; // indicate that object updated

      return true;
   }

   /** @summary Extract axes bins and ranges
     * @desc here functions are defined to convert index to axis value and back
     * was introduced to support non-equidistant bins */
   extractAxesProperties(ndim) {

      const assignTAxisFuncs = axis => {
         if (axis.fXbins.length >= axis.fNbins) {
            axis.regular = false;
            axis.GetBinCoord = function(bin) {
               let indx = Math.round(bin);
               if (indx <= 0) return this.fXmin;
               if (indx > this.fNbins) return this.fXmax;
               if (indx==bin) return this.fXbins[indx];
               let indx2 = (bin < indx) ? indx - 1 : indx + 1;
               return this.fXbins[indx] * Math.abs(bin-indx2) + this.fXbins[indx2] * Math.abs(bin-indx);
            };
            axis.FindBin = function(x,add) {
               for (let k = 1; k < this.fXbins.length; ++k)
                  if (x < this.fXbins[k]) return Math.floor(k-1+add);
               return this.fNbins;
            };
         } else {
            axis.regular = true;
            axis.binwidth = (axis.fXmax - axis.fXmin) / (axis.fNbins || 1);
            axis.GetBinCoord = function(bin) { return this.fXmin + bin*this.binwidth; };
            axis.FindBin = function(x,add) { return Math.floor((x - this.fXmin) / this.binwidth + add); };
         }
      };

      this.nbinsx = this.nbinsy = this.nbinsz = 0;

      const histo = this.getHisto();

      this.nbinsx = histo.fXaxis.fNbins;
      this.xmin = histo.fXaxis.fXmin;
      this.xmax = histo.fXaxis.fXmax;
      assignTAxisFuncs(histo.fXaxis);

      this.ymin = histo.fYaxis.fXmin;
      this.ymax = histo.fYaxis.fXmax;

      if (ndim > 1) {
         this.nbinsy = histo.fYaxis.fNbins;
         assignTAxisFuncs(histo.fYaxis);
      }

      if (ndim > 2) {
         this.nbinsz = histo.fZaxis.fNbins;
         this.zmin = histo.fZaxis.fXmin;
         this.zmax = histo.fZaxis.fXmax;
         assignTAxisFuncs(histo.fZaxis);
       }
   }

    /** @summary Draw axes for histogram
      * @desc axes can be drawn only for main histogram */
   drawAxes() {
      let fp = this.getFramePainter();
      if (!fp) return Promise.resolve(false);

      let histo = this.getHisto();

      // artificially add y range to display axes
      if (this.ymin === this.ymax) this.ymax += 1;

      if (!this.isMainPainter()) {
         let opts = {
            second_x: (this.options.AxisPos >= 10),
            second_y: (this.options.AxisPos % 10) == 1
         };

         if ((!opts.second_x && !opts.second_y) || fp.hasDrawnAxes(opts.second_x, opts.second_y))
            return Promise.resolve(false);

         fp.setAxes2Ranges(opts.second_x, histo.fXaxis, this.xmin, this.xmax, opts.second_y, histo.fYaxis, this.ymin, this.ymax);

         fp.createXY2(opts);

         return fp.drawAxes2(opts.second_x, opts.second_y);
      }

      if (this.options.adjustFrame) {
         let pad = this.getPadPainter().getRootPad();
         if (pad) {
            if (pad.fUxmin < pad.fUxmax) {
               fp.fX1NDC = (this.xmin - pad.fUxmin) / (pad.fUxmax - pad.fUxmin);
               fp.fX2NDC = (this.xmax - pad.fUxmin) / (pad.fUxmax - pad.fUxmin);
            }
            if (pad.fUymin < pad.fUymax) {
               fp.fY1NDC = (this.ymin - pad.fUymin) / (pad.fUymax - pad.fUymin);
               fp.fY2NDC = (this.ymax - pad.fUymin) / (pad.fUymax - pad.fUymin);
            }

            pad.fLeftMargin = fp.fX1NDC;
            pad.fRightMargin = 1 - fp.fX2NDC;
            pad.fBottomMargin = fp.fY1NDC;
            pad.fTopMargin = 1 - fp.fY2NDC;
            pad.fFrameLineColor = 0;
            pad.fFrameLineWidth = 0;
            fp.setRootPadRange(pad);

            fp.fillatt.setSolidColor('none');

            fp.redraw();
         }

         this.options.adjustFrame = false;
      }

      fp.setAxesRanges(histo.fXaxis, this.xmin, this.xmax, histo.fYaxis, this.ymin, this.ymax, histo.fZaxis, 0, 0);

      fp.createXY({ ndim: this.getDimension(),
                    check_pad_range: this.check_pad_range,
                    zoom_ymin: this.zoom_ymin,
                    zoom_ymax: this.zoom_ymax,
                    ymin_nz: this.ymin_nz,
                    swap_xy: (this.options.BarStyle >= 20),
                    reverse_x: this.options.RevX,
                    reverse_y: this.options.RevY,
                    symlog_x: this.options.SymlogX,
                    symlog_y: this.options.SymlogY,
                    Proj: this.options.Proj,
                    extra_y_space: this.options.Text && (this.options.BarStyle > 0) });
      delete this.check_pad_range;

      if (this.options.Same) return Promise.resolve(false);

      return fp.drawAxes(false, this.options.Axis < 0, (this.options.Axis < 0),
                         this.options.AxisPos, this.options.Zscale && this.options.Zvert, this.options.Zscale && !this.options.Zvert);
   }

   /** @summary Toggle histogram title drawing */
   toggleTitle(arg) {
      let histo = this.getHisto();
      if (!this.isMainPainter() || !histo)
         return false;
      if (arg==='only-check')
         return !histo.TestBit(TH1StatusBits.kNoTitle);
      histo.InvertBit(TH1StatusBits.kNoTitle);
      this.drawHistTitle();
   }

   /** @summary Draw histogram title
     * @returns {Promise} with painter */
   drawHistTitle() {

      // case when histogram drawn over other histogram (same option)
      if (!this.isMainPainter() || this.options.Same)
         return Promise.resolve(this);

      let histo = this.getHisto(), st = gStyle,
          pp = this.getPadPainter(),
          tpainter = pp?.findPainterFor(null, "title"),
          pt = tpainter?.getObject(),
          draw_title = !histo.TestBit(TH1StatusBits.kNoTitle) && (st.fOptTitle > 0);

      if (!pt && typeof pp?.findInPrimitives == "function")
         pt = pp.findInPrimitives("title", "TPaveText");

      if (pt) {
         pt.Clear();
         if (draw_title) pt.AddText(histo.fTitle);
         if (tpainter) return tpainter.redraw().then(() => this);
      } else if (draw_title && !tpainter && histo.fTitle && !this.options.PadTitle) {
         pt = create("TPaveText");
         Object.assign(pt, { fName: "title", fFillColor: st.fTitleColor, fFillStyle: st.fTitleStyle, fBorderSize: st.fTitleBorderSize,
                             fTextFont: st.fTitleFont, fTextSize: st.fTitleFontSize, fTextColor: st.fTitleTextColor, fTextAlign: st.fTitleAlign});
         pt.AddText(histo.fTitle);
         return TPavePainter.draw(this.getDom(), pt, "postitle").then(tp => {
            if (tp) tp.$secondary = true;
            return this;
         });
      }

      return Promise.resolve(this);
   }

   /** @summary Live change and update of title drawing
     * @desc Used from the GED */
   processTitleChange(arg) {

      let histo = this.getHisto(),
          pp = this.getPadPainter(),
          tpainter = pp?.findPainterFor(null, "title");

      if (!histo || !tpainter) return null;

      if (arg === "check")
         return (!this.isMainPainter() || this.options.Same) ? null : histo;

      tpainter.clearPave();
      tpainter.addText(histo.fTitle);

      tpainter.redraw();

      this.submitCanvExec('SetTitle("' + histo.fTitle + '")');
   }

   /** @summary Update statistics when web canvas is drawn */
   updateStatWebCanvas() {
      if (!this.snapid) return;

      let stat = this.findStat(),
          statpainter = this.getPadPainter()?.findPainterFor(stat);

      if (statpainter && !statpainter.snapid) statpainter.redraw();
   }

   /** @summary Find stats box
     * @desc either in list of functions or as object of correspondent painter */
   findStat() {
      if (this.options.PadStats)
         return this.getPadPainter()?.findPainterFor(null, "stats", "TPaveStats")?.getObject();

      return this.findFunction('TPaveStats', 'stats');
   }

   /** @summary Toggle stat box drawing
     * @private */
   toggleStat(arg) {

      let stat = this.findStat(), pp = this.getPadPainter(), statpainter;

      if (!arg) arg = "";

      if (!stat) {
         if (arg.indexOf('-check') > 0) return false;
         // when statbox created first time, one need to draw it
         stat = this.createStat(true);
      } else {
         statpainter = pp?.findPainterFor(stat);
      }

      if (arg == 'only-check')
         return statpainter?.Enabled || false;

      if (arg == 'fitpar-check')
         return stat?.fOptFit || false;

      if (arg == 'fitpar-toggle') {
         if (!stat) return false;
         stat.fOptFit = stat.fOptFit ? 0 : 1111; // for websocket command should be send to server
         statpainter?.redraw();
         return true;
      }

      if (statpainter) {
         statpainter.Enabled = !statpainter.Enabled;
         this.options.StatEnabled = statpainter.Enabled; // used only for interactive
         // when stat box is drawn, it always can be drawn individually while it
         // should be last for colz redrawPad is used
         statpainter.redraw();
         return statpainter.Enabled;
      }

      let prev_name = this.selectCurrentPad(this.getPadName());
      TPavePainter.draw(this.getDom(), stat).then(() => this.selectCurrentPad(prev_name));

      return true;
   }

   /** @summary Returns true if stats box fill can be ingored */
   isIgnoreStatsFill() {
      return !this.getObject() || (!this.draw_content && !this.create_stats && !this.snapid) || (this.options.Axis > 0);
   }

   /** @summary Create stat box for histogram if required */
   createStat(force) {

      let histo = this.getHisto();

      if (this.options.PadStats || !histo) return null;

      if (!force && !this.options.ForceStat) {
         if (this.options.NoStat || histo.TestBit(TH1StatusBits.kNoStats) || !settings.AutoStat) return null;

         if ((this.options.Axis > 0) || !this.isMainPainter()) return null;
      }

      let stats = this.findStat(), st = gStyle,
          optstat = this.options.optstat, optfit = this.options.optfit;

      if (optstat !== undefined) {
         if (stats) stats.fOptStat = optstat;
         delete this.options.optstat;
      } else {
         optstat = histo.$custom_stat || st.fOptStat;
      }

      if (optfit !== undefined) {
         if (stats) stats.fOptFit = optfit;
         delete this.options.optfit;
      } else {
         optfit = st.fOptFit;
      }

      if (!stats && !optstat && !optfit) return null;

      this.create_stats = true;

      if (stats) return stats;

      stats = create('TPaveStats');
      Object.assign(stats, {
         fName: 'stats', fOptStat: optstat, fOptFit: optfit,
         fX1NDC: st.fStatX - st.fStatW, fY1NDC: st.fStatY - st.fStatH, fX2NDC: st.fStatX, fY2NDC: st.fStatY,
         fTextAlign: 12
      });

      if (histo._typename.match(/^TProfile/) || histo._typename.match(/^TH2/))
         stats.fY1NDC = 0.67;

      stats.AddText(histo.fName);

      this.addFunction(stats);

      return stats;
   }

   /** @summary Find function in histogram list of functions */
   findFunction(type_name, obj_name) {
      let histo = this.getHisto(),
          funcs = histo && histo.fFunctions ? histo.fFunctions.arr : null;

      if (!funcs) return null;

      for (let i = 0; i < funcs.length; ++i) {
         if (obj_name && (funcs[i].fName !== obj_name)) continue;
         if (funcs[i]._typename === type_name) return funcs[i];
      }

      return null;
   }

   /** @summary Add function to histogram list of functions */
   addFunction(obj, asfirst) {
      let histo = this.getHisto();
      if (!histo || !obj) return;

      if (!histo.fFunctions)
         histo.fFunctions = create("TList");

      if (asfirst)
         histo.fFunctions.AddFirst(obj);
      else
         histo.fFunctions.Add(obj);
   }

   /** @summary Check if such function should be drawn directly */
   needDrawFunc(histo, func) {
      if (func._typename === 'TPaveStats')
          return !histo.TestBit(TH1StatusBits.kNoStats) && !this.options.NoStat;

       if (func._typename === 'TF1')
          return !func.TestBit(BIT(9));

       return func._typename !== 'TPaletteAxis';
   }

   /** @summary Method draws next function from the functions list
     * @returns {Promise} fulfilled when drawing is ready */
   drawNextFunction(indx) {
      let histo = this.getHisto();
      if (!this.options.Func || !histo.fFunctions || (indx >= histo.fFunctions.arr.length))
          return Promise.resolve(true);

      let func = histo.fFunctions.arr[indx],
          opt = histo.fFunctions.opt[indx],
          pp = this.getPadPainter(),
          do_draw = false,
          func_painter = pp?.findPainterFor(func);

      // no need to do something if painter for object was already done
      // object will be redraw automatically
      if (!func_painter && func)
         do_draw = this.needDrawFunc(histo, func);

      if (!do_draw)
         return this.drawNextFunction(indx+1);

      func.$histo = histo; // required to draw TF1 correctly

      let promise = TPavePainter.canDraw(func) ? TPavePainter.draw(this.getDom(), func, opt)
                                               : pp.drawObject(this.getDom(), func, opt);

      return promise.then(painter => {
         if (painter && (typeof painter == "object"))
            painter.child_painter_id = this.hist_painter_id;

         return this.drawNextFunction(indx+1);
      });
   }

   /** @summary Returns selected index for specified axis
     * @desc be aware - here indexes starts from 0 */
   getSelectIndex(axis, side, add) {
      let indx = 0,
          nbin = this['nbins'+axis] || 0,
          taxis = this.getAxis(axis);

      if (this.options.second_x && axis == "x") axis = "x2";
      if (this.options.second_y && axis == "y") axis = "y2";
      let main = this.getFramePainter(),
          min = main ? main['zoom_' + axis + 'min'] : 0,
          max = main ? main['zoom_' + axis + 'max'] : 0;

      if ((min !== max) && taxis) {
         if (side == "left")
            indx = taxis.FindBin(min, add || 0);
         else
            indx = taxis.FindBin(max, (add || 0) + 0.5);
         if (indx < 0) indx = 0; else if (indx > nbin) indx = nbin;
      } else {
         indx = (side == "left") ? 0 : nbin;
      }

      // TAxis object of histogram, where user range can be stored
      if (taxis) {
         if ((taxis.fFirst === taxis.fLast) || !taxis.TestBit(EAxisBits.kAxisRange) ||
             ((taxis.fFirst <= 1) && (taxis.fLast >= nbin))) taxis = undefined;
      }

      if (side == "left") {
         if (indx < 0) indx = 0;
         if (taxis && (taxis.fFirst > 1) && (indx < taxis.fFirst)) indx = taxis.fFirst-1;
      } else {
         if (indx > nbin) indx = nbin;
         if (taxis && (taxis.fLast <= nbin) && (indx>taxis.fLast)) indx = taxis.fLast;
      }

      return indx;
   }

   /** @summary Unzoom user range if any */
   unzoomUserRange(dox, doy, doz) {

      let res = false, histo = this.getHisto();

      if (!histo) return false;

      let unzoomTAxis = obj => {
         if (!obj || !obj.TestBit(EAxisBits.kAxisRange)) return false;
         if (obj.fFirst === obj.fLast) return false;
         if ((obj.fFirst <= 1) && (obj.fLast >= obj.fNbins)) return false;
         obj.InvertBit(EAxisBits.kAxisRange);
         return true;
      };

      let uzoomMinMax = ndim => {
         if (this.getDimension() !== ndim) return false;
         if ((this.options.minimum===-1111) && (this.options.maximum===-1111)) return false;
         if (!this.draw_content) return false; // if not drawing content, not change min/max
         this.options.minimum = this.options.maximum = -1111;
         this.scanContent(true); // to reset ymin/ymax
         return true;
      };

      if (dox && unzoomTAxis(histo.fXaxis)) res = true;
      if (doy && (unzoomTAxis(histo.fYaxis) || uzoomMinMax(1))) res = true;
      if (doz && (unzoomTAxis(histo.fZaxis) || uzoomMinMax(2))) res = true;

      return res;
   }

   /** @summary Add different interactive handlers
     * @desc only first (main) painter in list allowed to add interactive functionality
     * Most of interactivity now handled by frame
     * @returns {Promise} for ready */
   addInteractivity() {
      let ismain = this.isMainPainter(),
          second_axis = (this.options.AxisPos > 0),
          fp = ismain || second_axis ? this.getFramePainter() : null;
      return fp ? fp.addInteractivity(!ismain && second_axis) : Promise.resolve(false);
   }

   /** @summary Invoke dialog to enter and modify user range */
   changeUserRange(menu, arg) {
      let histo = this.getHisto(),
          taxis = histo ? histo['f'+arg+"axis"] : null;
      if (!taxis) return;

      let curr = "[1," + taxis.fNbins + "]";
      if (taxis.TestBit(EAxisBits.kAxisRange))
          curr = "[" + taxis.fFirst +"," + taxis.fLast +"]";

      menu.input(`Enter user range for axis ${arg} like [1,${taxis.fNbins}]`, curr).then(res => {
         if (!res) return;
         res = JSON.parse(res);
         if (!res || (res.length != 2)) return;
         let first = parseInt(res[0]), last = parseInt(res[1]);
         if (!Number.isInteger(first) || !Number.isInteger(last)) return;
         taxis.fFirst = first;
         taxis.fLast = last;

         let newflag = (taxis.fFirst < taxis.fLast) && (taxis.fFirst >= 1) && (taxis.fLast <= taxis.fNbins);

         if (newflag != taxis.TestBit(EAxisBits.kAxisRange))
            taxis.InvertBit(EAxisBits.kAxisRange);

         this.interactiveRedraw();
      });
   }

   /** @summary Start dialog to modify range of axis where histogram values are displayed */
   changeValuesRange(menu) {
      let curr;
      if ((this.options.minimum != -1111) && (this.options.maximum != -1111))
         curr = "[" + this.options.minimum + "," + this.options.maximum + "]";
      else
         curr = "[" + this.gminbin + "," + this.gmaxbin + "]";

      menu.input("Enter min/max hist values or empty string to reset", curr).then(res => {
         res = res ? JSON.parse(res) : [];

         if (!res || (typeof res != "object") || (res.length!=2) || !Number.isFinite(res[0]) || !Number.isFinite(res[1])) {
            this.options.minimum = this.options.maximum = -1111;
         } else {
            this.options.minimum = res[0];
            this.options.maximum = res[1];
          }

         this.interactiveRedraw();
       });
   }

   /** @summary Fill histogram context menu */
   fillContextMenu(menu) {

      let histo = this.getHisto(),
          fp = this.getFramePainter();
      if (!histo) return;

      menu.add("header:"+ histo._typename + "::" + histo.fName);

      if (this.options.Axis <= 0)
         menu.addchk(this.toggleStat('only-check'), "Show statbox", () => this.toggleStat());

      if (histo.fTitle && this.isMainPainter())
         menu.addchk(this.toggleTitle('only-check'), "Show title", () => this.toggleTitle());

      if (this.draw_content) {
         if (this.getDimension() == 1) {
            menu.add("User range X", () => this.changeUserRange(menu, "X"));
         } else {
            menu.add("sub:User ranges");
            menu.add("X", () => this.changeUserRange(menu, "X"));
            menu.add("Y", () => this.changeUserRange(menu, "Y"));
            if (this.getDimension() > 2)
               menu.add("Z", () => this.changeUserRange(menu, "Z"));
            else
               menu.add("Values", () => this.changeValuesRange(menu));
            menu.add("endsub:");
         }

         if (typeof this.fillHistContextMenu == 'function')
            this.fillHistContextMenu(menu);
      }

      if (this.options.Mode3D) {
         // menu for 3D drawings

         if (menu.size() > 0)
            menu.add("separator");

         let main = this.getMainPainter() || this;

         menu.addchk(main.isTooltipAllowed(), 'Show tooltips', function() {
            main.setTooltipAllowed("toggle");
         });

         menu.addchk(fp.enable_highlight, 'Highlight bins', function() {
            fp.enable_highlight = !fp.enable_highlight;
            if (!fp.enable_highlight && fp.highlightBin3D && fp.mode3d) fp.highlightBin3D(null);
         });

         if (typeof fp?.render3D == 'function') {
            menu.addchk(main.options.FrontBox, 'Front box', function() {
               main.options.FrontBox = !main.options.FrontBox;
               fp.render3D();
            });
            menu.addchk(main.options.BackBox, 'Back box', function() {
               main.options.BackBox = !main.options.BackBox;
               fp.render3D();
            });
         }

         if (this.draw_content) {
            menu.addchk(!this.options.Zero, 'Suppress zeros', function() {
               this.options.Zero = !this.options.Zero;
               this.interactiveRedraw("pad");
            });

            if ((this.options.Lego == 12) || (this.options.Lego == 14)) {
               menu.addchk(this.options.Zscale, "Z scale", () => this.toggleColz());
               if (this.fillPaletteMenu) this.fillPaletteMenu(menu);
            }
         }

         if (typeof main.control?.reset === 'function')
            menu.add('Reset camera', function() {
               main.control.reset();
            });
      }

      menu.addAttributesMenu(this);

      if (this.histogram_updated && fp.zoomChangedInteractive())
         menu.add('Let update zoom', function() {
            fp.zoomChangedInteractive('reset');
         });

      return true;
   }

   /** @summary Auto zoom into histogram non-empty range
     * @abstract */
   autoZoom() {}

   /** @summary Process click on histogram-defined buttons */
   clickButton(funcname) {
      let fp = this.getFramePainter();

      if (!this.isMainPainter() || !fp) return false;

      switch(funcname) {
         case "ToggleZoom":
            if ((fp.zoom_xmin !== fp.zoom_xmax) || (fp.zoom_ymin !== fp.zoom_ymax) || (fp.zoom_zmin !== fp.zoom_zmax)) {
               fp.unzoom();
               fp.zoomChangedInteractive('reset');
               return true;
            }
            if (this.draw_content) {
               this.autoZoom();
               return true;
            }
            break;
         case "ToggleLogX": fp.toggleAxisLog("x"); break;
         case "ToggleLogY": fp.toggleAxisLog("y"); break;
         case "ToggleLogZ": fp.toggleAxisLog("z"); break;
         case "ToggleStatBox": this.toggleStat(); return true;
      }
      return false;
   }

   /** @summary Fill pad toolbar with histogram-related functions */
   fillToolbar(not_shown) {
      let pp = this.getPadPainter();
      if (!pp) return;

      pp.addPadButton("auto_zoom", 'Toggle between unzoom and autozoom-in', 'ToggleZoom', "Ctrl *");
      pp.addPadButton("arrow_right", "Toggle log x", "ToggleLogX", "PageDown");
      pp.addPadButton("arrow_up", "Toggle log y", "ToggleLogY", "PageUp");
      if (this.getDimension() > 1)
         pp.addPadButton("arrow_diag", "Toggle log z", "ToggleLogZ");
      if (this.options.Axis <= 0)
         pp.addPadButton("statbox", 'Toggle stat box', "ToggleStatBox");
      if (!not_shown) pp.showPadButtons();
   }

   /** @summary Returns tooltip information for 3D drawings */
   get3DToolTip(indx) {
      let histo = this.getHisto(),
          tip = { bin: indx, name: histo.fName, title: histo.fTitle };
      switch (this.getDimension()) {
         case 1:
            tip.ix = indx; tip.iy = 1;
            tip.value = histo.getBinContent(tip.ix);
            tip.error = histo.getBinError(indx);
            tip.lines = this.getBinTooltips(indx-1);
            break;
         case 2:
            tip.ix = indx % (this.nbinsx + 2);
            tip.iy = (indx - tip.ix) / (this.nbinsx + 2);
            tip.value = histo.getBinContent(tip.ix, tip.iy);
            tip.error = histo.getBinError(indx);
            tip.lines = this.getBinTooltips(tip.ix-1, tip.iy-1);
            break;
         case 3:
            tip.ix = indx % (this.nbinsx+2);
            tip.iy = ((indx - tip.ix) / (this.nbinsx+2)) % (this.nbinsy+2);
            tip.iz = (indx - tip.ix - tip.iy * (this.nbinsx+2)) / (this.nbinsx+2) / (this.nbinsy+2);
            tip.value = histo.getBinContent(tip.ix, tip.iy, tip.iz);
            tip.error = histo.getBinError(indx);
            tip.lines = this.getBinTooltips(tip.ix-1, tip.iy-1, tip.iz-1);
            break;
      }

      return tip;
   }

   /** @summary Create contour object for histogram */
   createContour(nlevels, zmin, zmax, zminpositive, custom_levels) {

      let cntr = new HistContour(zmin, zmax);

      if (custom_levels) {
         cntr.createCustom(custom_levels);
      } else {
         if (nlevels < 2) nlevels = gStyle.fNumberContours;
         let pad = this.getPadPainter().getRootPad(true);
         cntr.createNormal(nlevels, pad ? pad.fLogz : 0, zminpositive);
      }

      cntr.configIndicies(this.options.Zero ? -1 : 0, (cntr.colzmin != 0) || !this.options.Zero || this.isTH2Poly() ? 0 : -1);

      let fp = this.getFramePainter();
      if ((this.getDimension() < 3) && fp) {
         fp.zmin = cntr.colzmin;
         fp.zmax = cntr.colzmax;
      }

      this.fContour = cntr;
      return cntr;
   }

   /** @summary Return contour object */
   getContour(force_recreate) {
      if (this.fContour && !force_recreate)
         return this.fContour;

      let main = this.getMainPainter(),
          fp = this.getFramePainter();

      if (main?.fContour && (main !== this)) {
         this.fContour = main.fContour;
         return this.fContour;
      }

      // if not initialized, first create contour array
      // difference from ROOT - fContour includes also last element with maxbin, which makes easier to build logz
      let histo = this.getObject(), nlevels = 0, apply_min,
          zmin = this.minbin, zmax = this.maxbin, zminpos = this.minposbin,
          custom_levels;
      if (zmin === zmax) { zmin = this.gminbin; zmax = this.gmaxbin; zminpos = this.gminposbin; }
      let gzmin = zmin, gzmax = zmax;
      if (this.options.minimum !== -1111) { zmin = this.options.minimum; gzmin = Math.min(gzmin,zmin); apply_min = true; }
      if (this.options.maximum !== -1111) { zmax = this.options.maximum; gzmax = Math.max(gzmax, zmax); apply_min = false; }
      if (zmin >= zmax) {
         if (apply_min) zmax = zmin + 1; else zmin = zmax - 1;
      }

      if (fp && (fp.zoom_zmin != fp.zoom_zmax)) {
         zmin = fp.zoom_zmin;
         zmax = fp.zoom_zmax;
      }

      if (histo.fContour && (histo.fContour.length > 1))
         if (histo.TestBit(TH1StatusBits.kUserContour))
            custom_levels = histo.fContour;
         else
            nlevels = histo.fContour.length;

      let cntr = this.createContour(nlevels, zmin, zmax, zminpos, custom_levels);

      if ((this.getDimension() < 3) && fp) {

         fp.zmin = gzmin;
         fp.zmax = gzmax;

         if ((gzmin != cntr.colzmin) || (gzmax != cntr.colzmax)) {
            fp.zoom_zmin = cntr.colzmin;
            fp.zoom_zmax = cntr.colzmax;
         } else {
            fp.zoom_zmin = fp.zoom_zmax = undefined;
         }
      }

      return cntr;
   }

   /** @summary Return levels from contour object */
   getContourLevels() {
      return this.getContour().getLevels();
   }

   /** @summary Returns color palette associated with histogram
     * @desc Create if required, checks pad and canvas for custom palette */
   getHistPalette(force) {
      if (force) this.fPalette = null;
      if (!this.fPalette && !this.options.Palette) {
         let pp = this.getPadPainter();
         if (typeof pp?.getCustomPalette == 'function')
            this.fPalette = pp.getCustomPalette();
      }
      if (!this.fPalette)
         this.fPalette = getColorPalette(this.options.Palette);
      return this.fPalette;
   }

   /** @summary Fill menu entries for palette */
   fillPaletteMenu(menu) {
      menu.addPaletteMenu(this.options.Palette || settings.Palette, arg => {
         this.options.Palette = parseInt(arg);
         this.getHistPalette(true);
         this.redraw(); // redraw histogram
      });
   }

   /** @summary draw color palette
     * @returns {Promise} when done */
   drawColorPalette(enabled, postpone_draw, can_move) {
      // only when create new palette, one could change frame size
      let mp = this.getMainPainter();
      if (mp !== this) {
         if (mp && (mp.draw_content !== false))
            return Promise.resolve(null);
      }

      let pal = this.findFunction('TPaletteAxis'),
          pp = this.getPadPainter(),
          pal_painter = pp?.findPainterFor(pal);

      if (this._can_move_colz) { can_move = true; delete this._can_move_colz; }

      if (!pal_painter && !pal) {
         pal_painter = pp?.findPainterFor(undefined, undefined, "TPaletteAxis");
         if (pal_painter) {
            pal = pal_painter.getObject();
            // add to list of functions
            this.addFunction(pal, true);
         }
      }

      if (!enabled) {
         if (pal_painter) {
            this.options.Zvert = pal_painter._palette_vertical;
            pal_painter.Enabled = false;
            pal_painter.removeG(); // completely remove drawing without need to redraw complete pad
         }

         return Promise.resolve(null);
      }

      if (!pal) {

         if (this.options.PadPalette)
            return Promise.resolve(null);

         pal = create('TPave');

         Object.assign(pal, { _typename: "TPaletteAxis", fName: "TPave", fH: null, fAxis: create('TGaxis'),
                               fX1NDC: 0.905, fX2NDC: 0.945, fY1NDC: 0.1, fY2NDC: 0.9, fInit: 1, $can_move: true } );

         if (!this.options.Zvert)
            Object.assign(pal, { fX1NDC: 0.1, fX2NDC: 0.9, fY1NDC: 0.805, fY2NDC: 0.845 });

         let zaxis = this.getHisto().fZaxis;

         Object.assign(pal.fAxis, { fTitle: zaxis.fTitle, fTitleSize: zaxis.fTitleSize, fChopt: "+",
                                    fLineColor: zaxis.fAxisColor, fLineSyle: 1, fLineWidth: 1,
                                    fTextAngle: 0, fTextSize: zaxis.fLabelSize, fTextAlign: 11,
                                    fTextColor: zaxis.fLabelColor, fTextFont: zaxis.fLabelFont });

         // place colz in the beginning, that stat box is always drawn on the top
         this.addFunction(pal, true);

         can_move = true;
      } else if (pp?._palette_vertical !== undefined) {
         this.options.Zvert = pp._palette_vertical;
      }

      let fp = this.getFramePainter();

      // keep palette width
      if (can_move && fp && pal.$can_move) {
         if (this.options.Zvert) {
            if (pal.fX1NDC > 0.5) {
               pal.fX2NDC = fp.fX2NDC + 0.005 + (pal.fX2NDC - pal.fX1NDC);
               pal.fX1NDC = fp.fX2NDC + 0.005;
            } else {
               pal.fX1NDC = fp.fX1NDC - 0.03 - (pal.fX2NDC - pal.fX1NDC);
               pal.fX2NDC = fp.fX1NDC - 0.03;
            }
            pal.fY1NDC = fp.fY1NDC;
            pal.fY2NDC = fp.fY2NDC;
         } else {
            pal.fX1NDC = fp.fX1NDC;
            pal.fX2NDC = fp.fX2NDC;
            if (pal.fY2NDC > 0.5) {
               pal.fY2NDC = fp.fY2NDC + 0.005 + (pal.fY2NDC - pal.fY1NDC);
               pal.fY1NDC = fp.fY2NDC + 0.005;
            } else {
               pal.fY1NDC = fp.fY1NDC - 0.05 - (pal.fY2NDC - pal.fY1NDC);
               pal.fY2NDC = fp.fY1NDC - 0.05;
            }
         }
      }

      //  required for z scale setting
      // TODO: use weak reference (via pad list of painters and any kind of string)
      pal.$main_painter = this;

      let arg = "", pr;
      if (postpone_draw) arg += ";postpone";
      if (can_move && !this.do_redraw_palette) arg += ";can_move";
      if (this.options.Cjust) arg+=";cjust";

      if (!pal_painter) {
         // when histogram drawn on sub pad, let draw new axis object on the same pad
         let prev = this.selectCurrentPad(this.getPadName());
         pr = TPavePainter.draw(this.getDom(), pal, arg).then(_palp => {
            pal_painter = _palp;
            this.selectCurrentPad(prev);
         });
      } else {
         pal_painter.Enabled = true;
         // real drawing will be perform at the end
         if (postpone_draw) return Promise.resolve(pal_painter);
         pr = pal_painter.drawPave(arg);
      }

      return pr.then(() => {

         // mark painter as secondary - not in list of TCanvas primitives
         pal_painter.$secondary = true;
         this.options.Zvert = pal_painter._palette_vertical;

         // make dummy redraw, palette will be updated only from histogram painter
         pal_painter.redraw = function() {};

         let need_redraw = false;

         // special code to adjust frame position to actual position of palette
         if (can_move && fp && !this.do_redraw_palette) {

            let pad = pp?.getRootPad(true);

            if (this.options.Zvert) {
               if ((pal.fX1NDC > 0.5) && (fp.fX2NDC > pal.fX1NDC)) {
                  need_redraw = true;
                  fp.fX2NDC = pal.fX1NDC - 0.01;

                  if (fp.fX1NDC > fp.fX2NDC - 0.1) fp.fX1NDC = Math.max(0, fp.fX2NDC - 0.1);
                } else if ((pal.fX2NDC < 0.5) && (fp.fX1NDC < pal.fX2NDC)) {
                  need_redraw = true;
                  fp.fX1NDC = pal.fX2NDC + 0.05;
                  if (fp.fX2NDC < fp.fX1NDC + 0.1) fp.fX2NDC = Math.min(1., fp.fX1NDC + 0.1);
                }
                if (need_redraw && pad) {
                   pad.fLeftMargin = fp.fX1NDC;
                   pad.fRightMargin = 1 - fp.fX2NDC;
                }
            } else {
               if ((pal.fY1NDC > 0.5) && (fp.fY2NDC > pal.fY1NDC)) {
                  need_redraw = true;
                  fp.fY2NDC = pal.fY1NDC - 0.01;
                  if (fp.fY1NDC > fp.fY2NDC - 0.1) fp.fY1NDC = Math.max(0, fp.fXYNDC - 0.1);
               } else if ((pal.fY2NDC < 0.5) && (fp.fY1NDC < pal.fY2NDC)) {
                  need_redraw = true;
                  fp.fY1NDC = pal.fY2NDC + 0.05;
                  if (fp.fXYNDC < fp.fY1NDC + 0.1) fp.fY2NDC = Math.min(1., fp.fY1NDC + 0.1);

               }
               if (need_redraw && pad) {
                  pad.fTopMargin = fp.fY1NDC;
                  pad.fBottomMargin = 1 - fp.fY2NDC;
               }
            }
         }

         if (!need_redraw)
            return pal_painter;

         this.do_redraw_palette = true;

         fp.redraw();

         let pr = !postpone_draw ? this.redraw() : Promise.resolve(true);
         return pr.then(() => {
             delete this.do_redraw_palette;
             return pal_painter;
         });
      });
   }

   /** @summary Toggle color z palette drawing */
   toggleColz() {
      let can_toggle = this.options.Mode3D ? (this.options.Lego === 12 || this.options.Lego === 14 || this.options.Surf === 11 || this.options.Surf === 12) :
                       this.options.Color || this.options.Contour;

      if (can_toggle) {
         this.options.Zscale = !this.options.Zscale;
         return this.drawColorPalette(this.options.Zscale, false, true);
      }
   }

   /** @summary Toggle 3D drawing mode */
   toggleMode3D() {
      this.options.Mode3D = !this.options.Mode3D;

      if (this.options.Mode3D) {
         if (!this.options.Surf && !this.options.Lego && !this.options.Error) {
            if ((this.nbinsx>=50) || (this.nbinsy>=50))
               this.options.Lego = this.options.Color ? 14 : 13;
            else
               this.options.Lego = this.options.Color ? 12 : 1;

            this.options.Zero = false; // do not show zeros by default
         }
      }

      this.copyOptionsToOthers();
      this.interactiveRedraw("pad","drawopt");
   }

   /** @summary Prepare handle for color draw */
   prepareDraw(args) {

      if (!args) args = { rounding: true, extra: 0, middle: 0 };

      if (args.extra === undefined) args.extra = 0;
      if (args.middle === undefined) args.middle = 0;

      let histo = this.getHisto(),
          xaxis = histo.fXaxis, yaxis = histo.fYaxis,
          pmain = this.getFramePainter(),
          hdim = this.getDimension(),
          i, j, x, y, binz, binarea,
          res = {
             i1: this.getSelectIndex("x", "left", 0 - args.extra),
             i2: this.getSelectIndex("x", "right", 1 + args.extra),
             j1: (hdim===1) ? 0 : this.getSelectIndex("y", "left", 0 - args.extra),
             j2: (hdim===1) ? 1 : this.getSelectIndex("y", "right", 1 + args.extra),
             min: 0, max: 0, sumz: 0, xbar1: 0, xbar2: 1, ybar1: 0, ybar2: 1
          };

      res.grx = new Float32Array(res.i2+1);
      res.gry = new Float32Array(res.j2+1);

      if (typeof histo.fBarOffset == 'number' && typeof histo.fBarWidth == 'number'
           && (histo.fBarOffset || histo.fBarWidth !== 1000)) {
             if (histo.fBarOffset <= 1000) {
                res.xbar1 = res.ybar1 = 0.001*histo.fBarOffset;
             } else if (histo.fBarOffset <= 3000) {
                res.xbar1 = 0.001*(histo.fBarOffset-2000);
             } else if (histo.fBarOffset <= 5000) {
                res.ybar1 = 0.001*(histo.fBarOffset-4000);
             }

             if (histo.fBarWidth <= 1000) {
                res.xbar2 = Math.min(1., res.xbar1 + 0.001*histo.fBarWidth);
                res.ybar2 = Math.min(1., res.ybar1 + 0.001*histo.fBarWidth);
             } else if (histo.fBarWidth <= 3000) {
                res.xbar2 = Math.min(1., res.xbar1 + 0.001*(histo.fBarWidth-2000));
                // res.ybar2 = res.ybar1 + 1;
             } else if (histo.fBarWidth <= 5000) {
                // res.xbar2 = res.xbar1 + 1;
                res.ybar2 = Math.min(1., res.ybar1 + 0.001*(histo.fBarWidth-4000));
             }
         }

      if (args.original) {
         res.original = true;
         res.origx = new Float32Array(res.i2+1);
         res.origy = new Float32Array(res.j2+1);
      }

      if (args.pixel_density) args.rounding = true;

      if (!pmain) {
         console.warn("cannot draw histogram without frame");
         return res;
      }

      let funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y);

       // calculate graphical coordinates in advance
      for (i = res.i1; i <= res.i2; ++i) {
         x = xaxis.GetBinCoord(i + args.middle);
         if (funcs.logx && (x <= 0)) { res.i1 = i+1; continue; }
         if (res.origx) res.origx[i] = x;
         res.grx[i] = funcs.grx(x);
         if (args.rounding) res.grx[i] = Math.round(res.grx[i]);

         if (args.use3d) {
            if (res.grx[i] < -pmain.size_x3d) { res.i1 = i; res.grx[i] = -pmain.size_x3d; }
            if (res.grx[i] > pmain.size_x3d) { res.i2 = i; res.grx[i] = pmain.size_x3d; }
         }
      }

      if (hdim===1) {
         res.gry[0] = funcs.gry(0);
         res.gry[1] = funcs.gry(1);
      } else
      for (j = res.j1; j <= res.j2; ++j) {
         y = yaxis.GetBinCoord(j + args.middle);
         if (funcs.logy && (y <= 0)) { res.j1 = j+1; continue; }
         if (res.origy) res.origy[j] = y;
         res.gry[j] = funcs.gry(y);
         if (args.rounding) res.gry[j] = Math.round(res.gry[j]);

         if (args.use3d) {
            if (res.gry[j] < -pmain.size_y3d) { res.j1 = j; res.gry[j] = -pmain.size_y3d; }
            if (res.gry[j] > pmain.size_y3d) { res.j2 = j; res.gry[j] = pmain.size_y3d; }
         }
      }

      //  find min/max values in selected range

      this.maxbin = this.minbin = this.minposbin = null;

      for (i = res.i1; i < res.i2; ++i) {
         for (j = res.j1; j < res.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            res.sumz += binz;
            if (args.pixel_density) {
               binarea = (res.grx[i+1]-res.grx[i])*(res.gry[j]-res.gry[j+1]);
               if (binarea <= 0) continue;
               res.max = Math.max(res.max, binz);
               if ((binz>0) && ((binz<res.min) || (res.min===0))) res.min = binz;
               binz = binz/binarea;
            }
            if (this.maxbin===null) {
               this.maxbin = this.minbin = binz;
            } else {
               this.maxbin = Math.max(this.maxbin, binz);
               this.minbin = Math.min(this.minbin, binz);
            }
            if (binz > 0)
               if ((this.minposbin===null) || (binz<this.minposbin)) this.minposbin = binz;
         }
      }

      // force recalculation of z levels
      this.fContour = null;

      return res;
   }

   /** @summary Get tip text for axis bin */
   getAxisBinTip(name, axis, bin) {
      let pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          handle = funcs[name+"_handle"],
          x1 = axis.GetBinLowEdge(bin+1);

      if (handle.kind === 'labels')
         return funcs.axisAsText(name, x1);

      let x2 = axis.GetBinLowEdge(bin+2);

      if (handle.kind === 'time')
         return funcs.axisAsText(name, (x1+x2)/2);

      return `[${funcs.axisAsText(name, x1)}, ${funcs.axisAsText(name, x2)})`;
   }

   /** @summary generic draw function for histograms
     * @private */
   static _drawHist(painter, opt) {

      return ensureTCanvas(painter).then(() => {

         painter.setAsMainPainter();

         painter.decodeOptions(opt);

         if (painter.isTH2Poly()) {
            if (painter.options.Mode3D)
               painter.options.Lego = 12; // lego always 12
            else if (!painter.options.Color)
               painter.options.Color = true; // default is color
         }

         painter.checkPadRange(!painter.options.Mode3D && (painter.options.Contour != 14));

         painter.scanContent();

         painter.createStat(); // only when required

         return painter.callDrawFunc();
      }).then(() => painter.drawNextFunction(0)).then(() => {
         if (!painter.Mode3D && painter.options.AutoZoom)
            return painter.autoZoom();
      }).then(() => {
         if (painter.options.Project && !painter.mode3d && painter.toggleProjection)
             return painter.toggleProjection(painter.options.Project);
      }).then(() => {
          painter.fillToolbar();
          return painter;
      });
   }

} // class THistPainter

export { THistPainter };
