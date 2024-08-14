import { gStyle, BIT, settings, constants, create, isObject, isFunc, isStr, getPromise,
         clTList, clTPaveText, clTPaveStats, clTPaletteAxis, clTProfile2D, clTProfile3D, clTPad,
         clTAxis, clTF1, clTF2, clTProfile, kNoZoom, clTCutG, kNoStats, kTitle } from '../core.mjs';
import { getColor, getColorPalette } from '../base/colors.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter, EAxisBits, kAxisTime, kAxisLabels } from '../base/ObjectPainter.mjs';
import { TPavePainter } from '../hist/TPavePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';


const kCARTESIAN = 1, kPOLAR = 2, kCYLINDRICAL = 3, kSPHERICAL = 4, kRAPIDITY = 5;

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
              Hist: 1, Line: false, Fill: false,
              Error: 0, ErrorKind: -1, errorX: gStyle.fErrorX,
              Mark: false, Same: false, Scat: false, ScatCoef: 1.0, Func: true, AllFunc: false,
              Arrow: false, Box: false, BoxStyle: 0,
              Text: false, TextAngle: 0, TextKind: '', Char: 0, Color: false, Contour: 0, Cjust: false,
              Lego: 0, Surf: 0, Off: 0, Tri: 0, Proj: 0, AxisPos: 0, Ortho: gStyle.fOrthoCamera,
              Spec: false, Pie: false, List: false, Zscale: false, Zvert: true, PadPalette: false,
              Candle: '', Violin: '', Scaled: null, Circular: 0,
              GLBox: 0, GLColor: false, Project: '', System: kCARTESIAN,
              AutoColor: false, NoStat: false, ForceStat: false, PadStats: false, PadTitle: false, AutoZoom: false,
              HighRes: 0, Zero: 1, Palette: 0, BaseLine: false,
              Optimize: settings.OptimizeDraw, adjustFrame: false,
              Mode3D: false, x3dscale: 1, y3dscale: 1,
              Render3D: constants.Render3D.Default,
              FrontBox: true, BackBox: true,
              need_fillcol: false,
              minimum: kNoZoom, maximum: kNoZoom, ymin: 0, ymax: 0, cutg: null,
              IgnoreMainScale: false, IgnorePalette: false });
   }

   isCartesian() { return this.System === kCARTESIAN; }

   is3d() { return this.Lego || this.Surf; }

   /** @summary Base on sumw2 values (re)set some bacis draw options, only for 1dim hist */
   decodeSumw2(histo, force) {
      const len = histo.fSumw2?.length ?? 0;
      let isany = false;
      for (let n = 0; n < len; ++n)
         if (histo.fSumw2[n] > 0) { isany = true; break; }

      if (Number.isInteger(this.Error) || force)
         this.Error = isany ? 1 : 0;

      if (Number.isInteger(this.Hist) || force)
         this.Hist = isany ? 0 : 1;

      if (Number.isInteger(this.Zero) || force)
         this.Zero = isany ? 0 : 1;
   }

   /** @summary Is palette can be used with current draw options */
   canHavePalette() {
      if (this.ndim !== 2)
         return false;

      if (this.Mode3D)
         return this.Lego === 12 || this.Lego === 14 || this.Surf === 11 || this.Surf === 12;

      if (this.Color || this.Contour || this.Hist || this.Axis)
         return true;

      return !this.Scat && !this.Box && !this.Arrow && !this.Proj && !this.Candle && !this.Violin && !this.Text;
   }

   /** @summary Decode histogram draw options */
   decode(opt, hdim, histo, pp, pad, painter) {
      this.orginal = opt; // will be overwritten by storeDrawOpt call

      this.cutg_name = '';
      if (isStr(opt) && (hdim === 2)) {
         const p1 = opt.lastIndexOf('['), p2 = opt.lastIndexOf(']');
         if ((p1 >= 0) && (p2 > p1+1)) {
            this.cutg_name = opt.slice(p1+1, p2);
            opt = opt.slice(0, p1) + opt.slice(p2+1);
            this.cutg = pp?.findInPrimitives(this.cutg_name, clTCutG);
            if (this.cutg) this.cutg.$redraw_pad = true;
         }
      }

      const d = new DrawOptions(opt);

      if (hdim === 1) this.decodeSumw2(histo, true);

      this.ndim = hdim || 1; // keep dimensions, used for now in GED

      // for old web canvas json
      // TODO: remove in version 8
      d.check('USE_PAD_TITLE');
      d.check('USE_PAD_PALETTE');
      d.check('USE_PAD_STATS');

      if (d.check('IGNORE_PALETTE'))
         this.IgnorePalette = true;

      if (d.check('PAL', true))
         this.Palette = d.partAsInt();
      // this is zooming of histo content
      if (d.check('MINIMUM:', true)) {
         this.ominimum = true;
         this.minimum = parseFloat(d.part);
      } else {
         this.ominimum = false;
         this.minimum = histo.fMinimum;
      }
      if (d.check('MAXIMUM:', true)) {
         this.omaximum = true;
         this.maximum = parseFloat(d.part);
      } else {
         this.omaximum = false;
         this.maximum = histo.fMaximum;
      }
      if (d.check('HMIN:', true)) {
         this.ohmin = true;
         this.hmin = parseFloat(d.part);
      } else {
         this.ohmin = false;
         delete this.hmin;
      }
      if (d.check('HMAX:', true)) {
         this.ohmax = true;
         this.hmax = parseFloat(d.part);
      } else {
         this.ohmax = false;
         delete this.hmax;
      }
      this.ignore_min_max = d.check('IGNORE_MIN_MAX');

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

      if (d.check('OPTSTAT', true)) this.optstat = d.partAsInt();
      if (d.check('OPTFIT', true)) this.optfit = d.partAsInt();

      if ((this.optstat || this.optstat) && histo?.TestBit(kNoStats))
         histo?.InvertBit(kNoStats);

      if (d.check('NOSTAT')) this.NoStat = true;
      if (d.check('STAT')) this.ForceStat = true;

      if (d.check('NOTOOLTIP') && painter) painter.setTooltipAllowed(false);
      if (d.check('TOOLTIP') && painter) painter.setTooltipAllowed(true);

      if (d.check('SYMLOGX', true)) this.SymlogX = d.partAsInt(0, 3);
      if (d.check('SYMLOGY', true)) this.SymlogY = d.partAsInt(0, 3);

      if (d.check('X3DSC', true)) this.x3dscale = d.partAsInt(0, 100) / 100;
      if (d.check('Y3DSC', true)) this.y3dscale = d.partAsInt(0, 100) / 100;

      if (d.check('PERSPECTIVE') || d.check('PERSP')) this.Ortho = false;
      if (d.check('ORTHO')) this.Ortho = true;

      let lx = 0, ly = 0, check3dbox = '';
      if (d.check('LOG2XY')) lx = ly = 2;
      if (d.check('LOGXY')) lx = ly = 1;
      if (d.check('LOG2X')) lx = 2;
      if (d.check('LOGX')) lx = 1;
      if (d.check('LOG2Y')) ly = 2;
      if (d.check('LOGY')) ly = 1;
      if (lx && pad) { pad.fLogx = lx; pad.fUxmin = 0; pad.fUxmax = 1; pad.fX1 = 0; pad.fX2 = 1; }
      if (ly && pad) { pad.fLogy = ly; pad.fUymin = 0; pad.fUymax = 1; pad.fY1 = 0; pad.fY2 = 1; }
      if (d.check('LOG2Z') && pad) pad.fLogz = 2;
      if (d.check('LOGZ') && pad) pad.fLogz = 1;
      if (d.check('LOGV') && pad) pad.fLogv = 1; // ficitional member, can be introduced in ROOT
      if (d.check('GRIDXY') && pad) pad.fGridx = pad.fGridy = 1;
      if (d.check('GRIDX') && pad) pad.fGridx = 1;
      if (d.check('GRIDY') && pad) pad.fGridy = 1;
      if (d.check('TICKXY') && pad) pad.fTickx = pad.fTicky = 1;
      if (d.check('TICKX') && pad) pad.fTickx = 1;
      if (d.check('TICKY') && pad) pad.fTicky = 1;
      if (d.check('TICKZ') && pad) pad.fTickz = 1;
      if (d.check('GRAYSCALE'))
         pp?.setGrayscale(true);

      if (d.check('FILL_', 'color')) {
         this.histoFillColor = d.color;
         this.histoFillPattern = 1001;
      }

      if (d.check('LINE_', 'color'))
         this.histoLineColor = getColor(d.color);

      if (d.check('XAXIS_', 'color'))
         histo.fXaxis.fAxisColor = histo.fXaxis.fLabelColor = histo.fXaxis.fTitleColor = d.color;

      if (d.check('YAXIS_', 'color'))
         histo.fYaxis.fAxisColor = histo.fYaxis.fLabelColor = histo.fYaxis.fTitleColor = d.color;

      const has_main = painter ? !!painter.getMainPainter() : false;

      if (d.check('X+')) { this.AxisPos = 10; this.second_x = has_main; }
      if (d.check('Y+')) { this.AxisPos += 1; this.second_y = has_main; }

      if (d.check('SAME0')) { this.Same = true; this.IgnoreMainScale = true; }
      if (d.check('SAMES')) { this.Same = true; this.ForceStat = true; }
      if (d.check('SAME')) { this.Same = true; this.Func = true; }

      if (d.check('SPEC')) this.Spec = true; // not used

      if (d.check('BASE0') || d.check('MIN0'))
         this.BaseLine = 0;
      else if (gStyle.fHistMinimumZero)
         this.BaseLine = 0;

      if (d.check('PIE')) this.Pie = true; // not used

      if (d.check('CANDLE', true)) this.Candle = d.part || '1';
      if (d.check('VIOLIN', true)) { this.Violin = d.part || '1'; delete this.Candle; }
      if (d.check('NOSCALED')) this.Scaled = false;
      if (d.check('SCALED')) this.Scaled = true;

      if (d.check('GLBOX', true)) this.GLBox = 10 + d.partAsInt();
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
         if (d.part.indexOf('H') >= 0) this.Zvert = false;
      }

      if (d.check('R3D_', true))
         this.Render3D = constants.Render3D.fromString(d.part.toLowerCase());

      if (d.check('POL')) this.System = kPOLAR;
      if (d.check('CYL')) this.System = kCYLINDRICAL;
      if (d.check('SPH')) this.System = kSPHERICAL;
      if (d.check('PSR')) this.System = kRAPIDITY;

      if (d.check('SURF', true)) {
         this.Surf = d.partAsInt(10, 1);
         check3dbox = d.part;
         if (d.part.indexOf('Z') >= 0) this.Zscale = true;
         if (d.part.indexOf('H') >= 0) this.Zvert = false;
      }

      if (d.check('TF3', true)) check3dbox = d.part;

      if (d.check('ISO', true)) check3dbox = d.part;

      if (d.check('LIST')) this.List = true; // not used

      if (d.check('CONT', true) && (hdim > 1)) {
         this.Contour = 1;
         if (d.part.indexOf('Z') >= 0) this.Zscale = true;
         if (d.part.indexOf('H') >= 0) this.Zvert = false;
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

      if (d.check('BOX', true)) {
         this.BoxStyle = 10;
         if (d.part.indexOf('1') >= 0) this.BoxStyle = 11; else
         if (d.part.indexOf('2') >= 0) this.BoxStyle = 12; else
         if (d.part.indexOf('3') >= 0) this.BoxStyle = 13;
         if (d.part.indexOf('Z') >= 0) this.Zscale = true;
         if (d.part.indexOf('H') >= 0) this.Zvert = false;
      }

      this.Box = this.BoxStyle > 0;

      if (d.check('CJUST')) this.Cjust = true;
      if (d.check('COL')) this.Color = true;
      if (d.check('CHAR')) this.Char = 1;
      if (d.check('ALLFUNC')) this.AllFunc = true;
      if (d.check('FUNC')) { this.Func = true; this.Hist = false; }
      if (d.check('AXIS')) this.Axis = 1;
      if (d.check('AXIG')) this.Axis = 2;

      if (d.check('TEXT', true)) {
         this.Text = true;
         this.Hist = false;
         this.TextAngle = Math.min(d.partAsInt(), 90);
         if (d.part.indexOf('N') >= 0) this.TextKind = 'N';
         if (d.part.indexOf('E0') >= 0) this.TextLine = true;
         if (d.part.indexOf('E') >= 0) this.TextKind = 'E';
      }

      if (d.check('SCAT=', true)) {
         this.Scat = true;
         this.ScatCoef = parseFloat(d.part);
         if (!Number.isFinite(this.ScatCoef) || (this.ScatCoef <= 0)) this.ScatCoef = 1.0;
      }

      if (d.check('SCAT')) this.Scat = true;

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
      if (d.check('MOLLWEIDE')) this.Proj = 5;
      if (this.Proj > 0) this.Contour = 14;

      if (d.check('PROJXY', true)) this.Project = 'XY' + d.partAsInt(0, 1);
      if (d.check('PROJX', true)) this.Project = 'X' + d.part;
      if (d.check('PROJY', true)) this.Project = 'Y' + d.part;
      if (d.check('PROJ')) this.Project = 'Y1';

      if (check3dbox) {
         if (check3dbox.indexOf('FB') >= 0) this.FrontBox = false;
         if (check3dbox.indexOf('BB') >= 0) this.BackBox = false;
      }

      if ((hdim === 3) && d.check('FB')) this.FrontBox = false;
      if ((hdim === 3) && d.check('BB')) this.BackBox = false;

      if (d.check('PFC') && !this._pfc)
         this._pfc = 2;
      if ((d.check('PLC') || this.AutoColor) && !this._plc)
         this._plc = 2;
      if (d.check('PMC') && !this._pmc)
         this._pmc = 2;

      const check_axis_bit = (opt, axis, bit) => {
         // ignore Z scale options for 2D plots
         if ((axis === 'fZaxis') && (hdim < 3) && !this.Lego && !this.Surf)
            return;
         let flag = d.check(opt);
         if (pad && pad['$'+opt]) { flag = true; pad['$'+opt] = undefined; }
         if (flag && histo) {
            if (!histo[axis].TestBit(bit))
               histo[axis].InvertBit(bit);
         }
      };

      check_axis_bit('OTX', 'fXaxis', EAxisBits.kOppositeTitle);
      check_axis_bit('OTY', 'fYaxis', EAxisBits.kOppositeTitle);
      check_axis_bit('OTZ', 'fZaxis', EAxisBits.kOppositeTitle);
      check_axis_bit('CTX', 'fXaxis', EAxisBits.kCenterTitle);
      check_axis_bit('CTY', 'fYaxis', EAxisBits.kCenterTitle);
      check_axis_bit('CTZ', 'fZaxis', EAxisBits.kCenterTitle);
      check_axis_bit('MLX', 'fXaxis', EAxisBits.kMoreLogLabels);
      check_axis_bit('MLY', 'fYaxis', EAxisBits.kMoreLogLabels);
      check_axis_bit('MLZ', 'fZaxis', EAxisBits.kMoreLogLabels);

      if (d.check('RX') || pad?.$RX) this.RevX = true;
      if (d.check('RY') || pad?.$RY) this.RevY = true;

      if (d.check('L')) { this.Line = true; this.Hist = false; }
      if (d.check('F')) { this.Fill = true; this.need_fillcol = true; }

      if (d.check('A')) this.Axis = -1;
      if (pad?.$ratio_pad === 'up') {
         if (!this.Same) this.Axis = 0; // draw both axes
         histo.fXaxis.fLabelSize = 0;
         histo.fXaxis.fTitle = '';
         histo.fYaxis.$use_top_pad = true;
      } else if (pad?.$ratio_pad === 'low') {
         if (!this.Same) this.Axis = 0; // draw both axes
         histo.fXaxis.$use_top_pad = true;
         histo.fYaxis.$use_top_pad = true;
         histo.fXaxis.fTitle = 'x';
         const fp = painter?.getCanvPainter().findPainterFor(null, 'upper_pad', clTPad)?.getFramePainter();
         if (fp) {
            painter.zoom_xmin = fp.scale_xmin;
            painter.zoom_xmax = fp.scale_xmax;
         }
      }

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
         if (hdim === 1) {
            this.Zero = false; // do not draw empty bins with errors
            if (this.Hist === 1) this.Hist = false;
            if (Number.isInteger(parseInt(d.part[0])))
               this.ErrorKind = parseInt(d.part[0]);
            if ((this.ErrorKind === 3) || (this.ErrorKind === 4)) this.need_fillcol = true;
            if (this.ErrorKind === 0) this.Zero = true; // enable drawing of empty bins
            if (d.part.indexOf('X0') >= 0) this.errorX = 0;
         }
      }
      if (d.check('9')) this.HighRes = 1;
      if (d.check('0')) this.Zero = false;
      if (this.Color && d.check('1')) this.Zero = false;

      // flag identifies 3D drawing mode for histogram
      if ((this.Lego > 0) || (hdim === 3) ||
          (((this.Surf > 0) || this.Error) && (hdim === 2))) this.Mode3D = true;

      // default draw options for TF1 is line and fill
      if (painter?.isTF1() && (hdim === 1) && (this.Hist === 1) && !this.Line && !this.Fill && !this.Curve && !this.Mark) {
         this.Hist = false;
         this.Curve = settings.FuncAsCurve;
         this.Line = !this.Curve;
         this.Fill = true;
      }

      if ((this.Surf === 15) && (this.System === kPOLAR || this.System === kCARTESIAN))
         this.Surf = 13;
   }

   /** @summary Tries to reconstruct string with hist draw options */
   asString(is_main_hist, pad) {
      let res = '', zopt = '';
      if (this.Zscale)
         zopt = this.Zvert ? 'Z' : 'HZ';
      if (this.Mode3D) {
         if (this.Lego) {
            res = 'LEGO';
            if (!this.Zero) res += '0';
            if (this.Lego > 10) res += (this.Lego-10);
            res += zopt;
         } else if (this.Surf) {
            res = 'SURF' + (this.Surf-10);
            res += zopt;
         }
         if (!this.FrontBox) res += 'FB';
         if (!this.BackBox) res += 'BB';

         if (this.x3dscale !== 1) res += `_X3DSC${Math.round(this.x3dscale * 100)}`;
         if (this.y3dscale !== 1) res += `_Y3DSC${Math.round(this.y3dscale * 100)}`;
      } else {
         if (this.Candle)
            res = 'CANDLE' + this.Candle;
          else if (this.Violin)
            res = 'VIOLIN' + this.Violin;
          else if (this.Scat)
            res = 'SCAT';
          else if (this.Color) {
            res = 'COL';
            if (!this.Zero) res += '0';
            res += zopt;
            if (this.Axis < 0) res += 'A';
         } else if (this.Contour) {
            res = 'CONT';
            if (this.Contour > 10) res += (this.Contour-10);
            res += zopt;
         } else if (this.Bar)
            res = (this.BaseLine === false) ? 'B' : 'B1';
          else if (this.Mark)
            res = this.Zero ? 'P0' : 'P'; // here invert logic with 0
          else if (this.Line) {
            res += 'L';
            if (this.Fill) res += 'F';
         } else if (this.Off)
            res = '][';

         if (this.Error) {
            res += 'E';
            if (this.ErrorKind >= 0)
               res += this.ErrorKind;
            if (this.errorX === 0)
               res += 'X0';
         }

         if (this.Cjust)
            res += ' CJUST';

         if (this.Hist === true)
            res += 'HIST';

         if (this.Text) {
            res += 'TEXT';
            if (this.TextAngle) res += this.TextAngle;
            res += this.TextKind;
         }
      }

      if (this.Palette && this.canHavePalette())
         res += `_PAL${this.Palette}`;

      if (this.is3d() && this.Ortho && is_main_hist)
         res += '_ORTHO';

      if (this.Same)
         res += this.ForceStat ? 'SAMES' : 'SAME';
      else if (is_main_hist && res) {
         if (this.ForceStat || (this.StatEnabled === true))
            res += '_STAT';
         else if (this.NoStat || (this.StatEnabled === false))
            res += '_NOSTAT';
      }

      if (is_main_hist && pad && res) {
         if (pad.fLogx === 2)
            res += '_LOG2X';
         else if (pad.fLogx)
            res += '_LOGX';
         if (pad.fLogy === 2)
            res += '_LOG2Y';
         else if (pad.fLogy)
            res += '_LOGY';
         if (pad.fLogz === 2)
            res += '_LOG2Z';
         else if (pad.fLogz)
            res += '_LOGZ';
         if (pad.fGridx) res += '_GRIDX';
         if (pad.fGridy) res += '_GRIDY';
         if (pad.fTickx) res += '_TICKX';
         if (pad.fTicky) res += '_TICKY';
         if (pad.fTickz) res += '_TICKZ';
      }

      if (this.cutg_name)
         res += ` [${this.cutg_name}]`;

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
            this.colzmax = 1.0;
         if (this.colzmin <= 0) {
            if ((zminpositive === undefined) || (zminpositive <= 0))
               this.colzmin = 0.0001*this.colzmax;
            else
               this.colzmin = ((zminpositive < 3) || (zminpositive > 100)) ? 0.3*zminpositive : 1;
         }
         if (this.colzmin >= this.colzmax)
            this.colzmin = 0.0001*this.colzmax;

         const logmin = Math.log(this.colzmin)/Math.log(10),
               logmax = Math.log(this.colzmax)/Math.log(10),
               dz = (logmax-logmin)/nlevels;
         this.arr.push(this.colzmin);
         for (let level = 1; level < nlevels; level++)
            this.arr.push(Math.exp((logmin + dz*level)*Math.log(10)));
         this.arr.push(this.colzmax);
         this.custom = true;
      } else {
         if ((this.colzmin === this.colzmax) && (this.colzmin !== 0)) {
            this.colzmax += 0.01*Math.abs(this.colzmax);
            this.colzmin -= 0.01*Math.abs(this.colzmin);
         }
         const dz = (this.colzmax-this.colzmin)/nlevels;
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
         const mid = Math.round((l+r)/2);
         if (this.arr[mid] > zc) r = mid; else l = mid;
      }
      return l;
   }

   /** @summary Get palette color */
   getPaletteColor(palette, zc) {
      const zindx = this.getContourIndex(zc);
      if (zindx < 0) return null;
      const pindx = palette.calcColorIndex(zindx, this.arr.length);
      return palette.getColor(pindx);
   }

   /** @summary Get palette index */
   getPaletteIndex(palette, zc) {
      const zindx = this.getContourIndex(zc);
      return (zindx < 0) ? null : palette.calcColorIndex(zindx, this.arr.length);
   }

} // class HistContour

/**
 * @summary Handle for updateing of secondary functions
 *
 * @private
 */

class FunctionsHandler {

   constructor(painter, pp, funcs, statpainter) {
      this.painter = painter;
      this.pp = pp;

      const painters = [], update_painters = [],
            only_draw = (statpainter === true);

      this.newfuncs = [];
      this.newopts = [];

      // find painters associated with histogram/graph/...
      if (!only_draw) {
         pp?.forEachPainterInPad(objp => {
            if (objp.isSecondary(painter) && objp._secondary_id?.match(/^func_|^indx_/))
               painters.push(objp);
         }, 'objects');
      }

      for (let n = 0; n < funcs?.arr.length; ++n) {
         const func = funcs.arr[n], fopt = funcs.opt[n];
         if (!func?._typename) continue;
         if (isFunc(painter.needDrawFunc) && !painter.needDrawFunc(painter.getObject(), func)) continue;

         let funcpainter = null, func_indx = -1;

         if (!only_draw) {
            // try to find matching object in associated list of painters
            for (let i = 0; i < painters.length; ++i) {
               if (painters[i].matchObjectType(func._typename) && (painters[i].getObjectName() === func.fName)) {
                  funcpainter = painters[i];
                  func_indx = i;
                  break;
               }
            }
            // or just in generic list of painted objects
            if (!funcpainter && func.fName)
               funcpainter = pp?.findPainterFor(null, func.fName, func._typename);
         }

         if (funcpainter) {
            funcpainter.updateObject(func, fopt);
            if (func_indx >= 0) {
               painters.splice(func_indx, 1);
               update_painters.push(funcpainter);
             }
         } else {
            // use arrays index while index is important
            this.newfuncs[n] = func;
            this.newopts[n] = fopt;
         }
      }

      // stat painter has to be kept even when no object exists in the list
      if (isObject(statpainter)) {
         const indx = painters.indexOf(statpainter);
         if (indx >= 0) painters.splice(indx, 1);
      }

      // remove all function which are not found in new list of functions
      if (painters.length > 0)
         pp?.cleanPrimitives(p => painters.indexOf(p) >= 0);

      if (update_painters.length > 0)
         this._extraPainters = update_painters;
   }

   /** @summary Draw/update functions selected before */
   drawNext(indx) {
      if (this._extraPainters) {
         const p = this._extraPainters.shift();
         if (this._extraPainters.length === 0)
            delete this._extraPainters;
         return getPromise(p.redraw()).then(() => this.drawNext(0));
      }

      if (!this.newfuncs || (indx >= this.newfuncs.length)) {
         delete this.newfuncs;
         delete this.newopts;
         return Promise.resolve(this.painter); // simplify drawing
      }

      const func = this.newfuncs[indx], fopt = this.newopts[indx];

      if (!func || this.pp?.findPainterFor(func))
         return this.drawNext(indx+1);

      const func_secondary_id = func?.fName ? `func_${func.fName}` : `indx_${indx}`;

      // Required to correctly draw multiple stats boxes
      // TODO: set reference via weak pointer
      func.$main_painter = this.painter;

      const promise = TPavePainter.canDraw(func)
            ? TPavePainter.draw(this.painter.getDom(), func, fopt)
            : this.pp.drawObject(this.painter.getDom(), func, fopt);

      return promise.then(fpainter => {
         fpainter.setSecondaryId(this.painter, func_secondary_id);

         return this.drawNext(indx+1);
      });
   }

} // class FunctionsHandler


// TH1 bits
//    kNoStats = BIT(9), don't draw stats box
const kUserContour = BIT(10), // user specified contour levels
//      kCanRebin    = BIT(11), // can rebin axis
//      kLogX        = BIT(15), // X-axis in log scale
      kIsZoomed = BIT(16), // bit set when zooming on Y axis
      kNoTitle = BIT(17); // don't draw the histogram title
//      kIsAverage   = BIT(18);  // Bin contents are average (used by Add)

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
      this.nbinsx = this.nbinsy = 0;
      this.accept_drops = true; // indicate that one can drop other objects like doing Draw('same')
      this.mode3d = false;
   }

   /** @summary Returns histogram object */
   getHisto() {
      return this.getObject();
   }

   /** @summary Returns histogram axis */
   getAxis(name) {
      const histo = this.getObject();
      switch (name) {
         case 'x': return histo?.fXaxis;
         case 'y': return histo?.fYaxis;
         case 'z': return histo?.fZaxis;
      }
      return null;
   }

   /** @summary Returns true if TProfile */
   isTProfile() {
      return this.matchObjectType(clTProfile);
   }

   /** @summary Returns true if histogram drawn instead of TF1/TF2 object */
   isTF1() { return false; }

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
      const fp = this.getFramePainter();
      if (isFunc(fp?.create3DScene))
         fp.create3DScene(-1);
      this.mode3d = false;
   }

   /** @summary Cleanup histogram painter */
   cleanup() {
      this.clear3DScene();

      delete this._color_palette;
      delete this.fContour;
      delete this.options;

      super.cleanup();
   }

   /** @summary Returns number of histogram dimensions */
   getDimension() {
      const histo = this.getHisto();
      if (!histo) return 0;
      if (histo._typename.match(/^TH2/)) return 2;
      if (histo._typename === clTProfile2D) return 2;
      if (histo._typename.match(/^TH3/)) return 3;
      if (histo._typename === clTProfile3D) return 3;
      if (this.isTH2Poly()) return 2;
      return 1;
   }

   /** @summary Decode options string opt and fill the option structure */
   decodeOptions(opt) {
      const histo = this.getHisto(),
            hdim = this.getDimension(),
            pp = this.getPadPainter(),
            pad = pp?.getRootPad(true);

      if (!this.options)
         this.options = new THistDrawOptions();
      else
         this.options.reset();

      // when changing draw option, reset attributes usage
      this.lineatt?.setUsed(false);
      this.fillatt?.setUsed(false);
      this.markeratt?.setUsed(false);

      this.options.decode(opt || histo.fOption, hdim, histo, pp, pad, this);

      this.storeDrawOpt(opt); // opt will be return as default draw option, used in webcanvas
   }

   /** @summary Copy draw options from other painter */
   copyOptionsFrom(src) {
      if (src === this) return;
      const o = this.options, o0 = src.options;

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
         if ((painter !== this) && isFunc(painter.copyOptionsFrom))
            painter.copyOptionsFrom(this);
      }, 'objects');
   }

   /** @summary Scan histogram content
     * @abstract */
   scanContent(/* when_axis_changed */) {
      // function will be called once new histogram or
      // new histogram content is assigned
      // one should find min,max,nbins, maxcontent values
      // if when_axis_changed === true specified, content will be scanned after axis zoom changed
   }

   /** @summary Check pad ranges when drawing of frame axes will be performed
     * @desc Only if histogram is main painter and drawn with SAME option, pad range can be used
     * In all other cases configured range must be derived from histogram itself */
   checkPadRange() {
      if (this.isMainPainter())
         this.check_pad_range = this.options.Same ? 'pad_range' : true;
   }

   /** @summary Create necessary histogram draw attributes */
   createHistDrawAttributes(only_check_auto) {
      const histo = this.getHisto(), o = this.options;

      if (o._pfc > 1 || o._plc > 1 || o._pmc > 1) {
         const pp = this.getPadPainter();
         if (isFunc(pp?.getAutoColor)) {
            const icolor = pp.getAutoColor(histo.$num_histos);
            this._auto_exec = ''; // can be reused when sending option back to server
            if (o._pfc > 1) { o._pfc = 1; histo.fFillColor = icolor; this._auto_exec += `SetFillColor(${icolor});;`; delete this.fillatt; }
            if (o._plc > 1) { o._plc = 1; histo.fLineColor = icolor; this._auto_exec += `SetLineColor(${icolor});;`; delete this.lineatt; }
            if (o._pmc > 1) { o._pmc = 1; histo.fMarkerColor = icolor; this._auto_exec += `SetMarkerColor(${icolor});;`; delete this.markeratt; }
         }
      }

      if (only_check_auto)
         this.deleteAttr();
      else {
         this.createAttFill({ attr: histo, color: this.options.histoFillColor, pattern: this.options.histoFillPattern, kind: 1 });
         this.createAttLine({ attr: histo, color0: this.options.histoLineColor });
      }
   }

   /** @summary Update axes attributes in target histogram
     * @private */
   updateAxes(tgt_histo, src_histo, fp) {
      const copyTAxisMembers = (tgt, src, copy_zoom) => {
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
         if (copy_zoom) {
            tgt.fFirst = src.fFirst;
            tgt.fLast = src.fLast;
            tgt.fBits = src.fBits;
         }
      };

      copyTAxisMembers(tgt_histo.fXaxis, src_histo.fXaxis, this.snapid && !fp?.zoomChangedInteractive('x'));
      copyTAxisMembers(tgt_histo.fYaxis, src_histo.fYaxis, this.snapid && !fp?.zoomChangedInteractive('y'));
      copyTAxisMembers(tgt_histo.fZaxis, src_histo.fZaxis, this.snapid && !fp?.zoomChangedInteractive('z'));
   }

   /** @summary Update histogram object
     * @param obj - new histogram instance
     * @param opt - new drawing option (optional)
     * @return {Boolean} - true if histogram was successfully updated */
   updateObject(obj, opt) {
      const histo = this.getHisto(),
            fp = this.getFramePainter(),
            pp = this.getPadPainter(),
            o = this.options;

      if (obj !== histo) {
         if (!this.matchObjectType(obj)) return false;

         // simple replace of object does not help - one can have different
         // complex relations between histo and stat box, histo and colz axis,
         // one could have THStack or TMultiGraph object
         // The only that could be done is update of content

         const statpainter = pp?.findPainterFor(this.findStat());

         // copy histogram bits
         if (histo.TestBit(kNoStats) !== obj.TestBit(kNoStats)) {
            histo.InvertBit(kNoStats);
            // here check only stats bit
            if (statpainter) statpainter.Enabled = !histo.TestBit(kNoStats) && !this.options.NoStat; // && (!this.options.Same || this.options.ForceStat)
         }

         if (histo.TestBit(kIsZoomed) !== obj.TestBit(kIsZoomed))
            histo.InvertBit(kIsZoomed);

         // special treatment for webcanvas - also name can be changed
         if (this.snapid !== undefined) {
            histo.fName = obj.fName;
            o._pfc = o._plc = o._pmc = 0; // auto colors should be processed in web canvas
         }

         if (!o._pfc)
            histo.fFillColor = obj.fFillColor;
         histo.fFillStyle = obj.fFillStyle;
         if (!o._plc)
            histo.fLineColor = obj.fLineColor;
         histo.fLineStyle = obj.fLineStyle;
         histo.fLineWidth = obj.fLineWidth;
         if (!o._pmc)
            histo.fMarkerColor = obj.fMarkerColor;
         histo.fMarkerSize = obj.fMarkerSize;
         histo.fMarkerStyle = obj.fMarkerStyle;

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

         this.updateAxes(histo, obj, fp);

         histo.fArray = obj.fArray;
         histo.fNcells = obj.fNcells;
         histo.fTitle = obj.fTitle;
         histo.fMinimum = obj.fMinimum;
         histo.fMaximum = obj.fMaximum;
         histo.fSumw2 = obj.fSumw2;

         if (!o.ominimum) o.minimum = histo.fMinimum;
         if (!o.omaximum) o.maximum = histo.fMaximum;

         if (this.getDimension() === 1)
            o.decodeSumw2(histo);

         if (this.isTProfile())
            histo.fBinEntries = obj.fBinEntries;
          else if (this.isTH1K()) {
            histo.fNIn = obj.fNIn;
            histo.fReady = 0;
         } else if (this.isTH2Poly())
            histo.fBins = obj.fBins;

         // remove old functions, update existing, prepare to draw new one
         this._funcHandler = new FunctionsHandler(this, pp, obj.fFunctions, statpainter);

         const changed_opt = (histo.fOption !== obj.fOption);
         histo.fOption = obj.fOption;

         if (((opt !== undefined) && (o.original !== opt)) || changed_opt)
            this.decodeOptions(opt || histo.fOption);
      }

      if (!o.ominimum)
         o.minimum = histo.fMinimum;
      if (!o.omaximum)
         o.maximum = histo.fMaximum;

      if (!fp || !fp.zoomChangedInteractive())
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
            axis.GetBinCoord = function(bin) {
               const indx = Math.round(bin);
               if (indx <= 0) return this.fXmin;
               if (indx > this.fNbins) return this.fXmax;
               if (indx === bin) return this.fXbins[indx];
               const indx2 = (bin < indx) ? indx - 1 : indx + 1;
               return this.fXbins[indx] * Math.abs(bin-indx2) + this.fXbins[indx2] * Math.abs(bin-indx);
            };
            axis.FindBin = function(x, add) {
               for (let k = 1; k < this.fXbins.length; ++k)
                  if (x < this.fXbins[k]) return Math.floor(k-1+add);
               return this.fNbins;
            };
         } else {
            axis.$binwidth = (axis.fXmax - axis.fXmin) / (axis.fNbins || 1);
            axis.GetBinCoord = function(bin) { return this.fXmin + bin*this.$binwidth; };
            axis.FindBin = function(x, add) { return Math.floor((x - this.fXmin) / this.$binwidth + add); };
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

      this._exact_y_range = (ndim === 1) && this.options.ohmin && this.options.ohmax;

      if (this._exact_y_range) {
         this.ymin = this.options.hmin;
         this.ymax = this.options.hmax;
      }

      if (ndim > 1) {
         this.nbinsy = histo.fYaxis.fNbins;
         assignTAxisFuncs(histo.fYaxis);

         this.zmin = histo.fZaxis.fXmin;
         this.zmax = histo.fZaxis.fXmax;

         if ((ndim === 2) && this.options.ohmin && this.options.ohmax) {
            this.zmin = this.options.hmin;
            this.zmax = this.options.hmax;
         }
      }

      if (ndim > 2) {
         this.nbinsz = histo.fZaxis.fNbins;
         assignTAxisFuncs(histo.fZaxis);
       }
   }

    /** @summary Draw axes for histogram
      * @desc axes can be drawn only for main histogram */
   async drawAxes() {
      const fp = this.getFramePainter();
      if (!fp) return false;

      const histo = this.getHisto();

      // artificially add y range to display axes
      if (this.ymin === this.ymax) this.ymax += 1;

      if (!this.isMainPainter()) {
         const opts = {
            second_x: (this.options.AxisPos >= 10),
            second_y: (this.options.AxisPos % 10) === 1,
            hist_painter: this
         };

         if ((!opts.second_x && !opts.second_y) || fp.hasDrawnAxes(opts.second_x, opts.second_y))
            return false;

         fp.setAxes2Ranges(opts.second_x, histo.fXaxis, this.xmin, this.xmax, opts.second_y, histo.fYaxis, this.ymin, this.ymax);

         fp.createXY2(opts);

         return fp.drawAxes2(opts.second_x, opts.second_y);
      }

      if (this.options.adjustFrame) {
         const pad = this.getPadPainter().getRootPad();
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
                    zoom_xmin: this.zoom_xmin,
                    zoom_xmax: this.zoom_xmax,
                    zoom_ymin: this.zoom_ymin,
                    zoom_ymax: this.zoom_ymax,
                    ymin_nz: this.ymin_nz,
                    swap_xy: (this.options.BarStyle >= 20),
                    reverse_x: this.options.RevX,
                    reverse_y: this.options.RevY,
                    symlog_x: this.options.SymlogX,
                    symlog_y: this.options.SymlogY,
                    Proj: this.options.Proj,
                    extra_y_space: this.options.Text && (this.options.BarStyle > 0),
                    hist_painter: this });

      delete this.check_pad_range;
      delete this.zoom_xmin;
      delete this.zoom_xmax;
      delete this.zoom_ymin;
      delete this.zoom_ymax;

      if (this.options.Same)
         return false;

      const disable_axis_draw = (this.options.Axis < 0) || (this.options.Axis === 2);

      return fp.drawAxes(false, disable_axis_draw, disable_axis_draw,
                         this.options.AxisPos, this.options.Zscale && this.options.Zvert,
                         this.options.Zscale && !this.options.Zvert, this.options.Axis !== 1);
   }

   /** @summary Inform web canvas that something changed in the histogram */
   processOnlineChange(kind) {
      const cp = this.getCanvPainter();
      if (isFunc(cp?.processChanges))
         cp.processChanges(kind, this);
   }

   /** @summary Fill option object used in TWebCanvas */
   fillWebObjectOptions(res) {
      if (this._auto_exec && res) {
         res.fcust = 'auto_exec:' + this._auto_exec;
         delete this._auto_exec;
      }
   }

   /** @summary Toggle histogram title drawing */
   toggleTitle(arg) {
      const histo = this.getHisto();
      if (!this.isMainPainter() || !histo)
         return false;
      if (arg === 'only-check')
         return !histo.TestBit(kNoTitle);
      histo.InvertBit(kNoTitle);
      this.updateHistTitle().then(() => this.processOnlineChange(`exec:SetBit(TH1::kNoTitle,${histo.TestBit(kNoTitle)?1:0})`));
   }

   /** @summary Only redraw histogram title
     * @return {Promise} with painter */
   async updateHistTitle() {
      // case when histogram drawn over other histogram (same option)
      if (!this.isMainPainter() || this.options.Same || (this.options.Axis > 0))
         return this;

      const tpainter = this.getPadPainter()?.findPainterFor(null, kTitle, clTPaveText),
            pt = tpainter?.getObject();

      if (!tpainter || !pt)
         return this;

      const histo = this.getHisto(), st = gStyle,
            draw_title = !histo.TestBit(kNoTitle) && (st.fOptTitle > 0);

      pt.Clear();
      if (draw_title) pt.AddText(histo.fTitle);
      return tpainter.redraw().then(() => this);
   }

   /** @summary Draw histogram title
     * @return {Promise} with painter */
   async drawHistTitle() {
      // case when histogram drawn over other histogram (same option)
      if (!this.isMainPainter() || this.options.Same || (this.options.Axis > 0))
         return this;

      const histo = this.getHisto(), st = gStyle,
            draw_title = !histo.TestBit(kNoTitle) && (st.fOptTitle > 0);

      let pt = this.getPadPainter()?.findInPrimitives(kTitle, clTPaveText);

      if (pt) {
         pt.Clear();
         if (draw_title) pt.AddText(histo.fTitle);
         return this;
      }

      pt = create(clTPaveText);
      Object.assign(pt, { fName: kTitle, fFillColor: st.fTitleColor, fFillStyle: st.fTitleStyle, fBorderSize: st.fTitleBorderSize,
                          fTextFont: st.fTitleFont, fTextSize: st.fTitleFontSize, fTextColor: st.fTitleTextColor, fTextAlign: st.fTitleAlign });
      if (draw_title) pt.AddText(histo.fTitle);
      return TPavePainter.draw(this.getDom(), pt, 'postitle');
   }

   /** @summary Live change and update of title drawing
     * @desc Used from the GED */
   processTitleChange(arg) {
      const histo = this.getHisto(),
            tpainter = this.getPadPainter()?.findPainterFor(null, kTitle);

      if (!histo || !tpainter) return null;

      if (arg === 'check')
         return (!this.isMainPainter() || this.options.Same) ? null : histo;

      tpainter.clearPave();
      tpainter.addText(histo.fTitle);

      tpainter.redraw();

      this.submitCanvExec(`SetTitle("${histo.fTitle}")`);
   }

   /** @summary Update statistics when web canvas is drawn */
   updateStatWebCanvas() {
      if (!this.snapid) return;

      const stat = this.findStat(),
            statpainter = this.getPadPainter()?.findPainterFor(stat);

      if (statpainter && !statpainter.snapid) statpainter.redraw();
   }

   /** @summary Find stats box in list of functions */
   findStat() {
      return this.findFunction(clTPaveStats, 'stats');
   }

   /** @summary Toggle statbox drawing
     * @private */
   toggleStat(arg) {
      let stat = this.findStat(), statpainter;

      if (!arg) arg = '';

      if (!stat) {
         if (arg.indexOf('-check') > 0) return false;
         // when statbox created first time, one need to draw it
         stat = this.createStat(true);
      } else
         statpainter = this.getPadPainter()?.findPainterFor(stat);


      if (arg === 'only-check')
         return statpainter?.Enabled || false;

      if (arg === 'fitpar-check')
         return stat?.fOptFit || false;

      if (arg === 'fitpar-toggle') {
         if (!stat) return false;
         stat.fOptFit = stat.fOptFit ? 0 : 1111; // for websocket command should be send to server
         statpainter?.redraw();
         return true;
      }

      let has_stats;

      if (statpainter) {
         statpainter.Enabled = !statpainter.Enabled;
         this.options.StatEnabled = statpainter.Enabled; // used only for interactive
         // when stat box is drawn, it always can be drawn individually while it
         // should be last for colz redrawPad is used
         statpainter.redraw();
         has_stats = statpainter.Enabled;
      } else {
         const prev_name = this.selectCurrentPad(this.getPadName());
         // return promise which will be used to process
         has_stats = TPavePainter.draw(this.getDom(), stat).then(() => this.selectCurrentPad(prev_name));
      }

      this.processOnlineChange(`exec:SetBit(TH1::kNoStats,${has_stats ? 0 : 1})`, this);

      return has_stats;
   }

   /** @summary Returns true if stats box fill can be ingored */
   isIgnoreStatsFill() {
      return !this.getObject() || (!this.draw_content && !this.create_stats && !this.snapid); // || (this.options.Axis > 0);
   }

   /** @summary Create stat box for histogram if required */
   createStat(force) {
      const histo = this.getHisto();
      if (!histo) return null;

      if (!force && !this.options.ForceStat) {
         if (this.options.NoStat || histo.TestBit(kNoStats) || !settings.AutoStat) return null;
         if (!this.isMainPainter()) return null;
      }

      const st = gStyle;
      let stats = this.findStat(),
          optstat = this.options.optstat,
          optfit = this.options.optfit;

      if (optstat !== undefined) {
         if (stats) stats.fOptStat = optstat;
         delete this.options.optstat;
      } else
         optstat = histo.$custom_stat || st.fOptStat;

      if (optfit !== undefined) {
         if (stats) stats.fOptFit = optfit;
         delete this.options.optfit;
      } else
         optfit = st.fOptFit;

      if (!stats && !optstat && !optfit) return null;

      this.create_stats = true;

      if (stats)
         return stats;

      stats = create(clTPaveStats);
      Object.assign(stats, {
         fName: 'stats', fOptStat: optstat, fOptFit: optfit,
         fX1NDC: st.fStatX - st.fStatW, fY1NDC: st.fStatY - st.fStatH, fX2NDC: st.fStatX, fY2NDC: st.fStatY,
         fTextAlign: 12
      });

      stats.AddText(histo.fName);

      this.addFunction(stats);

      return stats;
   }

   /** @summary Find function in histogram list of functions */
   findFunction(type_name, obj_name) {
      const funcs = this.getHisto()?.fFunctions?.arr;
      if (!funcs) return null;

      for (let i = 0; i < funcs.length; ++i) {
         const f = funcs[i];
         if (obj_name && (f.fName !== obj_name)) continue;
         if (f._typename === type_name) return f;
      }

      return null;
   }

   /** @summary Add function to histogram list of functions */
   addFunction(obj, asfirst) {
      const histo = this.getHisto();
      if (!histo || !obj) return;

      if (!histo.fFunctions)
         histo.fFunctions = create(clTList);

      if (asfirst)
         histo.fFunctions.AddFirst(obj);
      else
         histo.fFunctions.Add(obj);
   }

   /** @summary Check if such function should be drawn directly */
   needDrawFunc(histo, func) {
      if (func._typename === clTPaveStats)
          return (func.fName !== 'stats') || (!histo.TestBit(kNoStats) && !this.options.NoStat); // && (!this.options.Same || this.options.ForceStat))

       if ((func._typename === clTF1) || (func._typename === clTF2))
          return this.options.AllFunc || !func.TestBit(BIT(9)); // TF1::kNotDraw

       if ((func._typename === 'TGraphDelaunay') || (func._typename === 'TGraphDelaunay2D'))
          return false; // do not try to draw delaunay classes

       return func._typename !== clTPaletteAxis;
   }

   /** @summary Method draws functions from the histogram list of functions
     * @return {Promise} fulfilled when drawing is ready */
   async drawFunctions() {
      const handler = new FunctionsHandler(this, this.getPadPainter(), this.getHisto().fFunctions, true);
      return handler.drawNext(0); // returns this painter
   }

   /** @summary Method used to update functions which are prepared before
     * @return {Promise} fulfilled when drawing is ready */
   async updateFunctions() {
      const res = this._funcHandler?.drawNext(0) ?? this;
      delete this._funcHandler;
      return res;
   }

   /** @summary Returns selected index for specified axis
     * @desc be aware - here indexes starts from 0 */
   getSelectIndex(axis, side, add) {
      let indx = 0, taxis = this.getAxis(axis);
      const nbin = this[`nbins${axis}`] ?? 0;

      if (this.options.second_x && axis === 'x') axis = 'x2';
      if (this.options.second_y && axis === 'y') axis = 'y2';
      const main = this.getFramePainter(),
            min = main ? main[`zoom_${axis}min`] : 0,
            max = main ? main[`zoom_${axis}max`] : 0;

      if ((min !== max) && taxis) {
         if (side === 'left')
            indx = taxis.FindBin(min, add || 0);
         else
            indx = taxis.FindBin(max, (add || 0) + 0.5);
         if (indx < 0)
            indx = 0; else
         if (indx > nbin)
            indx = nbin;
      } else
         indx = (side === 'left') ? 0 : nbin;


      // TAxis object of histogram, where user range can be stored
      if (taxis) {
         if ((taxis.fFirst === taxis.fLast) || !taxis.TestBit(EAxisBits.kAxisRange) ||
             ((taxis.fFirst <= 1) && (taxis.fLast >= nbin))) taxis = undefined;
      }

      if (side === 'left') {
         if (indx < 0) indx = 0;
         if (taxis && (taxis.fFirst > 1) && (indx < taxis.fFirst)) indx = taxis.fFirst - 1;
      } else {
         if (indx > nbin) indx = nbin;
         if (taxis && (taxis.fLast <= nbin) && (indx>taxis.fLast)) indx = taxis.fLast;
      }

      return indx;
   }

   /** @summary Unzoom user range if any */
   unzoomUserRange(dox, doy, doz) {
      const histo = this.getHisto();
      if (!histo) return false;

      let res = false;

      const unzoomTAxis = obj => {
         if (!obj || !obj.TestBit(EAxisBits.kAxisRange)) return false;
         if (obj.fFirst === obj.fLast) return false;
         if ((obj.fFirst <= 1) && (obj.fLast >= obj.fNbins)) return false;
         obj.InvertBit(EAxisBits.kAxisRange);
         return true;
      },

      uzoomMinMax = ndim => {
         if (this.getDimension() !== ndim) return false;
         if ((this.options.minimum === kNoZoom) && (this.options.maximum === kNoZoom)) return false;
         if (!this.draw_content) return false; // if not drawing content, not change min/max
         this.options.minimum = this.options.maximum = kNoZoom;
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
     * @return {Promise} for ready */
   async addInteractivity() {
      const ismain = this.isMainPainter(),
            second_axis = (this.options.AxisPos > 0),
            fp = (ismain || second_axis) ? this.getFramePainter() : null;
      return fp?.addInteractivity(!ismain && second_axis) ?? false;
   }

   /** @summary Invoke dialog to enter and modify user range */
   changeUserRange(menu, arg) {
      const histo = this.getHisto(),
            taxis = histo ? histo[`f${arg}axis`] : null;
      if (!taxis) return;

      let curr = `[1,${taxis.fNbins}]`;
      if (taxis.TestBit(EAxisBits.kAxisRange))
          curr = `[${taxis.fFirst},${taxis.fLast}]`;

      menu.input(`Enter user range for axis ${arg} like [1,${taxis.fNbins}]`, curr).then(res => {
         if (!res) return;
         res = JSON.parse(res);
         if (!res || (res.length !== 2)) return;
         const first = parseInt(res[0]), last = parseInt(res[1]);
         if (!Number.isInteger(first) || !Number.isInteger(last)) return;
         taxis.fFirst = first;
         taxis.fLast = last;

         const newflag = (taxis.fFirst < taxis.fLast) && (taxis.fFirst >= 1) && (taxis.fLast <= taxis.fNbins);
         if (newflag !== taxis.TestBit(EAxisBits.kAxisRange))
            taxis.InvertBit(EAxisBits.kAxisRange);

         this.interactiveRedraw();
      });
   }

   /** @summary Start dialog to modify range of axis where histogram values are displayed */
   changeValuesRange(menu) {
      let curr;
      if ((this.options.minimum !== kNoZoom) && (this.options.maximum !== kNoZoom))
         curr = `[${this.options.minimum},${this.options.maximum}]`;
      else
         curr = `[${this.gminbin},${this.gmaxbin}]`;

      menu.input('Enter min/max hist values or empty string to reset', curr).then(res => {
         res = res ? JSON.parse(res) : [];

         if (!isObject(res) || (res.length !== 2) || !Number.isFinite(res[0]) || !Number.isFinite(res[1]))
            this.options.minimum = this.options.maximum = kNoZoom;
          else {
            this.options.minimum = res[0];
            this.options.maximum = res[1];
          }

         this.interactiveRedraw();
       });
   }

   /** @summary Execute histogram menu command
     * @desc Used to catch standard menu items and provide local implementation */
   executeMenuCommand(method, args) {
      if (super.executeMenuCommand(method, args))
         return true;

      if (method.fClassName === clTAxis) {
         const p = isStr(method.$execid) ? method.$execid.indexOf('#') : -1,
               kind = p > 0 ? method.$execid.slice(p+1) : 'x',
               fp = this.getFramePainter();
         if (method.fName === 'UnZoom') {
            fp?.unzoom(kind);
            return true;
         } else if (method.fName === 'SetRange') {
            const axis = fp?.getAxis(kind), bins = JSON.parse(`[${args}]`);
            if (axis && bins?.length === 2)
               fp?.zoom(kind, axis.GetBinLowEdge(bins[0]), axis.GetBinLowEdge(bins[1]+1));
            // let execute command on server
         } else if (method.fName === 'SetRangeUser') {
            const values = JSON.parse(`[${args}]`);
            if (values?.length === 2)
               fp?.zoom(kind, values[0], values[1]);
            // let execute command on server
         }
      }

      return false;
   }

   /** @summary Fill histogram context menu */
   fillContextMenuItems(menu) {
      const histo = this.getHisto(),
            fp = this.getFramePainter();
      if (!histo) return;

      if ((this.options.Axis <= 0) && !this.isTF1())
         menu.addchk(this.toggleStat('only-check'), 'Show statbox', () => this.toggleStat());

      if (histo.fTitle && this.isMainPainter())
         menu.addchk(this.toggleTitle('only-check'), 'Show title', () => this.toggleTitle());

      if (this.draw_content) {
         if (this.getDimension() === 1)
            menu.add('User range X', () => this.changeUserRange(menu, 'X'));
          else {
            menu.add('sub:User ranges');
            menu.add('X', () => this.changeUserRange(menu, 'X'));
            menu.add('Y', () => this.changeUserRange(menu, 'Y'));
            if (this.getDimension() > 2)
               menu.add('Z', () => this.changeUserRange(menu, 'Z'));
            else
               menu.add('Values', () => this.changeValuesRange(menu));
            menu.add('endsub:');
         }

         if (isFunc(this.fillHistContextMenu))
            this.fillHistContextMenu(menu);
      }

      if (this.options.Mode3D) {
         // menu for 3D drawings

         if (menu.size() > 0)
            menu.add('separator');

         const main = this.getMainPainter() || this;

         menu.addchk(main.isTooltipAllowed(), 'Show tooltips', () => main.setTooltipAllowed('toggle'));

         menu.addchk(fp?.enable_highlight, 'Highlight bins', () => {
            fp.enable_highlight = !fp.enable_highlight;
            if (!fp.enable_highlight && fp.mode3d && isFunc(fp.highlightBin3D))
               fp.highlightBin3D(null);
         });

         if (isFunc(fp?.render3D)) {
            menu.addchk(main.options.FrontBox, 'Front box', () => {
               main.options.FrontBox = !main.options.FrontBox;
               fp.render3D();
            });
            menu.addchk(main.options.BackBox, 'Back box', () => {
               main.options.BackBox = !main.options.BackBox;
               fp.render3D();
            });
            menu.addchk(fp.camera?.isOrthographicCamera, 'Orthographic camera', flag => {
               main.options.Ortho = flag;
               fp.change3DCamera(flag);
            });
         }

         if (this.draw_content) {
            menu.addchk(!this.options.Zero, 'Suppress zeros', () => {
               this.options.Zero = !this.options.Zero;
               this.interactiveRedraw('pad');
            });

            if ((this.options.Lego === 12) || (this.options.Lego === 14)) {
               menu.addchk(this.options.Zscale, 'Z scale', () => this.toggleColz());
               this.fillPaletteMenu(menu, true);
            }
         }

         if (isFunc(main.control?.reset))
            menu.add('Reset camera', () => main.control.reset());
      }

      if (this.histogram_updated && fp.zoomChangedInteractive())
         menu.add('Let update zoom', () => fp.zoomChangedInteractive('reset'));
   }

   /** @summary Returns snap id for object or subelement
     * @private */
   getSnapId(subelem) {
      if (!this.snapid)
         return '';
      let res = this.snapid.toString();
      if (subelem) {
         res += '#';
         if (this.isTF1() && (subelem === 'x' || subelem === 'y' || subelem === 'z'))
             res += 'hist#';
         res += subelem;
      }
      return res;
   }

   /** @summary Auto zoom into histogram non-empty range
     * @abstract */
   autoZoom() {}

   /** @summary Process click on histogram-defined buttons */
   clickButton(funcname) {
      const fp = this.getFramePainter();
      if (!this.isMainPainter() || !fp) return false;

      switch (funcname) {
         case 'ToggleZoom':
            if ((fp.zoom_xmin !== fp.zoom_xmax) || (fp.zoom_ymin !== fp.zoom_ymax) || (fp.zoom_zmin !== fp.zoom_zmax)) {
               const pr = fp.unzoom();
               fp.zoomChangedInteractive('reset');
               return pr;
            }
            if (this.draw_content)
               return this.autoZoom();
            break;
         case 'ToggleLogX': return fp.toggleAxisLog('x');
         case 'ToggleLogY': return fp.toggleAxisLog('y');
         case 'ToggleLogZ': return fp.toggleAxisLog('z');
         case 'ToggleStatBox': return getPromise(this.toggleStat());
      }
      return false;
   }

   /** @summary Fill pad toolbar with histogram-related functions */
   fillToolbar(not_shown) {
      const pp = this.getPadPainter();
      if (!pp) return;

      pp.addPadButton('auto_zoom', 'Toggle between unzoom and autozoom-in', 'ToggleZoom', 'Ctrl *');
      pp.addPadButton('arrow_right', 'Toggle log x', 'ToggleLogX', 'PageDown');
      pp.addPadButton('arrow_up', 'Toggle log y', 'ToggleLogY', 'PageUp');
      if (this.getDimension() > 1)
         pp.addPadButton('arrow_diag', 'Toggle log z', 'ToggleLogZ');
      pp.addPadButton('statbox', 'Toggle stat box', 'ToggleStatBox');
      if (!not_shown)
         pp.showPadButtons();
   }

   /** @summary Returns tooltip information for 3D drawings */
   get3DToolTip(indx) {
      const histo = this.getHisto(),
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
      const cntr = new HistContour(zmin, zmax),
            ndim = this.getDimension();

      if (custom_levels)
         cntr.createCustom(custom_levels);
      else {
         if (nlevels < 2) nlevels = gStyle.fNumberContours;
         const pad = this.getPadPainter().getRootPad(true),
               logv = pad?.fLogv ?? ((ndim === 2) && pad?.fLogz);

         cntr.createNormal(nlevels, logv ?? 0, zminpositive);
      }

      cntr.configIndicies(this.options.Zero ? -1 : 0, (cntr.colzmin !== 0) || !this.options.Zero || this.isTH2Poly() ? 0 : -1);

      const fp = this.getFramePainter();
      if (fp && (ndim < 3) && !fp.mode3d) {
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

      const main = this.getMainPainter(),
            fp = this.getFramePainter();

      if (main?.fContour && (main !== this) && !this.options.IgnoreMainScale) {
         this.fContour = main.fContour;
         return this.fContour;
      }

      // if not initialized, first create contour array
      // difference from ROOT - fContour includes also last element with maxbin, which makes easier to build logz
      // when no same0 draw option specified, use main painter for creating contour, also ignore scatter drawing for main painer
      const histo = this.getObject(),
            src = (this !== main) && (main?.minbin !== undefined) && !this.options.IgnoreMainScale && !main?.tt_handle?.ScatterPlot ? main : this;
      let nlevels = 0, apply_min,
          zmin = src.minbin, zmax = src.maxbin, zminpos = src.minposbin,
          custom_levels;
      if (zmin === zmax) { zmin = src.gminbin; zmax = src.gmaxbin; zminpos = src.gminposbin; }

      let gzmin = zmin, gzmax = zmax;
      if (this.options.minimum !== kNoZoom) { zmin = this.options.minimum; gzmin = Math.min(gzmin, zmin); apply_min = true; }
      if (this.options.maximum !== kNoZoom) { zmax = this.options.maximum; gzmax = Math.max(gzmax, zmax); apply_min = false; }
      if (zmin >= zmax) {
         if (apply_min)
            zmax = zmin + 1;
         else
            zmin = zmax - 1;
      }

      if (fp && (fp.zoom_zmin !== fp.zoom_zmax)) {
         zmin = fp.zoom_zmin;
         zmax = fp.zoom_zmax;
      }

      if (histo.fContour?.length > 1) {
         if (histo.TestBit(kUserContour))
            custom_levels = histo.fContour;
         else
            nlevels = histo.fContour.length;
      }

      const cntr = this.createContour(nlevels, zmin, zmax, zminpos, custom_levels);

      if ((this.getDimension() < 3) && fp) {
         fp.zmin = gzmin;
         fp.zmax = gzmax;

         if ((gzmin !== cntr.colzmin) || (gzmax !== cntr.colzmax)) {
            fp.zoom_zmin = cntr.colzmin;
            fp.zoom_zmax = cntr.colzmax;
         } else
            fp.zoom_zmin = fp.zoom_zmax = undefined;
      }

      return cntr;
   }

   /** @summary Return levels from contour object */
   getContourLevels(force_recreate) {
      return this.getContour(force_recreate).getLevels();
   }

   /** @summary Returns color palette associated with histogram
     * @desc Create if required, checks pad and canvas for custom palette */
   getHistPalette(force) {
      if (force) this._color_palette = null;
      const pp = this.getPadPainter();
      if (!this._color_palette && !this.options.Palette) {
         if (isFunc(pp?.getCustomPalette))
            this._color_palette = pp.getCustomPalette();
      }
      if (!this._color_palette)
         this._color_palette = getColorPalette(this.options.Palette, pp?.isGrayscale());
      return this._color_palette;
   }

   /** @summary Fill menu entries for palette */
   fillPaletteMenu(menu, only_palette) {
      menu.addPaletteMenu(this.options.Palette || settings.Palette, arg => {
         this.options.Palette = parseInt(arg);
         this.getHistPalette(true);
         this.redraw(); // redraw histogram
      });
      if (!only_palette) {
         menu.add('Default position', () => {
             this.drawColorPalette(this.options.Zscale, false, true)
                     .then(() => this.processOnlineChange('drawopt'));
         }, 'Set default position for palette');

         const pal = this.findFunction(clTPaletteAxis),
               is_vert = !pal ? true : pal.fX2NDC - pal.fX1NDC < pal.fY2NDC - pal.fY1NDC;
         menu.addchk(is_vert, 'Vertical', flag => {
            this.options.Zvert = flag;
            this.drawColorPalette(this.options.Zscale, false, 'toggle')
                     .then(() => this.processOnlineChange('drawopt'));
         }, 'Toggle palette vertical/horizontal flag');

         menu.add('Bring to front', () => this.getPadPainter()?.findPainterFor(pal)?.bringToFront());
      }
   }

   /** @summary draw color palette
     * @return {Promise} when done */
   async drawColorPalette(enabled, postpone_draw, can_move) {
      // in special cases like scatter palette drawing is ignored
      if (this.options.IgnorePalette)
         return null;

      // only when create new palette, one could change frame size
      const mp = this.getMainPainter(),
            pp = this.getPadPainter();
      if (mp !== this) {
         if (mp && (mp.draw_content !== false) && mp.options.Zscale)
            return null;
      }

      let pal = this.findFunction(clTPaletteAxis),
          pal_painter = pp?.findPainterFor(pal);

      const found_in_func = !!pal;

      if (this._can_move_colz) {
         delete this._can_move_colz;
         if (!can_move) can_move = true;
      }

      if (!pal_painter && !pal && !this.options.Axis) {
         pal_painter = pp?.findPainterFor(undefined, undefined, clTPaletteAxis);
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

         return null;
      }

      if (!pal) {
         pal = create(clTPaletteAxis);

         pal.fInit = 1;
         pal.$can_move = true;
         pal.$generated = true;

         if (!this.options.Zvert)
            Object.assign(pal, { fX1NDC: gStyle.fPadLeftMargin, fX2NDC: 1 - gStyle.fPadRightMargin, fY1NDC: 1.005 - gStyle.fPadTopMargin, fY2NDC: 1.045 - gStyle.fPadTopMargin });
         else
            Object.assign(pal, { fX1NDC: 1.005 - gStyle.fPadRightMargin, fX2NDC: 1.045 - gStyle.fPadRightMargin, fY1NDC: gStyle.fPadBottomMargin, fY2NDC: 1 - gStyle.fPadTopMargin });

         Object.assign(pal.fAxis, { fChopt: '+', fLineSyle: 1, fLineWidth: 1, fTextAngle: 0, fTextAlign: 11 });

         if (this.getDimension() === 2) {
            const zaxis = this.getHisto().fZaxis;
            Object.assign(pal.fAxis, { fTitle: zaxis.fTitle, fTitleSize: zaxis.fTitleSize,
                                       fTitleOffset: zaxis.fTitleOffset, fTitleColor: zaxis.fTitleColor,
                                       fLineColor: zaxis.fAxisColor, fTextSize: zaxis.fLabelSize,
                                       fTextColor: zaxis.fLabelColor, fTextFont: zaxis.fLabelFont,
                                       fLabelOffset: zaxis.fLabelOffset });
         }

         // place colz in the beginning, that stat box is always drawn on the top
         this.addFunction(pal, true);

         can_move = true;
      } else if (pp?._palette_vertical !== undefined)
         this.options.Zvert = pp._palette_vertical;

      const fp = this.getFramePainter();

      // keep palette width
      if (can_move && fp && pal.$can_move) {
         if (this.options.Zvert) {
            if (can_move === 'toggle') {
               const d = pal.fY2NDC - pal.fY1NDC;
               pal.fX1NDC = fp.fX2NDC + 0.005;
               pal.fX2NDC = pal.fX1NDC + d;
            }
            if (pal.fX1NDC > (fp.fX1NDC + fp.fX2NDC)*0.5) {
               pal.fX2NDC = fp.fX2NDC + 0.005 + (pal.fX2NDC - pal.fX1NDC);
               pal.fX1NDC = fp.fX2NDC + 0.005;
            } else {
               pal.fX1NDC = fp.fX1NDC - 0.03 - (pal.fX2NDC - pal.fX1NDC);
               pal.fX2NDC = fp.fX1NDC - 0.03;
            }
            pal.fY1NDC = fp.fY1NDC;
            pal.fY2NDC = fp.fY2NDC;
         } else {
            if (can_move === 'toggle') {
               const d = pal.fX2NDC - pal.fX1NDC;
               pal.fY1NDC = fp.fY2NDC + 0.005;
               pal.fY2NDC = pal.fY1NDC + d;
            }

            pal.fX1NDC = fp.fX1NDC;
            pal.fX2NDC = fp.fX2NDC;
            if (pal.fY2NDC > (fp.fY1NDC + fp.fY2NDC)*0.5) {
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

      let arg = '', pr;
      if (postpone_draw) arg += ';postpone';
      if (can_move && !this.do_redraw_palette) arg += ';can_move';
      if (this.options.Cjust) arg += ';cjust';

      if (!pal_painter) {
         // when histogram drawn on sub pad, let draw new axis object on the same pad
         const prev = this.selectCurrentPad(this.getPadName());
         pr = TPavePainter.draw(this.getDom(), pal, arg).then(_palp => {
            pal_painter = _palp;
            this.selectCurrentPad(prev);
            pal_painter.setSecondaryId(this, found_in_func && !pal.$generated ? `func_${pal.fName}` : undefined);
         });
      } else {
         pal_painter.Enabled = true;
         // real drawing will be perform at the end
         if (postpone_draw) return pal_painter;
         pr = pal_painter.drawPave(arg);
      }

      return pr.then(() => {
         // mark painter as secondary - not in list of TCanvas primitives
         this.options.Zvert = pal_painter._palette_vertical;

         // make dummy redraw, palette will be updated only from histogram painter
         pal_painter.redraw = () => {};

         let need_redraw = false;

         // special code to adjust frame position to actual position of palette
         if (can_move && fp && !this.do_redraw_palette) {
            const pad = pp?.getRootPad(true);

            if (this.options.Zvert) {
               if ((pal.fX1NDC > 0.5) && (fp.fX2NDC > pal.fX1NDC)) {
                  need_redraw = true;
                  fp.fX2NDC = pal.fX1NDC - 0.01;

                  if (fp.fX1NDC > fp.fX2NDC - 0.1) fp.fX1NDC = Math.max(0, fp.fX2NDC - 0.1);
                } else if ((pal.fX2NDC < 0.5) && (fp.fX1NDC < pal.fX2NDC)) {
                  need_redraw = true;
                  fp.fX1NDC = pal.fX2NDC + 0.05;
                  if (fp.fX2NDC < fp.fX1NDC + 0.1) fp.fX2NDC = Math.min(1, fp.fX1NDC + 0.1);
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
                  if (fp.fXYNDC < fp.fY1NDC + 0.1) fp.fY2NDC = Math.min(1, fp.fY1NDC + 0.1);
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

         const pr = !postpone_draw ? this.redraw() : Promise.resolve(true);
         return pr.then(() => {
             delete this.do_redraw_palette;
             return pal_painter;
         });
      });
   }

   /** @summary Toggle color z palette drawing */
   toggleColz() {
      if (this.options.canHavePalette()) {
         this.options.Zscale = !this.options.Zscale;
         return this.drawColorPalette(this.options.Zscale, false, true)
                    .then(() => this.processOnlineChange('drawopt'));
      }
   }

   /** @summary Toggle 3D drawing mode */
   toggleMode3D() {
      this.options.Mode3D = !this.options.Mode3D;

      if (this.options.Mode3D) {
         if (!this.options.Surf && !this.options.Lego && !this.options.Error) {
            if ((this.nbinsx >= 50) || (this.nbinsy >= 50))
               this.options.Lego = this.options.Scat ? 13 : 14;
            else
               this.options.Lego = this.options.Scat ? 1 : 12;

            this.options.Zero = false; // do not show zeros by default
         }
      }

      this.copyOptionsToOthers();
      return this.interactiveRedraw('pad', 'drawopt');
   }

   /** @summary Prepare handle for color draw */
   prepareDraw(args) {
      if (!args) args = { rounding: true, extra: 0, middle: 0 };

      if (args.extra === undefined) args.extra = 0;
      if (args.middle === undefined) args.middle = 0;

      const histo = this.getHisto(),
            xaxis = histo.fXaxis, yaxis = histo.fYaxis,
            pmain = this.getFramePainter(),
            hdim = this.getDimension(),
            res = {
               i1: args.nozoom ? 0 : this.getSelectIndex('x', 'left', 0 - args.extra),
               i2: args.nozoom ? this.nbinsx : this.getSelectIndex('x', 'right', 1 + args.extra),
               j1: (hdim === 1) ? 0 : (args.nozoom ? 0 : this.getSelectIndex('y', 'left', 0 - args.extra)),
               j2: (hdim === 1) ? 1 : (args.nozoom ? this.nbinsy : this.getSelectIndex('y', 'right', 1 + args.extra)),
               min: 0, max: 0, sumz: 0, xbar1: 0, xbar2: 1, ybar1: 0, ybar2: 1
            };

      if (args.cutg) {
         // if using cutg - define rectengular region
         let i1 = res.i2, i2 = res.i1, j1 = res.j2, j2 = res.j1;
         for (let ii = res.i1; ii < res.i2; ++ii) {
            for (let jj = res.j1; jj < res.j2; ++jj) {
               if (args.cutg.IsInside(xaxis.GetBinCoord(ii + args.middle), yaxis.GetBinCoord(jj + args.middle))) {
                  i1 = Math.min(i1, ii);
                  i2 = Math.max(i2, ii+1);
                  j1 = Math.min(j1, jj);
                  j2 = Math.max(j2, jj+1);
               }
            }
         }

         res.i1 = i1; res.i2 = i2; res.j1 = j1; res.j2 = j2;
      }

      let i, j, x, y, binz, binarea;

      res.grx = new Float32Array(res.i2+1);
      res.gry = new Float32Array(res.j2+1);

      if ((typeof histo.fBarOffset === 'number') && (typeof histo.fBarWidth === 'number') &&
           (histo.fBarOffset || histo.fBarWidth !== 1000)) {
             if (histo.fBarOffset <= 1000)
                res.xbar1 = res.ybar1 = 0.001*histo.fBarOffset;
             else if (histo.fBarOffset <= 3000)
                res.xbar1 = 0.001*(histo.fBarOffset-2000);
             else if (histo.fBarOffset <= 5000)
                res.ybar1 = 0.001*(histo.fBarOffset-4000);

             if (histo.fBarWidth <= 1000) {
                res.xbar2 = Math.min(1, res.xbar1 + 0.001*histo.fBarWidth);
                res.ybar2 = Math.min(1, res.ybar1 + 0.001*histo.fBarWidth);
             } else if (histo.fBarWidth <= 3000)
                res.xbar2 = Math.min(1, res.xbar1 + 0.001*(histo.fBarWidth-2000));
             else if (histo.fBarWidth <= 5000)
                res.ybar2 = Math.min(1, res.ybar1 + 0.001*(histo.fBarWidth-4000));
         }

      if (args.original) {
         res.original = true;
         res.origx = new Float32Array(res.i2+1);
         res.origy = new Float32Array(res.j2+1);
      }

      if (args.pixel_density) args.rounding = true;

      if (!pmain) {
         console.warn('cannot draw histogram without frame');
         return res;
      }

      const funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y);

      // calculate graphical coordinates in advance
      for (i = res.i1; i <= res.i2; ++i) {
         x = xaxis.GetBinCoord(i + args.middle);
         if (funcs.logx && (x <= 0)) { res.i1 = i+1; continue; }
         if (res.origx) res.origx[i] = x;
         res.grx[i] = funcs.grx(x);
         if (args.rounding) res.grx[i] = Math.round(res.grx[i]);

         if (args.use3d) {
            if (res.grx[i] < -pmain.size_x3d) {
               res.grx[i] = -pmain.size_x3d;
               if (this.options.RevX) res.i2 = i;
                                 else res.i1 = i;
            }
            if (res.grx[i] > pmain.size_x3d) {
               res.grx[i] = pmain.size_x3d;
               if (this.options.RevX) res.i1 = i;
                                 else res.i2 = i;
            }
         }
      }

      if (hdim === 1) {
         res.gry[0] = funcs.gry(0);
         res.gry[1] = funcs.gry(1);
      } else {
         for (j = res.j1; j <= res.j2; ++j) {
            y = yaxis.GetBinCoord(j + args.middle);
            if (funcs.logy && (y <= 0)) { res.j1 = j+1; continue; }
            if (res.origy) res.origy[j] = y;
            res.gry[j] = funcs.gry(y);
            if (args.rounding) res.gry[j] = Math.round(res.gry[j]);

            if (args.use3d) {
               if (res.gry[j] < -pmain.size_y3d) {
                  res.gry[j] = -pmain.size_y3d;
                  if (this.options.RevY) res.j2 = j;
                                    else res.j1 = j;
               }
               if (res.gry[j] > pmain.size_y3d) {
                  res.gry[j] = pmain.size_y3d;
                  if (this.options.RevY) res.j1 = j;
                                    else res.j2 = j;
               }
            }
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
               if ((binz > 0) && ((binz<res.min) || (res.min === 0))) res.min = binz;
               binz = binz/binarea;
            }
            if (this.maxbin === null)
               this.maxbin = this.minbin = binz;
             else {
               this.maxbin = Math.max(this.maxbin, binz);
               this.minbin = Math.min(this.minbin, binz);
            }
            if (binz > 0)
               if ((this.minposbin === null) || (binz < this.minposbin)) this.minposbin = binz;
         }
      }

      // force recalculation of z levels
      this.fContour = null;

      return res;
   }

   /** @summary Get tip text for axis bin */
   getAxisBinTip(name, axis, bin) {
      const pmain = this.getFramePainter(),
            funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
            handle = funcs[`${name}_handle`],
            x1 = axis.GetBinLowEdge(bin+1);

      if (handle.kind === kAxisLabels)
         return funcs.axisAsText(name, x1);

      const x2 = axis.GetBinLowEdge(bin+2);

      if ((handle.kind === kAxisTime) || this.isTF1())
         return funcs.axisAsText(name, (x1+x2)/2);

      return `[${funcs.axisAsText(name, x1)}, ${funcs.axisAsText(name, x2)})`;
   }

   /** @summary generic draw function for histograms
     * @private */
   static async _drawHist(painter, opt) {
      return ensureTCanvas(painter).then(() => {
         painter.setAsMainPainter();
         painter.decodeOptions(opt);

         if (painter.isTH2Poly()) {
            if (painter.options.Mode3D)
               painter.options.Lego = 12; // lego always 12
         }

         painter.checkPadRange();

         painter.scanContent();

         painter.createStat(); // only when required

         return painter.callDrawFunc();
      }).then(() => {
         return painter.drawFunctions();
      }).then(() => {
         return painter.drawHistTitle();
      }).then(() => {
         if (!painter.Mode3D && painter.options.AutoZoom)
            return painter.autoZoom();
      }).then(() => {
         if (painter.options.Project && !painter.mode3d && isFunc(painter.toggleProjection))
             return painter.toggleProjection(painter.options.Project);
      }).then(() => {
          painter.fillToolbar();
          return painter;
      });
   }

} // class THistPainter

export { THistPainter, FunctionsHandler, kNoZoom, HistContour, kCARTESIAN, kPOLAR, kCYLINDRICAL, kSPHERICAL, kRAPIDITY };
