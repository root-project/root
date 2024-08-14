import { gStyle, BIT, settings, create, createHistogram, setHistogramTitle, isFunc, isStr,
         clTPaveStats, clTCutG, clTH1I, clTH2I, clTF1, clTF2, clTPad, kNoZoom, kNoStats } from '../core.mjs';
import { select as d3_select } from '../d3.mjs';
import { DrawOptions, buildSvgCurve, makeTranslate, addHighlightStyle } from '../base/BasePainter.mjs';
import { ObjectPainter, kAxisNormal } from '../base/ObjectPainter.mjs';
import { FunctionsHandler } from './THistPainter.mjs';
import { TH1Painter, PadDrawOptions } from './TH1Painter.mjs';
import { kBlack, kWhite } from '../base/colors.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { assignContextMenu } from '../gui/menu.mjs';


const kNotEditable = BIT(18),   // bit set if graph is non editable
      clTGraphErrors = 'TGraphErrors',
      clTGraphAsymmErrors = 'TGraphAsymmErrors',
      clTGraphBentErrors = 'TGraphBentErrors',
      clTGraphMultiErrors = 'TGraphMultiErrors';

/**
 * @summary Painter for TGraph object.
 *
 * @private
 */


class TGraphPainter extends ObjectPainter {

   constructor(dom, graph) {
      super(dom, graph);
      this.axes_draw = false; // indicate if graph histogram was drawn for axes
      this.bins = null;
      this.xmin = this.ymin = this.xmax = this.ymax = 0;
      this.wheel_zoomy = true;
      this.is_bent = (graph._typename === clTGraphBentErrors);
      this.has_errors = (graph._typename === clTGraphErrors) ||
                        (graph._typename === clTGraphMultiErrors) ||
                        (graph._typename === clTGraphAsymmErrors) ||
                         this.is_bent || graph._typename.match(/^RooHist/);
   }

   /** @summary Return drawn graph object */
   getGraph() { return this.getObject(); }

   /** @summary Return histogram object used for axis drawings */
   getHistogram() { return this.getObject()?.fHistogram; }

   /** @summary Set histogram object to graph */
   setHistogram(histo) {
      const obj = this.getObject();
      if (obj) obj.fHistogram = histo;
   }

   /** @summary Redraw graph
     * @desc may redraw histogram which was used to draw axes
     * @return {Promise} for ready */
   async redraw() {
      let promise = Promise.resolve(true);

      if (this.$redraw_hist) {
         delete this.$redraw_hist;
         const hist_painter = this.getMainPainter();
         if (hist_painter?.isSecondary(this) && this.axes_draw)
            promise = hist_painter.redraw();
      }

      return promise.then(() => this.drawGraph()).then(() => {
         const res = this._funcHandler?.drawNext(0) ?? this;
         delete this._funcHandler;
         return res;
      });
   }

   /** @summary Cleanup graph painter */
   cleanup() {
      delete this.interactive_bin; // break mouse handling
      delete this.bins;
      super.cleanup();
   }

   /** @summary Returns object if this drawing TGraphMultiErrors object */
   get_gme() {
      const graph = this.getGraph();
      return graph?._typename === clTGraphMultiErrors ? graph : null;
   }

   /** @summary Decode options */
   decodeOptions(opt, first_time) {
      if (isStr(opt) && (opt.indexOf('same ') === 0))
         opt = opt.slice(5);

      const graph = this.getGraph(),
          is_gme = !!this.get_gme(),
          has_main = first_time ? !!this.getMainPainter() : !this.axes_draw;
      let blocks_gme = [];

      if (!this.options) this.options = {};

      // decode main draw options for the graph
      const decodeBlock = (d, res) => {
         Object.assign(res, { Line: 0, Curve: 0, Rect: 0, Mark: 0, Bar: 0, OutRange: 0, EF: 0, Fill: 0, MainError: 1, Ends: 1, ScaleErrX: 1 });

         if (is_gme && d.check('S=', true)) res.ScaleErrX = d.partAsFloat();

         if (d.check('L')) res.Line = 1;
         if (d.check('F')) res.Fill = 1;
         if (d.check('CC')) res.Curve = 2; // draw all points without reduction
         if (d.check('C')) res.Curve = 1;
         if (d.check('*')) res.Mark = 103;
         if (d.check('P0')) res.Mark = 104;
         if (d.check('P')) res.Mark = 1;
         if (d.check('B')) { res.Bar = 1; res.Errors = 0; }
         if (d.check('Z')) { res.Errors = 1; res.Ends = 0; }
         if (d.check('||')) { res.Errors = 1; res.MainError = 0; res.Ends = 1; }
         if (d.check('[]')) { res.Errors = 1; res.MainError = 0; res.Ends = 2; }
         if (d.check('|>')) { res.Errors = 1; res.Ends = 3; }
         if (d.check('>')) { res.Errors = 1; res.Ends = 4; }
         if (d.check('0')) { res.Mark = 1; res.Errors = 1; res.OutRange = 1; }
         if (d.check('1')) if (res.Bar === 1) res.Bar = 2;
         if (d.check('2')) { res.Rect = 1; res.Errors = 0; }
         if (d.check('3')) { res.EF = 1; res.Errors = 0; }
         if (d.check('4')) { res.EF = 2; res.Errors = 0; }
         if (d.check('5')) { res.Rect = 2; res.Errors = 0; }
         if (d.check('X')) res.Errors = 0;
      };

      Object.assign(this.options, { Axis: '', NoOpt: 0, PadStats: false, PadPalette: false, original: opt, second_x: false, second_y: false, individual_styles: false });

      if (is_gme && opt) {
         if (opt.indexOf(';') > 0) {
            blocks_gme = opt.split(';');
            opt = blocks_gme.shift();
         } else if (opt.indexOf('_') > 0) {
            blocks_gme = opt.split('_');
            opt = blocks_gme.shift();
         }
      }

      const res = this.options;
      let d = new DrawOptions(opt), hopt = '';

      PadDrawOptions.forEach(name => { if (d.check(name)) hopt += ';' + name; });
      if (d.check('XAXIS_', true)) hopt += ';XAXIS_' + d.part;
      if (d.check('YAXIS_', true)) hopt += ';YAXIS_' + d.part;

      if (d.empty()) {
         res.original = has_main ? 'lp' : 'alp';
         d = new DrawOptions(res.original);
      }

      if (d.check('NOOPT')) res.NoOpt = 1;

      if (d.check('POS3D_', true)) res.pos3d = d.partAsInt() - 0.5;

      if (d.check('PFC') && !res._pfc)
         res._pfc = 2;
      if (d.check('PLC') && !res._plc)
         res._plc = 2;
      if (d.check('PMC') && !res._pmc)
         res._pmc = 2;

      if (d.check('A')) res.Axis = d.check('I') ? 'A;' : ' '; // I means invisible axis
      if (d.check('X+')) { res.Axis += 'X+'; res.second_x = has_main; }
      if (d.check('Y+')) { res.Axis += 'Y+'; res.second_y = has_main; }
      if (d.check('RX')) res.Axis += 'RX';
      if (d.check('RY')) res.Axis += 'RY';

      if (is_gme) {
         res.blocks = [];
         res.skip_errors_x0 = res.skip_errors_y0 = false;
         if (d.check('X0')) res.skip_errors_x0 = true;
         if (d.check('Y0')) res.skip_errors_y0 = true;
      }

      decodeBlock(d, res);

      if (is_gme)
         if (d.check('S')) res.individual_styles = true;


      // if (d.check('E')) res.Errors = 1; // E option only defined for TGraphPolar

      if (res.Errors === undefined)
         res.Errors = this.has_errors && (!is_gme || !blocks_gme.length) ? 1 : 0;

      // special case - one could use svg:path to draw many pixels (
      if ((res.Mark === 1) && (graph.fMarkerStyle === 1)) res.Mark = 101;

      // if no drawing option is selected and if opt === '' nothing is done.
      if (res.Line + res.Fill + res.Curve + res.Mark + res.Bar + res.EF + res.Rect + res.Errors === 0)
         if (d.empty()) res.Line = 1;


      if (this.matchObjectType(clTGraphErrors)) {
         const len = graph.fEX.length;
         let m = 0;
         for (let k = 0; k < len; ++k)
            m = Math.max(m, graph.fEX[k], graph.fEY[k]);
         if (m < 1e-100)
            res.Errors = 0;
      }

      this._cutg = this.matchObjectType(clTCutG);
      this._cutg_lastsame = this._cutg && (graph.fNpoints > 3) &&
                            (graph.fX[0] === graph.fX[graph.fNpoints-1]) && (graph.fY[0] === graph.fY[graph.fNpoints-1]);

      if (!res.Axis) {
         // check if axis should be drawn
         // either graph drawn directly or
         // graph is first object in list of primitives
         const pad = this.getPadPainter()?.getRootPad(true);
         if (!pad || (pad?.fPrimitives?.arr[0] === this.getObject())) res.Axis = ' ';
      }

      res.Axis += hopt;

      for (let bl = 0; bl < blocks_gme.length; ++bl) {
         const subd = new DrawOptions(blocks_gme[bl]), subres = {};
         decodeBlock(subd, subres);
         subres.skip_errors_x0 = res.skip_errors_x0;
         subres.skip_errors_y0 = res.skip_errors_y0;
         res.blocks.push(subres);
      }
   }

   /** @summary Extract errors for TGraphMultiErrors */
   extractGmeErrors(nblock) {
      if (!this.bins) return;
      const gr = this.getGraph();
      this.bins.forEach(bin => {
         bin.eylow = gr.fEyL[nblock][bin.indx];
         bin.eyhigh = gr.fEyH[nblock][bin.indx];
      });
   }

   /** @summary Create bins for TF1 drawing */
   createBins() {
      const gr = this.getGraph();
      if (!gr) return;

      let kind = 0, npoints = gr.fNpoints;
      if (this._cutg && this._cutg_lastsame)
         npoints--;

      if (gr._typename === clTGraphErrors)
         kind = 1;
      else if (gr._typename === clTGraphMultiErrors)
         kind = 2;
      else if (gr._typename === clTGraphAsymmErrors || gr._typename === clTGraphBentErrors || gr._typename.match(/^RooHist/))
         kind = 3;

      this.bins = new Array(npoints);

      for (let p = 0; p < npoints; ++p) {
         const bin = this.bins[p] = { x: gr.fX[p], y: gr.fY[p], indx: p };
         switch (kind) {
            case 1:
               bin.exlow = bin.exhigh = gr.fEX[p];
               bin.eylow = bin.eyhigh = gr.fEY[p];
               break;
            case 2:
               bin.exlow = gr.fExL[p];
               bin.exhigh = gr.fExH[p];
               bin.eylow = gr.fEyL[0][p];
               bin.eyhigh = gr.fEyH[0][p];
               break;
            case 3:
               bin.exlow = gr.fEXlow[p];
               bin.exhigh = gr.fEXhigh[p];
               bin.eylow = gr.fEYlow[p];
               bin.eyhigh = gr.fEYhigh[p];
               break;
         }

         if (p === 0) {
            this.xmin = this.xmax = bin.x;
            this.ymin = this.ymax = bin.y;
         }

         if (kind > 0) {
            this.xmin = Math.min(this.xmin, bin.x - bin.exlow, bin.x + bin.exhigh);
            this.xmax = Math.max(this.xmax, bin.x - bin.exlow, bin.x + bin.exhigh);
            this.ymin = Math.min(this.ymin, bin.y - bin.eylow, bin.y + bin.eyhigh);
            this.ymax = Math.max(this.ymax, bin.y - bin.eylow, bin.y + bin.eyhigh);
         } else {
            this.xmin = Math.min(this.xmin, bin.x);
            this.xmax = Math.max(this.xmax, bin.x);
            this.ymin = Math.min(this.ymin, bin.y);
            this.ymax = Math.max(this.ymax, bin.y);
         }
      }
   }

   /** @summary Return margins for histogram ranges */
   getHistRangeMargin() { return 0.1; }

   /** @summary Create histogram for graph
     * @desc graph bins should be created when calling this function
     * @param {boolean} [set_x] - set X axis range
     * @param {boolean} [set_y] - set Y axis range */
   createHistogram(set_x, set_y) {
      if (!set_x && !set_y)
         set_x = set_y = true;

      const graph = this.getGraph(),
            xmin = this.xmin,
            margin = this.getHistRangeMargin();
      let xmax = this.xmax, ymin = this.ymin, ymax = this.ymax;

      if (xmin >= xmax) xmax = xmin + 1;
      if (ymin >= ymax) ymax = ymin + 1;
      const dx = (xmax - xmin) * margin, dy = (ymax - ymin) * margin;
      let uxmin = xmin - dx, uxmax = xmax + dx,
          minimum = ymin - dy, maximum = ymax + dy;

      if (!this._not_adjust_hrange) {
         const pad_logx = this.getPadPainter()?.getPadLog('x');

         if ((uxmin < 0) && (xmin >= 0))
            uxmin = pad_logx ? xmin * (1 - margin) : 0;
         if ((uxmax > 0) && (xmax <= 0))
            uxmax = pad_logx ? (1 + margin) * xmax : 0;
      }

      const minimum0 = minimum, maximum0 = maximum;
      let histo = this.getHistogram();

      if (!histo) {
         histo = this._need_2dhist ? createHistogram(clTH2I, 30, 30) : createHistogram(clTH1I, 100);
         histo.fName = graph.fName + '_h';
         histo.fBits |= kNoStats;
         this._own_histogram = true;
         this.setHistogram(histo);
      } else if ((histo.fMaximum !== kNoZoom) && (histo.fMinimum !== kNoZoom)) {
         minimum = histo.fMinimum;
         maximum = histo.fMaximum;
      }

      if (graph.fMinimum !== kNoZoom) minimum = ymin = graph.fMinimum;
      if (graph.fMaximum !== kNoZoom) maximum = graph.fMaximum;
      if ((minimum < 0) && (ymin >= 0)) minimum = (1 - margin)*ymin;

      setHistogramTitle(histo, this.getObject().fTitle);

      if (set_x && !histo.fXaxis.fLabels) {
         histo.fXaxis.fXmin = uxmin;
         histo.fXaxis.fXmax = uxmax;
      }

      if (set_y && !histo.fYaxis.fLabels) {
         histo.fYaxis.fXmin = Math.min(minimum0, minimum);
         histo.fYaxis.fXmax = Math.max(maximum0, maximum);
         if (!this._need_2dhist) {
            histo.fMinimum = minimum;
            histo.fMaximum = maximum;
         }
      }

      return histo;
   }

   /** @summary Check if user range can be unzommed
     * @desc Used when graph points covers larger range than provided histogram */
   unzoomUserRange(dox, doy /*, doz */) {
      const graph = this.getGraph();
      if (this._own_histogram || !graph) return false;

      const histo = this.getHistogram();

      dox = dox && histo && ((histo.fXaxis.fXmin > this.xmin) || (histo.fXaxis.fXmax < this.xmax));
      doy = doy && histo && ((histo.fYaxis.fXmin > this.ymin) || (histo.fYaxis.fXmax < this.ymax));
      if (!dox && !doy) return false;

      this.createHistogram(dox, doy);
      this.getMainPainter()?.extractAxesProperties(1); // just to enforce ranges extraction

      return true;
   }

   /** @summary Returns true if graph drawing can be optimize */
   canOptimize() {
      return (settings.OptimizeDraw > 0) && !this.options.NoOpt;
   }

   /** @summary Returns optimized bins - if optimization enabled */
   optimizeBins(maxpnt, filter_func) {
      if ((this.bins.length < 30) && !filter_func)
         return this.bins;

      let selbins = null;
      if (isFunc(filter_func)) {
         for (let n = 0; n < this.bins.length; ++n) {
            if (filter_func(this.bins[n], n)) {
               if (!selbins) selbins = (n === 0) ? [] : this.bins.slice(0, n);
            } else
               if (selbins) selbins.push(this.bins[n]);
         }
      }
      if (!selbins) selbins = this.bins;

      if (!maxpnt) maxpnt = 500000;

      if ((selbins.length < maxpnt) || !this.canOptimize()) return selbins;
      let step = Math.floor(selbins.length / maxpnt);
      if (step < 2) step = 2;
      const optbins = [];
      for (let n = 0; n < selbins.length; n+=step)
         optbins.push(selbins[n]);

      return optbins;
   }

   /** @summary Check if such function should be drawn directly */
   needDrawFunc(graph, func) {
      if (func._typename === clTPaveStats)
          return (func.fName !== 'stats') || !graph.TestBit(kNoStats); // kNoStats is same for graph and histogram

       if ((func._typename === clTF1) || (func._typename === clTF2))
          return !func.TestBit(BIT(9)); // TF1::kNotDraw

       return true;
   }

   /** @summary Returns tooltip for specified bin */
   getTooltips(d) {
      const pmain = this.get_main(), lines = [],
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          gme = this.get_gme();

      lines.push(this.getObjectHint());

      if (d && funcs) {
         if (d.indx !== undefined)
            lines.push('p = ' + d.indx);
         lines.push('x = ' + funcs.axisAsText('x', d.x), 'y = ' + funcs.axisAsText('y', d.y));
         if (gme)
            lines.push('error x = -' + funcs.axisAsText('x', gme.fExL[d.indx]) + '/+' + funcs.axisAsText('x', gme.fExH[d.indx]));
         else if (this.options.Errors && (funcs.x_handle.kind === kAxisNormal) && (d.exlow || d.exhigh))
            lines.push('error x = -' + funcs.axisAsText('x', d.exlow) + '/+' + funcs.axisAsText('x', d.exhigh));

         if (gme) {
            for (let ny = 0; ny < gme.fNYErrors; ++ny)
               lines.push(`error y${ny} = -${funcs.axisAsText('y', gme.fEyL[ny][d.indx])}/+${funcs.axisAsText('y', gme.fEyH[ny][d.indx])}`);
         } else if ((this.options.Errors || (this.options.EF > 0)) && (funcs.y_handle.kind === kAxisNormal) && (d.eylow || d.eyhigh))
            lines.push('error y = -' + funcs.axisAsText('y', d.eylow) + '/+' + funcs.axisAsText('y', d.eyhigh));
      }
      return lines;
   }

   /** @summary Provide frame painter for graph
     * @desc If not exists, emulate its behaviour */
   get_main() {
      let pmain = this.getFramePainter();

      if (pmain?.grx && pmain?.gry) return pmain;

      // FIXME: check if needed, can be removed easily
      const pp = this.getPadPainter(),
            rect = pp?.getPadRect() || { width: 800, height: 600 };

      pmain = {
          pad_layer: true,
          pad: pp?.getRootPad(true) ?? create(clTPad),
          pw: rect.width,
          ph: rect.height,
          fX1NDC: 0.1, fX2NDC: 0.9, fY1NDC: 0.1, fY2NDC: 0.9,
          getFrameWidth() { return this.pw; },
          getFrameHeight() { return this.ph; },
          grx(value) {
             if (this.pad.fLogx)
                value = (value > 0) ? Math.log10(value) : this.pad.fUxmin;
             else
                value = (value - this.pad.fX1) / (this.pad.fX2 - this.pad.fX1);
             return value * this.pw;
          },
          gry(value) {
             if (this.pad.fLogv ?? this.pad.fLogy)
                value = (value > 0) ? Math.log10(value) : this.pad.fUymin;
             else
                value = (value - this.pad.fY1) / (this.pad.fY2 - this.pad.fY1);
             return (1 - value) * this.ph;
          },
          revertAxis(name, v) {
            if (name === 'x')
               return v / this.pw * (this.pad.fX2 - this.pad.fX1) + this.pad.fX1;
            if (name === 'y')
               return (1 - v / this.ph) * (this.pad.fY2 - this.pad.fY1) + this.pad.fY1;
            return v;
          },
          getGrFuncs() { return this; }
      };

      return pmain.pad ? pmain : null;
   }

   /** @summary append exclusion area to created path */
   appendExclusion(is_curve, path, drawbins, excl_width) {
      const extrabins = [];
      for (let n = drawbins.length-1; n >= 0; --n) {
         const bin = drawbins[n],
             dlen = Math.sqrt(bin.dgrx**2 + bin.dgry**2);
         if (dlen > 1e-10) {
            // shift point
            bin.grx += excl_width*bin.dgry/dlen;
            bin.gry -= excl_width*bin.dgrx/dlen;
         }
         extrabins.push(bin);
      }

      const path2 = buildSvgCurve(extrabins, { cmd: 'L', line: !is_curve });

      this.draw_g.append('svg:path')
                 .attr('d', path + path2 + 'Z')
                 .call(this.fillatt.func)
                 .style('opacity', 0.75);
   }

   /** @summary draw TGraph bins with specified options
     * @desc Can be called several times */
   drawBins(funcs, options, draw_g, w, h, lineatt, fillatt, main_block) {
      const graph = this.getGraph();
      if (!graph?.fNpoints) return;

      let excl_width = 0, drawbins = null;

      if (main_block && lineatt.excl_side) {
         excl_width = lineatt.excl_width;
         if ((lineatt.width > 0) && !options.Line && !options.Curve) options.Line = 1;
      }

      if (options.EF) {
         drawbins = this.optimizeBins((options.EF > 1) ? 20000 : 0);

         // build lower part
         for (let n = 0; n < drawbins.length; ++n) {
            const bin = drawbins[n];
            bin.grx = funcs.grx(bin.x);
            bin.gry = funcs.gry(bin.y - bin.eylow);
         }

         const path1 = buildSvgCurve(drawbins, { line: options.EF < 2, qubic: true }),
             bins2 = [];

         for (let n = drawbins.length-1; n >= 0; --n) {
            const bin = drawbins[n];
            bin.gry = funcs.gry(bin.y + bin.eyhigh);
            bins2.push(bin);
         }

         // build upper part (in reverse direction)
         const path2 = buildSvgCurve(bins2, { line: options.EF < 2, cmd: 'L', qubic: true }),
            area = draw_g.append('svg:path')
               .attr('d', path1 + path2 + 'Z')
               .call(fillatt.func);

         // Let behaves as ROOT - see JIRA ROOT-8131
         if (fillatt.empty() && fillatt.colorindx)
            area.style('stroke', this.getColor(fillatt.colorindx));
         if (main_block)
            this.draw_kind = 'lines';
      }

      if (options.Line || options.Fill) {
         let close_symbol = '';
         if (this._cutg) {
            close_symbol = 'Z';
            if (!options.original) options.Fill = 1;
         }

         if (options.Fill) {
            close_symbol = 'Z'; // always close area if we want to fill it
            excl_width = 0;
         }

         if (!drawbins) drawbins = this.optimizeBins(0);

         for (let n = 0; n < drawbins.length; ++n) {
            const bin = drawbins[n];
            bin.grx = funcs.grx(bin.x);
            bin.gry = funcs.gry(bin.y);
         }

         const path = buildSvgCurve(drawbins, { line: true, calc: excl_width });

         if (excl_width)
             this.appendExclusion(false, path, drawbins, excl_width);

         const elem = draw_g.append('svg:path').attr('d', path + close_symbol);
         if (options.Line)
            elem.call(lineatt.func);

         if (options.Fill)
            elem.call(fillatt.func);
         else
            elem.style('fill', 'none');

         if (main_block)
            this.draw_kind = 'lines';
      }

      if (options.Curve) {
         let curvebins = drawbins;
         if ((this.draw_kind !== 'lines') || !curvebins || ((options.Curve === 1) && (curvebins.length > 20000))) {
            curvebins = this.optimizeBins((options.Curve === 1) ? 20000 : 0);
            for (let n = 0; n < curvebins.length; ++n) {
               const bin = curvebins[n];
               bin.grx = funcs.grx(bin.x);
               bin.gry = funcs.gry(bin.y);
            }
         }

         const path = buildSvgCurve(curvebins, { qubic: !excl_width });
         if (excl_width)
            this.appendExclusion(true, path, curvebins, excl_width);

         draw_g.append('svg:path')
               .attr('d', path)
               .call(lineatt.func)
               .style('fill', 'none');
         if (main_block)
            this.draw_kind = 'lines'; // handled same way as lines
      }

      let nodes = null;

      if (options.Errors || options.Rect || options.Bar) {
         drawbins = this.optimizeBins(5000, (pnt, i) => {
            const grx = funcs.grx(pnt.x);

            // when drawing bars, take all points
            if (!options.Bar && ((grx < 0) || (grx > w))) return true;

            const gry = funcs.gry(pnt.y);

            if (!options.Bar && !options.OutRange && ((gry < 0) || (gry > h))) return true;

            pnt.grx1 = Math.round(grx);
            pnt.gry1 = Math.round(gry);

            if (this.has_errors) {
               pnt.grx0 = Math.round(funcs.grx(pnt.x - options.ScaleErrX*pnt.exlow) - grx);
               pnt.grx2 = Math.round(funcs.grx(pnt.x + options.ScaleErrX*pnt.exhigh) - grx);
               pnt.gry0 = Math.round(funcs.gry(pnt.y - pnt.eylow) - gry);
               pnt.gry2 = Math.round(funcs.gry(pnt.y + pnt.eyhigh) - gry);

               if (this.is_bent) {
                  pnt.grdx0 = Math.round(funcs.gry(pnt.y + graph.fEXlowd[i]) - gry);
                  pnt.grdx2 = Math.round(funcs.gry(pnt.y + graph.fEXhighd[i]) - gry);
                  pnt.grdy0 = Math.round(funcs.grx(pnt.x + graph.fEYlowd[i]) - grx);
                  pnt.grdy2 = Math.round(funcs.grx(pnt.x + graph.fEYhighd[i]) - grx);
               } else
                  pnt.grdx0 = pnt.grdx2 = pnt.grdy0 = pnt.grdy2 = 0;
            }

            return false;
         });

         if (main_block)
            this.draw_kind = 'nodes';

         nodes = draw_g.selectAll('.grpoint')
                       .data(drawbins)
                       .enter()
                       .append('svg:g')
                       .attr('class', 'grpoint')
                       .attr('transform', d => makeTranslate(d.grx1, d.gry1));
      }

      if (options.Bar) {
         // calculate bar width

         let xmin = 0, xmax = 0;
         for (let i = 0; i < drawbins.length; ++i) {
            if (i === 0)
               xmin = xmax = drawbins[i].grx1;
            else {
               xmin = Math.min(xmin, drawbins[i].grx1);
               xmax = Math.max(xmax, drawbins[i].grx1);
            }
         }

         if (drawbins.length === 1)
            drawbins[0].width = w/4; // pathologic case of single bin
         else {
            for (let i = 0; i < drawbins.length; ++i)
               drawbins[i].width = (xmax - xmin) / drawbins.length * gStyle.fBarWidth;
         }

         const yy0 = Math.round(funcs.gry(0));
         let usefill = fillatt;

         if (main_block) {
            const fp = this.getFramePainter(),
                  fpcol = !fp?.fillatt?.empty() ? fp.fillatt.getFillColor() : -1;

            if (fpcol === fillatt.getFillColor())
               usefill = this.createAttFill({ color: fpcol === 'white' ? kBlack : kWhite, pattern: 1001, std: false });
         }

         nodes.append('svg:path')
              .attr('d', d => {
                 d.bar = true; // element drawn as bar
                 const dx = d.width > 1 ? Math.round(-d.width/2) : 0,
                       dw = d.width > 1 ? Math.round(d.width) : 1,
                       dy = (options.Bar !== 1) ? 0 : ((d.gry1 > yy0) ? yy0-d.gry1 : 0),
                       dh = (options.Bar !== 1) ? (h > d.gry1 ? h - d.gry1 : 0) : Math.abs(yy0 - d.gry1);
                 return `M${dx},${dy}h${dw}v${dh}h${-dw}z`;
              })
            .call(usefill.func);
      }

      if (options.Rect) {
         nodes.filter(d => (d.exlow > 0) && (d.exhigh > 0) && (d.eylow > 0) && (d.eyhigh > 0))
           .append('svg:path')
           .attr('d', d => {
               d.rect = true;
               return `M${d.grx0},${d.gry0}H${d.grx2}V${d.gry2}H${d.grx0}Z`;
            })
           .call(fillatt.func)
           .call(options.Rect === 2 ? lineatt.func : () => {});
      }

      this.error_size = 0;

      if (options.Errors) {
         // to show end of error markers, use line width attribute
         let lw = lineatt.width + gStyle.fEndErrorSize, bb = 0;
         const vv = options.Ends ? `m0,${lw}v${-2*lw}` : '',
               hh = options.Ends ? `m${lw},0h${-2*lw}` : '';
         let vleft = vv, vright = vv, htop = hh, hbottom = hh;

         const mainLine = (dx, dy) => {
            if (!options.MainError) return `M${dx},${dy}`;
            const res = 'M0,0';
            if (dx) return res + (dy ? `L${dx},${dy}` : `H${dx}`);
            return dy ? res + `V${dy}` : res;
         };

         switch (options.Ends) {
            case 2:  // option []
               bb = Math.max(lineatt.width+1, Math.round(lw*0.66));
               vleft = `m${bb},${lw}h${-bb}v${-2*lw}h${bb}`;
               vright = `m${-bb},${lw}h${bb}v${-2*lw}h${-bb}`;
               htop = `m${-lw},${bb}v${-bb}h${2*lw}v${bb}`;
               hbottom = `m${-lw},${-bb}v${bb}h${2*lw}v${-bb}`;
               break;
            case 3: // option |>
               lw = Math.max(lw, Math.round(graph.fMarkerSize*8*0.66));
               bb = Math.max(lineatt.width+1, Math.round(lw*0.66));
               vleft = `l${bb},${lw}v${-2*lw}l${-bb},${lw}`;
               vright = `l${-bb},${lw}v${-2*lw}l${bb},${lw}`;
               htop = `l${-lw},${bb}h${2*lw}l${-lw},${-bb}`;
               hbottom = `l${-lw},${-bb}h${2*lw}l${-lw},${bb}`;
               break;
            case 4: // option >
               lw = Math.max(lw, Math.round(graph.fMarkerSize*8*0.66));
               bb = Math.max(lineatt.width+1, Math.round(lw*0.66));
               vleft = `l${bb},${lw}m0,${-2*lw}l${-bb},${lw}`;
               vright = `l${-bb},${lw}m0,${-2*lw}l${bb},${lw}`;
               htop = `l${-lw},${bb}m${2*lw},0l${-lw},${-bb}`;
               hbottom = `l${-lw},${-bb}m${2*lw},0l${-lw},${bb}`;
               break;
         }

         this.error_size = lw;

         lw = Math.floor((lineatt.width-1)/2); // one should take into account half of end-cup line width

         let visible = nodes.filter(d => (d.exlow > 0) || (d.exhigh > 0) || (d.eylow > 0) || (d.eyhigh > 0));
         if (options.skip_errors_x0 || options.skip_errors_y0)
            visible = visible.filter(d => ((d.x !== 0) || !options.skip_errors_x0) && ((d.y !== 0) || !options.skip_errors_y0));

         if (!this.isBatchMode() && settings.Tooltip && main_block) {
            visible.append('svg:path')
                   .style('fill', 'none')
                   .style('pointer-events', 'visibleFill')
                   .attr('d', d => `M${d.grx0},${d.gry0}h${d.grx2-d.grx0}v${d.gry2-d.gry0}h${d.grx0-d.grx2}z`);
         }

         visible.append('svg:path')
             .call(lineatt.func)
             .style('fill', 'none')
             .attr('d', d => {
                d.error = true;
                return ((d.exlow > 0) ? mainLine(d.grx0+lw, d.grdx0) + vleft : '') +
                       ((d.exhigh > 0) ? mainLine(d.grx2-lw, d.grdx2) + vright : '') +
                       ((d.eylow > 0) ? mainLine(d.grdy0, d.gry0-lw) + hbottom : '') +
                       ((d.eyhigh > 0) ? mainLine(d.grdy2, d.gry2+lw) + htop : '');
              });
      }

      if (options.Mark) {
         // for tooltips use markers only if nodes were not created
         this.createAttMarker({ attr: graph, style: options.Mark - 100 });

         this.marker_size = this.markeratt.getFullSize();

         this.markeratt.resetPos();

         const want_tooltip = !this.isBatchMode() && settings.Tooltip && (!this.markeratt.fill || (this.marker_size < 7)) && !nodes && main_block,
               hsz = Math.max(5, Math.round(this.marker_size*0.7)),
               maxnummarker = 1000000 / (this.markeratt.getMarkerLength() + 7); // let produce SVG at maximum 1MB

         let path = '', pnt, grx, gry,
             hints_marker = '', step = 1;

         if (!drawbins)
            drawbins = this.optimizeBins(maxnummarker);
         else if (this.canOptimize() && (drawbins.length > 1.5*maxnummarker))
            step = Math.min(2, Math.round(drawbins.length/maxnummarker));

         for (let n = 0; n < drawbins.length; n += step) {
            pnt = drawbins[n];
            grx = funcs.grx(pnt.x);
            if ((grx > -this.marker_size) && (grx < w + this.marker_size)) {
               gry = funcs.gry(pnt.y);
               if ((gry > -this.marker_size) && (gry < h + this.marker_size)) {
                  path += this.markeratt.create(grx, gry);
                  if (want_tooltip) hints_marker += `M${grx-hsz},${gry-hsz}h${2*hsz}v${2*hsz}h${-2*hsz}z`;
               }
            }
         }

         if (path) {
            draw_g.append('svg:path')
                  .attr('d', path)
                  .call(this.markeratt.func);
            if ((nodes === null) && (this.draw_kind === 'none') && main_block)
               this.draw_kind = (options.Mark === 101) ? 'path' : 'mark';
         }
         if (want_tooltip && hints_marker) {
            draw_g.append('svg:path')
                  .attr('d', hints_marker)
                  .style('fill', 'none')
                  .style('pointer-events', 'visibleFill');
         }
      }
   }

   /** @summary append TGraphQQ part */
   appendQQ(funcs, graph) {
      const xqmin = Math.max(funcs.scale_xmin, graph.fXq1),
            xqmax = Math.min(funcs.scale_xmax, graph.fXq2),
            yqmin = Math.max(funcs.scale_ymin, graph.fYq1),
            yqmax = Math.min(funcs.scale_ymax, graph.fYq2),
            makeLine = (x1, y1, x2, y2) => `M${funcs.grx(x1)},${funcs.gry(y1)}L${funcs.grx(x2)},${funcs.gry(y2)}`,
            yxmin = (graph.fYq2 - graph.fYq1)*(funcs.scale_xmin-graph.fXq1)/(graph.fXq2-graph.fXq1) + graph.fYq1,
            yxmax = (graph.fYq2-graph.fYq1)*(funcs.scale_xmax-graph.fXq1)/(graph.fXq2-graph.fXq1) + graph.fYq1;

      let path2 = '';
      if (yxmin < funcs.scale_ymin) {
         const xymin = (graph.fXq2 - graph.fXq1)*(funcs.scale_ymin-graph.fYq1)/(graph.fYq2-graph.fYq1) + graph.fXq1;
         path2 = makeLine(xymin, funcs.scale_ymin, xqmin, yqmin);
      } else
         path2 = makeLine(funcs.scale_xmin, yxmin, xqmin, yqmin);


      if (yxmax > funcs.scale_ymax) {
         const xymax = (graph.fXq2-graph.fXq1)*(funcs.scale_ymax-graph.fYq1)/(graph.fYq2-graph.fYq1) + graph.fXq1;
         path2 += makeLine(xqmax, yqmax, xymax, funcs.scale_ymax);
      } else
         path2 += makeLine(xqmax, yqmax, funcs.scale_xmax, yxmax);


      const latt1 = this.createAttLine({ style: 1, width: 1, color: kBlack, std: false }),
            latt2 = this.createAttLine({ style: 2, width: 1, color: kBlack, std: false });

      this.draw_g.append('path')
                 .attr('d', makeLine(xqmin, yqmin, xqmax, yqmax))
                 .call(latt1.func)
                 .style('fill', 'none');

      this.draw_g.append('path')
                 .attr('d', path2)
                 .call(latt2.func)
                 .style('fill', 'none');
   }

   drawBins3D(/* fp, graph */) {
      console.log('Load ./hist/TGraphPainter.mjs to draw graph in 3D');
   }

   /** @summary Create necessary histogram draw attributes */
   createGraphDrawAttributes(only_check_auto) {
      const graph = this.getGraph(), o = this.options;
      if (o._pfc > 1 || o._plc > 1 || o._pmc > 1) {
         const pp = this.getPadPainter();
         if (isFunc(pp?.getAutoColor)) {
            const icolor = pp.getAutoColor(graph.$num_graphs);
            this._auto_exec = ''; // can be reused when sending option back to server
            if (o._pfc > 1) { o._pfc = 1; graph.fFillColor = icolor; this._auto_exec += `SetFillColor(${icolor});;`; delete this.fillatt; }
            if (o._plc > 1) { o._plc = 1; graph.fLineColor = icolor; this._auto_exec += `SetLineColor(${icolor});;`; delete this.lineatt; }
            if (o._pmc > 1) { o._pmc = 1; graph.fMarkerColor = icolor; this._auto_exec += `SetMarkerColor(${icolor});;`; delete this.markeratt; }
         }
      }

      if (only_check_auto)
         this.deleteAttr();
      else {
         this.createAttLine({ attr: graph, can_excl: true });
         this.createAttFill({ attr: graph });
      }
   }

   /** @summary draw TGraph */
   drawGraph() {
      const pmain = this.get_main(),
            graph = this.getGraph();
      if (!pmain) return;

      // special mode for TMultiGraph 3d drawing
      if (this.options.pos3d)
         return this.drawBins3D(pmain, graph);

      const is_gme = !!this.get_gme(),
            funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
            w = pmain.getFrameWidth(),
            h = pmain.getFrameHeight();

      this.createG(!pmain.pad_layer);

      this.createGraphDrawAttributes();

      this.fillatt.used = false; // mark used only when really used

      this.draw_kind = 'none'; // indicate if special svg:g were created for each bin
      this.marker_size = 0; // indicate if markers are drawn
      const draw_g = is_gme ? this.draw_g.append('svg:g') : this.draw_g;

      this.drawBins(funcs, this.options, draw_g, w, h, this.lineatt, this.fillatt, true);

      if (graph._typename === 'TGraphQQ')
         this.appendQQ(funcs, graph);

      if (is_gme) {
         for (let k = 0; k < graph.fNYErrors; ++k) {
            let lineatt = this.lineatt, fillatt = this.fillatt;
            if (this.options.individual_styles) {
               lineatt = this.createAttLine({ attr: graph.fAttLine[k], std: false });
               fillatt = this.createAttFill({ attr: graph.fAttFill[k], std: false });
            }
            const sub_g = this.draw_g.append('svg:g'),
                options = (k < this.options.blocks.length) ? this.options.blocks[k] : this.options;
            this.extractGmeErrors(k);
            this.drawBins(funcs, options, sub_g, w, h, lineatt, fillatt);
         }
         this.extractGmeErrors(0); // ensure that first block kept at the end
      }

      if (!this.isBatchMode()) {
         addMoveHandler(this, this.testEditable());
         assignContextMenu(this);
      }
   }

   /** @summary Provide tooltip at specified point */
   extractTooltip(pnt) {
      if (!pnt) return null;

      if ((this.draw_kind === 'lines') || (this.draw_kind === 'path') || (this.draw_kind === 'mark'))
         return this.extractTooltipForPath(pnt);

      if (this.draw_kind !== 'nodes') return null;

      const pmain = this.get_main(),
            height = pmain.getFrameHeight(),
            esz = this.error_size,
            isbar1 = (this.options.Bar === 1),
            funcs = isbar1 ? pmain.getGrFuncs(this.options.second_x, this.options.second_y) : null,
            msize = this.marker_size ? Math.round(this.marker_size/2 + 1.5) : 0;
      let findbin = null, best_dist2 = 1e10, best = null;

      this.draw_g.selectAll('.grpoint').each(function() {
         const d = d3_select(this).datum();
         if (d === undefined) return;
         let dist2 = (pnt.x - d.grx1) ** 2;
         if (pnt.nproc === 1) dist2 += (pnt.y - d.gry1) ** 2;
         if (dist2 >= best_dist2) return;

         let rect;

         if (d.error || d.rect || d.marker) {
            rect = { x1: Math.min(-esz, d.grx0, -msize),
                     x2: Math.max(esz, d.grx2, msize),
                     y1: Math.min(-esz, d.gry2, -msize),
                     y2: Math.max(esz, d.gry0, msize) };
         } else if (d.bar) {
             rect = { x1: -d.width/2, x2: d.width/2, y1: 0, y2: height - d.gry1 };

             if (isbar1) {
                const yy0 = funcs.gry(0);
                rect.y1 = (d.gry1 > yy0) ? yy0-d.gry1 : 0;
                rect.y2 = (d.gry1 > yy0) ? 0 : yy0-d.gry1;
             }
          } else
             rect = { x1: -5, x2: 5, y1: -5, y2: 5 };

          const matchx = (pnt.x >= d.grx1 + rect.x1) && (pnt.x <= d.grx1 + rect.x2),
              matchy = (pnt.y >= d.gry1 + rect.y1) && (pnt.y <= d.gry1 + rect.y2);

          if (matchx && (matchy || (pnt.nproc > 1))) {
             best_dist2 = dist2;
             findbin = this;
             best = rect;
             best.exact = /* matchx && */ matchy;
          }
       });

      if (findbin === null) return null;

      const d = d3_select(findbin).datum(),
            gr = this.getGraph(),
            res = { name: gr.fName, title: gr.fTitle,
                    x: d.grx1, y: d.gry1,
                    color1: this.lineatt.color,
                    lines: this.getTooltips(d),
                    rect: best, d3bin: findbin };

       res.user_info = { obj: gr, name: gr.fName, bin: d.indx, cont: d.y, grx: d.grx1, gry: d.gry1 };

      if (this.fillatt?.used && !this.fillatt?.empty())
         res.color2 = this.fillatt.getFillColor();

      if (best.exact) res.exact = true;
      res.menu = res.exact; // activate menu only when exactly locate bin
      res.menu_dist = 3; // distance always fixed
      res.bin = d;
      res.binindx = d.indx;

      return res;
   }

   /** @summary Show tooltip */
   showTooltip(hint) {
      let ttrect = this.draw_g?.selectChild('.tooltip_bin');

      if (!hint || !this.draw_g) {
         ttrect?.remove();
         return;
      }

      if (hint.usepath)
         return this.showTooltipForPath(hint);

      const d = d3_select(hint.d3bin).datum();

      if (ttrect.empty()) {
         ttrect = this.draw_g.append('svg:rect')
                             .attr('class', 'tooltip_bin')
                             .style('pointer-events', 'none')
                             .call(addHighlightStyle);
      }

      hint.changed = ttrect.property('current_bin') !== hint.d3bin;

      if (hint.changed) {
         ttrect.attr('x', d.grx1 + hint.rect.x1)
               .attr('width', hint.rect.x2 - hint.rect.x1)
               .attr('y', d.gry1 + hint.rect.y1)
               .attr('height', hint.rect.y2 - hint.rect.y1)
               .style('opacity', '0.3')
               .property('current_bin', hint.d3bin);
      }
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      const hint = this.extractTooltip(pnt);
      if (!pnt || !pnt.disabled) this.showTooltip(hint);
      return hint;
   }

   /** @summary Find best bin index for specified point */
   findBestBin(pnt) {
      if (!this.bins) return null;

      const islines = (this.draw_kind === 'lines'),
            funcs = this.get_main().getGrFuncs(this.options.second_x, this.options.second_y);
      let bestindx = -1,
          bestbin = null,
          bestdist = 1e10,
          dist, grx, gry, n, bin;

      for (n = 0; n < this.bins.length; ++n) {
         bin = this.bins[n];

         grx = funcs.grx(bin.x);
         gry = funcs.gry(bin.y);

         dist = (pnt.x-grx)**2 + (pnt.y-gry)**2;

         if (dist < bestdist) {
            bestdist = dist;
            bestbin = bin;
            bestindx = n;
         }
      }

      // check last point
      if ((bestdist > 100) && islines) bestbin = null;

      let radius = Math.max(this.lineatt.width + 3, 4);

      if (this.marker_size > 0) radius = Math.max(this.marker_size, radius);

      if (bestbin)
         bestdist = Math.sqrt((pnt.x-funcs.grx(bestbin.x))**2 + (pnt.y-funcs.gry(bestbin.y))**2);

      if (!islines && (bestdist > radius)) bestbin = null;

      if (!bestbin) bestindx = -1;

      const res = { bin: bestbin, indx: bestindx, dist: bestdist, radius: Math.round(radius) };

      if (!bestbin && islines) {
         bestdist = 1e10;

         const IsInside = (x, x1, x2) => ((x1 >= x) && (x >= x2)) || ((x1 <= x) && (x <= x2));

         let bin0 = this.bins[0], grx0 = funcs.grx(bin0.x), gry0, posy = 0;
         for (n = 1; n < this.bins.length; ++n) {
            bin = this.bins[n];
            grx = funcs.grx(bin.x);

            if (IsInside(pnt.x, grx0, grx)) {
               // if inside interval, check Y distance
               gry0 = funcs.gry(bin0.y);
               gry = funcs.gry(bin.y);

               if (Math.abs(grx - grx0) < 1) {
                  // very close x - check only y
                  posy = pnt.y;
                  dist = IsInside(pnt.y, gry0, gry) ? 0 : Math.min(Math.abs(pnt.y-gry0), Math.abs(pnt.y-gry));
               } else {
                  posy = gry0 + (pnt.x - grx0) / (grx - grx0) * (gry - gry0);
                  dist = Math.abs(posy - pnt.y);
               }

               if (dist < bestdist) {
                  bestdist = dist;
                  res.linex = pnt.x;
                  res.liney = posy;
               }
            }

            bin0 = bin;
            grx0 = grx;
         }

         if (bestdist < radius*0.5) {
            res.linedist = bestdist;
            res.closeline = true;
         }
      }

      return res;
   }

   /** @summary Check editable flag for TGraph
     * @desc if arg specified changes or toggles editable flag */
   testEditable(arg) {
      const obj = this.getGraph();
      if (!obj) return false;
      if ((arg === 'toggle') || ((arg !== undefined) && (!arg !== obj.TestBit(kNotEditable))))
         obj.InvertBit(kNotEditable);
      return !obj.TestBit(kNotEditable);
   }

   /** @summary Provide tooltip at specified point for path-based drawing */
   extractTooltipForPath(pnt) {
      if (this.bins === null) return null;

      const best = this.findBestBin(pnt);

      if (!best || (!best.bin && !best.closeline)) return null;

      const islines = (this.draw_kind === 'lines'),
          ismark = (this.draw_kind === 'mark'),
          pmain = this.get_main(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          gr = this.getGraph(),
          res = { name: gr.fName, title: gr.fTitle,
                  x: best.bin ? funcs.grx(best.bin.x) : best.linex,
                  y: best.bin ? funcs.gry(best.bin.y) : best.liney,
                  color1: this.lineatt.color,
                  lines: this.getTooltips(best.bin),
                  usepath: true };

      res.user_info = { obj: gr, name: gr.fName, bin: 0, cont: 0, grx: res.x, gry: res.y };

      res.ismark = ismark;
      res.islines = islines;

      if (best.closeline) {
         res.menu = res.exact = true;
         res.menu_dist = best.linedist;
      } else if (best.bin) {
         if (this.options.EF && islines) {
            res.gry1 = funcs.gry(best.bin.y - best.bin.eylow);
            res.gry2 = funcs.gry(best.bin.y + best.bin.eyhigh);
         } else
            res.gry1 = res.gry2 = funcs.gry(best.bin.y);


         res.binindx = best.indx;
         res.bin = best.bin;
         res.radius = best.radius;
         res.user_info.bin = best.indx;
         res.user_info.cont = best.bin.y;

         res.exact = (Math.abs(pnt.x - res.x) <= best.radius) &&
            ((Math.abs(pnt.y - res.gry1) <= best.radius) || (Math.abs(pnt.y - res.gry2) <= best.radius));

         res.menu = res.exact;
         res.menu_dist = Math.sqrt((pnt.x-res.x)**2 + Math.min(Math.abs(pnt.y-res.gry1), Math.abs(pnt.y-res.gry2))**2);
      }

      if (this.fillatt?.used && !this.fillatt?.empty())
         res.color2 = this.fillatt.getFillColor();

      if (!islines) {
         res.color1 = this.getColor(gr.fMarkerColor);
         if (!res.color2) res.color2 = res.color1;
      }

      return res;
   }

   /** @summary Show tooltip for path drawing */
   showTooltipForPath(hint) {
      let ttbin = this.draw_g?.selectChild('.tooltip_bin');

      if (!hint?.bin || !this.draw_g) {
         ttbin?.remove();
         return;
      }

      if (ttbin.empty())
         ttbin = this.draw_g.append('svg:g').attr('class', 'tooltip_bin');

      hint.changed = ttbin.property('current_bin') !== hint.bin;

      if (hint.changed) {
         ttbin.selectAll('*').remove(); // first delete all children
         ttbin.property('current_bin', hint.bin);

         if (hint.ismark) {
            ttbin.append('svg:rect')
                 .style('pointer-events', 'none')
                 .call(addHighlightStyle)
                 .style('opacity', '0.3')
                 .attr('x', Math.round(hint.x - hint.radius))
                 .attr('y', Math.round(hint.y - hint.radius))
                 .attr('width', 2*hint.radius)
                 .attr('height', 2*hint.radius);
         } else {
            ttbin.append('svg:circle').attr('cy', Math.round(hint.gry1));
            if (Math.abs(hint.gry1-hint.gry2) > 1)
               ttbin.append('svg:circle').attr('cy', Math.round(hint.gry2));

            const elem = ttbin.selectAll('circle')
                            .attr('r', hint.radius)
                            .attr('cx', Math.round(hint.x));

            if (!hint.islines)
               elem.style('stroke', hint.color1 === 'black' ? 'green' : 'black').style('fill', 'none');
             else {
               if (this.options.Line || this.options.Curve)
                  elem.call(this.lineatt.func);
               else
                  elem.style('stroke', 'black');
               if (this.options.Fill)
                  elem.call(this.fillatt.func);
               else
                  elem.style('fill', 'none');
            }
         }
      }
   }

   /** @summary Check if graph moving is enabled */
   moveEnabled() {
      return this.testEditable();
   }

   /** @summary Start moving of TGraph */
   moveStart(x, y) {
      this.pos_dx = this.pos_dy = 0;
      this.move_funcs = this.get_main().getGrFuncs(this.options.second_x, this.options.second_y);
      const hint = this.extractTooltip({ x, y });
      if (hint && hint.exact && (hint.binindx !== undefined)) {
         this.move_binindx = hint.binindx;
         this.move_bin = hint.bin;
         this.move_x0 = this.move_funcs.grx(this.move_bin.x);
         this.move_y0 = this.move_funcs.gry(this.move_bin.y);
      } else
         delete this.move_binindx;
   }

   /** @summary Perform moving */
   moveDrag(dx, dy) {
      this.pos_dx += dx;
      this.pos_dy += dy;

      if (this.move_binindx === undefined)
         makeTranslate(this.draw_g, this.pos_dx, this.pos_dy);
       else if (this.move_funcs && this.move_bin) {
         this.move_bin.x = this.move_funcs.revertAxis('x', this.move_x0 + this.pos_dx);
         this.move_bin.y = this.move_funcs.revertAxis('y', this.move_y0 + this.pos_dy);
         this.drawGraph();
      }
   }

   /** @summary Complete moving */
   moveEnd(not_changed) {
      const graph = this.getGraph(), last = graph?.fNpoints-1;
      let exec = '';

      const changeBin = bin => {
         exec += `SetPoint(${bin.indx},${bin.x},${bin.y});;`;
         graph.fX[bin.indx] = bin.x;
         graph.fY[bin.indx] = bin.y;
         if ((bin.indx === 0) && this._cutg_lastsame) {
            exec += `SetPoint(${last},${bin.x},${bin.y});;`;
            graph.fX[last] = bin.x;
            graph.fY[last] = bin.y;
         }
      };

      if (this.move_binindx === undefined) {
         this.draw_g.attr('transform', null);

         if (this.move_funcs && this.bins && !not_changed) {
            for (let k = 0; k < this.bins.length; ++k) {
               const bin = this.bins[k];
               bin.x = this.move_funcs.revertAxis('x', this.move_funcs.grx(bin.x) + this.pos_dx);
               bin.y = this.move_funcs.revertAxis('y', this.move_funcs.gry(bin.y) + this.pos_dy);
               changeBin(bin);
            }
            if (graph.$redraw_pad)
               this.redrawPad();
            else
               this.drawGraph();
         }
      } else {
         changeBin(this.move_bin);
         delete this.move_binindx;
         if (graph.$redraw_pad)
            this.redrawPad();
      }

      delete this.move_funcs;

      if (exec && !not_changed)
         this.submitCanvExec(exec);
   }

   /** @summary Fill option object used in TWebCanvas */
   fillWebObjectOptions(res) {
      if (this._auto_exec && res) {
         res.fcust = 'auto_exec:' + this._auto_exec;
         delete this._auto_exec;
      }
   }

   /** @summary Fill context menu */
   fillContextMenuItems(menu) {
      if (!this.snapid)
         menu.addchk(this.testEditable(), 'Editable', () => { this.testEditable('toggle'); this.drawGraph(); });
   }

   /** @summary Execute menu command
     * @private */
   executeMenuCommand(method, args) {
      if (super.executeMenuCommand(method, args)) return true;

      const canp = this.getCanvPainter(), pmain = this.get_main();

      if ((method.fName === 'RemovePoint') || (method.fName === 'InsertPoint')) {
         if (!canp || canp._readonly) return true; // ignore function

         const pnt = isFunc(pmain?.getLastEventPos) ? pmain.getLastEventPos() : null,
             hint = this.extractTooltip(pnt);

         if (method.fName === 'InsertPoint') {
            if (pnt) {
               const funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
                     userx = funcs.revertAxis('x', pnt.x) ?? 0,
                     usery = funcs.revertAxis('y', pnt.y) ?? 0;
               this.submitCanvExec(`AddPoint(${userx.toFixed(3)}, ${usery.toFixed(3)})`, method.$execid);
            }
         } else if (method.$execid && (hint?.binindx !== undefined))
            this.submitCanvExec(`RemovePoint(${hint.binindx})`, method.$execid);


         return true; // call is processed
      }

      return false;
   }

   /** @summary Update TGraph object members
     * @private */
   _updateMembers(graph, obj) {
      graph.fBits = obj.fBits;
      graph.fTitle = obj.fTitle;
      graph.fX = obj.fX;
      graph.fY = obj.fY;
      graph.fNpoints = obj.fNpoints;
      graph.fMinimum = obj.fMinimum;
      graph.fMaximum = obj.fMaximum;

      const o = this.options;

      if (this.snapid !== undefined)
         o._pfc = o._plc = o._pmc = 0; // auto colors should be processed in web canvas

      if (!o._pfc)
         graph.fFillColor = obj.fFillColor;
      graph.fFillStyle = obj.fFillStyle;
      if (!o._plc)
         graph.fLineColor = obj.fLineColor;
      graph.fLineStyle = obj.fLineStyle;
      graph.fLineWidth = obj.fLineWidth;
      if (!o._pmc)
         graph.fMarkerColor = obj.fMarkerColor;
      graph.fMarkerSize = obj.fMarkerSize;
      graph.fMarkerStyle = obj.fMarkerStyle;

      return obj.fFunctions;
   }

   /** @summary Update TGraph object */
   updateObject(obj, opt) {
      if (!this.matchObjectType(obj)) return false;

      if (opt && (opt !== this.options.original))
         this.decodeOptions(opt);

      const new_funcs = this._updateMembers(this.getObject(), obj);

      this.createBins();

      delete this.$redraw_hist;

      // if our own histogram was used as axis drawing, we need update histogram as well
      if (this.axes_draw) {
         const histo = this.createHistogram(),
               hist_painter = this.getMainPainter();
         if (hist_painter?.isSecondary(this)) {
            hist_painter.updateObject(histo, this.options.Axis);
            this.$redraw_hist = true;
         }
      }

      this._funcHandler = new FunctionsHandler(this, this.getPadPainter(), new_funcs);

      return true;
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range
     * @desc allow to zoom TGraph only when at least one point in the range */
   canZoomInside(axis, min, max) {
      const gr = this.getGraph();
      if (!gr || (axis !== (this.options.pos3d ? 'y' : 'x'))) return false;

      for (let n = 0; n < gr.fNpoints; ++n)
         if ((min < gr.fX[n]) && (gr.fX[n] < max)) return true;

      return false;
   }

   /** @summary Process click on graph-defined buttons */
   clickButton(funcname) {
      if (funcname !== 'ToggleZoom') return false;

      if ((this.xmin === this.xmax) && (this.ymin === this.ymax)) return false;

      return this.getFramePainter()?.zoom(this.xmin, this.xmax, this.ymin, this.ymax);
   }

   /** @summary Find TF1/TF2 in TGraph list of functions */
   findFunc() {
      return this.getGraph()?.fFunctions?.arr?.find(func => (func._typename === clTF1) || (func._typename === clTF2));
   }

   /** @summary Find stat box in TGraph list of functions */
   findStat() {
      return this.getGraph()?.fFunctions?.arr?.find(func => (func._typename === clTPaveStats) && (func.fName === 'stats'));
   }

   /** @summary Create stat box */
   createStat() {
      const func = this.findFunc();
      if (!func) return null;

      let stats = this.findStat();
      if (stats) return stats;

      // do not create stats box when drawing canvas
      if (this.getCanvPainter()?.normal_canvas) return null;

      this.create_stats = true;

      const st = gStyle;

      stats = create(clTPaveStats);
      Object.assign(stats, { fName: 'stats', fOptStat: 0, fOptFit: st.fOptFit || 111, fBorderSize: 1,
                             fX1NDC: st.fStatX - st.fStatW, fY1NDC: st.fStatY - st.fStatH, fX2NDC: st.fStatX, fY2NDC: st.fStatY,
                             fFillColor: st.fStatColor, fFillStyle: st.fStatStyle });

      stats.fTextAngle = 0;
      stats.fTextSize = st.fStatFontSize; // 9 ??
      stats.fTextAlign = 12;
      stats.fTextColor = st.fStatTextColor;
      stats.fTextFont = st.fStatFont;

      stats.AddText(func.fName);

      // while TF1 was found, one can be sure that stats is existing
      this.getGraph().fFunctions.Add(stats);

      return stats;
   }

   /** @summary Fill statistic */
   fillStatistic(stat, _dostat, dofit) {
      const func = this.findFunc();

      if (!func || !dofit) return false;

      stat.clearPave();

      stat.fillFunctionStat(func, (dofit === 1) ? 111 : dofit, 1);

      return true;
   }

   /** @summary Draw axis histogram
     * @private */
   async drawAxisHisto() {
      const histo = this.createHistogram();
      return TH1Painter.draw(this.getDom(), histo, this.options.Axis);
   }

   /** @summary Draw TGraph
     * @private */
   static async _drawGraph(painter, opt) {
      painter.decodeOptions(opt, true);
      painter.createBins();
      painter.createStat();
      const graph = painter.getGraph();
      if (!settings.DragGraphs && graph && !graph.TestBit(kNotEditable))
         graph.InvertBit(kNotEditable);

      let promise = Promise.resolve();

      if ((!painter.getMainPainter() || painter.options.second_x || painter.options.second_y) && painter.options.Axis) {
         promise = painter.drawAxisHisto().then(hist_painter => {
            hist_painter?.setSecondaryId(painter, 'hist');
            painter.axes_draw = !!hist_painter;
         });
      }

      return promise.then(() => {
         painter.addToPadPrimitives();
         return painter.drawGraph();
      }).then(() => {
         const handler = new FunctionsHandler(painter, painter.getPadPainter(), graph.fFunctions, true);
         return handler.drawNext(0); // returns painter
      });
   }

   static async draw(dom, graph, opt) {
      return TGraphPainter._drawGraph(new TGraphPainter(dom, graph), opt);
   }

} // class TGraphPainter

export { clTGraphAsymmErrors, TGraphPainter };
