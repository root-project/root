/// more ROOT classes

import { gStyle, BIT, settings, create, createHistogram, isBatchMode } from '../core.mjs';

import { select as d3_select } from '../d3.mjs';

import { DrawOptions, buildSvgPath } from '../base/BasePainter.mjs';

import { ObjectPainter } from '../base/ObjectPainter.mjs';

import { TH1Painter } from './TH1Painter.mjs';

import { TAttLineHandler } from '../base/TAttLineHandler.mjs';

import { TAttFillHandler } from '../base/TAttFillHandler.mjs';

import { addMoveHandler } from '../gui/utils.mjs';


const kNotEditable = BIT(18);   // bit set if graph is non editable

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
      this.is_bent = (graph._typename == 'TGraphBentErrors');
      this.has_errors = (graph._typename == 'TGraphErrors') ||
                        (graph._typename == 'TGraphMultiErrors') ||
                        (graph._typename == 'TGraphAsymmErrors') ||
                         this.is_bent || graph._typename.match(/^RooHist/);
   }

   /** @summary Redraw graph
     * @desc may redraw histogram which was used to draw axes
     * @returns {Promise} for ready */
   redraw() {
      let promise = Promise.resolve(true);

      if (this.$redraw_hist) {
         delete this.$redraw_hist;
         let hist_painter = this.getMainPainter();
         if (hist_painter && hist_painter.$secondary && this.axes_draw)
            promise = hist_painter.redraw();
      }

      return promise.then(() => this.drawGraph());
   }

   /** @summary Cleanup graph painter */
   cleanup() {
      delete this.interactive_bin; // break mouse handling
      delete this.bins;
      super.cleanup();
   }

   /** @summary Returns object if this drawing TGraphMultiErrors object */
   get_gme() {
      let graph = this.getObject();
      return graph._typename == "TGraphMultiErrors" ? graph : null;
   }

   /** @summary Decode options */
   decodeOptions(opt, first_time) {

      if ((typeof opt == "string") && (opt.indexOf("same ") == 0))
         opt = opt.slice(5);

      let graph = this.getObject(),
          is_gme = !!this.get_gme(),
          blocks_gme = [],
          has_main = first_time ? !!this.getMainPainter() : !this.axes_draw;

      if (!this.options) this.options = {};

      // decode main draw options for the graph
      const decodeBlock = (d, res) => {
         Object.assign(res, { Line: 0, Curve: 0, Rect: 0, Mark: 0, Bar: 0, OutRange: 0, EF:0, Fill: 0, MainError: 1, Ends: 1, ScaleErrX: 1 });

         if (is_gme && d.check("S=", true)) res.ScaleErrX = d.partAsFloat();

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
         if (d.check('1')) { if (res.Bar == 1) res.Bar = 2; }
         if (d.check('2')) { res.Rect = 1; res.Errors = 0; }
         if (d.check('3')) { res.EF = 1; res.Errors = 0;  }
         if (d.check('4')) { res.EF = 2; res.Errors = 0; }
         if (d.check('5')) { res.Rect = 2; res.Errors = 0; }
         if (d.check('X')) res.Errors = 0;
      };

      Object.assign(this.options, { Axis: "", NoOpt: 0, PadStats: false, original: opt, second_x: false, second_y: false, individual_styles: false });

      if (is_gme && opt) {
         if (opt.indexOf(";") > 0) {
            blocks_gme = opt.split(";");
            opt = blocks_gme.shift();
         } else if (opt.indexOf("_") > 0) {
            blocks_gme = opt.split("_");
            opt = blocks_gme.shift();
         }
      }

      let res = this.options,
          d = new DrawOptions(opt);

      // check pad options first
      res.PadStats = d.check("USE_PAD_STATS");
      let hopt = "", checkhopt = ["USE_PAD_TITLE", "LOGXY", "LOGX", "LOGY", "LOGZ", "GRIDXY", "GRIDX", "GRIDY", "TICKXY", "TICKX", "TICKY"];
      checkhopt.forEach(name => { if (d.check(name)) hopt += ";" + name; });
      if (d.check('XAXIS_', true)) hopt += ";XAXIS_" + d.part;
      if (d.check('YAXIS_', true)) hopt += ";YAXIS_" + d.part;

      if (d.empty()) {
         res.original = has_main ? "lp" : "alp";
         d = new DrawOptions(res.original);
      }

      if (d.check('NOOPT')) res.NoOpt = 1;

      if (d.check("POS3D_", true)) res.pos3d = d.partAsInt() - 0.5;

      res._pfc = d.check("PFC");
      res._plc = d.check("PLC");
      res._pmc = d.check("PMC");

      if (d.check('A')) res.Axis = d.check("I") ? "A" : "AXIS"; // I means invisible axis
      if (d.check('X+')) { res.Axis += "X+"; res.second_x = has_main; }
      if (d.check('Y+')) { res.Axis += "Y+"; res.second_y = has_main; }
      if (d.check('RX')) res.Axis += "RX";
      if (d.check('RY')) res.Axis += "RY";

      if (is_gme) {
         res.blocks = [];
         res.skip_errors_x0 = res.skip_errors_y0 = false;
         if (d.check('X0')) res.skip_errors_x0 = true;
         if (d.check('Y0')) res.skip_errors_y0 = true;
      }

      decodeBlock(d, res);

      if (is_gme) {
         if (d.check('S')) res.individual_styles = true;
      }

      // if (d.check('E')) res.Errors = 1; // E option only defined for TGraphPolar

      if (res.Errors === undefined)
         res.Errors = this.has_errors && (!is_gme || !blocks_gme.length) ? 1 : 0;

      // special case - one could use svg:path to draw many pixels (
      if ((res.Mark == 1) && (graph.fMarkerStyle == 1)) res.Mark = 101;

      // if no drawing option is selected and if opt=='' nothing is done.
      if (res.Line + res.Fill + res.Curve + res.Mark + res.Bar + res.EF + res.Rect + res.Errors == 0) {
         if (d.empty()) res.Line = 1;
      }

      if (graph._typename == 'TGraphErrors') {
         let len = graph.fEX.length, m = 0;
         for (let k = 0; k < len; ++k)
            m = Math.max(m, graph.fEX[k], graph.fEY[k]);
         if (m < 1e-100)
            res.Errors = 0;
      }

      if (!res.Axis) {
         // check if axis should be drawn
         // either graph drawn directly or
         // graph is first object in list of primitives
         let pp = this.getPadPainter(),
             pad = pp ? pp.getRootPad(true) : null;
         if (!pad || (pad.fPrimitives && (pad.fPrimitives.arr[0] === graph))) res.Axis = "AXIS";
      } else if (res.Axis.indexOf("A") < 0) {
         res.Axis = "AXIS," + res.Axis;
      }

      res.Axis += hopt;

      for (let bl = 0; bl < blocks_gme.length; ++bl) {
         let subd = new DrawOptions(blocks_gme[bl]), subres = {};
         decodeBlock(subd, subres);
         subres.skip_errors_x0 = res.skip_errors_x0;
         subres.skip_errors_y0 = res.skip_errors_y0;
         res.blocks.push(subres);
      }
   }

   /** @summary Extract errors for TGraphMultiErrors */
   extractGmeErrors(nblock) {
      if (!this.bins) return;
      let gr = this.getObject();
      this.bins.forEach(bin => {
         bin.eylow  = gr.fEyL[nblock][bin.indx];
         bin.eyhigh = gr.fEyH[nblock][bin.indx];
      });
   }

   /** @summary Create bins for TF1 drawing */
   createBins() {
      let gr = this.getObject();
      if (!gr) return;

      let kind = 0, npoints = gr.fNpoints;
      if ((gr._typename==="TCutG") && (npoints>3)) npoints--;

      if (gr._typename == 'TGraphErrors') kind = 1; else
      if (gr._typename == 'TGraphMultiErrors') kind = 2; else
      if (gr._typename == 'TGraphAsymmErrors' || gr._typename == 'TGraphBentErrors'
          || gr._typename.match(/^RooHist/)) kind = 3;

      this.bins = new Array(npoints);

      for (let p = 0; p < npoints; ++p) {
         let bin = this.bins[p] = { x: gr.fX[p], y: gr.fY[p], indx: p };
         switch(kind) {
            case 1:
               bin.exlow = bin.exhigh = gr.fEX[p];
               bin.eylow = bin.eyhigh = gr.fEY[p];
               break;
            case 2:
               bin.exlow  = gr.fExL[p];
               bin.exhigh = gr.fExH[p];
               bin.eylow  = gr.fEyL[0][p];
               bin.eyhigh = gr.fEyH[0][p];
               break;
            case 3:
               bin.exlow  = gr.fEXlow[p];
               bin.exhigh = gr.fEXhigh[p];
               bin.eylow  = gr.fEYlow[p];
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

   /** @summary Create histogram for graph
     * @descgraph bins should be created when calling this function
     * @param {object} histo - existing histogram instance
     * @param {boolean} only_set_ranges - when specified, just assign ranges */
   createHistogram(histo, set_x, set_y) {
      let xmin = this.xmin, xmax = this.xmax, ymin = this.ymin, ymax = this.ymax;

      if (xmin >= xmax) xmax = xmin+1;
      if (ymin >= ymax) ymax = ymin+1;
      let dx = (xmax-xmin)*0.1, dy = (ymax-ymin)*0.1,
          uxmin = xmin - dx, uxmax = xmax + dx,
          minimum = ymin - dy, maximum = ymax + dy;

      if ((uxmin < 0) && (xmin >= 0)) uxmin = xmin*0.9;
      if ((uxmax > 0) && (xmax <= 0)) uxmax = 0;

      let graph = this.getObject();

      if (graph.fMinimum != -1111) minimum = ymin = graph.fMinimum;
      if (graph.fMaximum != -1111) maximum = graph.fMaximum;
      if ((minimum < 0) && (ymin >=0)) minimum = 0.9*ymin;

      histo = graph.fHistogram;

      if (!set_x && !set_y) set_x = set_y = true;

      if (!histo) {
         histo = graph.fHistogram = createHistogram("TH1F", 100);
         histo.fName = graph.fName + "_h";
         let kNoStats = BIT(9);
         histo.fBits = histo.fBits | kNoStats;
         this._own_histogram = true;
      }

      histo.fTitle = graph.fTitle;

      if (set_x) {
         histo.fXaxis.fXmin = uxmin;
         histo.fXaxis.fXmax = uxmax;
      }

      if (set_y) {
         histo.fYaxis.fXmin = minimum;
         histo.fYaxis.fXmax = maximum;
         histo.fMinimum = minimum;
         histo.fMaximum = maximum;
      }

      return histo;
   }

   /** @summary Check if user range can be unzommed
     * @desc Used when graph points covers larger range than provided histogram */
   unzoomUserRange(dox, doy /*, doz*/) {
      let graph = this.getObject();
      if (this._own_histogram || !graph) return false;

      let histo = graph.fHistogram;

      dox = dox && histo && ((histo.fXaxis.fXmin > this.xmin) || (histo.fXaxis.fXmax < this.xmax));
      doy = doy && histo && ((histo.fYaxis.fXmin > this.ymin) || (histo.fYaxis.fXmax < this.ymax));
      if (!dox && !doy) return false;

      this.createHistogram(null, dox, doy);
      let hpainter = this.getMainPainter();
      if (hpainter) hpainter.extractAxesProperties(1); // just to enforce ranges extraction

      return true;
   }

   /** @summary Returns true if graph drawing can be optimize */
   canOptimize() {
      return (settings.OptimizeDraw > 0) && !this.options.NoOpt;
   }

   /** @summary Returns optimized bins - if optimization enabled */
   optimizeBins(maxpnt, filter_func) {
      if ((this.bins.length < 30) && !filter_func) return this.bins;

      let selbins = null;
      if (typeof filter_func == 'function') {
         for (let n = 0; n < this.bins.length; ++n) {
            if (filter_func(this.bins[n],n)) {
               if (!selbins) selbins = (n==0) ? [] : this.bins.slice(0, n);
            } else {
               if (selbins) selbins.push(this.bins[n]);
            }
         }
      }
      if (!selbins) selbins = this.bins;

      if (!maxpnt) maxpnt = 500000;

      if ((selbins.length < maxpnt) || !this.canOptimize()) return selbins;
      let step = Math.floor(selbins.length / maxpnt);
      if (step < 2) step = 2;
      let optbins = [];
      for (let n = 0; n < selbins.length; n+=step)
         optbins.push(selbins[n]);

      return optbins;
   }

   /** @summary Returns tooltip for specified bin */
   getTooltips(d) {
      let pmain = this.getFramePainter(), lines = [],
          funcs = pmain ? pmain.getGrFuncs(this.options.second_x, this.options.second_y) : null,
          gme = this.get_gme();

      lines.push(this.getObjectHint());

      if (d && funcs) {
         lines.push("x = " + funcs.axisAsText("x", d.x));
         lines.push("y = " + funcs.axisAsText("y", d.y));

         if (gme)
            lines.push("error x = -" + funcs.axisAsText("x", gme.fExL[d.indx]) + "/+" + funcs.axisAsText("x", gme.fExH[d.indx]));
         else if (this.options.Errors && (funcs.x_handle.kind=='normal') && (d.exlow || d.exhigh))
            lines.push("error x = -" + funcs.axisAsText("x", d.exlow) + "/+" + funcs.axisAsText("x", d.exhigh));

         if (gme) {
            for (let ny = 0; ny < gme.fNYErrors; ++ny)
               lines.push(`error y${ny} = -${funcs.axisAsText("y", gme.fEyL[ny][d.indx])}/+${funcs.axisAsText("y", gme.fEyH[ny][d.indx])}`);
         } else if ((this.options.Errors || (this.options.EF > 0)) && (funcs.y_handle.kind=='normal') && (d.eylow || d.eyhigh))
            lines.push("error y = -" + funcs.axisAsText("y", d.eylow) + "/+" + funcs.axisAsText("y", d.eyhigh));

      }
      return lines;
   }

   /** @summary Provide frame painter for graph
     * @desc If not exists, emulate its behaviour */
   get_main() {
      let pmain = this.getFramePainter();

      if (pmain && pmain.grx && pmain.gry) return pmain;

      // FIXME: check if needed, can be removed easily
      let pp = this.getPadPainter(),
          rect = pp ? pp.getPadRect() : { width: 800, height: 600 };

      pmain = {
          pad_layer: true,
          pad: pp.getRootPad(true),
          pw: rect.width,
          ph: rect.height,
          getFrameWidth: function() { return this.pw; },
          getFrameHeight: function() { return this.ph; },
          grx: function(value) {
             if (this.pad.fLogx)
                value = (value>0) ? Math.log10(value) : this.pad.fUxmin;
             else
                value = (value - this.pad.fX1) / (this.pad.fX2 - this.pad.fX1);
             return value*this.pw;
          },
          gry: function(value) {
             if (this.pad.fLogy)
                value = (value>0) ? Math.log10(value) : this.pad.fUymin;
             else
                value = (value - this.pad.fY1) / (this.pad.fY2 - this.pad.fY1);
             return (1-value)*this.ph;
          },
          getGrFuncs: function() { return this; }
      }

      return pmain.pad ? pmain : null;
   }

   /** @summary append exclusion area to created path */
   appendExclusion(is_curve, path, drawbins, excl_width) {
      let extrabins = [];
      for (let n = drawbins.length-1; n >= 0; --n) {
         let bin = drawbins[n],
             dlen = Math.sqrt(bin.dgrx*bin.dgrx + bin.dgry*bin.dgry);
         // shift point, using
         bin.grx += excl_width*bin.dgry/dlen;
         bin.gry -= excl_width*bin.dgrx/dlen;
         extrabins.push(bin);
      }

      let path2 = buildSvgPath("L" + (is_curve ? "bezier" : "line"), extrabins);

      this.draw_g.append("svg:path")
                 .attr("d", path.path + path2.path + "Z")
                 .call(this.fillatt.func)
                 .style('opacity', 0.75);
   }

   /** @summary draw TGraph bins with specified options
     * @desc Can be called several times */
   drawBins(funcs, options, draw_g, w, h, lineatt, fillatt, main_block) {
      let graph = this.getObject(),
          excl_width = 0, drawbins = null;

      if (main_block && (lineatt.excl_side != 0)) {
         excl_width = lineatt.excl_width;
         if ((lineatt.width > 0) && !options.Line && !options.Curve) options.Line = 1;
      }

      if (options.EF) {
         drawbins = this.optimizeBins((options.EF > 1) ? 20000 : 0);

         // build lower part
         for (let n = 0; n < drawbins.length; ++n) {
            let bin = drawbins[n];
            bin.grx = funcs.grx(bin.x);
            bin.gry = funcs.gry(bin.y - bin.eylow);
         }

         let path1 = buildSvgPath((options.EF > 1) ? "bezier" : "line", drawbins),
             bins2 = [];

         for (let n = drawbins.length-1; n >= 0; --n) {
            let bin = drawbins[n];
            bin.gry = funcs.gry(bin.y + bin.eyhigh);
            bins2.push(bin);
         }

         // build upper part (in reverse direction)
         let path2 = buildSvgPath((options.EF > 1) ? "Lbezier" : "Lline", bins2);

         draw_g.append("svg:path")
               .attr("d", path1.path + path2.path + "Z")
               .call(fillatt.func);
         if (main_block)
            this.draw_kind = "lines";
      }

      if (options.Line || options.Fill) {

         let close_symbol = "";
         if (graph._typename == "TCutG") options.Fill = 1;

         if (options.Fill) {
            close_symbol = "Z"; // always close area if we want to fill it
            excl_width = 0;
         }

         if (!drawbins) drawbins = this.optimizeBins(0);

         for (let n = 0; n < drawbins.length; ++n) {
            let bin = drawbins[n];
            bin.grx = funcs.grx(bin.x);
            bin.gry = funcs.gry(bin.y);
         }

         let kind = "line"; // simple line
         if (excl_width) kind += "calc"; // we need to calculated deltas to build exclusion points

         let path = buildSvgPath(kind, drawbins);

         if (excl_width)
             this.appendExclusion(false, path, drawbins, excl_width);

         let elem = draw_g.append("svg:path").attr("d", path.path + close_symbol);
         if (options.Line)
            elem.call(lineatt.func);

         if (options.Fill)
            elem.call(fillatt.func);
         else
            elem.style('fill', 'none');

         if (main_block)
            this.draw_kind = "lines";
      }

      if (options.Curve) {
         let curvebins = drawbins;
         if ((this.draw_kind != "lines") || !curvebins || ((options.Curve == 1) && (curvebins.length > 20000))) {
            curvebins = this.optimizeBins((options.Curve == 1) ? 20000 : 0);
            for (let n = 0; n < curvebins.length; ++n) {
               let bin = curvebins[n];
               bin.grx = funcs.grx(bin.x);
               bin.gry = funcs.gry(bin.y);
            }
         }

         let kind = "bezier";
         if (excl_width) kind += "calc"; // we need to calculated deltas to build exclusion points

         let path = buildSvgPath(kind, curvebins);

         if (excl_width)
             this.appendExclusion(true, path, curvebins, excl_width);

         draw_g.append("svg:path")
               .attr("d", path.path)
               .call(lineatt.func)
               .style('fill', 'none');
         if (main_block)
            this.draw_kind = "lines"; // handled same way as lines
      }

      let nodes = null;

      if (options.Errors || options.Rect || options.Bar) {

         drawbins = this.optimizeBins(5000, (pnt,i) => {

            let grx = funcs.grx(pnt.x);

            // when drawing bars, take all points
            if (!options.Bar && ((grx < 0) || (grx > w))) return true;

            let gry = funcs.gry(pnt.y);

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
               } else {
                  pnt.grdx0 = pnt.grdx2 = pnt.grdy0 = pnt.grdy2 = 0;
               }
            }

            return false;
         });

         if (main_block)
            this.draw_kind = "nodes";

         nodes = draw_g.selectAll(".grpoint")
                       .data(drawbins)
                       .enter()
                       .append("svg:g")
                       .attr("class", "grpoint")
                       .attr("transform", d => `translate(${d.grx1},${d.gry1})`);
      }

      if (options.Bar) {
         // calculate bar width
         for (let i = 1; i < drawbins.length-1; ++i)
            drawbins[i].width = Math.max(2, (drawbins[i+1].grx1 - drawbins[i-1].grx1) / 2 - 2);

         // first and last bins
         switch (drawbins.length) {
            case 0: break;
            case 1: drawbins[0].width = w/4; break; // pathologic case of single bin
            case 2: drawbins[0].width = drawbins[1].width = (drawbins[1].grx1-drawbins[0].grx1)/2; break;
            default:
               drawbins[0].width = drawbins[1].width;
               drawbins[drawbins.length-1].width = drawbins[drawbins.length-2].width;
         }

         let yy0 = Math.round(funcs.gry(0)), usefill = fillatt;

         if (main_block) {
            let fp = this.getFramePainter(),
                fpcol = fp && fp.fillatt && !fp.fillatt.empty() ? fp.fillatt.getFillColor() : -1;
            if (fpcol === fillatt.getFillColor())
               usefill = new TAttFillHandler({ color: fpcol == "white" ? 1 : 0, pattern: 1001 });
         }

         nodes.append("svg:path")
              .attr("d", d => {
                 d.bar = true; // element drawn as bar
                 let dx = Math.round(-d.width/2),
                     dw = Math.round(d.width),
                     dy = (options.Bar!==1) ? 0 : ((d.gry1 > yy0) ? yy0-d.gry1 : 0),
                     dh = (options.Bar!==1) ? (h > d.gry1 ? h - d.gry1 : 0) : Math.abs(yy0 - d.gry1);
                 return `M${dx},${dy}h${dw}v${dh}h${-dw}z`;
              })
            .call(usefill.func);
      }

      if (options.Rect) {
         nodes.filter(d => (d.exlow > 0) && (d.exhigh > 0) && (d.eylow > 0) && (d.eyhigh > 0))
           .append("svg:path")
           .attr("d", d => {
               d.rect = true;
               return `M${d.grx0},${d.gry0}H${d.grx2}V${d.gry2}H${d.grx0}Z`;
            })
           .call(fillatt.func)
           .call(options.Rect === 2 ? lineatt.func : () => {});
      }

      this.error_size = 0;

      if (options.Errors) {
         // to show end of error markers, use line width attribute
         let lw = lineatt.width + gStyle.fEndErrorSize, bb = 0,
             vv = options.Ends ? `m0,${lw}v${-2*lw}` : "",
             hh = options.Ends ? `m${lw},0h${-2*lw}` : "",
             vleft = vv, vright = vv, htop = hh, hbottom = hh;

         const mainLine = (dx,dy) => {
            if (!options.MainError) return `M${dx},${dy}`;
            let res = "M0,0";
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
            visible = visible.filter(d => ((d.x != 0) || !options.skip_errors_x0) && ((d.y != 0) || !options.skip_errors_y0));

         if (!isBatchMode() && settings.Tooltip && main_block)
            visible.append("svg:path")
                   .style("fill", "none")
                   .style("pointer-events", "visibleFill")
                   .attr("d", d => `M${d.grx0},${d.gry0}h${d.grx2-d.grx0}v${d.gry2-d.gry0}h${d.grx0-d.grx2}z`);

         visible.append("svg:path")
             .call(lineatt.func)
             .style("fill", "none")
             .attr("d", d => {
                d.error = true;
                return ((d.exlow > 0)  ? mainLine(d.grx0+lw, d.grdx0) + vleft : "") +
                       ((d.exhigh > 0) ? mainLine(d.grx2-lw, d.grdx2) + vright : "") +
                       ((d.eylow > 0)  ? mainLine(d.grdy0, d.gry0-lw) + hbottom : "") +
                       ((d.eyhigh > 0) ? mainLine(d.grdy2, d.gry2+lw) + htop : "");
              });
      }

      if (options.Mark) {
         // for tooltips use markers only if nodes were not created
         let path = "", pnt, grx, gry;

         this.createAttMarker({ attr: graph, style: options.Mark - 100 });

         this.marker_size = this.markeratt.getFullSize();

         this.markeratt.resetPos();

         let want_tooltip = !isBatchMode() && settings.Tooltip && (!this.markeratt.fill || (this.marker_size < 7)) && !nodes && main_block,
             hints_marker = "", hsz = Math.max(5, Math.round(this.marker_size*0.7)),
             maxnummarker = 1000000 / (this.markeratt.getMarkerLength() + 7), step = 1; // let produce SVG at maximum 1MB

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

         if (path.length > 0) {
            draw_g.append("svg:path")
                  .attr("d", path)
                  .call(this.markeratt.func);
            if ((nodes===null) && (this.draw_kind == "none") && main_block)
               this.draw_kind = (options.Mark == 101) ? "path" : "mark";
         }
         if (want_tooltip && hints_marker)
            draw_g.append("svg:path")
                .attr("d", hints_marker)
                .style("fill", "none")
                .style("pointer-events", "visibleFill");
      }
   }

   /** @summary append TGraphQQ part */
   appendQQ(funcs, graph) {
      let xqmin = Math.max(funcs.scale_xmin, graph.fXq1),
          xqmax = Math.min(funcs.scale_xmax, graph.fXq2),
          yqmin = Math.max(funcs.scale_ymin, graph.fYq1),
          yqmax = Math.min(funcs.scale_ymax, graph.fYq2),
          path2 = "",
          makeLine = (x1,y1,x2,y2) => `M${funcs.grx(x1)},${funcs.gry(y1)}L${funcs.grx(x2)},${funcs.gry(y2)}`;

      let yxmin = (graph.fYq2 - graph.fYq1)*(funcs.scale_xmin-graph.fXq1)/(graph.fXq2-graph.fXq1) + graph.fYq1;
      if (yxmin < funcs.scale_ymin){
         let xymin = (graph.fXq2 - graph.fXq1)*(funcs.scale_ymin-graph.fYq1)/(graph.fYq2-graph.fYq1) + graph.fXq1;
         path2 = makeLine(xymin, funcs.scale_ymin, xqmin, yqmin);
      } else {
         path2 = makeLine(funcs.scale_xmin, yxmin, xqmin, yqmin);
      }

      let yxmax = (graph.fYq2-graph.fYq1)*(funcs.scale_xmax-graph.fXq1)/(graph.fXq2-graph.fXq1) + graph.fYq1;
      if (yxmax > funcs.scale_ymax){
         let xymax = (graph.fXq2-graph.fXq1)*(funcs.scale_ymax-graph.fYq1)/(graph.fYq2-graph.fYq1) + graph.fXq1;
         path2 += makeLine(xqmax, yqmax, xymax, funcs.scale_ymax);
      } else {
         path2 += makeLine(xqmax, yqmax, funcs.scale_xmax, yxmax);
      }

      let latt1 = new TAttLineHandler({ style: 1, width: 1, color: "black" }),
          latt2 = new TAttLineHandler({ style: 2, width: 1, color: "black" });

      this.draw_g.append("path")
                 .attr("d", makeLine(xqmin,yqmin,xqmax,yqmax))
                 .call(latt1.func)
                 .style("fill","none");

      this.draw_g.append("path")
                 .attr("d", path2)
                 .call(latt2.func)
                 .style("fill","none");
   }

   drawBins3D(/*fp, graph*/) {
      console.log('Load ./hist/TGraphPainter.mjs to draw graph in 3D');
   }

   /** @summary draw TGraph */
   drawGraph() {

      let pmain = this.get_main(),
          graph = this.getObject();
      if (!pmain) return;

      // special mode for TMultiGraph 3d drawing
      if (this.options.pos3d)
         return this.drawBins3D(pmain, graph);

      let is_gme = !!this.get_gme(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          w = pmain.getFrameWidth(),
          h = pmain.getFrameHeight();

      this.createG(!pmain.pad_layer);

      if (this.options._pfc || this.options._plc || this.options._pmc) {
         let mp = this.getMainPainter();
         if (mp && mp.createAutoColor) {
            let icolor = mp.createAutoColor();
            if (this.options._pfc) { graph.fFillColor = icolor; delete this.fillatt; }
            if (this.options._plc) { graph.fLineColor = icolor; delete this.lineatt; }
            if (this.options._pmc) { graph.fMarkerColor = icolor; delete this.markeratt; }
            this.options._pfc = this.options._plc = this.options._pmc = false;
         }
      }

      this.createAttLine({ attr: graph, can_excl: true });
      this.createAttFill({ attr: graph });

      this.fillatt.used = false; // mark used only when really used

      this.draw_kind = "none"; // indicate if special svg:g were created for each bin
      this.marker_size = 0; // indicate if markers are drawn
      let draw_g = is_gme ? this.draw_g.append("svg:g") : this.draw_g;

      this.drawBins(funcs, this.options, draw_g, w, h, this.lineatt, this.fillatt, true);

      if (graph._typename == "TGraphQQ")
         this.appendQQ(funcs, graph);

      if (is_gme) {
         for (let k = 0; k < graph.fNYErrors; ++k) {
            let lineatt = this.lineatt, fillatt = this.fillatt;
            if (this.options.individual_styles) {
               lineatt = new TAttLineHandler({ attr: graph.fAttLine[k], std: false });
               fillatt = new TAttFillHandler({ attr: graph.fAttFill[k], std: false, svg: this.getCanvSvg() });
            }
            let sub_g = this.draw_g.append("svg:g");
            let options = k < this.options.blocks.length ? this.options.blocks[k] : this.options;
            this.extractGmeErrors(k);
            this.drawBins(funcs, options, sub_g, w, h, lineatt, fillatt);
         }
         this.extractGmeErrors(0); // ensure that first block kept at the end
      }

      if (!isBatchMode())
         addMoveHandler(this, this.testEditable());
   }

   /** @summary Provide tooltip at specified point */
   extractTooltip(pnt) {
      if (!pnt) return null;

      if ((this.draw_kind == "lines") || (this.draw_kind == "path") || (this.draw_kind == "mark"))
         return this.extractTooltipForPath(pnt);

      if (this.draw_kind != "nodes") return null;

      let pmain = this.getFramePainter(),
          height = pmain.getFrameHeight(),
          esz = this.error_size,
          isbar1 = (this.options.Bar === 1),
          funcs = isbar1 ? pmain.getGrFuncs(painter.options.second_x, painter.options.second_y) : null,
          findbin = null, best_dist2 = 1e10, best = null,
          msize = this.marker_size ? Math.round(this.marker_size/2 + 1.5) : 0;

      this.draw_g.selectAll('.grpoint').each(function() {
         let d = d3_select(this).datum();
         if (d===undefined) return;
         let dist2 = Math.pow(pnt.x - d.grx1, 2);
         if (pnt.nproc===1) dist2 += Math.pow(pnt.y - d.gry1, 2);
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
                let yy0 = funcs.gry(0);
                rect.y1 = (d.gry1 > yy0) ? yy0-d.gry1 : 0;
                rect.y2 = (d.gry1 > yy0) ? 0 : yy0-d.gry1;
             }
          } else {
             rect = { x1: -5, x2: 5, y1: -5, y2: 5 };
          }
          let matchx = (pnt.x >= d.grx1 + rect.x1) && (pnt.x <= d.grx1 + rect.x2),
              matchy = (pnt.y >= d.gry1 + rect.y1) && (pnt.y <= d.gry1 + rect.y2);

          if (matchx && (matchy || (pnt.nproc > 1))) {
             best_dist2 = dist2;
             findbin = this;
             best = rect;
             best.exact = /* matchx && */ matchy;
          }
       });

      if (findbin === null) return null;

      let d = d3_select(findbin).datum(),
          gr = this.getObject(),
          res = { name: gr.fName, title: gr.fTitle,
                  x: d.grx1, y: d.gry1,
                  color1: this.lineatt.color,
                  lines: this.getTooltips(d),
                  rect: best, d3bin: findbin  };

       res.user_info = { obj: gr,  name: gr.fName, bin: d.indx, cont: d.y, grx: d.grx1, gry: d.gry1 };

      if (this.fillatt && this.fillatt.used && !this.fillatt.empty()) res.color2 = this.fillatt.getFillColor();

      if (best.exact) res.exact = true;
      res.menu = res.exact; // activate menu only when exactly locate bin
      res.menu_dist = 3; // distance always fixed
      res.bin = d;
      res.binindx = d.indx;

      return res;
   }

   /** @summary Show tooltip */
   showTooltip(hint) {

      if (!hint) {
         if (this.draw_g) this.draw_g.select(".tooltip_bin").remove();
         return;
      }

      if (hint.usepath) return this.showTooltipForPath(hint);

      let d = d3_select(hint.d3bin).datum();

      let ttrect = this.draw_g.select(".tooltip_bin");

      if (ttrect.empty())
         ttrect = this.draw_g.append("svg:rect")
                             .attr("class","tooltip_bin h1bin")
                             .style("pointer-events","none");

      hint.changed = ttrect.property("current_bin") !== hint.d3bin;

      if (hint.changed)
         ttrect.attr("x", d.grx1 + hint.rect.x1)
               .attr("width", hint.rect.x2 - hint.rect.x1)
               .attr("y", d.gry1 + hint.rect.y1)
               .attr("height", hint.rect.y2 - hint.rect.y1)
               .style("opacity", "0.3")
               .property("current_bin", hint.d3bin);
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      let hint = this.extractTooltip(pnt);
      if (!pnt || !pnt.disabled) this.showTooltip(hint);
      return hint;
   }

   /** @summary Find best bin index for specified point */
   findBestBin(pnt) {
      if (!this.bins) return null;

      let islines = (this.draw_kind == "lines"),
          bestindx = -1,
          bestbin = null,
          bestdist = 1e10,
          pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          dist, grx, gry, n, bin;

      for (n = 0; n < this.bins.length; ++n) {
         bin = this.bins[n];

         grx = funcs.grx(bin.x);
         gry = funcs.gry(bin.y);

         dist = (pnt.x-grx)*(pnt.x-grx) + (pnt.y-gry)*(pnt.y-gry);

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
         bestdist = Math.sqrt(Math.pow(pnt.x-funcs.grx(bestbin.x),2) + Math.pow(pnt.y-funcs.gry(bestbin.y),2));

      if (!islines && (bestdist > radius)) bestbin = null;

      if (!bestbin) bestindx = -1;

      let res = { bin: bestbin, indx: bestindx, dist: bestdist, radius: Math.round(radius) };

      if (!bestbin && islines) {

         bestdist = 1e10;

         const IsInside = (x, x1, x2) => ((x1>=x) && (x>=x2)) || ((x1<=x) && (x<=x2));

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
      let obj = this.getObject();
      if (!obj) return false;
      if ((arg == "toggle") || ((arg!==undefined) && (!arg != obj.TestBit(kNotEditable))))
         obj.InvertBit(kNotEditable);
      return !obj.TestBit(kNotEditable);
   }

   /** @summary Provide tooltip at specified point for path-based drawing */
   extractTooltipForPath(pnt) {

      if (this.bins === null) return null;

      let best = this.findBestBin(pnt);

      if (!best || (!best.bin && !best.closeline)) return null;

      let islines = (this.draw_kind=="lines"),
          ismark = (this.draw_kind=="mark"),
          pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          gr = this.getObject(),
          res = { name: gr.fName, title: gr.fTitle,
                  x: best.bin ? funcs.grx(best.bin.x) : best.linex,
                  y: best.bin ? funcs.gry(best.bin.y) : best.liney,
                  color1: this.lineatt.color,
                  lines: this.getTooltips(best.bin),
                  usepath: true };

      res.user_info = { obj: gr,  name: gr.fName, bin: 0, cont: 0, grx: res.x, gry: res.y };

      res.ismark = ismark;
      res.islines = islines;

      if (best.closeline) {
         res.menu = res.exact = true;
         res.menu_dist = best.linedist;
      } else if (best.bin) {
         if (this.options.EF && islines) {
            res.gry1 = funcs.gry(best.bin.y - best.bin.eylow);
            res.gry2 = funcs.gry(best.bin.y + best.bin.eyhigh);
         } else {
            res.gry1 = res.gry2 = funcs.gry(best.bin.y);
         }

         res.binindx = best.indx;
         res.bin = best.bin;
         res.radius = best.radius;
         res.user_info.bin = best.indx;
         res.user_info.cont = best.bin.y;

         res.exact = (Math.abs(pnt.x - res.x) <= best.radius) &&
            ((Math.abs(pnt.y - res.gry1) <= best.radius) || (Math.abs(pnt.y - res.gry2) <= best.radius));

         res.menu = res.exact;
         res.menu_dist = Math.sqrt((pnt.x-res.x)*(pnt.x-res.x) + Math.pow(Math.min(Math.abs(pnt.y-res.gry1),Math.abs(pnt.y-res.gry2)),2));
      }

      if (this.fillatt && this.fillatt.used && !this.fillatt.empty())
         res.color2 = this.fillatt.getFillColor();

      if (!islines) {
         res.color1 = this.getColor(gr.fMarkerColor);
         if (!res.color2) res.color2 = res.color1;
      }

      return res;
   }

   /** @summary Show tooltip for path drawing */
   showTooltipForPath(hint) {

      let ttbin = this.draw_g.select(".tooltip_bin");

      if (!hint || !hint.bin) {
         ttbin.remove();
         return;
      }

      if (ttbin.empty())
         ttbin = this.draw_g.append("svg:g")
                             .attr("class","tooltip_bin");

      hint.changed = ttbin.property("current_bin") !== hint.bin;

      if (hint.changed) {
         ttbin.selectAll("*").remove(); // first delete all children
         ttbin.property("current_bin", hint.bin);

         if (hint.ismark) {
            ttbin.append("svg:rect")
                 .attr("class","h1bin")
                 .style("pointer-events","none")
                 .style("opacity", "0.3")
                 .attr("x", Math.round(hint.x - hint.radius))
                 .attr("y", Math.round(hint.y - hint.radius))
                 .attr("width", 2*hint.radius)
                 .attr("height", 2*hint.radius);
         } else {
            ttbin.append("svg:circle").attr("cy", Math.round(hint.gry1));
            if (Math.abs(hint.gry1-hint.gry2) > 1)
               ttbin.append("svg:circle").attr("cy", Math.round(hint.gry2));

            let elem = ttbin.selectAll("circle")
                            .attr("r", hint.radius)
                            .attr("cx", Math.round(hint.x));

            if (!hint.islines) {
               elem.style('stroke', hint.color1 == 'black' ? 'green' : 'black').style('fill','none');
            } else {
               if (this.options.Line || this.options.Curve)
                  elem.call(this.lineatt.func);
               else
                  elem.style('stroke','black');
               if (this.options.Fill)
                  elem.call(this.fillatt.func);
               else
                  elem.style('fill','none');
            }
         }
      }
   }

   /** @summary Check if graph moving is enabled */
   moveEnabled() {
      return this.testEditable();
   }

   /** @summary Start moving of TGraph */
   moveStart(x,y) {
      this.pos_dx = this.pos_dy = 0;
      let hint = this.extractTooltip({ x:x, y:y });
      if (hint && hint.exact && (hint.binindx !== undefined)) {
         this.move_binindx = hint.binindx;
         this.move_bin = hint.bin;
         let pmain = this.getFramePainter(),
             funcs = pmain ? pmain.getGrFuncs(this.options.second_x, this.options.second_y) : null;
         this.move_x0 = funcs ? funcs.grx(this.move_bin.x) : x;
         this.move_y0 = funcs ? funcs.gry(this.move_bin.y) : y;
      } else {
         delete this.move_binindx;
      }
   }

   /** @summary Perform moving */
   moveDrag(dx,dy) {
      this.pos_dx += dx;
      this.pos_dy += dy;

      if (this.move_binindx === undefined) {
         this.draw_g.attr("transform", `translate(${this.pos_dx},${this.pos_dy})`);
      } else {
         let pmain = this.getFramePainter(),
             funcs = pmain ? pmain.getGrFuncs(this.options.second_x, this.options.second_y) : null;
         if (funcs && this.move_bin) {
            this.move_bin.x = funcs.revertAxis("x", this.move_x0 + this.pos_dx);
            this.move_bin.y = funcs.revertAxis("y", this.move_y0 + this.pos_dy);
            this.drawGraph();
         }
      }
   }

   /** @summary Complete moving */
   moveEnd(not_changed) {
      let exec = "";

      if (this.move_binindx === undefined) {

         this.draw_g.attr("transform", null);

         let pmain = this.getFramePainter(),
             funcs = pmain ? pmain.getGrFuncs(this.options.second_x, this.options.second_y) : null;
         if (funcs && this.bins && !not_changed) {
            for (let k = 0; k < this.bins.length; ++k) {
               let bin = this.bins[k];
               bin.x = funcs.revertAxis("x", funcs.grx(bin.x) + this.pos_dx);
               bin.y = funcs.revertAxis("y", funcs.gry(bin.y) + this.pos_dy);
               exec += "SetPoint(" + bin.indx + "," + bin.x + "," + bin.y + ");;";
               if ((bin.indx == 0) && this.matchObjectType('TCutG'))
                  exec += "SetPoint(" + (this.getObject().fNpoints-1) + "," + bin.x + "," + bin.y + ");;";
            }
            this.drawGraph();
         }
      } else {
         exec = "SetPoint(" + this.move_bin.indx + "," + this.move_bin.x + "," + this.move_bin.y + ")";
         if ((this.move_bin.indx == 0) && this.matchObjectType('TCutG'))
            exec += ";;SetPoint(" + (this.getObject().fNpoints-1) + "," + this.move_bin.x + "," + this.move_bin.y + ")";
         delete this.move_binindx;
      }

      if (exec && !not_changed)
         this.submitCanvExec(exec);
   }

   /** @summary Fill context menu */
   fillContextMenu(menu) {
      super.fillContextMenu(menu);

      if (!this.snapid)
         menu.addchk(this.testEditable(), "Editable", () => { this.testEditable("toggle"); this.drawGraph(); });

      return menu.size() > 0;
   }

   /** @summary Execute menu command
     * @private */
   executeMenuCommand(method, args) {
      if (super.executeMenuCommand(method,args)) return true;

      let canp = this.getCanvPainter(), pmain = this.getFramePainter();

      if ((method.fName == 'RemovePoint') || (method.fName == 'InsertPoint')) {
         let pnt = pmain ? pmain.getLastEventPos() : null;

         if (!canp || canp._readonly || !pnt) return true; // ignore function

         let hint = this.extractTooltip(pnt);

         if (method.fName == 'InsertPoint') {
            let funcs = pmain ? pmain.getGrFuncs(this.options.second_x, this.options.second_y) : null,
                userx = funcs ? funcs.revertAxis("x", pnt.x) : 0,
                usery = funcs ? funcs.revertAxis("y", pnt.y) : 0;
            canp.showMessage('InsertPoint(' + userx.toFixed(3) + ',' + usery.toFixed(3) + ') not yet implemented');
         } else if (this.args_menu_id && hint && (hint.binindx !== undefined)) {
            this.submitCanvExec("RemovePoint(" + hint.binindx + ")", this.args_menu_id);
         }

         return true; // call is processed
      }

      return false;
   }

   /** @summary Update TGraph object */
   updateObject(obj, opt) {
      if (!this.matchObjectType(obj)) return false;

      if (opt && (opt != this.options.original))
         this.decodeOptions(opt);

      let graph = this.getObject();
      // TODO: make real update of TGraph object content
      graph.fBits = obj.fBits;
      graph.fTitle = obj.fTitle;
      graph.fX = obj.fX;
      graph.fY = obj.fY;
      graph.fNpoints = obj.fNpoints;
      graph.fMinimum = obj.fMinimum;
      graph.fMaximum = obj.fMaximum;
      this.createBins();

      delete this.$redraw_hist;

      // if our own histogram was used as axis drawing, we need update histogram as well
      if (this.axes_draw) {
         let histo = this.createHistogram(obj.fHistogram);
         histo.fTitle = graph.fTitle; // copy title

         let hist_painter = this.getMainPainter();
         if (hist_painter && hist_painter.$secondary) {
            hist_painter.updateObject(histo, this.options.Axis);
            this.$redraw_hist = true;
         }
      }

      return true;
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range
     * @desc allow to zoom TGraph only when at least one point in the range */
   canZoomInside(axis,min,max) {
      let gr = this.getObject();
      if (!gr || (axis !== (this.options.pos3d ? "y" : "x"))) return false;

      for (let n = 0; n < gr.fNpoints; ++n)
         if ((min < gr.fX[n]) && (gr.fX[n] < max)) return true;

      return false;
   }

   /** @summary Process click on graph-defined buttons */
   clickButton(funcname) {
      if (funcname !== "ToggleZoom") return false;

      let main = this.getFramePainter();
      if (!main) return false;

      if ((this.xmin === this.xmax) && (this.ymin === this.ymax)) return false;

      main.zoom(this.xmin, this.xmax, this.ymin, this.ymax);

      return true;
   }

   /** @summary Find TF1/TF2 in TGraph list of functions */
   findFunc() {
      let gr = this.getObject();
      if (gr && gr.fFunctions)
         for (let i = 0; i < gr.fFunctions.arr.length; ++i) {
            let func = gr.fFunctions.arr[i];
            if ((func._typename == 'TF1') || (func._typename == 'TF2')) return func;
         }
      return null;
   }

   /** @summary Find stat box in TGraph list of functions */
   findStat() {
      let gr = this.getObject();
      if (gr && gr.fFunctions)
         for (let i = 0; i < gr.fFunctions.arr.length; ++i) {
            let func = gr.fFunctions.arr[i];
            if ((func._typename == 'TPaveStats') && (func.fName == 'stats')) return func;
         }

      return null;
   }

   /** @summary Create stat box */
   createStat() {
      let func = this.findFunc();
      if (!func) return null;

      let stats = this.findStat();
      if (stats) return stats;

      // do not create stats box when drawing canvas
      let pp = this.getCanvPainter();
      if (pp && pp.normal_canvas) return null;

      if (this.options.PadStats) return null;

      this.create_stats = true;

      let st = gStyle;

      stats = create('TPaveStats');
      Object.assign(stats, { fName : 'stats',
                             fOptStat: 0,
                             fOptFit: st.fOptFit || 111,
                             fBorderSize : 1} );

      stats.fX1NDC = st.fStatX - st.fStatW;
      stats.fY1NDC = st.fStatY - st.fStatH;
      stats.fX2NDC = st.fStatX;
      stats.fY2NDC = st.fStatY;

      stats.fFillColor = st.fStatColor;
      stats.fFillStyle = st.fStatStyle;

      stats.fTextAngle = 0;
      stats.fTextSize = st.fStatFontSize; // 9 ??
      stats.fTextAlign = 12;
      stats.fTextColor = st.fStatTextColor;
      stats.fTextFont = st.fStatFont;

      stats.AddText(func.fName);

      // while TF1 was found, one can be sure that stats is existing
      this.getObject().fFunctions.Add(stats);

      return stats;
   }

   /** @summary Fill statistic */
   fillStatistic(stat, dostat, dofit) {

      // cannot fill stats without func
      let func = this.findFunc();

      if (!func || !dofit || !this.create_stats) return false;

      stat.clearPave();

      stat.fillFunctionStat(func, dofit);

      return true;
   }

   /** @summary method draws next function from the functions list
     * @returns {Promise} */
   drawNextFunction(indx) {

      let graph = this.getObject();

      if (!graph.fFunctions || (indx >= graph.fFunctions.arr.length))
         return Promise.resolve(this);

      let pp = this.getPadPainter(),
          func = graph.fFunctions.arr[indx],
          opt = graph.fFunctions.opt[indx];

      //  required for stats filling
      // TODO: use weak reference (via pad list of painters and any kind of string)
      func.$main_painter = this;

      return pp.drawObject(this.getDom(), func, opt).then(() => this.drawNextFunction(indx+1));
   }

   /** @summary Draw axis histogram
     * @private */
   drawAxisHisto() {
      let histo = this.createHistogram();
      return TH1Painter.draw(this.getDom(), histo, this.options.Axis)
   }

   /** @summary Draw TGraph
     * @private */
   static _drawGraph(painter, opt) {
      painter.decodeOptions(opt, true);
      painter.createBins();
      painter.createStat();
      if (!settings.DragGraphs && !graph.TestBit(kNotEditable))
         graph.InvertBit(kNotEditable);

      let promise = Promise.resolve();

      if ((!painter.getMainPainter() || painter.options.second_x || painter.options.second_y) && painter.options.Axis)
         promise = painter.drawAxisHisto().then(hist_painter => {
            if (hist_painter) {
               painter.axes_draw = true;
               if (!painter._own_histogram) painter.$primary = true;
               hist_painter.$secondary = true;
            }
         });

      return promise.then(() => {
         painter.addToPadPrimitives();
         return painter.drawGraph();
      }).then(() => painter.drawNextFunction(0));
   }

   static draw(dom, graph, opt) {
      return TGraphPainter._drawGraph(new TGraphPainter(dom, graph), opt);
   }

} // class TGraphPainter

export { TGraphPainter };
