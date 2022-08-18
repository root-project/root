import { gStyle, internals, createHistogram, createTPolyLine, isBatchMode } from '../core.mjs';
import { rgb as d3_rgb, chord as d3_chord, arc as d3_arc, ribbon as d3_ribbon } from '../d3.mjs';
import { TAttLineHandler } from '../base/TAttLineHandler.mjs';
import { TAttMarkerHandler } from '../base/TAttMarkerHandler.mjs';
import { TRandom, floatToString } from '../base/BasePainter.mjs';
import { EAxisBits } from '../gpad/TAxisPainter.mjs';
import { THistPainter } from './THistPainter.mjs';

/**
 * @summary Painter for TH2 classes
 * @private
 */

class TH2Painter extends THistPainter {

   /** @summary constructor
     * @param {object} histo - histogram object */
   constructor(dom, histo) {
      super(dom, histo);
      this.fPalette = null;
      this.wheel_zoomy = true;
      this._show_empty_bins = false;
   }

   /** @summary cleanup painter */
   cleanup() {
      delete this.tt_handle;

      super.cleanup();
   }

   /** @summary Toggle projection */
   toggleProjection(kind, width) {

      if ((kind == "Projections") || (kind == "Off")) kind = "";

      if ((typeof kind == 'string') && (kind.length>1)) {
          width = parseInt(kind.slice(1));
          kind = kind[0];
      }

      if (!width) width = 1;

      if (kind && (this.is_projection == kind)) {
         if (this.projection_width === width) {
            kind = "";
         } else {
            this.projection_width = width;
            return;
         }
      }

      delete this.proj_hist;

      let new_proj = (this.is_projection === kind) ? "" : kind;
      this.projection_width = width;
      this.is_projection = ""; // avoid projection handling until area is created

      this.provideSpecialDrawArea(new_proj).then(() => { this.is_projection = new_proj; return this.redrawProjection(); });
   }

   /** @summary Redraw projection */
   redrawProjection(ii1, ii2, jj1, jj2) {
      if (!this.is_projection) return;

      if (jj2 === undefined) {
         if (!this.tt_handle) return;
         ii1 = Math.round((this.tt_handle.i1 + this.tt_handle.i2)/2); ii2 = ii1+1;
         jj1 = Math.round((this.tt_handle.j1 + this.tt_handle.j2)/2); jj2 = jj1+1;
      }

      let canp = this.getCanvPainter(), histo = this.getHisto();

      if (canp && !canp._readonly && (this.snapid !== undefined)) {
         // this is when projection should be created on the server side
         let exec = `EXECANDSEND:D${this.is_projection}PROJ:${this.snapid}:`;
         if (this.is_projection == "X")
            exec += `ProjectionX("_projx",${jj1+1},${jj2},"")`;
         else
            exec += `ProjectionY("_projy",${ii1+1},${ii2},"")`;
         canp.sendWebsocket(exec);
         return;
      }

      if (!this.proj_hist) {
         if (this.is_projection == "X") {
            this.proj_hist = createHistogram("TH1D", this.nbinsx);
            Object.assign(this.proj_hist.fXaxis, histo.fXaxis);
            this.proj_hist.fName = "xproj";
            this.proj_hist.fTitle = "X projection";
         } else {
            this.proj_hist = createHistogram("TH1D", this.nbinsy);
            Object.assign(this.proj_hist.fXaxis, histo.fYaxis);
            this.proj_hist.fName = "yproj";
            this.proj_hist.fTitle = "Y projection";
         }
      }


      let first = 0, last = -1;
      if (this.is_projection == "X") {
         for (let i = 0; i < this.nbinsx; ++i) {
            let sum = 0;
            for (let j = jj1; j < jj2; ++j)
               sum += histo.getBinContent(i+1,j+1);
            this.proj_hist.setBinContent(i+1, sum);
         }
         this.proj_hist.fTitle = "X projection " + (jj1+1 == jj2 ? `bin ${jj2}` : `bins [${jj1+1} .. ${jj2}]`);
         if (this.tt_handle) { first = this.tt_handle.i1+1; last = this.tt_handle.i2; }

      } else {
         for (let j = 0; j < this.nbinsy; ++j) {
            let sum = 0;
            for (let i = ii1; i < ii2; ++i)
               sum += histo.getBinContent(i+1,j+1);
            this.proj_hist.setBinContent(j+1, sum);
         }
         this.proj_hist.fTitle = "Y projection " + (ii1+1 == ii2 ? `bin ${ii2}` : `bins [${ii1+1} .. ${ii2}]`);
         if (this.tt_handle) { first = this.tt_handle.j1+1; last = this.tt_handle.j2; }
      }

      if (first < last) {
         let axis = this.proj_hist.fXaxis;
         axis.fFirst = first;
         axis.fLast = last;

         if (((axis.fFirst == 1) && (axis.fLast == axis.fNbins)) == axis.TestBit(EAxisBits.kAxisRange))
            axis.InvertBit(EAxisBits.kAxisRange);
      }

      // reset statistic before display
      this.proj_hist.fEntries = 0;
      this.proj_hist.fTsumw = 0;

      return this.drawInSpecialArea(this.proj_hist);
   }

   /** @summary Execute TH2 menu command
     * @desc Used to catch standard menu items and provide local implementation */
   executeMenuCommand(method, args) {
      if (super.executeMenuCommand(method, args)) return true;

      if ((method.fName == 'SetShowProjectionX') || (method.fName == 'SetShowProjectionY')) {
         this.toggleProjection(method.fName[17], args && parseInt(args) ? parseInt(args) : 1);
         return true;
      }

      return false;
   }

   /** @summary Fill histogram context menu */
   fillHistContextMenu(menu) {
      if (!this.isTH2Poly()) {
         menu.add("sub:Projections", () => this.toggleProjection());
         let kind = this.is_projection || "";
         if (kind) kind += this.projection_width;
         const kinds = ["X1", "X2", "X3", "X5", "X10", "Y1", "Y2", "Y3", "Y5", "Y10"];
         if (this.is_projection) kinds.push("Off");
         for (let k = 0; k < kinds.length; ++k)
            menu.addchk(kind==kinds[k], kinds[k], kinds[k], arg => this.toggleProjection(arg));
         menu.add("endsub:");

         menu.add("Auto zoom-in", () => this.autoZoom());
      }

      let opts = this.getSupportedDrawOptions();

      menu.addDrawMenu("Draw with", opts, arg => {
         if (arg == 'inspect')
            return this.showInspector();
         this.decodeOptions(arg);
         this.interactiveRedraw("pad", "drawopt");
      });

      if (this.options.Color)
         this.fillPaletteMenu(menu);
   }

   /** @summary Process click on histogram-defined buttons */
   clickButton(funcname) {
      if (super.clickButton(funcname)) return true;

      if (this !== this.getMainPainter()) return false;

      switch(funcname) {
         case "ToggleColor": this.toggleColor(); break;
         case "ToggleColorZ": this.toggleColz(); break;
         case "Toggle3D": this.toggleMode3D(); break;
         default: return false;
      }

      // all methods here should not be processed further
      return true;
   }

   /** @summary Fill pad toolbar with histogram-related functions */
   fillToolbar() {
      super.fillToolbar(true);

      let pp = this.getPadPainter();
      if (!pp) return;

      if (!this.isTH2Poly())
         pp.addPadButton("th2color", "Toggle color", "ToggleColor");
      pp.addPadButton("th2colorz", "Toggle color palette", "ToggleColorZ");
      pp.addPadButton("th2draw3d", "Toggle 3D mode", "Toggle3D");
      pp.showPadButtons();
   }

   /** @summary Toggle color drawing mode */
   toggleColor() {

      if (this.options.Mode3D) {
         this.options.Mode3D = false;
         this.options.Color = true;
      } else {
         this.options.Color = !this.options.Color;
      }

      this._can_move_colz = true; // indicate that next redraw can move Z scale

      this.copyOptionsToOthers();

      this.redrawPad();
   }

   /** @summary Perform automatic zoom inside non-zero region of histogram */
   autoZoom() {
      if (this.isTH2Poly()) return; // not implemented

      let i1 = this.getSelectIndex("x", "left", -1),
          i2 = this.getSelectIndex("x", "right", 1),
          j1 = this.getSelectIndex("y", "left", -1),
          j2 = this.getSelectIndex("y", "right", 1),
          i,j, histo = this.getObject();

      if ((i1 == i2) || (j1 == j2)) return;

      // first find minimum
      let min = histo.getBinContent(i1 + 1, j1 + 1);
      for (i = i1; i < i2; ++i)
         for (j = j1; j < j2; ++j)
            min = Math.min(min, histo.getBinContent(i+1, j+1));
      if (min > 0) return; // if all points positive, no chance for autoscale

      let ileft = i2, iright = i1, jleft = j2, jright = j1;

      for (i = i1; i < i2; ++i)
         for (j = j1; j < j2; ++j)
            if (histo.getBinContent(i + 1, j + 1) > min) {
               if (i < ileft) ileft = i;
               if (i >= iright) iright = i + 1;
               if (j < jleft) jleft = j;
               if (j >= jright) jright = j + 1;
            }

      let xmin, xmax, ymin, ymax, isany = false;

      if ((ileft === iright-1) && (ileft > i1+1) && (iright < i2-1)) { ileft--; iright++; }
      if ((jleft === jright-1) && (jleft > j1+1) && (jright < j2-1)) { jleft--; jright++; }

      if ((ileft > i1 || iright < i2) && (ileft < iright - 1)) {
         xmin = histo.fXaxis.GetBinLowEdge(ileft+1);
         xmax = histo.fXaxis.GetBinLowEdge(iright+1);
         isany = true;
      }

      if ((jleft > j1 || jright < j2) && (jleft < jright - 1)) {
         ymin = histo.fYaxis.GetBinLowEdge(jleft+1);
         ymax = histo.fYaxis.GetBinLowEdge(jright+1);
         isany = true;
      }

      if (isany)
         return this.getFramePainter().zoom(xmin, xmax, ymin, ymax);
   }

   /** @summary Scan TH2 histogram content */
   scanContent(when_axis_changed) {

      // no need to rescan histogram while result does not depend from axis selection
      if (when_axis_changed && this.nbinsx && this.nbinsy) return;

      let i, j, histo = this.getObject();

      this.extractAxesProperties(2);

      if (this.isTH2Poly()) {
         this.gminposbin = null;
         this.gminbin = this.gmaxbin = 0;

         for (let n = 0, len=histo.fBins.arr.length; n < len; ++n) {
            let bin_content = histo.fBins.arr[n].fContent;
            if (n===0) this.gminbin = this.gmaxbin = bin_content;

            if (bin_content < this.gminbin) this.gminbin = bin_content; else
               if (bin_content > this.gmaxbin) this.gmaxbin = bin_content;

            if (bin_content > 0)
               if ((this.gminposbin===null) || (this.gminposbin > bin_content)) this.gminposbin = bin_content;
         }
      } else {
         // global min/max, used at the moment in 3D drawing
         this.gminbin = this.gmaxbin = histo.getBinContent(1, 1);
         this.gminposbin = null;
         for (i = 0; i < this.nbinsx; ++i) {
            for (j = 0; j < this.nbinsy; ++j) {
               let bin_content = histo.getBinContent(i+1, j+1);
               if (bin_content < this.gminbin)
                  this.gminbin = bin_content;
               else if (bin_content > this.gmaxbin)
                  this.gmaxbin = bin_content;
               if (bin_content > 0)
                  if ((this.gminposbin===null) || (this.gminposbin > bin_content)) this.gminposbin = bin_content;
            }
         }
      }

      // this value used for logz scale drawing
      if (this.gminposbin === null) this.gminposbin = this.gmaxbin*1e-4;

      if (this.options.Axis > 0) {
         // Paint histogram axis only
         this.draw_content = false;
      } else {
         this.draw_content = (this.gmaxbin > 0);
         if (!this.draw_content  && this.options.Zero && this.isTH2Poly()) {
            this.draw_content = true;
            this.options.Line = 1;
         }
      }
   }

   /** @summary Count TH2 histogram statistic
     * @desc Optionally one could provide condition function to select special range */
   countStat(cond) {
      let histo = this.getHisto(), xaxis = histo.fXaxis, yaxis = histo.fYaxis,
          stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0,
          stat_sumx2 = 0, stat_sumy2 = 0,
          xside, yside, xx, yy, zz,
          fp = this.getFramePainter(),
          funcs = fp.getGrFuncs(this.options.second_x, this.options.second_y),
          res = { name: histo.fName, entries: 0, integral: 0, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, matrix: [0,0,0,0,0,0,0,0,0], xmax: 0, ymax:0, wmax: null };

      if (this.isTH2Poly()) {

         let len = histo.fBins.arr.length, i, bin, n, gr, ngr, numgraphs, numpoints;

         for (i = 0; i < len; ++i) {
            bin = histo.fBins.arr[i];

            xside = 1; yside = 1;

            if (bin.fXmin > funcs.scale_xmax) xside = 2; else
            if (bin.fXmax < funcs.scale_xmin) xside = 0;
            if (bin.fYmin > funcs.scale_ymax) yside = 2; else
            if (bin.fYmax < funcs.scale_ymin) yside = 0;

            xx = yy = numpoints = 0;
            gr = bin.fPoly; numgraphs = 1;
            if (gr._typename === 'TMultiGraph') { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

            for (ngr = 0; ngr < numgraphs; ++ngr) {
               if (!gr || (ngr>0)) gr = bin.fPoly.fGraphs.arr[ngr];

               for (n = 0; n < gr.fNpoints; ++n) {
                  ++numpoints;
                  xx += gr.fX[n];
                  yy += gr.fY[n];
               }
            }

            if (numpoints > 1) {
               xx = xx / numpoints;
               yy = yy / numpoints;
            }

            zz = bin.fContent;

            res.entries += zz;

            res.matrix[yside * 3 + xside] += zz;

            if ((xside != 1) || (yside != 1)) continue;

            if (cond && !cond(xx,yy)) continue;

            if ((res.wmax === null) || (zz > res.wmax)) { res.wmax = zz; res.xmax = xx; res.ymax = yy; }

            stat_sum0 += zz;
            stat_sumx1 += xx * zz;
            stat_sumy1 += yy * zz;
            stat_sumx2 += xx * xx * zz;
            stat_sumy2 += yy * yy * zz;
         }
      } else {
         let xleft = this.getSelectIndex("x", "left"),
             xright = this.getSelectIndex("x", "right"),
             yleft = this.getSelectIndex("y", "left"),
             yright = this.getSelectIndex("y", "right"),
             xi, yi;

         for (xi = 0; xi <= this.nbinsx + 1; ++xi) {
            xside = (xi <= xleft) ? 0 : (xi > xright ? 2 : 1);
            xx = xaxis.GetBinCoord(xi - 0.5);

            for (yi = 0; yi <= this.nbinsy + 1; ++yi) {
               yside = (yi <= yleft) ? 0 : (yi > yright ? 2 : 1);
               yy = yaxis.GetBinCoord(yi - 0.5);

               zz = histo.getBinContent(xi, yi);

               res.entries += zz;

               res.matrix[yside * 3 + xside] += zz;

               if ((xside != 1) || (yside != 1)) continue;

               if (cond && !cond(xx,yy)) continue;

               if ((res.wmax === null) || (zz > res.wmax)) { res.wmax = zz; res.xmax = xx; res.ymax = yy; }

               stat_sum0 += zz;
               stat_sumx1 += xx * zz;
               stat_sumy1 += yy * zz;
               stat_sumx2 += xx**2 * zz;
               stat_sumy2 += yy**2 * zz;
               // stat_sumxy += xx * yy * zz;
            }
         }
      }

      if (!fp.isAxisZoomed("x") && !fp.isAxisZoomed("y") && (histo.fTsumw > 0)) {
         stat_sum0 = histo.fTsumw;
         stat_sumx1 = histo.fTsumwx;
         stat_sumx2 = histo.fTsumwx2;
         stat_sumy1 = histo.fTsumwy;
         stat_sumy2 = histo.fTsumwy2;
         // stat_sumxy = histo.fTsumwxy;
      }

      if (stat_sum0 > 0) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.rmsx = Math.sqrt(Math.abs(stat_sumx2 / stat_sum0 - res.meanx**2));
         res.rmsy = Math.sqrt(Math.abs(stat_sumy2 / stat_sum0 - res.meany**2));
      }

      if (res.wmax===null) res.wmax = 0;
      res.integral = stat_sum0;

      if (histo.fEntries > 1) res.entries = histo.fEntries;

      return res;
   }

   /** @summary Fill TH2 statistic in stat box */
   fillStatistic(stat, dostat, dofit) {

      // no need to refill statistic if histogram is dummy
      if (this.isIgnoreStatsFill()) return false;

      let data = this.countStat(),
          print_name = Math.floor(dostat % 10),
          print_entries = Math.floor(dostat / 10) % 10,
          print_mean = Math.floor(dostat / 100) % 10,
          print_rms = Math.floor(dostat / 1000) % 10,
          print_under = Math.floor(dostat / 10000) % 10,
          print_over = Math.floor(dostat / 100000) % 10,
          print_integral = Math.floor(dostat / 1000000) % 10,
          print_skew = Math.floor(dostat / 10000000) % 10,
          print_kurt = Math.floor(dostat / 100000000) % 10;

      stat.clearPave();

      if (print_name > 0)
         stat.addText(data.name);

      if (print_entries > 0)
         stat.addText("Entries = " + stat.format(data.entries,"entries"));

      if (print_mean > 0) {
         stat.addText("Mean x = " + stat.format(data.meanx));
         stat.addText("Mean y = " + stat.format(data.meany));
      }

      if (print_rms > 0) {
         stat.addText("Std Dev x = " + stat.format(data.rmsx));
         stat.addText("Std Dev y = " + stat.format(data.rmsy));
      }

      if (print_integral > 0)
         stat.addText("Integral = " + stat.format(data.matrix[4],"entries"));

      if (print_skew > 0) {
         stat.addText("Skewness x = <undef>");
         stat.addText("Skewness y = <undef>");
      }

      if (print_kurt > 0)
         stat.addText("Kurt = <undef>");

      if ((print_under > 0) || (print_over > 0)) {
         let m = data.matrix;

         stat.addText("" + m[6].toFixed(0) + " | " + m[7].toFixed(0) + " | "  + m[7].toFixed(0));
         stat.addText("" + m[3].toFixed(0) + " | " + m[4].toFixed(0) + " | "  + m[5].toFixed(0));
         stat.addText("" + m[0].toFixed(0) + " | " + m[1].toFixed(0) + " | "  + m[2].toFixed(0));
      }

      if (dofit) stat.fillFunctionStat(this.findFunction('TF2'), dofit);

      return true;
   }

   /** @summary Draw TH2 bins as colors */
   drawBinsColor() {
      const histo = this.getHisto(),
            handle = this.prepareDraw(),
            cntr = this.getContour(),
            palette = this.getHistPalette(),
            entries = [],
            show_empty = this._show_empty_bins,
            can_merge = (handle.ybar2 === 1) && (handle.ybar1 === 0);

      let dx, dy, x1, y2, binz, is_zero, colindx, last_entry = null,
          skip_zero = !this.options.Zero;

      const flush_last_entry = () => {
         last_entry.path += `h${dx}v${last_entry.y1-last_entry.y2}h${-dx}z`;
         last_entry = null;
      };

      // check in the beginning if zero can be skipped
      if (!skip_zero && !show_empty && (cntr.getPaletteIndex(palette, 0) === null)) skip_zero = true;

      // now start build
      for (let i = handle.i1; i < handle.i2; ++i) {

         dx = handle.grx[i+1] - handle.grx[i];
         x1 = Math.round(handle.grx[i] + dx*handle.xbar1);
         dx = Math.round(dx*(handle.xbar2 - handle.xbar1)) || 1;

         for (let j = handle.j2 - 1; j >= handle.j1; --j) {
            binz = histo.getBinContent(i + 1, j + 1);
            is_zero = (binz === 0);

            if (is_zero && skip_zero) {
               if (last_entry) flush_last_entry();
               continue;
            }

            colindx = cntr.getPaletteIndex(palette, binz);
            if (colindx === null) {
               if (is_zero && show_empty) {
                  colindx = 0;
                } else {
                   if (last_entry) flush_last_entry();
                   continue;
                }
            }

            dy = (handle.gry[j] - handle.gry[j+1]) || 1;
            if (can_merge) {
               y2 = handle.gry[j+1];
            } else {
               y2 = Math.round(handle.gry[j+1] + dy*handle.ybar2);
               dy = Math.round(dy*(handle.ybar2 - handle.ybar1)) || 1;
            }

            let cmd1 = `M${x1},${y2}`,
                entry = entries[colindx];
            if (!entry) {
               entry = entries[colindx] = { path: cmd1 };
            } else if (can_merge && (entry === last_entry)) {
               entry.y1 = y2 + dy;
               continue;
            } else {
               let ddx = x1 - entry.x1, ddy = y2 - entry.y2;
               if (ddx || ddy) {
                  let cmd2 = `m${ddx},${ddy}`;
                  entry.path += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
               }

            }
            if (last_entry) flush_last_entry();

            entry.x1 = x1;
            entry.y2 = y2;

            if (can_merge) {
               entry.y1 = y2 + dy;
               last_entry = entry;
            } else {
               entry.path += `h${dx}v${dy}h${-dx}z`;
            }
         }
         if (last_entry) flush_last_entry();
      }

      entries.forEach((entry,colindx) => {
        if (entry)
           this.draw_g
               .append("svg:path")
               .attr("fill", palette.getColor(colindx))
               .attr("d", entry.path);
      });

      return handle;
   }

   /** @summary Build histogram contour lines */
   buildContour(handle, levels, palette, contour_func) {

      const histo = this.getObject(),
            kMAXCONTOUR = 2004,
            kMAXCOUNT = 2000,
            // arguments used in the PaintContourLine
            xarr = new Float32Array(2*kMAXCONTOUR),
            yarr = new Float32Array(2*kMAXCONTOUR),
            itarr = new Int32Array(2*kMAXCONTOUR),
            nlevels = levels.length;
      let lj = 0, ipoly, poly, polys = [], np, npmax = 0,
          x = [0.,0.,0.,0.], y = [0.,0.,0.,0.], zc = [0.,0.,0.,0.], ir = [0,0,0,0],
          i, j, k, n, m, ljfill, count,
          xsave, ysave, itars, ix, jx;

      const BinarySearch = zc => {
         for (let kk = 0; kk < nlevels; ++kk)
            if (zc < levels[kk])
               return kk-1;
         return nlevels-1;
      }, PaintContourLine = (elev1, icont1, x1, y1,  elev2, icont2, x2, y2) => {
         /* Double_t *xarr, Double_t *yarr, Int_t *itarr, Double_t *levels */
         let vert = (x1 === x2),
             tlen = vert ? (y2 - y1) : (x2 - x1),
             n = icont1 +1,
             tdif = elev2 - elev1,
             ii = lj-1,
             maxii = kMAXCONTOUR/2 -3 + lj,
             icount = 0,
             xlen, pdif, diff, elev;

         while (n <= icont2 && ii <= maxii) {
//          elev = fH->GetContourLevel(n);
            elev = levels[n];
            diff = elev - elev1;
            pdif = diff/tdif;
            xlen = tlen*pdif;
            if (vert) {
               xarr[ii] = x1;
               yarr[ii] = y1 + xlen;
            } else {
               xarr[ii] = x1 + xlen;
               yarr[ii] = y1;
            }
            itarr[ii] = n;
            icount++;
            ii += 2;
            n++;
         }
         return icount;
      };

      let arrx = handle.original ? handle.origx : handle.grx,
          arry = handle.original ? handle.origy : handle.gry;

      for (j = handle.j1; j < handle.j2-1; ++j) {

         y[1] = y[0] = (arry[j] + arry[j+1])/2;
         y[3] = y[2] = (arry[j+1] + arry[j+2])/2;

         for (i = handle.i1; i < handle.i2-1; ++i) {

            zc[0] = histo.getBinContent(i+1, j+1);
            zc[1] = histo.getBinContent(i+2, j+1);
            zc[2] = histo.getBinContent(i+2, j+2);
            zc[3] = histo.getBinContent(i+1, j+2);

            for (k=0;k<4;k++)
               ir[k] = BinarySearch(zc[k]);

            if ((ir[0] !== ir[1]) || (ir[1] !== ir[2]) || (ir[2] !== ir[3]) || (ir[3] !== ir[0])) {
               x[3] = x[0] = (arrx[i] + arrx[i+1])/2;
               x[2] = x[1] = (arrx[i+1] + arrx[i+2])/2;

               if (zc[0] <= zc[1]) n = 0; else n = 1;
               if (zc[2] <= zc[3]) m = 2; else m = 3;
               if (zc[n] > zc[m]) n = m;
               n++;
               lj=1;
               for (ix=1;ix<=4;ix++) {
                  m = n%4 + 1;
                  ljfill = PaintContourLine(zc[n-1],ir[n-1],x[n-1],y[n-1],
                        zc[m-1],ir[m-1],x[m-1],y[m-1]);
                  lj += 2*ljfill;
                  n = m;
               }

               if (zc[0] <= zc[1]) n = 0; else n = 1;
               if (zc[2] <= zc[3]) m = 2; else m = 3;
               if (zc[n] > zc[m]) n = m;
               n++;
               lj=2;
               for (ix=1;ix<=4;ix++) {
                  if (n == 1) m = 4;
                  else        m = n-1;
                  ljfill = PaintContourLine(zc[n-1],ir[n-1],x[n-1],y[n-1],
                        zc[m-1],ir[m-1],x[m-1],y[m-1]);
                  lj += 2*ljfill;
                  n = m;
               }
               //     Re-order endpoints

               count = 0;
               for (ix=1; ix<=lj-5; ix +=2) {
                  //count = 0;
                  while (itarr[ix-1] != itarr[ix]) {
                     xsave = xarr[ix];
                     ysave = yarr[ix];
                     itars = itarr[ix];
                     for (jx=ix; jx<=lj-5; jx +=2) {
                        xarr[jx]  = xarr[jx+2];
                        yarr[jx]  = yarr[jx+2];
                        itarr[jx] = itarr[jx+2];
                     }
                     xarr[lj-3]  = xsave;
                     yarr[lj-3]  = ysave;
                     itarr[lj-3] = itars;
                     if (count > kMAXCOUNT) break;
                     count++;
                  }
               }

               if (count > kMAXCOUNT) continue;

               for (ix=1; ix<=lj-2; ix +=2) {

                  ipoly = itarr[ix-1];

                  if ((ipoly >= 0) && (ipoly < levels.length)) {
                     poly = polys[ipoly];
                     if (!poly)
                        poly = polys[ipoly] = createTPolyLine(kMAXCONTOUR*4, true);

                     np = poly.fLastPoint;
                     if (np < poly.fN-2) {
                        poly.fX[np+1] = Math.round(xarr[ix-1]); poly.fY[np+1] = Math.round(yarr[ix-1]);
                        poly.fX[np+2] = Math.round(xarr[ix]); poly.fY[np+2] = Math.round(yarr[ix]);
                        poly.fLastPoint = np+2;
                        npmax = Math.max(npmax, poly.fLastPoint+1);
                     } else {
                        // console.log('reject point??', poly.fLastPoint);
                     }
                  }
               }
            } // end of if (ir[0]
         } // end of j
      } // end of i

      let polysort = new Int32Array(levels.length), first = 0;
      //find first positive contour
      for (ipoly = 0; ipoly < levels.length; ipoly++) {
         if (levels[ipoly] >= 0) { first = ipoly; break; }
      }
      //store negative contours from 0 to minimum, then all positive contours
      k = 0;
      for (ipoly = first-1; ipoly >= 0; ipoly--) { polysort[k] = ipoly; k++; }
      for (ipoly = first; ipoly < levels.length; ipoly++) { polysort[k] = ipoly; k++; }

      let xp = new Float32Array(2*npmax),
          yp = new Float32Array(2*npmax);

      for (k = 0; k < levels.length; ++k) {

         ipoly = polysort[k];
         poly = polys[ipoly];
         if (!poly) continue;

         let colindx = palette.calcColorIndex(ipoly, levels.length),
             xx = poly.fX, yy = poly.fY, np = poly.fLastPoint+1,
             istart = 0, iminus, iplus, xmin = 0, ymin = 0, nadd;

         while (true) {

            iminus = npmax;
            iplus  = iminus+1;
            xp[iminus]= xx[istart];   yp[iminus] = yy[istart];
            xp[iplus] = xx[istart+1]; yp[iplus]  = yy[istart+1];
            xx[istart] = xx[istart+1] = xmin;
            yy[istart] = yy[istart+1] = ymin;
            while (true) {
               nadd = 0;
               for (i=2;i<np;i+=2) {
                  if ((iplus < 2*npmax-1) && (xx[i] === xp[iplus]) && (yy[i] === yp[iplus])) {
                     iplus++;
                     xp[iplus] = xx[i+1]; yp[iplus] = yy[i+1];
                     xx[i] = xx[i+1] = xmin;
                     yy[i] = yy[i+1] = ymin;
                     nadd++;
                  }
                  if ((iminus > 0) && (xx[i+1] === xp[iminus]) && (yy[i+1] === yp[iminus])) {
                     iminus--;
                     xp[iminus] = xx[i]; yp[iminus] = yy[i];
                     xx[i] = xx[i+1] = xmin;
                     yy[i] = yy[i+1] = ymin;
                     nadd++;
                  }
               }
               if (nadd == 0) break;
            }

            if ((iminus+1 < iplus) && (iminus>=0))
               contour_func(colindx, xp, yp, iminus, iplus, ipoly);

            istart = 0;
            for (i=2;i<np;i+=2) {
               if (xx[i] !== xmin && yy[i] !== ymin) {
                  istart = i;
                  break;
               }
            }

            if (istart === 0) break;
         }
      }
   }

   /** @summary Draw histogram bins as contour */
   drawBinsContour() {
      let handle = this.prepareDraw({ rounding: false, extra: 100, original: this.options.Proj != 0 }),
          main = this.getFramePainter(),
          frame_w = main.getFrameWidth(),
          frame_h = main.getFrameHeight(),
          funcs = main.getGrFuncs(this.options.second_x, this.options.second_y),
          levels = this.getContourLevels(),
          palette = this.getHistPalette(),
          func = main.getProjectionFunc();

      const buildPath = (xp,yp,iminus,iplus,do_close) => {
         let cmd = "", lastx, lasty, x0, y0, isany = false, matched, x, y;
         for (let i = iminus; i <= iplus; ++i) {
            if (func) {
               let pnt = func(xp[i], yp[i]);
               x = Math.round(funcs.grx(pnt.x));
               y = Math.round(funcs.gry(pnt.y));
            } else {
               x = Math.round(xp[i]);
               y = Math.round(yp[i]);
            }
            if (!cmd) {
               cmd = `M${x},${y}`; x0 = x; y0 = y;
            } else if ((i == iplus) && (iminus !== iplus) && (x == x0) && (y == y0)) {
               if (!isany) return ""; // all same points
               cmd += "z"; do_close = false; matched = true;
            } else {
               let dx = x - lastx, dy = y - lasty;
               if (dx) {
                  isany = true;
                  cmd += dy ? `l${dx},${dy}` : `h${dx}`;
               } else if (dy) {
                  isany = true;
                  cmd += `v${dy}`;
               }
            }

            lastx = x; lasty = y;
         }

         if (do_close && !matched && !func)
            return "<failed>";
         if (do_close) cmd += "z";
         return cmd;

      }, get_segm_intersection = (segm1, segm2) => {

          let s02_x, s02_y, s10_x, s10_y, s32_x, s32_y, s_numer, t_numer, denom, t;
          s10_x = segm1.x2 - segm1.x1;
          s10_y = segm1.y2 - segm1.y1;
          s32_x = segm2.x2 - segm2.x1;
          s32_y = segm2.y2 - segm2.y1;

          denom = s10_x * s32_y - s32_x * s10_y;
          if (denom == 0)
              return 0; // Collinear
          let denomPositive = denom > 0;

          s02_x = segm1.x1 - segm2.x1;
          s02_y = segm1.y1 - segm2.y1;
          s_numer = s10_x * s02_y - s10_y * s02_x;
          if ((s_numer < 0) == denomPositive)
              return null; // No collision

          t_numer = s32_x * s02_y - s32_y * s02_x;
          if ((t_numer < 0) == denomPositive)
              return null; // No collision

          if (((s_numer > denom) == denomPositive) || ((t_numer > denom) == denomPositive))
              return null; // No collision
          // Collision detected
          t = t_numer / denom;
          return { x: Math.round(segm1.x1 + (t * s10_x)), y: Math.round(segm1.y1 + (t * s10_y)) };

      }, buildPathOutside = (xp,yp,iminus,iplus,side) => {

         // try to build path which fills area to outside borders

         const points = [{x: 0, y: 0}, {x: frame_w, y: 0}, {x: frame_w, y: frame_h}, {x: 0, y: frame_h}];

         const get_intersect = (i,di) => {
            let segm = { x1: xp[i], y1: yp[i], x2: 2*xp[i] - xp[i+di], y2: 2*yp[i] - yp[i+di] };
            for (let i = 0; i < 4; ++i) {
               let res = get_segm_intersection(segm, { x1: points[i].x, y1: points[i].y, x2: points[(i+1)%4].x, y2: points[(i+1)%4].y});
               if (res) {
                  res.indx = i + 0.5;
                  return res;
               }
            }
            return null;
         };

         let pnt1, pnt2;
         iminus--;
         while ((iminus < iplus - 1) && !pnt1)
            pnt1 = get_intersect(++iminus, 1);
         iplus++;
         while ((iminus < iplus - 1) && pnt1 && !pnt2)
            pnt2 = get_intersect(--iplus, -1);
         if (!pnt1 || !pnt2) return "";

         // TODO: now side is always same direction, could be that side should be checked more precise

         let dd = buildPath(xp,yp,iminus,iplus),
             indx = pnt2.indx, step = side*0.5;

         dd += `L${pnt2.x},${pnt2.y}`;

         while (Math.abs(indx - pnt1.indx) > 0.1) {
            indx = Math.round(indx + step) % 4;
            dd += `L${points[indx].x},${points[indx].y}`;
            indx += step;
         }

         return dd + `L${pnt1.x},${pnt1.y}z`;
      };

      if (this.options.Contour === 14) {
         let dd = `M0,0h${frame_w}v${frame_h}h${-frame_w}z`;
         if (this.options.Proj) {
            let sz = handle.j2 - handle.j1, xd = new Float32Array(sz*2), yd = new Float32Array(sz*2);
            for (let i = 0; i < sz; ++i) {
               xd[i] = handle.origx[handle.i1];
               yd[i] = (handle.origy[handle.j1]*(i+0.5) + handle.origy[handle.j2]*(sz-0.5-i))/sz;
               xd[i+sz] = handle.origx[handle.i2];
               yd[i+sz] = (handle.origy[handle.j2]*(i+0.5) + handle.origy[handle.j1]*(sz-0.5-i))/sz;
            }
            dd = buildPath(xd,yd,0,2*sz-1, true);
         }

         this.draw_g
             .append("svg:path")
             .attr("d", dd)
             .style("fill", palette.calcColor(0, levels.length));
      }

      this.buildContour(handle, levels, palette, (colindx,xp,yp,iminus,iplus,ipoly) => {
         let icol = palette.getColor(colindx),
             fillcolor = icol, lineatt;

         switch (this.options.Contour) {
            case 1: break;
            case 11: fillcolor = 'none'; lineatt = new TAttLineHandler({ color: icol } ); break;
            case 12: fillcolor = 'none'; lineatt = new TAttLineHandler({ color: 1, style: (ipoly%5 + 1), width: 1 }); break;
            case 13: fillcolor = 'none'; lineatt = this.lineatt; break;
            case 14: break;
         }

         let dd = buildPath(xp, yp, iminus, iplus, fillcolor != 'none');
         if (dd == "<failed>")
            dd = buildPathOutside(xp, yp, iminus, iplus, 1);
         if (!dd) return;

         let elem = this.draw_g
                        .append("svg:path")
                        .attr("class","th2_contour")
                        .attr("d", dd)
                        .style("fill", fillcolor);

         if (lineatt)
            elem.call(lineatt.func);
      });

      handle.hide_only_zeros = true; // text drawing suppress only zeros

      return handle;
   }

   /** @summary Create poly bin */
   createPolyBin(funcs, bin, text_pos) {
      let cmd = "", grcmd = "", acc_x = 0, acc_y = 0, ngr, ngraphs = 1, gr = null;

      if (bin.fPoly._typename == 'TMultiGraph')
         ngraphs = bin.fPoly.fGraphs.arr.length;
      else
         gr = bin.fPoly;

      if (text_pos)
         bin._sumx = bin._sumy = bin._suml = 0;

      const addPoint = (x1,y1,x2,y2) => {
         const len = Math.sqrt((x1-x2)**2 + (y1-y2)**2);
         bin._sumx += (x1+x2)*len/2;
         bin._sumy += (y1+y2)*len/2;
         bin._suml += len;
      };

      const flush = () => {
         if (acc_x) { grcmd += "h" + acc_x; acc_x = 0; }
         if (acc_y) { grcmd += "v" + acc_y; acc_y = 0; }
      };

      for (ngr = 0; ngr < ngraphs; ++ ngr) {
         if (!gr || (ngr > 0)) gr = bin.fPoly.fGraphs.arr[ngr];

         const x = gr.fX, y = gr.fY;
         let n, nextx, nexty, npnts = gr.fNpoints, dx, dy,
             grx = Math.round(funcs.grx(x[0])),
             gry = Math.round(funcs.gry(y[0]));

         if ((npnts > 2) && (x[0] === x[npnts-1]) && (y[0] === y[npnts-1])) npnts--;

         let poscmd = `M${grx},${gry}`;

         grcmd = "";

         for (n = 1; n < npnts; ++n) {
            nextx = Math.round(funcs.grx(x[n]));
            nexty = Math.round(funcs.gry(y[n]));
            if (text_pos) addPoint(grx,gry, nextx, nexty);
            dx = nextx - grx;
            dy = nexty - gry;
            if (dx || dy) {
               if (dx === 0) {
                  if ((acc_y === 0) || ((dy < 0) !== (acc_y < 0))) flush();
                  acc_y += dy;
               } else if (dy === 0) {
                  if ((acc_x === 0) || ((dx < 0) !== (acc_x < 0))) flush();
                  acc_x += dx;
               } else {
                  flush();
                  grcmd += "l" + dx + "," + dy;
               }

               grx = nextx; gry = nexty;
            }
         }

         if (text_pos) addPoint(grx, gry, Math.round(funcs.grx(x[0])), Math.round(funcs.gry(y[0])));
         flush();

         if (grcmd)
            cmd += poscmd + grcmd + "z";
      }

      if (text_pos) {
         if (bin._suml > 0) {
            bin._midx = Math.round(bin._sumx / bin._suml);
            bin._midy = Math.round(bin._sumy / bin._suml);
         } else {
            bin._midx = Math.round(funcs.grx((bin.fXmin + bin.fXmax)/2));
            bin._midy = Math.round(funcs.gry((bin.fYmin + bin.fYmax)/2));
         }
      }

      return cmd;
   }

   /** @summary draw TH2Poly as color */
   drawPolyBinsColor() {
      let histo = this.getObject(),
          pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          h = pmain.getFrameHeight(),
          colPaths = [], textbins = [],
          colindx, cmd, bin, item,
          i, len = histo.fBins.arr.length;

      // force recalculations of contours
      // use global coordinates
      this.maxbin = this.gmaxbin;
      this.minbin = this.gminbin;
      this.minposbin = this.gminposbin;

      let cntr = this.getContour(true), palette = this.getHistPalette();

      for (i = 0; i < len; ++ i) {
         bin = histo.fBins.arr[i];
         colindx = cntr.getPaletteIndex(palette, bin.fContent);
         if (colindx === null) continue;
         if (bin.fContent === 0) {
            if (!this.options.Zero || !this.options.Line) continue;
            colindx = 0; // make dummy fill color to draw only line
         }

         // check if bin outside visible range
         if ((bin.fXmin > funcs.scale_xmax) || (bin.fXmax < funcs.scale_xmin) ||
             (bin.fYmin > funcs.scale_ymax) || (bin.fYmax < funcs.scale_ymin)) continue;

         cmd = this.createPolyBin(funcs, bin, this.options.Text && bin.fContent);

         if (colPaths[colindx] === undefined)
            colPaths[colindx] = cmd;
         else
            colPaths[colindx] += cmd;

         if (this.options.Text && bin.fContent) textbins.push(bin);
      }

      for (colindx = 0; colindx < colPaths.length; ++colindx)
         if (colPaths[colindx]) {
            item = this.draw_g
                     .append("svg:path")
                     .style("fill", colindx ? this.fPalette.getColor(colindx) : 'none')
                     .attr("d", colPaths[colindx]);
            if (this.options.Line)
               item.call(this.lineatt.func);
         }

      let pr = Promise.resolve(true);

      if (textbins.length > 0) {
         let text_col = this.getColor(histo.fMarkerColor),
             text_angle = -1*this.options.TextAngle,
             text_g = this.draw_g.append("svg:g").attr("class","th2poly_text"),
             text_size = 12;

         if ((histo.fMarkerSize!==1) && text_angle)
             text_size = Math.round(0.02*h*histo.fMarkerSize);

         this.startTextDrawing(42, text_size, text_g, text_size);

         for (i = 0; i < textbins.length; ++ i) {
            bin = textbins[i];

            let lbl = "";

            if (!this.options.TextKind) {
               lbl = (Math.round(bin.fContent) === bin.fContent) ? bin.fContent.toString() :
                          floatToString(bin.fContent, gStyle.fPaintTextFormat);
            } else {
               if (bin.fPoly) lbl = bin.fPoly.fName;
               if (lbl === "Graph") lbl = "";
               if (!lbl) lbl = bin.fNumber;
            }

            this.drawText({ align: 22, x: bin._midx, y: bin._midy, rotate: text_angle, text: lbl, color: text_col, latex: 0, draw_g: text_g });
         }

         pr = this.finishTextDrawing(text_g, true);
      }

      return pr.then(() => { return { poly: true }; });
   }

   /** @summary Draw TH2 bins as text */
   drawBinsText(handle) {
      let histo = this.getObject(),
          x, y, width, height,
          color = this.getColor(histo.fMarkerColor),
          rotate = -1*this.options.TextAngle,
          draw_g = this.draw_g.append("svg:g").attr("class","th2_text"),
          text_size = 20, text_offset = 0,
          profile2d = this.matchObjectType('TProfile2D') && (typeof histo.getBinEntries == 'function'),
          show_err = (this.options.TextKind == "E"),
          latex = (show_err && !this.options.TextLine) ? 1 : 0;

      if (!handle) handle = this.prepareDraw({ rounding: false });

      if ((histo.fMarkerSize!==1) && rotate)
         text_size = Math.round(0.02*histo.fMarkerSize*this.getFramePainter().getFrameHeight());

      if (histo.fBarOffset !== 0) text_offset = histo.fBarOffset*1e-3;

      this.startTextDrawing(42, text_size, draw_g, text_size);

      for (let i = handle.i1; i < handle.i2; ++i) {
         let binw = handle.grx[i+1] - handle.grx[i];
         for (let j = handle.j1; j < handle.j2; ++j) {
            let binz = histo.getBinContent(i+1, j+1);
            if ((binz === 0) && !this._show_empty_bins) continue;
            let binh = handle.gry[j] - handle.gry[j+1];

            if (profile2d)
               binz = histo.getBinEntries(i+1, j+1);

            let text = (binz === Math.round(binz)) ? binz.toString() :
                         floatToString(binz, gStyle.fPaintTextFormat);

            if (show_err) {
               let errz = histo.getBinError(histo.getBin(i+1,j+1)),
                   lble = (errz === Math.round(errz)) ? errz.toString() :
                            floatToString(errz, gStyle.fPaintTextFormat);
               if (this.options.TextLine)
                  text += '\xB1' + lble;
               else
                  text = `#splitline{${text}}{#pm${lble}}`;
            }

            if (rotate /*|| (histo.fMarkerSize!==1)*/) {
               x = Math.round(handle.grx[i] + binw*0.5);
               y = Math.round(handle.gry[j+1] + binh*(0.5 + text_offset));
               width = height = 0;
            } else {
               x = Math.round(handle.grx[i] + binw*0.1);
               y = Math.round(handle.gry[j+1] + binh*(0.1 + text_offset));
               width = Math.round(binw*0.8);
               height = Math.round(binh*0.8);
            }

            this.drawText({ align: 22, x, y, width, height, rotate, text, color, latex, draw_g });
         }
      }

      handle.hide_only_zeros = true; // text drawing suppress only zeros

      return this.finishTextDrawing(draw_g, true).then(() => handle);
   }

   /** @summary Draw TH2 bins as arrows */
   drawBinsArrow() {
      let histo = this.getObject(), cmd = "",
          i,j, dn = 1e-30, dx, dy, xc,yc,
          dxn,dyn,x1,x2,y1,y2, anr,si,co,
          handle = this.prepareDraw({ rounding: false }),
          scale_x = (handle.grx[handle.i2] - handle.grx[handle.i1])/(handle.i2 - handle.i1 + 1)/2,
          scale_y = (handle.gry[handle.j2] - handle.gry[handle.j1])/(handle.j2 - handle.j1 + 1)/2;

      const makeLine = (dx, dy) => {
         if (dx)
            return dy ? `l${dx},${dy}` : `h${dx}`;
         return dy ? `v${dy}` : "";
      };

      for (let loop = 0; loop < 2; ++loop)
         for (i = handle.i1; i < handle.i2; ++i)
            for (j = handle.j1; j < handle.j2; ++j) {

               if (i === handle.i1)
                  dx = histo.getBinContent(i+2, j+1) - histo.getBinContent(i+1, j+1);
               else if (i === handle.i2-1)
                  dx = histo.getBinContent(i+1, j+1) - histo.getBinContent(i, j+1);
               else
                  dx = 0.5*(histo.getBinContent(i+2, j+1) - histo.getBinContent(i, j+1));

               if (j === handle.j1)
                  dy = histo.getBinContent(i+1, j+2) - histo.getBinContent(i+1, j+1);
               else if (j === handle.j2-1)
                  dy = histo.getBinContent(i+1, j+1) - histo.getBinContent(i+1, j);
               else
                  dy = 0.5*(histo.getBinContent(i+1, j+2) - histo.getBinContent(i+1, j));

               if (loop === 0) {
                  dn = Math.max(dn, Math.abs(dx), Math.abs(dy));
               } else {
                  xc = (handle.grx[i] + handle.grx[i+1])/2;
                  yc = (handle.gry[j] + handle.gry[j+1])/2;
                  dxn = scale_x*dx/dn;
                  dyn = scale_y*dy/dn;
                  x1  = xc - dxn;
                  x2  = xc + dxn;
                  y1  = yc - dyn;
                  y2  = yc + dyn;
                  dx = Math.round(x2-x1);
                  dy = Math.round(y2-y1);

                  if (dx || dy) {
                     cmd += "M"+Math.round(x1)+","+Math.round(y1) + makeLine(dx,dy);

                     if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                        anr = Math.sqrt(9/(dx**2 + dy**2));
                        si  = Math.round(anr*(dx + dy));
                        co  = Math.round(anr*(dx - dy));
                        if (si || co)
                           cmd += `m${-si},${co}` + makeLine(si,-co) + makeLine(-co,-si);
                     }
                  }
               }
            }

      this.draw_g
         .append("svg:path")
         .attr("d", cmd)
         .style("fill", "none")
         .call(this.lineatt.func);

      return handle;
   }

   /** @summary Draw TH2 bins as boxes */
   drawBinsBox() {

      let histo = this.getObject(),
          handle = this.prepareDraw({ rounding: false }),
          main = this.getMainPainter();

      if (main === this) {
         if (main.maxbin === main.minbin) {
            main.maxbin = main.gmaxbin;
            main.minbin = main.gminbin;
            main.minposbin = main.gminposbin;
         }
         if (main.maxbin === main.minbin)
            main.minbin = Math.min(0, main.maxbin-1);
      }

      let absmax = Math.max(Math.abs(main.maxbin), Math.abs(main.minbin)),
          absmin = Math.max(0, main.minbin),
          i, j, binz, absz, res = "", cross = "", btn1 = "", btn2 = "",
          zdiff, dgrx, dgry, xx, yy, ww, hh, xyfactor,
          uselogz = false, logmin = 0,
          pad = this.getPadPainter().getRootPad(true);

      if (pad && pad.fLogz && (absmax > 0)) {
         uselogz = true;
         let logmax = Math.log(absmax);
         if (absmin > 0)
            logmin = Math.log(absmin);
         else if ((main.minposbin>=1) && (main.minposbin<100))
            logmin = Math.log(0.7);
         else
            logmin = (main.minposbin > 0) ? Math.log(0.7*main.minposbin) : logmax - 10;
         if (logmin >= logmax) logmin = logmax - 10;
         xyfactor = 1. / (logmax - logmin);
      } else {
         xyfactor = 1. / (absmax - absmin);
      }

      // now start build
      for (i = handle.i1; i < handle.i2; ++i) {
         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            absz = Math.abs(binz);
            if ((absz === 0) || (absz < absmin)) continue;

            zdiff = uselogz ? ((absz > 0) ? Math.log(absz) - logmin : 0) : (absz - absmin);
            // area of the box should be proportional to absolute bin content
            zdiff = 0.5 * ((zdiff < 0) ? 1 : (1 - Math.sqrt(zdiff * xyfactor)));
            // avoid oversized bins
            if (zdiff < 0) zdiff = 0;

            ww = handle.grx[i+1] - handle.grx[i];
            hh = handle.gry[j] - handle.gry[j+1];

            dgrx = zdiff * ww;
            dgry = zdiff * hh;

            xx = Math.round(handle.grx[i] + dgrx);
            yy = Math.round(handle.gry[j+1] + dgry);

            ww = Math.max(Math.round(ww - 2*dgrx), 1);
            hh = Math.max(Math.round(hh - 2*dgry), 1);

            res += `M${xx},${yy}v${hh}h${ww}v${-hh}z`;

            if ((binz < 0) && (this.options.BoxStyle === 10))
               cross += `M${xx},${yy}l${ww},${hh}m0,${-hh}l${-ww},${hh}`;

            if ((this.options.BoxStyle === 11) && (ww>5) && (hh>5)) {
               const pww = Math.round(ww*0.1),
                     phh = Math.round(hh*0.1),
                     side1 = `M${xx},${yy}h${ww}l${-pww},${phh}h${2*pww-ww}v${hh-2*phh}l${-pww},${phh}z`,
                     side2 = `M${xx+ww},${yy+hh}v${-hh}l${-pww},${phh}v${hh-2*phh}h${2*pww-ww}l${-pww},${phh}z`;
               if (binz < 0) { btn2 += side1; btn1 += side2; }
                        else { btn1 += side1; btn2 += side2; }
            }
         }
      }

      if (res.length > 0) {
         let elem = this.draw_g.append("svg:path")
                               .attr("d", res)
                               .call(this.fillatt.func);
         if ((this.options.BoxStyle !== 11) && this.fillatt.empty())
            elem.call(this.lineatt.func);
      }

      if ((btn1.length > 0) && this.fillatt.hasColor())
         this.draw_g.append("svg:path")
                    .attr("d", btn1)
                    .call(this.fillatt.func)
                    .style("fill", d3_rgb(this.fillatt.color).brighter(0.5).formatHex());

      if (btn2.length > 0)
         this.draw_g.append("svg:path")
                    .attr("d", btn2)
                    .call(this.fillatt.func)
                    .style("fill", !this.fillatt.hasColor() ? 'red' : d3_rgb(this.fillatt.color).darker(0.5).formatHex());

      if (cross.length > 0) {
         let elem = this.draw_g.append("svg:path")
                               .attr("d", cross)
                               .style("fill", "none");
         if (!this.lineatt.empty())
            elem.call(this.lineatt.func);
         else
            elem.style('stroke','black');
      }

      return handle;
   }

   /** @summary Draw histogram bins as candle plot */
   drawBinsCandle() {

      const kNoOption           = 0,
            kBox                = 1,
            kMedianLine         = 10,
            kMedianNotched      = 20,
            kMedianCircle       = 30,
            kMeanLine           = 100,
            kMeanCircle         = 300,
            kWhiskerAll         = 1000,
            kWhisker15          = 2000,
            kAnchor             = 10000,
            kPointsOutliers     = 100000,
            kPointsAll          = 200000,
            kPointsAllScat      = 300000,
            kHistoLeft          = 1000000,
            kHistoRight         = 2000000,
            kHistoViolin        = 3000000,
            kHistoZeroIndicator = 10000000,
            kHorizontal         = 100000000,
            fallbackCandle      = kBox + kMedianLine + kMeanCircle + kWhiskerAll + kAnchor,
            fallbackViolin      = kMeanCircle + kWhiskerAll + kHistoViolin + kHistoZeroIndicator;

      let fOption = kNoOption;

      const isOption = opt => {
         let mult = 1;
         while (opt >= mult) mult *= 10;
         mult /= 10;
         return Math.floor(fOption/mult) % 10 === Math.floor(opt/mult);

      }, parseOption = (opt, is_candle) => {

         let direction = '', preset = '',
             res = kNoOption, c0 = opt[0], c1 = opt[1];

         if (c0 >= 'A' && c0 <= 'Z') direction = c0;
         if (c0 >= '1' && c0 <= '9') preset = c0;
         if (c1 >= 'A' && c1 <= 'Z' && preset) direction = c1;
         if (c1 >= '1' && c1 <= '9' && direction) preset = c1;

         if (is_candle)
            switch(preset) {
               case '1': res += fallbackCandle; break;
               case '2': res += kBox + kMeanLine + kMedianLine + kWhisker15 + kAnchor + kPointsOutliers; break;
               case '3': res += kBox + kMeanCircle + kMedianLine + kWhisker15 + kAnchor + kPointsOutliers; break;
               case '4': res += kBox + kMeanCircle + kMedianNotched + kWhisker15 + kAnchor + kPointsOutliers; break;
               case '5': res += kBox + kMeanLine + kMedianLine + kWhisker15 + kAnchor + kPointsAll; break;
               case '6': res += kBox + kMeanCircle + kMedianLine + kWhisker15 + kAnchor + kPointsAllScat; break;
               default: res += fallbackCandle;
            }
         else
            switch(preset) {
               case '1': res += fallbackViolin; break;
               case '2': res += kMeanCircle + kWhisker15 + kHistoViolin + kHistoZeroIndicator + kPointsOutliers; break;
               default: res += fallbackViolin;
            }

         let l = opt.indexOf("("), r = opt.lastIndexOf(")");
         if ((l >= 0) && (r > l+1))
            res = parseInt(opt.slice(l+1, r));

         fOption = res;

         if ((direction == 'Y' || direction == 'H') && !isOption(kHorizontal))
            fOption += kHorizontal;

      }, extractQuantiles = (xx,proj,prob) => {

         let integral = 0, cnt = 0, sum1 = 0,
             res = { max: 0, first: -1, last: -1, entries: 0 };

         for (let j = 0; j < proj.length; ++j) {
            if (proj[j] > 0) {
               res.max = Math.max(res.max, proj[j]);
               if (res.first < 0) res.first = j;
               res.last = j;
            }
            integral += proj[j];
            sum1 += proj[j]*(xx[j]+xx[j+1])/2;
         }
         if (integral <= 0) return null;

         res.entries = integral;
         res.mean = sum1/integral;
         res.quantiles = new Array(prob.length);
         res.indx = new Array(prob.length);

         let sum = 0, nextv = 0;
         for (let j = 0; j < proj.length; ++j) {
            let v = nextv, x = xx[j];

            // special case - flat integral with const value
            if ((v === prob[cnt]) && (proj[j] === 0) && (v < 0.99)) {
               while ((proj[j] === 0) && (j < proj.length)) j++;
               x = (xx[j] + x) / 2; // this will be mid value
            }

            sum += proj[j];
            nextv = sum / integral;
            while ((prob[cnt] >= v) && (prob[cnt] < nextv)) {
               res.indx[cnt] = j;
               res.quantiles[cnt] = x + ((prob[cnt] - v)/(nextv-v))*(xx[j+1]-x);
               if (cnt++ == prob.length) return res;
               x = xx[j];
            }
         }

         while (cnt < prob.length) {
            res.indx[cnt] = proj.length-1;
            res.quantiles[cnt++] = xx[xx.length-1];
         }

         return res;
      };

      if (this.options.Candle)
         parseOption(this.options.Candle, true);
      else if (this.options.Violin)
         parseOption(this.options.Violin, false);

      let histo = this.getHisto(),
          handle = this.prepareDraw(),
          pmain = this.getFramePainter(), // used for axis values conversions
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          bars = "", lines = "", dashed_lines = "",
          hists = "", hlines = "",
          markers = "", cmarkers = "", attrcmarkers = null,
          xx, proj, swapXY = isOption(kHorizontal),
          scaledViolin = true, scaledCandle = false,
          maxContent = 0, maxIntegral = 0;

      if (this.options.Scaled !== null)
         scaledViolin = scaledCandle = this.options.Scaled;
      else if (histo.fTitle.indexOf('unscaled') >= 0)
         scaledViolin = scaledCandle = false;
      else if (histo.fTitle.indexOf('scaled') >= 0)
         scaledViolin = scaledCandle = true;

      if (scaledViolin && (isOption(kHistoRight) || isOption(kHistoLeft) || isOption(kHistoViolin)))
         for (let i = 0; i < this.nbinsx; ++i)
            for (let j = 0; j < this.nbinsy; ++j)
               maxContent = Math.max(maxContent, histo.getBinContent(i + 1, j + 1));

      const make_path = (...a) => {
         let l = a.length, i = 2, xx = a[0], yy = a[1],
             res = swapXY ? `M${yy},${xx}` : `M${xx},${yy}`;
         while (i < l) {
            switch(a[i]) {
               case 'Z': return res + "z";
               case 'V': if (yy != a[i+1]) { res += (swapXY ? 'h' : 'v') + (a[i+1] - yy); yy = a[i+1]; } break;
               case 'H': if (xx != a[i+1]) { res += (swapXY ? 'v' : 'h') + (a[i+1] - xx); xx = a[i+1]; } break;
               default: res += swapXY ? `l${a[i+1]-yy},${a[i]-xx}` : `l${a[i]-xx},${a[i+1]-yy}`; xx = a[i]; yy = a[i+1];
            }
            i += 2;
         }
         return res;
      }, make_marker = (x,y) => {
         if (!markers) {
            this.createAttMarker({ attr: histo, style: isOption(kPointsAllScat) ? 0 : 5 });
            this.markeratt.resetPos();
         }
         markers += swapXY ? this.markeratt.create(y,x) : this.markeratt.create(x,y);
      }, make_cmarker = (x,y) => {
         if (!attrcmarkers) {
            attrcmarkers = new TAttMarkerHandler({attr: histo, style: 24});
            attrcmarkers.resetPos();
         }
         cmarkers += swapXY ? attrcmarkers.create(y,x) : attrcmarkers.create(x,y);
      };

      //if ((histo.fFillStyle == 0) && (histo.fFillColor > 0) && (!this.fillatt || this.fillatt.empty()))
      //     this.createAttFill({ color: this.getColor(histo.fFillColor), pattern: 1001 });

      if (histo.fMarkerColor === 1) histo.fMarkerColor = histo.fLineColor;

      handle.candle = []; // array of drawn points

      // Determining the quantiles
      const fWhiskerRange = 1.0, fBoxRange = 0.5, // for now constants, later can be made configurable
            prob = [ (fWhiskerRange >= 1) ? 1e-15 : 0.5 - fWhiskerRange/2.,
                    (fBoxRange >= 1) ? 1E-14 : 0.5 - fBoxRange/2.,
                    0.5,
                    (fBoxRange >= 1) ? 1-1E-14 : 0.5 + fBoxRange/2.,
                    (fWhiskerRange >= 1) ? 1-1e-15 : 0.5 + fWhiskerRange/2.];

      const produceCandlePoint = (bin_indx, grx_left, grx_right, xindx1, xindx2) => {
         let res = extractQuantiles(xx, proj, prob);
         if (!res) return;

         let pnt = { bin: bin_indx, swapXY: swapXY, fBoxDown: res.quantiles[1], fMedian: res.quantiles[2], fBoxUp: res.quantiles[3] },
             fWhiskerDown = res.quantiles[0], fWhiskerUp = res.quantiles[4], iqr = pnt.fBoxUp - pnt.fBoxDown;

         if (isOption(kWhisker15)) { // Improved whisker definition, with 1.5*iqr
            let pos = pnt.fBoxDown-1.5*iqr, indx = res.indx[1];
            while ((xx[indx] > pos) && (indx > 0)) indx--;
            while (!proj[indx]) indx++;
            fWhiskerDown = xx[indx]; // use lower edge here
            pos = pnt.fBoxUp+1.5*iqr; indx = res.indx[3];
            while ((xx[indx] < pos) && (indx < proj.length)) indx++;
            while (!proj[indx]) indx--;
            fWhiskerUp = xx[indx+1]; // use upper index edge here
         }

         let fMean = res.mean,
             fMedianErr = 1.57*iqr/Math.sqrt(res.entries);

         //estimate quantiles... simple function... not so nice as GetQuantiles

         // exclude points with negative y when log scale is specified
         if (fWhiskerDown <= 0)
           if ((swapXY && funcs.logx) || (!swapXY && funcs.logy)) return;

         let w = (grx_right - grx_left), candleWidth, histoWidth,
             center = (grx_left + grx_right) / 2 + histo.fBarOffset/1000*w;
         if ((histo.fBarWidth > 0) && (histo.fBarWidth !== 1000)) {
            candleWidth = histoWidth = w * histo.fBarWidth / 1000;
         } else {
            candleWidth = w*0.66;
            histoWidth = w*0.8;
         }

         if (scaledViolin && (maxContent > 0))
            histoWidth *= res.max/maxContent;
         if (scaledCandle && (maxIntegral > 0))
            candleWidth *= res.entries/maxIntegral;

         pnt.x1 = Math.round(center - candleWidth/2);
         pnt.x2 = Math.round(center + candleWidth/2);
         center = Math.round(center);

         let x1d = Math.round(center - candleWidth/3),
             x2d = Math.round(center + candleWidth/3),
             fname = swapXY ? "grx" : "gry";

         pnt.yy1 = Math.round(funcs[fname](fWhiskerUp));
         pnt.y1 = Math.round(funcs[fname](pnt.fBoxUp));
         pnt.y0 = Math.round(funcs[fname](pnt.fMedian));
         pnt.y2 = Math.round(funcs[fname](pnt.fBoxDown));
         pnt.yy2 = Math.round(funcs[fname](fWhiskerDown));

         let y0m = Math.round(funcs[fname](fMean)),
             y01 = Math.round(funcs[fname](pnt.fMedian + fMedianErr)),
             y02 = Math.round(funcs[fname](pnt.fMedian - fMedianErr));

         if (isOption(kHistoZeroIndicator))
            hlines += make_path(center, Math.round(funcs[fname](xx[xindx1])),'V',Math.round(funcs[fname](xx[xindx2])));

         if (isOption(kMedianLine))
            lines += make_path(pnt.x1,pnt.y0,'H',pnt.x2);
         else if (isOption(kMedianNotched))
            lines += make_path(x1d,pnt.y0,'H',x2d);
         else if (isOption(kMedianCircle))
            make_cmarker(center, pnt.y0);

         if (isOption(kMeanCircle))
            make_cmarker(center, y0m);
         else if (isOption(kMeanLine))
            dashed_lines += make_path(pnt.x1,y0m,'H',pnt.x2);

         if (isOption(kBox))
            if (isOption(kMedianNotched))
               bars += make_path(pnt.x1, pnt.y1, "V", y01, x1d, pnt.y0, pnt.x1, y02, "V", pnt.y2, "H", pnt.x2, "V", y02, x2d, pnt.y0, pnt.x2, y01, "V", pnt.y1, "Z");
            else
               bars += make_path(pnt.x1, pnt.y1, "V", pnt.y2, "H", pnt.x2, "V", pnt.y1, "Z");

        if (isOption(kAnchor))  // Draw the anchor line
            lines += make_path(pnt.x1, pnt.yy1, "H", pnt.x2) + make_path(pnt.x1, pnt.yy2, "H", pnt.x2);

         if (isOption(kWhiskerAll) && !isOption(kHistoZeroIndicator)) { // Whiskers are dashed
            dashed_lines += make_path(center, pnt.y1, "V", pnt.yy1) + make_path(center, pnt.y2, "V", pnt.yy2);
         } else if ((isOption(kWhiskerAll) && isOption(kHistoZeroIndicator)) || isOption(kWhisker15)) {
            lines += make_path(center, pnt.y1, "V", pnt.yy1) + make_path(center, pnt.y2, "V", pnt.yy2);
         }

         if (isOption(kPointsOutliers) || isOption(kPointsAll) || isOption(kPointsAllScat)) {

            // reset seed for each projection to have always same pixels
            let rnd = new TRandom(bin_indx*7521 + Math.round(res.integral)),
                show_all = !isOption(kPointsOutliers),
                show_scat = isOption(kPointsAllScat);
            for (let ii = 0; ii < proj.length; ++ii) {
               let bin_content = proj[ii], binx = (xx[ii] + xx[ii+1])/2,
                   marker_x = center, marker_y = 0;

               if (!bin_content) continue;
               if (!show_all && (binx >= fWhiskerDown) && (binx <= fWhiskerUp)) continue;

               for (let k = 0; k < bin_content; k++) {
                  if (show_scat)
                     marker_x = center + Math.round(((rnd.random() - 0.5) * candleWidth));

                  if ((bin_content == 1) && !show_scat)
                     marker_y = Math.round(funcs[fname](binx));
                  else
                     marker_y = Math.round(funcs[fname](xx[ii] + rnd.random()*(xx[ii+1]-xx[ii])));

                  make_marker(marker_x, marker_y);
               }
            }
         }

         if ((isOption(kHistoRight) || isOption(kHistoLeft) || isOption(kHistoViolin)) && (res.max > 0) && (res.first >= 0)) {
            let arr = [], scale = (swapXY ? -0.5 : 0.5) *histoWidth/res.max;

            xindx1 = Math.max(xindx1, res.first);
            xindx2 = Math.min(xindx2-1, res.last);

            if (isOption(kHistoRight) || isOption(kHistoViolin)) {
               let prev_x = center, prev_y = Math.round(funcs[fname](xx[xindx1]));
               arr.push(prev_x, prev_y);
               for (let ii = xindx1; ii <= xindx2; ii++) {
                  let curr_x = Math.round(center + scale*proj[ii]),
                      curr_y = Math.round(funcs[fname](xx[ii+1]));
                  if (curr_x != prev_x) {
                     if (ii != xindx1) arr.push("V", prev_y);
                     arr.push("H", curr_x);
                  }
                  prev_x = curr_x;
                  prev_y = curr_y;
               }
               arr.push("V", prev_y);
            }

            if (isOption(kHistoLeft) || isOption(kHistoViolin)) {
               let prev_x = center, prev_y = Math.round(funcs[fname](xx[xindx2+1]));
               if (arr.length == 0)
                  arr.push(prev_x, prev_y);
               for (let ii = xindx2; ii >= xindx1; ii--) {
                  let curr_x = Math.round(center - scale*proj[ii]),
                      curr_y = Math.round(funcs[fname](xx[ii]));
                  if (curr_x != prev_x) {
                     if (ii != xindx2) arr.push("V", prev_y);
                     arr.push("H", curr_x);
                  }
                  prev_x = curr_x;
                  prev_y = curr_y;
               }
               arr.push("V", prev_y);
            }

            arr.push("H", center); // complete histogram

            hists += make_path(...arr);

            if (!this.fillatt.empty()) hists += "Z";
         }

         handle.candle.push(pnt); // keep point for the tooltip

      };

      if (swapXY) {
         xx = new Array(this.nbinsx+1);
         proj = new Array(this.nbinsx);
         for (let i = 0; i < this.nbinsx+1; ++i)
            xx[i] = histo.fXaxis.GetBinLowEdge(i+1);

         if(scaledCandle)
            for (let j = 0; j < this.nbinsy; ++j) {
               let sum = 0;
               for (let i = 0; i < this.nbinsx; ++i)
                  sum += histo.getBinContent(i+1,j+1);
               maxIntegral = Math.max(maxIntegral, sum);
            }

         for (let j = handle.j1; j < handle.j2; ++j) {
            for (let i = 0; i < this.nbinsx; ++i)
               proj[i] = histo.getBinContent(i+1,j+1);

            produceCandlePoint(j, handle.gry[j+1], handle.gry[j], handle.i1, handle.i2);
         }

      } else {
         xx = new Array(this.nbinsy+1);
         proj = new Array(this.nbinsy);

         for (let j = 0; j < this.nbinsy+1; ++j)
            xx[j] = histo.fYaxis.GetBinLowEdge(j+1);

         if(scaledCandle)
            for (let i = 0; i < this.nbinsx; ++i) {
               let sum = 0;
               for (let j = 0; j < this.nbinsy; ++j)
                  sum += histo.getBinContent(i+1,j+1);
               maxIntegral = Math.max(maxIntegral, sum);
            }

         // loop over visible x-bins
         for (let i = handle.i1; i < handle.i2; ++i) {
            for (let j = 0; j < this.nbinsy; ++j)
               proj[j] = histo.getBinContent(i+1,j+1);

            produceCandlePoint(i, handle.grx[i], handle.grx[i+1], handle.j1, handle.j2);
         }
      }

      if ((hlines.length > 0) && (histo.fFillColor > 0))
         this.draw_g.append("svg:path")
             .attr("d", hlines)
             .style("stroke", this.getColor(histo.fFillColor));

      let hline_color = (isOption(kHistoZeroIndicator) && (histo.fFillStyle != 0)) ? this.fillatt.color : this.lineatt.color;
      if (hists && (!this.fillatt.empty() || (hline_color != 'none')))
         this.draw_g.append("svg:path")
             .attr("d", hists)
             .style("stroke", (hline_color != 'none') ? hline_color : null)
             .style("pointer-events", isBatchMode() ? null : "visibleFill")
             .call(this.fillatt.func);

      if (bars)
         this.draw_g.append("svg:path")
             .attr("d", bars)
             .call(this.lineatt.func)
             .call(this.fillatt.func);

      if (lines)
         this.draw_g.append("svg:path")
             .attr("d", lines)
             .call(this.lineatt.func)
             .style('fill','none');

      if (dashed_lines) {
         let dashed = new TAttLineHandler({ attr: histo, style: 2 });
         this.draw_g.append("svg:path")
             .attr("d", dashed_lines)
             .call(dashed.func)
             .style('fill','none');
      }

      if (cmarkers)
         this.draw_g.append("svg:path")
             .attr("d", cmarkers)
             .call(attrcmarkers.func);

      if (markers)
         this.draw_g.append("svg:path")
             .attr("d", markers)
             .call(this.markeratt.func);

      return handle;
   }

   /** @summary Draw TH2 bins as scatter plot */
   drawBinsScatter() {
      let histo = this.getObject(),
          handle = this.prepareDraw({ rounding: true, pixel_density: true }),
          colPaths = [], currx = [], curry = [], cell_w = [], cell_h = [],
          colindx, cmd1, cmd2, i, j, binz, cw, ch, factor = 1.,
          scale = this.options.ScatCoef * ((this.gmaxbin) > 2000 ? 2000. / this.gmaxbin : 1.),
          rnd = new TRandom(handle.sumz);

      if (scale*handle.sumz < 1e5) {
         // one can use direct drawing of scatter plot without any patterns

         this.createAttMarker({ attr: histo });

         this.markeratt.resetPos();

         let path = "";
         for (i = handle.i1; i < handle.i2; ++i) {
            cw = handle.grx[i+1] - handle.grx[i];
            for (j = handle.j1; j < handle.j2; ++j) {
               ch = handle.gry[j] - handle.gry[j+1];
               binz = histo.getBinContent(i + 1, j + 1);

               let npix = Math.round(scale*binz);
               if (npix <= 0) continue;

               for (let k = 0; k < npix; ++k)
                  path += this.markeratt.create(
                            Math.round(handle.grx[i] + cw * rnd.random()),
                            Math.round(handle.gry[j+1] + ch * rnd.random()));
            }
         }

         this.draw_g
              .append("svg:path")
              .attr("d", path)
              .call(this.markeratt.func);

         return handle;
      }

      // limit filling factor, do not try to produce as many points as filled area;
      if (this.maxbin > 0.7) factor = 0.7/this.maxbin;

      let nlevels = Math.round(handle.max - handle.min),
          cntr = this.createContour((nlevels > 50) ? 50 : nlevels, this.minposbin, this.maxbin, this.minposbin);

      // now start build
      for (i = handle.i1; i < handle.i2; ++i) {
         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            if ((binz <= 0) || (binz < this.minbin)) continue;

            cw = handle.grx[i+1] - handle.grx[i];
            ch = handle.gry[j] - handle.gry[j+1];
            if (cw*ch <= 0) continue;

            colindx = cntr.getContourIndex(binz/cw/ch);
            if (colindx < 0) continue;

            cmd1 = `M${handle.grx[i]},${handle.gry[j+1]}`;
            if (colPaths[colindx] === undefined) {
               colPaths[colindx] = cmd1;
               cell_w[colindx] = cw;
               cell_h[colindx] = ch;
            } else{
               cmd2 = `m${handle.grx[i]-currx[colindx]},${handle.gry[j+1] - curry[colindx]}`;
               colPaths[colindx] += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
               cell_w[colindx] = Math.max(cell_w[colindx], cw);
               cell_h[colindx] = Math.max(cell_h[colindx], ch);
            }

            currx[colindx] = handle.grx[i];
            curry[colindx] = handle.gry[j+1];

            colPaths[colindx] += `v${ch}h${cw}v${-ch}z`;
         }
      }

      let layer = this.getFrameSvg().select('.main_layer'),
          defs = layer.select("defs");
      if (defs.empty() && (colPaths.length > 0))
         defs = layer.insert("svg:defs",":first-child");

      this.createAttMarker({ attr: histo });

      for (colindx = 0; colindx < colPaths.length; ++colindx)
        if ((colPaths[colindx] !== undefined) && (colindx < cntr.arr.length)) {
           let pattern_class = "scatter_" + colindx,
               pattern = defs.select('.' + pattern_class);
           if (pattern.empty())
              pattern = defs.append('svg:pattern')
                            .attr("class", pattern_class)
                            .attr("id", "jsroot_scatter_pattern_" + internals.id_counter++)
                            .attr("patternUnits","userSpaceOnUse");
           else
              pattern.selectAll("*").remove();

           let npix = Math.round(factor*cntr.arr[colindx]*cell_w[colindx]*cell_h[colindx]);
           if (npix < 1) npix = 1;

           let arrx = new Float32Array(npix), arry = new Float32Array(npix);

           if (npix === 1) {
              arrx[0] = arry[0] = 0.5;
           } else {
              for (let n = 0; n < npix; ++n) {
                 arrx[n] = rnd.random();
                 arry[n] = rnd.random();
              }
           }

           // arrx.sort();

           this.markeratt.resetPos();

           let path = "";

           for (let n = 0; n < npix; ++n)
              path += this.markeratt.create(arrx[n] * cell_w[colindx], arry[n] * cell_h[colindx]);

           pattern.attr("width", cell_w[colindx])
                  .attr("height", cell_h[colindx])
                  .append("svg:path")
                  .attr("d",path)
                  .call(this.markeratt.func);

           this.draw_g
               .append("svg:path")
               .attr("scatter-index", colindx)
               .style("fill", 'url(#' + pattern.attr("id") + ')')
               .attr("d", colPaths[colindx]);
        }

      return handle;
   }

   /** @summary Draw TH2 bins in 2D mode */
   draw2DBins() {

      if (this._hide_frame && this.isMainPainter()) {
         this.getFrameSvg().style('display', null);
         delete this._hide_frame;
      }

      if (!this.draw_content)
         return this.removeG();

      this.createHistDrawAttributes();

      this.createG(true);

      let handle, pr;

      if (this.isTH2Poly()) {
         pr = this.drawPolyBinsColor();
      } else {
         if (this.options.Scat)
            handle = this.drawBinsScatter();

         if (this.options.Color)
            handle = this.drawBinsColor();
         else if (this.options.Box)
            handle = this.drawBinsBox();
         else if (this.options.Arrow)
            handle = this.drawBinsArrow();
         else if (this.options.Contour)
            handle = this.drawBinsContour();
         else if (this.options.Candle || this.options.Violin)
            handle = this.drawBinsCandle();

         if (this.options.Text)
            pr = this.drawBinsText(handle);

         if (!handle && !pr)
            handle = this.drawBinsScatter();
      }

      if (handle)
         this.tt_handle = handle;
      else if (pr)
         return pr.then(tt => { this.tt_handle = tt; });
   }

   /** @summary Draw TH2 in circular mode */
   drawBinsCircular() {

      this.getFrameSvg().style('display', 'none');
      this._hide_frame = true;

      let rect = this.getPadPainter().getFrameRect(),
          hist = this.getHisto(),
          palette = this.options.Circular > 10 ? this.getHistPalette() : null,
          text_size = 20,
          circle_size = 16,
          axis = hist.fXaxis,
          getBinLabel = indx => {
            if (axis.fLabels)
               for (let i = 0; i < axis.fLabels.arr.length; ++i) {
                  const tstr = axis.fLabels.arr[i];
                  if (tstr.fUniqueID === indx+1) return tstr.fString;
               }
            return indx.toString();
          };

      this.createG();

      this.draw_g.attr('transform', `translate(${Math.round(rect.x + rect.width/2)},${Math.round(rect.y + rect.height/2)})`);

      let nbins = Math.min(this.nbinsx, this.nbinsy);

      this.startTextDrawing(42, text_size, this.draw_g);

      let pnts = [];

      for (let n = 0; n < nbins; n++) {
         let a = (0.5 - n/nbins)*Math.PI*2,
             cx = Math.round((0.9*rect.width/2 - 2*circle_size) * Math.cos(a)),
             cy = Math.round((0.9*rect.height/2 - 2*circle_size) * Math.sin(a)),
             x = Math.round(0.9*rect.width/2 * Math.cos(a)),
             y = Math.round(0.9*rect.height/2 * Math.sin(a)),
             rotate = Math.round(a/Math.PI*180), align = 12,
             color = palette ? palette.calcColor(n, nbins) : 'black';

         pnts.push({ x: cx, y: cy, a, color }); // remember points coordinates

         if ((rotate < -90) || (rotate > 90)) { rotate += 180; align = 32; }

         let s2 = Math.round(text_size/2), s1 = 2*s2;

         this.draw_g.append("path")
                    .attr("d",`M${cx-s2},${cy} a${s2},${s2},0,1,0,${s1},0a${s2},${s2},0,1,0,${-s1},0z`)
                    .style('stroke', color)
                    .style('fill','none');

         this.drawText({ align, rotate, x, y, text: getBinLabel(n)});
      }

      let max_width = circle_size/2, max_value = 0, min_value = 0;
      if (this.options.Circular > 11) {
         for (let i = 0; i < nbins - 1; ++i)
           for (let j = i+1; j < nbins; ++j) {
              let cont = hist.getBinContent(i+1, j+1);
              if (cont > 0) {
                 max_value = Math.max(max_value, cont);
                 if (!min_value || (cont < min_value)) min_value = cont;
              }
           }
      }

      for (let i = 0; i < nbins-1; ++i) {
         let path = "", pi = pnts[i];

         for (let j = i+1; j < nbins; ++j) {
            let cont = hist.getBinContent(i+1, j+1);
            if (cont <= 0) continue;

            let pj = pnts[j],
                a = (pi.a + pj.a)/2,
                qr = 0.5*(1-Math.abs(pi.a - pj.a)/Math.PI), // how far Q point will be away from center
                qx = Math.round(qr*rect.width/2 * Math.cos(a)),
                qy = Math.round(qr*rect.height/2 * Math.sin(a));

            path += `M${pi.x},${pi.y}Q${qx},${qy},${pj.x},${pj.y}`;

            if ((this.options.Circular > 11) && (max_value > min_value)) {
               let width = Math.round((cont - min_value) / (max_value - min_value) * (max_width - 1) + 1);
               this.draw_g.append("path").attr("d", path).style("stroke", pi.color).style("stroke-width", width).style('fill','none');
               path = "";
            }
         }
         if (path)
            this.draw_g.append("path").attr("d", path).style("stroke", pi.color).style('fill','none');
      }

      return this.finishTextDrawing();
   }

   /** @summary Draw histogram bins as chord diagram */
   drawBinsChord() {

      this.getFrameSvg().style('display', 'none');
      this._hide_frame = true;

      let fullsum = 0, used = [], isint = true,
          nbins = Math.min(this.nbinsx, this.nbinsy),
          hist = this.getHisto();
      for (let i = 0; i < nbins; ++i) {
         let sum = 0;
         for (let j = 0; j < nbins; ++j) {
            let cont = hist.getBinContent(i+1, j+1);
            if (cont > 0) {
               sum += cont;
               if (isint && (Math.round(cont) !== cont)) isint = false;
            }
         }
         if (sum > 0) used.push(i);
         fullsum += sum;
      }

      // do not show less than 2 elements
      if (used.length < 2) return Promise.resolve(true);

      let rect = this.getPadPainter().getFrameRect(),
          palette = this.getHistPalette(),
          outerRadius = Math.min(rect.width, rect.height) * 0.5 - 60,
          innerRadius = outerRadius - 10,
          data = [], labels = [],
          getColor = indx => palette.calcColor(indx, used.length),
          ndig = 0, tickStep = 1,
          formatValue = v => v.toString(),
          formatTicks = v => ndig > 3 ? v.toExponential(0) : v.toFixed(ndig),
          d3_descending = (a,b) => { return b < a ? -1 : b > a ? 1 : b >= a ? 0 : NaN; };

      if (!isint && fullsum < 10) {
         let lstep = Math.round(Math.log10(fullsum) - 2.3);
         ndig = -lstep;
         tickStep = Math.pow(10, lstep);
      } else if (fullsum > 200) {
         let lstep = Math.round(Math.log10(fullsum) - 2.3);
         tickStep = Math.pow(10, lstep);
      }

      if (tickStep * 250 < fullsum)
         tickStep *= 5;
      else if (tickStep * 100 < fullsum)
         tickStep *= 2;

      for (let i = 0; i < used.length; ++i) {
         data[i] = [];
         for (let j = 0; j < used.length; ++j)
            data[i].push(hist.getBinContent(used[i]+1, used[j]+1));
         let axis = hist.fXaxis, lbl = "indx_" + used[i].toString();
         if (axis.fLabels)
            for (let k = 0; k < axis.fLabels.arr.length; ++k) {
               const tstr = axis.fLabels.arr[k];
               if (tstr.fUniqueID === used[i]+1) { lbl = tstr.fString; break; }
            }
         labels.push(lbl);
      }

      this.createG();

      this.draw_g.attr('transform', `translate(${Math.round(rect.x + rect.width/2)},${Math.round(rect.y + rect.height/2)})`);

      const chord = d3_chord()
         .padAngle(10 / innerRadius)
         .sortSubgroups(d3_descending)
         .sortChords(d3_descending);

      const chords = chord(data);

      const group = this.draw_g.append("g")
         .attr("font-size", 10)
         .attr("font-family", "sans-serif")
         .selectAll("g")
         .data(chords.groups)
         .join("g");

      const arc = d3_arc().innerRadius(innerRadius).outerRadius(outerRadius);

      const ribbon = d3_ribbon().radius(innerRadius - 1).padAngle(1 / innerRadius);

      function ticks({ startAngle, endAngle, value }) {
         const k = (endAngle - startAngle) / value;
         let arr = [];
         for (let z = 0; z <= value; z += tickStep)
            arr.push({ value: z, angle: z * k + startAngle });
         return arr;
      }

      group.append("path")
         .attr("fill", d => getColor(d.index))
         .attr("d", arc);

      group.append("title").text(d => `${labels[d.index]} ${formatValue(d.value)}`);

      const groupTick = group.append("g")
         .selectAll("g")
         .data(ticks)
         .join("g")
         .attr("transform", d => `rotate(${d.angle * 180 / Math.PI - 90}) translate(${outerRadius},0)`);
      groupTick.append("line")
         .attr("stroke", "currentColor")
         .attr("x2", 6);

      groupTick.append("text")
         .attr("x", 8)
         .attr("dy", "0.35em")
         .attr("transform", d => d.angle > Math.PI ? "rotate(180) translate(-16)" : null)
         .attr("text-anchor", d => d.angle > Math.PI ? "end" : null)
         .text(d => formatTicks(d.value));

      group.select("text")
         .attr("font-weight", "bold")
         .text(function(d) {
            return this.getAttribute("text-anchor") === "end"
               ? `↑ ${labels[d.index]}` : `${labels[d.index]} ↓`;
         });

      this.draw_g.append("g")
         .attr("fill-opacity", 0.8)
         .selectAll("path")
         .data(chords)
         .join("path")
         .style("mix-blend-mode", "multiply")
         .attr("fill", d => getColor(d.source.index))
         .attr("d", ribbon)
         .append("title")
         .text(d => `${formatValue(d.source.value)} ${labels[d.target.index]} → ${labels[d.source.index]}${d.source.index === d.target.index ? "" : `\n${formatValue(d.target.value)} ${labels[d.source.index]} → ${labels[d.target.index]}`}`);

      return Promise.resolve(true);

   }

   /** @summary Provide text information (tooltips) for histogram bin */
   getBinTooltips(i, j) {
      let lines = [],
          histo = this.getHisto(),
          binz = histo.getBinContent(i+1,j+1);

      lines.push(this.getObjectHint());

      lines.push("x = " + this.getAxisBinTip("x", histo.fXaxis, i));
      lines.push("y = " + this.getAxisBinTip("y", histo.fYaxis, j));

      lines.push(`bin = ${i}, ${j}`);

      if (histo.$baseh) binz -= histo.$baseh.getBinContent(i+1,j+1);

      lines.push("entries = " + ((binz === Math.round(binz)) ? binz : floatToString(binz, gStyle.fStatFormat)));

      if ((this.options.TextKind == "E") || this.matchObjectType('TProfile2D')) {
         let errz = histo.getBinError(histo.getBin(i+1,j+1));
         lines.push("error = " + ((errz === Math.round(errz)) ? errz.toString() : floatToString(errz, gStyle.fPaintTextFormat)));
      }

      return lines;
   }

   /** @summary Provide text information (tooltips) for candle bin */
   getCandleTooltips(p) {
      let lines = [], pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          histo = this.getHisto();

      lines.push(this.getObjectHint());

      if (p.swapXY)
         lines.push("y = " + funcs.axisAsText("y", histo.fYaxis.GetBinLowEdge(p.bin+1)));
      else
         lines.push("x = " + funcs.axisAsText("x", histo.fXaxis.GetBinLowEdge(p.bin+1)));

      lines.push('m-25%  = ' + floatToString(p.fBoxDown, gStyle.fStatFormat))
      lines.push('median = ' + floatToString(p.fMedian, gStyle.fStatFormat))
      lines.push('m+25%  = ' + floatToString(p.fBoxUp, gStyle.fStatFormat))

      return lines;
   }

   /** @summary Provide text information (tooltips) for poly bin */
   getPolyBinTooltips(binindx, realx, realy) {

      let histo = this.getHisto(),
          bin = histo.fBins.arr[binindx],
          pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          binname = bin.fPoly.fName,
          lines = [], numpoints = 0;

      if (binname === "Graph") binname = "";
      if (binname.length === 0) binname = bin.fNumber;

      if ((realx===undefined) && (realy===undefined)) {
         realx = realy = 0;
         let gr = bin.fPoly, numgraphs = 1;
         if (gr._typename === 'TMultiGraph') { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

         for (let ngr = 0; ngr < numgraphs; ++ngr) {
            if (!gr || (ngr > 0)) gr = bin.fPoly.fGraphs.arr[ngr];

            for (let n = 0; n < gr.fNpoints; ++n) {
               ++numpoints;
               realx += gr.fX[n];
               realy += gr.fY[n];
            }
         }

         if (numpoints > 1) {
            realx = realx / numpoints;
            realy = realy / numpoints;
         }
      }

      lines.push(this.getObjectHint());
      lines.push("x = " + funcs.axisAsText("x", realx));
      lines.push("y = " + funcs.axisAsText("y", realy));
      if (numpoints > 0) lines.push("npnts = " + numpoints);
      lines.push("bin = " + binname);
      if (bin.fContent === Math.round(bin.fContent))
         lines.push("content = " + bin.fContent);
      else
         lines.push("content = " + floatToString(bin.fContent, gStyle.fStatFormat));
      return lines;
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      if (!pnt || !this.draw_content || !this.draw_g || !this.tt_handle || this.options.Proj) {
         if (this.draw_g)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      let histo = this.getHisto(),
          h = this.tt_handle,
          ttrect = this.draw_g.select(".tooltip_bin");

      if (h.poly) {
         // process tooltips from TH2Poly

         let pmain = this.getFramePainter(),
             funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
             foundindx = -1, bin;
         const realx = funcs.revertAxis("x", pnt.x),
               realy = funcs.revertAxis("y", pnt.y);

         if ((realx !== undefined) && (realy !== undefined)) {
            const len = histo.fBins.arr.length;

            for (let i = 0; (i < len) && (foundindx < 0); ++i) {
               bin = histo.fBins.arr[i];

               // found potential bins candidate
               if ((realx < bin.fXmin) || (realx > bin.fXmax) ||
                   (realy < bin.fYmin) || (realy > bin.fYmax)) continue;

               // ignore empty bins with col0 option
               if ((bin.fContent === 0) && !this.options.Zero) continue;

               let gr = bin.fPoly, numgraphs = 1;
               if (gr._typename === 'TMultiGraph') { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

               for (let ngr = 0; ngr < numgraphs; ++ngr) {
                  if (!gr || (ngr > 0)) gr = bin.fPoly.fGraphs.arr[ngr];
                  if (gr.IsInside(realx,realy)) {
                     foundindx = i;
                     break;
                  }
               }
            }
         }

         if (foundindx < 0) {
            ttrect.remove();
            return null;
         }

         let res = { name: histo.fName, title: histo.fTitle,
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.getFillColorAlt('blue') : "blue",
                     exact: true, menu: true,
                     lines: this.getPolyBinTooltips(foundindx, realx, realy) };

         if (pnt.disabled) {
            ttrect.remove();
            res.changed = true;
         } else {

            if (ttrect.empty())
               ttrect = this.draw_g.append("svg:path")
                            .attr("class","tooltip_bin h1bin")
                            .style("pointer-events","none");

            res.changed = ttrect.property("current_bin") !== foundindx;

            if (res.changed)
                  ttrect.attr("d", this.createPolyBin(funcs, bin))
                        .style("opacity", "0.7")
                        .property("current_bin", foundindx);
         }

         if (res.changed)
            res.user_info = { obj: histo, name: histo.fName,
                              bin: foundindx,
                              cont: bin.fContent,
                              grx: pnt.x, gry: pnt.y };

         return res;

      } else if (h.candle) {
         // process tooltips for candle

         let i, p, match;

         for (i = 0; i < h.candle.length; ++i) {
            p = h.candle[i];
            match = p.swapXY ? ((p.x1 <= pnt.y) && (pnt.y <= p.x2) && (p.yy1 >= pnt.x) && (pnt.x >= p.yy2))
                             : ((p.x1 <= pnt.x) && (pnt.x <= p.x2) && (p.yy1 <= pnt.y) && (pnt.y <= p.yy2));
            if (match) break;
         }

         if (!match) {
            ttrect.remove();
            return null;
         }

         let res = { name: histo.fName, title: histo.fTitle,
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.getFillColorAlt('blue') : "blue",
                     lines: this.getCandleTooltips(p), exact: true, menu: true };

         if (pnt.disabled) {
            ttrect.remove();
            res.changed = true;
         } else {

            if (ttrect.empty())
               ttrect = this.draw_g.append("svg:path")
                                   .attr("class","tooltip_bin h1bin")
                                   .style("pointer-events","none")
                                   .style("opacity", "0.7");

            res.changed = ttrect.property("current_bin") !== i;

            if (res.changed)
               ttrect.attr("d", p.swapXY ? `M${p.yy1},${p.x1}H${p.yy2}V${p.x2}H${p.yy1}Z` : `M${p.x1},${p.yy1}H${p.x2}V${p.yy2}H${p.x1}Z`)
                     .property("current_bin", i);
         }

         if (res.changed)
            res.user_info = { obj: histo,  name: histo.fName,
                              bin: i+1, cont: p.fMedian, binx: i+1, biny: 1,
                              grx: pnt.x, gry: pnt.y };

         return res;
      }

      let i, j, binz = 0, colindx = null,
          i1, i2, j1, j2, x1, x2, y1, y2,
          pmain = this.getFramePainter();

      // search bins position
      if (pmain.reverse_x) {
         for (i = h.i1; i < h.i2; ++i)
            if ((pnt.x<=h.grx[i]) && (pnt.x>=h.grx[i+1])) break;
      } else {
         for (i = h.i1; i < h.i2; ++i)
            if ((pnt.x>=h.grx[i]) && (pnt.x<=h.grx[i+1])) break;
      }

      if (pmain.reverse_y) {
         for (j = h.j1; j < h.j2; ++j)
            if ((pnt.y <= h.gry[j+1]) && (pnt.y >= h.gry[j])) break;
      } else {
         for (j = h.j1; j < h.j2; ++j)
            if ((pnt.y >= h.gry[j+1]) && (pnt.y <= h.gry[j])) break;
      }

      if ((i < h.i2) && (j < h.j2)) {

         i1 = i; i2 = i+1; j1 = j; j2 = j+1;
         x1 = h.grx[i1]; x2 = h.grx[i2];
         y1 = h.gry[j2]; y2 = h.gry[j1];

         let match = true;

         if (this.options.Color) {
            // take into account bar settings
            let dx = x2 - x1, dy = y2 - y1;
            x2 = Math.round(x1 + dx*h.xbar2);
            x1 = Math.round(x1 + dx*h.xbar1);
            y2 = Math.round(y1 + dy*h.ybar2);
            y1 = Math.round(y1 + dy*h.ybar1);
            if (pmain.reverse_x) {
               if ((pnt.x>x1) || (pnt.x<=x2)) match = false;
            } else {
               if ((pnt.x<x1) || (pnt.x>=x2)) match = false;
            }
            if (pmain.reverse_y) {
               if ((pnt.y>y1) || (pnt.y<=y2)) match = false;
            } else {
               if ((pnt.y<y1) || (pnt.y>=y2)) match = false;
            }
         }

         binz = histo.getBinContent(i+1,j+1);
         if (this.is_projection) {
            colindx = 0; // just to avoid hide
         } else if (!match) {
            colindx = null;
         } else if (h.hide_only_zeros) {
            colindx = (binz === 0) && !this._show_empty_bins ? null : 0;
         } else {
            colindx = this.getContour().getPaletteIndex(this.getHistPalette(), binz);
            if ((colindx === null) && (binz === 0) && this._show_empty_bins) colindx = 0;
         }
      }

      if (colindx === null) {
         ttrect.remove();
         return null;
      }

      let res = { name: histo.fName, title: histo.fTitle,
                  x: pnt.x, y: pnt.y,
                  color1: this.lineatt ? this.lineatt.color : 'green',
                  color2: this.fillatt ? this.fillatt.getFillColorAlt('blue') : "blue",
                  lines: this.getBinTooltips(i, j), exact: true, menu: true };

      if (this.options.Color) res.color2 = this.getHistPalette().getColor(colindx);

      if (pnt.disabled && !this.is_projection) {
         ttrect.remove();
         res.changed = true;
      } else {
         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:path")
                                .attr("class","tooltip_bin h1bin")
                                .style("pointer-events","none");

         let binid = i*10000 + j;

         if (this.is_projection == "X") {
            x1 = 0; x2 = this.getFramePainter().getFrameWidth();
            if (this.projection_width > 1) {
               let dd = (this.projection_width-1)/2;
               if (j2+dd >= h.j2) { j2 = Math.min(Math.round(j2+dd), h.j2); j1 = Math.max(j2 - this.projection_width, h.j1); }
                             else { j1 = Math.max(Math.round(j1-dd), h.j1); j2 = Math.min(j1 + this.projection_width, h.j2); }
            }
            y1 = h.gry[j2]; y2 = h.gry[j1];
            binid = j1*777 + j2*333;
         } else if (this.is_projection == "Y") {
            y1 = 0; y2 = this.getFramePainter().getFrameHeight();
            if (this.projection_width > 1) {
               let dd = (this.projection_width-1)/2;
               if (i2+dd >= h.i2) { i2 = Math.min(Math.round(i2+dd), h.i2); i1 = Math.max(i2 - this.projection_width, h.i1); }
                             else { i1 = Math.max(Math.round(i1-dd), h.i1); i2 = Math.min(i1 + this.projection_width, h.i2); }
            }
            x1 = h.grx[i1], x2 = h.grx[i2],
            binid = i1*777 + i2*333;
         }

         res.changed = ttrect.property("current_bin") !== binid;

         if (res.changed)
            ttrect.attr("d", "M"+x1+","+y1 + "h"+(x2-x1) + "v"+(y2-y1) + "h"+(x1-x2) + "z")
                  .style("opacity", "0.7")
                  .property("current_bin", binid);

         if (this.is_projection && res.changed)
            this.redrawProjection(i1, i2, j1, j2);
      }

      if (res.changed)
         res.user_info = { obj: histo, name: histo.fName,
                           bin: histo.getBin(i+1, j+1), cont: binz, binx: i+1, biny: j+1,
                           grx: pnt.x, gry: pnt.y };

      return res;
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis, min, max) {

      if (axis=="z") return true;

      let obj = this.getHisto();
      if (obj) obj = (axis=="y") ? obj.fYaxis : obj.fXaxis;

      return !obj || (obj.FindBin(max,0.5) - obj.FindBin(min,0) > 1);
   }

   /** @summary Complete paletted drawing */
   completePalette(pp) {
      if (!pp) return true;

      pp.$main_painter = this;
      this.options.Zvert = pp._palette_vertical;

      // redraw palette till the end when contours are available
      return pp.drawPave(this.options.Cjust ? "cjust" : "");
   }

   /** @summary Performs 2D drawing of histogram
     * @returns {Promise} when ready */
   draw2D(/* reason */) {

      this.clear3DScene();

      let need_palette = this.options.Zscale && (this.options.Color || this.options.Contour);
      // draw new palette, resize frame if required

      return this.drawColorPalette(need_palette, true).then(pp => {

         let pr;
         if (this.options.Circular && this.isMainPainter()) {
            pr = this.drawBinsCircular();
         } else if (this.options.Chord && this.isMainPainter()) {
            pr = this.drawBinsChord();
         } else {
            pr = this.drawAxes().then(() => this.draw2DBins())
         }

         return pr.then(() => this.completePalette(pp));
      }).then(() => this.drawHistTitle()).then(() => {
         this.updateStatWebCanvas();
         return this.addInteractivity();
      });
   }

   /** @summary Should performs 3D drawing of histogram
     * @desc Disabled in 2D case. just draw default draw options
     * @returns {Promise} when ready */
   draw3D(reason) {
      console.log('3D drawing is disabled, load ./hist/TH2Painter.mjs');
      return this.draw2D(reason);
   }

   /** @summary Call drawing function depending from 3D mode */
   callDrawFunc(reason) {

      let main = this.getMainPainter(),
          fp = this.getFramePainter();

     if ((main !== this) && fp && (fp.mode3d !== this.options.Mode3D))
        this.copyOptionsFrom(main);

      return this.options.Mode3D ? this.draw3D(reason) : this.draw2D(reason);
   }

   /** @summary Redraw histogram */
   redraw(reason) {
      return this.callDrawFunc(reason);
   }

   /** @summary draw TH2 object */
   static draw(dom, histo, opt) {
      return TH2Painter._drawHist(new TH2Painter(dom, histo), opt);
   }

} // class TH2Painter

export { TH2Painter };
