import { gStyle, internals, createTPolyLine } from '../core.mjs';
import { rgb as d3_rgb } from '../d3.mjs';
import { TAttLineHandler } from '../base/TAttLineHandler.mjs';
import { floatToString, TRandom } from '../base/BasePainter.mjs';
import { RHistPainter } from './RHistPainter.mjs';
import { ensureRCanvas } from '../gpad/RCanvasPainter.mjs';

/**
 * @summary Painter for RH2 classes
 *
 * @private
 */

class RH2Painter extends RHistPainter {

   /** @summary constructor
     * @param {object|string} dom - DOM element or id
     * @param {object} histo - histogram object */
   constructor(dom, histo) {
      super(dom, histo);
      this.wheel_zoomy = true;
   }

   /** @summary Cleanup painter */
   cleanup() {
      delete this.tt_handle;
      super.cleanup();
   }

   /** @summary Returns histogram dimension */
   getDimension() { return 2; }

   /** @summary Toggle projection */
   toggleProjection(kind, width) {

      if (kind=="Projections") kind = "";

      if ((typeof kind == 'string') && (kind.length>1)) {
          width = parseInt(kind.slice(1));
          kind = kind[0];
      }

      if (!width) width = 1;

      if (kind && (this.is_projection==kind)) {
         if (this.projection_width === width) {
            kind = "";
         } else {
            this.projection_width = width;
            return;
         }
      }

      delete this.proj_hist;

      let new_proj = (this.is_projection === kind) ? "" : kind;
      this.is_projection = ""; // disable projection redraw until callback
      this.projection_width = width;

      this.provideSpecialDrawArea(new_proj).then(() => { this.is_projection = new_proj; return this.redrawProjection(); });
   }

   /** @summary Readraw projections */
   redrawProjection(/* ii1, ii2 , jj1, jj2*/) {
      // do nothing for the moment

      if (!this.is_projection) return;
   }

   /** @summary Execute menu command */
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
      menu.add("sub:Projections", () => this.toggleProjection());
      let kind = this.is_projection || "";
      if (kind) kind += this.projection_width;
      let kinds = ["X1", "X2", "X3", "X5", "X10", "Y1", "Y2", "Y3", "Y5", "Y10"];
      for (let k=0;k<kinds.length;++k)
         menu.addchk(kind==kinds[k], kinds[k], kinds[k], arg => this.toggleProjection(arg));
      menu.add("endsub:");

      menu.add("Auto zoom-in", () => this.autoZoom());

      let opts = this.getSupportedDrawOptions();

      menu.addDrawMenu("Draw with", opts, arg => {
         if (arg==='inspect')
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

      switch(funcname) {
         case "ToggleColor": this.toggleColor(); break;
         case "Toggle3D": this.toggleMode3D(); break;
         default: return false;
      }

      // all methods here should not be processed further
      return true;
   }

   /** @summary Fill pad toolbar with RH2-related functions */
   fillToolbar() {
      super.fillToolbar(true);

      let pp = this.getPadPainter();
      if (!pp) return;

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

      this.redraw();
   }

   /** @summary Perform automatic zoom inside non-zero region of histogram */
   autoZoom() {
      let i1 = this.getSelectIndex("x", "left", -1),
          i2 = this.getSelectIndex("x", "right", 1),
          j1 = this.getSelectIndex("y", "left", -1),
          j2 = this.getSelectIndex("y", "right", 1),
          i,j, histo = this.getHisto(), xaxis = this.getAxis("x"), yaxis = this.getAxis("y");

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
         xmin = xaxis.GetBinCoord(ileft);
         xmax = xaxis.GetBinCoord(iright);
         isany = true;
      }

      if ((jleft > j1 || jright < j2) && (jleft < jright - 1)) {
         ymin = yaxis.GetBinCoord(jleft);
         ymax = yaxis.GetBinCoord(jright);
         isany = true;
      }

      if (isany)
         return this.getFramePainter().zoom(xmin, xmax, ymin, ymax);
   }

   /** @summary Scan content of 2-dim histogram */
   scanContent(when_axis_changed) {

      // no need to rescan histogram while result does not depend from axis selection
      if (when_axis_changed && this.nbinsx && this.nbinsy) return;

      let i, j, histo = this.getHisto();

      this.extractAxesProperties(2);

      if (this.isDisplayItem()) {
         // take min/max values from the display item
         this.gminbin = histo.fContMin;
         this.gminposbin = histo.fContMinPos > 0 ? histo.fContMinPos : null;
         this.gmaxbin = histo.fContMax;
      } else {
         // global min/max, used at the moment in 3D drawing
         this.gminbin = this.gmaxbin = histo.getBinContent(1, 1);
         this.gminposbin = null;
         for (i = 0; i < this.nbinsx; ++i) {
            for (j = 0; j < this.nbinsy; ++j) {
               let bin_content = histo.getBinContent(i+1, j+1);
               if (bin_content < this.gminbin) this.gminbin = bin_content; else
                  if (bin_content > this.gmaxbin) this.gmaxbin = bin_content;
               if (bin_content > 0)
                  if ((this.gminposbin===null) || (this.gminposbin > bin_content)) this.gminposbin = bin_content;
            }
         }
      }

      this.zmin = this.gminbin;
      this.zmax = this.gmaxbin;

      // this value used for logz scale drawing
      if (this.gminposbin === null) this.gminposbin = this.gmaxbin*1e-4;

      if (this.options.Axis > 0) { // Paint histogram axis only
         this.draw_content = false;
      } else {
         this.draw_content = this.gmaxbin > 0;
      }
   }

   /** @summary Count statistic */
   countStat(cond) {
      let histo = this.getHisto(),
          stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0,
          stat_sumx2 = 0, stat_sumy2 = 0,
          xside, yside, xx, yy, zz,
          res = { name: "histo", entries: 0, integral: 0, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, matrix: [0,0,0,0,0,0,0,0,0], xmax: 0, ymax:0, wmax: null };

      let xleft = this.getSelectIndex("x", "left"),
          xright = this.getSelectIndex("x", "right"),
          yleft = this.getSelectIndex("y", "left"),
          yright = this.getSelectIndex("y", "right"),
          xi, yi, xaxis = this.getAxis("x"), yaxis = this.getAxis("y");

      // TODO: account underflow/overflow bins, now stored in different array and only by histogram itself
      for (xi = 1; xi <= this.nbinsx; ++xi) {
         xside = (xi <= xleft+1) ? 0 : (xi > xright+1 ? 2 : 1);
         xx = xaxis.GetBinCoord(xi - 0.5);

         for (yi = 1; yi <= this.nbinsy; ++yi) {
            yside = (yi <= yleft+1) ? 0 : (yi > yright+1 ? 2 : 1);
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
         }
      }

      if (stat_sum0 > 0) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.rmsx = Math.sqrt(Math.abs(stat_sumx2 / stat_sum0 - res.meanx**2));
         res.rmsy = Math.sqrt(Math.abs(stat_sumy2 / stat_sum0 - res.meany**2));
      }

      if (res.wmax === null) res.wmax = 0;
      res.integral = stat_sum0;
      return res;
   }

   /** @summary Fill statistic into statbox */
   fillStatistic(stat, dostat /*, dofit*/) {

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

      stat.clearStat();

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
         stat.addText("Integral = " + stat.format(data.matrix[4], "entries"));

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

      return true;
   }

   /** @summary Draw histogram bins as color */
   drawBinsColor() {
      const histo = this.getHisto(),
            handle = this.prepareDraw(),
            di = handle.stepi, dj = handle.stepj,
            entries = [],
            can_merge = true;
      let colindx, cmd1, cmd2, i, j, binz, dx, dy, entry, last_entry;

      const flush_last_entry = () => {
         last_entry.path += `h${dx}v${last_entry.y2-last_entry.y}h${-dx}z`;
         last_entry.dy = 0;
         last_entry = null;
      };

      // now start build
      for (i = handle.i1; i < handle.i2; i += di) {
         dx = (handle.grx[i+di] - handle.grx[i]) || 1;

         for (j = handle.j1; j < handle.j2; j += dj) {
            binz = histo.getBinContent(i+1, j+1);
            colindx = handle.palette.getContourIndex(binz);
            if (binz === 0) {
               if (!this.options.Zero)
                  colindx = null;
               else if ((colindx === null) && this._show_empty_bins)
                  colindx = 0;
            }
            if (colindx === null) {
               if (last_entry) flush_last_entry();
               continue;
            }

            cmd1 = `M${handle.grx[i]},${handle.gry[j]}`;

            dy = (handle.gry[j+dj] - handle.gry[j]) || -1;

            entry = entries[colindx];

            if (entry === undefined) {
               entry = entries[colindx] = { path: cmd1 };
            } else if (can_merge && (entry === last_entry)) {
               entry.y2 = handle.gry[j] + dy;
               continue;
            } else {
               cmd2 = `m${handle.grx[i]-entry.x},${handle.gry[j]-entry.y}`;
               entry.path += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
            }
            if (last_entry) flush_last_entry();
            entry.x = handle.grx[i];
            entry.y = handle.gry[j];
            if (can_merge) {
               entry.y2 = handle.gry[j] + dy;
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
               .style("fill", handle.palette.getColor(colindx))
               .attr("d", entry.path);
      });

      this.updatePaletteDraw();

      return handle;
   }

   /** @summary Build histogram contour lines */
   buildContour(handle, levels, palette, contour_func) {
      let histo = this.getHisto(),
          kMAXCONTOUR = 2004,
          kMAXCOUNT = 2000,
          // arguments used in the PaintContourLine
          xarr = new Float32Array(2*kMAXCONTOUR),
          yarr = new Float32Array(2*kMAXCONTOUR),
          itarr = new Int32Array(2*kMAXCONTOUR),
          lj = 0, ipoly, poly, polys = [], np, npmax = 0,
          x = [0.,0.,0.,0.], y = [0.,0.,0.,0.], zc = [0.,0.,0.,0.], ir = [0,0,0,0],
          i, j, k, n, m, ix, ljfill, count,
          xsave, ysave, itars, jx,
          di = handle.stepi, dj = handle.stepj;

      function BinarySearch(zc) {
         for (let kk=0;kk<levels.length;++kk)
            if (zc<levels[kk]) return kk-1;
         return levels.length-1;
      }

      function PaintContourLine(elev1, icont1, x1, y1,  elev2, icont2, x2, y2) {
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
            ii +=2;
            n++;
         }
         return icount;
      }

      let arrx = handle.original ? handle.origx : handle.grx,
          arry = handle.original ? handle.origy : handle.gry;

      for (j = handle.j1; j < handle.j2-dj; j += dj) {

         y[1] = y[0] = (arry[j] + arry[j+dj])/2;
         y[3] = y[2] = (arry[j+dj] + arry[j+2*dj])/2;

         for (i = handle.i1; i < handle.i2-di; i += di) {

            zc[0] = histo.getBinContent(i+1, j+1);
            zc[1] = histo.getBinContent(i+1+di, j+1);
            zc[2] = histo.getBinContent(i+1+di, j+1+dj);
            zc[3] = histo.getBinContent(i+1, j+1+dj);

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
      for (ipoly=0;ipoly<levels.length;ipoly++) {
         if (levels[ipoly] >= 0) { first = ipoly; break; }
      }
      //store negative contours from 0 to minimum, then all positive contours
      k = 0;
      for (ipoly=first-1;ipoly>=0;ipoly--) {polysort[k] = ipoly; k++;}
      for (ipoly=first;ipoly<levels.length;ipoly++) { polysort[k] = ipoly; k++;}

      let xp = new Float32Array(2*npmax),
          yp = new Float32Array(2*npmax);

      for (k=0;k<levels.length;++k) {

         ipoly = polysort[k];
         poly = polys[ipoly];
         if (!poly) continue;

         let colindx = ipoly,
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
   drawBinsContour(funcs, frame_w,frame_h) {
      let handle = this.prepareDraw({ rounding: false, extra: 100, original: this.options.Proj != 0 }),
          main = this.getFramePainter(),
          palette = main.getHistPalette(),
          levels = palette.getContour(),
          func = main.getProjectionFunc();

      let BuildPath = (xp,yp,iminus,iplus,do_close) => {
         let cmd = "", last, pnt, first, isany;
         for (let i = iminus; i <= iplus; ++i) {
            if (func) {
               pnt = func(xp[i], yp[i]);
               pnt.x = Math.round(funcs.grx(pnt.x));
               pnt.y = Math.round(funcs.gry(pnt.y));
            } else {
               pnt = { x: Math.round(xp[i]), y: Math.round(yp[i]) };
            }
            if (!cmd) {
               cmd = "M" + pnt.x + "," + pnt.y; first = pnt;
            } else if ((i == iplus) && first && (pnt.x == first.x) && (pnt.y == first.y)) {
               if (!isany) return ""; // all same points
               cmd += "z"; do_close = false;
            } else if ((pnt.x != last.x) && (pnt.y != last.y)) {
               cmd +=  "l" + (pnt.x - last.x) + "," + (pnt.y - last.y); isany = true;
            } else if (pnt.x != last.x) {
               cmd +=  "h" + (pnt.x - last.x); isany = true;
            } else if (pnt.y != last.y) {
               cmd +=  "v" + (pnt.y - last.y); isany = true;
            }
            last = pnt;
         }
         if (do_close) cmd += "z";
         return cmd;
      };

      if (this.options.Contour===14) {
         let dd = "M0,0h"+frame_w+"v"+frame_h+"h-"+frame_w+"z";
         if (this.options.Proj) {
            let dj = handle.stepj, sz = parseInt((handle.j2 - handle.j1)/dj),
                xd = new Float32Array(sz*2), yd = new Float32Array(sz*2);
            for (let i=0;i<sz;++i) {
               xd[i] = handle.origx[handle.i1];
               yd[i] = (handle.origy[handle.j1]*(i*dj+0.5) + handle.origy[handle.j2]*(sz-0.5-i*dj))/sz;
               xd[i+sz] = handle.origx[handle.i2];
               yd[i+sz] = (handle.origy[handle.j2]*(i*dj+0.5) + handle.origy[handle.j1]*(sz-0.5-i*dj))/sz;
            }
            dd = BuildPath(xd,yd,0,2*sz-1,true);
         }

         this.draw_g
             .append("svg:path")
             .attr("d", dd)
             .style("fill", palette.getColor(0));
      }

      this.buildContour(handle, levels, palette,
         (colindx,xp,yp,iminus,iplus) => {
            let icol = palette.getColor(colindx),
                fillcolor = icol, lineatt;

            switch (this.options.Contour) {
               case 1: break;
               case 11: fillcolor = 'none'; lineatt = new TAttLineHandler({ color: icol }); break;
               case 12: fillcolor = 'none'; lineatt = new TAttLineHandler({ color:1, style: (colindx%5 + 1), width: 1 }); break;
               case 13: fillcolor = 'none'; lineatt = this.lineatt; break;
               case 14: break;
            }

            let dd = BuildPath(xp, yp, iminus, iplus, fillcolor != 'none');
            if (!dd) return;

            let elem = this.draw_g
                          .append("svg:path")
                          .attr("class","th2_contour")
                          .attr("d", dd)
                          .style("fill", fillcolor);

            if (lineatt)
               elem.call(lineatt.func);
         }
      );

      handle.hide_only_zeros = true; // text drawing suppress only zeros

      return handle;
   }

   /** @summary Create polybin */
   createPolyBin(pmain, bin, text_pos) {
      let cmd = "", ngr, ngraphs = 1, gr = null,
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y);

      if (bin.fPoly._typename=='TMultiGraph')
         ngraphs = bin.fPoly.fGraphs.arr.length;
      else
         gr = bin.fPoly;

      if (text_pos)
         bin._sumx = bin._sumy = bin._suml = 0;

      function addPoint(x1,y1,x2,y2) {
         let len = Math.sqrt((x1-x2)**2 + (y1-y2)**2);
         bin._sumx += (x1+x2)*len/2;
         bin._sumy += (y1+y2)*len/2;
         bin._suml += len;
      }

      for (ngr = 0; ngr < ngraphs; ++ ngr) {
         if (!gr || (ngr>0)) gr = bin.fPoly.fGraphs.arr[ngr];

         let npnts = gr.fNpoints, n,
             x = gr.fX, y = gr.fY,
             grx = Math.round(funcs.grx(x[0])),
             gry = Math.round(funcs.gry(y[0])),
             nextx, nexty;

         if ((npnts>2) && (x[0]==x[npnts-1]) && (y[0]==y[npnts-1])) npnts--;

         cmd += "M"+grx+","+gry;

         for (n=1;n<npnts;++n) {
            nextx = Math.round(funcs.grx(x[n]));
            nexty = Math.round(funcs.gry(y[n]));
            if (text_pos) addPoint(grx,gry, nextx, nexty);
            if ((grx!==nextx) || (gry!==nexty)) {
               if (grx===nextx)
                  cmd += "v" + (nexty - gry);
               else if (gry===nexty)
                  cmd += "h" + (nextx - grx);
               else
                  cmd += "l" + (nextx - grx) + "," + (nexty - gry);
            }
            grx = nextx; gry = nexty;
         }

         if (text_pos) addPoint(grx, gry, Math.round(funcs.grx(x[0])), Math.round(funcs.gry(y[0])));
         cmd += "z";
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

   /** @summary Draw RH2 bins as text */
   drawBinsText(handle) {
      let histo = this.getHisto(),
          i, j, binz, binw, binh, text, x, y, width, height;

      if (handle===null) handle = this.prepareDraw({ rounding: false });

      let textFont  = this.v7EvalFont("text", { size: 20, color: "black", align: 22 }),
          text_offset = 0,
          text_g = this.draw_g.append("svg:g").attr("class","th2_text"),
          di = handle.stepi, dj = handle.stepj,
          profile2d = (this.options.TextKind == "E") &&
                      this.matchObjectType('TProfile2D') && (typeof histo.getBinEntries=='function');

      if (this.options.BarOffset) text_offset = this.options.BarOffset;

      this.startTextDrawing(textFont, 'font', text_g);

      for (i = handle.i1; i < handle.i2; i += di)
         for (j = handle.j1; j < handle.j2; j += dj) {
            binz = histo.getBinContent(i+1, j+1);
            if ((binz === 0) && !this._show_empty_bins) continue;

            binw = handle.grx[i+di] - handle.grx[i];
            binh = handle.gry[j] - handle.gry[j+dj];

            if (profile2d)
               binz = histo.getBinEntries(i+1, j+1);

            text = (binz === Math.round(binz)) ? binz.toString() :
                      floatToString(binz, gStyle.fPaintTextFormat);

            if (textFont.angle) {
               x = Math.round(handle.grx[i] + binw*0.5);
               y = Math.round(handle.gry[j+dj] + binh*(0.5 + text_offset));
               width = height = 0;
            } else {
               x = Math.round(handle.grx[i] + binw*0.1);
               y = Math.round(handle.gry[j+dj] + binh*(0.1 + text_offset));
               width = Math.round(binw*0.8);
               height = Math.round(binh*0.8);
            }

            this.drawText({ align: 22, x, y, width, height, text, latex: 0, draw_g: text_g });
         }

      return this.finishTextDrawing(text_g, true).then(() => {

         handle.hide_only_zeros = true; // text drawing suppress only zeros

         return handle;
      });
   }

   /** @summary Draw RH2 bins as arrows */
   drawBinsArrow() {
      let histo = this.getHisto(), cmd = "",
          i,j, dn = 1e-30, dx, dy, xc,yc,
          dxn,dyn,x1,x2,y1,y2, anr,si,co,
          handle = this.prepareDraw({ rounding: false }),
          scale_x = (handle.grx[handle.i2] - handle.grx[handle.i1])/(handle.i2 - handle.i1 + 1-0.03)/2,
          scale_y = (handle.gry[handle.j2] - handle.gry[handle.j1])/(handle.j2 - handle.j1 + 1-0.03)/2,
          di = handle.stepi, dj = handle.stepj;

      const makeLine = (dx, dy) => {
         if (dx)
            return dy ? `l${dx},${dy}` : `h${dx}`;
         return dy ? `v${dy}` : "";
      };

      for (let loop = 0; loop < 2; ++loop)
         for (i = handle.i1; i < handle.i2; i += di)
            for (j = handle.j1; j < handle.j2; j += dj) {

               if (i === handle.i1) {
                  dx = histo.getBinContent(i+1+di, j+1) - histo.getBinContent(i+1, j+1);
               } else if (i >= handle.i2-di) {
                  dx = histo.getBinContent(i+1, j+1) - histo.getBinContent(i+1-di, j+1);
               } else {
                  dx = 0.5*(histo.getBinContent(i+1+di, j+1) - histo.getBinContent(i+1-di, j+1));
               }
               if (j === handle.j1) {
                  dy = histo.getBinContent(i+1, j+1+dj) - histo.getBinContent(i+1, j+1);
               } else if (j >= handle.j2-dj) {
                  dy = histo.getBinContent(i+1, j+1) - histo.getBinContent(i+1, j+1-dj);
               } else {
                  dy = 0.5*(histo.getBinContent(i+1, j+1+dj) - histo.getBinContent(i+1, j+1-dj));
               }

               if (loop===0) {
                  dn = Math.max(dn, Math.abs(dx), Math.abs(dy));
               } else {
                  xc = (handle.grx[i] + handle.grx[i+di])/2;
                  yc = (handle.gry[j] + handle.gry[j+dj])/2;
                  dxn = scale_x*dx/dn;
                  dyn = scale_y*dy/dn;
                  x1  = xc - dxn;
                  x2  = xc + dxn;
                  y1  = yc - dyn;
                  y2  = yc + dyn;
                  dx = Math.round(x2-x1);
                  dy = Math.round(y2-y1);

                  if ((dx!==0) || (dy!==0)) {
                     cmd += "M"+Math.round(x1)+","+Math.round(y1) + makeLine(dx,dy);;

                     if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                        anr = Math.sqrt(2/(dx**2 + dy**2));
                        si  = Math.round(anr*(dx + dy));
                        co  = Math.round(anr*(dx - dy));
                        if (si || co)
                           cmd += `m${-si},${co}` + makeLine(si,-co) + makeLine(-co,-si);;
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

   /** @summary Draw RH2 bins as boxes */
   drawBinsBox() {

      let histo = this.getHisto(),
          handle = this.prepareDraw({ rounding: false }),
          main = this.getFramePainter();

      if (main.maxbin === main.minbin) {
         main.maxbin = this.gmaxbin;
         main.minbin = this.gminbin;
         main.minposbin = this.gminposbin;
      }
      if (main.maxbin === main.minbin)
         main.minbin = Math.min(0, main.maxbin-1);

      let absmax = Math.max(Math.abs(main.maxbin), Math.abs(main.minbin)),
          absmin = Math.max(0, main.minbin),
          i, j, binz, absz, res = "", cross = "", btn1 = "", btn2 = "",
          zdiff, dgrx, dgry, xx, yy, ww, hh,
          xyfactor, uselogz = false, logmin = 0,
          di = handle.stepi, dj = handle.stepj;

      if (main.logz && (absmax>0)) {
         uselogz = true;
         let logmax = Math.log(absmax);
         if (absmin > 0)
            logmin = Math.log(absmin);
         else if ((main.minposbin >= 1) && (main.minposbin < 100))
            logmin = Math.log(0.7);
          else
            logmin = (main.minposbin > 0) ? Math.log(0.7*main.minposbin) : logmax - 10;
         if (logmin >= logmax) logmin = logmax - 10;
         xyfactor = 1. / (logmax - logmin);
      } else {
         xyfactor = 1. / (absmax - absmin);
      }

      // now start build
      for (i = handle.i1; i < handle.i2; i += di) {
         for (j = handle.j1; j < handle.j2; j += dj) {
            binz = histo.getBinContent(i + 1, j + 1);
            absz = Math.abs(binz);
            if ((absz === 0) || (absz < absmin)) continue;

            zdiff = uselogz ? ((absz > 0) ? Math.log(absz) - logmin : 0) : (absz - absmin);
            // area of the box should be proportional to absolute bin content
            zdiff = 0.5 * ((zdiff < 0) ? 1 : (1 - Math.sqrt(zdiff * xyfactor)));
            // avoid oversized bins
            if (zdiff < 0) zdiff = 0;

            ww = handle.grx[i+di] - handle.grx[i];
            hh = handle.gry[j] - handle.gry[j+dj];

            dgrx = zdiff * ww;
            dgry = zdiff * hh;

            xx = Math.round(handle.grx[i] + dgrx);
            yy = Math.round(handle.gry[j+dj] + dgry);

            ww = Math.max(Math.round(ww - 2*dgrx), 1);
            hh = Math.max(Math.round(hh - 2*dgry), 1);

            res += `M${xx},${yy}v${hh}h${ww}v${-hh}z`;

            if ((binz < 0) && (this.options.BoxStyle === 10))
               cross += `M${xx},${yy}l${ww},${hh}M${xx+ww},${yy}l${-ww},${hh}`;

            if ((this.options.BoxStyle === 11) && (ww>5) && (hh>5)) {
               let pww = Math.round(ww*0.1),
                   phh = Math.round(hh*0.1),
                   side1 = `M${xx},${yy}h${ww}l${-pww},${phh}h${2*pww-ww}v${hh-2*phh}l${-pww},${phh}z`,
                   side2 = `M${xx+ww},${yy+hh}v${-hh}l${-pww},${phh}v${hh-2*phh}h${2*pww-ww}l${-pww},${phh}z`;
               if (binz < 0) { btn2 += side1; btn1 += side2; }
                        else { btn1 += side1; btn2 += side2; }
            }
         }
      }

      if (res.length > 0) {
         let elem = this.draw_g
                        .append("svg:path")
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
      }

      return handle;
   }

   /** @summary Draw histogram bins as candle plot */
   drawBinsCandle(funcs, w) {
      let histo = this.getHisto(), yaxis = this.getAxis("y"),
          handle = this.prepareDraw(),
          i, j, y, sum1, cont, center, counter, integral, pnt,
          bars = "", markers = "", posy;

      // create attribute only when necessary
      this.createv7AttMarker();

      // reset absolution position for markers
      this.markeratt.resetPos();

      handle.candle = []; // array of drawn points

      // loop over visible x-bins
      for (i = handle.i1; i < handle.i2; ++i) {
         sum1 = 0;
         //estimate integral
         integral = 0;
         counter = 0;
         for (j = 0; j < this.nbinsy; ++j) {
            integral += histo.getBinContent(i+1,j+1);
         }
         pnt = { bin:i, meany:0, m25y:0, p25y:0, median:0, iqr:0, whiskerp:0, whiskerm:0};
         //estimate quantiles... simple function... not so nice as GetQuantiles
         for (j = 0; j < this.nbinsy; ++j) {
            cont = histo.getBinContent(i+1,j+1);
            posy = yaxis.GetBinCoord(j + 0.5);
            if (counter/integral < 0.001 && (counter + cont)/integral >=0.001) pnt.whiskerm = posy; // Lower whisker
            if (counter/integral < 0.25 && (counter + cont)/integral >=0.25) pnt.m25y = posy; // Lower edge of box
            if (counter/integral < 0.5 && (counter + cont)/integral >=0.5) pnt.median = posy; //Median
            if (counter/integral < 0.75 && (counter + cont)/integral >=0.75) pnt.p25y = posy; //Upper edge of box
            if (counter/integral < 0.999 && (counter + cont)/integral >=0.999) pnt.whiskerp = posy; // Upper whisker
            counter += cont;
            y = posy; // center of y bin coordinate
            sum1 += cont*y;
         }
         if (counter > 0) {
            pnt.meany = sum1/counter;
         }
         pnt.iqr = pnt.p25y-pnt.m25y;

         //Whiskers cannot exceed 1.5*iqr from box
         if ((pnt.m25y-1.5*pnt.iqr) > pnt.whsikerm)  {
            pnt.whiskerm = pnt.m25y-1.5*pnt.iqr;
         }
         if ((pnt.p25y+1.5*pnt.iqr) < pnt.whiskerp) {
            pnt.whiskerp = pnt.p25y+1.5*pnt.iqr;
         }

         // exclude points with negative y when log scale is specified
         if (funcs.logy && (pnt.whiskerm<=0)) continue;

         w = handle.grx[i+1] - handle.grx[i];
         w *= 0.66;
         center = (handle.grx[i+1] + handle.grx[i]) / 2 + this.options.BarOffset*w;
         if (this.options.BarWidth > 0) w = w * this.options.BarWidth;

         pnt.x1 = Math.round(center - w/2);
         pnt.x2 = Math.round(center + w/2);
         center = Math.round(center);

         pnt.y0 = Math.round(funcs.gry(pnt.median));
         // mean line
         bars += "M" + pnt.x1 + "," + pnt.y0 + "h" + (pnt.x2-pnt.x1);

         pnt.y1 = Math.round(funcs.gry(pnt.p25y));
         pnt.y2 = Math.round(funcs.gry(pnt.m25y));

         // rectangle
         bars += "M" + pnt.x1 + "," + pnt.y1 +
         "v" + (pnt.y2-pnt.y1) + "h" + (pnt.x2-pnt.x1) + "v-" + (pnt.y2-pnt.y1) + "z";

         pnt.yy1 = Math.round(funcs.gry(pnt.whiskerp));
         pnt.yy2 = Math.round(funcs.gry(pnt.whiskerm));

         // upper part
         bars += "M" + center + "," + pnt.y1 + "v" + (pnt.yy1-pnt.y1);
         bars += "M" + pnt.x1 + "," + pnt.yy1 + "h" + (pnt.x2-pnt.x1);

         // lower part
         bars += "M" + center + "," + pnt.y2 + "v" + (pnt.yy2-pnt.y2);
         bars += "M" + pnt.x1 + "," + pnt.yy2 + "h" + (pnt.x2-pnt.x1);

         //estimate outliers
         for (j = 0; j < this.nbinsy; ++j) {
            cont = histo.getBinContent(i+1,j+1);
            posy = yaxis.GetBinCoord(j + 0.5);
            if (cont > 0 && posy < pnt.whiskerm) markers += this.markeratt.create(center, posy);
            if (cont > 0 && posy > pnt.whiskerp) markers += this.markeratt.create(center, posy);         }

         handle.candle.push(pnt); // keep point for the tooltip
      }

      if (bars.length > 0)
         this.draw_g.append("svg:path")
             .attr("d", bars)
             .call(this.lineatt.func)
             .call(this.fillatt.func);

      if (markers.length > 0)
         this.draw_g.append("svg:path")
             .attr("d", markers)
             .call(this.markeratt.func);

      return handle;
   }

   /** @summary Draw RH2 bins as scatter plot */
   drawBinsScatter() {
      let histo = this.getHisto(),
          handle = this.prepareDraw({ rounding: true, pixel_density: true, scatter_plot: true }),
          colPaths = [], currx = [], curry = [], cell_w = [], cell_h = [],
          colindx, cmd1, cmd2, i, j, binz, cw, ch, factor = 1.,
          scale = this.options.ScatCoef * ((this.gmaxbin) > 2000 ? 2000. / this.gmaxbin : 1.),
          di = handle.stepi, dj = handle.stepj;

      let rnd = new TRandom(handle.sumz);

      if (scale*handle.sumz < 1e5) {
         // one can use direct drawing of scatter plot without any patterns

         this.createv7AttMarker();

         this.markeratt.resetPos();

         let path = "", k, npix;
         for (i = handle.i1; i < handle.i2; i += di) {
            cw = handle.grx[i+di] - handle.grx[i];
            for (j = handle.j1; j < handle.j2; j += dj) {
               ch = handle.gry[j] - handle.gry[j+dj];
               binz = histo.getBinContent(i + 1, j + 1);

               npix = Math.round(scale*binz);
               if (npix <= 0) continue;

               for (k = 0; k < npix; ++k)
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

      // let nlevels = Math.round(handle.max - handle.min);

      // now start build
      for (i = handle.i1; i < handle.i2; i += di) {
         for (j = handle.j1; j < handle.j2; j += dj) {
            binz = histo.getBinContent(i + 1, j + 1);
            if ((binz <= 0) || (binz < this.minbin)) continue;

            cw = handle.grx[i+di] - handle.grx[i];
            ch = handle.gry[j] - handle.gry[j+dj];
            if (cw*ch <= 0) continue;

            colindx = handle.palette.getContourIndex(binz/cw/ch);
            if (colindx < 0) continue;

            cmd1 = "M"+handle.grx[i]+","+handle.gry[j+dj];
            if (colPaths[colindx] === undefined) {
               colPaths[colindx] = cmd1;
               cell_w[colindx] = cw;
               cell_h[colindx] = ch;
            } else{
               cmd2 = "m" + (handle.grx[i]-currx[colindx]) + "," + (handle.gry[j+dj] - curry[colindx]);
               colPaths[colindx] += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
               cell_w[colindx] = Math.max(cell_w[colindx], cw);
               cell_h[colindx] = Math.max(cell_h[colindx], ch);
            }

            currx[colindx] = handle.grx[i];
            curry[colindx] = handle.gry[j+dj];

            colPaths[colindx] += "v"+ch+"h"+cw+"v-"+ch+"z";
         }
      }

      let layer = this.getFrameSvg().select('.main_layer'),
          defs = layer.select("defs");
      if (defs.empty() && (colPaths.length>0))
         defs = layer.insert("svg:defs",":first-child");

      this.createv7AttMarker();

      let cntr = handle.palette.getContour();

      for (colindx=0;colindx<colPaths.length;++colindx)
        if ((colPaths[colindx] !== undefined) && (colindx<cntr.length)) {
           let pattern_class = "scatter_" + colindx,
               pattern = defs.select('.' + pattern_class);
           if (pattern.empty())
              pattern = defs.append('svg:pattern')
                            .attr("class", pattern_class)
                            .attr("id", "jsroot_scatter_pattern_" + internals.id_counter++)
                            .attr("patternUnits","userSpaceOnUse");
           else
              pattern.selectAll("*").remove();

           let npix = Math.round(factor*cntr[colindx]*cell_w[colindx]*cell_h[colindx]);
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

   /** @summary Draw RH2 bins in 2D mode */
   draw2DBins() {

      if (!this.draw_content) {
         this.removeG();
         return Promise.resolve(false);
      }

      this.createHistDrawAttributes();

      this.createG(true);

      let pmain = this.getFramePainter(),
          rect = pmain.getFrameRect(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          handle = null, pr = null;

      // if (this.lineatt.empty()) this.lineatt.color = 'cyan';

      if (this.options.Scat)
         handle = this.drawBinsScatter();
      else if (this.options.Color)
         handle = this.drawBinsColor();
      else if (this.options.Box)
         handle = this.drawBinsBox();
      else if (this.options.Arrow)
         handle = this.drawBinsArrow();
      else if (this.options.Contour > 0)
         handle = this.drawBinsContour(funcs, rect.width, rect.height);
      else if (this.options.Candle)
         handle = this.drawBinsCandle(funcs, rect.width);

      if (this.options.Text)
         pr = this.drawBinsText(handle);

      if (!handle && !pr)
         handle = this.drawBinsColor();

      if (!pr) pr = Promise.resolve(handle);

      return pr.then(h => {
         this.tt_handle = h;
         return this;
      });
   }

   /** @summary Provide text information (tooltips) for histogram bin */
   getBinTooltips(i, j) {
      let lines = [],
           histo = this.getHisto(),
           binz = histo.getBinContent(i+1,j+1),
           di = 1, dj = 1;

      if (this.isDisplayItem()) {
         di = histo.stepx || 1;
         dj = histo.stepy || 1;
      }

      lines.push(this.getObjectHint() || "histo<2>");
      lines.push("x = " + this.getAxisBinTip("x", i, di));
      lines.push("y = " + this.getAxisBinTip("y", j, dj));

      lines.push("bin = " + i + ", " + j);

      if (histo.$baseh) binz -= histo.$baseh.getBinContent(i+1,j+1);

      let lbl = "entries = " + ((di>1) || (dj>1) ? "~" : "");

      if (binz === Math.round(binz))
         lines.push(lbl + binz);
      else
         lines.push(lbl + floatToString(binz, gStyle.fStatFormat));

      return lines;
   }

   /** @summary Provide text information (tooltips) for candle bin */
   getCandleTooltips(p) {
      let lines = [], main = this.getFramePainter(), xaxis = this.getAxis("y");

      lines.push(this.getObjectHint() || "histo");

      lines.push("x = " + main.axisAsText("x", xaxis.GetBinCoord(p.bin)));

      lines.push('mean y = ' + floatToString(p.meany, gStyle.fStatFormat))
      lines.push('m25 = ' + floatToString(p.m25y, gStyle.fStatFormat))
      lines.push('p25 = ' + floatToString(p.p25y, gStyle.fStatFormat))

      return lines;
   }

   /** @summary Provide text information (tooltips) for poly bin */
   getPolyBinTooltips(binindx, realx, realy) {

      let histo = this.getHisto(),
          bin = histo.fBins.arr[binindx],
          pmain = this.getFramePainter(),
          binname = bin.fPoly.fName,
          lines = [], numpoints = 0;

      if (binname === "Graph") binname = "";
      if (binname.length === 0) binname = bin.fNumber;

      if ((realx===undefined) && (realy===undefined)) {
         realx = realy = 0;
         let gr = bin.fPoly, numgraphs = 1;
         if (gr._typename === 'TMultiGraph') { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

         for (let ngr=0;ngr<numgraphs;++ngr) {
            if (!gr || (ngr>0)) gr = bin.fPoly.fGraphs.arr[ngr];

            for (let n=0;n<gr.fNpoints;++n) {
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

      lines.push(this.getObjectHint() || "histo");
      lines.push("x = " + pmain.axisAsText("x", realx));
      lines.push("y = " + pmain.axisAsText("y", realy));
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

         let pmain = this.getFramePainter(), foundindx = -1, bin;
         const realx = pmain.revertAxis("x", pnt.x),
               realy = pmain.revertAxis("y", pnt.y);

         if ((realx !== undefined) && (realy !== undefined)) {
            const len = histo.fBins.arr.length;

            for (let i = 0; (i < len) && (foundindx < 0); ++ i) {
               bin = histo.fBins.arr[i];

               // found potential bins candidate
               if ((realx < bin.fXmin) || (realx > bin.fXmax) ||
                    (realy < bin.fYmin) || (realy > bin.fYmax)) continue;

               // ignore empty bins with col0 option
               if ((bin.fContent === 0) && !this.options.Zero) continue;

               let gr = bin.fPoly, numgraphs = 1;
               if (gr._typename === 'TMultiGraph') { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

               for (let ngr=0;ngr<numgraphs;++ngr) {
                  if (!gr || (ngr>0)) gr = bin.fPoly.fGraphs.arr[ngr];
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

         let res = { name: "histo", title: histo.fTitle || "title",
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.getFillColorAlt('blue') : 'blue',
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
                  ttrect.attr("d", this.createPolyBin(pmain, bin))
                        .style("opacity", "0.7")
                        .property("current_bin", foundindx);
         }

         if (res.changed)
            res.user_info = { obj: histo,  name: "histo",
                              bin: foundindx,
                              cont: bin.fContent,
                              grx: pnt.x, gry: pnt.y };

         return res;

      } else

      if (h.candle) {
         // process tooltips for candle

         let p, i;

         for (i=0;i<h.candle.length;++i) {
            p = h.candle[i];
            if ((p.x1 <= pnt.x) && (pnt.x <= p.x2) && (p.yy1 <= pnt.y) && (pnt.y <= p.yy2)) break;
         }

         if (i>=h.candle.length) {
            ttrect.remove();
            return null;
         }

         let res = { name: "histo", title: histo.fTitle || "title",
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.getFillColorAlt('blue') : 'blue',
                     lines: this.getCandleTooltips(p), exact: true, menu: true };

         if (pnt.disabled) {
            ttrect.remove();
            res.changed = true;
         } else {

            if (ttrect.empty())
               ttrect = this.draw_g.append("svg:rect")
                                   .attr("class","tooltip_bin h1bin")
                                   .style("pointer-events","none");

            res.changed = ttrect.property("current_bin") !== i;

            if (res.changed)
               ttrect.attr("x", p.x1)
                     .attr("width", p.x2-p.x1)
                     .attr("y", p.yy1)
                     .attr("height", p.yy2- p.yy1)
                     .style("opacity", "0.7")
                     .property("current_bin", i);
         }

         if (res.changed)
            res.user_info = { obj: histo,  name: "histo",
                              bin: i+1, cont: p.median, binx: i+1, biny: 1,
                              grx: pnt.x, gry: pnt.y };

         return res;
      }

      let i, j, binz = 0, colindx = null;

      // search bins position
      for (i = h.i1; i < h.i2; ++i)
         if ((pnt.x>=h.grx[i]) && (pnt.x<=h.grx[i+1])) break;

      for (j = h.j1; j < h.j2; ++j)
         if ((pnt.y>=h.gry[j+1]) && (pnt.y<=h.gry[j])) break;

      if ((i < h.i2) && (j < h.j2)) {
         binz = histo.getBinContent(i+1,j+1);
         if (this.is_projection) {
            colindx = 0; // just to avoid hide
         } else if (h.hide_only_zeros) {
            colindx = (binz === 0) && !this._show_empty_bins ? null : 0;
         } else {
            colindx = h.palette.getContourIndex(binz);
            if ((colindx === null) && (binz === 0) && this._show_empty_bins) colindx = 0;
         }
      }

      if (colindx === null) {
         ttrect.remove();
         return null;
      }

      let res = { name: "histo", title: histo.fTitle || "title",
                  x: pnt.x, y: pnt.y,
                  color1: this.lineatt ? this.lineatt.color : 'green',
                  color2: this.fillatt ? this.fillatt.getFillColorAlt('blue') : 'blue',
                  lines: this.getBinTooltips(i, j), exact: true, menu: true };

      if (this.options.Color) res.color2 = h.palette.getColor(colindx);

      if (pnt.disabled && !this.is_projection) {
         ttrect.remove();
         res.changed = true;
      } else {
         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:rect")
                                .attr("class","tooltip_bin h1bin")
                                .style("pointer-events","none");

         let i1 = i, i2 = i+1,
             j1 = j, j2 = j+1,
             x1 = h.grx[i1], x2 = h.grx[i2],
             y1 = h.gry[j2], y2 = h.gry[j1],
             binid = i*10000 + j;

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
            ttrect.attr("x", x1)
                  .attr("width", x2 - x1)
                  .attr("y", y1)
                  .attr("height", y2 - y1)
                  .style("opacity", "0.7")
                  .property("current_bin", binid);

         if (this.is_projection && res.changed)
            this.redrawProjection(i1, i2, j1, j2);
      }

      if (res.changed)
         res.user_info = { obj: histo, name: "histo",
                           bin: histo.getBin(i+1, j+1), cont: binz, binx: i+1, biny: j+1,
                           grx: pnt.x, gry: pnt.y };

      return res;
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis,min,max) {
      if (axis=="z") return true;
      let obj = this.getAxis(axis);
      return (obj.FindBin(max,0.5) - obj.FindBin(min,0) > 1);
   }

   /** @summary Performs 2D drawing of histogram
     * @returns {Promise} when ready */
   draw2D(reason) {
      this.clear3DScene();

      return this.drawFrameAxes().then(res => {
        return res ? this.drawingBins(reason) : false;
      }).then(res => {
         if (res) return this.draw2DBins().then(() => this.addInteractivity());
      }).then(() => this);
   }

   /** @summary Performs 3D drawing of histogram
     * @returns {Promise} when ready */
   draw3D(reason) {
      console.log('3D drawing is disabled, load ./hist/RH1Painter.mjs');
      return this.draw2D(reason);
   }

   /** @summary Call drawing function depending from 3D mode */
   callDrawFunc(reason) {
      let main = this.getFramePainter();

      if (main && (main.mode3d !== this.options.Mode3D) && !this.isMainPainter())
         this.options.Mode3D = main.mode3d;

      return this.options.Mode3D ? this.draw3D(reason) : this.draw2D(reason);
   }

   /** @summary Redraw histogram */
   redraw(reason) {
      return this.callDrawFunc(reason);
   }

   static _draw(painter, opt) {
      return ensureRCanvas(painter).then(() => {

         painter.setAsMainPainter();

         painter.options = { Hist: false, Error: false, Zero: false, Mark: false,
                             Line: false, Fill: false, Lego: 0, Surf: 0,
                             Text: true, TextAngle: 0, TextKind: "",
                             BaseLine: false, Mode3D: false, AutoColor: 0,
                             Color: false, Scat: false, ScatCoef: 1, Candle: "", Box: false, BoxStyle: 0, Arrow: false, Contour: 0, Proj: 0,
                             BarOffset: 0., BarWidth: 1., minimum: -1111, maximum: -1111 };

         let kind = painter.v7EvalAttr("kind", ""),
             sub = painter.v7EvalAttr("sub", 0),
             o = painter.options;

         o.Text = painter.v7EvalAttr("drawtext", false);

         switch(kind) {
            case "lego": o.Lego = sub > 0 ? 10+sub : 12; o.Mode3D = true; break;
            case "surf": o.Surf = sub > 0 ? 10+sub : 1; o.Mode3D = true; break;
            case "box": o.Box = true; o.BoxStyle = 10 + sub; break;
            case "err": o.Error = true; o.Mode3D = true; break;
            case "cont": o.Contour = sub > 0 ? 10+sub : 1; break;
            case "arr": o.Arrow = true; break;
            case "scat": o.Scat = true; break;
            case "col": o.Color = true; break;
            default: if (!o.Text) o.Color = true;
         }

         // here we deciding how histogram will look like and how will be shown
         // painter.decodeOptions(opt);

         painter._show_empty_bins = false;

         painter.scanContent();

         return painter.callDrawFunc();
      });
   }

   /** @summary draw RH2 object */
   static draw(dom, obj, opt) {
      // create painter and add it to canvas
      return RH2Painter._draw(new RH2Painter(dom, obj), opt);
   }

} //  class RH2Painter


export { RH2Painter };
