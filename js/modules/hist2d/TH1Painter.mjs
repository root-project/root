import { gStyle, settings, isBatchMode } from '../core.mjs';

import { rgb as d3_rgb } from '../d3.mjs';

import { floatToString, buildSvgPath } from '../base/BasePainter.mjs';

import { THistPainter } from './THistPainter.mjs';

/**
 * @summary Painter for TH1 classes
 * @private
 */

class TH1Painter extends THistPainter {

   /** @summary Convert TH1K into normal binned histogram */
   convertTH1K() {
      let histo = this.getObject();
      if (histo.fReady) return;

      let arr = histo.fArray, entries = histo.fEntries; // array of values
      histo.fNcells = histo.fXaxis.fNbins + 2;
      histo.fArray = new Float64Array(histo.fNcells).fill(0);
      for (let n = 0; n < histo.fNIn; ++n)
         histo.Fill(arr[n]);
      histo.fReady = true;
      histo.fEntries = entries;
   }

   /** @summary Scan content of 1-D histogram
     * @desc Detect min/max values for x and y axis
     * @param {boolean} when_axis_changed - true when zooming was changed, some checks may be skipped */
   scanContent(when_axis_changed) {

      if (when_axis_changed && !this.nbinsx) when_axis_changed = false;

      if (this.isTH1K()) this.convertTH1K();

      let histo = this.getHisto();

      if (!when_axis_changed)
         this.extractAxesProperties(1);

      let left = this.getSelectIndex("x", "left"),
          right = this.getSelectIndex("x", "right");

      if (when_axis_changed) {
         if ((left === this.scan_xleft) && (right === this.scan_xright)) return;
      }

      // Paint histogram axis only
      this.draw_content = !(this.options.Axis > 0);

      this.scan_xleft = left;
      this.scan_xright = right;

      let hmin = 0, hmin_nz = 0, hmax = 0, hsum = 0, first = true,
          profile = this.isTProfile(), value, err;

      for (let i = 0; i < this.nbinsx; ++i) {
         value = histo.getBinContent(i + 1);
         hsum += profile ? histo.fBinEntries[i + 1] : value;

         if ((i<left) || (i>=right)) continue;

         if ((value > 0) && ((hmin_nz == 0) || (value < hmin_nz))) hmin_nz = value;

         if (first) {
            hmin = hmax = value;
            first = false;
         }

         err = this.options.Error ? histo.getBinError(i + 1) : 0;

         hmin = Math.min(hmin, value - err);
         hmax = Math.max(hmax, value + err);
      }

      // account overflow/underflow bins
      if (profile)
         hsum += histo.fBinEntries[0] + histo.fBinEntries[this.nbinsx + 1];
      else
         hsum += histo.getBinContent(0) + histo.getBinContent(this.nbinsx + 1);

      this.stat_entries = hsum;
      if (histo.fEntries > 1) this.stat_entries = histo.fEntries;

      this.hmin = hmin;
      this.hmax = hmax;

      this.ymin_nz = hmin_nz; // value can be used to show optimal log scale

      if ((this.nbinsx == 0) || ((Math.abs(hmin) < 1e-300) && (Math.abs(hmax) < 1e-300)))
         this.draw_content = false;

      let set_zoom = false;

      if (this.draw_content) {
         if (hmin >= hmax) {
            if (hmin == 0) { this.ymin = 0; this.ymax = 1; }
            else if (hmin < 0) { this.ymin = 2 * hmin; this.ymax = 0; }
            else { this.ymin = 0; this.ymax = hmin * 2; }
         } else {
            let dy = (hmax - hmin) * 0.05;
            this.ymin = hmin - dy;
            if ((this.ymin < 0) && (hmin >= 0)) this.ymin = 0;
            this.ymax = hmax + dy;
         }
      }

      hmin = this.options.minimum;
      hmax = this.options.maximum;

      if ((hmin === hmax) && (hmin !== -1111)) {
         if (hmin < 0) {
            hmin *= 2; hmax = 0;
         } else {
            hmin = 0; hmax*=2; if (!hmax) hmax = 1;
         }
      }

      if ((hmin != -1111) && (hmax != -1111) && !this.draw_content) {
         this.ymin = hmin;
         this.ymax = hmax;
      } else {
         if (hmin != -1111) {
            if (hmin < this.ymin) this.ymin = hmin; else set_zoom = true;
         }
         if (hmax != -1111) {
            if (hmax > this.ymax) this.ymax = hmax; else set_zoom = true;
         }
      }

      if (!when_axis_changed) {
         if (set_zoom && this.draw_content) {
            this.zoom_ymin = (hmin == -1111) ? this.ymin : hmin;
            this.zoom_ymax = (hmax == -1111) ? this.ymax : hmax;
         } else {
            delete this.zoom_ymin;
            delete this.zoom_ymax;
         }
      }

      // used in FramePainter.isAllowedDefaultYZooming
      this.wheel_zoomy = (this.getDimension() > 1) || !this.draw_content;
   }

   /** @summary Count histogram statistic */
   countStat(cond) {
      let profile = this.isTProfile(),
          histo = this.getHisto(), xaxis = histo.fXaxis,
          left = this.getSelectIndex("x", "left"),
          right = this.getSelectIndex("x", "right"),
          stat_sumw = 0, stat_sumwx = 0, stat_sumwx2 = 0, stat_sumwy = 0, stat_sumwy2 = 0,
          i, xx = 0, w = 0, xmax = null, wmax = null,
          fp = this.getFramePainter(),
          res = { name: histo.fName, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, integral: 0, entries: this.stat_entries, xmax:0, wmax:0 };

      for (i = left; i < right; ++i) {
         xx = xaxis.GetBinCoord(i + 0.5);

         if (cond && !cond(xx)) continue;

         if (profile) {
            w = histo.fBinEntries[i + 1];
            stat_sumwy += histo.fArray[i + 1];
            stat_sumwy2 += histo.fSumw2[i + 1];
         } else {
            w = histo.getBinContent(i + 1);
         }

         if ((xmax === null) || (w > wmax)) { xmax = xx; wmax = w; }

         stat_sumw += w;
         stat_sumwx += w * xx;
         stat_sumwx2 += w * xx * xx;
      }

      // when no range selection done, use original statistic from histogram
      if (!fp.isAxisZoomed("x") && (histo.fTsumw > 0)) {
         stat_sumw = histo.fTsumw;
         stat_sumwx = histo.fTsumwx;
         stat_sumwx2 = histo.fTsumwx2;
      }

      res.integral = stat_sumw;

      if (stat_sumw > 0) {
         res.meanx = stat_sumwx / stat_sumw;
         res.meany = stat_sumwy / stat_sumw;
         res.rmsx = Math.sqrt(Math.abs(stat_sumwx2 / stat_sumw - res.meanx * res.meanx));
         res.rmsy = Math.sqrt(Math.abs(stat_sumwy2 / stat_sumw - res.meany * res.meany));
      }

      if (xmax!==null) {
         res.xmax = xmax;
         res.wmax = wmax;
      }

      return res;
   }

   /** @summary Fill stat box */
   fillStatistic(stat, dostat, dofit) {

      // no need to refill statistic if histogram is dummy
      if (this.isIgnoreStatsFill()) return false;

      let histo = this.getHisto(),
          data = this.countStat(),
          print_name = dostat % 10,
          print_entries = Math.floor(dostat / 10) % 10,
          print_mean = Math.floor(dostat / 100) % 10,
          print_rms = Math.floor(dostat / 1000) % 10,
          print_under = Math.floor(dostat / 10000) % 10,
          print_over = Math.floor(dostat / 100000) % 10,
          print_integral = Math.floor(dostat / 1000000) % 10,
          print_skew = Math.floor(dostat / 10000000) % 10,
          print_kurt = Math.floor(dostat / 100000000) % 10;

      // make empty at the beginning
      stat.clearPave();

      if (print_name > 0)
         stat.addText(data.name);

      if (this.isTProfile()) {

         if (print_entries > 0)
            stat.addText("Entries = " + stat.format(data.entries,"entries"));

         if (print_mean > 0) {
            stat.addText("Mean = " + stat.format(data.meanx));
            stat.addText("Mean y = " + stat.format(data.meany));
         }

         if (print_rms > 0) {
            stat.addText("Std Dev = " + stat.format(data.rmsx));
            stat.addText("Std Dev y = " + stat.format(data.rmsy));
         }

      } else {

         if (print_entries > 0)
            stat.addText("Entries = " + stat.format(data.entries, "entries"));

         if (print_mean > 0)
            stat.addText("Mean = " + stat.format(data.meanx));

         if (print_rms > 0)
            stat.addText("Std Dev = " + stat.format(data.rmsx));

         if (print_under > 0)
            stat.addText("Underflow = " + stat.format((histo.fArray.length > 0) ? histo.fArray[0] : 0, "entries"));

         if (print_over > 0)
            stat.addText("Overflow = " + stat.format((histo.fArray.length > 0) ? histo.fArray[histo.fArray.length - 1] : 0, "entries"));

         if (print_integral > 0)
            stat.addText("Integral = " + stat.format(data.integral, "entries"));

         if (print_skew > 0)
            stat.addText("Skew = <not avail>");

         if (print_kurt > 0)
            stat.addText("Kurt = <not avail>");
      }

      if (dofit) stat.fillFunctionStat(this.findFunction('TF1'), dofit);

      return true;
   }

   /** @summary Draw histogram as bars */
   drawBars(height, pmain, funcs) {

      this.createG(true);

      let left = this.getSelectIndex("x", "left", -1),
          right = this.getSelectIndex("x", "right", 1),
          histo = this.getHisto(), xaxis = histo.fXaxis,
          show_text = this.options.Text, text_col, text_angle, text_size,
          i, x1, x2, grx1, grx2, y, gry1, gry2, w,
          bars = "", barsl = "", barsr = "",
          side = (this.options.BarStyle > 10) ? this.options.BarStyle % 10 : 0;

      if (side>4) side = 4;
      gry2 = pmain.swap_xy ? 0 : height;
      if (Number.isFinite(this.options.BaseLine))
         if (this.options.BaseLine >= funcs.scale_ymin)
            gry2 = Math.round(funcs.gry(this.options.BaseLine));

      if (show_text) {
         text_col = this.getColor(histo.fMarkerColor);
         text_angle = -1*this.options.TextAngle;
         text_size = 20;

         if ((histo.fMarkerSize!==1) && text_angle)
            text_size = 0.02*height*histo.fMarkerSize;

         this.startTextDrawing(42, text_size, this.draw_g, text_size);
      }

      for (i = left; i < right; ++i) {
         x1 = xaxis.GetBinLowEdge(i+1);
         x2 = xaxis.GetBinLowEdge(i+2);

         if (pmain.logx && (x2 <= 0)) continue;

         grx1 = Math.round(funcs.grx(x1));
         grx2 = Math.round(funcs.grx(x2));

         y = histo.getBinContent(i+1);
         if (funcs.logy && (y < funcs.scale_ymin)) continue;
         gry1 = Math.round(funcs.gry(y));

         w = grx2 - grx1;
         grx1 += Math.round(histo.fBarOffset/1000*w);
         w = Math.round(histo.fBarWidth/1000*w);

         if (pmain.swap_xy)
            bars += `M${gry2},${grx1}h${gry1-gry2}v${w}h${gry2-gry1}z`;
         else
            bars += `M${grx1},${gry1}h${w}v${gry2-gry1}h${-w}z`;

         if (side > 0) {
            grx2 = grx1 + w;
            w = Math.round(w * side / 10);
            if (pmain.swap_xy) {
               barsl += `M${gry2},${grx1}h${gry1-gry2}v${w}h${gry2-gry1}z`;
               barsr += `M${gry2},${grx2}h${gry1-gry2}v${-w}h${gry2-gry1}z`;
            } else {
               barsl += `M${grx1},${gry1}h${w}v${gry2-gry1}h${-w}z`;
               barsr += `M${grx2},${gry1}h${-w}v${gry2-gry1}h${w}z`;
            }
         }

         if (show_text && y) {
            let lbl = (y === Math.round(y)) ? y.toString() : floatToString(y, gStyle.fPaintTextFormat);

            if (pmain.swap_xy)
               this.drawText({ align: 12, x: Math.round(gry1 + text_size/2), y: Math.round(grx1+0.1), height: Math.round(w*0.8), text: lbl, color: text_col, latex: 0 });
            else if (text_angle)
               this.drawText({ align: 12, x: grx1+w/2, y: Math.round(gry1 - 2 - text_size/5), width: 0, height: 0, rotate: text_angle, text: lbl, color: text_col, latex: 0 });
            else
               this.drawText({ align: 22, x: Math.round(grx1 + w*0.1), y: Math.round(gry1-2-text_size), width: Math.round(w*0.8), height: text_size, text: lbl, color: text_col, latex: 0 });
         }
      }

      if (bars)
         this.draw_g.append("svg:path")
                    .attr("d", bars)
                    .call(this.fillatt.func);

      if (barsl)
         this.draw_g.append("svg:path")
               .attr("d", barsl)
               .call(this.fillatt.func)
               .style("fill", d3_rgb(this.fillatt.color).brighter(0.5).formatHex());

      if (barsr)
         this.draw_g.append("svg:path")
               .attr("d", barsr)
               .call(this.fillatt.func)
               .style("fill", d3_rgb(this.fillatt.color).darker(0.5).formatHex());

      if (show_text)
         return this.finishTextDrawing();
   }

   /** @summary Draw histogram as filled errors */
   drawFilledErrors(funcs) {
      this.createG(true);

      let left = this.getSelectIndex("x", "left", -1),
          right = this.getSelectIndex("x", "right", 1),
          histo = this.getHisto(), xaxis = histo.fXaxis,
          i, x, grx, y, yerr, gry1, gry2,
          bins1 = [], bins2 = [];

      for (i = left; i < right; ++i) {
         x = xaxis.GetBinCoord(i+0.5);
         if (funcs.logx && (x <= 0)) continue;
         grx = Math.round(funcs.grx(x));

         y = histo.getBinContent(i+1);
         yerr = histo.getBinError(i+1);
         if (funcs.logy && (y-yerr < funcs.scale_ymin)) continue;

         gry1 = Math.round(funcs.gry(y + yerr));
         gry2 = Math.round(funcs.gry(y - yerr));

         bins1.push({ grx:grx, gry: gry1 });
         bins2.unshift({ grx:grx, gry: gry2 });
      }

      let kind = (this.options.ErrorKind === 4) ? "bezier" : "line",
          path1 = buildSvgPath(kind, bins1),
          path2 = buildSvgPath("L"+kind, bins2);

      this.draw_g.append("svg:path")
                 .attr("d", path1.path + path2.path + "Z")
                 .call(this.fillatt.func);
   }

   /** @summary Draw TH1 bins in SVG element
     * @returns Promise or scalar value */
   draw1DBins() {

      this.createHistDrawAttributes();

      let pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          width = pmain.getFrameWidth(), height = pmain.getFrameHeight();

      if (!this.draw_content || (width <= 0) || (height <= 0))
          return this.removeG();

      if (this.options.Bar)
         return this.drawBars(height, pmain, funcs);

      if ((this.options.ErrorKind === 3) || (this.options.ErrorKind === 4))
         return this.drawFilledErrors(pmain, funcs);

      let left = this.getSelectIndex("x", "left", -1),
          right = this.getSelectIndex("x", "right", 2),
          histo = this.getHisto(),
          want_tooltip = !isBatchMode() && settings.Tooltip,
          xaxis = histo.fXaxis,
          res = "", lastbin = false,
          startx, currx, curry, x, grx, y, gry, curry_min, curry_max, prevy, prevx, i, bestimin, bestimax,
          exclude_zero = !this.options.Zero,
          show_errors = this.options.Error,
          show_markers = this.options.Mark,
          show_line = this.options.Line || this.options.Curve,
          show_text = this.options.Text,
          text_profile = show_text && (this.options.TextKind == "E") && this.isTProfile() && histo.fBinEntries,
          path_fill = null, path_err = null, path_marker = null, path_line = null,
          hints_err = null, hints_marker = null, hsz = 5,
          do_marker = false, do_err = false,
          dend = 0, dlw = 0, my, yerr1, yerr2, bincont, binerr, mx1, mx2, midx, mmx1, mmx2,
          text_col, text_angle, text_size;

      if (show_errors && !show_markers && (histo.fMarkerStyle > 1))
         show_markers = true;

      if (this.options.ErrorKind === 2) {
         if (this.fillatt.empty()) show_markers = true;
                              else path_fill = "";
      } else if (this.options.Error) {
         path_err = "";
         hints_err = want_tooltip ? "" : null;
         do_err = true;
      }

      if (show_line) path_line = "";

      dlw = this.lineatt.width + gStyle.fEndErrorSize;
      if (this.options.ErrorKind === 1)
         dend = Math.floor((this.lineatt.width-1)/2);

      if (show_markers) {
         // draw markers also when e2 option was specified
         this.createAttMarker({ attr: histo, style: this.options.MarkStyle }); // when style not configured, it will be ignored
         if (this.markeratt.size > 0) {
            // simply use relative move from point, can optimize in the future
            path_marker = "";
            do_marker = true;
            this.markeratt.resetPos();
            if ((hints_err === null) && want_tooltip && (!this.markeratt.fill || (this.markeratt.getFullSize() < 7))) {
               hints_marker = "";
               hsz = Math.max(5, Math.round(this.markeratt.getFullSize()*0.7));
             }
         } else {
            show_markers = false;
         }
      }

      let draw_markers = show_errors || show_markers,
          draw_any_but_hist = draw_markers || show_text || show_line,
          draw_hist = this.options.Hist && (!this.lineatt.empty() || !this.fillatt.empty());

      if (!draw_hist && !draw_any_but_hist)
         return this.removeG();

      this.createG(true);

      if (show_text) {
         text_col = this.getColor(histo.fMarkerColor);
         text_angle = -1*this.options.TextAngle;
         text_size = 20;

         if ((histo.fMarkerSize!==1) && text_angle)
            text_size = 0.02*height*histo.fMarkerSize;

         if (!text_angle && !this.options.TextKind) {
             let space = width / (right - left + 1);
             if (space < 3 * text_size) {
                text_angle = 270;
                text_size = Math.round(space*0.7);
             }
         }

         this.startTextDrawing(42, text_size, this.draw_g, text_size);
      }

      // if there are too many points, exclude many vertical drawings at the same X position
      // instead define min and max value and made min-max drawing
      let use_minmax = draw_any_but_hist || ((right - left) > 3*width);

      // just to get correct values for the specified bin
      const extract_bin = bin => {
         bincont = histo.getBinContent(bin+1);
         if (exclude_zero && (bincont===0)) return false;
         mx1 = Math.round(funcs.grx(xaxis.GetBinLowEdge(bin+1)));
         mx2 = Math.round(funcs.grx(xaxis.GetBinLowEdge(bin+2)));
         midx = Math.round((mx1+mx2)/2);
         my = Math.round(funcs.gry(bincont));
         if (show_errors) {
            binerr = histo.getBinError(bin+1);
            yerr1 = Math.round(my - funcs.gry(bincont + binerr)); // up
            yerr2 = Math.round(funcs.gry(bincont - binerr) - my); // down
         } else {
            yerr1 = yerr2 = 20;
         }
         return true;
      };

      let draw_errbin = () => {
         let edx = 5;
         if (this.options.errorX > 0) {
            edx = Math.round((mx2-mx1)*this.options.errorX);
            mmx1 = midx - edx;
            mmx2 = midx + edx;
            if (this.options.ErrorKind === 1)
               path_err += `M${mmx1+dend},${my-dlw}v${2*dlw}m0,-${dlw}h${mmx2-mmx1-2*dend}m0,-${dlw}v${2*dlw}`;
            else
               path_err += `M${mmx1+dend},${my}h${mmx2-mmx1-2*dend}`;
         }
         if (this.options.ErrorKind === 1)
            path_err += `M${midx-dlw},${my-yerr1+dend}h${2*dlw}m${-dlw},0v${yerr1+yerr2-2*dend}m${-dlw},0h${2*dlw}`;
         else
            path_err += `M${midx},${my-yerr1+dend}v${yerr1+yerr2-2*dend}`;
         if (hints_err !== null)
            hints_err += `M${midx-edx},${my-yerr1}h${2*edx}v${yerr1+yerr2}h${-2*edx}z`;
      };

      const draw_bin = bin => {
         if (extract_bin(bin)) {
            if (show_text) {
               let cont = text_profile ? histo.fBinEntries[bin+1] : bincont;

               if (cont!==0) {
                  let lbl = (cont === Math.round(cont)) ? cont.toString() : floatToString(cont, gStyle.fPaintTextFormat);

                  if (text_angle)
                     this.drawText({ align: 12, x: midx, y: Math.round(my - 2 - text_size/5), width: 0, height: 0, rotate: text_angle, text: lbl, color: text_col, latex: 0 });
                  else
                     this.drawText({ align: 22, x: Math.round(mx1 + (mx2-mx1)*0.1), y: Math.round(my-2-text_size), width: Math.round((mx2-mx1)*0.8), height: text_size, text: lbl, color: text_col, latex: 0 });
               }
            }

            if (show_line && (path_line !== null))
               path_line += ((path_line.length===0) ? "M" : "L") + midx + "," + my;

            if (draw_markers) {
               if ((my >= -yerr1) && (my <= height + yerr2)) {
                  if (path_fill !== null)
                     path_fill += `M${mx1},${my-yerr1}h${mx2-mx1}v${yerr1+yerr2+1}h${mx1-mx2}z`;
                  if ((path_marker !== null) && do_marker) {
                     path_marker += this.markeratt.create(midx, my);
                     if (hints_marker !== null)
                        hints_marker += `M${midx-hsz},${my-hsz}h${2*hsz}v${2*hsz}h${-2*hsz}z`;
                  }

                  if ((path_err !== null) && do_err)
                     draw_errbin();
               }
            }
         }
      };

      // check if we should draw markers or error marks directly, skipping optimization
      if (do_marker || do_err)
         if (!settings.OptimizeDraw || ((right-left < 50000) && (settings.OptimizeDraw == 1))) {
            for (i = left; i < right; ++i) {
               if (extract_bin(i)) {
                  if (path_marker !== null)
                     path_marker += this.markeratt.create(midx, my);
                  if (hints_marker !== null)
                     hints_marker += `M${midx-hsz},${my-hsz}h${2*hsz}v${2*hsz}h${-2*hsz}z`;
                  if (path_err !== null)
                     draw_errbin();
               }
            }
            do_err = do_marker = false;
         }


      for (i = left; i <= right; ++i) {

         x = xaxis.GetBinLowEdge(i+1);

         if (this.logx && (x <= 0)) continue;

         grx = Math.round(funcs.grx(x));

         lastbin = (i === right);

         if (lastbin && (left<right)) {
            gry = curry;
         } else {
            y = histo.getBinContent(i+1);
            gry = Math.round(funcs.gry(y));
         }

         if (res.length === 0) {
            bestimin = bestimax = i;
            prevx = startx = currx = grx;
            prevy = curry_min = curry_max = curry = gry;
            res = "M"+currx+","+curry;
         } else if (use_minmax) {
            if ((grx === currx) && !lastbin) {
               if (gry < curry_min) bestimax = i; else
               if (gry > curry_max) bestimin = i;

               curry_min = Math.min(curry_min, gry);
               curry_max = Math.max(curry_max, gry);
               curry = gry;
            } else {

               if (draw_any_but_hist) {
                  if (bestimin === bestimax) { draw_bin(bestimin); } else
                  if (bestimin < bestimax) { draw_bin(bestimin); draw_bin(bestimax); } else {
                     draw_bin(bestimax); draw_bin(bestimin);
                  }
               }

               // when several points at same X differs, need complete logic
               if (draw_hist && ((curry_min !== curry_max) || (prevy !== curry_min))) {

                  if (prevx !== currx)
                     res += "h"+(currx-prevx);

                  if (curry === curry_min) {
                     if (curry_max !== prevy)
                        res += "v" + (curry_max - prevy);
                     if (curry_min !== curry_max)
                        res += "v" + (curry_min - curry_max);
                  } else {
                     if (curry_min !== prevy)
                        res += "v" + (curry_min - prevy);
                     if (curry_max !== curry_min)
                        res += "v" + (curry_max - curry_min);
                     if (curry !== curry_max)
                       res += "v" + (curry - curry_max);
                  }

                  prevx = currx;
                  prevy = curry;
               }

               if (lastbin && (prevx !== grx))
                  res += "h"+(grx-prevx);

               bestimin = bestimax = i;
               curry_min = curry_max = curry = gry;
               currx = grx;
            }
            // end of use_minmax
         } else if ((gry !== curry) || lastbin) {
            if (grx !== currx) res += "h"+(grx-currx);
            if (gry !== curry) res += "v"+(gry-curry);
            curry = gry;
            currx = grx;
         }
      }

      let fill_for_interactive = want_tooltip && this.fillatt.empty() && draw_hist && !draw_markers && !show_line,
          h0 = height + 3;
      if (!fill_for_interactive) {
         let gry0 = Math.round(funcs.gry(0));
         if (gry0 <= 0) h0 = -3; else if (gry0 < height) h0 = gry0;
      }
      let close_path = `L${currx},${h0}H${startx}Z`;

      if (draw_markers || show_line) {
         if ((path_fill !== null) && (path_fill.length > 0))
            this.draw_g.append("svg:path")
                       .attr("d", path_fill)
                       .call(this.fillatt.func);

         if (path_err)
               this.draw_g.append("svg:path")
                   .attr("d", path_err)
                   .call(this.lineatt.func);

          if (hints_err)
               this.draw_g.append("svg:path")
                   .attr("d", hints_err)
                   .style("fill", "none")
                   .style("pointer-events", isBatchMode() ? null : "visibleFill");

         if (path_line) {
            if (!this.fillatt.empty() && !draw_hist)
               this.draw_g.append("svg:path")
                     .attr("d", path_line + close_path)
                     .call(this.fillatt.func);

            this.draw_g.append("svg:path")
                   .attr("d", path_line)
                   .style("fill", "none")
                   .call(this.lineatt.func);
         }

         if (path_marker)
            this.draw_g.append("svg:path")
                .attr("d", path_marker)
                .call(this.markeratt.func);

         if (hints_marker)
            this.draw_g.append("svg:path")
                .attr("d", hints_marker)
                .style("fill", "none")
                .style("pointer-events", isBatchMode() ? null : "visibleFill");
      }

      if (res && draw_hist)
         this.draw_g.append("svg:path")
                    .attr("d", res + ((!this.fillatt.empty() || fill_for_interactive) ? close_path : ""))
                    .style("stroke-linejoin","miter")
                    .call(this.lineatt.func)
                    .call(this.fillatt.func);

      if (show_text)
         return this.finishTextDrawing();
   }

   /** @summary Provide text information (tooltips) for histogram bin */
   getBinTooltips(bin) {
      let tips = [],
          name = this.getObjectHint(),
          pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          histo = this.getHisto(),
          x1 = histo.fXaxis.GetBinLowEdge(bin+1),
          x2 = histo.fXaxis.GetBinLowEdge(bin+2),
          cont = histo.getBinContent(bin+1),
          xlbl = this.getAxisBinTip("x", histo.fXaxis, bin);

      if (name.length > 0) tips.push(name);

      if (this.options.Error || this.options.Mark) {
         tips.push("x = " + xlbl);
         tips.push("y = " + funcs.axisAsText("y", cont));
         if (this.options.Error) {
            if (xlbl[0] == "[") tips.push("error x = " + ((x2 - x1) / 2).toPrecision(4));
            tips.push("error y = " + histo.getBinError(bin + 1).toPrecision(4));
         }
      } else {
         tips.push("bin = " + (bin+1));
         tips.push("x = " + xlbl);
         if (histo['$baseh']) cont -= histo['$baseh'].getBinContent(bin+1);
         if (cont === Math.round(cont))
            tips.push("entries = " + cont);
         else
            tips.push("entries = " + floatToString(cont, gStyle.fStatFormat));
      }

      return tips;
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      if (!pnt || !this.draw_content || !this.draw_g || this.options.Mode3D) {
         if (this.draw_g)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      const pmain = this.getFramePainter(),
           funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
           histo = this.getHisto(),
           left = this.getSelectIndex("x", "left", -1),
           right = this.getSelectIndex("x", "right", 2);
      let width = pmain.getFrameWidth(),
          height = pmain.getFrameHeight(),
          findbin = null, show_rect,
          grx1, midx, grx2, gry1, midy, gry2, gapx = 2,
          l = left, r = right, pnt_x = pnt.x, pnt_y = pnt.y;

      const GetBinGrX = i => {
         let xx = histo.fXaxis.GetBinLowEdge(i+1);
         return (funcs.logx && (xx<=0)) ? null : funcs.grx(xx);
      }, GetBinGrY = i => {
         let yy = histo.getBinContent(i + 1);
         if (funcs.logy && (yy < funcs.scale_ymin))
            return funcs.swap_xy ? -1000 : 10*height;
         return Math.round(funcs.gry(yy));
      };

      if (funcs.swap_xy) {
         let d = pnt.x; pnt_x = pnt_y; pnt_y = d;
         d = height; height = width; width = d;
      }

      while (l < r-1) {
         let m = Math.round((l+r)*0.5), xx = GetBinGrX(m);
         if ((xx === null) || (xx < pnt_x - 0.5)) {
            if (funcs.swap_xy) r = m; else l = m;
         } else if (xx > pnt_x + 0.5) {
            if (funcs.swap_xy) l = m; else r = m;
         } else { l++; r--; }
      }

      findbin = r = l;
      grx1 = GetBinGrX(findbin);

      if (pmain.swap_xy) {
         while ((l > left) && (GetBinGrX(l-1) < grx1 + 2)) --l;
         while ((r < right) && (GetBinGrX(r+1) > grx1 - 2)) ++r;
      } else {
         while ((l > left) && (GetBinGrX(l-1) > grx1 - 2)) --l;
         while ((r < right) && (GetBinGrX(r+1) < grx1 + 2)) ++r;
      }

      if (l < r) {
         // many points can be assigned with the same cursor position
         // first try point around mouse y
         let best = height;
         for (let m = l; m <= r; m++) {
            let dist = Math.abs(GetBinGrY(m) - pnt_y);
            if (dist < best) { best = dist; findbin = m; }
         }

         // if best distance still too far from mouse position, just take from between
         if (best > height/10)
            findbin = Math.round(l + (r-l) / height * pnt_y);

         grx1 = GetBinGrX(findbin);
      }

      grx1 = Math.round(grx1);
      grx2 = Math.round(GetBinGrX(findbin+1));

      if (this.options.Bar) {
         let w = grx2 - grx1;
         grx1 += Math.round(histo.fBarOffset/1000*w);
         grx2 = grx1 + Math.round(histo.fBarWidth/1000*w);
      }

      if (grx1 > grx2) { let d = grx1; grx1 = grx2; grx2 = d; }

      midx = Math.round((grx1+grx2)/2);

      midy = gry1 = gry2 = GetBinGrY(findbin);

      if (this.options.Bar) {
         show_rect = true;

         gapx = 0;

         gry1 = Math.round(funcs.gry(((this.options.BaseLine!==false) && (this.options.BaseLine > funcs.scale_ymin)) ? this.options.BaseLine : funcs.scale_ymin));

         if (gry1 > gry2) { let d = gry1; gry1 = gry2; gry2 = d; }

         if (!pnt.touch && (pnt.nproc === 1))
            if ((pnt_y < gry1) || (pnt_y > gry2)) findbin = null;

      } else if (this.options.Error || this.options.Mark || this.options.Line || this.options.Curve)  {

         show_rect = true;

         let msize = 3;
         if (this.markeratt) msize = Math.max(msize, this.markeratt.getFullSize());

         if (this.options.Error) {
            let cont = histo.getBinContent(findbin+1),
                binerr = histo.getBinError(findbin+1);

            gry1 = Math.round(funcs.gry(cont + binerr)); // up
            gry2 = Math.round(funcs.gry(cont - binerr)); // down

            if ((cont==0) && this.isTProfile()) findbin = null;

            let dx = (grx2-grx1)*this.options.errorX;
            grx1 = Math.round(midx - dx);
            grx2 = Math.round(midx + dx);
         }

         // show at least 6 pixels as tooltip rect
         if (grx2 - grx1 < 2*msize) { grx1 = midx-msize; grx2 = midx+msize; }

         gry1 = Math.min(gry1, midy - msize);
         gry2 = Math.max(gry2, midy + msize);

         if (!pnt.touch && (pnt.nproc === 1))
            if ((pnt_y<gry1) || (pnt_y>gry2)) findbin = null;

      } else {

         // if histogram alone, use old-style with rects
         // if there are too many points at pixel, use circle
         show_rect = (pnt.nproc === 1) && (right-left < width);

         if (show_rect) {
            gry2 = height;

            if (!this.fillatt.empty()) {
               gry2 = Math.round(funcs.gry(0));
               if (gry2 < 0) gry2 = 0; else if (gry2 > height) gry2 = height;
               if (gry2 < gry1) { let d = gry1; gry1 = gry2; gry2 = d; }
            }

            // for mouse events pointer should be between y1 and y2
            if (((pnt.y < gry1) || (pnt.y > gry2)) && !pnt.touch) findbin = null;
         }
      }

      if (findbin !== null) {
         // if bin on boundary found, check that x position is ok
         if ((findbin === left) && (grx1 > pnt_x + gapx))  findbin = null; else
         if ((findbin === right-1) && (grx2 < pnt_x - gapx)) findbin = null; else
         // if bars option used check that bar is not match
         if ((pnt_x < grx1 - gapx) || (pnt_x > grx2 + gapx)) findbin = null; else
         // exclude empty bin if empty bins suppressed
         if (!this.options.Zero && (histo.getBinContent(findbin+1)===0)) findbin = null;
      }

      let ttrect = this.draw_g.select(".tooltip_bin");

      if ((findbin === null) || ((gry2 <= 0) || (gry1 >= height))) {
         ttrect.remove();
         return null;
      }

      let res = { name: histo.fName, title: histo.fTitle,
                  x: midx, y: midy, exact: true,
                  color1: this.lineatt ? this.lineatt.color : 'green',
                  color2: this.fillatt ? this.fillatt.getFillColorAlt('blue') : 'blue',
                  lines: this.getBinTooltips(findbin) };

      if (pnt.disabled) {
         // case when tooltip should not highlight bin

         ttrect.remove();
         res.changed = true;
      } else if (show_rect) {

         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:rect")
                                .attr("class","tooltip_bin h1bin")
                                .style("pointer-events","none");

         res.changed = ttrect.property("current_bin") !== findbin;

         if (res.changed)
            ttrect.attr("x", funcs.swap_xy ? gry1 : grx1)
                  .attr("width", funcs.swap_xy ? gry2-gry1 : grx2-grx1)
                  .attr("y", funcs.swap_xy ? grx1 : gry1)
                  .attr("height", funcs.swap_xy ? grx2-grx1 : gry2-gry1)
                  .style("opacity", "0.3")
                  .property("current_bin", findbin);

         res.exact = (Math.abs(midy - pnt_y) <= 5) || ((pnt_y >= gry1) && (pnt_y <= gry2));

         res.menu = true; // one could show context menu
         // distance to middle point, use to decide which menu to activate
         res.menu_dist = Math.sqrt((midx-pnt_x)*(midx-pnt_x) + (midy-pnt_y)*(midy-pnt_y));

      } else {
         let radius = this.lineatt.width + 3;

         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:circle")
                                .attr("class","tooltip_bin")
                                .style("pointer-events","none")
                                .attr("r", radius)
                                .call(this.lineatt.func)
                                .call(this.fillatt.func);

         res.exact = (Math.abs(midx - pnt.x) <= radius) && (Math.abs(midy - pnt.y) <= radius);

         res.menu = res.exact; // show menu only when mouse pointer exactly over the histogram
         res.menu_dist = Math.sqrt((midx-pnt.x)*(midx-pnt.x) + (midy-pnt.y)*(midy-pnt.y));

         res.changed = ttrect.property("current_bin") !== findbin;

         if (res.changed)
            ttrect.attr("cx", midx)
                  .attr("cy", midy)
                  .property("current_bin", findbin);
      }

      if (res.changed)
         res.user_info = { obj: histo,  name: histo.fName,
                           bin: findbin, cont: histo.getBinContent(findbin+1),
                           grx: midx, gry: midy };

      return res;
   }

   /** @summary Fill histogram context menu */
   fillHistContextMenu(menu) {

      menu.add("Auto zoom-in", () => this.autoZoom());

      let opts = this.getSupportedDrawOptions();

      menu.addDrawMenu("Draw with", opts, arg => {
         if (arg==='inspect')
            return this.showInspector();

         this.decodeOptions(arg);

         if (this.options.need_fillcol && this.fillatt && this.fillatt.empty())
            this.fillatt.change(5,1001);

         // redraw all objects in pad, inform dependent objects
         this.interactiveRedraw("pad", "drawopt");
      });

      if (!this.snapid && !this.isTProfile())
         menu.addRebinMenu(sz => this.rebinHist(sz));
   }

   /** @summary Rebin histogram, used via context menu */
   rebinHist(sz) {
      let histo = this.getHisto(),
          xaxis = histo.fXaxis,
          nbins = Math.floor(xaxis.fNbins/ sz);
      if (nbins < 2) return;

      let arr = new Array(nbins+2), xbins = null;

      if (xaxis.fXbins.length > 0)
         xbins = new Array(nbins);

      arr[0] = histo.fArray[0];
      let indx = 1;

      for (let i = 1; i <= nbins; ++i) {
         if (xbins) xbins[i-1] = xaxis.fXbins[indx-1];
         let sum = 0;
         for (let k = 0; k < sz; ++k)
           sum += histo.fArray[indx++];
         arr[i] = sum;

      }

      if (xbins) {
         if (indx <= xaxis.fXbins.length)
            xaxis.fXmax = xaxis.fXbins[indx-1];
         xaxis.fXbins = xbins;
      } else {
         xaxis.fXmax = xaxis.fXmin + (xaxis.fXmax - xaxis.fXmin) / xaxis.fNbins * nbins * sz;
      }

      xaxis.fNbins = nbins;

      let overflow = 0;
      while (indx < histo.fArray.length)
         overflow += histo.fArray[indx++];
      arr[nbins+1] = overflow;

      histo.fArray = arr;
      histo.fSumw2 = [];

      this.scanContent();

      this.interactiveRedraw("pad");
   }

   /** @summary Perform automatic zoom inside non-zero region of histogram */
   autoZoom() {
      let left = this.getSelectIndex("x", "left", -1),
          right = this.getSelectIndex("x", "right", 1),
          dist = right - left, histo = this.getHisto();

      if ((dist == 0) || !histo) return;

      // first find minimum
      let min = histo.getBinContent(left + 1);
      for (let indx = left; indx < right; ++indx)
         min = Math.min(min, histo.getBinContent(indx+1));
      if (min > 0) return; // if all points positive, no chance for autoscale

      while ((left < right) && (histo.getBinContent(left+1) <= min)) ++left;
      while ((left < right) && (histo.getBinContent(right) <= min)) --right;

      // if singular bin
      if ((left === right-1) && (left > 2) && (right < this.nbinsx-2)) {
         --left; ++right;
      }

      if ((right - left < dist) && (left < right))
         return this.getFramePainter().zoom(histo.fXaxis.GetBinLowEdge(left+1), histo.fXaxis.GetBinLowEdge(right+1));
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis,min,max) {
      let histo = this.getHisto();

      if ((axis=="x") && histo && (histo.fXaxis.FindBin(max,0.5) - histo.fXaxis.FindBin(min,0) > 1)) return true;

      if ((axis=="y") && (Math.abs(max-min) > Math.abs(this.ymax-this.ymin)*1e-6)) return true;

      // check if it makes sense to zoom inside specified axis range
      return false;
   }

   /** @summary Call drawing function depending from 3D mode */
   callDrawFunc(reason) {

      let main = this.getMainPainter(),
          fp = this.getFramePainter();

     if ((main !== this) && fp && (fp.mode3d !== this.options.Mode3D))
        this.copyOptionsFrom(main);

      return this.options.Mode3D ? this.draw3D(reason) : this.draw2D(reason);
   }

   /** @summary Performs 2D drawing of histogram
     * @returns {Promise} when ready */
   draw2D(/* reason */) {
      this.clear3DScene();

      this.scanContent(true);

      let pr = this.isMainPainter() ? this.drawColorPalette(false) : Promise.resolve(true);

      return pr.then(() => this.drawAxes())
               .then(() => this.draw1DBins())
               .then(() => this.drawHistTitle())
               .then(() => {
                   this.updateStatWebCanvas();
                   return this.addInteractivity();
               });
   }

   /** @summary Should performs 3D drawing of histogram
     * @desc Disable in 2D case, just draw with default options
     * @returns {Promise} when ready */
   draw3D(reason) {
      console.log('3D drawing is disabled, load ./hist/TH1Painter.mjs');
      return this.draw2D(reason);
   }

   /** @summary Redraw histogram */
   redraw(reason) {
      return this.callDrawFunc(reason);
   }

   /** @summary draw TH1 object */
   static draw(dom, histo, opt) {
      return TH1Painter._drawHist(new TH1Painter(dom, histo), opt);
   }

} // class TH1Painter

export { TH1Painter };
