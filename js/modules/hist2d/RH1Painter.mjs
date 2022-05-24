import { gStyle, settings, constants, isBatchMode } from '../core.mjs';

import { rgb as d3_rgb } from '../d3.mjs';

import { floatToString, DrawOptions, buildSvgPath } from '../base/BasePainter.mjs';

import { RHistPainter } from './RHistPainter.mjs';

import { ensureRCanvas } from '../gpad/RCanvasPainter.mjs';

/**
 * @summary Painter for RH1 classes
 *
 * @private
 */

class RH1Painter extends RHistPainter {

   /** @summary Constructor
     * @param {object|string} dom - DOM element or id
     * @param {object} histo - histogram object */
   constructor(dom, histo) {
      super(dom, histo);
      this.wheel_zoomy = false;
   }

   /** @summary Scan content */
   scanContent(when_axis_changed) {
      // if when_axis_changed === true specified, content will be scanned after axis zoom changed

      let histo = this.getHisto();
      if (!histo) return;

      if (!this.nbinsx && when_axis_changed) when_axis_changed = false;

      if (!when_axis_changed)
         this.extractAxesProperties(1);

      let hmin = 0, hmin_nz = 0, hmax = 0, hsum = 0;

      if (this.isDisplayItem()) {
         // take min/max values from the display item
         hmin = histo.fContMin;
         hmin_nz = histo.fContMinPos;
         hmax = histo.fContMax;
         hsum = hmax;
      } else {

         let left = this.getSelectIndex("x", "left"),
             right = this.getSelectIndex("x", "right");

         if (when_axis_changed) {
            if ((left === this.scan_xleft) && (right === this.scan_xright)) return;
         }

         this.scan_xleft = left;
         this.scan_xright = right;

         let first = true, value, err;

         for (let i = 0; i < this.nbinsx; ++i) {
            value = histo.getBinContent(i+1);
            hsum += value;

            if ((i<left) || (i>=right)) continue;

            if (value > 0)
               if ((hmin_nz == 0) || (value<hmin_nz)) hmin_nz = value;
            if (first) {
               hmin = hmax = value;
               first = false;
            }

            err =  0;

            hmin = Math.min(hmin, value - err);
            hmax = Math.max(hmax, value + err);
         }
      }

      this.stat_entries = hsum;

      this.hmin = hmin;
      this.hmax = hmax;

      this.ymin_nz = hmin_nz; // value can be used to show optimal log scale

      if ((this.nbinsx == 0) || ((Math.abs(hmin) < 1e-300 && Math.abs(hmax) < 1e-300))) {
         this.draw_content = false;
      } else {
         this.draw_content = true;
      }

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
   }

   /** @summary Count statistic */
   countStat(cond) {
      let histo = this.getHisto(), xaxis = this.getAxis("x"),
          left = this.getSelectIndex("x", "left"),
          right = this.getSelectIndex("x", "right"),
          stat_sumw = 0, stat_sumwx = 0, stat_sumwx2 = 0, stat_sumwy = 0, stat_sumwy2 = 0,
          i, xx = 0, w = 0, xmax = null, wmax = null,
          fp = this.getFramePainter(),
          res = { name: "histo", meanx: 0, meany: 0, rmsx: 0, rmsy: 0, integral: 0, entries: this.stat_entries, xmax:0, wmax:0 };

      for (i = left; i < right; ++i) {
         xx = xaxis.GetBinCoord(i+0.5);

         if (cond && !cond(xx)) continue;

         w = histo.getBinContent(i + 1);

         if ((xmax === null) || (w > wmax)) { xmax = xx; wmax = w; }

         stat_sumw += w;
         stat_sumwx += w * xx;
         stat_sumwx2 += w * xx * xx;
      }

      // when no range selection done, use original statistic from histogram
      if (!fp.isAxisZoomed("x") && histo.fTsumw) {
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

      if (xmax !== null) {
         res.xmax = xmax;
         res.wmax = wmax;
      }

      return res;
   }

   /** @summary Fill statistic */
   fillStatistic(stat, dostat/*, dofit*/) {

      let data = this.countStat(),
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
      stat.clearStat();

      if (print_name > 0)
         stat.addText(data.name);

      if (print_entries > 0)
         stat.addText("Entries = " + stat.format(data.entries,"entries"));

      if (print_mean > 0)
         stat.addText("Mean = " + stat.format(data.meanx));

      if (print_rms > 0)
         stat.addText("Std Dev = " + stat.format(data.rmsx));

      if (print_under > 0)
         stat.addText("Underflow = " + stat.format(histo.getBinContent(0), "entries"));

      if (print_over > 0)
         stat.addText("Overflow = " + stat.format(histo.getBinContent(this.nbinsx+1), "entries"));

      if (print_integral > 0)
         stat.addText("Integral = " + stat.format(data.integral,"entries"));

      if (print_skew > 0)
         stat.addText("Skew = <not avail>");

      if (print_kurt > 0)
         stat.addText("Kurt = <not avail>");

      return true;
   }

   /** @summary Draw histogram as bars */
   drawBars(handle, funcs, width, height) {

      this.createG(true);

      let left = handle.i1, right = handle.i2, di = handle.stepi,
          pmain = this.getFramePainter(),
          histo = this.getHisto(), xaxis = this.getAxis("x"),
          i, x1, x2, grx1, grx2, y, gry1, gry2, w,
          bars = "", barsl = "", barsr = "";

      gry2 = pmain.swap_xy ? 0 : height;
      if (Number.isFinite(this.options.BaseLine))
         if (this.options.BaseLine >= funcs.scale_ymin)
            gry2 = Math.round(funcs.gry(this.options.BaseLine));

      for (i = left; i < right; i += di) {
         x1 = xaxis.GetBinCoord(i);
         x2 = xaxis.GetBinCoord(i+di);

         if (funcs.logx && (x2 <= 0)) continue;

         grx1 = Math.round(funcs.grx(x1));
         grx2 = Math.round(funcs.grx(x2));

         y = histo.getBinContent(i+1);
         if (funcs.logy && (y < funcs.scale_ymin)) continue;
         gry1 = Math.round(funcs.gry(y));

         w = grx2 - grx1;
         grx1 += Math.round(this.options.BarOffset*w);
         w = Math.round(this.options.BarWidth*w);

         if (pmain.swap_xy)
            bars += "M"+gry2+","+grx1 + "h"+(gry1-gry2) + "v"+w + "h"+(gry2-gry1) + "z";
         else
            bars += "M"+grx1+","+gry1 + "h"+w + "v"+(gry2-gry1) + "h"+(-w)+ "z";

         if (this.options.BarStyle > 0) {
            grx2 = grx1 + w;
            w = Math.round(w / 10);
            if (pmain.swap_xy) {
               barsl += "M"+gry2+","+grx1 + "h"+(gry1-gry2) + "v" + w + "h"+(gry2-gry1) + "z";
               barsr += "M"+gry2+","+grx2 + "h"+(gry1-gry2) + "v" + (-w) + "h"+(gry2-gry1) + "z";
            } else {
               barsl += "M"+grx1+","+gry1 + "h"+w + "v"+(gry2-gry1) + "h"+(-w)+ "z";
               barsr += "M"+grx2+","+gry1 + "h"+(-w) + "v"+(gry2-gry1) + "h"+w + "z";
            }
         }
      }

      if (this.fillatt.empty()) this.fillatt.setSolidColor("blue");

      if (bars.length > 0)
         this.draw_g.append("svg:path")
                    .attr("d", bars)
                    .call(this.fillatt.func);

      if (barsl.length > 0)
         this.draw_g.append("svg:path")
               .attr("d", barsl)
               .call(this.fillatt.func)
               .style("fill", d3_rgb(this.fillatt.color).brighter(0.5).formatHex());

      if (barsr.length > 0)
         this.draw_g.append("svg:path")
               .attr("d", barsr)
               .call(this.fillatt.func)
               .style("fill", d3_rgb(this.fillatt.color).darker(0.5).formatHex());

       return Promise.resolve(true);
   }

   /** @summary Draw histogram as filled errors */
   drawFilledErrors(handle, funcs /*, width, height*/) {
      this.createG(true);

      let left = handle.i1, right = handle.i2, di = handle.stepi,
          histo = this.getHisto(), xaxis = this.getAxis("x"),
          i, x, grx, y, yerr, gry1, gry2,
          bins1 = [], bins2 = [];

      for (i = left; i < right; i += di) {
         x = xaxis.GetBinCoord(i+0.5);
         if (funcs.logx && (x <= 0)) continue;
         grx = Math.round(funcs.grx(x));

         y = histo.getBinContent(i+1);
         yerr = histo.getBinError(i+1);
         if (funcs.logy && (y-yerr < funcs.scale_ymin)) continue;

         gry1 = Math.round(funcs.gry(y + yerr));
         gry2 = Math.round(funcs.gry(y - yerr));

         bins1.push({grx: grx, gry: gry1});
         bins2.unshift({grx: grx, gry: gry2});
      }

      let kind = (this.options.ErrorKind === 4) ? "bezier" : "line",
          path1 = buildSvgPath(kind, bins1),
          path2 = buildSvgPath("L"+kind, bins2);

      if (this.fillatt.empty()) this.fillatt.setSolidColor("blue");

      this.draw_g.append("svg:path")
                 .attr("d", path1.path + path2.path + "Z")
                 .call(this.fillatt.func);

      return Promise.resolve(true);
   }

   /** @summary Draw 1D histogram as SVG */
   draw1DBins() {

      let pmain = this.getFramePainter(),
          rect = pmain.getFrameRect();

      if (!this.draw_content || (rect.width <= 0) || (rect.height <= 0)) {
         this.removeG()
         return Promise.resolve(false);
      }

      this.createHistDrawAttributes();

      let handle = this.prepareDraw({ extra: 1, only_indexes: true }),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y);

      if (this.options.Bar)
         return this.drawBars(handle, funcs, rect.width, rect.height);

      if ((this.options.ErrorKind === 3) || (this.options.ErrorKind === 4))
         return this.drawFilledErrors(handle, funcs, rect.width, rect.height);

      return this.drawHistBins(handle, funcs, rect.width, rect.height);
   }

   /** @summary Draw histogram bins */
   drawHistBins(handle, funcs, width, height) {
      this.createG(true);

      let options = this.options,
          left = handle.i1,
          right = handle.i2,
          di = handle.stepi,
          histo = this.getHisto(),
          want_tooltip = !isBatchMode() && settings.Tooltip,
          xaxis = this.getAxis("x"),
          res = "", lastbin = false,
          startx, currx, curry, x, grx, y, gry, curry_min, curry_max, prevy, prevx, i, bestimin, bestimax,
          exclude_zero = !options.Zero,
          show_errors = options.Error,
          show_markers = options.Mark,
          show_line = options.Line,
          show_text = options.Text,
          path_fill = null, path_err = null, path_marker = null, path_line = null,
          hints_err = null,
          endx = "", endy = "", dend = 0, my, yerr1, yerr2, bincont, binerr, mx1, mx2, midx,
          text_font;

      if (show_errors && !show_markers && (this.v7EvalAttr("marker_style",1) > 1))
         show_markers = true;

      if (options.ErrorKind === 2) {
         if (this.fillatt.empty()) show_markers = true;
                              else path_fill = "";
      } else if (options.Error) {
         path_err = "";
         hints_err = want_tooltip ? "" : null;
      }

      if (show_line) path_line = "";

      if (show_markers) {
         // draw markers also when e2 option was specified
         this.createv7AttMarker();
         if (this.markeratt.size > 0) {
            // simply use relative move from point, can optimize in the future
            path_marker = "";
            this.markeratt.resetPos();
         } else {
            show_markers = false;
         }
      }

      if (show_text) {
         text_font = this.v7EvalFont("text", { size: 20, color: "black", align: 22 });

         if (!text_font.angle && !options.TextKind) {
             let space = width / (right - left + 1);
             if (space < 3 * text_font.size) {
                text_font.setAngle(270);
                text_font.setSize(Math.round(space*0.7));
             }
         }

         this.startTextDrawing(text_font, 'font');
      }

      // if there are too many points, exclude many vertical drawings at the same X position
      // instead define min and max value and made min-max drawing
      let use_minmax = ((right-left) > 3*width);

      if (options.ErrorKind === 1) {
         let lw = this.lineatt.width + gStyle.fEndErrorSize;
         endx = "m0," + lw + "v-" + 2*lw + "m0," + lw;
         endy = "m" + lw + ",0h-" + 2*lw + "m" + lw + ",0";
         dend = Math.floor((this.lineatt.width-1)/2);
      }

      let draw_markers = show_errors || show_markers;

      if (draw_markers || show_text || show_line) use_minmax = true;

      let draw_bin = besti => {
         bincont = histo.getBinContent(besti+1);
         if (!exclude_zero || (bincont!==0)) {
            mx1 = Math.round(funcs.grx(xaxis.GetBinCoord(besti)));
            mx2 = Math.round(funcs.grx(xaxis.GetBinCoord(besti+di)));
            midx = Math.round((mx1+mx2)/2);
            my = Math.round(funcs.gry(bincont));
            yerr1 = yerr2 = 20;
            if (show_errors) {
               binerr = histo.getBinError(besti+1);
               yerr1 = Math.round(my - funcs.gry(bincont + binerr)); // up
               yerr2 = Math.round(funcs.gry(bincont - binerr) - my); // down
            }

            if (show_text && (bincont !== 0)) {
               let lbl = (bincont === Math.round(bincont)) ? bincont.toString() : floatToString(bincont, gStyle.fPaintTextFormat);

               if (text_font.angle)
                  this.drawText({ align: 12, x: midx, y: Math.round(my - 2 - text_font.size / 5), text: lbl, latex: 0 });
               else
                  this.drawText({ x: Math.round(mx1 + (mx2 - mx1) * 0.1), y: Math.round(my - 2 - text_font.size), width: Math.round((mx2 - mx1) * 0.8), height: text_font.size, text: lbl, latex: 0 });
            }

            if (show_line && (path_line !== null))
               path_line += ((path_line.length === 0) ? "M" : "L") + midx + "," + my;

            if (draw_markers) {
               if ((my >= -yerr1) && (my <= height + yerr2)) {
                  if (path_fill !== null)
                     path_fill += "M" + mx1 +","+(my-yerr1) +
                                  "h" + (mx2-mx1) + "v" + (yerr1+yerr2+1) + "h-" + (mx2-mx1) + "z";
                  if (path_marker !== null)
                     path_marker += this.markeratt.create(midx, my);
                  if (path_err !== null) {
                     let edx = 5;
                     if (this.options.errorX > 0) {
                        edx = Math.round((mx2-mx1)*this.options.errorX);
                        let mmx1 = midx - edx, mmx2 = midx + edx;
                        path_err += "M" + (mmx1+dend) +","+ my + endx + "h" + (mmx2-mmx1-2*dend) + endx;
                     }
                     path_err += "M" + midx +"," + (my-yerr1+dend) + endy + "v" + (yerr1+yerr2-2*dend) + endy;
                     if (hints_err !== null)
                        hints_err += "M" + (midx-edx) + "," + (my-yerr1) + "h" + (2*edx) + "v" + (yerr1+yerr2) + "h" + (-2*edx) + "z";
                  }
               }
            }
         }
      };

      for (i = left; i <= right; i += di) {

         x = xaxis.GetBinCoord(i);

         if (funcs.logx && (x <= 0)) continue;

         grx = Math.round(funcs.grx(x));

         lastbin = (i > right - di);

         if (lastbin && (left < right)) {
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
         } else
         if (use_minmax) {
            if ((grx === currx) && !lastbin) {
               if (gry < curry_min) bestimax = i; else
               if (gry > curry_max) bestimin = i;
               curry_min = Math.min(curry_min, gry);
               curry_max = Math.max(curry_max, gry);
               curry = gry;
            } else {

               if (draw_markers || show_text || show_line) {
                  if (bestimin === bestimax) { draw_bin(bestimin); } else
                     if (bestimin < bestimax) { draw_bin(bestimin); draw_bin(bestimax); } else {
                        draw_bin(bestimax); draw_bin(bestimin);
                     }
               }

               // when several points as same X differs, need complete logic
               if (!draw_markers && ((curry_min !== curry_max) || (prevy !== curry_min))) {

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
         } else
         if ((gry !== curry) || lastbin) {
            if (grx !== currx) res += "h"+(grx-currx);
            if (gry !== curry) res += "v"+(gry-curry);
            curry = gry;
            currx = grx;
         }
      }

      let close_path = "",
          fill_for_interactive = !isBatchMode() && this.fillatt.empty() && options.Hist && settings.Tooltip && !draw_markers && !show_line;
      if (!this.fillatt.empty() || fill_for_interactive) {
         let h0 = height + 3;
         if (fill_for_interactive) {
            let gry0 = Math.round(funcs.gry(0));
            if (gry0 <= 0)
               h0 = -3;
            else if (gry0 < height)
               h0 = gry0;
         }
         close_path = `L${currx},${h0}H${startx}Z`;
         if (res.length > 0) res += close_path;
      }

      if (draw_markers || show_line) {
         if ((path_fill !== null) && (path_fill.length > 0))
            this.draw_g.append("svg:path")
                       .attr("d", path_fill)
                       .call(this.fillatt.func);

         if ((path_err !== null) && (path_err.length > 0))
               this.draw_g.append("svg:path")
                   .attr("d", path_err)
                   .call(this.lineatt.func);

         if ((hints_err !== null) && (hints_err.length > 0))
               this.draw_g.append("svg:path")
                   .attr("d", hints_err)
                   .style("fill", "none")
                   .style("pointer-events", isBatchMode() ? null : "visibleFill");

         if ((path_line !== null) && (path_line.length > 0)) {
            if (!this.fillatt.empty())
               this.draw_g.append("svg:path")
                     .attr("d", options.Fill ? (path_line + close_path) : res)
                     .call(this.fillatt.func);

            this.draw_g.append("svg:path")
                   .attr("d", path_line)
                   .style("fill", "none")
                   .call(this.lineatt.func);
         }

         if ((path_marker !== null) && (path_marker.length > 0))
            this.draw_g.append("svg:path")
                .attr("d", path_marker)
                .call(this.markeratt.func);

      } else if (res && options.Hist) {
         this.draw_g.append("svg:path")
                    .attr("d", res)
                    .style("stroke-linejoin","miter")
                    .call(this.lineatt.func)
                    .call(this.fillatt.func);
      }

      return show_text ? this.finishTextDrawing() : Promise.resolve(true);
   }

   /** @summary Provide text information (tooltips) for histogram bin */
   getBinTooltips(bin) {
      let tips = [],
          name = this.getObjectHint(),
          pmain = this.getFramePainter(),
          histo = this.getHisto(),
          xaxis = this.getAxis("x"),
          di = this.isDisplayItem() ? histo.stepx : 1,
          x1 = xaxis.GetBinCoord(bin),
          x2 = xaxis.GetBinCoord(bin+di),
          cont = histo.getBinContent(bin+1),
          xlbl = this.getAxisBinTip("x", bin, di);

      if (name.length>0) tips.push(name);

      if (this.options.Error || this.options.Mark) {
         tips.push("x = " + xlbl);
         tips.push("y = " + pmain.axisAsText("y", cont));
         if (this.options.Error) {
            if (xlbl[0] == "[") tips.push("error x = " + ((x2 - x1) / 2).toPrecision(4));
            tips.push("error y = " + histo.getBinError(bin + 1).toPrecision(4));
         }
      } else {
         tips.push("bin = " + bin);
         tips.push("x = " + xlbl);
         if (histo['$baseh']) cont -= histo['$baseh'].getBinContent(bin+1);
         let lbl = "entries = " + (di > 1 ? "~" : "");
         if (cont === Math.round(cont))
            tips.push(lbl + cont);
         else
            tips.push(lbl + floatToString(cont, gStyle.fStatFormat));
      }

      return tips;
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      if (!pnt || !this.draw_content || this.options.Mode3D || !this.draw_g) {
         if (this.draw_g)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      let pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          width = pmain.getFrameWidth(),
          height = pmain.getFrameHeight(),
          histo = this.getHisto(), xaxis = this.getAxis("x"),
          findbin = null, show_rect,
          grx1, midx, grx2, gry1, midy, gry2, gapx = 2,
          left = this.getSelectIndex("x", "left", -1),
          right = this.getSelectIndex("x", "right", 2),
          l = left, r = right;

      function GetBinGrX(i) {
         let xx = xaxis.GetBinCoord(i);
         return (funcs.logx && (xx<=0)) ? null : funcs.grx(xx);
      }

      function GetBinGrY(i) {
         let yy = histo.getBinContent(i + 1);
         if (funcs.logy && (yy < funcs.scale_ymin))
            return funcs.swap_xy ? -1000 : 10*height;
         return Math.round(funcs.gry(yy));
      }

      let pnt_x = funcs.swap_xy ? pnt.y : pnt.x,
          pnt_y = funcs.swap_xy ? pnt.x : pnt.y;

      while (l < r-1) {
         let m = Math.round((l+r)*0.5),
             xx = GetBinGrX(m);
         if ((xx === null) || (xx < pnt_x - 0.5)) {
            if (funcs.swap_xy) r = m; else l = m;
         } else if (xx > pnt_x + 0.5) {
            if (funcs.swap_xy) l = m; else r = m;
         } else { l++; r--; }
      }

      findbin = r = l;
      grx1 = GetBinGrX(findbin);

      if (funcs.swap_xy) {
         while ((l>left) && (GetBinGrX(l-1) < grx1 + 2)) --l;
         while ((r<right) && (GetBinGrX(r+1) > grx1 - 2)) ++r;
      } else {
         while ((l>left) && (GetBinGrX(l-1) > grx1 - 2)) --l;
         while ((r<right) && (GetBinGrX(r+1) < grx1 + 2)) ++r;
      }

      if (l < r) {
         // many points can be assigned with the same cursor position
         // first try point around mouse y
         let best = height;
         for (let m=l;m<=r;m++) {
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
         grx1 += Math.round(this.options.BarOffset*w);
         grx2 = grx1 + Math.round(this.options.BarWidth*w);
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
            if ((pnt_y<gry1) || (pnt_y>gry2)) findbin = null;

      } else if (this.options.Error || this.options.Mark) {

         show_rect = true;

         let msize = 3;
         if (this.markeratt) msize = Math.max(msize, this.markeratt.getFullSize());

         if (this.options.Error) {
            let cont = histo.getBinContent(findbin+1),
                binerr = histo.getBinError(findbin+1);

            gry1 = Math.round(funcs.gry(cont + binerr)); // up
            gry2 = Math.round(funcs.gry(cont - binerr)); // down

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

      } else if (this.options.Line) {

         show_rect = false;

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

      if (findbin!==null) {
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

      let res = { name: "histo", title: histo.fTitle,
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
            ttrect.attr("x", pmain.swap_xy ? gry1 : grx1)
                  .attr("width", pmain.swap_xy ? gry2-gry1 : grx2-grx1)
                  .attr("y", pmain.swap_xy ? grx1 : gry1)
                  .attr("height", pmain.swap_xy ? grx2-grx1 : gry2-gry1)
                  .style("opacity", "0.3")
                  .property("current_bin", findbin);

         res.exact = (Math.abs(midy - pnt_y) <= 5) || ((pnt_y>=gry1) && (pnt_y<=gry2));

         res.menu = res.exact; // one could show context menu
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
         res.user_info = { obj: histo,  name: "histo",
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

         this.decodeOptions(arg); // obsolete, should be implemented differently

         if (this.options.need_fillcol && this.fillatt && this.fillatt.empty())
            this.fillatt.change(5,1001);

         // redraw all objects
         this.interactiveRedraw("pad", "drawopt");
      });
   }

   /** @summary Perform automatic zoom inside non-zero region of histogram */
   autoZoom() {
      let left = this.getSelectIndex("x", "left", -1),
          right = this.getSelectIndex("x", "right", 1),
          dist = right - left, histo = this.getHisto(), xaxis = this.getAxis("x");

      if (dist == 0) return;

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
         return this.getFramePainter().zoom(xaxis.GetBinCoord(left), xaxis.GetBinCoord(right));
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis,min,max) {
      let xaxis = this.getAxis("x");

      if ((axis=="x") && (xaxis.FindBin(max,0.5) - xaxis.FindBin(min,0) > 1)) return true;

      if ((axis=="y") && (Math.abs(max-min) > Math.abs(this.ymax-this.ymin)*1e-6)) return true;

      // check if it makes sense to zoom inside specified axis range
      return false;
   }

   /** @summary Call appropriate draw function */
   callDrawFunc(reason) {
      let main = this.getFramePainter();

      if (main && (main.mode3d !== this.options.Mode3D) && !this.isMainPainter())
         this.options.Mode3D = main.mode3d;

      return this.options.Mode3D ? this.draw3D(reason) : this.draw2D(reason);
   }

   /** @summary Draw in 2d */
   draw2D(reason) {
      this.clear3DScene();

      return this.drawFrameAxes().then(res => {
         return res ? this.drawingBins(reason) : false;
      }).then(res => {
         if (res)
            return this.draw1DBins().then(() => this.addInteractivity());
      }).then(() => this);
   }

   /** @summary Draw in 3d */
   draw3D(reason) {
      console.log('3D drawing is disabled, load ./hist/RH1Painter.mjs');
      return this.draw2D(reason);
   }

   /** @summary Readraw histogram */
   redraw(reason) {
      return this.callDrawFunc(reason);
   }

   static _draw(painter, opt) {
      return ensureRCanvas(painter).then(() => {

         painter.setAsMainPainter();

         painter.options = { Hist: false, Bar: false, BarStyle: 0,
                             Error: false, ErrorKind: -1, errorX: gStyle.fErrorX,
                             Zero: false, Mark: false,
                             Line: false, Fill: false, Lego: 0, Surf: 0,
                             Text: false, TextAngle: 0, TextKind: "", AutoColor: 0,
                             BarOffset: 0., BarWidth: 1., BaseLine: false, Mode3D: false };

         let d = new DrawOptions(opt);
         if (d.check('R3D_', true))
            painter.options.Render3D = constants.Render3D.fromString(d.part.toLowerCase());

         let kind = painter.v7EvalAttr("kind", "hist"),
             sub = painter.v7EvalAttr("sub", 0),
             has_main = !!painter.getMainPainter(),
             o = painter.options;

         o.Text = painter.v7EvalAttr("drawtext", false);
         o.BarOffset = painter.v7EvalAttr("baroffset", 0.);
         o.BarWidth = painter.v7EvalAttr("barwidth", 1.);
         o.second_x = has_main && painter.v7EvalAttr("secondx", false);
         o.second_y = has_main && painter.v7EvalAttr("secondy", false);

         switch(kind) {
            case "bar": o.Bar = true; o.BarStyle = sub; break;
            case "err": o.Error = true; o.ErrorKind = sub; break;
            case "p": o.Mark = true; break;
            case "l": o.Line = true; break;
            case "lego": o.Lego = sub > 0 ? 10+sub : 12; o.Mode3D = true; break;
            default: o.Hist = true;
         }

         painter.scanContent();

         return painter.callDrawFunc();
      });
   }

   /** @summary draw RH1 object */
   static draw(dom, histo, opt) {
      return RH1Painter._draw(new RH1Painter(dom, histo), opt);
   }

} // class RH1Painter

export { RH1Painter };
