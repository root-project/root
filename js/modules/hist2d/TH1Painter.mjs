import { gStyle, settings, clTF1, kNoZoom, kInspect, isFunc } from '../core.mjs';
import { rgb as d3_rgb } from '../d3.mjs';
import { floatToString, buildSvgCurve, addHighlightStyle } from '../base/BasePainter.mjs';
import { THistPainter } from './THistPainter.mjs';
import { getTF1Value } from '../base/func.mjs';


const PadDrawOptions = ['LOGXY', 'LOGX', 'LOGY', 'LOGZ', 'LOGV', 'LOG', 'LOG2X', 'LOG2Y', 'LOG2',
                        'LNX', 'LNY', 'LN', 'GRIDXY', 'GRIDX', 'GRIDY', 'TICKXY', 'TICKX', 'TICKY', 'TICKZ', 'FB', 'GRAYSCALE'];

/**
 * @summary Painter for TH1 classes
 * @private
 */

class TH1Painter extends THistPainter {

   /** @summary Convert TH1K into normal binned histogram */
   convertTH1K() {
      const histo = this.getObject();
      if (histo.fReady) return;

      const arr = histo.fArray, entries = histo.fEntries; // array of values
      histo.fNcells = histo.fXaxis.fNbins + 2;
      histo.fArray = new Float64Array(histo.fNcells).fill(0);
      for (let n = 0; n < histo.fNIn; ++n)
         histo.Fill(arr[n]);
      histo.fReady = 1;
      histo.fEntries = entries;
   }

   /** @summary Scan content of 1-D histogram
     * @desc Detect min/max values for x and y axis
     * @param {boolean} when_axis_changed - true when zooming was changed, some checks may be skipped */
   scanContent(when_axis_changed) {
      if (when_axis_changed && !this.nbinsx)
         when_axis_changed = false;

      if (this.isTH1K())
         this.convertTH1K();

      const histo = this.getHisto();

      if (!when_axis_changed)
         this.extractAxesProperties(1);

      const left = this.getSelectIndex('x', 'left'),
            right = this.getSelectIndex('x', 'right'),
            pad_logy = this.getPadPainter()?.getPadLog(this.options.BarStyle >= 20 ? 'x' : 'y'),
            f1 = this.options.Func ? this.findFunction(clTF1) : null;

      if (when_axis_changed && (left === this.scan_xleft) && (right === this.scan_xright))
         return;

      // Paint histogram axis only
      this.draw_content = !(this.options.Axis > 0);

      this.scan_xleft = left;
      this.scan_xright = right;

      const profile = this.isTProfile();
      let hmin = 0, hmin_nz = 0, hmax = 0, hsum = 0, first = true, value, err;

      for (let i = 0; i < this.nbinsx; ++i) {
         value = histo.getBinContent(i + 1);
         hsum += profile ? histo.fBinEntries[i + 1] : value;

         if ((i < left) || (i >= right))
            continue;

         if ((value > 0) && ((hmin_nz === 0) || (value < hmin_nz)))
            hmin_nz = value;

         if (first) {
            hmin = hmax = value;
            first = false;
         }

         err = this.options.Error ? histo.getBinError(i + 1) : 0;

         hmin = Math.min(hmin, value - err);
         hmax = Math.max(hmax, value + err);

         if (f1) {
            // similar code as in THistPainter, line 7196
            const x = histo.fXaxis.GetBinCenter(i + 1),
                  v = getTF1Value(f1, x);
            if (v !== undefined) {
               hmax = Math.max(hmax, v);
               if (pad_logy && (value > 0) && (v > 0.3 * value))
                  hmin_nz = Math.min(hmin_nz, v);
            }
         }
      }

      // account overflow/underflow bins
      if (profile)
         hsum += histo.fBinEntries[0] + histo.fBinEntries[this.nbinsx + 1];
      else
         hsum += histo.getBinContent(0) + histo.getBinContent(this.nbinsx + 1);

      this.stat_entries = (histo.fEntries > 1) ? histo.fEntries : hsum;

      this.hmin = hmin;
      this.hmax = hmax;

      // this.ymin_nz = hmin_nz; // value can be used to show optimal log scale

      if ((this.nbinsx === 0) || ((Math.abs(hmin) < 1e-300) && (Math.abs(hmax) < 1e-300)))
         this.draw_content = false;

      let set_zoom = false;

      if (this.draw_content || (this.isMainPainter() && (this.options.Axis > 0) && !this.options.ohmin && !this.options.ohmax && (histo.fMinimum === kNoZoom) && (histo.fMaximum === kNoZoom))) {
         if (hmin >= hmax) {
            if (hmin === 0) {
               this.ymin = 0; this.ymax = 1;
            } else if (hmin < 0) {
               this.ymin = 2 * hmin; this.ymax = 0;
            } else {
               this.ymin = 0; this.ymax = hmin * 2;
            }
         } else {
            if (pad_logy) {
               this.ymin = (hmin_nz || hmin) * 0.5;
               this.ymax = hmax*2*(0.9/0.95);
            } else {
               this.ymin = hmin;
               this.ymax = hmax;
            }
         }
      }

      // final adjustment like in THistPainter.cxx line 7309
      if (!this._exact_y_range && !this._set_y_range && !pad_logy) {
         if ((this.options.BaseLine !== false) && (this.ymin >= 0))
            this.ymin = 0;
         else {
            const positive = (this.ymin >= 0);
            this.ymin -= gStyle.fHistTopMargin*(this.ymax - this.ymin);
            if (positive && (this.ymin < 0))
               this.ymin = 0;
         }
         this.ymax += gStyle.fHistTopMargin*(this.ymax - this.ymin);
      }

      if (this.options.ignore_min_max)
         hmin = hmax = kNoZoom;
      else {
         hmin = this.options.minimum;
         hmax = this.options.maximum;
      }

      if ((hmin === hmax) && (hmin !== kNoZoom)) {
         if (hmin < 0) {
            hmin *= 2; hmax = 0;
         } else {
            hmin = 0; hmax *= 2;
            if (!hmax) hmax = 1;
         }
      }

      this._set_y_range = false;

      if (this.options.ohmin && this.options.ohmax && !this.draw_content) {
         // case of hstack drawing - histogram range used for zooming, but only for stack
         set_zoom = !this.options.ignore_min_max;
      } else if ((hmin !== kNoZoom) && (hmax !== kNoZoom) && !this.draw_content &&
          ((this.ymin === this.ymax) || (this.ymin > hmin) || (this.ymax < hmax))) {
         this.ymin = hmin;
         this.ymax = hmax;
         this._set_y_range = true;
      } else {
         if (hmin !== kNoZoom) {
            this._set_y_range = true;
            if (hmin < this.ymin)
               this.ymin = hmin;
             set_zoom = true;
         }
         if (hmax !== kNoZoom) {
            this._set_y_range = true;
            if (hmax > this.ymax)
               this.ymax = hmax;
            set_zoom = true;
         }
      }

      // always set zoom when hmin/hmax is configured
      // fMinimum/fMaximum values is a way how ROOT handles Y scale zooming for TH1

      if (!when_axis_changed) {
         if (set_zoom) {
            this.zoom_ymin = (hmin === kNoZoom) ? this.ymin : hmin;
            this.zoom_ymax = (hmax === kNoZoom) ? this.ymax : hmax;
         } else {
            delete this.zoom_ymin;
            delete this.zoom_ymax;
         }
      }

      // used in FramePainter.isAllowedDefaultYZooming
      this.wheel_zoomy = (this.getDimension() > 1) || !this.draw_content;
   }

   /** @summary Count histogram statistic */
   countStat(cond, count_skew) {
      const profile = this.isTProfile(),
            histo = this.getHisto(), xaxis = histo.fXaxis,
            left = this.getSelectIndex('x', 'left'),
            right = this.getSelectIndex('x', 'right'),
            fp = this.getFramePainter(),
            res = { name: histo.fName, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, integral: 0,
                    entries: this.stat_entries, eff_entries: 0, xmax: 0, wmax: 0, skewx: 0, skewd: 0, kurtx: 0, kurtd: 0 },
            has_counted_stat = !fp.isAxisZoomed('x') && (Math.abs(histo.fTsumw) > 1e-300);
      let stat_sumw = 0, stat_sumw2 = 0, stat_sumwx = 0, stat_sumwx2 = 0, stat_sumwy = 0, stat_sumwy2 = 0,
          i, xx = 0, w = 0, xmax = null, wmax = null;

      if (!isFunc(cond)) cond = null;

      for (i = left; i < right; ++i) {
         xx = xaxis.GetBinCoord(i + 0.5);

         if (cond && !cond(xx)) continue;

         if (profile) {
            w = histo.fBinEntries[i + 1];
            stat_sumwy += histo.fArray[i + 1];
            stat_sumwy2 += histo.fSumw2[i + 1];
         } else
            w = histo.getBinContent(i + 1);


         if ((xmax === null) || (w > wmax)) {
            xmax = xx;
            wmax = w;
         }

         if (!has_counted_stat) {
            stat_sumw += w;
            stat_sumw2 += w * w;
            stat_sumwx += w * xx;
            stat_sumwx2 += w * xx**2;
         }
      }

      // when no range selection done, use original statistic from histogram
      if (has_counted_stat) {
         stat_sumw = histo.fTsumw;
         stat_sumw2 = histo.fTsumw2;
         stat_sumwx = histo.fTsumwx;
         stat_sumwx2 = histo.fTsumwx2;
      }

      res.integral = stat_sumw;

      res.eff_entries = stat_sumw2 ? stat_sumw*stat_sumw/stat_sumw2 : Math.abs(stat_sumw);

      if (Math.abs(stat_sumw) > 1e-300) {
         res.meanx = stat_sumwx / stat_sumw;
         res.meany = stat_sumwy / stat_sumw;
         res.rmsx = Math.sqrt(Math.abs(stat_sumwx2 / stat_sumw - res.meanx**2));
         res.rmsy = Math.sqrt(Math.abs(stat_sumwy2 / stat_sumw - res.meany**2));
      }

      if (xmax !== null) {
         res.xmax = xmax;
         res.wmax = wmax;
      }

      if (count_skew) {
         let sum3 = 0, sum4 = 0, np = 0;
         for (i = left; i < right; ++i) {
            xx = xaxis.GetBinCoord(i + 0.5);
            if (cond && !cond(xx)) continue;
            w = profile ? histo.fBinEntries[i + 1] : histo.getBinContent(i + 1);
            np += w;
            sum3 += w * Math.pow(xx - res.meanx, 3);
            sum4 += w * Math.pow(xx - res.meanx, 4);
         }

         const stddev3 = Math.pow(res.rmsx, 3), stddev4 = Math.pow(res.rmsx, 4);
         if (np * stddev3 !== 0)
            res.skewx = sum3 / (np * stddev3);
         res.skewd = res.eff_entries > 0 ? Math.sqrt(6/res.eff_entries) : 0;
         if (np * stddev4 !== 0)
            res.kurtx = sum4 / (np * stddev4) - 3;
         res.kurtd = res.eff_entries > 0 ? Math.sqrt(24/res.eff_entries) : 0;
      }

      return res;
   }

   /** @summary Fill stat box */
   fillStatistic(stat, dostat, dofit) {
      // no need to refill statistic if histogram is dummy
      if (this.isIgnoreStatsFill()) return false;

      if (dostat === 1) dostat = 1111;
      if (dofit === 1) dofit = 111;

      const histo = this.getHisto(),
            print_name = dostat % 10,
            print_entries = Math.floor(dostat / 10) % 10,
            print_mean = Math.floor(dostat / 100) % 10,
            print_rms = Math.floor(dostat / 1000) % 10,
            print_under = Math.floor(dostat / 10000) % 10,
            print_over = Math.floor(dostat / 100000) % 10,
            print_integral = Math.floor(dostat / 1000000) % 10,
            print_skew = Math.floor(dostat / 10000000) % 10,
            print_kurt = Math.floor(dostat / 100000000) % 10,
            data = this.countStat(undefined, (print_skew > 0) || (print_kurt > 0));


      // make empty at the beginning
      stat.clearPave();

      if (print_name > 0)
         stat.addText(data.name);

      if (this.isTProfile()) {
         if (print_entries > 0)
            stat.addText('Entries = ' + stat.format(data.entries, 'entries'));

         if (print_mean > 0) {
            stat.addText('Mean = ' + stat.format(data.meanx));
            stat.addText('Mean y = ' + stat.format(data.meany));
         }

         if (print_rms > 0) {
            stat.addText('Std Dev = ' + stat.format(data.rmsx));
            stat.addText('Std Dev y = ' + stat.format(data.rmsy));
         }
      } else {
         if (print_entries > 0)
            stat.addText('Entries = ' + stat.format(data.entries, 'entries'));

         if (print_mean > 0)
            stat.addText('Mean = ' + stat.format(data.meanx));

         if (print_rms > 0)
            stat.addText('Std Dev = ' + stat.format(data.rmsx));

         if (print_under > 0)
            stat.addText('Underflow = ' + stat.format((histo.fArray.length > 0) ? histo.fArray[0] : 0, 'entries'));

         if (print_over > 0)
            stat.addText('Overflow = ' + stat.format((histo.fArray.length > 0) ? histo.fArray[histo.fArray.length - 1] : 0, 'entries'));

         if (print_integral > 0)
            stat.addText('Integral = ' + stat.format(data.integral, 'entries'));

         if (print_skew === 2)
            stat.addText(`Skewness = ${stat.format(data.skewx)} #pm ${stat.format(data.skewd)}`);
         else if (print_skew > 0)
            stat.addText(`Skewness = ${stat.format(data.skewx)}`);

         if (print_kurt === 2)
            stat.addText(`Kurtosis = ${stat.format(data.kurtx)} #pm ${stat.format(data.kurtd)}`);
         else if (print_kurt > 0)
            stat.addText(`Kurtosis = ${stat.format(data.kurtx)}`);
      }

      if (dofit) stat.fillFunctionStat(this.findFunction(clTF1), dofit, 1);

      return true;
   }

   /** @summary Draw histogram as bars */
   async drawBars(funcs, height) {
      const left = this.getSelectIndex('x', 'left', -1),
            right = this.getSelectIndex('x', 'right', 1),
            histo = this.getHisto(), xaxis = histo.fXaxis,
            show_text = this.options.Text;
      let text_col, text_angle, text_size,
          i, x1, x2, grx1, grx2, y, gry1, gry2, w,
          bars = '', barsl = '', barsr = '',
          side = (this.options.BarStyle > 10) ? this.options.BarStyle % 10 : 0;

      if (side > 4) side = 4;
      gry2 = funcs.swap_xy ? 0 : height;
      if (Number.isFinite(this.options.BaseLine)) {
         if (this.options.BaseLine >= funcs.scale_ymin)
            gry2 = Math.round(funcs.gry(this.options.BaseLine));
       }

      if (show_text) {
         text_col = this.getColor(histo.fMarkerColor);
         text_angle = -1*this.options.TextAngle;
         text_size = 20;

         if ((histo.fMarkerSize !== 1) && text_angle)
            text_size = 0.02*height*histo.fMarkerSize;

         this.startTextDrawing(42, text_size, this.draw_g, text_size);
      }

      for (i = left; i < right; ++i) {
         x1 = xaxis.GetBinLowEdge(i+1);
         x2 = xaxis.GetBinLowEdge(i+2);

         if (funcs.logx && (x2 <= 0)) continue;

         grx1 = Math.round(funcs.grx(x1));
         grx2 = Math.round(funcs.grx(x2));

         y = histo.getBinContent(i+1);
         if (funcs.logy && (y < funcs.scale_ymin)) continue;
         gry1 = Math.round(funcs.gry(y));

         w = grx2 - grx1;
         grx1 += Math.round(histo.fBarOffset/1000*w);
         w = Math.round(histo.fBarWidth/1000*w);

         if (funcs.swap_xy)
            bars += `M${gry2},${grx1}h${gry1-gry2}v${w}h${gry2-gry1}z`;
         else
            bars += `M${grx1},${gry1}h${w}v${gry2-gry1}h${-w}z`;

         if (side > 0) {
            grx2 = grx1 + w;
            w = Math.round(w * side / 10);
            if (funcs.swap_xy) {
               barsl += `M${gry2},${grx1}h${gry1-gry2}v${w}h${gry2-gry1}z`;
               barsr += `M${gry2},${grx2}h${gry1-gry2}v${-w}h${gry2-gry1}z`;
            } else {
               barsl += `M${grx1},${gry1}h${w}v${gry2-gry1}h${-w}z`;
               barsr += `M${grx2},${gry1}h${-w}v${gry2-gry1}h${w}z`;
            }
         }

         if (show_text && y) {
            const text = (y === Math.round(y)) ? y.toString() : floatToString(y, gStyle.fPaintTextFormat);

            if (funcs.swap_xy)
               this.drawText({ align: 12, x: Math.round(gry1 + text_size/2), y: Math.round(grx1+0.1), height: Math.round(w*0.8), text, color: text_col, latex: 0 });
            else if (text_angle)
               this.drawText({ align: 12, x: grx1+w/2, y: Math.round(gry1 - 2 - text_size/5), width: 0, height: 0, rotate: text_angle, text, color: text_col, latex: 0 });
            else
               this.drawText({ align: 22, x: Math.round(grx1 + w*0.1), y: Math.round(gry1 - 2 - text_size), width: Math.round(w*0.8), height: text_size, text, color: text_col, latex: 0 });
         }
      }

      if (bars) {
         this.draw_g.append('svg:path')
                    .attr('d', bars)
                    .call(this.fillatt.func);
      }

      if (barsl) {
         this.draw_g.append('svg:path')
             .attr('d', barsl)
             .call(this.fillatt.func)
             .style('fill', d3_rgb(this.fillatt.color).brighter(0.5).formatHex());
      }

      if (barsr) {
         this.draw_g.append('svg:path')
               .attr('d', barsr)
               .call(this.fillatt.func)
               .style('fill', d3_rgb(this.fillatt.color).darker(0.5).formatHex());
      }

      if (show_text)
         return this.finishTextDrawing();
   }

   /** @summary Draw histogram as filled errors */
   drawFilledErrors(funcs) {
      const left = this.getSelectIndex('x', 'left', 0),
            right = this.getSelectIndex('x', 'right', 0),
            histo = this.getHisto(), bins1 = [], bins2 = [];

      for (let i = left; i < right; ++i) {
         const x = histo.fXaxis.GetBinCoord(i+0.5);
         if (funcs.logx && (x <= 0)) continue;
         const grx = Math.round(funcs.grx(x)),
               y = histo.getBinContent(i+1),
               yerr = histo.getBinError(i+1);
         if (funcs.logy && (y-yerr < funcs.scale_ymin)) continue;

         bins1.push({ grx, gry: Math.round(funcs.gry(y + yerr)) });
         bins2.unshift({ grx, gry: Math.round(funcs.gry(y - yerr)) });
      }

      const line = this.options.ErrorKind !== 4,
            path1 = buildSvgCurve(bins1, { line }),
            path2 = buildSvgCurve(bins2, { line, cmd: 'L' });

      this.draw_g.append('svg:path')
                 .attr('d', path1 + path2 + 'Z')
                 .call(this.fillatt.func);
   }

   /** @summary Draw TH1 as hist/line/curve
     * @return Promise or scalar value */
   drawNormal(funcs, width, height) {
      const left = this.getSelectIndex('x', 'left', -1),
            right = this.getSelectIndex('x', 'right', 2),
            histo = this.getHisto(),
            want_tooltip = !this.isBatchMode() && settings.Tooltip,
            xaxis = histo.fXaxis,
            exclude_zero = !this.options.Zero,
            show_errors = this.options.Error,
            show_curve = this.options.Curve,
            show_text = this.options.Text,
            text_profile = show_text && (this.options.TextKind === 'E') && this.isTProfile() && histo.fBinEntries,
            grpnts = [];
      let res = '', lastbin = false,
          show_markers = this.options.Mark,
          show_line = this.options.Line,
          startx, startmidx, currx, curry, x, grx, y, gry, curry_min, curry_max, prevy, prevx, i, bestimin, bestimax,
          path_fill = null, path_err = null, path_marker = null, path_line = '',
          hints_err = null, hints_marker = null, hsz = 5,
          do_marker = false, do_err = false,
          dend = 0, dlw = 0, my, yerr1, yerr2, bincont, binerr, mx1, mx2, midx, lx, ly, mmx1, mmx2,
          text_col, text_angle, text_size;

      if (show_errors && !show_markers && (histo.fMarkerStyle > 1))
         show_markers = true;

      if (this.options.ErrorKind === 2) {
         if (this.fillatt.empty()) show_markers = true;
                              else path_fill = '';
      } else if (show_errors) {
         show_line = false;
         path_err = '';
         hints_err = want_tooltip ? '' : null;
         do_err = true;
      }

      dlw = this.lineatt.width + gStyle.fEndErrorSize;
      if (this.options.ErrorKind === 1)
         dend = Math.floor((this.lineatt.width-1)/2);

      if (show_markers) {
         // draw markers also when e2 option was specified
         let style = this.options.MarkStyle;
         if (!style && (histo.fMarkerStyle === 1)) style = 8; // as in recent ROOT changes
         this.createAttMarker({ attr: histo, style }); // when style not configured, it will be ignored
         if (this.markeratt.size > 0) {
            // simply use relative move from point, can optimize in the future
            path_marker = '';
            do_marker = true;
            this.markeratt.resetPos();
            if ((hints_err === null) && want_tooltip && (!this.markeratt.fill || (this.markeratt.getFullSize() < 7))) {
               hints_marker = '';
               hsz = Math.max(5, Math.round(this.markeratt.getFullSize()*0.7));
             }
         } else
            show_markers = false;
      }

      const draw_markers = show_errors || show_markers,
            draw_any_but_hist = draw_markers || show_text || show_line || show_curve,
            draw_hist = this.options.Hist && (!this.lineatt.empty() || !this.fillatt.empty());

      if (!draw_hist && !draw_any_but_hist)
         return this.removeG();

      if (show_text) {
         text_col = this.getColor(histo.fMarkerColor);
         text_angle = -1*this.options.TextAngle;
         text_size = 20;

         if ((histo.fMarkerSize !== 1) && text_angle)
            text_size = 0.02*height*histo.fMarkerSize;

         if (!text_angle && !this.options.TextKind) {
             const space = width / (right - left + 1);
             if (space < 3 * text_size) {
                text_angle = 270;
                text_size = Math.round(space*0.7);
             }
         }

         this.startTextDrawing(42, text_size, this.draw_g, text_size);
      }

      // if there are too many points, exclude many vertical drawings at the same X position
      // instead define min and max value and made min-max drawing
      const use_minmax = draw_any_but_hist || ((right - left) > 3*width),

      // just to get correct values for the specified bin
       extract_bin = bin => {
         bincont = histo.getBinContent(bin+1);
         if (exclude_zero && (bincont === 0)) return false;
         mx1 = Math.round(funcs.grx(xaxis.GetBinLowEdge(bin+1)));
         mx2 = Math.round(funcs.grx(xaxis.GetBinLowEdge(bin+2)));
         midx = Math.round((mx1 + mx2) / 2);
         if (startmidx === undefined) startmidx = midx;
         my = Math.round(funcs.gry(bincont));
         if (show_errors) {
            binerr = histo.getBinError(bin+1);
            yerr1 = Math.round(my - funcs.gry(bincont + binerr)); // up
            yerr2 = Math.round(funcs.gry(bincont - binerr) - my); // down
         } else
            yerr1 = yerr2 = 20;

         return true;
      }, draw_errbin = () => {
         let edx = 5;
         if (this.options.errorX > 0) {
            edx = Math.round((mx2 - mx1) * this.options.errorX);
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
         if (hints_err !== null) {
            const he1 = Math.max(yerr1, 5), he2 = Math.max(yerr2, 5);
            hints_err += `M${midx-edx},${my-he1}h${2*edx}v${he1+he2}h${-2*edx}z`;
         }
      }, draw_bin = bin => {
         if (extract_bin(bin)) {
            if (show_text) {
               const cont = text_profile ? histo.fBinEntries[bin+1] : bincont;

               if (cont !== 0) {
                  const lbl = (cont === Math.round(cont)) ? cont.toString() : floatToString(cont, gStyle.fPaintTextFormat);

                  if (text_angle)
                     this.drawText({ align: 12, x: midx, y: Math.round(my - 2 - text_size/5), width: 0, height: 0, rotate: text_angle, text: lbl, color: text_col, latex: 0 });
                  else
                     this.drawText({ align: 22, x: Math.round(mx1 + (mx2-mx1)*0.1), y: Math.round(my-2-text_size), width: Math.round((mx2-mx1)*0.8), height: text_size, text: lbl, color: text_col, latex: 0 });
               }
            }

            if (show_line) {
               if (path_line.length === 0)
                  path_line = `M${midx},${my}`;
               else if (lx === midx)
                  path_line += `v${my-ly}`;
               else if (ly === my)
                  path_line += `h${midx-lx}`;
               else
                  path_line += `l${midx-lx},${my-ly}`;
               lx = midx; ly = my;
            } else if (show_curve)
               grpnts.push({ grx: (mx1 + mx2) / 2, gry: funcs.gry(bincont) });

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
      if (do_marker || do_err) {
         if (!settings.OptimizeDraw || ((right-left < 50000) && (settings.OptimizeDraw === 1))) {
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
      }


      for (i = left; i <= right; ++i) {
         x = xaxis.GetBinLowEdge(i+1);

         if (this.logx && (x <= 0)) continue;

         grx = Math.round(funcs.grx(x));

         lastbin = (i === right);

         if (lastbin && (left<right))
            gry = curry;
          else {
            y = histo.getBinContent(i+1);
            gry = Math.round(funcs.gry(y));
         }

         if (res.length === 0) {
            bestimin = bestimax = i;
            prevx = startx = currx = grx;
            prevy = curry_min = curry_max = curry = gry;
            res = `M${currx},${curry}`;
         } else if (use_minmax) {
            if ((grx === currx) && !lastbin) {
               if (gry < curry_min)
                  bestimax = i;
               else if (gry > curry_max)
                  bestimin = i;

               curry_min = Math.min(curry_min, gry);
               curry_max = Math.max(curry_max, gry);
               curry = gry;
            } else {
               if (draw_any_but_hist) {
                  if (bestimin === bestimax)
                     draw_bin(bestimin);
                  else if (bestimin < bestimax) {
                     draw_bin(bestimin); draw_bin(bestimax);
                  } else {
                     draw_bin(bestimax); draw_bin(bestimin);
                  }
               }

               // when several points at same X differs, need complete logic
               if (draw_hist && ((curry_min !== curry_max) || (prevy !== curry_min))) {
                  if (prevx !== currx)
                     res += 'h'+(currx-prevx);

                  if (curry === curry_min) {
                     if (curry_max !== prevy)
                        res += 'v' + (curry_max - prevy);
                     if (curry_min !== curry_max)
                        res += 'v' + (curry_min - curry_max);
                  } else {
                     if (curry_min !== prevy)
                        res += 'v' + (curry_min - prevy);
                     if (curry_max !== curry_min)
                        res += 'v' + (curry_max - curry_min);
                     if (curry !== curry_max)
                       res += 'v' + (curry - curry_max);
                  }

                  prevx = currx;
                  prevy = curry;
               }

               if (lastbin && (prevx !== grx))
                  res += 'h' + (grx-prevx);

               bestimin = bestimax = i;
               curry_min = curry_max = curry = gry;
               currx = grx;
            }
            // end of use_minmax
         } else if ((gry !== curry) || lastbin) {
            if (grx !== currx) res += `h${grx-currx}`;
            if (gry !== curry) res += `v${gry-curry}`;
            curry = gry;
            currx = grx;
         }
      }

      const fill_for_interactive = want_tooltip && this.fillatt.empty() && draw_hist && !draw_markers && !show_line && !show_curve,
      add_hist = () => {
         this.draw_g.append('svg:path')
                    .attr('d', res + ((!this.fillatt.empty() || fill_for_interactive) ? close_path : ''))
                    .style('stroke-linejoin', 'miter')
                    .call(this.lineatt.func)
                    .call(this.fillatt.func);
      };
      let h0 = height + 3;
      if (!fill_for_interactive) {
         const gry0 = Math.round(funcs.gry(0));
         if (gry0 <= 0)
            h0 = -3;
         else if (gry0 < height)
            h0 = gry0;
      }
      const close_path = `L${currx},${h0}H${startx}Z`;

      if (res && draw_hist && !this.fillatt.empty()) {
         add_hist();
         res = '';
      }

      if (draw_markers || show_line || show_curve) {
         if (!path_line && grpnts.length)
            path_line = buildSvgCurve(grpnts);

         if (path_fill) {
            this.draw_g.append('svg:path')
                       .attr('d', path_fill)
                       .call(this.fillatt.func);
         } else if (path_line && !this.fillatt.empty() && !draw_hist) {
            this.draw_g.append('svg:path')
                .attr('d', path_line + `L${midx},${h0}H${startmidx}Z`)
                .call(this.fillatt.func);
         }

         if (path_err) {
             this.draw_g.append('svg:path')
                   .attr('d', path_err)
                   .call(this.lineatt.func);
         }

         if (hints_err) {
               this.draw_g.append('svg:path')
                   .attr('d', hints_err)
                   .style('fill', 'none')
                   .style('pointer-events', this.isBatchMode() ? null : 'visibleFill');
         }

         if (path_line) {
            this.draw_g.append('svg:path')
                   .attr('d', path_line)
                   .style('fill', 'none')
                   .call(this.lineatt.func);
         }

         if (path_marker) {
            this.draw_g.append('svg:path')
                .attr('d', path_marker)
                .call(this.markeratt.func);
         }

         if (hints_marker) {
            this.draw_g.append('svg:path')
                .attr('d', hints_marker)
                .style('fill', 'none')
                .style('pointer-events', this.isBatchMode() ? null : 'visibleFill');
         }
      }

      if (res && draw_hist)
         add_hist();

      if (show_text)
         return this.finishTextDrawing();
   }

   /** @summary Draw TH1 bins in SVG element
     * @return Promise or scalar value */
   draw1DBins() {
      this.createHistDrawAttributes();

      const pmain = this.getFramePainter(),
          funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
          width = pmain.getFrameWidth(), height = pmain.getFrameHeight();

      if (!this.draw_content || (width <= 0) || (height <= 0))
          return this.removeG();

      this.createG(true);

      if (this.options.Bar) {
         return this.drawBars(funcs, height).then(() => {
            if (this.options.ErrorKind === 1)
               return this.drawNormal(funcs, width, height);
         });
      }

      if ((this.options.ErrorKind === 3) || (this.options.ErrorKind === 4))
         return this.drawFilledErrors(funcs);

      return this.drawNormal(funcs, width, height);
   }

   /** @summary Provide text information (tooltips) for histogram bin */
   getBinTooltips(bin) {
      const tips = [],
            name = this.getObjectHint(),
            pmain = this.getFramePainter(),
            funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
            histo = this.getHisto(),
            x1 = histo.fXaxis.GetBinLowEdge(bin+1),
            x2 = histo.fXaxis.GetBinLowEdge(bin+2),
            xlbl = this.getAxisBinTip('x', histo.fXaxis, bin);
      let cont = histo.getBinContent(bin+1);

      if (name) tips.push(name);

      if (this.options.Error || this.options.Mark || this.isTF1()) {
         tips.push(`x = ${xlbl}`, `y = ${funcs.axisAsText('y', cont)}`);
         if (this.options.Error) {
            if (xlbl[0] === '[') tips.push(`error x = ${((x2 - x1) / 2).toPrecision(4)}`);
            tips.push(`error y = ${histo.getBinError(bin + 1).toPrecision(4)}`);
         }
      } else {
         tips.push(`bin = ${bin+1}`, `x = ${xlbl}`);
         if (histo.$baseh) cont -= histo.$baseh.getBinContent(bin+1);
         if (cont === Math.round(cont))
            tips.push(`entries = ${cont}`);
         else
            tips.push(`entries = ${floatToString(cont, gStyle.fStatFormat)}`);
      }

      return tips;
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      if (!pnt || !this.draw_content || !this.draw_g || this.options.Mode3D) {
         this.draw_g?.selectChild('.tooltip_bin').remove();
         return null;
      }

      const pmain = this.getFramePainter(),
            funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y),
            histo = this.getHisto(),
            left = this.getSelectIndex('x', 'left', -1),
            right = this.getSelectIndex('x', 'right', 2);
      let width = pmain.getFrameWidth(),
          height = pmain.getFrameHeight(),
          findbin = null, show_rect,
          grx1, grx2, gry1, gry2, gapx = 2,
          l = left, r = right, pnt_x = pnt.x, pnt_y = pnt.y;

      const GetBinGrX = i => {
         const xx = histo.fXaxis.GetBinLowEdge(i+1);
         return (funcs.logx && (xx <= 0)) ? null : funcs.grx(xx);
      }, GetBinGrY = i => {
         const yy = histo.getBinContent(i + 1);
         if (funcs.logy && (yy < funcs.scale_ymin))
            return funcs.swap_xy ? -1000 : 10*height;
         return Math.round(funcs.gry(yy));
      };

      if (funcs.swap_xy)
         [pnt_x, pnt_y, width, height] = [pnt_y, pnt_x, height, width];

      const descent_order = funcs.swap_xy !== pmain.x_handle.reverse;

      while (l < r-1) {
         const m = Math.round((l+r)*0.5), xx = GetBinGrX(m);
         if ((xx === null) || (xx < pnt_x - 0.5))
            if (descent_order) r = m; else l = m;
          else if (xx > pnt_x + 0.5)
            if (descent_order) l = m; else r = m;
          else { l++; r--; }
      }

      findbin = r = l;
      grx1 = GetBinGrX(findbin);

      if (descent_order) {
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
            const dist = Math.abs(GetBinGrY(m) - pnt_y);
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
         const w = grx2 - grx1;
         grx1 += Math.round(histo.fBarOffset / 1000 * w);
         grx2 = grx1 + Math.round(histo.fBarWidth / 1000 * w);
      }

      if (grx1 > grx2)
         [grx1, grx2] = [grx2, grx1];

      const midx = Math.round((grx1 + grx2) / 2),
         midy = gry1 = gry2 = GetBinGrY(findbin);

      if (this.options.Bar) {
         show_rect = true;

         gapx = 0;

         gry1 = Math.round(funcs.gry(((this.options.BaseLine !== false) && (this.options.BaseLine > funcs.scale_ymin)) ? this.options.BaseLine : funcs.scale_ymin));

         if (gry1 > gry2)
            [gry1, gry2] = [gry2, gry1];

         if (!pnt.touch && (pnt.nproc === 1))
            if ((pnt_y < gry1) || (pnt_y > gry2)) findbin = null;
      } else if ((this.options.Error && (this.options.Hist !== true)) || this.options.Mark || this.options.Line || this.options.Curve) {
         show_rect = !this.isTF1();

         let msize = 3;
         if (this.markeratt) msize = Math.max(msize, this.markeratt.getFullSize());

         if (this.options.Error) {
            const cont = histo.getBinContent(findbin+1),
                binerr = histo.getBinError(findbin+1);

            gry1 = Math.round(funcs.gry(cont + binerr)); // up
            gry2 = Math.round(funcs.gry(cont - binerr)); // down

            if ((cont === 0) && this.isTProfile()) findbin = null;

            const dx = (grx2-grx1)*this.options.errorX;
            grx1 = Math.round(midx - dx);
            grx2 = Math.round(midx + dx);
         }

         // show at least 6 pixels as tooltip rect
         if (grx2 - grx1 < 2*msize) { grx1 = midx-msize; grx2 = midx+msize; }

         gry1 = Math.min(gry1, midy - msize);
         gry2 = Math.max(gry2, midy + msize);

         if (!pnt.touch && (pnt.nproc === 1))
            if ((pnt_y < gry1) || (pnt_y > gry2)) findbin = null;
      } else {
         // if histogram alone, use old-style with rects
         // if there are too many points at pixel, use circle
         show_rect = (pnt.nproc === 1) && (right-left < width);

         if (show_rect) {
            gry2 = height;

            if (!this.fillatt.empty()) {
               gry2 = Math.min(height, Math.max(0, Math.round(funcs.gry(0))));
               if (gry2 < gry1)
                 [gry1, gry2] = [gry2, gry1];
            }

            // for mouse events pointer should be between y1 and y2
            if (((pnt.y < gry1) || (pnt.y > gry2)) && !pnt.touch) findbin = null;
         }
      }

      if (findbin !== null) {
         // if bin on boundary found, check that x position is ok
         if ((findbin === left) && (grx1 > pnt_x + gapx))
            findbin = null;
         else if ((findbin === right-1) && (grx2 < pnt_x - gapx))
            findbin = null;
         else if ((pnt_x < grx1 - gapx) || (pnt_x > grx2 + gapx))
            findbin = null; // if bars option used check that bar is not match
         else if (!this.options.Zero && (histo.getBinContent(findbin+1) === 0))
            findbin = null; // exclude empty bin if empty bins suppressed
      }

      let ttrect = this.draw_g.selectChild('.tooltip_bin');

      if ((findbin === null) || ((gry2 <= 0) || (gry1 >= height))) {
         ttrect.remove();
         return null;
      }

      const res = { name: this.getObjectName(), title: histo.fTitle,
                    x: midx, y: midy, exact: true,
                    color1: this.lineatt?.color ?? 'green',
                    color2: this.fillatt?.getFillColorAlt('blue') ?? 'blue',
                    lines: this.getBinTooltips(findbin) };

      if (pnt.disabled) {
         // case when tooltip should not highlight bin
         ttrect.remove();
         res.changed = true;
      } else if (show_rect) {
         if (ttrect.empty()) {
            ttrect = this.draw_g.append('svg:rect')
                                .attr('class', 'tooltip_bin')
                                .style('pointer-events', 'none')
                                .call(addHighlightStyle);
         }

         res.changed = ttrect.property('current_bin') !== findbin;

         if (res.changed) {
            ttrect.attr('x', funcs.swap_xy ? gry1 : grx1)
                  .attr('width', funcs.swap_xy ? gry2-gry1 : grx2-grx1)
                  .attr('y', funcs.swap_xy ? grx1 : gry1)
                  .attr('height', funcs.swap_xy ? grx2-grx1 : gry2-gry1)
                  .style('opacity', '0.3')
                  .property('current_bin', findbin);
         }

         res.exact = (Math.abs(midy - pnt_y) <= 5) || ((pnt_y >= gry1) && (pnt_y <= gry2));

         res.menu = res.exact; // one could show context menu when histogram is selected
         // distance to middle point, use to decide which menu to activate
         res.menu_dist = Math.sqrt((midx-pnt_x)**2 + (midy-pnt_y)**2);
      } else {
         const radius = this.lineatt.width + 3;

         if (ttrect.empty()) {
            ttrect = this.draw_g.append('svg:circle')
                                .attr('class', 'tooltip_bin')
                                .style('pointer-events', 'none')
                                .attr('r', radius)
                                .call(this.lineatt.func)
                                .call(this.fillatt.func);
         }

         res.exact = (Math.abs(midx - pnt.x) <= radius) && (Math.abs(midy - pnt.y) <= radius);

         res.menu = res.exact; // show menu only when mouse pointer exactly over the histogram
         res.menu_dist = Math.sqrt((midx-pnt.x)**2 + (midy-pnt.y)**2);

         res.changed = ttrect.property('current_bin') !== findbin;

         if (res.changed) {
            ttrect.attr('cx', midx)
                  .attr('cy', midy)
                  .property('current_bin', findbin);
         }
      }

      if (res.changed) {
         res.user_info = { obj: histo, name: histo.fName,
                           bin: findbin, cont: histo.getBinContent(findbin+1),
                           grx: midx, gry: midy };
      }

      return res;
   }

   /** @summary Fill histogram context menu */
   fillHistContextMenu(menu) {
      menu.add('Auto zoom-in', () => this.autoZoom());

      const opts = this.getSupportedDrawOptions();

      menu.addDrawMenu('Draw with', opts, arg => {
         if (arg.indexOf(kInspect) === 0)
            return this.showInspector(arg);

         this.decodeOptions(arg);

         if (this.options.need_fillcol && this.fillatt?.empty())
            this.fillatt.change(5, 1001);

         // redraw all objects in pad, inform dependent objects
         this.interactiveRedraw('pad', 'drawopt');
      });

      if (!this.snapid && !this.isTProfile() && !this.isTF1())
         menu.addRebinMenu(sz => this.rebinHist(sz));
   }

   /** @summary Rebin histogram, used via context menu */
   rebinHist(sz) {
      const histo = this.getHisto(),
            xaxis = histo.fXaxis,
            nbins = Math.floor(xaxis.fNbins/ sz);
      if (nbins < 2) return;

      const arr = new Array(nbins+2),
            xbins = (xaxis.fXbins.length > 0) ? new Array(nbins) : null;

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
      } else
         xaxis.fXmax = xaxis.fXmin + (xaxis.fXmax - xaxis.fXmin) / xaxis.fNbins * nbins * sz;


      xaxis.fNbins = nbins;

      let overflow = 0;
      while (indx < histo.fArray.length)
         overflow += histo.fArray[indx++];
      arr[nbins+1] = overflow;

      histo.fArray = arr;
      histo.fSumw2 = [];

      this.scanContent();

      this.interactiveRedraw('pad');
   }

   /** @summary Perform automatic zoom inside non-zero region of histogram */
   autoZoom() {
      let left = this.getSelectIndex('x', 'left', -1),
          right = this.getSelectIndex('x', 'right', 1);
      const dist = right - left,
            histo = this.getHisto();

      if ((dist === 0) || !histo) return;

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
   canZoomInside(axis, min, max) {
      const histo = this.getHisto();

      if ((axis === 'x') && histo && (histo.fXaxis.FindBin(max, 0.5) - histo.fXaxis.FindBin(min, 0) > 1)) return true;

      if ((axis === 'y') && (Math.abs(max-min) > Math.abs(this.ymax-this.ymin)*1e-6)) return true;

      return false;
   }

   /** @summary Call drawing function depending from 3D mode */
   async callDrawFunc(reason) {
      const main = this.getMainPainter(),
            fp = this.getFramePainter();

     if ((main !== this) && fp && (fp.mode3d !== this.options.Mode3D))
        this.copyOptionsFrom(main);

      return this.options.Mode3D ? this.draw3D(reason) : this.draw2D(reason);
   }

   /** @summary Performs 2D drawing of histogram
     * @return {Promise} when ready */
   async draw2D(/* reason */) {
      this.clear3DScene();

      this.scanContent(true);

      const pr = this.isMainPainter() ? this.drawColorPalette(false) : Promise.resolve(true);

      return pr.then(() => this.drawAxes())
               .then(() => this.draw1DBins())
               .then(() => this.updateFunctions())
               .then(() => this.updateHistTitle())
               .then(() => {
                   this.updateStatWebCanvas();
                   return this.addInteractivity();
               });
   }

   /** @summary Should performs 3D drawing of histogram
     * @desc Disable in 2D case, just draw with default options
     * @return {Promise} when ready */
   async draw3D(reason) {
      console.log('3D drawing is disabled, load ./hist/TH1Painter.mjs');
      return this.draw2D(reason);
   }

   /** @summary Redraw histogram */
   redraw(reason) {
      return this.callDrawFunc(reason);
   }

   /** @summary draw TH1 object */
   static async draw(dom, histo, opt) {
      return THistPainter._drawHist(new TH1Painter(dom, histo), opt);
   }

} // class TH1Painter

export { TH1Painter, PadDrawOptions };
