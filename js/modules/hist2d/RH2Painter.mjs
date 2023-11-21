import { gStyle, isStr, kNoZoom, kInspect } from '../core.mjs';
import { rgb as d3_rgb } from '../d3.mjs';
import { floatToString, TRandom, addHighlightStyle } from '../base/BasePainter.mjs';
import { RHistPainter } from './RHistPainter.mjs';
import { ensureRCanvas } from '../gpad/RCanvasPainter.mjs';
import { buildHist2dContour } from '../hist2d/TH2Painter.mjs';

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
      if ((kind === 'Projections') || (kind === 'Off'))
         kind = '';

      let widthX = width, widthY = width;

      if (isStr(kind) && (kind.indexOf('XY') === 0)) {
         const ws = kind.length > 2 ? kind.slice(2) : '';
         kind = 'XY';
         widthX = widthY = parseInt(ws);
      } else if (isStr(kind) && (kind.length > 1)) {
         const ps = kind.indexOf('_');
         if ((ps > 0) && (kind[0] === 'X') && (kind[ps+1] === 'Y')) {
            widthX = parseInt(kind.slice(1, ps)) || 1;
            widthY = parseInt(kind.slice(ps+2)) || 1;
            kind = 'XY';
         } else if ((ps > 0) && (kind[0] === 'Y') && (kind[ps+1] === 'X')) {
            widthY = parseInt(kind.slice(1, ps)) || 1;
            widthX = parseInt(kind.slice(ps+2)) || 1;
            kind = 'XY';
         } else {
            widthX = widthY = parseInt(kind.slice(1)) || 1;
            kind = kind[0];
         }
      }

      if (!widthX && !widthY)
         widthX = widthY = 1;

      if (kind && (this.is_projection === kind)) {
         if ((this.projection_widthX === widthX) && (this.projection_widthY === widthY))
            kind = '';
          else {
            this.projection_widthX = widthX;
            this.projection_widthY = widthY;
            return;
         }
      }

      delete this.proj_hist;

      const new_proj = (this.is_projection === kind) ? '' : kind;
      this.projection_widthX = widthX;
      this.projection_widthY = widthY;
      this.is_projection = ''; // avoid projection handling until area is created

      this.provideSpecialDrawArea(new_proj).then(() => { this.is_projection = new_proj; return this.redrawProjection(); });
   }

   /** @summary Readraw projections */
   redrawProjection(/* ii1, ii2, jj1, jj2 */) {
      // do nothing for the moment
      // if (!this.is_projection) return;
   }

   /** @summary Execute menu command */
   executeMenuCommand(method, args) {
      if (super.executeMenuCommand(method, args)) return true;

      if ((method.fName === 'SetShowProjectionX') || (method.fName === 'SetShowProjectionY')) {
         this.toggleProjection(method.fName[17], args && parseInt(args) ? parseInt(args) : 1);
         return true;
      }

      return false;
   }

   /** @summary Fill histogram context menu */
   fillHistContextMenu(menu) {
      if (this.getPadPainter()?.iscan) {
         let kind = this.is_projection || '';
         if (kind) kind += this.projection_widthX;
         if ((this.projection_widthX !== this.projection_widthY) && (this.is_projection === 'XY'))
            kind = `X${this.projection_widthX}_Y${this.projection_widthY}`;
         const kinds = ['X1', 'X2', 'X3', 'X5', 'X10', 'Y1', 'Y2', 'Y3', 'Y5', 'Y10', 'XY1', 'XY2', 'XY3', 'XY5', 'XY10'];
         if (kind) kinds.unshift('Off');

         menu.add('sub:Projections', () => menu.input('Input projection kind X1 or XY2 or X3_Y4', kind, 'string').then(val => this.toggleProjection(val)));
         for (let k = 0; k < kinds.length; ++k)
            menu.addchk(kind === kinds[k], kinds[k], kinds[k], arg => this.toggleProjection(arg));
         menu.add('endsub:');
      }

      menu.add('Auto zoom-in', () => this.autoZoom());

      const opts = this.getSupportedDrawOptions();

      menu.addDrawMenu('Draw with', opts, arg => {
         if (arg.indexOf(kInspect) === 0)
            return this.showInspector(arg);
         this.decodeOptions(arg);
         this.interactiveRedraw('pad', 'drawopt');
      });

      if (this.options.Color)
         this.fillPaletteMenu(menu);
   }

   /** @summary Process click on histogram-defined buttons */
   clickButton(funcname) {
      const res = super.clickButton(funcname);
      if (res) return res;

      switch (funcname) {
         case 'ToggleColor': return this.toggleColor();
         case 'Toggle3D': return this.toggleMode3D();
      }

      // all methods here should not be processed further
      return false;
   }

   /** @summary Fill pad toolbar with RH2-related functions */
   fillToolbar() {
      super.fillToolbar(true);

      const pp = this.getPadPainter();
      if (!pp) return;

      pp.addPadButton('th2color', 'Toggle color', 'ToggleColor');
      pp.addPadButton('th2colorz', 'Toggle color palette', 'ToggleColorZ');
      pp.addPadButton('th2draw3d', 'Toggle 3D mode', 'Toggle3D');
      pp.showPadButtons();
   }

   /** @summary Toggle color drawing mode */
   toggleColor() {
      if (this.options.Mode3D) {
         this.options.Mode3D = false;
         this.options.Color = true;
      } else
         this.options.Color = !this.options.Color;

      return this.redraw();
   }

   /** @summary Perform automatic zoom inside non-zero region of histogram */
   autoZoom() {
      const i1 = this.getSelectIndex('x', 'left', -1),
            i2 = this.getSelectIndex('x', 'right', 1),
            j1 = this.getSelectIndex('y', 'left', -1),
            j2 = this.getSelectIndex('y', 'right', 1),
            histo = this.getHisto(), xaxis = this.getAxis('x'), yaxis = this.getAxis('y');

      if ((i1 === i2) || (j1 === j2)) return;

      // first find minimum
      let min = histo.getBinContent(i1 + 1, j1 + 1);
      for (let i = i1; i < i2; ++i) {
         for (let j = j1; j < j2; ++j)
            min = Math.min(min, histo.getBinContent(i+1, j+1));
      }
      if (min > 0) return; // if all points positive, no chance for autoscale

      let ileft = i2, iright = i1, jleft = j2, jright = j1;

      for (let i = i1; i < i2; ++i) {
         for (let j = j1; j < j2; ++j) {
            if (histo.getBinContent(i + 1, j + 1) > min) {
               if (i < ileft) ileft = i;
               if (i >= iright) iright = i + 1;
               if (j < jleft) jleft = j;
               if (j >= jright) jright = j + 1;
            }
         }
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

      const histo = this.getHisto();

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
         for (let i = 0; i < this.nbinsx; ++i) {
            for (let j = 0; j < this.nbinsy; ++j) {
               const bin_content = histo.getBinContent(i+1, j+1);
               if (bin_content < this.gminbin) this.gminbin = bin_content; else
                  if (bin_content > this.gmaxbin) this.gmaxbin = bin_content;
               if (bin_content > 0)
                  if ((this.gminposbin === null) || (this.gminposbin > bin_content)) this.gminposbin = bin_content;
            }
         }
      }

      this.zmin = this.gminbin;
      this.zmax = this.gmaxbin;

      // this value used for logz scale drawing
      if ((this.gminposbin === null) && (this.gmaxbin > 0))
         this.gminposbin = this.gmaxbin*1e-4;

      if (this.options.Axis > 0)  // Paint histogram axis only
         this.draw_content = false;
      else
         this.draw_content = (this.gmaxbin !== 0) || (this.gminbin !== 0);
   }

   /** @summary Count statistic */
   countStat(cond) {
      const histo = this.getHisto(),
            res = { name: 'histo', entries: 0, integral: 0, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, matrix: [0, 0, 0, 0, 0, 0, 0, 0, 0], xmax: 0, ymax: 0, wmax: null },
            xleft = this.getSelectIndex('x', 'left'),
            xright = this.getSelectIndex('x', 'right'),
            yleft = this.getSelectIndex('y', 'left'),
            yright = this.getSelectIndex('y', 'right'),
            xaxis = this.getAxis('x'), yaxis = this.getAxis('y');
      let stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0,
          stat_sumx2 = 0, stat_sumy2 = 0,
          xside, yside, xx, yy, zz,
          xi, yi;

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

            if ((xside !== 1) || (yside !== 1)) continue;

            if (cond && !cond(xx, yy)) continue;

            if ((res.wmax === null) || (zz > res.wmax)) { res.wmax = zz; res.xmax = xx; res.ymax = yy; }

            stat_sum0 += zz;
            stat_sumx1 += xx * zz;
            stat_sumy1 += yy * zz;
            stat_sumx2 += xx**2 * zz;
            stat_sumy2 += yy**2 * zz;
         }
      }

      if (Math.abs(stat_sum0) > 1e-300) {
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
   fillStatistic(stat, dostat /*, dofit */) {
      const data = this.countStat(),
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
         stat.addText('Entries = ' + stat.format(data.entries, 'entries'));

      if (print_mean > 0) {
         stat.addText('Mean x = ' + stat.format(data.meanx));
         stat.addText('Mean y = ' + stat.format(data.meany));
      }

      if (print_rms > 0) {
         stat.addText('Std Dev x = ' + stat.format(data.rmsx));
         stat.addText('Std Dev y = ' + stat.format(data.rmsy));
      }

      if (print_integral > 0)
         stat.addText('Integral = ' + stat.format(data.matrix[4], 'entries'));

      if (print_skew > 0) {
         stat.addText('Skewness x = <undef>');
         stat.addText('Skewness y = <undef>');
      }

      if (print_kurt > 0)
         stat.addText('Kurt = <undef>');

      if ((print_under > 0) || (print_over > 0)) {
         const m = data.matrix;

         stat.addText('' + m[6].toFixed(0) + ' | ' + m[7].toFixed(0) + ' | ' + m[7].toFixed(0));
         stat.addText('' + m[3].toFixed(0) + ' | ' + m[4].toFixed(0) + ' | ' + m[5].toFixed(0));
         stat.addText('' + m[0].toFixed(0) + ' | ' + m[1].toFixed(0) + ' | ' + m[2].toFixed(0));
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

            if (entry === undefined)
               entry = entries[colindx] = { path: cmd1 };
             else if (can_merge && (entry === last_entry)) {
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
            } else
               entry.path += `h${dx}v${dy}h${-dx}z`;
         }
         if (last_entry) flush_last_entry();
      }

      entries.forEach((entry, colindx) => {
         if (entry) {
            this.draw_g
                .append('svg:path')
                .style('fill', handle.palette.getColor(colindx))
                .attr('d', entry.path);
         }
      });

      this.updatePaletteDraw();

      return handle;
   }

   /** @summary Draw histogram bins as contour */
   drawBinsContour(funcs, frame_w, frame_h) {
      const handle = this.prepareDraw({ rounding: false, extra: 100 }),
          main = this.getFramePainter(),
          palette = main.getHistPalette(),
          levels = palette.getContour(),
          func = main.getProjectionFunc(),

       BuildPath = (xp, yp, iminus, iplus, do_close) => {
         let cmd = '', last, pnt, first, isany;
         for (let i = iminus; i <= iplus; ++i) {
            if (func) {
               pnt = func(xp[i], yp[i]);
               pnt.x = Math.round(funcs.grx(pnt.x));
               pnt.y = Math.round(funcs.gry(pnt.y));
            } else
               pnt = { x: Math.round(xp[i]), y: Math.round(yp[i]) };

            if (!cmd) {
               cmd = `M${pnt.x},${pnt.y}`; first = pnt;
            } else if ((i === iplus) && first && (pnt.x === first.x) && (pnt.y === first.y)) {
               if (!isany) return ''; // all same points
               cmd += 'z'; do_close = false;
            } else if ((pnt.x !== last.x) && (pnt.y !== last.y)) {
               cmd += `l${pnt.x - last.x},${pnt.y - last.y}`; isany = true;
            } else if (pnt.x !== last.x) {
               cmd += `h${pnt.x - last.x}`; isany = true;
            } else if (pnt.y !== last.y) {
               cmd += `v${pnt.y - last.y}`; isany = true;
            }
            last = pnt;
         }
         if (do_close) cmd += 'z';
         return cmd;
      };

      if (this.options.Contour === 14) {
         this.draw_g
             .append('svg:path')
             .attr('d', `M0,0h${frame_w}v${frame_h}h${-frame_w}z`)
             .style('fill', palette.getColor(0));
      }

      buildHist2dContour(this.getHisto(), handle, levels, palette,
         (colindx, xp, yp, iminus, iplus) => {
            const icol = palette.getColor(colindx);
            let fillcolor = icol, lineatt;

            switch (this.options.Contour) {
               case 1: break;
               case 11: fillcolor = 'none'; lineatt = this.createAttLine({ color: icol, std: false }); break;
               case 12: fillcolor = 'none'; lineatt = this.createAttLine({ color: 1, style: (colindx%5 + 1), width: 1, std: false }); break;
               case 13: fillcolor = 'none'; lineatt = this.lineatt; break;
               case 14: break;
            }

            const dd = BuildPath(xp, yp, iminus, iplus, fillcolor !== 'none');
            if (!dd) return;

            const elem = this.draw_g
                          .append('svg:path')
                          .attr('d', dd)
                          .style('fill', fillcolor);

            if (lineatt)
               elem.call(lineatt.func);
         }
      );

      handle.hide_only_zeros = true; // text drawing suppress only zeros

      return handle;
   }

   /** @summary Create polybin */
   createPolyBin() {
      // see how TH2Painter is implemented
      return '';
   }

   /** @summary Draw RH2 bins as text */
   drawBinsText(handle) {
      if (handle === null) handle = this.prepareDraw({ rounding: false });

      const histo = this.getHisto(),
            textFont = this.v7EvalFont('text', { size: 20, color: 'black', align: 22 }),
            text_offset = this.options.BarOffset || 0,
            text_g = this.draw_g.append('svg:g').attr('class', 'th2_text'),
            di = handle.stepi, dj = handle.stepj,
            profile2d = false;
      let i, j, binz, binw, binh, text, x, y, width, height;

      this.startTextDrawing(textFont, 'font', text_g);

      for (i = handle.i1; i < handle.i2; i += di) {
         for (j = handle.j1; j < handle.j2; j += dj) {
            binz = histo.getBinContent(i+1, j+1);
            if ((binz === 0) && !this._show_empty_bins) continue;

            binw = handle.grx[i+di] - handle.grx[i];
            binh = handle.gry[j] - handle.gry[j+dj];

            if (profile2d)
               binz = histo.getBinEntries(i+1, j+1);

            text = (binz === Math.round(binz)) ? binz.toString() : floatToString(binz, gStyle.fPaintTextFormat);

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
      }

      return this.finishTextDrawing(text_g, true).then(() => {
         handle.hide_only_zeros = true; // text drawing suppress only zeros
         return handle;
      });
   }

   /** @summary Draw RH2 bins as arrows */
   drawBinsArrow() {
      const histo = this.getHisto(),
            handle = this.prepareDraw({ rounding: false }),
            scale_x = (handle.grx[handle.i2] - handle.grx[handle.i1])/(handle.i2 - handle.i1 + 1-0.03)/2,
            scale_y = (handle.gry[handle.j2] - handle.gry[handle.j1])/(handle.j2 - handle.j1 + 1-0.03)/2,
            di = handle.stepi, dj = handle.stepj,
            makeLine = (dx, dy) => dx ? (dy ? `l${dx},${dy}` : `h${dx}`) : (dy ? `v${dy}` : '');
      let cmd = '', i, j, dn = 1e-30, dx, dy, xc, yc,
          dxn, dyn, x1, x2, y1, y2, anr, si, co;

      for (let loop = 0; loop < 2; ++loop) {
         for (i = handle.i1; i < handle.i2; i += di) {
            for (j = handle.j1; j < handle.j2; j += dj) {
               if (i === handle.i1)
                  dx = histo.getBinContent(i+1+di, j+1) - histo.getBinContent(i+1, j+1);
                else if (i >= handle.i2-di)
                  dx = histo.getBinContent(i+1, j+1) - histo.getBinContent(i+1-di, j+1);
                else
                  dx = 0.5*(histo.getBinContent(i+1+di, j+1) - histo.getBinContent(i+1-di, j+1));

               if (j === handle.j1)
                  dy = histo.getBinContent(i+1, j+1+dj) - histo.getBinContent(i+1, j+1);
                else if (j >= handle.j2-dj)
                  dy = histo.getBinContent(i+1, j+1) - histo.getBinContent(i+1, j+1-dj);
                else
                  dy = 0.5*(histo.getBinContent(i+1, j+1+dj) - histo.getBinContent(i+1, j+1-dj));


               if (loop === 0)
                  dn = Math.max(dn, Math.abs(dx), Math.abs(dy));
                else {
                  xc = (handle.grx[i] + handle.grx[i+di])/2;
                  yc = (handle.gry[j] + handle.gry[j+dj])/2;
                  dxn = scale_x*dx/dn;
                  dyn = scale_y*dy/dn;
                  x1 = xc - dxn;
                  x2 = xc + dxn;
                  y1 = yc - dyn;
                  y2 = yc + dyn;
                  dx = Math.round(x2-x1);
                  dy = Math.round(y2-y1);

                  if ((dx !== 0) || (dy !== 0)) {
                     cmd += 'M'+Math.round(x1)+','+Math.round(y1) + makeLine(dx, dy);

                     if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                        anr = Math.sqrt(2/(dx**2 + dy**2));
                        si = Math.round(anr*(dx + dy));
                        co = Math.round(anr*(dx - dy));
                        if (si || co)
                           cmd += `m${-si},${co}` + makeLine(si, -co) + makeLine(-co, -si);
                     }
                  }
               }
            }
         }
      }

      this.draw_g
         .append('svg:path')
         .attr('d', cmd)
         .style('fill', 'none')
         .call(this.lineatt.func);

      return handle;
   }

   /** @summary Draw RH2 bins as boxes */
   drawBinsBox() {
      const histo = this.getHisto(),
            handle = this.prepareDraw({ rounding: false }),
            main = this.getFramePainter();

      if (main.maxbin === main.minbin) {
         main.maxbin = this.gmaxbin;
         main.minbin = this.gminbin;
         main.minposbin = this.gminposbin;
      }
      if (main.maxbin === main.minbin)
         main.minbin = Math.min(0, main.maxbin-1);

      const absmax = Math.max(Math.abs(main.maxbin), Math.abs(main.minbin)),
            absmin = Math.max(0, main.minbin),
            di = handle.stepi, dj = handle.stepj;
      let i, j, binz, absz, res = '', cross = '', btn1 = '', btn2 = '',
          zdiff, dgrx, dgry, xx, yy, ww, hh,
          xyfactor, uselogz = false, logmin = 0;

      if (main.logz && (absmax > 0)) {
         uselogz = true;
         const logmax = Math.log(absmax);
         if (absmin > 0)
            logmin = Math.log(absmin);
         else if ((main.minposbin >= 1) && (main.minposbin < 100))
            logmin = Math.log(0.7);
          else
            logmin = (main.minposbin > 0) ? Math.log(0.7*main.minposbin) : logmax - 10;
         if (logmin >= logmax) logmin = logmax - 10;
         xyfactor = 1.0 / (logmax - logmin);
      } else
         xyfactor = 1.0 / (absmax - absmin);


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
               const pww = Math.round(ww*0.1),
                     phh = Math.round(hh*0.1),
                     side1 = `M${xx},${yy}h${ww}l${-pww},${phh}h${2*pww-ww}v${hh-2*phh}l${-pww},${phh}z`,
                     side2 = `M${xx+ww},${yy+hh}v${-hh}l${-pww},${phh}v${hh-2*phh}h${2*pww-ww}l${-pww},${phh}z`;
               btn2 += (binz < 0) ? side1 : side2;
               btn1 += (binz < 0) ? side2 : side1;
            }
         }
      }

      if (res) {
         const elem = this.draw_g
                        .append('svg:path')
                        .attr('d', res)
                        .call(this.fillatt.func);
         if ((this.options.BoxStyle !== 11) && this.fillatt.empty())
            elem.call(this.lineatt.func);
      }

      if (btn1 && this.fillatt.hasColor()) {
         this.draw_g.append('svg:path')
                    .attr('d', btn1)
                    .call(this.fillatt.func)
                    .style('fill', d3_rgb(this.fillatt.color).brighter(0.5).formatHex());
      }

      if (btn2) {
         this.draw_g.append('svg:path')
                    .attr('d', btn2)
                    .call(this.fillatt.func)
                    .style('fill', !this.fillatt.hasColor() ? 'red' : d3_rgb(this.fillatt.color).darker(0.5).formatHex());
      }

      if (cross) {
         const elem = this.draw_g.append('svg:path')
                               .attr('d', cross)
                               .style('fill', 'none');
         if (!this.lineatt.empty())
            elem.call(this.lineatt.func);
      }

      return handle;
   }

   /** @summary Draw RH2 bins as scatter plot */
   drawBinsScatter() {
      const histo = this.getHisto(),
            handle = this.prepareDraw({ rounding: true, pixel_density: true, scatter_plot: true }),
            colPaths = [], currx = [], curry = [], cell_w = [], cell_h = [],
            scale = this.options.ScatCoef * ((this.gmaxbin) > 2000 ? 2000 / this.gmaxbin : 1),
            di = handle.stepi, dj = handle.stepj,
            rnd = new TRandom(handle.sumz);
      let colindx, cmd1, cmd2, i, j, binz, cw, ch, factor = 1;

      if (scale*handle.sumz < 1e5) {
         // one can use direct drawing of scatter plot without any patterns

         this.createv7AttMarker();

         this.markeratt.resetPos();

         let path = '', k, npix;
         for (i = handle.i1; i < handle.i2; i += di) {
            cw = handle.grx[i+di] - handle.grx[i];
            for (j = handle.j1; j < handle.j2; j += dj) {
               ch = handle.gry[j] - handle.gry[j+dj];
               binz = histo.getBinContent(i + 1, j + 1);

               npix = Math.round(scale*binz);
               if (npix <= 0) continue;

               for (k = 0; k < npix; ++k) {
                  path += this.markeratt.create(
                            Math.round(handle.grx[i] + cw * rnd.random()),
                            Math.round(handle.gry[j+1] + ch * rnd.random()));
               }
            }
         }

         this.draw_g
              .append('svg:path')
              .attr('d', path)
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

            cmd1 = `M${handle.grx[i]},${handle.gry[j+dj]}`;
            if (colPaths[colindx] === undefined) {
               colPaths[colindx] = cmd1;
               cell_w[colindx] = cw;
               cell_h[colindx] = ch;
            } else {
               cmd2 = `m${handle.grx[i]-currx[colindx]},${handle.gry[j+dj]-curry[colindx]}`;
               colPaths[colindx] += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
               cell_w[colindx] = Math.max(cell_w[colindx], cw);
               cell_h[colindx] = Math.max(cell_h[colindx], ch);
            }

            currx[colindx] = handle.grx[i];
            curry[colindx] = handle.gry[j+dj];

            colPaths[colindx] += `v${ch}h${cw}v${-ch}z`;
         }
      }

      const layer = this.getFrameSvg().selectChild('.main_layer');
      let defs = layer.selectChild('def');
      if (defs.empty() && (colPaths.length > 0))
         defs = layer.insert('svg:defs', ':first-child');

      this.createv7AttMarker();

      const cntr = handle.palette.getContour();

      for (colindx = 0; colindx < colPaths.length; ++colindx) {
         if ((colPaths[colindx] !== undefined) && (colindx<cntr.length)) {
            const pattern_id = (this.pad_name || 'canv') + `_scatter_${colindx}`;
            let pattern = defs.selectChild(`#${pattern_id}`);
            if (pattern.empty()) {
               pattern = defs.append('svg:pattern')
                             .attr('id', pattern_id)
                             .attr('patternUnits', 'userSpaceOnUse');
            } else
               pattern.selectAll('*').remove();

            let npix = Math.round(factor*cntr[colindx]*cell_w[colindx]*cell_h[colindx]);
            if (npix < 1) npix = 1;

            const arrx = new Float32Array(npix), arry = new Float32Array(npix);

           if (npix === 1)
              arrx[0] = arry[0] = 0.5;
            else {
              for (let n = 0; n < npix; ++n) {
                 arrx[n] = rnd.random();
                 arry[n] = rnd.random();
              }
           }

           this.markeratt.resetPos();

           let path = '';

           for (let n = 0; n < npix; ++n)
              path += this.markeratt.create(arrx[n] * cell_w[colindx], arry[n] * cell_h[colindx]);

           pattern.attr('width', cell_w[colindx])
                  .attr('height', cell_h[colindx])
                  .append('svg:path')
                  .attr('d', path)
                  .call(this.markeratt.func);

           this.draw_g
               .append('svg:path')
               .attr('scatter-index', colindx)
               .style('fill', `url(#${pattern_id})`)
               .attr('d', colPaths[colindx]);
        }
}

      return handle;
   }

   /** @summary Draw RH2 bins in 2D mode */
   async draw2DBins() {
      if (!this.draw_content) {
         this.removeG();
         return false;
      }

      this.createHistDrawAttributes();

      this.createG(true);

      const pmain = this.getFramePainter(),
            rect = pmain.getFrameRect(),
            funcs = pmain.getGrFuncs(this.options.second_x, this.options.second_y);
      let handle = null, pr = null;

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
      const lines = [],
            histo = this.getHisto();
      let binz = histo.getBinContent(i+1, j+1),
          di = 1, dj = 1;

      if (this.isDisplayItem()) {
         di = histo.stepx || 1;
         dj = histo.stepy || 1;
      }

      lines.push(this.getObjectHint() || 'histo<2>',
                 'x = ' + this.getAxisBinTip('x', i, di),
                 'y = ' + this.getAxisBinTip('y', j, dj),
                 `bin = ${i+1}, ${j+1}`);

      if (histo.$baseh) binz -= histo.$baseh.getBinContent(i+1, j+1);

      const lbl = 'entries = ' + ((di > 1) || (dj > 1) ? '~' : '');

      if (binz === Math.round(binz))
         lines.push(lbl + binz);
      else
         lines.push(lbl + floatToString(binz, gStyle.fStatFormat));

      return lines;
   }

   /** @summary Provide text information (tooltips) for poly bin */
   getPolyBinTooltips() {
      // see how TH2Painter is implemented
      return [];
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      const histo = this.getHisto(),
            h = this.tt_handle;
      let ttrect = this.draw_g?.selectChild('.tooltip_bin');

      if (!pnt || !this.draw_content || !this.draw_g || !h || this.options.Proj) {
         ttrect?.remove();
         return null;
      }

      if (h.poly) {
         // process tooltips from TH2Poly - see TH2Painter
         return null;
      }

      let i, j, binz = 0, colindx = null;

      // search bins position
      for (i = h.i1; i < h.i2; ++i)
         if ((pnt.x>=h.grx[i]) && (pnt.x<=h.grx[i+1])) break;

      for (j = h.j1; j < h.j2; ++j)
         if ((pnt.y>=h.gry[j+1]) && (pnt.y<=h.gry[j])) break;

      if ((i < h.i2) && (j < h.j2)) {
         binz = histo.getBinContent(i+1, j+1);
         if (this.is_projection)
            colindx = 0; // just to avoid hide
          else if (h.hide_only_zeros)
            colindx = (binz === 0) && !this._show_empty_bins ? null : 0;
          else {
            colindx = h.palette.getContourIndex(binz);
            if ((colindx === null) && (binz === 0) && this._show_empty_bins) colindx = 0;
         }
      }

      if (colindx === null) {
         ttrect.remove();
         return null;
      }

      const res = { name: 'histo', title: histo.fTitle || 'title',
                  x: pnt.x, y: pnt.y,
                  color1: this.lineatt?.color ?? 'green',
                  color2: this.fillatt?.getFillColorAlt('blue') ?? 'blue',
                  lines: this.getBinTooltips(i, j), exact: true, menu: true };

      if (this.options.Color)
         res.color2 = h.palette.getColor(colindx);

      if (pnt.disabled && !this.is_projection) {
         ttrect.remove();
         res.changed = true;
      } else {
         if (ttrect.empty()) {
            ttrect = this.draw_g.append('svg:path')
                                .attr('class', 'tooltip_bin')
                                .style('pointer-events', 'none')
                                .call(addHighlightStyle);
         }

         const pmain = this.getFramePainter();
         let i1 = i, i2 = i+1,
             j1 = j, j2 = j+1,
             x1 = h.grx[i1], x2 = h.grx[i2],
             y1 = h.gry[j2], y2 = h.gry[j1],
             binid = i*10000 + j, path;

         if (this.is_projection) {
            const pwx = this.projection_widthX || 1, ddx = (pwx - 1) / 2;
            if ((this.is_projection.indexOf('X')) >= 0 && (pwx > 1)) {
               if (j2+ddx >= h.j2) {
                  j2 = Math.min(Math.round(j2+ddx), h.j2);
                  j1 = Math.max(j2-pwx, h.j1);
               } else {
                  j1 = Math.max(Math.round(j1-ddx), h.j1);
                  j2 = Math.min(j1+pwx, h.j2);
               }
            }
            const pwy = this.projection_widthY || 1, ddy = (pwy - 1) / 2;
            if ((this.is_projection.indexOf('Y')) >= 0 && (pwy > 1)) {
               if (i2+ddy >= h.i2) {
                  i2 = Math.min(Math.round(i2+ddy), h.i2);
                  i1 = Math.max(i2-pwy, h.i1);
               } else {
                  i1 = Math.max(Math.round(i1-ddy), h.i1);
                  i2 = Math.min(i1+pwy, h.i2);
               }
            }
         }

         if (this.is_projection === 'X') {
            x1 = 0; x2 = pmain.getFrameWidth();
            y1 = h.gry[j2]; y2 = h.gry[j1];
            binid = j1*777 + j2*333;
         } else if (this.is_projection === 'Y') {
            y1 = 0; y2 = pmain.getFrameHeight();
            x1 = h.grx[i1]; x2 = h.grx[i2];
            binid = i1*777 + i2*333;
         } else if (this.is_projection === 'XY') {
            y1 = h.gry[j2]; y2 = h.gry[j1];
            x1 = h.grx[i1]; x2 = h.grx[i2];
            binid = i1*789 + i2*653 + j1*12345 + j2*654321;
            path = `M${x1},0H${x2}V${y1}H${pmain.getFrameWidth()}V${y2}H${x2}V${pmain.getFrameHeight()}H${x1}V${y2}H0V${y1}H${x1}Z`;
         }

         res.changed = ttrect.property('current_bin') !== binid;

         if (res.changed) {
            ttrect.attr('d', path || `M${x1},${y1}H${x2}V${y2}H${x1}Z`)
                  .style('opacity', '0.7')
                  .property('current_bin', binid);
         }

         if (this.is_projection && res.changed)
            this.redrawProjection(i1, i2, j1, j2);
      }

      if (res.changed) {
         res.user_info = { obj: histo, name: 'histo',
                           bin: histo.getBin(i+1, j+1), cont: binz, binx: i+1, biny: j+1,
                           grx: pnt.x, gry: pnt.y };
      }

      return res;
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis, min, max) {
      if (axis === 'z') return true;
      const obj = this.getAxis(axis);
      return obj.FindBin(max, 0.5) - obj.FindBin(min, 0) > 1;
   }

   /** @summary Performs 2D drawing of histogram
     * @return {Promise} when ready */
   async draw2D(reason) {
      this.clear3DScene();

      return this.drawFrameAxes().then(res => {
        return res ? this.drawingBins(reason) : false;
      }).then(res => {
         if (res) return this.draw2DBins().then(() => this.addInteractivity());
      }).then(() => this);
   }

   /** @summary Performs 3D drawing of histogram
     * @return {Promise} when ready */
   async draw3D(reason) {
      console.log('3D drawing is disabled, load ./hist/RH1Painter.mjs');
      return this.draw2D(reason);
   }

   /** @summary Call drawing function depending from 3D mode */
   async callDrawFunc(reason) {
      const main = this.getFramePainter();

      if (main && (main.mode3d !== this.options.Mode3D) && !this.isMainPainter())
         this.options.Mode3D = main.mode3d;

      return this.options.Mode3D ? this.draw3D(reason) : this.draw2D(reason);
   }

   /** @summary Redraw histogram */
   async redraw(reason) {
      return this.callDrawFunc(reason);
   }

   /** @summary Draw histogram using painter instance
     * @private */
   static async _draw(painter /* , opt */) {
      return ensureRCanvas(painter).then(() => {
         painter.setAsMainPainter();

         painter.options = { Hist: false, Error: false, Zero: false, Mark: false,
                             Line: false, Fill: false, Lego: 0, Surf: 0,
                             Text: true, TextAngle: 0, TextKind: '',
                             BaseLine: false, Mode3D: false, AutoColor: 0,
                             Color: false, Scat: false, ScatCoef: 1, Box: false, BoxStyle: 0, Arrow: false, Contour: 0, Proj: 0,
                             BarOffset: 0, BarWidth: 1, minimum: kNoZoom, maximum: kNoZoom,
                             FrontBox: false, BackBox: false };

         const kind = painter.v7EvalAttr('kind', ''),
             sub = painter.v7EvalAttr('sub', 0),
             o = painter.options;

         o.Text = painter.v7EvalAttr('drawtext', false);

         switch (kind) {
            case 'lego': o.Lego = sub > 0 ? 10+sub : 12; o.Mode3D = true; break;
            case 'surf': o.Surf = sub > 0 ? 10+sub : 1; o.Mode3D = true; break;
            case 'box': o.Box = true; o.BoxStyle = 10 + sub; break;
            case 'err': o.Error = true; o.Mode3D = true; break;
            case 'cont': o.Contour = sub > 0 ? 10+sub : 1; break;
            case 'arr': o.Arrow = true; break;
            case 'scat': o.Scat = true; break;
            case 'col': o.Color = true; break;
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
   static async draw(dom, obj, opt) {
      // create painter and add it to canvas
      return RH2Painter._draw(new RH2Painter(dom, obj), opt);
   }

} //  class RH2Painter

export { RH2Painter };
