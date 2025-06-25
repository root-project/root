import { gStyle, settings, kInspect, clTF1, clTF3, clTProfile3D, BIT, isFunc } from '../core.mjs';
import { TRandom, floatToString } from '../base/BasePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { TAxisPainter } from '../gpad/TAxisPainter.mjs';
import { createLineSegments, PointsCreator, Box3D, THREE } from '../base/base3d.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';
import { assignFrame3DMethods } from './hist3d.mjs';
import { proivdeEvalPar, getTF1Value } from '../base/func.mjs';


/**
 * @summary Painter for TH3 classes
 * @private
 */

class TH3Painter extends THistPainter {

   #box_option; // actual box option

   /** @summary Returns number of histogram dimensions */
   getDimension() { return 3; }

   /** @summary Scan TH3 histogram content */
   scanContent(when_axis_changed) {
      // no need to re-scan histogram while result does not depend from axis selection
      if (when_axis_changed && this.nbinsx && this.nbinsy && this.nbinsz) return;

      const histo = this.getHisto();

      this.extractAxesProperties(3);

      // global min/max, used at the moment in 3D drawing
      this.gminbin = this.gmaxbin = histo.getBinContent(1, 1, 1);
      this.gminposbin = null;

      for (let i = 0; i < this.nbinsx; ++i) {
         for (let j = 0; j < this.nbinsy; ++j) {
            for (let k = 0; k < this.nbinsz; ++k) {
               const bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (bin_content < this.gminbin)
                  this.gminbin = bin_content;
               else if (bin_content > this.gmaxbin)
                  this.gmaxbin = bin_content;

               if ((bin_content > 0) && ((this.gminposbin === null) || (this.gminposbin > bin_content)))
                  this.gminposbin = bin_content;
            }
         }
      }

      if ((this.gminposbin === null) && (this.gmaxbin > 0))
         this.gminposbin = this.gmaxbin*1e-4;

      this.draw_content = this.gmaxbin || this.gminbin;

      this.transferFunc = this.findFunction(clTF1, 'TransferFunction');
      this.transferFunc?.SetBit(BIT(9), true); // TF1::kNotDraw
   }

   /** @summary Count TH3 statistic */
   countStat(cond, count_skew) {
      const histo = this.getHisto(), xaxis = histo.fXaxis, yaxis = histo.fYaxis, zaxis = histo.fZaxis,
            i1 = this.getSelectIndex('x', 'left'),
            i2 = this.getSelectIndex('x', 'right'),
            j1 = this.getSelectIndex('y', 'left'),
            j2 = this.getSelectIndex('y', 'right'),
            k1 = this.getSelectIndex('z', 'left'),
            k2 = this.getSelectIndex('z', 'right'),
            fp = this.getFramePainter(),
            res = { name: histo.fName, entries: 0, eff_entries: 0, integral: 0,
                    meanx: 0, meany: 0, meanz: 0, rmsx: 0, rmsy: 0, rmsz: 0,
                    skewx: 0, skewy: 0, skewz: 0, skewd: 0, kurtx: 0, kurty: 0, kurtz: 0, kurtd: 0 },
            has_counted_stat = (Math.abs(histo.fTsumw) > 1e-300) && !fp.isAxisZoomed('x') && !fp.isAxisZoomed('y') && !fp.isAxisZoomed('z');
      let xi, yi, zi, xx, xside, yy, yside, zz, zside, cont,
          stat_sum0 = 0, stat_sumw2 = 0, stat_sumx1 = 0, stat_sumy1 = 0,
          stat_sumz1 = 0, stat_sumx2 = 0, stat_sumy2 = 0, stat_sumz2 = 0;

      if (!isFunc(cond))
         cond = null;

      for (xi = 0; xi < this.nbinsx+2; ++xi) {
         xx = xaxis.GetBinCoord(xi - 0.5);
         xside = (xi < i1) ? 0 : (xi > i2 ? 2 : 1);

         for (yi = 0; yi < this.nbinsy+2; ++yi) {
            yy = yaxis.GetBinCoord(yi - 0.5);
            yside = (yi < j1) ? 0 : (yi > j2 ? 2 : 1);

            for (zi = 0; zi < this.nbinsz+2; ++zi) {
               zz = zaxis.GetBinCoord(zi - 0.5);
               zside = (zi < k1) ? 0 : (zi > k2 ? 2 : 1);

               if (cond && !cond(xx, yy, zz)) continue;

               cont = histo.getBinContent(xi, yi, zi);
               res.entries += cont;

               if (!has_counted_stat && (xside === 1) && (yside === 1) && (zside === 1)) {
                  stat_sum0 += cont;
                  stat_sumw2 += cont * cont;
                  stat_sumx1 += xx * cont;
                  stat_sumy1 += yy * cont;
                  stat_sumz1 += zz * cont;
                  stat_sumx2 += xx**2 * cont;
                  stat_sumy2 += yy**2 * cont;
                  stat_sumz2 += zz**2 * cont;
               }
            }
         }
      }

      if (has_counted_stat) {
         stat_sum0 = histo.fTsumw;
         stat_sumw2 = histo.fTsumw2;
         stat_sumx1 = histo.fTsumwx;
         stat_sumx2 = histo.fTsumwx2;
         stat_sumy1 = histo.fTsumwy;
         stat_sumy2 = histo.fTsumwy2;
         stat_sumz1 = histo.fTsumwz;
         stat_sumz2 = histo.fTsumwz2;
      }

      if (Math.abs(stat_sum0) > 1e-300) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.meanz = stat_sumz1 / stat_sum0;
         res.rmsx = Math.sqrt(Math.abs(stat_sumx2 / stat_sum0 - res.meanx * res.meanx));
         res.rmsy = Math.sqrt(Math.abs(stat_sumy2 / stat_sum0 - res.meany * res.meany));
         res.rmsz = Math.sqrt(Math.abs(stat_sumz2 / stat_sum0 - res.meanz * res.meanz));
      }

      res.integral = stat_sum0;

      if (histo.fEntries > 0)
         res.entries = histo.fEntries;

      res.eff_entries = stat_sumw2 ? stat_sum0*stat_sum0/stat_sumw2 : Math.abs(stat_sum0);

      if (count_skew && !this.isTH2Poly()) {
         let sumx3 = 0, sumy3 = 0, sumz3 = 0, sumx4 = 0, sumy4 = 0, sumz4 = 0, np = 0;
         for (xi = i1; xi < i2; ++xi) {
            xx = xaxis.GetBinCoord(xi + 0.5);
            for (yi = j1; yi < j2; ++yi) {
               yy = yaxis.GetBinCoord(yi + 0.5);
               for (zi = k1; zi < k2; ++zi) {
                  zz = zaxis.GetBinCoord(zi + 0.5);
                  if (cond && !cond(xx, yy, zz)) continue;
                  const w = histo.getBinContent(xi + 1, yi + 1, zi + 1);
                  np += w;
                  sumx3 += w * Math.pow(xx - res.meanx, 3);
                  sumy3 += w * Math.pow(yy - res.meany, 3);
                  sumz3 += w * Math.pow(zz - res.meany, 3);
                  sumx4 += w * Math.pow(xx - res.meanx, 4);
                  sumy4 += w * Math.pow(yy - res.meany, 4);
                  sumz4 += w * Math.pow(yy - res.meany, 4);
               }
            }
         }

         const stddev3x = Math.pow(res.rmsx, 3),
               stddev3y = Math.pow(res.rmsy, 3),
               stddev3z = Math.pow(res.rmsz, 3),
               stddev4x = Math.pow(res.rmsx, 4),
               stddev4y = Math.pow(res.rmsy, 4),
               stddev4z = Math.pow(res.rmsz, 4);

         if (np * stddev3x)
            res.skewx = sumx3 / (np * stddev3x);
         if (np * stddev3y)
            res.skewy = sumy3 / (np * stddev3y);
         if (np * stddev3z)
            res.skewz = sumz3 / (np * stddev3z);
         res.skewd = res.eff_entries > 0 ? Math.sqrt(6/res.eff_entries) : 0;

         if (np * stddev4x)
            res.kurtx = sumx4 / (np * stddev4x) - 3;
         if (np * stddev4y)
            res.kurty = sumy4 / (np * stddev4y) - 3;
         if (np * stddev4z)
            res.kurtz = sumz4 / (np * stddev4z) - 3;
         res.kurtd = res.eff_entries > 0 ? Math.sqrt(24/res.eff_entries) : 0;
      }

      return res;
   }

   /** @summary Fill TH3 statistic in stat box */
   fillStatistic(stat, dostat, dofit) {
      // no need to refill statistic if histogram is dummy
      if (this.isIgnoreStatsFill())
         return false;

      if (dostat === 1) dostat = 1111;

      const print_name = dostat % 10,
            print_entries = Math.floor(dostat / 10) % 10,
            print_mean = Math.floor(dostat / 100) % 10,
            print_rms = Math.floor(dostat / 1000) % 10,
            print_integral = Math.floor(dostat / 1000000) % 10,
            print_skew = Math.floor(dostat / 10000000) % 10,
            print_kurt = Math.floor(dostat / 100000000) % 10,
            data = this.countStat(undefined, (print_skew > 0) || (print_kurt > 0));

      stat.clearPave();

      if (print_name > 0)
         stat.addText(data.name);

      if (print_entries > 0)
         stat.addText('Entries = ' + stat.format(data.entries, 'entries'));

      if (print_mean > 0) {
         stat.addText('Mean x = ' + stat.format(data.meanx));
         stat.addText('Mean y = ' + stat.format(data.meany));
         stat.addText('Mean z = ' + stat.format(data.meanz));
      }

      if (print_rms > 0) {
         stat.addText('Std Dev x = ' + stat.format(data.rmsx));
         stat.addText('Std Dev y = ' + stat.format(data.rmsy));
         stat.addText('Std Dev z = ' + stat.format(data.rmsz));
      }

      if (print_integral > 0)
         stat.addText('Integral = ' + stat.format(data.integral, 'entries'));

      if (print_skew === 2) {
         stat.addText(`Skewness x = ${stat.format(data.skewx)} #pm ${stat.format(data.skewd)}`);
         stat.addText(`Skewness y = ${stat.format(data.skewy)} #pm ${stat.format(data.skewd)}`);
         stat.addText(`Skewness z = ${stat.format(data.skewz)} #pm ${stat.format(data.skewd)}`);
      } else if (print_skew > 0) {
         stat.addText(`Skewness x = ${stat.format(data.skewx)}`);
         stat.addText(`Skewness y = ${stat.format(data.skewy)}`);
         stat.addText(`Skewness z = ${stat.format(data.skewz)}`);
      }

      if (print_kurt === 2) {
         stat.addText(`Kurtosis x = ${stat.format(data.kurtx)} #pm ${stat.format(data.kurtd)}`);
         stat.addText(`Kurtosis y = ${stat.format(data.kurty)} #pm ${stat.format(data.kurtd)}`);
         stat.addText(`Kurtosis z = ${stat.format(data.kurtz)} #pm ${stat.format(data.kurtd)}`);
      } else if (print_kurt > 0) {
         stat.addText(`Kurtosis x = ${stat.format(data.kurtx)}`);
         stat.addText(`Kurtosis y = ${stat.format(data.kurty)}`);
         stat.addText(`Kurtosis z = ${stat.format(data.kurtz)}`);
      }

      if (dofit) stat.fillFunctionStat(this.findFunction(clTF3), dofit, 3);

      return true;
   }

   /** @summary Provide text information (tooltips) for histogram bin */
   getBinTooltips(ix, iy, iz) {
      const lines = [], histo = this.getHisto();

      lines.push(this.getObjectHint(),
                 `x = ${this.getAxisBinTip('x', histo.fXaxis, ix)}  xbin=${ix+1}`,
                 `y = ${this.getAxisBinTip('y', histo.fYaxis, iy)}  ybin=${iy+1}`,
                 `z = ${this.getAxisBinTip('z', histo.fZaxis, iz)}  zbin=${iz+1}`);

      const binz = histo.getBinContent(ix+1, iy+1, iz+1);
      if (binz === Math.round(binz))
         lines.push(`entries = ${binz}`);
      else
         lines.push(`entries = ${floatToString(binz, gStyle.fStatFormat)}`);

      if (this.matchObjectType(clTProfile3D)) {
         const errz = histo.getBinError(histo.getBin(ix+1, iy+1, iz+1));
         lines.push('error = ' + ((errz === Math.round(errz)) ? errz.toString() : floatToString(errz, gStyle.fPaintTextFormat)));
      }

      return lines;
   }

   /** @summary draw 3D histogram as scatter plot
     * @desc If there are too many points, box will be displayed
     * @return {Promise|false} either Promise or just false that drawing cannot be performed */
   draw3DScatter() {
      const histo = this.getObject(),
            fp = this.getFramePainter(),
            i1 = this.getSelectIndex('x', 'left', 0.5),
            i2 = this.getSelectIndex('x', 'right', 0),
            j1 = this.getSelectIndex('y', 'left', 0.5),
            j2 = this.getSelectIndex('y', 'right', 0),
            k1 = this.getSelectIndex('z', 'left', 0.5),
            k2 = this.getSelectIndex('z', 'right', 0);
      let i, j, k, bin_content;

      if ((i2 <= i1) || (j2 <= j1) || (k2 <= k1))
         return Promise.resolve(true);

      // scale down factor if too large values
      const coef = (this.gmaxbin > 1000) ? 1000/this.gmaxbin : 1,
            content_lmt = Math.max(0, this.gminbin);
      let numpixels = 0, sumz = 0;

      for (i = i1; i < i2; ++i) {
         for (j = j1; j < j2; ++j) {
            for (k = k1; k < k2; ++k) {
               bin_content = histo.getBinContent(i+1, j+1, k+1);
               sumz += bin_content;
               if (bin_content <= content_lmt) continue;
               numpixels += Math.round(bin_content*coef);
            }
         }
      }

      // too many pixels - use box drawing
      if (numpixels > (fp.webgl ? 100000 : 30000))
         return false;

      const pnts = new PointsCreator(numpixels, fp.webgl, fp.size_x3d / 200),
            bins = new Int32Array(numpixels),
            rnd = new TRandom(sumz);
      let nbin = 0;

      for (i = i1; i < i2; ++i) {
         for (j = j1; j < j2; ++j) {
            for (k = k1; k < k2; ++k) {
               bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (bin_content <= content_lmt) continue;
               const num = Math.round(bin_content*coef);

               for (let n = 0; n < num; ++n) {
                  const binx = histo.fXaxis.GetBinCoord(i + rnd.random()),
                      biny = histo.fYaxis.GetBinCoord(j + rnd.random()),
                      binz = histo.fZaxis.GetBinCoord(k + rnd.random());

                  // remember bin index for tooltip
                  bins[nbin++] = histo.getBin(i+1, j+1, k+1);

                  pnts.addPoint(fp.grx(binx), fp.gry(biny), fp.grz(binz));
               }
            }
         }
      }

      return pnts.createPoints({ color: this.getColor(histo.fMarkerColor) }).then(mesh => {
         fp.add3DMesh(mesh);

         mesh.bins = bins;
         mesh.painter = this;
         mesh.tip_color = histo.fMarkerColor === 3 ? 0xFF0000 : 0x00FF00;

         mesh.tooltip = function(intersect) {
            const indx = Math.floor(intersect.index / this.nvertex);
            if ((indx < 0) || (indx >= this.bins.length)) return null;

            const p = this.painter,
                  thisto = p.getHisto(),
                  tip = p.get3DToolTip(this.bins[indx]);

            tip.x1 = fp.grx(thisto.fXaxis.GetBinLowEdge(tip.ix));
            tip.x2 = fp.grx(thisto.fXaxis.GetBinLowEdge(tip.ix+1));
            tip.y1 = fp.gry(thisto.fYaxis.GetBinLowEdge(tip.iy));
            tip.y2 = fp.gry(thisto.fYaxis.GetBinLowEdge(tip.iy+1));
            tip.z1 = fp.grz(thisto.fZaxis.GetBinLowEdge(tip.iz));
            tip.z2 = fp.grz(thisto.fZaxis.GetBinLowEdge(tip.iz+1));
            tip.color = this.tip_color;
            tip.opacity = 0.3;

            return tip;
         };

         return true;
      });
   }

   /** @summary Drawing of 3D histogram */
   async draw3DBins() {
      if (!this.draw_content)
         return false;

      const o = this.getOptions();

      let box_option = o.BoxStyle;

      if (!box_option && o.Scat) {
         const promise = this.draw3DScatter();
         if (promise !== false) return promise;
         box_option = 12; // fall back to box2 draw option
      } else if (!box_option && !o.GLBox && !o.GLColor && !o.Lego)
         box_option = 12; // default draw option

      const histo = this.getHisto(),
            fp = this.getFramePainter();

      let use_lambert = false,
          use_helper = false, use_colors = false, use_opacity = 1, exclude_content = -1,
          logv = this.getPadPainter()?.getRootPad()?.fLogv,
          use_scale = true, scale_offset = 0,
          fillcolor = this.getColor(histo.fFillColor),
          tipscale = 0.5, single_bin_geom;

      if (!box_option && o.Lego)
         box_option = (o.Lego === 1) ? 10 : o.Lego;

      if ((o.GLBox === 11) || (o.GLBox === 12)) {
         tipscale = 0.4;
         use_lambert = true;
         if (o.GLBox === 12)
            use_colors = true;

         single_bin_geom = new THREE.SphereGeometry(0.5, fp.webgl ? 16 : 8, fp.webgl ? 12 : 6);
         single_bin_geom.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI/2));
         single_bin_geom.computeVertexNormals();
      } else {
         const indicies = Box3D.Indexes,
               normals = Box3D.Normals,
               vertices = Box3D.Vertices,
               buffer_size = indicies.length*3,
               single_bin_verts = new Float32Array(buffer_size),
               single_bin_norms = new Float32Array(buffer_size);

         for (let k = 0, nn = -3; k < indicies.length; ++k) {
            const vert = vertices[indicies[k]];
            single_bin_verts[k*3] = vert.x-0.5;
            single_bin_verts[k*3+1] = vert.y-0.5;
            single_bin_verts[k*3+2] = vert.z-0.5;

            if (k%6 === 0) nn+=3;
            single_bin_norms[k*3] = normals[nn];
            single_bin_norms[k*3+1] = normals[nn+1];
            single_bin_norms[k*3+2] = normals[nn+2];
         }
         use_helper = true;

         if (box_option === 12)
            use_colors = true;
         else if (box_option === 13) {
            use_colors = true;
            use_helper = false;
         } else if (o.GLColor) {
            use_colors = true;
            use_opacity = 0.5;
            use_scale = false;
            use_helper = false;
            exclude_content = 0;
            use_lambert = true;
         }

         single_bin_geom = new THREE.BufferGeometry();
         single_bin_geom.setAttribute('position', new THREE.BufferAttribute(single_bin_verts, 3));
         single_bin_geom.setAttribute('normal', new THREE.BufferAttribute(single_bin_norms, 3));
      }

      this.#box_option = box_option;

      if (use_scale && logv) {
         if (this.gminposbin && (this.gmaxbin > this.gminposbin)) {
            scale_offset = Math.log(this.gminposbin) - 0.1;
            use_scale = 1/(Math.log(this.gmaxbin) - scale_offset);
         } else {
            logv = 0;
            use_scale = 1;
         }
      } else if (use_scale)
         use_scale = (this.gminbin || this.gmaxbin) ? 1 / Math.max(Math.abs(this.gminbin), Math.abs(this.gmaxbin)) : 1;

      const get_bin_weight = content => {
         if ((exclude_content >= 0) && (content < exclude_content)) return 0;
         if (!use_scale) return 1;
         if (logv) {
            if (content <= 0) return 0;
            content = Math.log(content) - scale_offset;
         }
         return Math.pow(Math.abs(content*use_scale), 0.3333);
      }, i1 = this.getSelectIndex('x', 'left', 0.5),
         i2 = this.getSelectIndex('x', 'right', 0),
         j1 = this.getSelectIndex('y', 'left', 0.5),
         j2 = this.getSelectIndex('y', 'right', 0),
         k1 = this.getSelectIndex('z', 'left', 0.5),
         k2 = this.getSelectIndex('z', 'right', 0);

      if ((i2 <= i1) || (j2 <= j1) || (k2 <= k1))
         return false;

      const cntr = use_colors ? this.getContour() : null,
            palette = use_colors ? this.getHistPalette() : null,
            bins_matrixes = [], bins_colors = [], bins_ids = [], negative_matrixes = [], bin_opacities = [],
            transfer = (this.transferFunc && proivdeEvalPar(this.transferFunc, true)) ? this.transferFunc : null;

      for (let i = i1; i < i2; ++i) {
         const grx1 = fp.grx(histo.fXaxis.GetBinLowEdge(i+1)),
               grx2 = fp.grx(histo.fXaxis.GetBinLowEdge(i+2));
         for (let j = j1; j < j2; ++j) {
            const gry1 = fp.gry(histo.fYaxis.GetBinLowEdge(j+1)),
                  gry2 = fp.gry(histo.fYaxis.GetBinLowEdge(j+2));
            for (let k = k1; k < k2; ++k) {
               const bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (!o.GLColor && ((bin_content === 0) || (bin_content < this.gminbin))) continue;

               const wei = get_bin_weight(bin_content);
               if (wei < 1e-3) continue; // do not show very small bins

               if (use_colors) {
                  const colindx = cntr.getPaletteIndex(palette, bin_content);
                  if (colindx === null)
                     continue;
                  bins_colors.push(palette.getColor(colindx));
                  if (transfer) {
                     const op = getTF1Value(transfer, bin_content, false) * 3;
                     bin_opacities.push((!op || op < 0) ? 0 : (op > 1 ? 1 : op));
                  }
               }

               const grz1 = fp.grz(histo.fZaxis.GetBinLowEdge(k+1)),
                     grz2 = fp.grz(histo.fZaxis.GetBinLowEdge(k+2));

               // remember bin index for tooltip
               bins_ids.push(histo.getBin(i+1, j+1, k+1));

               const bin_matrix = new THREE.Matrix4();
               bin_matrix.scale(new THREE.Vector3((grx2 - grx1) * wei, (gry2 - gry1) * wei, (grz2 - grz1) * wei));
               bin_matrix.setPosition((grx2 + grx1) / 2, (gry2 + gry1) / 2, (grz2 + grz1) / 2);
               bins_matrixes.push(bin_matrix);
               if (bin_content < 0)
                  negative_matrixes.push(bin_matrix);
            }
         }
      }

      function getBinTooltip(intersect) {
         let binid = this.binid;

         if (binid === undefined) {
            if ((intersect.instanceId === undefined) || (intersect.instanceId >= this.bins.length))
               return;
            binid = this.bins[intersect.instanceId];
         }

         const p = this.painter,
               thisto = p.getHisto(),
               tip = p.get3DToolTip(binid),
               grx1 = fp.grx(thisto.fXaxis.GetBinCoord(tip.ix-1)),
               grx2 = fp.grx(thisto.fXaxis.GetBinCoord(tip.ix)),
               gry1 = fp.gry(thisto.fYaxis.GetBinCoord(tip.iy-1)),
               gry2 = fp.gry(thisto.fYaxis.GetBinCoord(tip.iy)),
               grz1 = fp.grz(thisto.fZaxis.GetBinCoord(tip.iz-1)),
               grz2 = fp.grz(thisto.fZaxis.GetBinCoord(tip.iz)),
               wei2 = this.get_weight(tip.value) * this.tipscale;

         tip.x1 = (grx2 + grx1) / 2 - (grx2 - grx1) * wei2;
         tip.x2 = (grx2 + grx1) / 2 + (grx2 - grx1) * wei2;
         tip.y1 = (gry2 + gry1) / 2 - (gry2 - gry1) * wei2;
         tip.y2 = (gry2 + gry1) / 2 + (gry2 - gry1) * wei2;
         tip.z1 = (grz2 + grz1) / 2 - (grz2 - grz1) * wei2;
         tip.z2 = (grz2 + grz1) / 2 + (grz2 - grz1) * wei2;
         tip.color = this.tip_color;

         return tip;
      }

      if (use_colors && (transfer || (use_opacity !== 1))) {
         // create individual meshes for each bin
         for (let n = 0; n < bins_matrixes.length; ++n) {
            const opacity = transfer ? bin_opacities[n] : use_opacity,
                  color = new THREE.Color(bins_colors[n]),
                  material = use_lambert ? new THREE.MeshLambertMaterial({ color, opacity, transparent: opacity < 1, vertexColors: false })
                                         : new THREE.MeshBasicMaterial({ color, opacity, transparent: opacity < 1, vertexColors: false }),
                  bin_mesh = new THREE.Mesh(single_bin_geom, material);

            bin_mesh.applyMatrix4(bins_matrixes[n]);

            bin_mesh.painter = this;
            bin_mesh.binid = bins_ids[n];
            bin_mesh.tipscale = tipscale;
            bin_mesh.tip_color = (histo.fFillColor === 3) ? 0xFF0000 : 0x00FF00;
            bin_mesh.get_weight = get_bin_weight;
            bin_mesh.tooltip = getBinTooltip;

            fp.add3DMesh(bin_mesh);
         }
      } else {
         if (use_colors)
            fillcolor = new THREE.Color(1, 1, 1);

         const material = use_lambert ? new THREE.MeshLambertMaterial({ color: fillcolor, vertexColors: false })
                                      : new THREE.MeshBasicMaterial({ color: fillcolor, vertexColors: false }),
               all_bins_mesh = new THREE.InstancedMesh(single_bin_geom, material, bins_matrixes.length);

         for (let n = 0; n < bins_matrixes.length; ++n) {
            all_bins_mesh.setMatrixAt(n, bins_matrixes[n]);
            if (use_colors)
               all_bins_mesh.setColorAt(n, new THREE.Color(bins_colors[n]));
         }

         all_bins_mesh.painter = this;
         all_bins_mesh.bins = bins_ids;
         all_bins_mesh.tipscale = tipscale;
         all_bins_mesh.tip_color = (histo.fFillColor === 3) ? 0xFF0000 : 0x00FF00;
         all_bins_mesh.get_weight = get_bin_weight;
         all_bins_mesh.tooltip = getBinTooltip;

         fp.add3DMesh(all_bins_mesh);
      }

      if (use_helper) {
         const helper_material = new THREE.LineBasicMaterial({ color: this.getColor(histo.fLineColor) });
         function addLines(segments, matrixes) {
            if (!matrixes)
               return;
            const positions = new Float32Array(matrixes.length * segments.length * 3);
            for (let i = 0, vvv = 0; i < matrixes.length; ++i) {
               const m = matrixes[i].elements;
               for (let n = 0; n < segments.length; ++n, vvv += 3) {
                  const vert = Box3D.Vertices[segments[n]];
                  positions[vvv] = m[12] + (vert.x - 0.5) * m[0];
                  positions[vvv+1] = m[13] + (vert.y - 0.5) * m[5];
                  positions[vvv+2] = m[14] + (vert.z - 0.5) * m[10];
               }
            }
            fp.add3DMesh(createLineSegments(positions, helper_material));
         };

         addLines(Box3D.Segments, bins_matrixes);
         addLines(Box3D.Crosses, negative_matrixes);
      }

      return true;
   }


   /** @summary Redraw TH3 histogram */
   async redraw(reason) {
      const fp = this.getFramePainter(), // who makes axis and 3D drawing
            histo = this.getHisto(),
            o = this.getOptions();

      let pr = Promise.resolve(true), full_draw = true;

      if (reason === 'resize') {
         const res = fp.resize3D();
         if (res !== 1) {
            full_draw = false;
            if (res)
               fp.render3D();
         }
      }

      if (full_draw) {
         assignFrame3DMethods(fp);
         pr = fp.create3DScene(o.Render3D, o.x3dscale, o.y3dscale, o.Ortho).then(() => {
            fp.setAxesRanges(histo.fXaxis, this.xmin, this.xmax, histo.fYaxis, this.ymin, this.ymax, histo.fZaxis, this.zmin, this.zmax, this);
            fp.set3DOptions(o);
            fp.drawXYZ(fp.toplevel, TAxisPainter, {
               ndim: 3, hist_painter: this, zoom: settings.Zooming,
               draw: o.Axis !== -1, drawany: o.isCartesian()
            });
            return this.draw3DBins();
         }).then(() => {
            fp.render3D();
            this.updateStatWebCanvas();
            fp.addKeysHandler();
         });
      }

      if (this.isMainPainter())
        pr = pr.then(() => this.drawColorPalette(o.Zscale && (this.#box_option === 12 || this.#box_option === 13 || o.GLBox === 12)));

      return pr.then(() => this.updateFunctions())
               .then(() => this.updateHistTitle())
               .then(() => this);
   }

   /** @summary Fill pad toolbar with TH3-related functions */
   fillToolbar() {
      const pp = this.getPadPainter();
      if (!pp) return;

      pp.addPadButton('auto_zoom', 'Unzoom all axes', 'ToggleZoom', 'Ctrl *');
      if (this.draw_content)
         pp.addPadButton('statbox', 'Toggle stat box', 'ToggleStatBox');
      pp.addPadButton('th2colorz', 'Toggle color palette', 'ToggleColorZ');
      pp.showPadButtons();
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis, min, max) {
      let obj = this.getHisto();
      if (obj) obj = obj[`f${axis.toUpperCase()}axis`];
      return !obj || (obj.FindBin(max, 0.5) - obj.FindBin(min, 0) > 1);
   }

   /** @summary Perform automatic zoom inside non-zero region of histogram */
   autoZoom() {
      const i1 = this.getSelectIndex('x', 'left'),
            i2 = this.getSelectIndex('x', 'right'),
            j1 = this.getSelectIndex('y', 'left'),
            j2 = this.getSelectIndex('y', 'right'),
            k1 = this.getSelectIndex('z', 'left'),
            k2 = this.getSelectIndex('z', 'right'),
            histo = this.getObject();
      let i, j, k;

      if ((i1 === i2) || (j1 === j2) || (k1 === k2)) return;

      // first find minimum
      let min = histo.getBinContent(i1+1, j1+1, k1+1);
      for (i = i1; i < i2; ++i) {
         for (j = j1; j < j2; ++j) {
            for (k = k1; k < k2; ++k)
               min = Math.min(min, histo.getBinContent(i+1, j+1, k+1));
         }
      }
      if (min > 0) return; // if all points positive, no chance for auto-scale

      let ileft = i2, iright = i1, jleft = j2, jright = j1, kleft = k2, kright = k1;

      for (i = i1; i < i2; ++i) {
         for (j = j1; j < j2; ++j) {
            for (k = k1; k < k2; ++k) {
               if (histo.getBinContent(i+1, j+1, k+1) > min) {
                  if (i < ileft) ileft = i;
                  if (i >= iright) iright = i + 1;
                  if (j < jleft) jleft = j;
                  if (j >= jright) jright = j + 1;
                  if (k < kleft) kleft = k;
                  if (k >= kright) kright = k + 1;
               }
            }
         }
      }

      let xmin, xmax, ymin, ymax, zmin, zmax, isany = false;

      if ((ileft === iright-1) && (ileft > i1+1) && (iright < i2-1)) { ileft--; iright++; }
      if ((jleft === jright-1) && (jleft > j1+1) && (jright < j2-1)) { jleft--; jright++; }
      if ((kleft === kright-1) && (kleft > k1+1) && (kright < k2-1)) { kleft--; kright++; }

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

      if ((kleft > k1 || kright < k2) && (kleft < kright - 1)) {
         zmin = histo.fZaxis.GetBinLowEdge(kleft+1);
         zmax = histo.fZaxis.GetBinLowEdge(kright+1);
         isany = true;
      }

      if (isany)
         return this.getFramePainter().zoom(xmin, xmax, ymin, ymax, zmin, zmax);
   }

   /** @summary Fill histogram context menu */
   fillHistContextMenu(menu) {
      const opts = this.getSupportedDrawOptions();

      menu.addDrawMenu('Draw with', opts, arg => {
         if (arg.indexOf(kInspect) === 0)
            return this.showInspector(arg);

         this.decodeOptions(arg);

         this.interactiveRedraw(true, 'drawopt');
      });
   }

   /** @summary draw TH3 object */
   static async draw(dom, histo, opt) {
      const painter = new TH3Painter(dom, histo);
      painter.mode3d = true;

      return ensureTCanvas(painter, '3d').then(() => {
         painter.setAsMainPainter();
         painter.decodeOptions(opt);
         painter.checkPadRange();
         painter.scanContent();
         painter.createStat(); // only when required
         return painter.redraw();
      })
      .then(() => painter.drawFunctions())
      .then(() => {
         painter.fillToolbar();
         return painter;
      });
   }

} // class TH3Painter

export { TH3Painter };
