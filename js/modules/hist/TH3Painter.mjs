import { gStyle, settings } from '../core.mjs';
import { REVISION, Matrix4,
         BufferGeometry, BufferAttribute, Mesh, MeshBasicMaterial, MeshLambertMaterial,
         LineBasicMaterial, SphereGeometry } from '../three.mjs';
import { TRandom, floatToString } from '../base/BasePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { TAxisPainter } from '../gpad/TAxisPainter.mjs';
import { createLineSegments, PointsCreator, Box3D } from '../base/base3d.mjs';
import { TPavePainter } from '../hist/TPavePainter.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';
import { assignFrame3DMethods } from './hist3d.mjs';


/**
 * @summary Painter for TH3 classes
 * @private
 */

class TH3Painter extends THistPainter {

   /** @summary Scan TH3 histogram content */
   scanContent(when_axis_changed) {

      // no need to rescan histogram while result does not depend from axis selection
      if (when_axis_changed && this.nbinsx && this.nbinsy && this.nbinsz) return;

      let histo = this.getHisto();

      this.extractAxesProperties(3);

      // global min/max, used at the moment in 3D drawing
      this.gminbin = this.gmaxbin = histo.getBinContent(1,1,1);

      for (let i = 0; i < this.nbinsx; ++i)
         for (let j = 0; j < this.nbinsy; ++j)
            for (let k = 0; k < this.nbinsz; ++k) {
               let bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (bin_content < this.gminbin) this.gminbin = bin_content; else
               if (bin_content > this.gmaxbin) this.gmaxbin = bin_content;
            }

      this.draw_content = this.gmaxbin > 0;
   }

   /** @summary Count TH3 statistic */
   countStat() {
      let histo = this.getHisto(), xaxis = histo.fXaxis, yaxis = histo.fYaxis, zaxis = histo.fZaxis,
          stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0,
          stat_sumz1 = 0, stat_sumx2 = 0, stat_sumy2 = 0, stat_sumz2 = 0,
          i1 = this.getSelectIndex("x", "left"),
          i2 = this.getSelectIndex("x", "right"),
          j1 = this.getSelectIndex("y", "left"),
          j2 = this.getSelectIndex("y", "right"),
          k1 = this.getSelectIndex("z", "left"),
          k2 = this.getSelectIndex("z", "right"),
          fp = this.getFramePainter(),
          res = { name: histo.fName, entries: 0, integral: 0, meanx: 0, meany: 0, meanz: 0, rmsx: 0, rmsy: 0, rmsz: 0 },
          xi, yi, zi, xx, xside, yy, yside, zz, zside, cont;

      for (xi = 0; xi < this.nbinsx+2; ++xi) {

         xx = xaxis.GetBinCoord(xi - 0.5);
         xside = (xi < i1) ? 0 : (xi > i2 ? 2 : 1);

         for (yi = 0; yi < this.nbinsy+2; ++yi) {

            yy = yaxis.GetBinCoord(yi - 0.5);
            yside = (yi < j1) ? 0 : (yi > j2 ? 2 : 1);

            for (zi = 0; zi < this.nbinsz+2; ++zi) {

               zz = zaxis.GetBinCoord(zi - 0.5);
               zside = (zi < k1) ? 0 : (zi > k2 ? 2 : 1);

               cont = histo.getBinContent(xi, yi, zi);
               res.entries += cont;

               if ((xside==1) && (yside==1) && (zside==1)) {
                  stat_sum0 += cont;
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

      if ((histo.fTsumw > 0) && !fp.isAxisZoomed("x") && !fp.isAxisZoomed("y") && !fp.isAxisZoomed("z")) {
         stat_sum0  = histo.fTsumw;
         stat_sumx1 = histo.fTsumwx;
         stat_sumx2 = histo.fTsumwx2;
         stat_sumy1 = histo.fTsumwy;
         stat_sumy2 = histo.fTsumwy2;
         stat_sumz1 = histo.fTsumwz;
         stat_sumz2 = histo.fTsumwz2;
      }

      if (stat_sum0 > 0) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.meanz = stat_sumz1 / stat_sum0;
         res.rmsx = Math.sqrt(Math.abs(stat_sumx2 / stat_sum0 - res.meanx * res.meanx));
         res.rmsy = Math.sqrt(Math.abs(stat_sumy2 / stat_sum0 - res.meany * res.meany));
         res.rmsz = Math.sqrt(Math.abs(stat_sumz2 / stat_sum0 - res.meanz * res.meanz));
      }

      res.integral = stat_sum0;

      if (histo.fEntries > 1) res.entries = histo.fEntries;

      return res;
   }

   /** @summary Fill TH3 statistic in stat box */
   fillStatistic(stat, dostat, dofit) {

      // no need to refill statistic if histogram is dummy
      if (this.isIgnoreStatsFill()) return false;

      let data = this.countStat(),
          print_name = dostat % 10,
          print_entries = Math.floor(dostat / 10) % 10,
          print_mean = Math.floor(dostat / 100) % 10,
          print_rms = Math.floor(dostat / 1000) % 10,
          print_integral = Math.floor(dostat / 1000000) % 10;
          // print_under = Math.floor(dostat / 10000) % 10,
          // print_over = Math.floor(dostat / 100000) % 10,
          // print_skew = Math.floor(dostat / 10000000) % 10,
          // print_kurt = Math.floor(dostat / 100000000) % 10;

      stat.clearPave();

      if (print_name > 0)
         stat.addText(data.name);

      if (print_entries > 0)
         stat.addText("Entries = " + stat.format(data.entries,"entries"));

      if (print_mean > 0) {
         stat.addText("Mean x = " + stat.format(data.meanx));
         stat.addText("Mean y = " + stat.format(data.meany));
         stat.addText("Mean z = " + stat.format(data.meanz));
      }

      if (print_rms > 0) {
         stat.addText("Std Dev x = " + stat.format(data.rmsx));
         stat.addText("Std Dev y = " + stat.format(data.rmsy));
         stat.addText("Std Dev z = " + stat.format(data.rmsz));
      }

      if (print_integral > 0) {
         stat.addText("Integral = " + stat.format(data.integral,"entries"));
      }

      if (dofit) stat.fillFunctionStat(this.findFunction('TF3'), dofit);

      return true;
   }

   /** @summary Provide text information (tooltips) for histogram bin */
   getBinTooltips(ix, iy, iz) {
      let lines = [], histo = this.getHisto();

      lines.push(this.getObjectHint());

      lines.push("x = " + this.getAxisBinTip("x", histo.fXaxis, ix) + "  xbin=" + (ix+1));
      lines.push("y = " + this.getAxisBinTip("y", histo.fYaxis, iy) + "  ybin=" + (iy+1));
      lines.push("z = " + this.getAxisBinTip("z", histo.fZaxis, iz) + "  zbin=" + (iz+1));

      let binz = histo.getBinContent(ix+1, iy+1, iz+1);
      if (binz === Math.round(binz))
         lines.push("entries = " + binz);
      else
         lines.push("entries = " + floatToString(binz, gStyle.fStatFormat));

      return lines;
   }

   /** @summary draw 3D histogram as scatter plot
     * @desc If there are too many points, box will be displayed */
   draw3DScatter() {

      let histo = this.getObject(),
          main = this.getFramePainter(),
          i1 = this.getSelectIndex("x", "left", 0.5),
          i2 = this.getSelectIndex("x", "right", 0),
          j1 = this.getSelectIndex("y", "left", 0.5),
          j2 = this.getSelectIndex("y", "right", 0),
          k1 = this.getSelectIndex("z", "left", 0.5),
          k2 = this.getSelectIndex("z", "right", 0),
          i, j, k, bin_content;

      if ((i2 <= i1) || (j2 <= j1) || (k2 <= k1))
         return Promise.resolve(true);

      // scale down factor if too large values
      let coef = (this.gmaxbin > 1000) ? 1000/this.gmaxbin : 1,
          numpixels = 0, sumz = 0, content_lmt = Math.max(0, this.gminbin);

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
      if (numpixels > (main.webgl ? 100000 : 30000))
         return false;

      let pnts = new PointsCreator(numpixels, main.webgl, main.size_x3d/200),
          bins = new Int32Array(numpixels), nbin = 0,
          rnd = new TRandom(sumz);

      for (i = i1; i < i2; ++i) {
         for (j = j1; j < j2; ++j) {
            for (k = k1; k < k2; ++k) {
               bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (bin_content <= content_lmt) continue;
               let num = Math.round(bin_content*coef);

               for (let n=0;n<num;++n) {
                  let binx = histo.fXaxis.GetBinCoord(i + rnd.random()),
                      biny = histo.fYaxis.GetBinCoord(j + rnd.random()),
                      binz = histo.fZaxis.GetBinCoord(k + rnd.random());

                  // remember bin index for tooltip
                  bins[nbin++] = histo.getBin(i+1, j+1, k+1);

                  pnts.addPoint(main.grx(binx), main.gry(biny), main.grz(binz));
               }
            }
         }
      }

      return pnts.createPoints({ color: this.getColor(histo.fMarkerColor) }).then(mesh => {
         main.toplevel.add(mesh);

         mesh.bins = bins;
         mesh.painter = this;
         mesh.tip_color = (histo.fMarkerColor===3) ? 0xFF0000 : 0x00FF00;

         mesh.tooltip = function(intersect) {
            if (!Number.isInteger(intersect.index)) {
               console.error(`intersect.index not provided, three.js version ${REVISION}`);
               return null;
            }

            let indx = Math.floor(intersect.index / this.nvertex);
            if ((indx < 0) || (indx >= this.bins.length)) return null;

            let p = this.painter, histo = p.getHisto(),
                main = p.getFramePainter(),
                tip = p.get3DToolTip(this.bins[indx]);

            tip.x1 = main.grx(histo.fXaxis.GetBinLowEdge(tip.ix));
            tip.x2 = main.grx(histo.fXaxis.GetBinLowEdge(tip.ix+1));
            tip.y1 = main.gry(histo.fYaxis.GetBinLowEdge(tip.iy));
            tip.y2 = main.gry(histo.fYaxis.GetBinLowEdge(tip.iy+1));
            tip.z1 = main.grz(histo.fZaxis.GetBinLowEdge(tip.iz));
            tip.z2 = main.grz(histo.fZaxis.GetBinLowEdge(tip.iz+1));
            tip.color = this.tip_color;
            tip.opacity = 0.3;

            return tip;
         };

         return true;
      });
   }

   /** @summary Drawing of 3D histogram */
   draw3DBins() {

      if (!this.draw_content)
         return Promise.resolve(false);

      let box_option = this.options.Box ? this.options.BoxStyle : 0;

      if (!box_option && !this.options.GLBox && !this.options.GLColor && !this.options.Lego) {
         let promise = this.draw3DScatter();
         if (promise !== false) return promise;
         box_option = 12; // fall back to box2 draw option
      }

      let histo = this.getHisto(),
          fillcolor = this.getColor(histo.fFillColor),
          main = this.getFramePainter(),
          buffer_size = 0, use_lambert = false,
          use_helper = false, use_colors = false, use_opacity = 1, use_scale = true,
          single_bin_verts, single_bin_norms,

          tipscale = 0.5;

      if (!box_option && this.options.Lego)
         box_option = (this.options.Lego===1) ? 10 : this.options.Lego;

      if ((this.options.GLBox === 11) || (this.options.GLBox === 12)) {

         tipscale = 0.4;
         use_lambert = true;
         if (this.options.GLBox === 12) use_colors = true;

         let geom = main.webgl ? new SphereGeometry(0.5, 16, 12) : new SphereGeometry(0.5, 8, 6);
         geom.applyMatrix4( new Matrix4().makeRotationX( Math.PI / 2 ) );
         geom.computeVertexNormals();

         let indx = geom.getIndex().array,
             pos = geom.getAttribute('position').array,
             norm = geom.getAttribute('normal').array;

         buffer_size = indx.length*3;
         single_bin_verts = new Float32Array(buffer_size);
         single_bin_norms = new Float32Array(buffer_size);

         for (let k = 0; k < indx.length; ++k) {
            let iii = indx[k]*3;
            single_bin_verts[k*3] = pos[iii];
            single_bin_verts[k*3+1] = pos[iii+1];
            single_bin_verts[k*3+2] = pos[iii+2];
            single_bin_norms[k*3] = norm[iii];
            single_bin_norms[k*3+1] = norm[iii+1];
            single_bin_norms[k*3+2] = norm[iii+2];
         }

      } else {

         let indicies = Box3D.Indexes,
             normals = Box3D.Normals,
             vertices = Box3D.Vertices;

         buffer_size = indicies.length*3;
         single_bin_verts = new Float32Array(buffer_size);
         single_bin_norms = new Float32Array(buffer_size);

         for (let k = 0, nn = -3; k < indicies.length; ++k) {
            let vert = vertices[indicies[k]];
            single_bin_verts[k*3]   = vert.x-0.5;
            single_bin_verts[k*3+1] = vert.y-0.5;
            single_bin_verts[k*3+2] = vert.z-0.5;

            if (k%6===0) nn+=3;
            single_bin_norms[k*3]   = normals[nn];
            single_bin_norms[k*3+1] = normals[nn+1];
            single_bin_norms[k*3+2] = normals[nn+2];
         }
         use_helper = true;

         if (box_option===12) { use_colors = true; } else
         if (box_option===13) { use_colors = true; use_helper = false; }  else
         if (this.options.GLColor) { use_colors = true; use_opacity = 0.5; use_scale = false; use_helper = false; use_lambert = true; }
      }

      if (use_scale)
         use_scale = (this.gminbin || this.gmaxbin) ? 1 / Math.max(Math.abs(this.gminbin), Math.abs(this.gmaxbin)) : 1;

      let i1 = this.getSelectIndex("x", "left", 0.5),
          i2 = this.getSelectIndex("x", "right", 0),
          j1 = this.getSelectIndex("y", "left", 0.5),
          j2 = this.getSelectIndex("y", "right", 0),
          k1 = this.getSelectIndex("z", "left", 0.5),
          k2 = this.getSelectIndex("z", "right", 0);

      if ((i2 <= i1) || (j2 <= j1) || (k2 <= k1))
         return Promise.resolve(false);

      let scalex = (main.grx(histo.fXaxis.GetBinLowEdge(i2+1)) - main.grx(histo.fXaxis.GetBinLowEdge(i1+1))) / (i2-i1),
          scaley = (main.gry(histo.fYaxis.GetBinLowEdge(j2+1)) - main.gry(histo.fYaxis.GetBinLowEdge(j1+1))) / (j2-j1),
          scalez = (main.grz(histo.fZaxis.GetBinLowEdge(k2+1)) - main.grz(histo.fZaxis.GetBinLowEdge(k1+1))) / (k2-k1);

      let nbins = 0, i, j, k, wei, bin_content, cols_size = [], num_colors = 0, cols_sequence = [],
          cntr = use_colors ? this.getContour() : null,
          palette = use_colors ? this.getHistPalette() : null;

      for (i = i1; i < i2; ++i) {
         for (j = j1; j < j2; ++j) {
            for (k = k1; k < k2; ++k) {
               bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (!this.options.GLColor && ((bin_content===0) || (bin_content < this.gminbin))) continue;
               wei = use_scale ? Math.pow(Math.abs(bin_content*use_scale), 0.3333) : 1;
               if (wei < 1e-3) continue; // do not draw empty or very small bins

               nbins++;

               if (!use_colors) continue;

               let colindx = cntr.getPaletteIndex(palette, bin_content);
               if (colindx !== null) {
                  if (cols_size[colindx] === undefined) {
                     cols_size[colindx] = 0;
                     cols_sequence[colindx] = num_colors++;
                  }
                  cols_size[colindx]+=1;
               } else {
                  console.error('not found color for', bin_content);
               }
            }
         }
      }

      if (!use_colors) {
         cols_size.push(nbins);
         num_colors = 1;
         cols_sequence = [0];
      }

      const cols_nbins = new Array(num_colors),
            bin_verts = new Array(num_colors),
            bin_norms = new Array(num_colors),
            bin_tooltips = new Array(num_colors),
            helper_kind = new Array(num_colors),
            helper_indexes = new Array(num_colors),  // helper_kind == 1, use original vertices
            helper_positions = new Array(num_colors);  // helper_kind == 2, all vertices copied into separate buffer

      for(let ncol = 0; ncol < cols_size.length; ++ncol) {
         if (!cols_size[ncol]) continue; // ignore dummy colors

         nbins = cols_size[ncol]; // how many bins with specified color
         let nseq = cols_sequence[ncol];

         cols_nbins[nseq] = helper_kind[nseq] = 0; // counter for the filled bins

         // 1 - use same vertices to create helper, one can use maximal 64K vertices
         // 2 - all vertices copied into separate buffer
         if (use_helper)
            helper_kind[nseq] = (nbins * buffer_size / 3 > 0xFFF0) ? 2 : 1;

         bin_verts[nseq] = new Float32Array(nbins * buffer_size);
         bin_norms[nseq] = new Float32Array(nbins * buffer_size);
         bin_tooltips[nseq] = new Int32Array(nbins);

         if (helper_kind[nseq] === 1)
            helper_indexes[nseq] = new Uint16Array(nbins * Box3D.MeshSegments.length);

         if (helper_kind[nseq] === 2)
            helper_positions[nseq] = new Float32Array(nbins * Box3D.Segments.length * 3);
      }

      let binx, grx, biny, gry, binz, grz;

      for (i = i1; i < i2; ++i) {
         binx = histo.fXaxis.GetBinCenter(i+1); grx = main.grx(binx);
         for (j = j1; j < j2; ++j) {
            biny = histo.fYaxis.GetBinCenter(j+1); gry = main.gry(biny);
            for (k = k1; k < k2; ++k) {
               bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (!this.options.GLColor && ((bin_content===0) || (bin_content < this.gminbin))) continue;

               wei = use_scale ? Math.pow(Math.abs(bin_content*use_scale), 0.3333) : 1;
               if (wei < 1e-3) continue; // do not show very small bins

               let nseq = 0;
               if (use_colors) {
                  let colindx = cntr.getPaletteIndex(palette, bin_content);
                  if (colindx === null) continue;
                  nseq = cols_sequence[colindx];
               }

               nbins = cols_nbins[nseq];

               binz = histo.fZaxis.GetBinCenter(k+1); grz = main.grz(binz);

               // remember bin index for tooltip
               bin_tooltips[nseq][nbins] = histo.getBin(i+1, j+1, k+1);

               let vvv = nbins * buffer_size, bin_v = bin_verts[nseq], bin_n = bin_norms[nseq];

               // Grab the coordinates and scale that are being assigned to each bin
               for (let vi = 0; vi < buffer_size; vi+=3, vvv+=3) {
                  bin_v[vvv]   = grx + single_bin_verts[vi]*scalex*wei;
                  bin_v[vvv+1] = gry + single_bin_verts[vi+1]*scaley*wei;
                  bin_v[vvv+2] = grz + single_bin_verts[vi+2]*scalez*wei;

                  bin_n[vvv]   = single_bin_norms[vi];
                  bin_n[vvv+1] = single_bin_norms[vi+1];
                  bin_n[vvv+2] = single_bin_norms[vi+2];
               }

               if (helper_kind[nseq]===1) {
                  // reuse vertices created for the mesh
                  let helper_segments = Box3D.MeshSegments;
                  vvv = nbins * helper_segments.length;
                  let shift = Math.round(nbins * buffer_size/3),
                      helper_i = helper_indexes[nseq];
                  for (let n=0;n<helper_segments.length;++n)
                     helper_i[vvv+n] = shift + helper_segments[n];
               }

               if (helper_kind[nseq]===2) {
                  let helper_segments = Box3D.Segments,
                      helper_p = helper_positions[nseq];
                  vvv = nbins * helper_segments.length * 3;
                  for (let n=0;n<helper_segments.length;++n, vvv+=3) {
                     let vert = Box3D.Vertices[helper_segments[n]];
                     helper_p[vvv]   = grx + (vert.x-0.5)*scalex*wei;
                     helper_p[vvv+1] = gry + (vert.y-0.5)*scaley*wei;
                     helper_p[vvv+2] = grz + (vert.z-0.5)*scalez*wei;
                  }
               }

               cols_nbins[nseq] = nbins+1;
            }
         }
      }

      for(let ncol = 0; ncol < cols_size.length; ++ncol) {
         if (!cols_size[ncol]) continue; // ignore dummy colors

         let nseq = cols_sequence[ncol];

         // BufferGeometries that store geometry of all bins
         let all_bins_buffgeom = new BufferGeometry();

         // Create mesh from bin buffergeometry
         all_bins_buffgeom.setAttribute('position', new BufferAttribute(bin_verts[nseq], 3));
         all_bins_buffgeom.setAttribute('normal', new BufferAttribute(bin_norms[nseq], 3));

         if (use_colors) fillcolor = this.fPalette.getColor(ncol);

         const material = use_lambert ? new MeshLambertMaterial({ color: fillcolor, opacity: use_opacity, transparent: (use_opacity < 1), vertexColors: false })
                                      : new MeshBasicMaterial({ color: fillcolor, opacity: use_opacity, vertexColors: false }),
              combined_bins = new Mesh(all_bins_buffgeom, material);

         combined_bins.bins = bin_tooltips[nseq];
         combined_bins.bins_faces = buffer_size/9;
         combined_bins.painter = this;

         combined_bins.scalex = tipscale*scalex;
         combined_bins.scaley = tipscale*scaley;
         combined_bins.scalez = tipscale*scalez;
         combined_bins.tip_color = (histo.fFillColor === 3) ? 0xFF0000 : 0x00FF00;
         combined_bins.use_scale = use_scale;

         combined_bins.tooltip = function(intersect) {
            if (!Number.isInteger(intersect.faceIndex)) {
               console.error(`intersect.faceIndex not provided, three.js version ${REVISION}`);
               return null;
            }
            let indx = Math.floor(intersect.faceIndex / this.bins_faces);
            if ((indx<0) || (indx >= this.bins.length)) return null;

            let p = this.painter,
                histo = p.getHisto(),
                main = p.getFramePainter(),
                tip = p.get3DToolTip(this.bins[indx]),
                grx = main.grx(histo.fXaxis.GetBinCoord(tip.ix-0.5)),
                gry = main.gry(histo.fYaxis.GetBinCoord(tip.iy-0.5)),
                grz = main.grz(histo.fZaxis.GetBinCoord(tip.iz-0.5)),
                wei = this.use_scale ? Math.pow(Math.abs(tip.value*this.use_scale), 0.3333) : 1;

            tip.x1 = grx - this.scalex*wei; tip.x2 = grx + this.scalex*wei;
            tip.y1 = gry - this.scaley*wei; tip.y2 = gry + this.scaley*wei;
            tip.z1 = grz - this.scalez*wei; tip.z2 = grz + this.scalez*wei;

            tip.color = this.tip_color;

            return tip;
         };

         main.toplevel.add(combined_bins);

         if (helper_kind[nseq] > 0) {
            let lcolor = this.getColor(histo.fLineColor),
                helper_material = new LineBasicMaterial({ color: lcolor }),
                lines = null;

            if (helper_kind[nseq] === 1) {
               // reuse positions from the mesh - only special index was created
               lines = createLineSegments(bin_verts[nseq], helper_material, helper_indexes[nseq]);
            } else {
               lines = createLineSegments(helper_positions[nseq], helper_material);
            }

            main.toplevel.add(lines);
         }
      }

      return Promise.resolve(true);
   }

   /** @summary Redraw TH3 histogram */
   redraw(reason) {

      let main = this.getFramePainter(), // who makes axis and 3D drawing
          histo = this.getHisto(),
          pr = Promise.resolve(true);

      if (reason == "resize") {

         if (main.resize3D()) main.render3D();

      } else {
         assignFrame3DMethods(main);
         pr = main.create3DScene(this.options.Render3D, this.options.x3dscale, this.options.y3dscale).then(() => {
            main.setAxesRanges(histo.fXaxis, this.xmin, this.xmax, histo.fYaxis, this.ymin, this.ymax, histo.fZaxis, this.zmin, this.zmax);
            main.set3DOptions(this.options);
            main.drawXYZ(main.toplevel, TAxisPainter, { zoom: settings.Zooming, ndim: 3, draw: this.options.Axis !== -1 });
            return this.draw3DBins();
         }).then(() => {
            main.render3D();
            this.updateStatWebCanvas();
            main.addKeysHandler();
         });
      }

      return pr.then(() => this.drawHistTitle()).then(() => this);
   }

   /** @summary Fill pad toolbar with TH3-related functions */
   fillToolbar() {
      let pp = this.getPadPainter();
      if (!pp) return;

      pp.addPadButton("auto_zoom", 'Unzoom all axes', 'ToggleZoom', "Ctrl *");
      if (this.draw_content)
         pp.addPadButton("statbox", 'Toggle stat box', "ToggleStatBox");
      pp.showPadButtons();
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis,min,max) {
      let obj = this.getHisto();
      if (obj) obj = obj["f"+axis.toUpperCase()+"axis"];
      return !obj || (obj.FindBin(max,0.5) - obj.FindBin(min,0) > 1);
   }

   /** @summary Perform automatic zoom inside non-zero region of histogram */
   autoZoom() {
      let i1 = this.getSelectIndex("x", "left"),
          i2 = this.getSelectIndex("x", "right"),
          j1 = this.getSelectIndex("y", "left"),
          j2 = this.getSelectIndex("y", "right"),
          k1 = this.getSelectIndex("z", "left"),
          k2 = this.getSelectIndex("z", "right"),
          i,j,k, histo = this.getObject();

      if ((i1 === i2) || (j1 === j2) || (k1 === k2)) return;

      // first find minimum
      let min = histo.getBinContent(i1 + 1, j1 + 1, k1+1);
      for (i = i1; i < i2; ++i)
         for (j = j1; j < j2; ++j)
            for (k = k1; k < k2; ++k)
               min = Math.min(min, histo.getBinContent(i+1, j+1, k+1));

      if (min > 0) return; // if all points positive, no chance for autoscale

      let ileft = i2, iright = i1, jleft = j2, jright = j1, kleft = k2, kright = k1;

      for (i = i1; i < i2; ++i)
         for (j = j1; j < j2; ++j)
            for (k = k1; k < k2; ++k)
               if (histo.getBinContent(i+1, j+1, k+1) > min) {
                  if (i < ileft) ileft = i;
                  if (i >= iright) iright = i + 1;
                  if (j < jleft) jleft = j;
                  if (j >= jright) jright = j + 1;
                  if (k < kleft) kleft = k;
                  if (k >= kright) kright = k + 1;
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

      let opts = this.getSupportedDrawOptions();

      menu.addDrawMenu("Draw with", opts, arg => {
         if (arg==='inspect')
            return this.showInspector();

         this.decodeOptions(arg);

         this.interactiveRedraw(true, "drawopt");
      });
   }

   /** @summary draw TH3 object */
   static draw(dom, histo, opt) {

      let painter = new TH3Painter(dom, histo);
      painter.mode3d = true;

      return ensureTCanvas(painter, "3d").then(() => {
         painter.setAsMainPainter();
         painter.decodeOptions(opt);
         painter.checkPadRange();
         painter.scanContent();
         return painter.redraw();
      }).then(() => {
         let stats = painter.createStat(); // only when required
         if (stats)
            return TPavePainter.draw(dom, stats, "");
      }).then(() => {
         painter.fillToolbar();
         return painter;
      });
   }

} // class TH3Painter

export { TH3Painter };
