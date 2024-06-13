import { gStyle, settings, kNoZoom, kInspect } from '../core.mjs';
import { Matrix4, Mesh, MeshBasicMaterial, MeshLambertMaterial, SphereGeometry,
         LineBasicMaterial, BufferAttribute, BufferGeometry } from '../three.mjs';
import { floatToString, TRandom } from '../base/BasePainter.mjs';
import { ensureRCanvas } from '../gpad/RCanvasPainter.mjs';
import { RAxisPainter } from '../gpad/RAxisPainter.mjs';
import { RHistPainter } from '../hist2d/RHistPainter.mjs';
import { createLineSegments, PointsCreator, Box3D } from '../base/base3d.mjs';
import { RH1Painter } from './RH1Painter.mjs';
import { RH2Painter } from './RH2Painter.mjs';
import { assignFrame3DMethods } from './hist3d.mjs';

/**
 * @summary Painter for RH3 classes
 *
 * @private
 */

class RH3Painter extends RHistPainter {

   /** @summary Returns histogram dimension */
   getDimension() { return 3; }

   scanContent(when_axis_changed) {
      // no need to rescan histogram while result does not depend from axis selection
      if (when_axis_changed && this.nbinsx && this.nbinsy && this.nbinsz) return;

      const histo = this.getHisto();
      if (!histo) return;

      this.extractAxesProperties(3);

      // global min/max, used at the moment in 3D drawing

      if (this.isDisplayItem()) {
         // take min/max values from the display item
         this.gminbin = histo.fContMin;
         this.gminposbin = histo.fContMinPos > 0 ? histo.fContMinPos : null;
         this.gmaxbin = histo.fContMax;
      } else {
         this.gminbin = this.gmaxbin = histo.getBinContent(1, 1, 1);

         for (let i = 0; i < this.nbinsx; ++i) {
            for (let j = 0; j < this.nbinsy; ++j) {
               for (let k = 0; k < this.nbinsz; ++k) {
                  const bin_content = histo.getBinContent(i+1, j+1, k+1);
                  if (bin_content < this.gminbin) this.gminbin = bin_content; else
                  if (bin_content > this.gmaxbin) this.gmaxbin = bin_content;
               }
            }
         }
      }

      this.draw_content = (this.gmaxbin !== 0) || (this.gminbin !== 0);
   }

  /** @summary Count histogram statistic */
   countStat() {
      const histo = this.getHisto(),
            xaxis = this.getAxis('x'),
            yaxis = this.getAxis('y'),
            zaxis = this.getAxis('z'),
            i1 = this.getSelectIndex('x', 'left'),
            i2 = this.getSelectIndex('x', 'right'),
            j1 = this.getSelectIndex('y', 'left'),
            j2 = this.getSelectIndex('y', 'right'),
            k1 = this.getSelectIndex('z', 'left'),
            k2 = this.getSelectIndex('z', 'right'),
            res = { name: histo.fName, entries: 0, integral: 0, meanx: 0, meany: 0, meanz: 0, rmsx: 0, rmsy: 0, rmsz: 0 };
      let stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0,
          stat_sumz1 = 0, stat_sumx2 = 0, stat_sumy2 = 0, stat_sumz2 = 0,
          xi, yi, zi, xx, xside, yy, yside, zz, zside, cont;

      for (xi = 1; xi <= this.nbinsx; ++xi) {
         xx = xaxis.GetBinCoord(xi - 0.5);
         xside = (xi <= i1+1) ? 0 : (xi > i2+1 ? 2 : 1);

         for (yi = 1; yi <= this.nbinsy; ++yi) {
            yy = yaxis.GetBinCoord(yi - 0.5);
            yside = (yi <= j1+1) ? 0 : (yi > j2+1 ? 2 : 1);

            for (zi = 1; zi <= this.nbinsz; ++zi) {
               zz = zaxis.GetBinCoord(zi - 0.5);
               zside = (zi <= k1+1) ? 0 : (zi > k2+1 ? 2 : 1);

               cont = histo.getBinContent(xi, yi, zi);
               res.entries += cont;

               if ((xside === 1) && (yside === 1) && (zside === 1)) {
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

      if (Math.abs(stat_sum0) > 1e-300) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.meanz = stat_sumz1 / stat_sum0;
         res.rmsx = Math.sqrt(Math.abs(stat_sumx2 / stat_sum0 - res.meanx**2));
         res.rmsy = Math.sqrt(Math.abs(stat_sumy2 / stat_sum0 - res.meany**2));
         res.rmsz = Math.sqrt(Math.abs(stat_sumz2 / stat_sum0 - res.meanz**2));
      }

      res.integral = stat_sum0;

      if (histo.fEntries > 1)
         res.entries = histo.fEntries;

      return res;
   }

   /** @summary Fill statistic */
   fillStatistic(stat, dostat /*, dofit */) {
      const data = this.countStat(),
          print_name = dostat % 10,
          print_entries = Math.floor(dostat / 10) % 10,
          print_mean = Math.floor(dostat / 100) % 10,
          print_rms = Math.floor(dostat / 1000) % 10,
          // print_under = Math.floor(dostat / 10000) % 10,
          // print_over = Math.floor(dostat / 100000) % 10,
          print_integral = Math.floor(dostat / 1000000) % 10;
          // print_skew = Math.floor(dostat / 10000000) % 10;
          // print_kurt = Math.floor(dostat / 100000000) % 10;

      stat.clearStat();

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


      return true;
   }

   /** @summary Provide text information (tooltips) for histogram bin */
   getBinTooltips(ix, iy, iz) {
      const lines = [], histo = this.getHisto();
      let dx = 1, dy = 1, dz = 1;

      if (this.isDisplayItem()) {
         dx = histo.stepx || 1;
         dy = histo.stepy || 1;
         dz = histo.stepz || 1;
      }

      lines.push(this.getObjectHint(),
                 `x = ${this.getAxisBinTip('x', ix, dx)}  xbin=${ix+1}`,
                 `y = ${this.getAxisBinTip('y', iy, dy)}  ybin=${iy+1}`,
                 `z = ${this.getAxisBinTip('z', iz, dz)}  zbin=${iz+1}`);

      const binz = histo.getBinContent(ix+1, iy+1, iz+1),
            lbl = 'entries = '+ ((dx > 1) || (dy > 1) || (dz > 1) ? '~' : '');
      if (binz === Math.round(binz))
         lines.push(lbl + binz);
      else
         lines.push(lbl + floatToString(binz, gStyle.fStatFormat));

      return lines;
   }

   /** @summary Try to draw 3D histogram as scatter plot
     * @desc If there are too many points, returns promise with false */
   async draw3DScatter(handle) {
      const histo = this.getHisto(),
            main = this.getFramePainter(),
            i1 = handle.i1, i2 = handle.i2, di = handle.stepi,
            j1 = handle.j1, j2 = handle.j2, dj = handle.stepj,
            k1 = handle.k1, k2 = handle.k2, dk = handle.stepk;

      if ((i2 <= i1) || (j2 <= j1) || (k2 <= k1))
         return true;

      // scale down factor if too large values
      const coef = (this.gmaxbin > 1000) ? 1000/this.gmaxbin : 1,
            content_lmt = Math.max(0, this.gminbin);
      let i, j, k, bin_content, numpixels = 0, sumz = 0;

      for (i = i1; i < i2; i += di) {
         for (j = j1; j < j2; j += dj) {
            for (k = k1; k < k2; k += dk) {
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

      const pnts = new PointsCreator(numpixels, main.webgl, main.size_x3d/200),
            bins = new Int32Array(numpixels),
            xaxis = this.getAxis('x'), yaxis = this.getAxis('y'), zaxis = this.getAxis('z'),
            rnd = new TRandom(sumz);
      let nbin = 0;

      for (i = i1; i < i2; i += di) {
         for (j = j1; j < j2; j += dj) {
            for (k = k1; k < k2; k += dk) {
               bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (bin_content <= content_lmt) continue;
               const num = Math.round(bin_content*coef);

               for (let n=0; n<num; ++n) {
                  const binx = xaxis.GetBinCoord(i + rnd.random()),
                      biny = yaxis.GetBinCoord(j + rnd.random()),
                      binz = zaxis.GetBinCoord(k + rnd.random());

                  // remember bin index for tooltip
                  bins[nbin++] = histo.getBin(i+1, j+1, k+1);

                  pnts.addPoint(main.grx(binx), main.gry(biny), main.grz(binz));
               }
            }
         }
      }

      return pnts.createPoints({ color: this.v7EvalColor('fill_color', 'red') }).then(mesh => {
         main.add3DMesh(mesh);

         mesh.bins = bins;
         mesh.painter = this;
         mesh.tip_color = 0x00FF00;

         mesh.tooltip = function(intersect) {
            const indx = Math.floor(intersect.index / this.nvertex);
            if ((indx < 0) || (indx >= this.bins.length)) return null;

            const p = this.painter,
                main = p.getFramePainter(),
                tip = p.get3DToolTip(this.bins[indx]);

            tip.x1 = main.grx(p.getAxis('x').GetBinLowEdge(tip.ix));
            tip.x2 = main.grx(p.getAxis('x').GetBinLowEdge(tip.ix+di));
            tip.y1 = main.gry(p.getAxis('y').GetBinLowEdge(tip.iy));
            tip.y2 = main.gry(p.getAxis('y').GetBinLowEdge(tip.iy+dj));
            tip.z1 = main.grz(p.getAxis('z').GetBinLowEdge(tip.iz));
            tip.z2 = main.grz(p.getAxis('z').GetBinLowEdge(tip.iz+dk));
            tip.color = this.tip_color;
            tip.opacity = 0.3;

            return tip;
         };

         return true;
      });
   }

   /** @summary Drawing of 3D histogram */
   draw3DBins(handle) {
      const main = this.getFramePainter();
      let fillcolor = this.v7EvalColor('fill_color', 'red'),
          buffer_size = 0, use_lambert = false,
          use_helper = false, use_colors = false, use_opacity = 1, use_scale = true,
          single_bin_verts, single_bin_norms,
          tipscale = 0.5;

      if (this.options.Sphere) {
         // drawing spheres
         tipscale = 0.4;
         use_lambert = true;
         if (this.options.Sphere === 11) use_colors = true;

         const geom = main.webgl ? new SphereGeometry(0.5, 16, 12) : new SphereGeometry(0.5, 8, 6);
         geom.applyMatrix4(new Matrix4().makeRotationX(Math.PI/2));
         geom.computeVertexNormals();

         const indx = geom.getIndex().array,
             pos = geom.getAttribute('position').array,
             norm = geom.getAttribute('normal').array;

         buffer_size = indx.length*3;
         single_bin_verts = new Float32Array(buffer_size);
         single_bin_norms = new Float32Array(buffer_size);

         for (let k=0; k<indx.length; ++k) {
            const iii = indx[k]*3;
            single_bin_verts[k*3] = pos[iii];
            single_bin_verts[k*3+1] = pos[iii+1];
            single_bin_verts[k*3+2] = pos[iii+2];
            single_bin_norms[k*3] = norm[iii];
            single_bin_norms[k*3+1] = norm[iii+1];
            single_bin_norms[k*3+2] = norm[iii+2];
         }
      } else {
         const indicies = Box3D.Indexes,
             normals = Box3D.Normals,
             vertices = Box3D.Vertices;

         buffer_size = indicies.length*3;
         single_bin_verts = new Float32Array(buffer_size);
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

         if (this.options.Box === 11) use_colors = true; else
         if (this.options.Box === 12) { use_colors = true; use_helper = false; } else
         if (this.options.Color) { use_colors = true; use_opacity = 0.5; use_scale = false; use_helper = false; use_lambert = true; }
      }

      if (use_scale)
         use_scale = (this.gminbin || this.gmaxbin) ? 1 / Math.max(Math.abs(this.gminbin), Math.abs(this.gmaxbin)) : 1;

      const histo = this.getHisto(),
            i1 = handle.i1, i2 = handle.i2, di = handle.stepi,
            j1 = handle.j1, j2 = handle.j2, dj = handle.stepj,
            k1 = handle.k1, k2 = handle.k2, dk = handle.stepk;
      let palette = null;

      if (use_colors) {
         palette = main.getHistPalette();
         this.createContour(main, palette);
      }

      if ((i2 <= i1) || (j2 <= j1) || (k2 <= k1))
         return true;

      const cols_size = [];
      let nbins = 0, i, j, k, wei, bin_content, num_colors = 0, cols_sequence = [];

      for (i = i1; i < i2; i += di) {
         for (j = j1; j < j2; j += dj) {
            for (k = k1; k < k2; k += dk) {
               bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (!this.options.Color && ((bin_content === 0) || (bin_content < this.gminbin))) continue;
               wei = use_scale ? Math.pow(Math.abs(bin_content*use_scale), 0.3333) : 1;
               if (wei < 1e-3) continue; // do not draw empty or very small bins

               nbins++;

               if (!use_colors) continue;

               const colindx = palette.getContourIndex(bin_content);
               if (colindx >= 0) {
                  if (cols_size[colindx] === undefined) {
                     cols_size[colindx] = 0;
                     cols_sequence[colindx] = num_colors++;
                  }
                  cols_size[colindx]+=1;
               } else
                  console.error(`not found color for value = ${bin_content}`);
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
            helper_indexes = new Array(num_colors),  // helper_kind === 1, use original vertices
            helper_positions = new Array(num_colors);  // helper_kind === 2, all vertices copied into separate buffer

      for (let ncol = 0; ncol < cols_size.length; ++ncol) {
         if (!cols_size[ncol]) continue; // ignore dummy colors

         nbins = cols_size[ncol]; // how many bins with specified color
         const nseq = cols_sequence[ncol];

         cols_nbins[nseq] = 0; // counter for the filled bins

         helper_kind[nseq] = 0;

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

      const xaxis = this.getAxis('x'), yaxis = this.getAxis('y'), zaxis = this.getAxis('z');
      let grx1, grx2, gry1, gry2, grz1, grz2;

      for (i = i1; i < i2; i += di) {
         grx1 = main.grx(xaxis.GetBinLowEdge(i+1));
         grx2 = main.grx(xaxis.GetBinLowEdge(i+2));
         for (j = j1; j < j2; j += dj) {
            gry1 = main.gry(yaxis.GetBinLowEdge(j+1));
            gry2 = main.gry(yaxis.GetBinLowEdge(j+2));
            for (k = k1; k < k2; k +=dk) {
               bin_content = histo.getBinContent(i+1, j+1, k+1);
               if (!this.options.Color && ((bin_content === 0) || (bin_content < this.gminbin))) continue;

               wei = use_scale ? Math.pow(Math.abs(bin_content*use_scale), 0.3333) : 1;
               if (wei < 1e-3) continue; // do not show very small bins

               let nseq = 0;
               if (use_colors) {
                  const colindx = palette.getContourIndex(bin_content);
                  if (colindx < 0) continue;
                  nseq = cols_sequence[colindx];
               }

               nbins = cols_nbins[nseq];

               grz1 = main.grz(zaxis.GetBinLowEdge(k+1));
               grz2 = main.grz(zaxis.GetBinLowEdge(k+2));

               // remember bin index for tooltip
               bin_tooltips[nseq][nbins] = histo.getBin(i+1, j+1, k+1);

               let vvv = nbins * buffer_size;
               const bin_v = bin_verts[nseq], bin_n = bin_norms[nseq];

               // Grab the coordinates and scale that are being assigned to each bin
               for (let vi = 0; vi < buffer_size; vi += 3, vvv += 3) {
                  bin_v[vvv] = (grx2 + grx1) / 2 + single_bin_verts[vi] * (grx2 - grx1) * wei;
                  bin_v[vvv+1] = (gry2 + gry1) / 2 + single_bin_verts[vi+1] * (gry2 - gry1) * wei;
                  bin_v[vvv+2] = (grz2 + grz1) / 2 + single_bin_verts[vi+2] * (grz2 - grz1) * wei;

                  bin_n[vvv] = single_bin_norms[vi];
                  bin_n[vvv+1] = single_bin_norms[vi+1];
                  bin_n[vvv+2] = single_bin_norms[vi+2];
               }

               if (helper_kind[nseq] === 1) {
                  // reuse vertices created for the mesh
                  const helper_segments = Box3D.MeshSegments;
                  vvv = nbins * helper_segments.length;
                  const shift = Math.round(nbins * buffer_size/3),
                        helper_i = helper_indexes[nseq];
                  for (let n = 0; n < helper_segments.length; ++n)
                     helper_i[vvv+n] = shift + helper_segments[n];
               }

               if (helper_kind[nseq] === 2) {
                  const helper_segments = Box3D.Segments,
                        helper_p = helper_positions[nseq];
                  vvv = nbins * helper_segments.length * 3;
                  for (let n = 0; n < helper_segments.length; ++n, vvv += 3) {
                     const vert = Box3D.Vertices[helper_segments[n]];
                     helper_p[vvv] = (grx2 + grx1) / 2 + (vert.x - 0.5) * (grx2 - grx1) * wei;
                     helper_p[vvv+1] = (gry2 + gry1) / 2 + (vert.y - 0.5) * (gry2 - gry1) * wei;
                     helper_p[vvv+2] = (grz2 + grz1) / 2 + (vert.z - 0.5) * (grz2 - grz1) * wei;
                  }
               }

               cols_nbins[nseq] = nbins+1;
            }
         }
      }

      for (let ncol = 0; ncol < cols_size.length; ++ncol) {
         if (!cols_size[ncol]) continue; // ignore dummy colors

         const nseq = cols_sequence[ncol],
              // BufferGeometries that store geometry of all bins
              all_bins_buffgeom = new BufferGeometry();

         // Create mesh from bin buffergeometry
         all_bins_buffgeom.setAttribute('position', new BufferAttribute(bin_verts[nseq], 3));
         all_bins_buffgeom.setAttribute('normal', new BufferAttribute(bin_norms[nseq], 3));

         if (use_colors) fillcolor = palette.getColor(ncol);

         const material = use_lambert
                           ? new MeshLambertMaterial({ color: fillcolor, opacity: use_opacity, transparent: use_opacity < 1, vertexColors: false })
                           : new MeshBasicMaterial({ color: fillcolor, opacity: use_opacity, transparent: use_opacity < 1, vertexColors: false }),
               combined_bins = new Mesh(all_bins_buffgeom, material);

         combined_bins.bins = bin_tooltips[nseq];
         combined_bins.bins_faces = buffer_size/9;
         combined_bins.painter = this;
         combined_bins.tipscale = tipscale;
         combined_bins.tip_color = 0x00FF00;
         combined_bins.use_scale = use_scale;

         combined_bins.tooltip = function(intersect) {
            const indx = Math.floor(intersect.faceIndex / this.bins_faces);
            if ((indx < 0) || (indx >= this.bins.length)) return null;

            const p = this.painter,
                  main = p.getFramePainter(),
                  tip = p.get3DToolTip(this.bins[indx]),
                  grx1 = main.grx(xaxis.GetBinCoord(tip.ix-1)),
                  grx2 = main.grx(xaxis.GetBinCoord(tip.ix)),
                  gry1 = main.gry(yaxis.GetBinCoord(tip.iy-1)),
                  gry2 = main.gry(yaxis.GetBinCoord(tip.iy)),
                  grz1 = main.grz(zaxis.GetBinCoord(tip.iz-1)),
                  grz2 = main.grz(zaxis.GetBinCoord(tip.iz)),
                  wei2 = (this.use_scale ? Math.pow(Math.abs(tip.value*this.use_scale), 0.3333) : 1) * this.tipscale;

            tip.x1 = (grx2 + grx1) / 2 - (grx2 - grx1) * wei2;
            tip.x2 = (grx2 + grx1) / 2 + (grx2 - grx1) * wei2;
            tip.y1 = (gry2 + gry1) / 2 - (gry2 - gry1) * wei2;
            tip.y2 = (gry2 + gry1) / 2 + (gry2 - gry1) * wei2;
            tip.z1 = (grz2 + grz1) / 2 - (grz2 - grz1) * wei2;
            tip.z2 = (grz2 + grz1) / 2 + (grz2 - grz1) * wei2;
            tip.color = this.tip_color;

            return tip;
         };

         main.add3DMesh(combined_bins);

         if (helper_kind[nseq] > 0) {
            const lcolor = this.v7EvalColor('line_color', 'lightblue'),
                  helper_material = new LineBasicMaterial({ color: lcolor }),
                  lines = (helper_kind[nseq] === 1)
                          // reuse positions from the mesh - only special index was created
                          ? createLineSegments(bin_verts[nseq], helper_material, helper_indexes[nseq])
                          : createLineSegments(helper_positions[nseq], helper_material);

            main.add3DMesh(lines);
         }
      }

      if (use_colors)
         this.updatePaletteDraw();
   }

   draw3D() {
      if (!this.draw_content)
         return false;

      // this.options.Scatter = false;
      // this.options.Box = true;

      const handle = this.prepareDraw({ only_indexes: true, extra: -0.5, right_extra: -1 }),
            pr = this.options.Scatter ? this.draw3DScatter(handle) : Promise.resolve(false);

      return pr.then(res => {
         return res || this.draw3DBins(handle);
      });
   }

   /** @summary Redraw histogram */
   redraw(reason) {
      const main = this.getFramePainter(); // who makes axis and 3D drawing

      if (reason === 'resize') {
         if (main.resize3D()) main.render3D();
         return this;
      }

      assignFrame3DMethods(main);
      return main.create3DScene(this.options.Render3D).then(() => {
         main.setAxesRanges(this.getAxis('x'), this.xmin, this.xmax, this.getAxis('y'), this.ymin, this.ymax, this.getAxis('z'), this.zmin, this.zmax);
         main.set3DOptions(this.options);
         main.drawXYZ(main.toplevel, RAxisPainter, { zoom: settings.Zooming, ndim: 3, draw: true, v7: true });
         return this.drawingBins(reason);
      }).then(() => this.draw3D()).then(() => {
         main.render3D();
         main.addKeysHandler();
         return this;
      });
   }

   /** @summary Fill pad toolbar with RH3-related functions */
   fillToolbar() {
      const pp = this.getPadPainter();
      if (!pp) return;

      pp.addPadButton('auto_zoom', 'Unzoom all axes', 'ToggleZoom', 'Ctrl *');
      if (this.draw_content)
         pp.addPadButton('statbox', 'Toggle stat box', 'ToggleStatBox');
      pp.showPadButtons();
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis, min, max) {
      let obj = this.getHisto();
      if (obj) obj = obj['f'+axis.toUpperCase()+'axis'];
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
            histo = this.getHisto();
      let i, j, k;

      if ((i1 === i2) || (j1 === j2) || (k1 === k2)) return;

      // first find minimum
      let min = histo.getBinContent(i1 + 1, j1 + 1, k1+1);
      for (i = i1; i < i2; ++i) {
         for (j = j1; j < j2; ++j) {
            for (k = k1; k < k2; ++k)
               min = Math.min(min, histo.getBinContent(i+1, j+1, k+1));
         }
      }

      if (min > 0) return; // if all points positive, no chance for autoscale

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
         xmin = this.getAxis('x').GetBinLowEdge(ileft+1);
         xmax = this.getAxis('x').GetBinLowEdge(iright+1);
         isany = true;
      }

      if ((jleft > j1 || jright < j2) && (jleft < jright - 1)) {
         ymin = this.getAxis('y').GetBinLowEdge(jleft+1);
         ymax = this.getAxis('y').GetBinLowEdge(jright+1);
         isany = true;
      }

      if ((kleft > k1 || kright < k2) && (kleft < kright - 1)) {
         zmin = this.getAxis('z').GetBinLowEdge(kleft+1);
         zmax = this.getAxis('z').GetBinLowEdge(kright+1);
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

   /** @summary draw RH3 object */
  static async draw(dom, histo /* ,opt */) {
      const painter = new RH3Painter(dom, histo);
      painter.mode3d = true;

      return ensureRCanvas(painter, '3d').then(() => {
         painter.setAsMainPainter();

         painter.options = { Box: 0, Scatter: false, Sphere: 0, Color: false, minimum: kNoZoom, maximum: kNoZoom, FrontBox: false, BackBox: false };

         const kind = painter.v7EvalAttr('kind', ''),
               sub = painter.v7EvalAttr('sub', 0),
               o = painter.options;

         switch (kind) {
            case 'box': o.Box = 10 + sub; break;
            case 'sphere': o.Sphere = 10 + sub; break;
            case 'col': o.Color = true; break;
            case 'scat': o.Scatter = true; break;
            default: o.Box = 10;
         }

         painter.scanContent();
         return painter.redraw();
      });
   }

} // class RH3Painter

/** @summary draw RHistDisplayItem  object
  * @private */
function drawHistDisplayItem(dom, obj, opt) {
   if (!obj)
      return null;

   if (obj.fAxes.length === 1)
      return RH1Painter.draw(dom, obj, opt);

   if (obj.fAxes.length === 2)
      return RH2Painter.draw(dom, obj, opt);

   if (obj.fAxes.length === 3)
      return RH3Painter.draw(dom, obj, opt);

   return null;
}

export { RH3Painter, drawHistDisplayItem };
