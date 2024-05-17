import { settings, gStyle, clTMultiGraph, kNoZoom } from '../core.mjs';
import { Vector2, BufferGeometry, BufferAttribute, Mesh, MeshBasicMaterial, ShapeUtils } from '../three.mjs';
import { getMaterialArgs } from '../base/base3d.mjs';
import { assignFrame3DMethods, drawBinsLego, drawBinsError3D, drawBinsContour3D, drawBinsSurf3D } from './hist3d.mjs';
import { TAxisPainter } from '../gpad/TAxisPainter.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';
import { TH2Painter as TH2Painter2D } from '../hist2d/TH2Painter.mjs';


/** @summary Draw TH2Poly histogram as lego
  * @private */
function drawTH2PolyLego(painter) {
   const histo = painter.getHisto(),
         pmain = painter.getFramePainter(),
         axis_zmin = pmain.z_handle.getScaleMin(),
         axis_zmax = pmain.z_handle.getScaleMax(),
         len = histo.fBins.arr.length,
         z0 = pmain.grz(axis_zmin);
   let colindx, bin, i, z1;

   // use global coordinates
   painter.maxbin = painter.gmaxbin;
   painter.minbin = painter.gminbin;
   painter.minposbin = painter.gminposbin;

   const cntr = painter.getContour(true), palette = painter.getHistPalette();

   for (i = 0; i < len; ++i) {
      bin = histo.fBins.arr[i];
      if (bin.fContent < axis_zmin) continue;

      colindx = cntr.getPaletteIndex(palette, bin.fContent);
      if (colindx === null) continue;

      // check if bin outside visible range
      if ((bin.fXmin > pmain.scale_xmax) || (bin.fXmax < pmain.scale_xmin) ||
          (bin.fYmin > pmain.scale_ymax) || (bin.fYmax < pmain.scale_ymin)) continue;

      z1 = pmain.grz((bin.fContent > axis_zmax) ? axis_zmax : bin.fContent);

      const all_pnts = [], all_faces = [];
      let ngraphs = 1, gr = bin.fPoly, nfaces = 0;

      if (gr._typename === clTMultiGraph) {
         ngraphs = bin.fPoly.fGraphs.arr.length;
         gr = null;
      }

      for (let ngr = 0; ngr < ngraphs; ++ngr) {
         if (!gr || (ngr > 0)) gr = bin.fPoly.fGraphs.arr[ngr];

         const x = gr.fX, y = gr.fY;
         let npnts = gr.fNpoints;
         while ((npnts>2) && (x[0]===x[npnts-1]) && (y[0]===y[npnts-1])) --npnts;

         let pnts, faces;

         for (let ntry = 0; ntry < 2; ++ntry) {
            // run two loops - on the first try to compress data, on second - run as is (removing duplication)

            let lastx, lasty, currx, curry,
                dist2 = pmain.size_x3d*pmain.size_z3d;
            const dist2limit = (ntry > 0) ? 0 : dist2/1e6;

            pnts = []; faces = null;

            for (let vert = 0; vert < npnts; ++vert) {
               currx = pmain.grx(x[vert]);
               curry = pmain.gry(y[vert]);
               if (vert > 0)
                  dist2 = (currx-lastx)*(currx-lastx) + (curry-lasty)*(curry-lasty);
               if (dist2 > dist2limit) {
                  pnts.push(new Vector2(currx, curry));
                  lastx = currx;
                  lasty = curry;
               }
            }

            try {
               if (pnts.length > 2)
                  faces = ShapeUtils.triangulateShape(pnts, []);
            } catch (e) {
               faces = null;
            }

            if (faces && (faces.length > pnts.length-3)) break;
         }

         if (faces?.length && pnts) {
            all_pnts.push(pnts);
            all_faces.push(faces);

            nfaces += faces.length * 2;
            if (z1 > z0) nfaces += pnts.length*2;
         }
      }

      const pos = new Float32Array(nfaces*9);
      let indx = 0;

      for (let ngr = 0; ngr < all_pnts.length; ++ngr) {
         const pnts = all_pnts[ngr], faces = all_faces[ngr];

         for (let layer = 0; layer < 2; ++layer) {
            for (let n = 0; n < faces.length; ++n) {
               const face = faces[n],
                   pnt1 = pnts[face[0]],
                   pnt2 = pnts[face[layer === 0 ? 2 : 1]],
                   pnt3 = pnts[face[layer === 0 ? 1 : 2]];

               pos[indx] = pnt1.x;
               pos[indx+1] = pnt1.y;
               pos[indx+2] = layer ? z1 : z0;
               indx+=3;

               pos[indx] = pnt2.x;
               pos[indx+1] = pnt2.y;
               pos[indx+2] = layer ? z1 : z0;
               indx+=3;

               pos[indx] = pnt3.x;
               pos[indx+1] = pnt3.y;
               pos[indx+2] = layer ? z1 : z0;
               indx+=3;
            }
         }

         if (z1>z0) {
            for (let n = 0; n < pnts.length; ++n) {
               const pnt1 = pnts[n], pnt2 = pnts[n > 0 ? n - 1 : pnts.length - 1];

               pos[indx] = pnt1.x;
               pos[indx+1] = pnt1.y;
               pos[indx+2] = z0;
               indx+=3;

               pos[indx] = pnt2.x;
               pos[indx+1] = pnt2.y;
               pos[indx+2] = z0;
               indx+=3;

               pos[indx] = pnt2.x;
               pos[indx+1] = pnt2.y;
               pos[indx+2] = z1;
               indx+=3;

               pos[indx] = pnt1.x;
               pos[indx+1] = pnt1.y;
               pos[indx+2] = z0;
               indx+=3;

               pos[indx] = pnt2.x;
               pos[indx+1] = pnt2.y;
               pos[indx+2] = z1;
               indx+=3;

               pos[indx] = pnt1.x;
               pos[indx+1] = pnt1.y;
               pos[indx+2] = z1;
               indx+=3;
            }
         }
      }

      const geometry = new BufferGeometry();
      geometry.setAttribute('position', new BufferAttribute(pos, 3));
      geometry.computeVertexNormals();

      const material = new MeshBasicMaterial(getMaterialArgs(painter._color_palette?.getColor(colindx), { vertexColors: false })),
            mesh = new Mesh(geometry, material);

      pmain.add3DMesh(mesh);

      mesh.painter = painter;
      mesh.bins_index = i;
      mesh.draw_z0 = z0;
      mesh.draw_z1 = z1;
      mesh.tip_color = 0x00FF00;

      mesh.tooltip = function(/* intersects */) {
         const p = this.painter, main = p.getFramePainter(),
             bin = p.getObject().fBins.arr[this.bins_index],

          tip = {
           use_itself: true, // indicate that use mesh itself for highlighting
           x1: main.grx(bin.fXmin),
           x2: main.grx(bin.fXmax),
           y1: main.gry(bin.fYmin),
           y2: main.gry(bin.fYmax),
           z1: this.draw_z0,
           z2: this.draw_z1,
           bin: this.bins_index,
           value: bin.fContent,
           color: this.tip_color,
           lines: p.getPolyBinTooltips(this.bins_index)
         };

         return tip;
      };
   }
}

/** @summary Draw 2-D histogram in 3D
  * @private */
class TH2Painter extends TH2Painter2D {

   /** @summary draw TH2 object in 3D mode */
   async draw3D(reason) {
      this.mode3d = true;

      const main = this.getFramePainter(), // who makes axis drawing
            is_main = this.isMainPainter(), // is main histogram
            histo = this.getHisto();
      let pr = Promise.resolve(true);

      if (reason === 'resize') {
         if (is_main && main.resize3D()) main.render3D();
      } else {
         const pad = this.getPadPainter().getRootPad(true),
               logz = pad?.fLogv ?? pad?.fLogz;
         let zmult = 1;

         if (this.options.minimum !== kNoZoom && this.options.maximum !== kNoZoom) {
            this.zmin = this.options.minimum;
            this.zmax = this.options.maximum;
         } else if (this.draw_content || (this.gmaxbin !== 0)) {
            this.zmin = logz ? this.gminposbin * 0.3 : this.gminbin;
            this.zmax = this.gmaxbin;
            zmult = 1 + 2*gStyle.fHistTopMargin;
         }

         if (logz && (this.zmin <= 0))
            this.zmin = this.zmax * 1e-5;

         this.createHistDrawAttributes(true);

         if (is_main) {
            assignFrame3DMethods(main);
            pr = main.create3DScene(this.options.Render3D, this.options.x3dscale, this.options.y3dscale, this.options.Ortho).then(() => {
               main.setAxesRanges(histo.fXaxis, this.xmin, this.xmax, histo.fYaxis, this.ymin, this.ymax, histo.fZaxis, this.zmin, this.zmax, this);
               main.set3DOptions(this.options);
               main.drawXYZ(main.toplevel, TAxisPainter, { zmult, zoom: settings.Zooming, ndim: 2,
                  draw: this.options.Axis !== -1, drawany: this.options.isCartesian(),
                  reverse_x: this.options.RevX, reverse_y: this.options.RevY });
            });
         }

         if (main.mode3d) {
            pr = pr.then(() => {
               if (this.draw_content) {
                  if (this.isTH2Poly())
                     drawTH2PolyLego(this);
                  else if (this.options.Contour)
                     drawBinsContour3D(this, true);
                  else if (this.options.Surf)
                     drawBinsSurf3D(this);
                  else if (this.options.Error)
                     drawBinsError3D(this);
                  else
                     drawBinsLego(this);
               } else if (this.options.Axis && this.options.Zscale) {
                  this.getContourLevels(true);
                  this.getHistPalette();
               }
               main.render3D();
               this.updateStatWebCanvas();
               main.addKeysHandler();
            });
         }
      }

      //  (re)draw palette by resize while canvas may change dimension
      if (is_main) {
         pr = pr.then(() => this.drawColorPalette(this.options.Zscale && ((this.options.Lego === 12) || (this.options.Lego === 14) ||
                                                  (this.options.Surf === 11) || (this.options.Surf === 12))));
      }

      return pr.then(() => this.updateFunctions())
               .then(() => this.updateHistTitle())
               .then(() => this);
   }

   /** @summary draw TH2 object */
   static async draw(dom, histo, opt) {
      return THistPainter._drawHist(new TH2Painter(dom, histo), opt);
   }

} // class TH2Painter

export { TH2Painter };
