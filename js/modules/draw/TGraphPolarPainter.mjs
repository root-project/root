import { settings, create, isBatchMode } from '../core.mjs';
import { scaleLinear, select as d3_select, pointer as d3_pointer } from '../d3.mjs';
import { DrawOptions, buildSvgPath } from '../base/BasePainter.mjs';
import { ObjectPainter, getElementMainPainter } from '../base/ObjectPainter.mjs';
import { TAttLineHandler } from '../base/TAttLineHandler.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { TooltipHandler } from '../gpad/TFramePainter.mjs';


/**
 * @summary Painter for TGraphPolargram objects.
 *
 * @private */

class TGraphPolargramPainter extends ObjectPainter {

   /** @summary Create painter
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} polargram - object to draw */
   constructor(dom, polargram) {
      super(dom, polargram);
      this.$polargram = true; // indicate that this is polargram
      this.zoom_rmin = this.zoom_rmax = 0;
   }

   /** @summary Translate coordinates */
   translate(angle, radius, keep_float) {
      let _rx = this.r(radius), _ry = _rx/this.szx*this.szy,
          pos = {
            x: _rx * Math.cos(-angle - this.angle),
            y: _ry * Math.sin(-angle - this.angle),
            rx: _rx,
            ry: _ry
         };

      if (!keep_float) {
         pos.x = Math.round(pos.x);
         pos.y = Math.round(pos.y);
         pos.rx =  Math.round(pos.rx);
         pos.ry =  Math.round(pos.ry);
      }
      return pos;
   }

   /** @summary format label for radius ticks */
   format(radius) {

      if (radius === Math.round(radius)) return radius.toString();
      if (this.ndig>10) return radius.toExponential(4);

      return radius.toFixed((this.ndig > 0) ? this.ndig : 0);
   }

   /** @summary Convert axis values to text */
   axisAsText(axis, value) {

      if (axis == "r") {
         if (value === Math.round(value)) return value.toString();
         if (this.ndig>10) return value.toExponential(4);
         return value.toFixed(this.ndig+2);
      }

      value *= 180/Math.PI;
      return (value === Math.round(value)) ? value.toString() : value.toFixed(1);
   }

   /** @summary Returns coordinate of frame - without using frame itself */
   getFrameRect() {
      let pp = this.getPadPainter(),
          pad = pp.getRootPad(true),
          w = pp.getPadWidth(),
          h = pp.getPadHeight(),
          rect = {};

      if (pad) {
         rect.szx = Math.round(Math.max(0.1, 0.5 - Math.max(pad.fLeftMargin, pad.fRightMargin))*w);
         rect.szy = Math.round(Math.max(0.1, 0.5 - Math.max(pad.fBottomMargin, pad.fTopMargin))*h);
      } else {
         rect.szx = Math.round(0.5*w);
         rect.szy = Math.round(0.5*h);
      }

      rect.width = 2*rect.szx;
      rect.height = 2*rect.szy;
      rect.x = Math.round(w/2 - rect.szx);
      rect.y = Math.round(h/2 - rect.szy);

      rect.hint_delta_x = rect.szx;
      rect.hint_delta_y = rect.szy;

      rect.transform = `translate(${rect.x},${rect.y})`;

      return rect;
   }

   /** @summary Process mouse event */
   mouseEvent(kind, evnt) {
      let layer = this.getLayerSvg("primitives_layer"),
          interactive = layer.select(".interactive_ellipse");
      if (interactive.empty()) return;

      let pnt = null;

      if (kind !== 'leave') {
         let pos = d3_pointer(evnt, interactive.node());
         pnt = { x: pos[0], y: pos[1], touch: false };
      }

      this.processFrameTooltipEvent(pnt);
   }

   /** @summary Process mouse wheel event */
   mouseWheel(evnt) {
      evnt.stopPropagation();
      evnt.preventDefault();

      this.processFrameTooltipEvent(null); // remove all tooltips

      let polar = this.getObject();

      if (!polar) return;

      let delta = evnt.wheelDelta ? -evnt.wheelDelta : (evnt.deltaY || evnt.detail);
      if (!delta) return;

      delta = (delta<0) ? -0.2 : 0.2;

      let rmin = this.scale_rmin, rmax = this.scale_rmax, range = rmax - rmin;

      // rmin -= delta*range;
      rmax += delta*range;

      if ((rmin<polar.fRwrmin) || (rmax>polar.fRwrmax)) rmin = rmax = 0;

      if ((this.zoom_rmin != rmin) || (this.zoom_rmax != rmax)) {
         this.zoom_rmin = rmin;
         this.zoom_rmax = rmax;
         this.redrawPad();
      }
   }

   /** @summary Redraw polargram */
   redraw() {
      if (!this.isMainPainter()) return;

      let polar = this.getObject(),
          rect = this.getPadPainter().getFrameRect();

      this.createG();

      this.draw_g.attr("transform", `translate(${Math.round(rect.x + rect.width/2)},${Math.round(rect.y + rect.height/2)})`);
      this.szx = rect.szx;
      this.szy = rect.szy;

      this.scale_rmin = polar.fRwrmin;
      this.scale_rmax = polar.fRwrmax;
      if (this.zoom_rmin != this.zoom_rmax) {
         this.scale_rmin = this.zoom_rmin;
         this.scale_rmax = this.zoom_rmax;
      }

      this.r = scaleLinear().domain([this.scale_rmin, this.scale_rmax]).range([ 0, this.szx ]);
      this.angle = polar.fAxisAngle || 0;

      let ticks = this.r.ticks(5),
          nminor = Math.floor((polar.fNdivRad % 10000) / 100);

      this.createAttLine({ attr: polar });
      if (!this.gridatt) this.gridatt = new TAttLineHandler({ color: polar.fLineColor, style: 2, width: 1 });

      let range = Math.abs(polar.fRwrmax - polar.fRwrmin);
      this.ndig = (range <= 0) ? -3 : Math.round(Math.log10(ticks.length / range));

      // verify that all radius labels are unique
      let lbls = [], indx = 0;
      while (indx<ticks.length) {
         let lbl = this.format(ticks[indx]);
         if (lbls.indexOf(lbl)>=0) {
            if (++this.ndig>10) break;
            lbls = []; indx = 0; continue;
          }
         lbls.push(lbl);
         indx++;
      }

      let exclude_last = false;

      if ((ticks[ticks.length-1] < polar.fRwrmax) && (this.zoom_rmin == this.zoom_rmax)) {
         ticks.push(polar.fRwrmax);
         exclude_last = true;
      }

      this.startTextDrawing(polar.fRadialLabelFont, Math.round(polar.fRadialTextSize * this.szy * 2));

      for (let n = 0; n < ticks.length; ++n) {
         let rx = this.r(ticks[n]), ry = rx/this.szx*this.szy;
         this.draw_g.append("ellipse")
             .attr("cx",0)
             .attr("cy",0)
             .attr("rx",Math.round(rx))
             .attr("ry",Math.round(ry))
             .style("fill", "none")
             .call(this.lineatt.func);

         if ((n < ticks.length-1) || !exclude_last)
            this.drawText({ align: 23, x: Math.round(rx), y: Math.round(polar.fRadialTextSize * this.szy * 0.5),
                            text: this.format(ticks[n]), color: this.getColor(polar.fRadialLabelColor), latex: 0 });

         if ((nminor>1) && ((n < ticks.length-1) || !exclude_last)) {
            let dr = (ticks[1] - ticks[0]) / nminor;
            for (let nn = 1; nn < nminor; ++nn) {
               let gridr = ticks[n] + dr*nn;
               if (gridr > this.scale_rmax) break;
               rx = this.r(gridr); ry = rx/this.szx*this.szy;
               this.draw_g.append("ellipse")
                   .attr("cx",0)
                   .attr("cy",0)
                   .attr("rx",Math.round(rx))
                   .attr("ry",Math.round(ry))
                   .style("fill", "none")
                   .call(this.gridatt.func);
            }
         }
      }

      let nmajor = polar.fNdivPol % 100;
      if ((nmajor !== 8) && (nmajor !== 3)) nmajor = 8;

      return this.finishTextDrawing().then(() => {

         let fontsize = Math.round(polar.fPolarTextSize * this.szy * 2);
         this.startTextDrawing(polar.fPolarLabelFont, fontsize);

         lbls = (nmajor==8) ? ["0", "#frac{#pi}{4}", "#frac{#pi}{2}", "#frac{3#pi}{4}", "#pi", "#frac{5#pi}{4}", "#frac{3#pi}{2}", "#frac{7#pi}{4}"] : ["0", "#frac{2#pi}{3}", "#frac{4#pi}{3}"];
         let aligns = [12, 11, 21, 31, 32, 33, 23, 13];

         for (let n = 0; n < nmajor; ++n) {
            let angle = -n*2*Math.PI/nmajor - this.angle;
            this.draw_g.append("svg:path")
                .attr("d",`M0,0L${Math.round(this.szx*Math.cos(angle))},${Math.round(this.szy*Math.sin(angle))}`)
                .call(this.lineatt.func);

            let aindx = Math.round(16 -angle/Math.PI*4) % 8; // index in align table, here absolute angle is important

            this.drawText({ align: aligns[aindx],
                            x: Math.round((this.szx+fontsize)*Math.cos(angle)),
                            y: Math.round((this.szy + fontsize/this.szx*this.szy)*(Math.sin(angle))),
                            text: lbls[n],
                            color: this.getColor(polar.fPolarLabelColor), latex: 1 });
         }

         return this.finishTextDrawing();
      }).then(() => {

         nminor = Math.floor((polar.fNdivPol % 10000) / 100);

         if (nminor > 1)
            for (let n = 0; n < nmajor*nminor; ++n) {
               if (n % nminor === 0) continue;
               let angle = -n*2*Math.PI/nmajor/nminor - this.angle;
               this.draw_g.append("svg:path")
                   .attr("d",`M0,0L${Math.round(this.szx*Math.cos(angle))},${Math.round(this.szy*Math.sin(angle))}`)
                   .call(this.gridatt.func);
            }

         if (isBatchMode()) return;

         TooltipHandler.assign(this);

         let layer = this.getLayerSvg("primitives_layer"),
             interactive = layer.select(".interactive_ellipse");

         if (interactive.empty())
            interactive = layer.append("g")
                               .classed("most_upper_primitives", true)
                               .append("ellipse")
                               .classed("interactive_ellipse", true)
                               .attr("cx",0)
                               .attr("cy",0)
                               .style("fill", "none")
                               .style("pointer-events","visibleFill")
                               .on('mouseenter', evnt => this.mouseEvent('enter', evnt))
                               .on('mousemove', evnt => this.mouseEvent('move', evnt))
                               .on('mouseleave', evnt => this.mouseEvent('leave', evnt));

         interactive.attr("rx", this.szx).attr("ry", this.szy);

         d3_select(interactive.node().parentNode).attr("transform", this.draw_g.attr("transform"));

         if (settings.Zooming && settings.ZoomWheel)
            interactive.on("wheel", evnt => this.mouseWheel(evnt));
      });
   }

   /** @summary Draw TGraphPolargram */
   static draw(dom, polargram /*, opt*/) {

      let main = getElementMainPainter(dom);
      if (main) {
         if (main.getObject() === polargram)
            return main;
         throw Error("Cannot superimpose TGraphPolargram with any other drawings");
      }

      let painter = new TGraphPolargramPainter(dom, polargram);
      return ensureTCanvas(painter, false).then(() => {
         painter.setAsMainPainter();
         return painter.redraw();
      }).then(() => painter);
   }

} // class TGraphPolargramPainter


/**
 * @summary Painter for TGraphPolar objects.
 *
 * @private
 */

class TGraphPolarPainter extends ObjectPainter {

   /** @summary Redraw TGraphPolar */
   redraw() {
      this.drawGraphPolar();
   }

   /** @summary Decode options for drawing TGraphPolar */
   decodeOptions(opt) {

      let d = new DrawOptions(opt || "L");

      if (!this.options) this.options = {};

      Object.assign(this.options, {
          mark: d.check("P"),
          err: d.check("E"),
          fill: d.check("F"),
          line: d.check("L"),
          curve: d.check("C")
      });

      this.storeDrawOpt(opt);
   }

   /** @summary Drawing TGraphPolar */
   drawGraphPolar() {
      let graph = this.getObject(),
          main = this.getMainPainter();

      if (!graph || !main?.$polargram) return;

      if (this.options.mark) this.createAttMarker({ attr: graph });
      if (this.options.err || this.options.line || this.options.curve) this.createAttLine({ attr: graph });
      if (this.options.fill) this.createAttFill({ attr: graph });

      this.createG();

      this.draw_g.attr("transform", main.draw_g.attr("transform"));

      let mpath = "", epath = "", lpath = "", bins = [];

      for (let n = 0; n < graph.fNpoints; ++n) {

         if (graph.fY[n] > main.scale_rmax) continue;

         if (this.options.err) {
            let pos1 = main.translate(graph.fX[n], graph.fY[n] - graph.fEY[n]),
                pos2 = main.translate(graph.fX[n], graph.fY[n] + graph.fEY[n]);
            epath += `M${pos1.x},${pos1.y}L${pos2.x},${pos2.y}`;

            pos1 = main.translate(graph.fX[n] + graph.fEX[n], graph.fY[n]);
            pos2 = main.translate(graph.fX[n] - graph.fEX[n], graph.fY[n]);

            epath += `M${pos1.x},${pos1.y}A${pos2.rx},${pos2.ry},0,0,1,${pos2.x},${pos2.y}`;
         }

         let pos = main.translate(graph.fX[n], graph.fY[n]);

         if (this.options.mark) {
            mpath += this.markeratt.create(pos.x, pos.y);
         }

         if (this.options.line || this.options.fill) {
            lpath += (lpath ? "L" : "M") + pos.x + "," + pos.y;
         }

         if (this.options.curve) {
            pos.grx = pos.x;
            pos.gry = pos.y;
            bins.push(pos);
         }
      }

      if (this.options.fill && lpath)
         this.draw_g.append("svg:path")
             .attr("d", lpath + "Z")
             .call(this.fillatt.func);

      if (this.options.line && lpath)
         this.draw_g.append("svg:path")
             .attr("d", lpath)
             .style("fill", "none")
             .call(this.lineatt.func);

      if (this.options.curve && bins.length)
         this.draw_g.append("svg:path")
                 .attr("d", buildSvgPath("bezier", bins).path)
                 .style("fill", "none")
                 .call(this.lineatt.func);

      if (epath)
         this.draw_g.append("svg:path")
             .attr("d", epath)
             .style("fill","none")
             .call(this.lineatt.func);

      if (mpath)
         this.draw_g.append("svg:path")
               .attr("d", mpath)
               .call(this.markeratt.func);
   }

   /** @summary Create polargram object */
   createPolargram() {
      let polargram = create("TGraphPolargram"),
          gr = this.getObject();

      let rmin = gr.fY[0] || 0, rmax = rmin;
      for (let n = 0; n < gr.fNpoints; ++n) {
         rmin = Math.min(rmin, gr.fY[n] - gr.fEY[n]);
         rmax = Math.max(rmax, gr.fY[n] + gr.fEY[n]);
      }

      polargram.fRwrmin = rmin - (rmax-rmin)*0.1;
      polargram.fRwrmax = rmax + (rmax-rmin)*0.1;

      return polargram;
   }

   /** @summary Provide tooltip at specified point */
   extractTooltip(pnt) {
      if (!pnt) return null;

      let graph = this.getObject(),
          main = this.getMainPainter(),
          best_dist2 = 1e10, bestindx = -1, bestpos = null;

      for (let n = 0; n < graph.fNpoints; ++n) {
         let pos = main.translate(graph.fX[n], graph.fY[n]),
             dist2 = (pos.x-pnt.x)**2 + (pos.y-pnt.y)**2;
         if (dist2 < best_dist2) { best_dist2 = dist2; bestindx = n; bestpos = pos; }
      }

      let match_distance = 5;
      if (this.markeratt && this.markeratt.used) match_distance = this.markeratt.getFullSize();

      if (Math.sqrt(best_dist2) > match_distance) return null;

      let res = { name: this.getObject().fName, title: this.getObject().fTitle,
                  x: bestpos.x, y: bestpos.y,
                  color1: this.markeratt && this.markeratt.used ? this.markeratt.color : this.lineatt.color,
                  exact: Math.sqrt(best_dist2) < 4,
                  lines: [ this.getObjectHint() ],
                  binindx: bestindx,
                  menu_dist: match_distance,
                  radius: match_distance
                };

      res.lines.push("r = " + main.axisAsText("r", graph.fY[bestindx]));
      res.lines.push("phi = " + main.axisAsText("phi",graph.fX[bestindx]));

      if (graph.fEY && graph.fEY[bestindx])
         res.lines.push("error r = " + main.axisAsText("r", graph.fEY[bestindx]));

      if (graph.fEX && graph.fEX[bestindx])
         res.lines.push("error phi = " + main.axisAsText("phi", graph.fEX[bestindx]));

      return res;
   }

   /** @summary Show tooltip */
   showTooltip(hint) {

      if (!this.draw_g) return;

      let ttcircle = this.draw_g.select(".tooltip_bin");

      if (!hint) {
         ttcircle.remove();
         return;
      }

      if (ttcircle.empty())
         ttcircle = this.draw_g.append("svg:ellipse")
                             .attr("class","tooltip_bin")
                             .style("pointer-events","none");

      hint.changed = ttcircle.property("current_bin") !== hint.binindx;

      if (hint.changed)
         ttcircle.attr("cx", hint.x)
               .attr("cy", hint.y)
               .attr("rx", Math.round(hint.radius))
               .attr("ry", Math.round(hint.radius))
               .style("fill", "none")
               .style("stroke", hint.color1)
               .property("current_bin", hint.binindx);
   }

   /** @summary Process tooltip event */
   processTooltipEvent(pnt) {
      let hint = this.extractTooltip(pnt);
      if (!pnt || !pnt.disabled) this.showTooltip(hint);
      return hint;
   }

   /** @summary Draw TGraphPolar */
   static draw(dom, graph, opt) {
      let painter = new TGraphPolarPainter(dom, graph);
      painter.decodeOptions(opt);

      let main = painter.getMainPainter();
      if (main && !main.$polargram) {
         console.error('Cannot superimpose TGraphPolar with plain histograms');
         return null;
      }

      let pr = Promise.resolve(null);
      if (!main) {
         if (!graph.fPolargram)
            graph.fPolargram = painter.createPolargram();
         pr = TGraphPolargramPainter.draw(dom, graph.fPolargram);
      }

      return pr.then(() => {
         painter.addToPadPrimitives();
         painter.drawGraphPolar();
         return painter;
      });
   }

} // class TGraphPolarPainter

export { TGraphPolargramPainter, TGraphPolarPainter };
