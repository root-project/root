import { gStyle, settings, create, isBatchMode } from '../core.mjs';
import { pointer as d3_pointer } from '../d3.mjs';
import { getSvgLineStyle } from '../base/TAttLineHandler.mjs';
import { TAxisPainter } from './TAxisPainter.mjs';
import { RAxisPainter } from './RAxisPainter.mjs';
import { FrameInteractive } from './TFramePainter.mjs';
import { RObjectPainter } from '../base/RObjectPainter.mjs';


/**
 * @summary Painter class for RFrame, main handler for interactivity
 *
 * @private
 */

class RFramePainter extends RObjectPainter {

   /** @summary constructor
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} tframe - RFrame object */
   constructor(dom, tframe) {
      super(dom, tframe, "", "frame");
      this.mode3d = false;
      this.xmin = this.xmax = 0; // no scale specified, wait for objects drawing
      this.ymin = this.ymax = 0; // no scale specified, wait for objects drawing
      this.axes_drawn = false;
      this.keys_handler = null;
      this.projection = 0; // different projections
      this.v7_frame = true; // indicator of v7, used in interactive part
   }

   /** @summary Returns frame painter - object itself */
   getFramePainter() { return this; }

   /** @summary Returns true if it is ROOT6 frame
    * @private */
   is_root6() { return false; }

   /** @summary Set active flag for frame - can block some events
    * @private */
   setFrameActive(on) {
      this.enabledKeys = on && settings.HandleKeys ? true : false;
      // used only in 3D mode
      if (this.control)
         this.control.enableKeys = this.enabledKeys;
   }

   setLastEventPos(pnt) {
      // set position of last context menu event, can be
      this.fLastEventPnt = pnt;
   }

   getLastEventPos() {
      // return position of last event
      return this.fLastEventPnt;
   }

   /** @summary Update graphical attributes */
   updateAttributes(force) {
      if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {

         let rect = this.getPadPainter().getPadRect();
         this.fX1NDC = this.v7EvalLength("margins_left", rect.width, settings.FrameNDC.fX1NDC) / rect.width;
         this.fY1NDC = this.v7EvalLength("margins_bottom", rect.height, settings.FrameNDC.fY1NDC) / rect.height;
         this.fX2NDC = 1 - this.v7EvalLength("margins_right", rect.width, 1-settings.FrameNDC.fX2NDC) / rect.width;
         this.fY2NDC = 1 - this.v7EvalLength("margins_top", rect.height, 1-settings.FrameNDC.fY2NDC) / rect.height;
      }

      if (!this.fillatt)
         this.createv7AttFill();

      this.createv7AttLine("border_");
   }

   /** @summary Returns coordinates transformation func */
   getProjectionFunc() {
      switch (this.projection) {
         // Aitoff2xy
         case 1: return (l, b) => {
            const DegToRad = Math.PI/180,
                  alpha2 = (l/2)*DegToRad,
                  delta  = b*DegToRad,
                  r2     = Math.sqrt(2),
                  f      = 2*r2/Math.PI,
                  cdec   = Math.cos(delta),
                  denom  = Math.sqrt(1. + cdec*Math.cos(alpha2));
            return {
               x: cdec*Math.sin(alpha2)*2.*r2/denom/f/DegToRad,
               y: Math.sin(delta)*r2/denom/f/DegToRad
            };
         };
         // mercator
         case 2: return (l, b) => { return { x: l, y: Math.log(Math.tan((Math.PI/2 + b/180*Math.PI)/2)) }; };
         // sinusoidal
         case 3: return (l, b) => { return { x: l*Math.cos(b/180*Math.PI), y: b } };
         // parabolic
         case 4: return (l, b) => { return { x: l*(2.*Math.cos(2*b/180*Math.PI/3) - 1), y: 180*Math.sin(b/180*Math.PI/3) }; };
      }
   }

   /** @summary Rcalculate frame ranges using specified projection functions
     * @desc Not yet used in v7 */
   recalculateRange(Proj) {
      this.projection = Proj || 0;

      if ((this.projection == 2) && ((this.scale_ymin <= -90 || this.scale_ymax >=90))) {
         console.warn("Mercator Projection", "Latitude out of range", this.scale_ymin, this.scale_ymax);
         this.projection = 0;
      }

      let func = this.getProjectionFunc();
      if (!func) return;

      let pnts = [ func(this.scale_xmin, this.scale_ymin),
                   func(this.scale_xmin, this.scale_ymax),
                   func(this.scale_xmax, this.scale_ymax),
                   func(this.scale_xmax, this.scale_ymin) ];
      if (this.scale_xmin<0 && this.scale_xmax>0) {
         pnts.push(func(0, this.scale_ymin));
         pnts.push(func(0, this.scale_ymax));
      }
      if (this.scale_ymin<0 && this.scale_ymax>0) {
         pnts.push(func(this.scale_xmin, 0));
         pnts.push(func(this.scale_xmax, 0));
      }

      this.original_xmin = this.scale_xmin;
      this.original_xmax = this.scale_xmax;
      this.original_ymin = this.scale_ymin;
      this.original_ymax = this.scale_ymax;

      this.scale_xmin = this.scale_xmax = pnts[0].x;
      this.scale_ymin = this.scale_ymax = pnts[0].y;

      for (let n = 1; n < pnts.length; ++n) {
         this.scale_xmin = Math.min(this.scale_xmin, pnts[n].x);
         this.scale_xmax = Math.max(this.scale_xmax, pnts[n].x);
         this.scale_ymin = Math.min(this.scale_ymin, pnts[n].y);
         this.scale_ymax = Math.max(this.scale_ymax, pnts[n].y);
      }
   }

   /** @summary Draw axes grids
     * @desc Called immediately after axes drawing */
   drawGrids() {
      let layer = this.getFrameSvg().select(".grid_layer");

      layer.selectAll(".xgrid").remove();
      layer.selectAll(".ygrid").remove();

      let h = this.getFrameHeight(),
          w = this.getFrameWidth(),
          gridx = this.v7EvalAttr("gridX", false),
          gridy = this.v7EvalAttr("gridY", false),
          grid_style = getSvgLineStyle(gStyle.fGridStyle),
          grid_color = (gStyle.fGridColor > 0) ? this.getColor(gStyle.fGridColor) : "black";

      if (this.x_handle)
         this.x_handle.draw_grid = gridx;

      // add a grid on x axis, if the option is set
      if (this.x_handle && this.x_handle.draw_grid) {
         let grid = "";
         for (let n = 0; n < this.x_handle.ticks.length; ++n)
            if (this.swap_xy)
               grid += "M0,"+(h+this.x_handle.ticks[n])+"h"+w;
            else
               grid += "M"+this.x_handle.ticks[n]+",0v"+h;

         if (grid.length > 0)
            layer.append("svg:path")
                 .attr("class", "xgrid")
                 .attr("d", grid)
                 .style('stroke',grid_color)
                 .style("stroke-width", gStyle.fGridWidth)
                 .style("stroke-dasharray", grid_style);
      }

      if (this.y_handle)
         this.y_handle.draw_grid = gridy;

      // add a grid on y axis, if the option is set
      if (this.y_handle && this.y_handle.draw_grid) {
         let grid = "";
         for (let n = 0; n < this.y_handle.ticks.length; ++n)
            if (this.swap_xy)
               grid += "M"+this.y_handle.ticks[n]+",0v"+h;
            else
               grid += "M0,"+(h+this.y_handle.ticks[n])+"h"+w;

         if (grid.length > 0)
          layer.append("svg:path")
               .attr("class", "ygrid")
               .attr("d", grid)
               .style('stroke', grid_color)
               .style("stroke-width", gStyle.fGridWidth)
               .style("stroke-dasharray", grid_style);
      }
   }

   /** @summary Converts "raw" axis value into text */
   axisAsText(axis, value) {
      let handle = this[axis+"_handle"];

      if (handle)
         return handle.axisAsText(value, settings[axis.toUpperCase() + "ValuesFormat"]);

      return value.toPrecision(4);
   }

   /** @summary Set axix range */
   _setAxisRange(prefix, vmin, vmax) {
      let nmin = prefix + "min", nmax = prefix + "max";
      if (this[nmin] != this[nmax]) return;
      let min = this.v7EvalAttr(prefix + "_min"),
          max = this.v7EvalAttr(prefix + "_max");

      if (min !== undefined) vmin = min;
      if (max !== undefined) vmax = max;

      if (vmin < vmax) {
         this[nmin] = vmin;
         this[nmax] = vmax;
      }

      let nzmin = "zoom_" + prefix + "min", nzmax = "zoom_" + prefix + "max";

      if ((this[nzmin] == this[nzmax]) && !this.zoomChangedInteractive(prefix)) {
         min = this.v7EvalAttr(prefix + "_zoomMin");
         max = this.v7EvalAttr(prefix + "_zoomMax");

         if ((min !== undefined) || (max !== undefined)) {
            this[nzmin] = (min === undefined) ? this[nmin] : min;
            this[nzmax] = (max === undefined) ? this[nmax] : max;
         }
      }
   }

   /** @summary Set axes ranges for drawing, check configured attributes if range already specified */
   setAxesRanges(xaxis, xmin, xmax, yaxis, ymin, ymax, zaxis, zmin, zmax) {
      if (this.axes_drawn) return;
      this.xaxis = xaxis;
      this._setAxisRange("x", xmin, xmax);
      this.yaxis = yaxis;
      this._setAxisRange("y", ymin, ymax);
      this.zaxis = zaxis;
      this._setAxisRange("z", zmin, zmax);
   }

   /** @summary Set secondary axes ranges */
   setAxes2Ranges(second_x, xaxis, xmin, xmax, second_y, yaxis, ymin, ymax) {
      if (second_x) {
         this.x2axis = xaxis;
         this._setAxisRange("x2", xmin, xmax);
      }
      if (second_y) {
         this.y2axis = yaxis;
         this._setAxisRange("y2", ymin, ymax);
      }
   }

   /** @summary Create x,y objects which maps user coordinates into pixels
     * @desc Must be used only for v6 objects, see TFramePainter for more details
     * @private */
   createXY(opts) {
      if (this.self_drawaxes) return;

      this.cleanXY(); // remove all previous configurations

      if (!opts) opts = {};

      this.v6axes = true;
      this.swap_xy = opts.swap_xy || false;
      this.reverse_x = opts.reverse_x || false;
      this.reverse_y = opts.reverse_y || false;

      this.logx = this.v7EvalAttr("x_log", 0);
      this.logy = this.v7EvalAttr("y_log", 0);

      let w = this.getFrameWidth(), h = this.getFrameHeight();

      this.scale_xmin = this.xmin;
      this.scale_xmax = this.xmax;

      this.scale_ymin = this.ymin;
      this.scale_ymax = this.ymax;

      if (opts.extra_y_space) {
         let log_scale = this.swap_xy ? this.logx : this.logy;
         if (log_scale && (this.scale_ymax > 0))
            this.scale_ymax = Math.exp(Math.log(this.scale_ymax)*1.1);
         else
            this.scale_ymax += (this.scale_ymax - this.scale_ymin)*0.1;
      }

      // if (opts.check_pad_range) {
         // take zooming out of pad or axis attributes - skip!
      // }

      if ((this.zoom_ymin == this.zoom_ymax) && (opts.zoom_ymin != opts.zoom_ymax) && !this.zoomChangedInteractive("y")) {
         this.zoom_ymin = opts.zoom_ymin;
         this.zoom_ymax = opts.zoom_ymax;
      }

      if (this.zoom_xmin != this.zoom_xmax) {
         this.scale_xmin = this.zoom_xmin;
         this.scale_xmax = this.zoom_xmax;
      }

      if (this.zoom_ymin != this.zoom_ymax) {
         this.scale_ymin = this.zoom_ymin;
         this.scale_ymax = this.zoom_ymax;
      }

      let xaxis = this.xaxis, yaxis = this.yaxis;
      if (xaxis?._typename != "TAxis") xaxis = create("TAxis");
      if (yaxis?._typename != "TAxis") yaxis = create("TAxis");

      this.x_handle = new TAxisPainter(this.getDom(), xaxis, true);
      this.x_handle.setPadName(this.getPadName());
      this.x_handle.optionUnlab = this.v7EvalAttr("x_labels_hide", false);

      this.x_handle.configureAxis("xaxis", this.xmin, this.xmax, this.scale_xmin, this.scale_xmax, this.swap_xy, this.swap_xy ? [0,h] : [0,w],
                                      { reverse: this.reverse_x,
                                        log: this.swap_xy ? this.logy : this.logx,
                                        symlog: this.swap_xy ? opts.symlog_y : opts.symlog_x,
                                        logcheckmin: this.swap_xy,
                                        logminfactor: 0.0001 });

      this.x_handle.assignFrameMembers(this,"x");

      this.y_handle = new TAxisPainter(this.getDom(), yaxis, true);
      this.y_handle.setPadName(this.getPadName());
      this.y_handle.optionUnlab = this.v7EvalAttr("y_labels_hide", false);

      this.y_handle.configureAxis("yaxis", this.ymin, this.ymax, this.scale_ymin, this.scale_ymax, !this.swap_xy, this.swap_xy ? [0,w] : [0,h],
                                      { reverse: this.reverse_y,
                                        log: this.swap_xy ? this.logx : this.logy,
                                        symlog: this.swap_xy ? opts.symlog_x : opts.symlog_y,
                                        logcheckmin: (opts.ndim < 2) || this.swap_xy,
                                        log_min_nz: opts.ymin_nz && (opts.ymin_nz < 0.01*this.ymax) ? 0.3 * opts.ymin_nz : 0,
                                        logminfactor: 3e-4 });

      this.y_handle.assignFrameMembers(this,"y");
   }

   /** @summary Identify if requested axes are drawn
     * @desc Checks if x/y axes are drawn. Also if second side is already there */
   hasDrawnAxes(second_x, second_y) {
      return !second_x && !second_y ? this.axes_drawn : false;
   }

   /** @summary Draw configured axes on the frame
     * @desc axes can be drawn only for main histogram  */
   drawAxes() {

      if (this.axes_drawn || (this.xmin == this.xmax) || (this.ymin == this.ymax))
         return Promise.resolve(this.axes_drawn);

      let ticksx = this.v7EvalAttr("ticksX", 1),
          ticksy = this.v7EvalAttr("ticksY", 1),
          sidex = 1, sidey = 1;

      if (this.v7EvalAttr("swapX", false)) sidex = -1;
      if (this.v7EvalAttr("swapY", false)) sidey = -1;

      let w = this.getFrameWidth(), h = this.getFrameHeight();

      if (!this.v6axes) {
         // this is partially same as v6 createXY method

         this.cleanupAxes();

         this.swap_xy = false;

         if (this.zoom_xmin != this.zoom_xmax) {
            this.scale_xmin = this.zoom_xmin;
            this.scale_xmax = this.zoom_xmax;
         } else {
            this.scale_xmin = this.xmin;
            this.scale_xmax = this.xmax;
         }

         if (this.zoom_ymin != this.zoom_ymax) {
            this.scale_ymin = this.zoom_ymin;
            this.scale_ymax = this.zoom_ymax;
         } else {
            this.scale_ymin = this.ymin;
            this.scale_ymax = this.ymax;
         }

         this.recalculateRange(0);

         this.x_handle = new RAxisPainter(this.getDom(), this, this.xaxis, "x_");
         this.x_handle.setPadName(this.getPadName());
         this.x_handle.snapid = this.snapid;
         this.x_handle.draw_swapside = (sidex < 0);
         this.x_handle.draw_ticks = ticksx;

         this.y_handle = new RAxisPainter(this.getDom(), this, this.yaxis, "y_");
         this.y_handle.setPadName(this.getPadName());
         this.y_handle.snapid = this.snapid;
         this.y_handle.draw_swapside = (sidey < 0);
         this.y_handle.draw_ticks = ticksy;

         this.z_handle = new RAxisPainter(this.getDom(), this, this.zaxis, "z_");
         this.z_handle.setPadName(this.getPadName());
         this.z_handle.snapid = this.snapid;

         this.x_handle.configureAxis("xaxis", this.xmin, this.xmax, this.scale_xmin, this.scale_xmax, false, [0,w], w, { reverse: false });
         this.x_handle.assignFrameMembers(this,"x");

         this.y_handle.configureAxis("yaxis", this.ymin, this.ymax, this.scale_ymin, this.scale_ymax, true, [h,0], -h, { reverse: false });
         this.y_handle.assignFrameMembers(this,"y");

         // only get basic properties like log scale
         this.z_handle.configureZAxis("zaxis", this);
      }

      let layer = this.getFrameSvg().select(".axis_layer");

      this.x_handle.has_obstacle = false;

      let draw_horiz = this.swap_xy ? this.y_handle : this.x_handle,
          draw_vertical = this.swap_xy ? this.x_handle : this.y_handle,
          pp = this.getPadPainter(), pr;

      if (pp?._fast_drawing) {
         pr = Promise.resolve(true); // do nothing
      } else if (this.v6axes) {

         // in v7 ticksx/y values shifted by 1 relative to v6
         // In v7 ticksx==0 means no ticks, ticksx==1 equivalent to ==0 in v6

         let can_adjust_frame = false, disable_x_draw = false, disable_y_draw = false;

         draw_horiz.disable_ticks = (ticksx <= 0);
         draw_vertical.disable_ticks = (ticksy <= 0);

         let pr1 = draw_horiz.drawAxis(layer, w, h,
                                   draw_horiz.invert_side ? undefined : `translate(0,${h})`,
                                   (ticksx > 1) ? -h : 0, disable_x_draw,
                                   undefined, false);

         let pr2 =  draw_vertical.drawAxis(layer, w, h,
                                      draw_vertical.invert_side ? `translate(${w})` : undefined,
                                      (ticksy > 1) ? w : 0, disable_y_draw,
                                      draw_vertical.invert_side ? 0 : this._frame_x, can_adjust_frame);

         pr = Promise.all([pr1,pr2]).then(() => this.drawGrids());

      } else {

         let arr = [];

         if (ticksx > 0)
            arr.push(draw_horiz.drawAxis(layer, (sidex > 0) ? `translate(0,${h})` : "", sidex));

         if (ticksy > 0)
            arr.push(draw_vertical.drawAxis(layer, (sidey > 0) ? `translate(0,${h})` : `translate(${w},${h})`, sidey));

         pr = Promise.all(arr).then(() => {
            arr = [];
            if (ticksx > 1)
               arr.push(draw_horiz.drawAxisOtherPlace(layer, (sidex < 0) ? `translate(0,${h})` : "", -sidex, ticksx == 2));

            if (ticksy > 1)
               arr.push(draw_vertical.drawAxisOtherPlace(layer, (sidey < 0) ? `translate(0,${h})` : `translate(${w},${h})`, -sidey, ticksy == 2));
            return Promise.all(arr);
         }).then(() => this.drawGrids());
      }

      return pr.then(() => {
         this.axes_drawn = true;
         return true;
      });
   }

   /** @summary Draw secondary configuread axes */
   drawAxes2(second_x, second_y) {
      let w = this.getFrameWidth(), h = this.getFrameHeight(),
          layer = this.getFrameSvg().select(".axis_layer"),
          pr1, pr2;

      if (second_x) {
         if (this.zoom_x2min != this.zoom_x2max) {
            this.scale_x2min = this.zoom_x2min;
            this.scale_x2max = this.zoom_x2max;
         } else {
           this.scale_x2min = this.x2min;
           this.scale_x2max = this.x2max;
         }
         this.x2_handle = new RAxisPainter(this.getDom(), this, this.x2axis, "x2_");
         this.x2_handle.setPadName(this.getPadName());
         this.x2_handle.snapid = this.snapid;

         this.x2_handle.configureAxis("x2axis", this.x2min, this.x2max, this.scale_x2min, this.scale_x2max, false, [0,w], w, { reverse: false });
         this.x2_handle.assignFrameMembers(this,"x2");

         pr1 = this.x2_handle.drawAxis(layer, "", -1);
      }

      if (second_y) {
         if (this.zoom_y2min != this.zoom_y2max) {
            this.scale_y2min = this.zoom_y2min;
            this.scale_y2max = this.zoom_y2max;
         } else {
            this.scale_y2min = this.y2min;
            this.scale_y2max = this.y2max;
         }

         this.y2_handle = new RAxisPainter(this.getDom(), this, this.y2axis, "y2_");
         this.y2_handle.setPadName(this.getPadName());
         this.y2_handle.snapid = this.snapid;

         this.y2_handle.configureAxis("y2axis", this.y2min, this.y2max, this.scale_y2min, this.scale_y2max, true, [h,0], -h, { reverse: false });
         this.y2_handle.assignFrameMembers(this,"y2");

         pr2 = this.y2_handle.drawAxis(layer, `translate(${w},${h})`, -1);
      }

      return Promise.all([pr1,pr2]);
   }

   /** @summary Return functions to create x/y points based on coordinates
     * @desc In default case returns frame painter itself
     * @private */
   getGrFuncs(second_x, second_y) {
      let use_x2 = second_x && this.grx2,
          use_y2 = second_y && this.gry2;
      if (!use_x2 && !use_y2) return this;

      return {
         use_x2: use_x2,
         grx: use_x2 ? this.grx2 : this.grx,
         x_handle: use_x2 ? this.x2_handle : this.x_handle,
         logx: use_x2 ? this.x2_handle.log : this.x_handle.log,
         scale_xmin: use_x2 ? this.scale_x2min : this.scale_xmin,
         scale_xmax: use_x2 ? this.scale_x2max : this.scale_xmax,
         use_y2: use_y2,
         gry: use_y2 ? this.gry2 : this.gry,
         y_handle: use_y2 ? this.y2_handle : this.y_handle,
         logy: use_y2 ? this.y2_handle.log : this.y_handle.log,
         scale_ymin: use_y2 ? this.scale_y2min : this.scale_ymin,
         scale_ymax: use_y2 ? this.scale_y2max : this.scale_ymax,
         swap_xy: this.swap_xy,
         fp: this,
         revertAxis(name, v) {
            if ((name == "x") && this.use_x2) name = "x2";
            if ((name == "y") && this.use_y2) name = "y2";
            return this.fp.revertAxis(name, v);
         },
         axisAsText(name, v) {
            if ((name == "x") && this.use_x2) name = "x2";
            if ((name == "y") && this.use_y2) name = "y2";
            return this.fp.axisAsText(name, v);
         }
      };
   }

   /** @summary function called at the end of resize of frame
     * @desc Used to update attributes on the server
     * @private */
   sizeChanged() {

      let changes = {};
      this.v7AttrChange(changes, "margins_left", this.fX1NDC);
      this.v7AttrChange(changes, "margins_bottom", this.fY1NDC);
      this.v7AttrChange(changes, "margins_right", 1 - this.fX2NDC);
      this.v7AttrChange(changes, "margins_top", 1 - this.fY2NDC);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server

      this.redrawPad();
   }

   /** @summary Remove all x/y functions
     * @private */
   cleanXY() {
      // remove all axes drawings
      let clean = (name,grname) => {
         if (this[name]) {
            this[name].cleanup();
            delete this[name];
         }
         delete this[grname];
      };

      clean("x_handle", "grx");
      clean("y_handle", "gry");
      clean("z_handle", "grz");
      clean("x2_handle", "grx2");
      clean("y2_handle", "gry2");

      delete this.v6axes; // marker that v6 axes are used
   }

   /** @summary Remove all axes drawings
     * @private */
   cleanupAxes() {
      this.cleanXY();

      if (this.draw_g) {
         this.draw_g.select(".grid_layer").selectAll("*").remove();
         this.draw_g.select(".axis_layer").selectAll("*").remove();
      }
      this.axes_drawn = false;
   }

   /** @summary Removes all drawn elements of the frame
     * @private */
   cleanFrameDrawings() {
      // cleanup all 3D drawings if any
      if (typeof this.create3DScene === 'function')
         this.create3DScene(-1);

      this.cleanupAxes();

      let clean = (name) => {
         this[name+"min"] = this[name+"max"] = 0;
         this["zoom_"+name+"min"] = this["zoom_"+name+"max"] = 0;
         this["scale_"+name+"min"] = this["scale_"+name+"max"] = 0;
      };

      clean("x");
      clean("y");
      clean("z");
      clean("x2");
      clean("y2");

      if (this.draw_g) {
         this.draw_g.select(".main_layer").selectAll("*").remove();
         this.draw_g.select(".upper_layer").selectAll("*").remove();
      }
   }

   /** @summary Fully cleanup frame
     * @private */
   cleanup() {

      this.cleanFrameDrawings();

      if (this.draw_g) {
         this.draw_g.selectAll("*").remove();
         this.draw_g.on("mousedown", null)
                    .on("dblclick", null)
                    .on("wheel", null)
                    .on("contextmenu", null)
                    .property('interactive_set', null);
      }

      if (this.keys_handler) {
         window.removeEventListener('keydown', this.keys_handler, false);
         this.keys_handler = null;
      }
      delete this.enabledKeys;
      delete this.self_drawaxes;

      delete this.xaxis;
      delete this.yaxis;
      delete this.zaxis;
      delete this.x2axis;
      delete this.y2axis;

      delete this.draw_g; // frame <g> element managet by the pad

      delete this._click_handler;
      delete this._dblclick_handler;

      let pp = this.getPadPainter();
      if (pp?.frame_painter_ref === this)
         delete pp.frame_painter_ref;

      super.cleanup();
   }

   /** @summary Redraw frame
     * @private */
   redraw() {

      let pp = this.getPadPainter();
      if (pp) pp.frame_painter_ref = this;

      // first update all attributes from objects
      this.updateAttributes();

      let rect = pp ? pp.getPadRect() : { width: 10, height: 10 },
          lm = Math.round(rect.width * this.fX1NDC),
          w = Math.round(rect.width * (this.fX2NDC - this.fX1NDC)),
          tm = Math.round(rect.height * (1 - this.fY2NDC)),
          h = Math.round(rect.height * (this.fY2NDC - this.fY1NDC)),
          rotate = false, fixpos = false, trans;

      if (pp && pp.options) {
         if (pp.options.RotateFrame) rotate = true;
         if (pp.options.FixFrame) fixpos = true;
      }

      if (rotate) {
         trans = `rotate(-90,${lm},${tm}) translate(${lm-h},${tm})`;
         let d = w; w = h; h = d;
      } else {
         trans = `translate(${lm},${tm})`;
      }

      // update values here to let access even when frame is not really updated
      this._frame_x = lm;
      this._frame_y = tm;
      this._frame_width = w;
      this._frame_height = h;
      this._frame_rotate = rotate;
      this._frame_fixpos = fixpos;

      if (this.mode3d) return this; // no need for real draw in mode3d

      // this is svg:g object - container for every other items belonging to frame
      this.draw_g = this.getFrameSvg();

      let top_rect, main_svg;

      if (this.draw_g.empty()) {

         this.draw_g = this.getLayerSvg("primitives_layer").append("svg:g").attr("class", "root_frame");

         if (!isBatchMode())
            this.draw_g.append("svg:title").text("");

         top_rect = this.draw_g.append("svg:rect");

         // append for the moment three layers - for drawing and axis
         this.draw_g.append('svg:g').attr('class','grid_layer');

         main_svg = this.draw_g.append('svg:svg')
                           .attr('class','main_layer')
                           .attr("x", 0)
                           .attr("y", 0)
                           .attr('overflow', 'hidden');

         this.draw_g.append('svg:g').attr('class','axis_layer');
         this.draw_g.append('svg:g').attr('class','upper_layer');
      } else {
         top_rect = this.draw_g.select("rect");
         main_svg = this.draw_g.select(".main_layer");
      }

      this.axes_drawn = false;

      this.draw_g.attr("transform", trans);

      top_rect.attr("x", 0)
              .attr("y", 0)
              .attr("width", w)
              .attr("height", h)
              .attr("rx", this.lineatt.rx || null)
              .attr("ry", this.lineatt.ry || null)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      main_svg.attr("width", w)
              .attr("height", h)
              .attr("viewBox", `0 0 ${w} ${h}`);

      let pr = Promise.resolve(true);

      if (this.v7EvalAttr("drawAxes")) {
         this.self_drawaxes = true;
         this.setAxesRanges();
         pr = this.drawAxes().then(() => this.addInteractivity());
      }

      return pr.then(() => {
         if (!isBatchMode()) {
            top_rect.style("pointer-events", "visibleFill");  // let process mouse events inside frame

            FrameInteractive.assign(this);
            this.addBasicInteractivity();
         }

         return this;
      });
   }

   /** @summary Returns frame width */
   getFrameWidth() { return this._frame_width || 0; }

   /** @summary Returns frame height */
   getFrameHeight() { return this._frame_height || 0; }

   /** @summary Returns frame rectangle plus extra info for hint display */
   getFrameRect() {
      return {
         x: this._frame_x || 0,
         y: this._frame_y || 0,
         width: this.getFrameWidth(),
         height: this.getFrameHeight(),
         transform: this.draw_g ? this.draw_g.attr("transform") : "",
         hint_delta_x: 0,
         hint_delta_y: 0
      }
   }

   /** @summary Returns palette associated with frame */
   getHistPalette() {
      return this.getPadPainter().getHistPalette();
   }

   /** @summary Configure user-defined click handler
     * @desc Function will be called every time when frame click was perfromed
     * As argument, tooltip object with selected bins will be provided
     * If handler function returns true, default handling of click will be disabled */
   configureUserClickHandler(handler) {
      this._click_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   /** @summary Configure user-defined dblclick handler
     * @desc Function will be called every time when double click was called
     * As argument, tooltip object with selected bins will be provided
     * If handler function returns true, default handling of dblclick (unzoom) will be disabled */
   configureUserDblclickHandler(handler) {
      this._dblclick_handler = handler && (typeof handler == 'function') ? handler : null;
   }

   /** @summary function can be used for zooming into specified range
     * @desc if both limits for each axis 0 (like xmin==xmax==0), axis will be unzoomed
     * @returns {Promise} with boolean flag if zoom operation was performed */
   zoom(xmin, xmax, ymin, ymax, zmin, zmax) {

      // disable zooming when axis conversion is enabled
      if (this.projection) return Promise.resolve(false);

      if (xmin==="x") { xmin = xmax; xmax = ymin; ymin = undefined; } else
      if (xmin==="y") { ymax = ymin; ymin = xmax; xmin = xmax = undefined; } else
      if (xmin==="z") { zmin = xmax; zmax = ymin; xmin = xmax = ymin = undefined; }

      let zoom_x = (xmin !== xmax), zoom_y = (ymin !== ymax), zoom_z = (zmin !== zmax),
          unzoom_x = false, unzoom_y = false, unzoom_z = false;

      if (zoom_x) {
         let cnt = 0;
         if (xmin <= this.xmin) { xmin = this.xmin; cnt++; }
         if (xmax >= this.xmax) { xmax = this.xmax; cnt++; }
         if (cnt === 2) { zoom_x = false; unzoom_x = true; }
      } else {
         unzoom_x = (xmin === xmax) && (xmin === 0);
      }

      if (zoom_y) {
         let cnt = 0;
         if (ymin <= this.ymin) { ymin = this.ymin; cnt++; }
         if (ymax >= this.ymax) { ymax = this.ymax; cnt++; }
         if (cnt === 2) { zoom_y = false; unzoom_y = true; }
      } else {
         unzoom_y = (ymin === ymax) && (ymin === 0);
      }

      if (zoom_z) {
         let cnt = 0;
         // if (this.logz && this.ymin_nz && this.getDimension()===2) main_zmin = 0.3*this.ymin_nz;
         if (zmin <= this.zmin) { zmin = this.zmin; cnt++; }
         if (zmax >= this.zmax) { zmax = this.zmax; cnt++; }
         if (cnt === 2) { zoom_z = false; unzoom_z = true; }
      } else {
         unzoom_z = (zmin === zmax) && (zmin === 0);
      }

      let changed = false,
          r_x = "", r_y = "", r_z = "", is_any_check = false,
         req = {
            _typename: "ROOT::Experimental::RFrame::RUserRanges",
            values: [0, 0, 0, 0, 0, 0],
            flags: [false, false, false, false, false, false]
         };

      const checkZooming = (painter, force) => {
         if (!force && (typeof painter.canZoomInside != 'function')) return;

         is_any_check = true;

         if (zoom_x && (force || painter.canZoomInside("x", xmin, xmax))) {
            this.zoom_xmin = xmin;
            this.zoom_xmax = xmax;
            changed = true; r_x = "0";
            zoom_x = false;
            req.values[0] = xmin; req.values[1] = xmax;
            req.flags[0] = req.flags[1] = true;
         }
         if (zoom_y && (force || painter.canZoomInside("y", ymin, ymax))) {
            this.zoom_ymin = ymin;
            this.zoom_ymax = ymax;
            changed = true; r_y = "1";
            zoom_y = false;
            req.values[2] = ymin; req.values[3] = ymax;
            req.flags[2] = req.flags[3] = true;
         }
         if (zoom_z && (force || painter.canZoomInside("z", zmin, zmax))) {
            this.zoom_zmin = zmin;
            this.zoom_zmax = zmax;
            changed = true; r_z = "2";
            zoom_z = false;
            req.values[4] = zmin; req.values[5] = zmax;
            req.flags[4] = req.flags[5] = true;
         }
      };

      // first process zooming (if any)
      if (zoom_x || zoom_y || zoom_z)
         this.forEachPainter(painter => checkZooming(painter));

      // force zooming when no any other painter can verify zoom range
      if (!is_any_check && this.self_drawaxes)
         checkZooming(null, true);

      // and process unzoom, if any
      if (unzoom_x || unzoom_y || unzoom_z) {
         if (unzoom_x) {
            if (this.zoom_xmin !== this.zoom_xmax) { changed = true; r_x = "0"; }
            this.zoom_xmin = this.zoom_xmax = 0;
            req.values[0] = req.values[1] = -1;
         }
         if (unzoom_y) {
            if (this.zoom_ymin !== this.zoom_ymax) { changed = true; r_y = "1"; }
            this.zoom_ymin = this.zoom_ymax = 0;
            req.values[2] = req.values[3] = -1;
         }
         if (unzoom_z) {
            if (this.zoom_zmin !== this.zoom_zmax) { changed = true; r_z = "2"; }
            this.zoom_zmin = this.zoom_zmax = 0;
            req.values[4] = req.values[5] = -1;
         }
      }

      if (!changed) return Promise.resolve(false);

      if (this.v7NormalMode())
         this.v7SubmitRequest("zoom", { _typename: "ROOT::Experimental::RFrame::RZoomRequest", ranges: req });

      return this.interactiveRedraw("pad", "zoom" + r_x + r_y + r_z).then(() => true);
   }

   /** @summary Provide zooming of single axis
     * @desc One can specify names like x/y/z but also second axis x2 or y2 */
   zoomSingle(name, vmin, vmax) {

      let names = ["x","y","z","x2","y2"], indx = names.indexOf(name);

      // disable zooming when axis conversion is enabled
      if (this.projection || !this[name+"_handle"] || (indx < 0))
         return Promise.resolve(false);

      let zoom_v = (vmin !== vmax), unzoom_v = false;

      if (zoom_v) {
         let cnt = 0;
         if (vmin <= this[name+"min"]) { vmin = this[name+"min"]; cnt++; }
         if (vmax >= this[name+"max"]) { vmax = this[name+"max"]; cnt++; }
         if (cnt === 2) { zoom_v = false; unzoom_v = true; }
      } else {
         unzoom_v = (vmin === vmax) && (vmin === 0);
      }

      let changed = false, is_any_check = false,
          req = {
             _typename: "ROOT::Experimental::RFrame::RUserRanges",
             values: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             flags: [false, false, false, false, false, false, false, false, false, false]
          };

      let checkZooming = (painter, force) => {
         if (!force && (typeof painter?.canZoomInside != 'function')) return;

         is_any_check = true;

         if (zoom_v && (force || painter.canZoomInside(name[0], vmin, vmax))) {
            this["zoom_" + name + "min"] = vmin;
            this["zoom_" + name + "max"] = vmax;
            changed = true;
            zoom_v = false;
            req.values[indx*2] = vmin; req.values[indx*2+1] = vmax;
            req.flags[indx*2] = req.flags[indx*2+1] = true;
         }
      };

      // first process zooming (if any)
      if (zoom_v)
         this.forEachPainter(painter => checkZooming(painter));

      // force zooming when no any other painter can verify zoom range
      if (!is_any_check && this.self_drawaxes)
         checkZooming(null, true);

      if (unzoom_v) {
         if (this[`zoom_${name}min`] !== this[`zoom_${name}max`]) changed = true;
         this[`zoom_${name}min`] = this[`zoom_${name}max`] = 0;
         req.values[indx*2] = req.values[indx*2+1] = -1;
      }

      if (!changed) return Promise.resolve(false);

      if (this.v7NormalMode())
         this.v7SubmitRequest("zoom", { _typename: "ROOT::Experimental::RFrame::RZoomRequest", ranges: req });

      return this.interactiveRedraw("pad", "zoom" + indx).then(() => true);
   }

   /** @summary Checks if specified axis zoomed */
   isAxisZoomed(axis) {
      return this['zoom_'+axis+'min'] !== this['zoom_'+axis+'max'];
   }

   /** @summary Unzoom specified axes
     * @returns {Promise} with boolean flag if zoom is changed */
   unzoom(dox, doy, doz) {
      if (dox == "all")
         return this.unzoom("x2").then(() => this.unzoom("y2")).then(() => this.unzoom("xyz"));

      if ((dox == "x2") || (dox == "y2"))
         return this.zoomSingle(dox, 0, 0).then(changed => {
            if (changed) this.zoomChangedInteractive(dox, "unzoom");
            return changed;
         });

      if (typeof dox === 'undefined') { dox = doy = doz = true; } else
      if (typeof dox === 'string') { doz = dox.indexOf("z") >= 0; doy = dox.indexOf("y") >= 0; dox = dox.indexOf("x") >= 0; }

      return this.zoom(dox ? 0 : undefined, dox ? 0 : undefined,
                       doy ? 0 : undefined, doy ? 0 : undefined,
                       doz ? 0 : undefined, doz ? 0 : undefined).then(changed => {

         if (changed && dox) this.zoomChangedInteractive("x", "unzoom");
         if (changed && doy) this.zoomChangedInteractive("y", "unzoom");
         if (changed && doz) this.zoomChangedInteractive("z", "unzoom");

         return changed;
      });
   }

   /** @summary Mark/check if zoom for specific axis was changed interactively
     * @private */
   zoomChangedInteractive(axis, value) {
      if (axis == 'reset') {
         this.zoom_changed_x = this.zoom_changed_y = this.zoom_changed_z = undefined;
         return;
      }
      if (!axis || axis == 'any')
         return this.zoom_changed_x || this.zoom_changed_y  || this.zoom_changed_z;

      if ((axis !== 'x') && (axis !== 'y') && (axis !== 'z')) return;

      let fld = "zoom_changed_" + axis;
      if (value === undefined) return this[fld];

      if (value === 'unzoom') {
         // special handling of unzoom, only if was never changed before flag set to true
         this[fld] = (this[fld] === undefined);
         return;
      }

      if (value) this[fld] = true;
   }

   /** @summary Fill menu for frame when server is not there */
   fillObjectOfflineMenu(menu, kind) {
      if ((kind!="x") && (kind!="y")) return;

      menu.add("Unzoom", () => this.unzoom(kind));

      //if (this[kind+"_kind"] == "normal")
      //   menu.addchk(this["log"+kind], "SetLog"+kind, this.toggleAxisLog.bind(this, kind));

      // here should be all axes attributes in offline
   }

   /** @summary Set grid drawing for specified axis */
   changeFrameAttr(attr, value) {
      let changes = {};
      this.v7AttrChange(changes, attr, value);
      this.v7SetAttr(attr, value);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server
      this.redrawPad();
   }

   /** @summary Fill context menu */
   fillContextMenu(menu, kind, /* obj */) {

      // when fill and show context menu, remove all zooming

      if ((kind=="x") || (kind=="y") || (kind=="x2") || (kind=="y2")) {
         let handle = this[kind+"_handle"];
         if (!handle) return false;
         menu.add("header: " + kind.toUpperCase() + " axis");
         return handle.fillAxisContextMenu(menu, kind);
      }

      let alone = menu.size()==0;

      if (alone)
         menu.add("header:Frame");
      else
         menu.add("separator");

      if (this.zoom_xmin !== this.zoom_xmax)
         menu.add("Unzoom X", () => this.unzoom("x"));
      if (this.zoom_ymin !== this.zoom_ymax)
         menu.add("Unzoom Y", () => this.unzoom("y"));
      if (this.zoom_zmin !== this.zoom_zmax)
         menu.add("Unzoom Z", () => this.unzoom("z"));
      if (this.zoom_x2min !== this.zoom_x2max)
         menu.add("Unzoom X2", () => this.unzoom("x2"));
      if (this.zoom_y2min !== this.zoom_y2max)
         menu.add("Unzoom Y2", () => this.unzoom("y2"));
      menu.add("Unzoom all", () => this.unzoom("all"));

      menu.add("separator");

      menu.addchk(this.isTooltipAllowed(), "Show tooltips", () => this.setTooltipAllowed("toggle"));

      if (this.x_handle)
         menu.addchk(this.x_handle.draw_grid, "Grid x", flag => this.changeFrameAttr("gridX", flag));
      if (this.y_handle)
         menu.addchk(this.y_handle.draw_grid, "Grid y", flag => this.changeFrameAttr("gridY", flag));
      if (this.x_handle && !this.x2_handle)
         menu.addchk(this.x_handle.draw_swapside, "Swap x", flag => this.changeFrameAttr("swapX", flag));
      if (this.y_handle && !this.y2_handle)
         menu.addchk(this.y_handle.draw_swapside, "Swap y", flag => this.changeFrameAttr("swapY", flag));
      if (this.x_handle && !this.x2_handle) {
         menu.add("sub:Ticks x");
         menu.addchk(this.x_handle.draw_ticks == 0, "off", () => this.changeFrameAttr("ticksX", 0));
         menu.addchk(this.x_handle.draw_ticks == 1, "normal", () => this.changeFrameAttr("ticksX", 1));
         menu.addchk(this.x_handle.draw_ticks == 2, "ticks on both sides", () => this.changeFrameAttr("ticksX", 2));
         menu.addchk(this.x_handle.draw_ticks == 3, "labels on both sides", () => this.changeFrameAttr("ticksX", 3));
         menu.add("endsub:");
       }
      if (this.y_handle && !this.y2_handle) {
         menu.add("sub:Ticks y");
         menu.addchk(this.y_handle.draw_ticks == 0, "off", () => this.changeFrameAttr("ticksY", 0));
         menu.addchk(this.y_handle.draw_ticks == 1, "normal", () => this.changeFrameAttr("ticksY", 1));
         menu.addchk(this.y_handle.draw_ticks == 2, "ticks on both sides", () => this.changeFrameAttr("ticksY", 2));
         menu.addchk(this.y_handle.draw_ticks == 3, "labels on both sides", () => this.changeFrameAttr("ticksY", 3));
         menu.add("endsub:");
       }

      menu.addAttributesMenu(this, alone ? "" : "Frame ");
      menu.add("separator");
      menu.add("Save as frame.png", () => this.getPadPainter().saveAs("png", 'frame', 'frame.png'));
      menu.add("Save as frame.svg", () => this.getPadPainter().saveAs("svg", 'frame', 'frame.svg'));

      return true;
   }

   /** @summary Convert graphical coordinate into axis value */
   revertAxis(axis, pnt) {
      let handle = this[axis+"_handle"];
      return handle ? handle.revertPoint(pnt) : 0;
   }

   /** @summary Show axis status message
     * @desc method called normally when mouse enter main object element
     * @private */
   showAxisStatus(axis_name, evnt) {

      let taxis = null, hint_name = axis_name, hint_title = "axis",
          m = d3_pointer(evnt, this.getFrameSvg().node()), id = (axis_name=="x") ? 0 : 1;

      if (taxis) { hint_name = taxis.fName; hint_title = taxis.fTitle || "axis object"; }

      if (this.swap_xy) id = 1-id;

      let axis_value = this.revertAxis(axis_name, m[id]);

      this.showObjectStatus(hint_name, hint_title, axis_name + " : " + this.axisAsText(axis_name, axis_value), Math.round(m[0])+","+Math.round(m[1]));
   }

   /** @summary Add interactive keys handlers
    * @private */
   addKeysHandler() {
      if (isBatchMode()) return;
      FrameInteractive.assign(this);
      this.addFrameKeysHandler();
   }

   /** @summary Add interactive functionality to the frame
    * @private */
   addInteractivity(for_second_axes) {

      if (isBatchMode() || (!settings.Zooming && !settings.ContextMenu))
         return true;
      FrameInteractive.assign(this);
      return this.addFrameInteractivity(for_second_axes);
   }

   /** @summary Set selected range back to pad object - to be implemented
     * @private */
   setRootPadRange(/* pad, is3d */) {
      // TODO: change of pad range and send back to root application
   }

   /** @summary Toggle log scale on the specified axes */
   toggleAxisLog(axis) {
      let handle = this[axis+"_handle"];
      if (handle) handle.changeAxisLog('toggle');
   }

} // class RFramePainter

export { RFramePainter };
