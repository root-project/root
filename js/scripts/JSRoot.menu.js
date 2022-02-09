/// @file JSRoot.menu.js
/// JSROOT menu implementation

JSROOT.define(['d3', 'painter'], (d3, jsrp) => {

   "use strict";

  /** @summary Produce exec string for WebCanas to set color value
    * @desc Color can be id or string, but should belong to list of known colors
    * For higher color numbers TColor::GetColor(r,g,b) will be invoked to ensure color is exists
    * @private */
   function getColorExec(col, method) {
      let id = -1, arr = jsrp.root_colors;
      if (typeof col == "string") {
         if (!col || (col == "none")) id = 0; else
            for (let k = 1; k < arr.length; ++k)
               if (arr[k] == col) { id = k; break; }
         if ((id < 0) && (col.indexOf("rgb") == 0)) id = 9999;
      } else if (Number.isInteger(col) && arr[col]) {
         id = col;
         col = arr[id];
      }

      if (id < 0) return "";

      if (id >= 50) {
         // for higher color numbers ensure that such color exists
         let c = d3.color(col);
         id = `TColor::GetColor(${c.r},${c.g},${c.b})`;
      }

      return `exec:${method}(${id})`;
   }

   /**
    * @summary Abstract class for creating context menu
    *
    * @memberof JSROOT.Painter
    * @desc Use {@link JSROOT.Painter.createMenu} to create instance of the menu
    * @private
    */

   class JSRootMenu {
      constructor(painter, menuname, show_event) {
         this.painter = painter;
         this.menuname = menuname;
         if (show_event && (typeof show_event == "object"))
            if ((show_event.clientX !== undefined) && (show_event.clientY !== undefined))
               this.show_evnt = { clientX: show_event.clientX, clientY: show_event.clientY };

         this.remove_handler = () => this.remove();
         this.element = null;
         this.cnt = 0;
      }

      load() { return Promise.resolve(this); }

      /** @summary Returns object with mouse event position when context menu was actiavted
       * @desc Return object will have members "clientX" and "clientY" */
      getEventPosition() { return this.show_evnt; }

      add(/*name, arg, func, title*/) {
         throw Error("add() method has to be implemented in the menu");
      }

      /** @summary Returns menu size */
      size() { return this.cnt; }

      /** @summary Close and remove menu */
      remove() {
         if (this.element!==null) {
            this.element.remove();
            if (this.resolveFunc) {
               this.resolveFunc();
               delete this.resolveFunc;
            }
            document.body.removeEventListener('click', this.remove_handler);
         }
         this.element = null;
      }

      show(/*event*/) {
         throw Error("show() method has to be implemented in the menu class");
      }

      /** @summary Add checked menu item
        * @param {boolean} flag - flag
        * @param {string} name - item name
        * @param {function} func - func called when item is selected */
      addchk(flag, name, arg, func) {
         let handler = func;
         if (typeof arg == 'function') {
            func = arg;
            handler = res => func(res=="1");
            arg = flag ? "0" : "1";
         }
         this.add((flag ? "chk:" : "unk:") + name, arg, handler);
      }

      /** @summary Add draw sub-menu with draw options
        * @protected */
      addDrawMenu(top_name, opts, call_back) {
         if (!opts) opts = [];
         if (opts.length==0) opts.push("");

         let without_sub = false;
         if (top_name.indexOf("nosub:")==0) {
            without_sub = true;
            top_name = top_name.substr(6);
         }

         if (opts.length === 1) {
            if (opts[0]==='inspect') top_name = top_name.replace("Draw", "Inspect");
            this.add(top_name, opts[0], call_back);
            return;
         }

         if (!without_sub) this.add("sub:" + top_name, opts[0], call_back);

         for (let i = 0; i < opts.length; ++i) {
            let name = opts[i];
            if (name=="") name = this._use_plain_text ? '<dflt>' : '&lt;dflt&gt;';

            let group = i+1;
            if ((opts.length > 5) && (name.length > 0)) {
               // check if there are similar options, which can be grouped once again
               while ((group<opts.length) && (opts[group].indexOf(name)==0)) group++;
            }

            if (without_sub) name = top_name + " " + name;

            if (group < i+2) {
               this.add(name, opts[i], call_back);
            } else {
               this.add("sub:" + name, opts[i], call_back);
               for (let k = i+1; k < group; ++k)
                  this.add(opts[k], opts[k], call_back);
               this.add("endsub:");
               i = group-1;
            }
         }
         if (!without_sub) this.add("endsub:");
      }

      /** @summary Add color selection menu entries
        * @protected */
      addColorMenu(name, value, set_func, fill_kind) {
         if (value === undefined) return;
         let useid = (typeof value !== 'string');
         this.add("sub:" + name, () => {
            this.input("Enter color " + (useid ? "(only id number)" : "(name or id)"), value, useid ? "int" : "text", useid ? 0 : undefined, useid ? 9999 : undefined).then(col => {
               let id = parseInt(col);
               if (Number.isInteger(id) && jsrp.getColor(id)) {
                  col = jsrp.getColor(id);
               } else {
                  if (useid) return;
               }
               set_func(useid ? id : col);
            });
         });
         for (let n = -1; n < 11; ++n) {
            if ((n < 0) && useid) continue;
            if ((n == 10) && (fill_kind !== 1)) continue;
            let col = (n < 0) ? 'none' : jsrp.getColor(n);
            if ((n == 0) && (fill_kind == 1)) col = 'none';
            let svg = "<svg width='100' height='18' style='margin:0px;background-color:" + col + "'><text x='4' y='12' style='font-size:12px' fill='" + (n == 1 ? "white" : "black") + "'>" + col + "</text></svg>";
            this.addchk((value == (useid ? n : col)), svg, (useid ? n : col), res => set_func(useid ? parseInt(res) : res));
         }
         this.add("endsub:");
      }

      /** @summary Add size selection menu entries
        * @protected */
      addSizeMenu(name, min, max, step, size_value, set_func) {
         if (size_value === undefined) return;

         this.add("sub:" + name, () => {
            let entry = size_value.toFixed(4);
            if (step >= 0.1) entry = size_value.toFixed(2);
            if (step >= 1) entry = size_value.toFixed(0);
            this.input("Enter value of " + name, entry, (step >= 1) ? "int" : "float").then(set_func);
         });
         for (let sz = min; sz <= max; sz += step) {
            let entry = sz.toFixed(2);
            if (step >= 0.1) entry = sz.toFixed(1);
            if (step >= 1) entry = sz.toFixed(0);
            this.addchk((Math.abs(size_value - sz) < step / 2), entry,
                        sz, res => set_func((step >= 1) ? parseInt(res) : parseFloat(res)));
         }
         this.add("endsub:");
      }

      /** @summary Add palette menu entries
        * @protected */
      addPaletteMenu(curr, set_func) {
         const add = (id, name, more) => this.addchk((id === curr) || more, '<nobr>' + name + '</nobr>', id, set_func);

         this.add("sub:Palette", () => this.input("Enter palette code [1..113]", curr, "int", 1, 113).then(set_func));

         add(50, "ROOT 5", (curr>=10) && (curr<51));
         add(51, "Deep Sea");
         add(52, "Grayscale", (curr>0) && (curr<10));
         add(53, "Dark body radiator");
         add(54, "Two-color hue");
         add(55, "Rainbow");
         add(56, "Inverted dark body radiator");
         add(57, "Bird", (curr>113));
         add(58, "Cubehelix");
         add(59, "Green Red Violet");
         add(60, "Blue Red Yellow");
         add(61, "Ocean");
         add(62, "Color Printable On Grey");
         add(63, "Alpine");
         add(64, "Aquamarine");
         add(65, "Army");
         add(66, "Atlantic");

         this.add("endsub:");
      }

      /** @summary Add rebin menu entries
        * @protected */
      addRebinMenu(rebin_func) {
        this.add("sub:Rebin", () => {
            this.input("Enter rebin value", 2, "int", 2).then(rebin_func);
         });
         for (let sz = 2; sz <= 7; sz++) {
            this.add(sz.toString(), sz, res => rebin_func(parseInt(res)));
         }
         this.add("endsub:");
      }

      /** @summary Add selection menu entries
        * @protected */
      addSelectMenu(name, values, value, set_func) {
         this.add("sub:" + name);
         for (let n = 0; n < values.length; ++n)
            this.addchk(values[n] == value, values[n], values[n], res => set_func(res));
         this.add("endsub:");
      }

      /** @summary Add RColor selection menu entries
        * @protected */
      addRColorMenu(name, value, set_func) {
         // if (value === undefined) return;
         let colors = ['default', 'black', 'white', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan'];

         this.add("sub:" + name, () => {
            this.input("Enter color name - empty string will reset color", value).then(set_func);
         });
         let fillcol = 'black';
         for (let n = 0; n < colors.length; ++n) {
            let coltxt = colors[n], match = false, bkgr = '';
            if (n > 0) {
               bkgr = "background-color:" + coltxt;
               fillcol = (coltxt == 'white') ? 'black' : 'white';

               if ((typeof value === 'string') && value && (value != 'auto') && (value[0] != '['))
                  match = (d3.rgb(value).toString() == d3.rgb(coltxt).toString());
            } else {
               match = !value;
            }
            let svg = `<svg width='100' height='18' style='margin:0px;${bkgr}'><text x='4' y='12' style='font-size:12px' fill='${fillcol}'>${coltxt}</text></svg>`;
            this.addchk(match, svg, coltxt, res => set_func(res == 'default' ? null : res));
         }
         this.add("endsub:");
      }

      /** @summary Add items to change RAttrText
        * @protected */
      addRAttrTextItems(fontHandler, opts, set_func) {
         if (!opts) opts = {};
         this.addRColorMenu("color", fontHandler.color, sel => set_func({ name: "color", value: sel }));
         if (fontHandler.scaled)
            this.addSizeMenu("size", 0.01, 0.10, 0.01, fontHandler.size /fontHandler.scale, sz => set_func({ name: "size", value: sz }));
         else
            this.addSizeMenu("size", 6, 20, 2, fontHandler.size, sz => set_func({ name: "size", value: sz }));

         this.addSelectMenu("family", ["Arial", "Times New Roman", "Courier New", "Symbol"], fontHandler.name, res => set_func( {name: "font_family", value: res }));

         this.addSelectMenu("style", ["normal", "italic", "oblique"], fontHandler.style || "normal", res => set_func( {name: "font_style", value: res == "normal" ? null : res }));

         this.addSelectMenu("weight", ["normal", "lighter", "bold", "bolder"], fontHandler.weight || "normal", res => set_func( {name: "font_weight", value: res == "normal" ? null : res }));

         if (!opts.noalign)
            this.add("align");
         if (!opts.noangle)
            this.add("angle");
      }

      /** @summary Fill context menu for text attributes
        * @private */
      addTextAttributesMenu(painter, prefix) {
         // for the moment, text attributes accessed directly from objects

         let obj = painter.getObject();
         if (!obj || !('fTextColor' in obj)) return;

         this.add("sub:" + (prefix ? prefix : "Text"));
         this.addColorMenu("color", obj.fTextColor,
            arg => { obj.fTextColor = arg; painter.interactiveRedraw(true, getColorExec(arg, "SetTextColor")); });

         let align = [11, 12, 13, 21, 22, 23, 31, 32, 33];

         this.add("sub:align");
         for (let n = 0; n < align.length; ++n) {
            this.addchk(align[n] == obj.fTextAlign,
               align[n], align[n],
               // align[n].toString() + "_h:" + hnames[Math.floor(align[n]/10) - 1] + "_v:" + vnames[align[n]%10-1], align[n],
               function(arg) { this.getObject().fTextAlign = parseInt(arg); this.interactiveRedraw(true, "exec:SetTextAlign(" + arg + ")"); }.bind(painter));
         }
         this.add("endsub:");

         this.add("sub:font");
         for (let n = 1; n < 16; ++n) {
            this.addchk(n == Math.floor(obj.fTextFont / 10), n, n,
               function(arg) { this.getObject().fTextFont = parseInt(arg) * 10 + 2; this.interactiveRedraw(true, "exec:SetTextFont(" + this.getObject().fTextFont + ")"); }.bind(painter));
         }
         this.add("endsub:");

         this.add("endsub:");
      }

      /** @summary Fill context menu for graphical attributes in painter
        * @private */
      addAttributesMenu(painter, preffix) {
         // this method used to fill entries for different attributes of the object
         // like TAttFill, TAttLine, ....
         // all menu call-backs need to be rebind, while menu can be used from other painter

         if (!preffix) preffix = "";

         if (painter.lineatt && painter.lineatt.used) {
            this.add("sub:" + preffix + "Line att");
            this.addSizeMenu("width", 1, 10, 1, painter.lineatt.width,
               arg => { painter.lineatt.change(undefined, arg); painter.interactiveRedraw(true, `exec:SetLineWidth(${arg})`); });
            this.addColorMenu("color", painter.lineatt.color,
               arg => { painter.lineatt.change(arg); painter.interactiveRedraw(true, getColorExec(arg, "SetLineColor")); });
            this.add("sub:style", () => {
               this.input("Enter line style id (1-solid)", painter.lineatt.style, "int", 1, 11).then(id => {
                  if (!jsrp.root_line_styles[id]) return;
                  painter.lineatt.change(undefined, undefined, id);
                  painter.interactiveRedraw(true, `exec:SetLineStyle(${id})`);
               });
            });
            for (let n = 1; n < 11; ++n) {
               let dash = jsrp.root_line_styles[n],
                   svg = "<svg width='100' height='18'><text x='1' y='12' style='font-size:12px'>" + n + "</text><line x1='30' y1='8' x2='100' y2='8' stroke='black' stroke-width='3' stroke-dasharray='" + dash + "'></line></svg>";

               this.addchk((painter.lineatt.style == n), svg, n, arg => { painter.lineatt.change(undefined, undefined, parseInt(arg)); painter.interactiveRedraw(true, `exec:SetLineStyle(${arg})`); });
            }
            this.add("endsub:");
            this.add("endsub:");

            if (('excl_side' in painter.lineatt) && (painter.lineatt.excl_side !== 0)) {
               this.add("sub:Exclusion");
               this.add("sub:side");
               for (let side = -1; side <= 1; ++side)
                  this.addchk((painter.lineatt.excl_side == side), side, side, function(arg) {
                     this.lineatt.changeExcl(parseInt(arg));
                     this.interactiveRedraw();
                  }.bind(painter));
               this.add("endsub:");

               this.addSizeMenu("width", 10, 100, 10, painter.lineatt.excl_width,
                  arg => { painter.lineatt.changeExcl(undefined, arg); painter.interactiveRedraw(); });

               this.add("endsub:");
            }
         }

         if (painter.fillatt && painter.fillatt.used) {
            this.add("sub:" + preffix + "Fill att");
            this.addColorMenu("color", painter.fillatt.colorindx,
               arg => { painter.fillatt.change(arg, undefined, painter.getCanvSvg()); painter.interactiveRedraw(true, getColorExec(arg, "SetFillColor")); }, painter.fillatt.kind);
            this.add("sub:style", () => {
               this.input("Enter fill style id (1001-solid, 3000..3010)", painter.fillatt.pattern, "int", 0, 4000).then(id => {
                  if ((id < 0) || (id > 4000)) return;
                  painter.fillatt.change(undefined, id, painter.getCanvSvg());
                  painter.interactiveRedraw(true, "exec:SetFillStyle(" + id + ")");
               });
            });

            let supported = [1, 1001, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3010, 3021, 3022];

            for (let n = 0; n < supported.length; ++n) {
               let sample = painter.createAttFill({ std: false, pattern: supported[n], color: painter.fillatt.colorindx || 1 }),
                   svg = "<svg width='100' height='18'><text x='1' y='12' style='font-size:12px'>" + supported[n].toString() + "</text><rect x='40' y='0' width='60' height='18' stroke='none' fill='" + sample.getFillColor() + "'></rect></svg>";
               this.addchk(painter.fillatt.pattern == supported[n], svg, supported[n], arg => {
                  painter.fillatt.change(undefined, parseInt(arg), painter.getCanvSvg());
                  painter.interactiveRedraw(true, `exec:SetFillStyle(${arg})`);
               });
            }
            this.add("endsub:");
            this.add("endsub:");
         }

         if (painter.markeratt && painter.markeratt.used) {
            this.add("sub:" + preffix + "Marker att");
            this.addColorMenu("color", painter.markeratt.color,
               arg => { painter.markeratt.change(arg); painter.interactiveRedraw(true, getColorExec(arg, "SetMarkerColor"));});
            this.addSizeMenu("size", 0.5, 6, 0.5, painter.markeratt.size,
               arg => { painter.markeratt.change(undefined, undefined, arg); painter.interactiveRedraw(true, `exec:SetMarkerSize(${arg})`); });

            this.add("sub:style");
            let supported = [1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34];

            for (let n = 0; n < supported.length; ++n) {

               let clone = new JSROOT.TAttMarkerHandler({ style: supported[n], color: painter.markeratt.color, size: 1.7 }),
                   svg = "<svg width='60' height='18'><text x='1' y='12' style='font-size:12px'>" + supported[n].toString() + "</text><path stroke='black' fill='" + (clone.fill ? "black" : "none") + "' d='" + clone.create(40, 8) + "'></path></svg>";

               this.addchk(painter.markeratt.style == supported[n], svg, supported[n],
                  function(arg) { this.markeratt.change(undefined, parseInt(arg)); this.interactiveRedraw(true, "exec:SetMarkerStyle(" + arg + ")"); }.bind(painter));
            }
            this.add("endsub:");
            this.add("endsub:");
         }
      }

      /** @summary Fill context menu for axis
        * @private */
      addTAxisMenu(painter, faxis, kind) {
         this.add("Divisions", () => this.input("Set Ndivisions", faxis.fNdivisions, "int", 0).then(val => {
            faxis.fNdivisions = val;
            painter.interactiveRedraw("pad", `exec:SetNdivisions(${val})`, kind);
         }));

         this.add("sub:Labels");
         this.addchk(faxis.TestBit(JSROOT.EAxisBits.kCenterLabels), "Center",
               arg => { faxis.InvertBit(JSROOT.EAxisBits.kCenterLabels); painter.interactiveRedraw("pad", `exec:CenterLabels(${arg})`, kind); });
         this.addchk(faxis.TestBit(JSROOT.EAxisBits.kLabelsVert), "Rotate",
               arg => { faxis.InvertBit(JSROOT.EAxisBits.kLabelsVert); painter.interactiveRedraw("pad", `exec:SetBit(TAxis::kLabelsVert,${arg})`, kind); });
         this.addColorMenu("Color", faxis.fLabelColor,
               arg => { faxis.fLabelColor = arg; painter.interactiveRedraw("pad", getColorExec(arg, "SetLabelColor"), kind); });
         this.addSizeMenu("Offset", 0, 0.1, 0.01, faxis.fLabelOffset,
               arg => { faxis.fLabelOffset = arg; painter.interactiveRedraw("pad", `exec:SetLabelOffset(${arg})`, kind); } );
         this.addSizeMenu("Size", 0.02, 0.11, 0.01, faxis.fLabelSize,
               arg => { faxis.fLabelSize = arg; painter.interactiveRedraw("pad", `exec:SetLabelSize(${arg})`, kind); } );
         this.add("endsub:");
         this.add("sub:Title");
         this.add("SetTitle", () => {
            this.input("Enter axis title", faxis.fTitle).then(t => {
               faxis.fTitle = t;
               painter.interactiveRedraw("pad", `exec:SetTitle("${t}")`, kind);
            });
         });
         this.addchk(faxis.TestBit(JSROOT.EAxisBits.kCenterTitle), "Center",
               arg => { faxis.InvertBit(JSROOT.EAxisBits.kCenterTitle); painter.interactiveRedraw("pad", `exec:CenterTitle(${arg})`, kind); });
         this.addchk(faxis.TestBit(JSROOT.EAxisBits.kOppositeTitle), "Opposite",
                () => { faxis.InvertBit(JSROOT.EAxisBits.kOppositeTitle); painter.redrawPad(); });
         this.addchk(faxis.TestBit(JSROOT.EAxisBits.kRotateTitle), "Rotate",
               arg => { faxis.InvertBit(JSROOT.EAxisBits.kRotateTitle); painter.interactiveRedraw("pad", `exec:RotateTitle(${arg})`, kind); });
         this.addColorMenu("Color", faxis.fTitleColor,
               arg => { faxis.fTitleColor = arg; painter.interactiveRedraw("pad", getColorExec(arg, "SetTitleColor"), kind); });
         this.addSizeMenu("Offset", 0, 3, 0.2, faxis.fTitleOffset,
                         arg => { faxis.fTitleOffset = arg; painter.interactiveRedraw("pad", `exec:SetTitleOffset(${arg})`, kind); });
         this.addSizeMenu("Size", 0.02, 0.11, 0.01, faxis.fTitleSize,
                         arg => { faxis.fTitleSize = arg; painter.interactiveRedraw("pad", `exec:SetTitleSize(${arg})`, kind); });
         this.add("endsub:");
         this.add("sub:Ticks");
         if (faxis._typename == "TGaxis") {
            this.addColorMenu("Color", faxis.fLineColor,
                     arg => { faxis.fLineColor = arg; painter.interactiveRedraw("pad"); });
            this.addSizeMenu("Size", -0.05, 0.055, 0.01, faxis.fTickSize,
                     arg => { faxis.fTickSize = arg; painter.interactiveRedraw("pad"); } );
         } else {
            this.addColorMenu("Color", faxis.fAxisColor,
                     arg => { faxis.fAxisColor = arg; painter.interactiveRedraw("pad", getColorExec(arg, "SetAxisColor"), kind); });
            this.addSizeMenu("Size", -0.05, 0.055, 0.01, faxis.fTickLength,
                     arg => { faxis.fTickLength = arg; painter.interactiveRedraw("pad", `exec:SetTickLength(${arg})`, kind); });
         }
         this.add("endsub:");
      }

      /** @summary Run modal dialog
        * @returns {Promise} with html element inside dialg
        * @private */
      runModal() {
         throw Error('runModal() must be reimplemented');
      }

      /** @summary Show modal info dialog
        * @param {String} title - title
        * @param {String} message - message
        * @protected */
      info(title, message) {
         return this.runModal(title,`<p>${message}</p>`, { height: 120, width: 400, resizable: true });
      }

      /** @summary Show confirm dialog
        * @param {String} title - title
        * @param {String} message - message
        * @returns {Promise} with true when "Ok" pressed or false when "Cancel" pressed
        * @protected */
      confirm(title, message) {
         return this.runModal(title, message, { btns: true, height: 120, width: 400 }).then(elem => { return !!elem; });
      }

      /** @summary Input value
        * @returns {Promise} with input value
        * @param {string} title - input dialog title
        * @param value - initial value
        * @param {string} [kind] - use "text" (default), "number", "float" or "int"
        * @protected */
      input(title, value, kind, min, max) {

         if (!kind) kind = "text";
         let inp_type = (kind == "int") ? "number" : "text", ranges = "";
         if ((value === undefined) || (value === null)) value = "";
         if (kind == "int") {
             if (min !== undefined) ranges += ` min="${min}"`;
             if (max !== undefined) ranges += ` max="${max}"`;
          }

         let main_content =
            '<form><fieldset style="padding:0; border:0">'+
               `<input type="${inp_type}" value="${value}" ${ranges} style="width:98%;display:block" class="jsroot_dlginp"/>`+
            '</fieldset></form>';

         return new Promise(resolveFunc => {

            this.runModal(title, main_content, { btns: true, height: 150, width: 400 }).then(element => {
               if (!element) return;
               let val = element.querySelector(`.jsroot_dlginp`).value;
               if (kind == "float") {
                  val = parseFloat(val);
                  if (Number.isFinite(val))
                     resolveFunc(val);
               } else if (kind == "int") {
                  val = parseInt(val);
                  if (Number.isInteger(val))
                     resolveFunc(val);
               } else {
                  resolveFunc(val);
              }
            });

         });
      }

      /** @summary Let input arguments from the method
        * @returns {Promise} with method argument */
      showMethodArgsDialog(method) {
         let dlg_id = this.menuname + "_dialog",
             main_content = '<form> <fieldset style="padding:0; border:0">';

         for (let n = 0; n < method.fArgs.length; ++n) {
            let arg = method.fArgs[n];
            arg.fValue = arg.fDefault;
            if (arg.fValue == '\"\"') arg.fValue = "";
            main_content += `<label for="${dlg_id}_inp${n}">${arg.fName}</label>
                             <input type="text" tabindex="${n+1}" id="${dlg_id}_inp${n}" value="${arg.fValue}" style="width:100%;display:block"/>`;
         }

         main_content += '</fieldset></form>';

         return new Promise(resolveFunc => {

            this.runModal(method.fClassName + '::' + method.fName, main_content, { btns: true, height: 100 + method.fArgs.length*60, width: 400, resizable: true }).then(element => {
               if (!element) return;
               let args = "";

               for (let k = 0; k < method.fArgs.length; ++k) {
                  let arg = method.fArgs[k];
                  let value = element.querySelector(`#${dlg_id}_inp${k}`).value;
                  if (value === "") value = arg.fDefault;
                  if ((arg.fTitle=="Option_t*") || (arg.fTitle=="const char*")) {
                     // check quotes,
                     // TODO: need to make more precise checking of escape characters
                     if (!value) value = '""';
                     if (value[0]!='"') value = '"' + value;
                     if (value[value.length-1] != '"') value += '"';
                  }

                  args += (k > 0 ? "," : "") + value;
               }

               resolveFunc(args);
            });
         });

      }

      /** @summary Let input arguments from the Command
        * @returns {Promise} with command argument */
      showCommandArgsDialog(cmdname, args) {
         let dlg_id = this.menuname + "_dialog",
             main_content = '<form> <fieldset style="padding:0; border:0">';

         for (let n = 0; n < args.length; ++n)
            main_content += `<label for="${dlg_id}_inp${n}">arg${n+1}</label>
                             <input type="text" id="${dlg_id}_inp${n}" value="${args[n]}" style="width:100%;display:block"/>`;

         main_content += '</fieldset></form>';

         return new Promise(resolveFunc => {

            this.runModal("Arguments for command " + cmdname, main_content, { btns: true, height: 110 + args.length*60, width: 400, resizable: true}).then(element => {
               if (!element)
                  return resolveFunc(null);

               let resargs = [];
               for (let k = 0; k < args.length; ++k)
                  resargs.push(element.querySelector(`#${dlg_id}_inp${k}`).value);
               resolveFunc(resargs);
            });
         });
      }

   } // class JSRootMenu

   /**
    * @summary Context menu class using plain HTML/JavaScript
    *
    * @memberof JSROOT.Painter
    * @desc Use {@link JSROOT.Painter.createMenu} to create instance of the menu
    * based on {@link https://github.com/L1quidH2O/ContextMenu.js}
    * @private
    */

   class StandaloneMenu extends JSRootMenu {

      constructor(painter, menuname, show_event) {
         super(painter, menuname, show_event);

         this.code = [];
         this._use_plain_text = true;
         this.stack = [ this.code ];
      }

     /** @summary Load required modules, noop for that menu class */
     load() { return Promise.resolve(this); }

      /** @summary Add menu item
        * @param {string} name - item name
        * @param {function} func - func called when item is selected */
      add(name, arg, func, title) {
         let curr = this.stack[this.stack.length-1];

         if (name == "separator")
            return curr.push({ divider: true });

         if (name.indexOf("header:")==0)
            return curr.push({ text: name.substr(7), header: true });

         if (name=="endsub:") return this.stack.pop();

         if (typeof arg == 'function') { title = func; func = arg; arg = name; }

         let elem = {};
         curr.push(elem);

         if (name.indexOf("sub:")==0) {
            name = name.substr(4);
            elem.sub = [];
            this.stack.push(elem.sub);
         }

         if (name.indexOf("chk:")==0) { elem.checked = true; name = name.substr(4); } else
         if (name.indexOf("unk:")==0) { elem.checked = false; name = name.substr(4); }

         elem.text = name;
         elem.title = title;
         elem.arg = arg;
         elem.func = func;
      }

      /** @summary Returns size of main menu */
      size() { return this.code.length; }

      /** @summary Build HTML elements of the menu
        * @private */
      _buildContextmenu(menu, left, top, loc) {

         let outer = document.createElement('div');
         outer.className = "jsroot_ctxt_container";

         //if loc !== document.body then its a submenu, so it needs to have position: relative;
         if (loc === document.body) {
            //delete all elements with className jsroot_ctxt_container
            let deleteElems = document.getElementsByClassName('jsroot_ctxt_container');
            while (deleteElems.length > 0)
               deleteElems[0].parentNode.removeChild(deleteElems[0]);

            outer.style.position = 'fixed';
            outer.style.left = left + 'px';
            outer.style.top = top + 'px';
         } else {
            outer.style.left = -loc.offsetLeft + loc.offsetWidth + 'px';
         }

         let need_check_area = false;
         menu.forEach(d => { if (d.checked !== undefined) need_check_area = true; });

         menu.forEach(d => {
            if (d.divider) {
               let hr = document.createElement('hr');
               hr.className = "jsroot_ctxt_divider";
               outer.appendChild(hr);
               return;
            }

            let item = document.createElement('div');
            item.style.position = 'relative';
            outer.appendChild(item);

            if (d.header) {
               item.className = "jsroot_ctxt_header";
               item.innerHTML = d.text;
               return;
            }

            let hovArea = document.createElement('div');
            hovArea.style.width = '100%';
            hovArea.style.height = '100%';
            hovArea.className = "jsroot_ctxt_item";
            hovArea.style.display = 'flex';
            hovArea.style.justifyContent = 'space-between';
            hovArea.style.cursor = 'pointer';
            if (d.title) hovArea.setAttribute("title", d.title);

            item.appendChild(hovArea);
            if (!d.text) d.text = "item";

            let text = document.createElement('div');
            text.className = "jsroot_ctxt_text";
            if (d.text.indexOf("<svg") >= 0) {
               text.innerHTML = d.text;
            } else {
               if (need_check_area) {
                  let chk = document.createElement('span');
                  chk.innerHTML = d.checked ? "\u2713" : "";
                  chk.style.display = "inline-block";
                  chk.style.width = "1em";
                  text.appendChild(chk);
               }

               let sub = document.createElement('span');
               if (d.text.indexOf("<nobr>") == 0)
                  sub.textContent = d.text.substr(6, d.text.length-13);
               else
                  sub.textContent = d.text;
               text.appendChild(sub);
            }
            hovArea.appendChild(text);

            if (d.hasOwnProperty('extraText') || d.sub) {
               let extraText = document.createElement('span');
               extraText.className = "jsroot_ctxt_extraText jsroot_ctxt_text";
               extraText.textContent = d.sub ? "\u25B6" : d.extraText;
               hovArea.appendChild(extraText);
            }

            hovArea.addEventListener('mouseenter', () => {
               let focused = outer.childNodes;
               focused.forEach(d => {
                  if (d.classList.contains('jsroot_ctxt_focus')) {
                     d.removeChild(d.getElementsByClassName('jsroot_ctxt_container')[0]);
                     d.classList.remove('jsroot_ctxt_focus');
                  }
               })
            });

            if (d.sub)
               hovArea.addEventListener('mouseenter', () => {
                  item.classList.add('jsroot_ctxt_focus');
                  this._buildContextmenu(d.sub, 0, 0, item);
               });


            if (d.func)
               item.addEventListener('click', evnt => {
                  let func = this.painter ? d.func.bind(this.painter) : d.func;
                  func(d.arg);
                  evnt.stopPropagation();
                  this.remove();
               });
         });

         loc.appendChild(outer);

         let docWidth = document.documentElement.clientWidth, docHeight = document.documentElement.clientHeight;

         //Now determine where the contextmenu will be
         if (loc === document.body) {

            if (left + outer.offsetWidth > docWidth) {
               //Does sub-contextmenu overflow window width?
               outer.style.left = docWidth - outer.offsetWidth + 'px';
            }

            if (outer.offsetHeight > docHeight) {
               //is the contextmenu height larger than the window height?
               outer.style.top = 0;
               outer.style.overflowY = 'scroll';
               outer.style.overflowX = 'hidden';
               outer.style.height = docHeight + 'px';
            }
            else if (top + outer.offsetHeight > docHeight) {
               //Does contextmenu overflow window height?
               outer.style.top = docHeight - outer.offsetHeight + 'px';
            }

         } else {

            //if its sub-contextmenu

            let dimensionsLoc = loc.getBoundingClientRect(), dimensionsOuter = outer.getBoundingClientRect();

            //Does sub-contextmenu overflow window width?
            if (dimensionsOuter.left + dimensionsOuter.width > docWidth) {
               outer.style.left = -loc.offsetLeft - dimensionsOuter.width + 'px'
            }

            if (dimensionsOuter.height > docHeight) {
               //is the sub-contextmenu height larger than the window height?

               outer.style.top = -dimensionsOuter.top + 'px'
               outer.style.overflowY = 'scroll'
               outer.style.overflowX = 'hidden'
               outer.style.height = docHeight + 'px'
            }
            else if (dimensionsOuter.height < docHeight && dimensionsOuter.height > docHeight / 2) {
               //is the sub-contextmenu height smaller than the window height AND larger than half of window height?

               if (dimensionsOuter.top - docHeight / 2 >= 0) { //If sub-contextmenu is closer to bottom of the screen
                  outer.style.top = -dimensionsOuter.top - dimensionsOuter.height + docHeight + 'px'
               }
               else { //If sub-contextmenu is closer to top of the screen
                  outer.style.top = -dimensionsOuter.top + 'px'
               }

            }
            else if (dimensionsOuter.top + dimensionsOuter.height > docHeight) {
               //Does sub-contextmenu overflow window height?
               outer.style.top = -dimensionsOuter.height + dimensionsLoc.height + 'px'
            }

         }
         return outer;
      }

      /** @summary Show standalone menu */
      show(event) {
         this.remove();

         if (!event && this.show_evnt) event = this.show_evnt;

         document.body.addEventListener('click', this.remove_handler);

         let oldmenu = document.getElementById(this.menuname);
         if (oldmenu) oldmenu.remove();

         this.element = this._buildContextmenu(this.code, event.clientX + window.pageXOffset, event.clientY + window.pageYOffset, document.body);

         this.element.setAttribute('id', this.menuname);

         return Promise.resolve(this);
      }

      /** @summary Run modal elements with standalone code */
      runModal(title, main_content, args) {
         if (!args) args = {};
         let dlg_id = this.menuname + "_dialog";
         d3.select("#" + dlg_id).remove();
         d3.select("#" + dlg_id+"_block").remove();

         let block = d3.select('body').append('div').attr('id', dlg_id+"_block").attr("class", "jsroot_dialog_block");

         let element = d3.select('body')
                         .append('div')
                         .attr('id',dlg_id)
                         .attr("class","jsroot_dialog").style("width",(args.width || 450) + "px")
                         .attr("tabindex", "0")
                         .html(
            `<div class="jsroot_dialog_body">
               <div class="jsroot_dialog_header">${title}</div>
               <div class="jsroot_dialog_content">${main_content}</div>
               <div class="jsroot_dialog_footer">
                  <button class="jsroot_dialog_button">Ok</button>
                  ${args.btns ? '<button class="jsroot_dialog_button">Cancel</button>' : ''}
              </div>
             </div>`);

         return new Promise(resolveFunc => {
            element.on("keyup", evnt => {
               if ((evnt.keyCode == 13) || (evnt.keyCode == 27)) {
                  evnt.preventDefault();
                  evnt.stopPropagation();
                  resolveFunc(evnt.keyCode == 13 ? element.node() : null);
                  element.remove();
                  block.remove();
               }
            });
            element.on("keydown", evnt => {
               if ((evnt.keyCode == 13) || (evnt.keyCode == 27)) {
                  evnt.preventDefault();
                  evnt.stopPropagation();
               }
            });
            element.selectAll('.jsroot_dialog_button').on("click", evnt => {
               resolveFunc(args.btns && (d3.select(evnt.target).text() == "Ok") ? element.node() : null);
               element.remove();
               block.remove();
            });

            let f = element.select('.jsroot_dialog_content').select('input');
            if (f.empty()) f = element.select('.jsroot_dialog_footer').select('button');
            if (!f.empty()) f.node().focus();
         });
      }

   } // class StandaloneMenu

   /**
    * @summary Context menu class using Bootstrap
    *
    * @memberof JSROOT.Painter
    * @desc Use {@link JSROOT.Painter.createMenu} to create instance of the menu
    * @private
    */

   class BootstrapMenu extends JSRootMenu {

      constructor(painter, menuname, show_event) {
         super(painter, menuname, show_event);

         this.code = "";
         this.funcs = {};
         this.lvl = 0;
      }

      /** @summary Load bootstrap functionality, required for menu
        * @private */
      loadBS(with_js) {
         let ext = 'https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.0.2/';

         let promise = JSROOT._.bs_path ? Promise.resolve(true) :
                         JSROOT.loadScript(JSROOT.source_dir + 'style/bootstrap.min.css')
                               .then(() => { JSROOT._.bs_path = JSROOT.source_dir + 'scripts/'; })
                               .catch(() => { JSROOT._.bs_path = ext + "js/"; return JSROOT.loadScript(ext + 'css/bootstrap.min.css'); });
         return promise.then(() => (!with_js || (typeof bootstrap != 'undefined')) ? true : JSROOT.loadScript(JSROOT._.bs_path + 'bootstrap.bundle.min.js'));
      }

      /** @summary Load bootstrap functionality */
      load() { return this.loadBS().then(() => this); }

      /** @summary Add menu item
        * @param {string} name - item name
        * @param {function} func - func called when item is selected */
      add(name, arg, func, title) {
         if (name == "separator") {
            this.code += '<hr class="dropdown-divider">';
            return;
         }

         if (name.indexOf("header:")==0) {
            this.code += `<h6 class="dropdown-header">${name.substr(7)}</h6>`;
            return;
         }

         let newlevel = false, extras = "", cl = "dropdown-item btn-sm", checked = "";

         if (name=="endsub:") {
            this.lvl--;
            this.code += "</li>";
            this.code += "</ul>";
            return;
         }
         if (name.indexOf("sub:")==0) { name = name.substr(4); newlevel = true; }

         if (typeof arg == 'function') { func = arg; arg = name; }

         if (name.indexOf("chk:")==0) {
            checked = '\u2713';
            name  = name.substr(4);
         } else if (name.indexOf("unk:")==0) {
            name = name.substr(4);
         }

         if (title) extras += ` title="${title}"`;
         if (arg !== undefined) extras += ` arg="${arg}"`;
         if (newlevel) { extras += ` data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false"`; cl += " dropdown-toggle"; }

         let item = `<button id="${this.menuname}${this.cnt}" ${extras} class="${cl}" type="button"><span style="width:1em;display:inline-block">${checked}</span>${name}</button>`;

         if (newlevel) item = '<li class="dropend">' + item;
                  else item = "<li>" + item + "</li>";

         this.code += item;

         if (newlevel) {
            this.code += `<ul class="dropdown-menu" aria-labelledby="${this.menuname}${this.cnt}">`;
            this.lvl++;
         }

         if (typeof func == 'function') this.funcs[this.cnt] = func; // keep call-back function

         this.cnt++;
      }

      /** @summary Show menu */
      show(event) {
         this.remove();

         if (!event && this.show_evnt) event = this.show_evnt;

         document.body.addEventListener('click', this.remove_handler);

         let oldmenu = document.getElementById(this.menuname);
         if (oldmenu) oldmenu.parentNode.removeChild(oldmenu);

         return this.loadBS().then(() => {

            let ww = window.innerWidth, wh = window.innerHeight;

            this.element = document.createElement('div');
            this.element.id = this.menuname;
            this.element.setAttribute('class', "dropdown");
            this.element.innerHTML = `<ul class="dropdown-menu dropend" style="display:block">${this.code}</ul>`;

            document.body.appendChild(this.element);

            this.element.style.position = 'absolute';
            this.element.style.background = 'white';
            this.element.style.display = 'block';
            this.element.style.left = (event.clientX + window.pageXOffset) + 'px';
            this.element.style.top = (event.clientY + window.pageYOffset) + 'px';

            let menu = this;

            let myItems = this.element.getElementsByClassName('dropdown-item');

            for (let i = 0; i < myItems.length; i++)
               myItems[i].addEventListener('click', function() {
                  let arg = this.getAttribute('arg'),
                      cnt = this.getAttribute('id').substr(menu.menuname.length),
                      func = cnt ? menu.funcs[cnt] : null;
                  menu.remove();
                  if (typeof func == 'function') {
                     if (menu.painter)
                        func.bind(menu.painter)(arg); // if 'painter' field set, returned as this to callback
                     else
                        func(arg);
                  }
               });

            let myDropdown = this.element.getElementsByClassName('dropdown-toggle');
            for (let i=0; i < myDropdown.length; i++) {
               myDropdown[i].addEventListener('mouseenter', function() {
                  let el = this.nextElementSibling;
                  el.style.display = (el.style.display == 'block') ? 'none' : 'block';
                  el.style.left = this.scrollWidth + 'px';
                  let rect = el.getBoundingClientRect();
                  if (rect.bottom > wh) el.style.top = (wh - rect.bottom - 5) + 'px';
                  if (rect.right > ww) el.style.left = (-rect.width) + 'px';
               });
               myDropdown[i].addEventListener('mouseleave', function() {
                  let el = this.nextElementSibling;
                  el.was_entered = false;
                  setTimeout(function() { if (!el.was_entered) el.style.display = 'none'; }, 200);
               });
            }

            let myMenus = this.element.getElementsByClassName('dropdown-menu');
            for (let i = 0; i < myMenus.length; i++)
               myMenus[i].addEventListener('mouseenter', function() {
                  this.was_entered = true;
               });


            let newx = null, newy = null, rect = this.element.firstChild.getBoundingClientRect();

            if (event.clientX + rect.width > ww) newx = ww - rect.width - 10;
            if (event.clientY + rect.height > wh) newy = wh - rect.height - 10;

            if (newx!==null) this.element.style.left = ((newx>0 ? newx : 0) + window.pageXOffset) + 'px';
            if (newy!==null) this.element.style.top = ((newy>0 ? newy : 0) + window.pageYOffset) + 'px';

            return new Promise(resolve => {
               this.resolveFunc = resolve;
            });
         });
      }

      /** @summary Run modal elements with bootstrap code */
      runModal(title, main_content, args) {
         if (!args) args = {};

         let dlg_id = this.menuname + "_dialog",
             old_dlg = document.getElementById(dlg_id);
         if (old_dlg) old_dlg.remove();

         return this.loadBS(true).then(() => {

            let myModalEl = document.createElement('div');
            myModalEl.setAttribute('id', dlg_id);
            myModalEl.setAttribute('class', 'modal fade');
            myModalEl.setAttribute('role', "dialog");
            myModalEl.setAttribute('tabindex', "-1");
            myModalEl.setAttribute('aria-hidden', "true");
            let close_btn = args.btns ? '<button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>' : '';

            myModalEl.innerHTML =
               `<div class="modal-dialog">
                 <div class="modal-content">
                  <div class="modal-header">
                   <h5 class="modal-title">${title}</h5>
                   <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                  </div>
                  <div class="modal-body">
                     ${main_content}
                  </div>
                  <div class="modal-footer">
                     ${close_btn}
                     <button type="button" class="btn btn-primary jsroot_okbtn" data-bs-dismiss="modal">Ok</button>
                  </div>
                 </div>
                </div>`;

            document.body.appendChild(myModalEl);

            let myModal = new bootstrap.Modal(myModalEl, { keyboard: true, backdrop: 'static' });
            myModal.show();

            return new Promise(resolveFunc => {
               let pressOk = false;
               myModalEl.querySelector(`.jsroot_okbtn`).addEventListener('click', () => { pressOk = true; });

               myModalEl.addEventListener('hidden.bs.modal', () => {
                  if (pressOk) resolveFunc(myModalEl);
                  myModalEl.remove();
               });
            });

        });
      }

   }


   /** @summary Create JSROOT menu
     * @desc See {@link JSROOT.Painter.jQueryMenu} class for detailed list of methods
     * @memberof JSROOT.Painter
     * @param {object} [evnt] - event object like mouse context menu event
     * @param {object} [handler] - object with handling function, in this case one not need to bind function
     * @param {string} [menuname] - optional menu name
     * @example
     * JSROOT.require("painter")
     *       .then(jsrp => jsrp.createMenu())
     *       .then(menu => {
     *          menu.add("First", () => console.log("Click first"));
     *          let flag = true;
     *          menu.addchk(flag, "Checked", arg => console.log(`Now flag is ${arg}`));
     *          menu.show();
     *        }); */
   function createMenu(evnt, handler, menuname) {
      let menu = JSROOT.settings.Bootstrap ?  new BootstrapMenu(handler, menuname || 'root_ctx_menu', evnt)
                                           : new StandaloneMenu(handler, menuname || 'root_ctx_menu', evnt);
      return menu.load();
   }

   /** @summary Close previousely created and shown JSROOT menu
     * @param {string} [menuname] - optional menu name
     * @memberof JSROOT.Painter */
   function closeMenu(menuname) {
      let x = document.getElementById(menuname || 'root_ctx_menu');
      if (x) { x.parentNode.removeChild(x); return true; }
      return false;
   }

   jsrp.createMenu = createMenu;
   jsrp.closeMenu = closeMenu;

   if (JSROOT.nodejs) module.exports = jsrp;
   return jsrp;
});
