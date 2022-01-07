/// @file JSRoot.menu.js
/// JSROOT menu implementation

JSROOT.define(['d3', 'jquery', 'painter', 'jquery-ui'], (d3, $, jsrp) => {

   "use strict";

   if (typeof jQuery === 'undefined') globalThis.jQuery = $;

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
         id = "TColor::GetColor(" + c.r + "," + c.g + "," + c.b + ")";
      }

      return "exec:" + method + "(" + id + ")";
   }

   /**
    * @summary Class for creating context menu
    *
    * @class
    * @memberof JSROOT.Painter
    * @desc Use {@link JSROOT.Painter.createMenu} to create instance of the menu
    * @private
    */

   class JQueryMenu {
      constructor(painter, menuname, show_event) {
         this.painter = painter;
         this.menuname = menuname;
         this.element = null;
         this.code = "";
         this.cnt = 1;
         this.funcs = {};
         this.separ = false;
         if (show_event && (typeof show_event == "object"))
            if ((show_event.clientX !== undefined) && (show_event.clientY !== undefined))
               this.show_evnt = { clientX: show_event.clientX, clientY: show_event.clientY };

         this.remove_bind = this.remove.bind(this);
      }

      /** @summary Returns object with mouse event position when context menu was actiavted
        * @desc Return object will have members "clientX" and "clientY" */
      getEventPosition() { return this.show_evnt; }

      /** @summary Add menu item
        * @param {string} name - item name
        * @param {function} func - func called when item is selected */
      add(name, arg, func, title) {
         if (name == "separator") { this.code += "<li>-</li>"; this.separ = true; return; }

         if (name.indexOf("header:")==0) {
            this.code += "<li class='ui-widget-header' style='padding:3px; padding-left:5px;'>"+name.substr(7)+"</li>";
            return;
         }

         if (name=="endsub:") { this.code += "</ul></li>"; return; }

         let item = "", close_tag = "</li>";
         title = title ? " title='" + title + "'" : "";
         if (name.indexOf("sub:")==0) { name = name.substr(4); close_tag = "<ul>"; }

         if (typeof arg == 'function') { func = arg; arg = name; }

         if (name.indexOf("chk:")==0) { item = "<span class='ui-icon ui-icon-check' style='margin:1px'></span>"; name = name.substr(4); } else
         if (name.indexOf("unk:")==0) { item = "<span class='ui-icon ui-icon-blank' style='margin:1px'></span>"; name = name.substr(4); }

         // special handling of first versions with menu support
         if (($.ui.version.indexOf("1.10")==0) || ($.ui.version.indexOf("1.9")==0))
            item = '<a href="#">' + item + name + '</a>';
         else if ($.ui.version.indexOf("1.11")==0)
            item += name;
         else
            item = "<div" + title + ">" + item + name + "</div>";

         this.code += "<li cnt='" + this.cnt + ((arg !== undefined) ? "' arg='" + arg : "") + "'>" + item + close_tag;
         if (typeof func == 'function') this.funcs[this.cnt] = func; // keep call-back function

         this.cnt++;
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

      /** @summary Returns menu size */
      size() { return this.cnt-1; }

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

         for (let i=0;i<opts.length;++i) {
            let name = opts[i];
            if (name=="") name = '&lt;dflt&gt;';

            let group = i+1;
            if ((opts.length>5) && (name.length>0)) {
               // check if there are similar options, which can be grouped once again
               while ((group<opts.length) && (opts[group].indexOf(name)==0)) group++;
            }

            if (without_sub) name = top_name + " " + name;

            if (group < i+2) {
               this.add(name, opts[i], call_back);
            } else {
               this.add("sub:" + name, opts[i], call_back);
               for (let k=i+1;k<group;++k)
                  this.add(opts[k], opts[k], call_back);
               this.add("endsub:");
               i = group-1;
            }
         }
         if (!without_sub) this.add("endsub:");
      }

      /** @summary Show modal info dialog
        * @protected */
      info(title, message) {
         let dlg_id = this.menuname + "_dialog";
         let old_dlg = document.getElementById(dlg_id);
         if (old_dlg) old_dlg.parentNode.removeChild(old_dlg);
         $(document.body).append(
            `<div id="${dlg_id}">
            <p tabindex="0">${message}</p>
            </div>`);
         let dialog = $("#" + dlg_id).dialog({
            height: 120,
            width: 400,
            modal: true,
            resizable: true,
            title: title,
            close: () => dialog.remove()
          });
      }

      /** @summary Input value
        * @returns {Promise} with input value
        * @protected */
      input(title, value, kind) {
         let dlg_id = this.menuname + "_dialog";
         let old_dlg = document.getElementById(dlg_id);
         if (old_dlg) old_dlg.parentNode.removeChild(old_dlg);
         if (!kind) kind = "text";
         let inp_type = (kind == "int") ? "number" : "text";
         if ((value === undefined) || (value === null)) value = "";

         $(document.body).append(
            `<div id="${dlg_id}">
              <form>
                <fieldset style="padding:0; border:0">
                   <input type="${inp_type}" tabindex="0" name="${dlg_id}_inp" id="${dlg_id}_inp" value="${value}" style="width:100%;display:block" class="text ui-widget-content ui-corner-all"/>
                   <input type="submit" tabindex="-1" style="position:absolute; top:-1000px; display:block"/>
               </fieldset>
              </form>
            </div>`);

         return new Promise(resolveFunc => {
            let dialog, pressEnter = () => {
               let val = $("#" + dlg_id + "_inp").val();
               dialog.dialog("close");
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
            }

            dialog = $("#" + dlg_id).dialog({
               height: 150,
               width: 400,
               modal: true,
               resizable: false,
               title: title,
               buttons: {
                  "Ok": pressEnter,
                  "Cancel": () => dialog.dialog( "close" )
               },
               close: () => dialog.remove()
             });

             dialog.find( "form" ).on( "submit", event => {
                event.preventDefault();
                pressEnter();
             });
          });
      }

      /** @summary Let input arguments from the command
        * @returns {Promise} with command argument */
      showMethodArgsDialog(method) {
         let dlg_id = this.menuname + "_dialog";
         let old_dlg = document.getElementById(dlg_id);
         if (old_dlg) old_dlg.parentNode.removeChild(old_dlg);

         let inputs = "";

         for (let n = 0; n < method.fArgs.length; ++n) {
            let arg = method.fArgs[n];
            arg.fValue = arg.fDefault;
            if (arg.fValue == '\"\"') arg.fValue = "";
            inputs += `<label for="${dlg_id}_inp${n}">${arg.fName}</label>
                       <input type="text" tabindex="0" name="${dlg_id}_inp${n}" id="${dlg_id}_inp${n}" value="${arg.fValue}" style="width:100%;display:block" class="text ui-widget-content ui-corner-all"/>`;
         }

         $(document.body).append(
            `<div id="${dlg_id}">
              <form>
                <fieldset style="padding:0; border:0">
                   ${inputs}
                   <input type="submit" tabindex="-1" style="position:absolute; top:-1000px; display:block"/>
               </fieldset>
              </form>
            </div>`);

         return new Promise(resolveFunc => {
            let dialog, pressEnter = () => {
               let args = "";

               for (let k = 0; k < method.fArgs.length; ++k) {
                  let arg = method.fArgs[k];
                  let value = $("#" + dlg_id + "_inp" + k).val();
                  if (value==="") value = arg.fDefault;
                  if ((arg.fTitle=="Option_t*") || (arg.fTitle=="const char*")) {
                     // check quotes,
                     // TODO: need to make more precise checking of escape characters
                     if (!value) value = '""';
                     if (value[0]!='"') value = '"' + value;
                     if (value[value.length-1] != '"') value += '"';
                  }

                  args += (k>0 ? "," : "") + value;
               }

               dialog.dialog("close");
               resolveFunc(args);
            }

            dialog = $("#" + dlg_id).dialog({
               height: 100 + method.fArgs.length*60,
               width: 400,
               modal: true,
               resizable: true,
               title: method.fClassName + '::' + method.fName,
               buttons: {
                  "Ok": pressEnter,
                  "Cancel": () => dialog.dialog( "close" )
               },
               close: () => dialog.remove()
             });

             dialog.find( "form" ).on( "submit", event => {
                event.preventDefault();
                pressEnter();
             });
          });
      }

      /** @summary Add color selection menu entries
        * @protected */
      addColorMenu(name, value, set_func, fill_kind) {
         if (value === undefined) return;
         let useid = (typeof value !== 'string');
         this.add("sub:" + name, () => {
            this.input("Enter color " + (useid ? "(only id number)" : "(name or id)"), value, useid ? "int" : "text").then(col => {
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

      addRebinMenu(rebin_func) {
        this.add("sub:Rebin", () => {
            this.input("Enter rebin value", 2, "int").then(rebin_func);
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
               this.input("Enter line style id (1-solid)", painter.lineatt.style, "int").then(id => {
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
               this.input("Enter fill style id (1001-solid, 3000..3010)", painter.fillatt.pattern, "int").then(id => {
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
         this.add("Divisions", () => this.input("Set Ndivisions", faxis.fNdivisions, "int").then(val => {
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

      /** @summary Close and remove menu */
      remove() {
         if (this.element!==null) {
            this.element.remove();
            if (this.resolveFunc) {
               this.resolveFunc();
               delete this.resolveFunc;
            }
            document.body.removeEventListener('click', this.remove_bind);
         }
         this.element = null;
      }

      /** @summary Show menu */
      show(event) {
         this.remove();

         if (!event && this.show_evnt) event = this.show_evnt;

         document.body.addEventListener('click', this.remove_bind);

         let oldmenu = document.getElementById(this.menuname);
         if (oldmenu) oldmenu.parentNode.removeChild(oldmenu);

         $(document.body).append('<ul class="jsroot_ctxmenu">' + this.code + '</ul>');

         this.element = $('.jsroot_ctxmenu');

         let menu = this;

         this.element
            .attr('id', this.menuname)
            .css('left', event.clientX + window.pageXOffset)
            .css('top', event.clientY + window.pageYOffset)
//            .css('font-size', '80%')
            .css('position', 'absolute') // this overrides ui-menu-items class property
            .menu({
               items: "> :not(.ui-widget-header)",
               select: function( event, ui ) {
                  let arg = ui.item.attr('arg'),
                      cnt = ui.item.attr('cnt'),
                      func = cnt ? menu.funcs[cnt] : null;
                  menu.remove();
                  if (typeof func == 'function') {
                     if (menu.painter)
                        func.bind(menu.painter)(arg); // if 'painter' field set, returned as this to callback
                     else
                        func(arg);
                  }
              }
            });

         return JSROOT.loadScript('$$$style/jquery-ui').then(() => {

            let newx = null, newy = null;

            if (event.clientX + this.element.width() > $(window).width()) newx = $(window).width() - this.element.width() - 20;
            if (event.clientY + this.element.height() > $(window).height()) newy = $(window).height() - this.element.height() - 20;

            if (newx!==null) this.element.css('left', (newx>0 ? newx : 0) + window.pageXOffset);
            if (newy!==null) this.element.css('top', (newy>0 ? newy : 0) + window.pageYOffset);

            return new Promise(resolve => {
               this.resolveFunc = resolve;
            });
         });
      }

   } // class JQueryMenu

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
      let menu = new JQueryMenu(handler, menuname || 'root_ctx_menu', evnt);
      return Promise.resolve(menu);
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
