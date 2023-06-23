import { settings, browser, gStyle, isObject, isFunc, isStr, clTGaxis } from '../core.mjs';
import { rgb as d3_rgb, select as d3_select } from '../d3.mjs';
import { selectgStyle, saveSettings, readSettings, saveStyle, getColorExec } from './utils.mjs';
import { getColor } from '../base/colors.mjs';
import { TAttMarkerHandler } from '../base/TAttMarkerHandler.mjs';
import { getSvgLineStyle } from '../base/TAttLineHandler.mjs';
import { FontHandler } from '../base/FontHandler.mjs';

const kToFront = '__front__';

/**
 * @summary Abstract class for creating context menu
 *
 * @desc Use {@link createMenu} to create instance of the menu
 * @private
 */

class JSRootMenu {
   constructor(painter, menuname, show_event) {
      this.painter = painter;
      this.menuname = menuname;
      if (isObject(show_event) && (show_event.clientX !== undefined) && (show_event.clientY !== undefined))
         this.show_evnt = { clientX: show_event.clientX, clientY: show_event.clientY, skip_close: show_event.skip_close };

      this.remove_handler = () => this.remove();
      this.element = null;
      this.cnt = 0;
   }

   native() { return false; }

   async load() { return this; }

   /** @summary Returns object with mouse event position when context menu was actiavted
     * @desc Return object will have members 'clientX' and 'clientY' */
   getEventPosition() { return this.show_evnt; }

   add(/*name, arg, func, title*/) {
      throw Error('add() method has to be implemented in the menu');
   }

   /** @summary Returns menu size */
   size() { return this.cnt; }

   /** @summary Close and remove menu */
   remove() {
      if (!this.element)
         return;

      if (this.show_evnt?.skip_close) {
         this.show_evnt.skip_close = 0;
         return;
      }

      this.element.remove();
      this.element = null;
      if (isFunc(this.resolveFunc)) {
         let func = this.resolveFunc;
         delete this.resolveFunc;
         func();
      }
      document.body.removeEventListener('click', this.remove_handler);
   }

   show(/*event*/) {
      throw Error('show() method has to be implemented in the menu class');
   }

   /** @summary Add checked menu item
     * @param {boolean} flag - flag
     * @param {string} name - item name
     * @param {function} func - func called when item is selected */
   addchk(flag, name, arg, func, title) {
      let handler = func;
      if (isFunc(arg)) {
         title = func;
         func = arg;
         handler = res => func(res == '1');
         arg = flag ? '0' : '1';
      }
      this.add((flag ? 'chk:' : 'unk:') + name, arg, handler, title);
   }

   /** @summary Add draw sub-menu with draw options
     * @protected */
   addDrawMenu(top_name, opts, call_back) {
      if (!opts || !opts.length)
         return;

      let without_sub = false;
      if (top_name.indexOf('nosub:') == 0) {
         without_sub = true;
         top_name = top_name.slice(6);
      }

      if (opts.length === 1) {
         if (opts[0] === 'inspect')
            top_name = top_name.replace('Draw', 'Inspect');
         this.add(top_name, opts[0], call_back);
         return;
      }

      if (!without_sub) this.add('sub:' + top_name, () => {
         let opt = isFunc(this.painter?.getDrawOpt) ? this.painter.getDrawOpt() : opts[0];
         this.input('Provide draw option', opt, 'text').then(call_back);
      });

      for (let i = 0; i < opts.length; ++i) {
         let name = opts[i] || (this._use_plain_text ? '<dflt>' : '&lt;dflt&gt;'),
             group = i+1;
         if ((opts.length > 5) && name) {
            // check if there are similar options, which can be grouped once again
            while ((group < opts.length) && (opts[group].indexOf(name) == 0)) group++;
         }

         if (without_sub)
            name = top_name + ' ' + name;

         if (group < i+2) {
            this.add(name, opts[i], call_back);
         } else {
            this.add('sub:' + name, opts[i], call_back);
            for (let k = i+1; k < group; ++k)
               this.add(opts[k], opts[k], call_back);
            this.add('endsub:');
            i = group-1;
         }
      }
      if (!without_sub) this.add('endsub:');
   }

   /** @summary Add color selection menu entries
     * @protected */
   addColorMenu(name, value, set_func, fill_kind) {
      if (value === undefined) return;
      let useid = !isStr(value);
      this.add('sub:' + name, () => {
         this.input('Enter color ' + (useid ? '(only id number)' : '(name or id)'), value, useid ? 'int' : 'text', useid ? 0 : undefined, useid ? 9999 : undefined).then(col => {
            let id = parseInt(col);
            if (Number.isInteger(id) && getColor(id)) {
               col = getColor(id);
            } else {
               if (useid) return;
            }
            set_func(useid ? id : col);
         });
      });

      for(let ncolumn = 0; ncolumn < 5; ++ncolumn) {
         this.add('column:');

         for (let nrow = 0; nrow < 10; nrow++) {
            let n = ncolumn*10 + nrow;
            if (!useid) --n; // use -1 as none color

            let col = (n < 0) ? 'none' : getColor(n);
            if ((n == 0) && (fill_kind == 1)) col = 'none';
            let lbl = (n <= 0) || (col[0] != '#') ? col : `col ${n}`,
                fill = (n == 1) ? 'white' : 'black',
                stroke = (n == 1) ? 'red' : 'black',
                rect = (value == (useid ? n : col)) ? `<rect width="50" height="18" style="fill:none;stroke-width:3px;stroke:${stroke}"></rect>` : '',
                svg = `<svg width="50" height="18" style="margin:0px;background-color:${col}">${rect}<text x="4" y="12" style='font-size:12px' fill="${fill}">${lbl}</text></svg>`;

            this.add(svg, (useid ? n : col), res => set_func(useid ? parseInt(res) : res), 'Select color ' + col);
         }

         this.add('endcolumn:');
         if (!this.native()) break;
      }

      this.add('endsub:');
   }

   /** @summary Add size selection menu entries
     * @protected */
   addSizeMenu(name, min, max, step, size_value, set_func, title) {

      if (size_value === undefined) return;

      let values = [], miss_current = false;
      if (isObject(step)) {
         values = step; step = 1;
      } else for (let sz = min; sz <= max; sz += step)
         values.push(sz);

      const match = v => Math.abs(v-size_value) < (max - min)*1e-5,
            conv = (v, more) => {
               if ((v === size_value) && miss_current) more = true;
               if (step >= 1) return v.toFixed(0);
               if (step >= 0.1) return v.toFixed(more ? 2 : 1);
               return v.toFixed(more ? 4 : 2);
           };

      if (values.findIndex(match) < 0) {
         miss_current = true;
         values.push(size_value);
         values = values.sort((a,b) => a > b);
      }

      this.add('sub:' + name, () => this.input('Enter value of ' + name, conv(size_value, true), (step >= 1) ? 'int' : 'float').then(set_func), title);
      values.forEach(v => this.addchk(match(v), conv(v), v, res => set_func((step >= 1) ? parseInt(res) : parseFloat(res))));
      this.add('endsub:');
   }

   /** @summary Add palette menu entries
     * @protected */
   addPaletteMenu(curr, set_func) {
      const add = (id, name, title, more) => {
         if (!name)
            name = `pal ${id}`;
         else if (!title)
            title = name;
         if (title) title += `, code ${id}`;
         this.addchk((id === curr) || more, '<nobr>' + name + '</nobr>', id, set_func, title || name);
      };

      this.add('sub:Palette', () => this.input('Enter palette code [1..113]', curr, 'int', 1, 113).then(set_func));

      this.add('column:');

      add(57, 'Bird', 'Default color palette', (curr > 113));
      add(55, 'Rainbow');
      add(51, 'Deep Sea');
      add(52, 'Grayscale', 'New gray scale');
      add(1,  '', 'Old gray scale', (curr > 0) && (curr < 10));
      add(50, 'ROOT 5', 'Default color palette in ROOT 5', (curr >= 10) && (curr < 51));
      add(53, '', 'Dark body radiator');
      add(54, '', 'Two-color hue');
      add(56, '', 'Inverted dark body radiator');
      add(58, 'Cubehelix');
      add(59, '', 'Green Red Violet');
      add(60, '', 'Blue Red Yellow');
      add(61, 'Ocean');

      this.add('endcolumn:');

      if (!this.native())
         return this.add('endsub:');

      this.add('column:');

      add(62, '', 'Color Printable On Grey');
      add(63, 'Alpine');
      add(64, 'Aquamarine');
      add(65, 'Army');
      add(66, 'Atlantic');
      add(67, 'Aurora');
      add(68, 'Avocado');
      add(69, 'Beach');
      add(70, 'Black Body');
      add(71, '', 'Blue Green Yellow');
      add(72, 'Brown Cyan');
      add(73, 'CMYK');
      add(74, 'Candy');

      this.add('endcolumn:');
      this.add('column:');

      add(75, 'Cherry');
      add(76, 'Coffee');
      add(77, '', 'Dark Rain Bow');
      add(78, '', 'Dark Terrain');
      add(79, 'Fall');
      add(80, 'Fruit Punch');
      add(81, 'Fuchsia');
      add(82, 'Grey Yellow');
      add(83, '', 'Green Brown Terrain');
      add(84, 'Green Pink');
      add(85, 'Island');
      add(86, 'Lake');
      add(87, '', 'Light Temperature');

      this.add('endcolumn:');
      this.add('column:');

      add(88, '', 'Light Terrain');
      add(89, 'Mint');
      add(90, 'Neon');
      add(91, 'Pastel');
      add(92, 'Pearl');
      add(93, 'Pigeon');
      add(94, 'Plum');
      add(95, 'Red Blue');
      add(96, 'Rose');
      add(97, 'Rust');
      add(98, '', 'Sandy Terrain');
      add(99, 'Sienna');
      add(100, 'Solar');

      this.add('endcolumn:');
      this.add('column:');

      add(101, '', 'South West');
      add(102, '', 'Starry Night');
      add(103, '', 'Sunset');
      add(104, '', 'Temperature Map');
      add(105, '', 'Thermometer');
      add(106, 'Valentine');
      add(107, '', 'Visible Spectrum');
      add(108, '', 'Water Melon');
      add(109, 'Cool');
      add(110, 'Copper');
      add(111, '', 'Gist Earth');
      add(112, 'Viridis');
      add(113, 'Cividis');

      this.add('endcolumn:');

      this.add('endsub:');
   }

   /** @summary Add rebin menu entries
     * @protected */
   addRebinMenu(rebin_func) {
      this.add('sub:Rebin', () => this.input('Enter rebin value', 2, 'int', 2).then(rebin_func));
      for (let sz = 2; sz <= 7; sz++)
         this.add(sz.toString(), sz, res => rebin_func(parseInt(res)));
      this.add('endsub:');
   }

   /** @summary Add selection menu entries
     * @param {String} name - name of submenu
     * @param {Array} values - array of string entries used as list for selection
     * @param {String|Number} value - currently elected value, either name or index
     * @param {Function} set_func - function called when item selected, either name or index depending from value parameter
     * @protected */
   addSelectMenu(name, values, value, set_func) {
      let use_number = (typeof value == 'number');
      this.add('sub:' + name);
      for (let n = 0; n < values.length; ++n)
         this.addchk(use_number ? (n == value) : (values[n] == value), values[n], use_number ? n : values[n], res => set_func(use_number ? Number.parseInt(res) : res));
      this.add('endsub:');
   }

   /** @summary Add RColor selection menu entries
     * @protected */
   addRColorMenu(name, value, set_func) {
      // if (value === undefined) return;
      let colors = ['default', 'black', 'white', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan'];

      this.add('sub:' + name, () => {
         this.input('Enter color name - empty string will reset color', value).then(set_func);
      });
      let fillcol = 'black';
      for (let n = 0; n < colors.length; ++n) {
         let coltxt = colors[n], match = false, bkgr = '';
         if (n > 0) {
            bkgr = 'background-color:' + coltxt;
            fillcol = (coltxt == 'white') ? 'black' : 'white';

            if (isStr(value) && value && (value != 'auto') && (value[0] != '['))
               match = (d3_rgb(value).toString() == d3_rgb(coltxt).toString());
         } else {
            match = !value;
         }
         let svg = `<svg width='100' height='18' style='margin:0px;${bkgr}'><text x='4' y='12' style='font-size:12px' fill='${fillcol}'>${coltxt}</text></svg>`;
         this.addchk(match, svg, coltxt, res => set_func(res == 'default' ? null : res));
      }
      this.add('endsub:');
   }

   /** @summary Add items to change RAttrText
     * @protected */
   addRAttrTextItems(fontHandler, opts, set_func) {
      if (!opts) opts = {};
      this.addRColorMenu('color', fontHandler.color, value => set_func({ name: 'color', value }));
      if (fontHandler.scaled)
         this.addSizeMenu('size', 0.01, 0.10, 0.01, fontHandler.size /fontHandler.scale, value => set_func({ name: 'size', value }));
      else
         this.addSizeMenu('size', 6, 20, 2, fontHandler.size, value => set_func({ name: 'size', value }));

      this.addSelectMenu('family', ['Arial', 'Times New Roman', 'Courier New', 'Symbol'], fontHandler.name, value => set_func({ name: 'font_family', value }));

      this.addSelectMenu('style', ['normal', 'italic', 'oblique'], fontHandler.style || 'normal', res => set_func({ name: 'font_style', value: res == 'normal' ? null : res }));

      this.addSelectMenu('weight', ['normal', 'lighter', 'bold', 'bolder'], fontHandler.weight || 'normal', res => set_func({ name: 'font_weight', value: res == 'normal' ? null : res }));

      if (!opts.noalign)
         this.add('align');
      if (!opts.noangle)
         this.add('angle');
   }

   /** @summary Add line style menu
     * @private */
   addLineStyleMenu(name, value, set_func) {
      this.add('sub:'+name, () => this.input('Enter line style id (1-solid)', value, 'int', 1, 11).then(val => {
         if (getSvgLineStyle(val)) set_func(val);
      }));
      for (let n = 1; n < 11; ++n) {
         let dash = getSvgLineStyle(n),
             svg = `<svg width='100' height='14'><text x='2' y='13' style='font-size:12px'>${n}</text><line x1='30' y1='7' x2='100' y2='7' stroke='black' stroke-width='3' stroke-dasharray='${dash}'></line></svg>`;

         this.addchk((value == n), svg, n, arg => set_func(parseInt(arg)));
      }
      this.add('endsub:');
   }

   /** @summary Add fill style menu
     * @private */
   addFillStyleMenu(name, value, color_index, painter, set_func) {
      this.add('sub:' + name, () => {
         this.input('Enter fill style id (1001-solid, 3000..3010)', value, 'int', 0, 4000).then(id => {
            if ((id >= 0) && (id <= 4000)) set_func(id);
         });
      });

      let supported = [1, 1001, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3010, 3021, 3022];

      for (let n = 0; n < supported.length; ++n) {
         let svg = supported[n];
         if (painter) {
            let sample = painter.createAttFill({ std: false, pattern: supported[n], color: color_index || 1 });
            svg = `<svg width='100' height='18'><text x='1' y='12' style='font-size:12px'>${supported[n].toString()}</text><rect x='40' y='0' width='60' height='18' stroke='none' fill='${sample.getFillColor()}'></rect></svg>`;
         }
         this.addchk(value == supported[n], svg, supported[n], arg => set_func(parseInt(arg)));
      }
      this.add('endsub:');
   }

   /** @summary Add font selection menu
     * @private */
   addFontMenu(name, value, set_func) {
      let prec = value && Number.isInteger(value) ? value % 10 : 2;

      this.add('sub:' + name, () => {
         this.input('Enter font id from [0..20]', Math.floor(value/10), 'int', 0, 20).then(id => {
            if ((id >= 0) && (id <= 20)) set_func(id*10 + prec);
         });
      });

      this.add('column:');

      for (let n = 1; n < 20; ++n) {
         let id = n*10 + prec,
             handler = new FontHandler(id, 14),
             txt = d3_select(document.createElementNS('http://www.w3.org/2000/svg', 'text')),
             fullname = handler.getFontName(), qual = '';
         if (handler.weight) { qual += 'b'; fullname += ' ' + handler.weight; }
         if (handler.style) { qual += handler.style[0]; fullname += ' ' + handler.style; }
         if (qual) qual = ' ' + qual;
         txt.attr('x', 1).attr('y',15).text(fullname.split(' ')[0] + qual);
         handler.setFont(txt);

         let rect = (value != id) ? '' : `<rect width='90' height='18' style='fill:none;stroke:black'></rect>`,
             svg = `<svg width='90' height='18'>${txt.node().outerHTML}${rect}</svg>`;
         this.add(svg, id, arg => set_func(parseInt(arg)), `${id}: ${fullname}`);

         if (n == 10) {
            this.add('endcolumn:');
            this.add('column:');
         }
      }

      this.add('endcolumn:');
      this.add('endsub:');
   }

   /** @summary Add align selection menu
     * @private */
   addAlignMenu(name, value, set_func) {
      this.add(`sub:${name}`, () => {
         this.input('Enter align like 12 or 31', value).then(arg => {
            let id = parseInt(arg);
            if ((id < 11) || (id > 33)) return;
            let h = Math.floor(id/10), v = id % 10;
            if ((h > 0) && (h < 4) && (v > 0) && (v < 4)) set_func(id);
         });
      });

      const hnames = ['left', 'middle', 'right'], vnames = ['bottom', 'centered', 'top'];
      for (let h = 1; h < 4; ++h)
         for (let v = 1; v < 4; ++v)
            this.addchk(h*10+v == value, `${h*10+v}: ${hnames[h-1]} ${vnames[h-1]}`, h*10+v, arg => set_func(parseInt(arg)));

      this.add('endsub:');
   }

   /** @summary Fill context menu for graphical attributes in painter
     * @private */
   addAttributesMenu(painter, preffix) {
      // this method used to fill entries for different attributes of the object
      // like TAttFill, TAttLine, ....
      // all menu call-backs need to be rebind, while menu can be used from other painter

      if (!preffix) preffix = '';

      if (painter.lineatt?.used) {
         this.add(`sub:${preffix}Line att`);
         this.addSizeMenu('width', 1, 10, 1, painter.lineatt.width,
            arg => { painter.lineatt.change(undefined, arg); painter.interactiveRedraw(true, `exec:SetLineWidth(${arg})`); });
         this.addColorMenu('color', painter.lineatt.color,
            arg => { painter.lineatt.change(arg); painter.interactiveRedraw(true, getColorExec(arg, 'SetLineColor')); });
         this.addLineStyleMenu('style', painter.lineatt.style, id => {
            painter.lineatt.change(undefined, undefined, id);
            painter.interactiveRedraw(true, `exec:SetLineStyle(${id})`);
         });
         this.add('endsub:');

         if (painter.lineatt?.excl_side) {
            this.add('sub:Exclusion');
            this.add('sub:side');
            for (let side = -1; side <= 1; ++side)
               this.addchk((painter.lineatt.excl_side == side), side, side,
                  arg => { painter.lineatt.changeExcl(parseInt(arg)); painter.interactiveRedraw(); });
            this.add('endsub:');

            this.addSizeMenu('width', 10, 100, 10, painter.lineatt.excl_width,
               arg => { painter.lineatt.changeExcl(undefined, arg); painter.interactiveRedraw(); });

            this.add('endsub:');
         }
      }

      if (painter.fillatt?.used) {
         this.add(`sub:${preffix}Fill att`);
         this.addColorMenu('color', painter.fillatt.colorindx, arg => {
            painter.fillatt.change(arg, undefined, painter.getCanvSvg());
            painter.interactiveRedraw(true, getColorExec(arg, 'SetFillColor'));
         }, painter.fillatt.kind);
         this.addFillStyleMenu('style', painter.fillatt.pattern, painter.fillatt.colorindx, painter, id => {
            painter.fillatt.change(undefined, id, painter.getCanvSvg());
            painter.interactiveRedraw(true, `exec:SetFillStyle(${id})`);
         });
         this.add('endsub:');
      }

      if (painter.markeratt?.used) {
         this.add(`sub:${preffix}Marker att`);
         this.addColorMenu('color', painter.markeratt.color,
            arg => { painter.markeratt.change(arg); painter.interactiveRedraw(true, getColorExec(arg, 'SetMarkerColor'));});
         this.addSizeMenu('size', 0.5, 6, 0.5, painter.markeratt.size,
            arg => { painter.markeratt.change(undefined, undefined, arg); painter.interactiveRedraw(true, `exec:SetMarkerSize(${arg})`); });

         this.add('sub:style');
         let supported = [1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34];

         for (let n = 0; n < supported.length; ++n) {
            let clone = new TAttMarkerHandler({ style: supported[n], color: painter.markeratt.color, size: 1.7 }),
                svg = `<svg width='60' height='18'><text x='1' y='12' style='font-size:12px'>${supported[n].toString()}</text><path stroke='black' fill='${clone.fill?'black':'none'}' d='${clone.create(40, 8)}'></path></svg>`;

            this.addchk(painter.markeratt.style == supported[n], svg, supported[n],
               arg => { painter.markeratt.change(undefined, parseInt(arg)); painter.interactiveRedraw(true, `exec:SetMarkerStyle(${arg})`); });
         }
         this.add('endsub:');
         this.add('endsub:');
      }

      if (painter.textatt?.used) {
         this.add(`sub:${preffix}Text att`);

         this.addFontMenu('font', painter.textatt.font,
                         arg => { painter.textatt.change(arg); painter.interactiveRedraw(true, `exec:SetTextFont(${arg})`); });

         let rel = painter.textatt.size < 1.;

         this.addSizeMenu('size', rel ? 0.03 : 6, rel ? 0.20 : 26, rel ? 0.01 : 2, painter.textatt.size,
            arg => { painter.textatt.change(undefined, parseFloat(arg)); painter.interactiveRedraw(true, `exec:SetTextSize(${arg})`); });

         this.addColorMenu('color', painter.textatt.color,
            arg => { painter.textatt.change(undefined, undefined, arg); painter.interactiveRedraw(true, getColorExec(arg, 'SetTextColor')); });

         this.addAlignMenu('align', painter.textatt.align, arg => {
            painter.textatt.change(undefined, undefined, undefined, arg); painter.interactiveRedraw(true, `exec:SetTextAlign(${arg})`);
         });

         this.addSizeMenu('angle', -180, 180, 45, painter.textatt.angle,
            arg => { painter.textatt.change(undefined,  undefined, undefined, undefined, parseFloat(arg)); painter.interactiveRedraw(true, `exec:SetTextAngle(${arg})`); });

         this.add('endsub:');
      }

   }

   /** @summary Fill context menu for axis
     * @private */
   addTAxisMenu(EAxisBits, painter, faxis, kind) {
      let is_gaxis = faxis._typename == clTGaxis;

      this.add('Divisions', () => this.input('Set Ndivisions', faxis.fNdivisions, 'int', 0).then(val => {
         faxis.fNdivisions = val;  painter.interactiveRedraw('pad', `exec:SetNdivisions(${val})`, kind);
      }));

      this.add('sub:Labels');
      this.addchk(faxis.TestBit(EAxisBits.kCenterLabels), 'Center',
            arg => { faxis.InvertBit(EAxisBits.kCenterLabels); painter.interactiveRedraw('pad', `exec:CenterLabels(${arg})`, kind); });
      this.addchk(faxis.TestBit(EAxisBits.kLabelsVert), 'Rotate',
            arg => { faxis.InvertBit(EAxisBits.kLabelsVert); painter.interactiveRedraw('pad', `exec:SetBit(TAxis::kLabelsVert,${arg})`, kind); });
      this.addColorMenu('Color', faxis.fLabelColor,
            arg => { faxis.fLabelColor = arg; painter.interactiveRedraw('pad', getColorExec(arg, 'SetLabelColor'), kind); });
      this.addSizeMenu('Offset', 0, 0.1, 0.01, faxis.fLabelOffset,
            arg => { faxis.fLabelOffset = arg; painter.interactiveRedraw('pad', `exec:SetLabelOffset(${arg})`, kind); });
      let a = faxis.fLabelSize >= 1;
      this.addSizeMenu('Size', a ? 2 : 0.02, a ? 30 : 0.11, a ? 2 : 0.01, faxis.fLabelSize,
            arg => { faxis.fLabelSize = arg; painter.interactiveRedraw('pad', `exec:SetLabelSize(${arg})`, kind); });
      this.add('endsub:');
      this.add('sub:Title');
      this.add('SetTitle', () => {
         this.input('Enter axis title', faxis.fTitle).then(t => {
            faxis.fTitle = t;
            painter.interactiveRedraw('pad', `exec:SetTitle("${t}")`, kind);
         });
      });
      this.addchk(faxis.TestBit(EAxisBits.kCenterTitle), 'Center',
            arg => { faxis.InvertBit(EAxisBits.kCenterTitle); painter.interactiveRedraw('pad', `exec:CenterTitle(${arg})`, kind); });
      this.addchk(faxis.TestBit(EAxisBits.kOppositeTitle), 'Opposite',
             () => { faxis.InvertBit(EAxisBits.kOppositeTitle); painter.redrawPad(); });
      this.addchk(faxis.TestBit(EAxisBits.kRotateTitle), 'Rotate',
            arg => { faxis.InvertBit(EAxisBits.kRotateTitle); painter.interactiveRedraw('pad', is_gaxis ? `exec:SetBit(TAxis::kRotateTitle, ${arg ? true : false})` : `exec:RotateTitle(${arg})`, kind); });
      if (is_gaxis) {
         this.addColorMenu('Color', faxis.fTextColor,
               arg => { faxis.fTextColor = arg; painter.interactiveRedraw('pad', getColorExec(arg, 'SetTitleColor'), kind); });
      } else {
         this.addColorMenu('Color', faxis.fTitleColor,
               arg => { faxis.fTitleColor = arg; painter.interactiveRedraw('pad', getColorExec(arg, 'SetTitleColor'), kind); });
      }
      this.addSizeMenu('Offset', 0, 3, 0.2, faxis.fTitleOffset,
                      arg => { faxis.fTitleOffset = arg; painter.interactiveRedraw('pad', `exec:SetTitleOffset(${arg})`, kind); });
      a = faxis.fTitleSize >= 1;
      this.addSizeMenu('Size', a ? 2 : 0.02, a ? 30 : 0.11, a ? 2 : 0.01, faxis.fTitleSize,
                      arg => { faxis.fTitleSize = arg; painter.interactiveRedraw('pad', `exec:SetTitleSize(${arg})`, kind); });
      this.add('endsub:');
      this.add('sub:Ticks');
      if (is_gaxis) {
         this.addColorMenu('Color', faxis.fLineColor,
                  arg => { faxis.fLineColor = arg; painter.interactiveRedraw('pad', getColorExec(arg, 'SetLineColor'), kind); });
         this.addSizeMenu('Size', -0.05, 0.055, 0.01, faxis.fTickSize,
                  arg => { faxis.fTickSize = arg; painter.interactiveRedraw('pad', `exec:SetTickLength(${arg})`, kind); });
      } else {
         this.addColorMenu('Color', faxis.fAxisColor,
                  arg => { faxis.fAxisColor = arg; painter.interactiveRedraw('pad', getColorExec(arg, 'SetAxisColor'), kind); });
         this.addSizeMenu('Size', -0.05, 0.055, 0.01, faxis.fTickLength,
                  arg => { faxis.fTickLength = arg; painter.interactiveRedraw('pad', `exec:SetTickLength(${arg})`, kind); });
      }
      this.add('endsub:');

      if (is_gaxis)
         this.add('Options', () => this.input('Enter TGaxis options like +L or -G', faxis.fChopt, 'string').then(arg => {
             faxis.fChopt = arg; painter.interactiveRedraw('pad', `exec:SetOption("${arg}")`, kind);
         }));
   }

   /** @summary Fill menu to edit settings properties
     * @private */
   addSettingsMenu(with_hierarchy, alone, handle_func) {
      if (alone)
         this.add('header:Settings');
      else
         this.add('sub:Settings');

      this.add('sub:Files');

      if (with_hierarchy) {
         this.addchk(settings.OnlyLastCycle, 'Last cycle', flag => {
            settings.OnlyLastCycle = flag;
            if (handle_func) handle_func('refresh');
         });

         this.addchk(!settings.SkipStreamerInfos, 'Streamer infos', flag => {
            settings.SkipStreamerInfos = !flag;
            if (handle_func) handle_func('refresh');
         });
      }

      this.addchk(settings.UseStamp, 'Use stamp arg', flag => { settings.UseStamp = flag; });
      this.addSizeMenu('Max ranges', 1, 1000, [1, 10, 20, 50, 200, 1000], settings.MaxRanges, value => { settings.MaxRanges = value; }, 'Maximal number of ranges in single http request');

      this.addchk(settings.HandleWrongHttpResponse, 'Handle wrong http response', flag => { settings.HandleWrongHttpResponse = flag; });
      this.addchk(settings.WithCredentials, 'With credentials', flag => { settings.WithCredentials = flag; }, 'Submit http request with user credentials');

      this.add('endsub:');

      this.add('sub:Toolbar');
      this.addchk(settings.ToolBar === false, 'Off', flag => { settings.ToolBar = !flag; });
      this.addchk(settings.ToolBar === true, 'On', flag => { settings.ToolBar = flag; });
      this.addchk(settings.ToolBar === 'popup', 'Popup', flag => { settings.ToolBar = flag ? 'popup' : false; });
      this.add('separator');
      this.addchk(settings.ToolBarSide == 'left', 'Left side', flag => { settings.ToolBarSide = flag ? 'left' : 'right'; });
      this.addchk(settings.ToolBarVert, 'Vertical', flag => { settings.ToolBarVert = flag; });
      this.add('endsub:');

      this.add('sub:Interactive');
      this.addchk(settings.Tooltip, 'Tooltip', flag => { settings.Tooltip = flag; });
      this.addchk(settings.ContextMenu, 'Context menus', flag => { settings.ContextMenu = flag; });
      this.add('sub:Zooming');
      this.addchk(settings.Zooming,   'Global', flag => { settings.Zooming = flag; });
      this.addchk(settings.ZoomMouse, 'Mouse',  flag => { settings.ZoomMouse = flag; });
      this.addchk(settings.ZoomWheel, 'Wheel',  flag => { settings.ZoomWheel = flag; });
      this.addchk(settings.ZoomTouch, 'Touch',  flag => { settings.ZoomTouch = flag; });
      this.add('endsub:');
      this.addchk(settings.HandleKeys, 'Keypress handling', flag => { settings.HandleKeys = flag; });
      this.addchk(settings.MoveResize, 'Move and resize', flag => { settings.MoveResize = flag; });
      this.addchk(settings.DragAndDrop,'Drag and drop', flag => { settings.DragAndDrop = flag; });
      this.addchk(settings.DragGraphs, 'Drag graph points', flag => { settings.DragGraphs = flag; });
      this.addchk(settings.ProgressBox,'Progress box', flag => { settings.ProgressBox = flag; });
      this.add('endsub:');

      this.add('sub:Drawing');
      this.addSelectMenu('Optimize', ['None', 'Smart', 'Always'], settings.OptimizeDraw, value => { settings.OptimizeDraw = value; });
      this.addPaletteMenu(settings.Palette, pal => { settings.Palette = pal; });
      this.addchk(settings.AutoStat, 'Auto stat box', flag => { settings.AutoStat = flag; });
      this.addSelectMenu('Latex', ['Off', 'Symbols', 'Normal', 'MathJax', 'Force MathJax'], settings.Latex, value => { settings.Latex = value; });
      this.addSelectMenu('3D rendering', ['Default', 'WebGL', 'Image'], settings.Render3D, value => { settings.Render3D = value; });
      this.addSelectMenu('WebGL embeding', ['Default', 'Overlay', 'Embed'], settings.Embed3D, value => { settings.Embed3D = value; });

      this.add('endsub:');

      this.add('sub:Geometry');
      this.add('Grad per segment:  ' + settings.GeoGradPerSegm, () => this.input('Grad per segment in geometry', settings.GeoGradPerSegm, 'int', 1, 60).then(val => { settings.GeoGradPerSegm = val; }));
      this.addchk(settings.GeoCompressComp, 'Compress composites', flag => { settings.GeoCompressComp = flag; });
      this.add('endsub:');

      if (with_hierarchy)
         this.add('Hierarchy limit:  ' + settings.HierarchyLimit, () => this.input('Max number of items in hierarchy', settings.HierarchyLimit, 'int', 10, 100000).then(val => {
            settings.HierarchyLimit = val;
            if (handle_func) handle_func('refresh');
         }));

      this.add('Dark mode: ' + (settings.DarkMode ? 'On' : 'Off'), () => {
         settings.DarkMode = !settings.DarkMode;
         if (handle_func) handle_func('dark');
      });

      const setStyleField = arg => { gStyle[arg.slice(1)] = parseInt(arg[0]); },
            addStyleIntField = (name, field, arr) => {
         this.add('sub:' + name);
         for (let v = 0; v < arr.length; ++v)
            this.addchk(gStyle[field] == v, arr[v], `${v}${field}`, setStyleField);
         this.add('endsub:');
      };

      this.add('sub:gStyle');

      this.add('sub:Canvas');
      this.addColorMenu('Color', gStyle.fCanvasColor, col => { gStyle.fCanvasColor = col; });
      this.addchk(gStyle.fOptDate, 'Draw date', flag => { gStyle.fOptDate = flag ? 1 : 0; });
      this.addchk(gStyle.fOptFile, 'Draw item', flag => { gStyle.fOptFile = flag ? 1 : 0; });
      this.addSizeMenu('Date X', 0.01, 0.1, 0.01, gStyle.fDateX, x => { gStyle.fDateX = x; }, 'configure gStyle.fDateX for date/item name drawings');
      this.addSizeMenu('Date Y', 0.01, 0.1, 0.01, gStyle.fDateY, y => { gStyle.fDateY = y; }, 'configure gStyle.fDateY for date/item name drawings');
      this.add('endsub:');

      this.add('sub:Pad');
      this.addColorMenu('Color', gStyle.fPadColor, col => { gStyle.fPadColor = col; });
      this.add('sub:Grid');
      this.addchk(gStyle.fPadGridX, 'X', flag => { gStyle.fPadGridX = flag; });
      this.addchk(gStyle.fPadGridY, 'Y', flag => { gStyle.fPadGridY = flag; });
      this.addColorMenu('Color', gStyle.fGridColor, col => { gStyle.fGridColor = col; });
      this.addSizeMenu('Width', 1, 10, 1, gStyle.fGridWidth, w => { gStyle.fGridWidth = w; });
      this.addLineStyleMenu('Style', gStyle.fGridStyle, st => { gStyle.fGridStyle = st; });
      this.add('endsub:');
      addStyleIntField('Ticks X', 'fPadTickX', ['normal', 'ticks on both sides', 'labels on both sides']);
      addStyleIntField('Ticks Y', 'fPadTickY', ['normal', 'ticks on both sides', 'labels on both sides']);
      addStyleIntField('Log X', 'fOptLogx', ['off', 'on', 'log 2']);
      addStyleIntField('Log Y', 'fOptLogy', ['off', 'on', 'log 2']);
      addStyleIntField('Log Z', 'fOptLogz', ['off', 'on', 'log 2']);
      this.addchk(gStyle.fOptTitle == 1, 'Hist title', flag => { gStyle.fOptTitle = flag ? 1 : 0; });
      this.add('endsub:');

      this.add('sub:Frame');
      this.addColorMenu('Fill color', gStyle.fFrameFillColor, col => { gStyle.fFrameFillColor = col; });
      this.addFillStyleMenu('Fill style', gStyle.fFrameFillStyle, gStyle.fFrameFillColor, null, id => { gStyle.fFrameFillStyle = id; });
      this.addColorMenu('Line color', gStyle.fFrameLineColor, col => { gStyle.fFrameLineColor = col; });
      this.addSizeMenu('Line width', 1, 10, 1, gStyle.fFrameLineWidth, w => { gStyle.fFrameLineWidth = w; });
      this.addLineStyleMenu('Line style', gStyle.fFrameLineStyle, st => { gStyle.fFrameLineStyle = st; });
      this.addSizeMenu('Border size', 0, 10, 1, gStyle.fFrameBorderSize, sz => { gStyle.fFrameBorderSize = sz; });
      // fFrameBorderMode: 0,
      this.add('sub:Margins');
      this.addSizeMenu('Bottom', 0, 0.5, 0.05, gStyle.fPadBottomMargin, v => { gStyle.fPadBottomMargin = v; });
      this.addSizeMenu('Top', 0, 0.5, 0.05, gStyle.fPadTopMargin, v => { gStyle.fPadTopMargin = v; });
      this.addSizeMenu('Left', 0, 0.5, 0.05, gStyle.fPadLeftMargin, v => { gStyle.fPadLeftMargin = v; });
      this.addSizeMenu('Right', 0, 0.5, 0.05, gStyle.fPadRightMargin, v => { gStyle.fPadRightMargin = v; });
      this.add('endsub:');
      this.add('endsub:');

      this.add('sub:Title');
      this.addColorMenu('Fill color', gStyle.fTitleColor, col => { gStyle.fTitleColor = col; });
      this.addFillStyleMenu('Fill style', gStyle.fTitleStyle, gStyle.fTitleColor, null, id => { gStyle.fTitleStyle = id; });
      this.addColorMenu('Text color', gStyle.fTitleTextColor, col => { gStyle.fTitleTextColor = col; });
      this.addSizeMenu('Border size', 0, 10, 1, gStyle.fTitleBorderSize, sz => { gStyle.fTitleBorderSize = sz; });
      this.addSizeMenu('Font size', 0.01, 0.1, 0.01, gStyle.fTitleFontSize, sz => { gStyle.fTitleFontSize = sz; });
      this.addFontMenu('Font', gStyle.fTitleFont, fnt => { gStyle.fTitleFont = fnt; });
      this.addSizeMenu('X: ' + gStyle.fTitleX.toFixed(2), 0., 1., 0.1, gStyle.fTitleX, v => { gStyle.fTitleX = v; });
      this.addSizeMenu('Y: ' + gStyle.fTitleY.toFixed(2), 0., 1., 0.1, gStyle.fTitleY, v => { gStyle.fTitleY = v; });
      this.addSizeMenu('W: ' + gStyle.fTitleW.toFixed(2), 0., 1., 0.1, gStyle.fTitleW, v => { gStyle.fTitleW = v; });
      this.addSizeMenu('H: ' + gStyle.fTitleH.toFixed(2), 0., 1., 0.1, gStyle.fTitleH, v => { gStyle.fTitleH = v; });
      this.add('endsub:');

      this.add('sub:Stat box');
      this.addColorMenu('Fill color', gStyle.fStatColor, col => { gStyle.fStatColor = col; });
      this.addFillStyleMenu('Fill style', gStyle.fStatStyle, gStyle.fStatColor, null, id => { gStyle.fStatStyle = id; });
      this.addColorMenu('Text color', gStyle.fStatTextColor, col => { gStyle.fStatTextColor = col; });
      this.addSizeMenu('Border size', 0, 10, 1, gStyle.fStatBorderSize, sz => { gStyle.fStatBorderSize = sz; });
      this.addSizeMenu('Font size', 0, 30, 5, gStyle.fStatFontSize, sz => { gStyle.fStatFontSize = sz; });
      this.addFontMenu('Font', gStyle.fStatFont, fnt => { gStyle.fStatFont = fnt; });
      this.add('Stat format', () => this.input('Stat format', gStyle.fStatFormat).then(fmt => { gStyle.fStatFormat = fmt; }));
      this.addSizeMenu('X: ' + gStyle.fStatX.toFixed(2), 0.2, 1., 0.1, gStyle.fStatX, v => { gStyle.fStatX = v; });
      this.addSizeMenu('Y: ' + gStyle.fStatY.toFixed(2), 0.2, 1., 0.1, gStyle.fStatY, v => { gStyle.fStatY = v; });
      this.addSizeMenu('Width: ' + gStyle.fStatW.toFixed(2), 0.1, 1., 0.1, gStyle.fStatW, v => { gStyle.fStatW = v; });
      this.addSizeMenu('Height: ' + gStyle.fStatH.toFixed(2), 0.1, 1., 0.1, gStyle.fStatH, v => { gStyle.fStatH = v; });
      this.add('endsub:');

      this.add('sub:Legend');
      this.addColorMenu('Fill color', gStyle.fLegendFillColor, col => { gStyle.fLegendFillColor = col; });
      this.addSizeMenu('Border size', 0, 10, 1, gStyle.fLegendBorderSize, sz => { gStyle.fLegendBorderSize = sz; });
      this.addFontMenu('Font', gStyle.fLegendFont, fnt => { gStyle.fLegendFont = fnt; });
      this.addSizeMenu('Text size', 0, 0.1, 0.01, gStyle.fLegendTextSize, v => { gStyle.fLegendTextSize = v; }, 'legend text size, when 0 - auto adjustment is used');
      this.add('endsub:');

      this.add('sub:Histogram');
      this.addchk(gStyle.fHistMinimumZero, 'Base0', flag => { gStyle.fHistMinimumZero = flag; }, 'when true, BAR and LEGO drawing using base = 0');
      this.add('Text format', () => this.input('Paint text format', gStyle.fPaintTextFormat).then(fmt => { gStyle.fPaintTextFormat = fmt; }));
      this.add('Time offset', () => this.input('Time offset in seconds, default is 788918400 for 1/1/1995', gStyle.fTimeOffset, 'int').then(ofset => { gStyle.fTimeOffset = ofset; }));
      this.addSizeMenu('ErrorX: ' + gStyle.fErrorX.toFixed(2), 0., 1., 0.1, gStyle.fErrorX, v => { gStyle.fErrorX = v; });
      this.addSizeMenu('End error', 0, 12, 1, gStyle.fEndErrorSize, v => { gStyle.fEndErrorSize = v; }, 'size in pixels of end error for E1 draw options, gStyle.fEndErrorSize');
      this.addSizeMenu('Top margin', 0., 0.5, 0.05, gStyle.fHistTopMargin, v => { gStyle.fHistTopMargin = v; }, 'Margin between histogram top and frame top');
      this.addColorMenu('Fill color', gStyle.fHistFillColor, col => { gStyle.fHistFillColor = col; });
      this.addFillStyleMenu('Fill style', gStyle.fHistFillStyle, gStyle.fHistFillColor, null, id => { gStyle.fHistFillStyle = id; });
      this.addColorMenu('Line color', gStyle.fHistLineColor, col => { gStyle.fHistLineColor = col; });
      this.addSizeMenu('Line width', 1, 10, 1, gStyle.fHistLineWidth, w => { gStyle.fHistLineWidth = w; });
      this.addLineStyleMenu('Line style', gStyle.fHistLineStyle, st => { gStyle.fHistLineStyle = st; });
      this.add('endsub:');

      this.add('separator');
      this.add('sub:Predefined');
      ['Modern', 'Plain', 'Bold'].forEach(name => this.addchk((gStyle.fName == name), name, name, selectgStyle));
      this.add('endsub:');

      this.add('endsub:'); // gStyle

      this.add('separator');

      this.add('Save settings', () => {
         let promise = readSettings(true) ? Promise.resolve(true) : this.confirm('Save settings', 'Pressing OK one agreess that JSROOT will store settings as browser cookies');
         promise.then(res => { if (res) { saveSettings(); saveStyle(); } });
      }, 'Store settings and gStyle as cookies');
      this.add('Delete settings', () => { saveSettings(-1); saveStyle(-1); }, 'Delete settings and gStyle from cookies');

      if (!alone) this.add('endsub:');
   }

   /** @summary Run modal dialog
     * @return {Promise} with html element inside dialg
     * @private */
   async runModal() {
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
     * @return {Promise} with true when 'Ok' pressed or false when 'Cancel' pressed
     * @protected */
   async confirm(title, message) {
      return this.runModal(title, message, { btns: true, height: 120, width: 400 }).then(elem => { return !!elem; });
   }

   /** @summary Input value
     * @return {Promise} with input value
     * @param {string} title - input dialog title
     * @param value - initial value
     * @param {string} [kind] - use 'text' (default), 'number', 'float' or 'int'
     * @protected */
   async input(title, value, kind, min, max) {

      if (!kind) kind = 'text';
      let inp_type = (kind == 'int') ? 'number' : 'text', ranges = '';
      if ((value === undefined) || (value === null)) value = '';
      if (kind == 'int') {
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
            if (kind == 'float') {
               val = parseFloat(val);
               if (Number.isFinite(val))
                  resolveFunc(val);
            } else if (kind == 'int') {
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
     * @return {Promise} with method argument */
   async showMethodArgsDialog(method) {
      let dlg_id = this.menuname + '_dialog',
          main_content = '<form> <fieldset style="padding:0; border:0">';

      for (let n = 0; n < method.fArgs.length; ++n) {
         let arg = method.fArgs[n];
         arg.fValue = arg.fDefault;
         if (arg.fValue == '\"\"') arg.fValue = '';
         main_content += `<label for="${dlg_id}_inp${n}">${arg.fName}</label>
                          <input type='text' tabindex="${n+1}" id="${dlg_id}_inp${n}" value="${arg.fValue}" style="width:100%;display:block"/>`;
      }

      main_content += '</fieldset></form>';

      return new Promise(resolveFunc => {

         this.runModal(method.fClassName + '::' + method.fName, main_content, { btns: true, height: 100 + method.fArgs.length*60, width: 400, resizable: true }).then(element => {
            if (!element) return;
            let args = '';

            for (let k = 0; k < method.fArgs.length; ++k) {
               let arg = method.fArgs[k];
               let value = element.querySelector(`#${dlg_id}_inp${k}`).value;
               if (value === '') value = arg.fDefault;
               if ((arg.fTitle == 'Option_t*') || (arg.fTitle == 'const char*')) {
                  // check quotes,
                  // TODO: need to make more precise checking of escape characters
                  if (!value) value = '""';
                  if (value[0] != '"') value = '"' + value;
                  if (value[value.length-1] != '"') value += '"';
               }

               args += (k > 0 ? ',' : '') + value;
            }

            resolveFunc(args);
         });
      });
   }

   /** @summary Let input arguments from the Command
     * @return {Promise} with command argument */
   async showCommandArgsDialog(cmdname, args) {
      let dlg_id = this.menuname + '_dialog',
          main_content = '<form> <fieldset style="padding:0; border:0">';

      for (let n = 0; n < args.length; ++n)
         main_content += `<label for="${dlg_id}_inp${n}">arg${n+1}</label>`+
                         `<input type='text' id="${dlg_id}_inp${n}" value="${args[n]}" style="width:100%;display:block"/>`;

      main_content += '</fieldset></form>';

      return new Promise(resolveFunc => {

         this.runModal('Arguments for command ' + cmdname, main_content, { btns: true, height: 110 + args.length*60, width: 400, resizable: true}).then(element => {
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
 * @desc Use {@link createMenu} to create instance of the menu
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

   native() { return true; }

   /** @summary Load required modules, noop for that menu class */
   async load() { return this; }

   /** @summary Add menu item
     * @param {string} name - item name
     * @param {function} func - func called when item is selected */
   add(name, arg, func, title) {
      let curr = this.stack[this.stack.length-1];

      if (name == 'separator')
         return curr.push({ divider: true });

      if (name.indexOf('header:') == 0)
         return curr.push({ text: name.slice(7), header: true });

      if (name == 'endsub:') {
         this.stack.pop();
         curr = this.stack[this.stack.length-1];
         if (curr[curr.length-1].sub.length == 0)
            curr[curr.length-1].sub = undefined;
         return;
      }

      if (name == 'endcolumn:')
         return this.stack.pop();


      if (isFunc(arg)) { title = func; func = arg; arg = name; }

      let elem = {};
      curr.push(elem);

      if (name == 'column:') {
         elem.column = true;
         elem.sub = [];
         this.stack.push(elem.sub);
         return;
      }

      if (name.indexOf('sub:') == 0) {
         name = name.slice(4);
         elem.sub = [];
         this.stack.push(elem.sub);
      }

      if (name.indexOf('chk:') == 0) {
         elem.checked = true;
         name = name.slice(4);
      } else if (name.indexOf('unk:') == 0) {
         elem.checked = false;
         name = name.slice(4);
      }

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

      const container_style =
         'position: absolute; top: 0; user-select: none; z-index: 100000; background-color: rgb(250, 250, 250); margin: 0; padding: 0px; width: auto;'+
         'min-width: 100px; box-shadow: 0px 0px 10px rgb(0, 0, 0, 0.2); border: 3px solid rgb(215, 215, 215); font-family: Arial, helvetica, sans-serif, serif;'+
         'font-size: 13px; color: rgb(0, 0, 0, 0.8); line-height: 15px;';

      //if loc !== document.body then its a submenu, so it needs to have position: relative;
      if (loc === document.body) {
         //delete all elements with className jsroot_ctxt_container
         let deleteElems = document.getElementsByClassName('jsroot_ctxt_container');
         while (deleteElems.length > 0)
            deleteElems[0].parentNode.removeChild(deleteElems[0]);

         outer.className = 'jsroot_ctxt_container';
         outer.style = container_style;
         outer.style.position = 'fixed';
         outer.style.left = left + 'px';
         outer.style.top = top + 'px';

      } else if ((left < 0) && (top == left)) {
         // column
         outer.className = 'jsroot_ctxt_column';
         outer.style.float = 'left';
         outer.style.width = (100/-left).toFixed(1) + '%';
      } else {
         outer.className = 'jsroot_ctxt_container';
         outer.style = container_style;
         outer.style.left = -loc.offsetLeft + loc.offsetWidth + 'px';
      }

      let need_check_area = false, ncols = 0;
      menu.forEach(d => {
         if (d.checked) need_check_area = true;
         if (d.column) ncols++;
      });

      menu.forEach(d => {
         if (ncols > 0) {
            outer.style.display = 'flex';
            if (d.column) this._buildContextmenu(d.sub, -ncols, -ncols, outer);
            return;
         }

         if (d.divider) {
            let hr = document.createElement('hr');
            hr.style = 'width: 85%; margin: 3px auto; border: 1px solid rgb(0, 0, 0, 0.15)';
            outer.appendChild(hr);
            return;
         }

         let item = document.createElement('div');
         item.style.position = 'relative';
         outer.appendChild(item);

         if (d.header) {
            item.style = 'background-color: lightblue; padding: 3px 7px; font-weight: bold; border-bottom: 1px;';
            item.innerHTML = d.text;
            return;
         }

         let hovArea = document.createElement('div');
         hovArea.style.width = '100%';
         hovArea.style.height = '100%';
         hovArea.style.display = 'flex';
         hovArea.style.justifyContent = 'space-between';
         hovArea.style.cursor = 'pointer';
         if (d.title) hovArea.setAttribute('title', d.title);

         item.appendChild(hovArea);
         if (!d.text) d.text = 'item';

         let text = document.createElement('div');
         text.style = 'margin: 0; padding: 3px 7px; pointer-events: none; white-space: nowrap';

         if (d.text.indexOf('<svg') >= 0) {
            if (need_check_area) {
               text.style.display = 'flex';

               let chk = document.createElement('span');
               chk.innerHTML = d.checked ? '\u2713' : '';
               chk.style.display = 'inline-block';
               chk.style.width = '1em';
               text.appendChild(chk);

               let sub = document.createElement('div');
               sub.innerHTML = d.text;
               text.appendChild(sub);
            } else {
               text.innerHTML = d.text;
            }

         } else {
            if (need_check_area) {
               let chk = document.createElement('span');
               chk.innerHTML = d.checked ? '\u2713' : '';
               chk.style.display = 'inline-block';
               chk.style.width = '1em';
               text.appendChild(chk);
            }

            let sub = document.createElement('span');
            if (d.text.indexOf('<nobr>') == 0)
               sub.textContent = d.text.slice(6, d.text.length-7);
            else
               sub.textContent = d.text;
            text.appendChild(sub);

         }

         hovArea.appendChild(text);

         function changeFocus(item, on) {
            if (on) {
               item.classList.add('jsroot_ctxt_focus');
               item.style['background-color'] = 'rgb(220, 220, 220)';
            } else if (item.classList.contains('jsroot_ctxt_focus')) {
               item.style['background-color'] = null;
               item.classList.remove('jsroot_ctxt_focus');
               item.querySelector('.jsroot_ctxt_container')?.remove()
            }
         }

         if (d.hasOwnProperty('extraText') || d.sub) {
            let extraText = document.createElement('span');
            extraText.className = 'jsroot_ctxt_extraText';
            extraText.style = 'margin: 0; padding: 3px 7px; color: rgb(0, 0, 0, 0.6);';
            extraText.textContent = d.sub ? '\u25B6' : d.extraText;
            hovArea.appendChild(extraText);

            if (d.sub && browser.touches)
               extraText.addEventListener('click', evnt => {
                  evnt.preventDefault();
                  evnt.stopPropagation();
                  let was_active = item.parentNode.querySelector('.jsroot_ctxt_focus');

                  if (was_active)
                     changeFocus(was_active, false);

                  if (item !== was_active) {
                     changeFocus(item, true);
                     this._buildContextmenu(d.sub, 0, 0, item);
                  }
               });
         }

         if (!browser.touches)
         hovArea.addEventListener('mouseenter', () => {

            if (this.prevHovArea)
               this.prevHovArea.style['background-color'] = null;
            hovArea.style['background-color'] = 'rgb(235, 235, 235)';
            this.prevHovArea = hovArea;

            outer.childNodes.forEach(chld => changeFocus(chld, false));

            if (d.sub) {
               changeFocus(item, true);
               this._buildContextmenu(d.sub, 0, 0, item);
            }
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
            outer.style.left = (docWidth - outer.offsetWidth) + 'px';
         }
         if (outer.offsetHeight > docHeight) {
            //is the contextmenu height larger than the window height?
            outer.style.top = 0;
            outer.style.overflowY = 'scroll';
            outer.style.overflowX = 'hidden';
            outer.style.height = docHeight + 'px';
         } else if (top + outer.offsetHeight > docHeight) {
            //Does contextmenu overflow window height?
            outer.style.top = (docHeight - outer.offsetHeight) + 'px';
         }

      } else if (outer.className != 'jsroot_ctxt_column') {

         // if its sub-contextmenu
         let dimensionsLoc = loc.getBoundingClientRect(), dimensionsOuter = outer.getBoundingClientRect();

         //Does sub-contextmenu overflow window width?
         if (dimensionsOuter.left + dimensionsOuter.width > docWidth) {
            outer.style.left = (-loc.offsetLeft - dimensionsOuter.width) + 'px';
         }

         if (dimensionsOuter.height > docHeight) {
            //is the sub-contextmenu height larger than the window height?
            outer.style.top = -dimensionsOuter.top + 'px';
            outer.style.overflowY = 'scroll';
            outer.style.overflowX = 'hidden';
            outer.style.height = docHeight + 'px';
         } else if (dimensionsOuter.height < docHeight && dimensionsOuter.height > docHeight / 2) {
            //is the sub-contextmenu height smaller than the window height AND larger than half of window height?
            if (dimensionsOuter.top - docHeight / 2 >= 0) { //If sub-contextmenu is closer to bottom of the screen
               outer.style.top = (-dimensionsOuter.top - dimensionsOuter.height + docHeight) + 'px';
            } else { //If sub-contextmenu is closer to top of the screen
               outer.style.top = (-dimensionsOuter.top) + 'px';
            }
         } else if (dimensionsOuter.top + dimensionsOuter.height > docHeight) {
            //Does sub-contextmenu overflow window height?
            outer.style.top = (-dimensionsOuter.height + dimensionsLoc.height) + 'px';
         }
      }
      return outer;
   }

   /** @summary Show standalone menu */
   async show(event) {
      this.remove();

      if (!event && this.show_evnt) event = this.show_evnt;

      document.body.addEventListener('click', this.remove_handler);

      let oldmenu = document.getElementById(this.menuname);
      if (oldmenu) oldmenu.remove();

      this.element = this._buildContextmenu(this.code, (event?.clientX || 0) + window.pageXOffset, (event?.clientY || 0) + window.pageYOffset, document.body);

      this.element.setAttribute('id', this.menuname);

      return this;
   }

   /** @summary Run modal elements with standalone code */
   async runModal(title, main_content, args) {
      if (!args) args = {};
      let dlg_id = this.menuname + '_dialog';
      d3_select(`#${dlg_id}`).remove();
      d3_select(`#${dlg_id}_block`).remove();

      let w = Math.min(args.width || 450, Math.round(0.9*browser.screenWidth)),
          block = d3_select('body').append('div')
                                   .attr('id', `${dlg_id}_block`)
                                   .attr('class', 'jsroot_dialog_block')
                                   .attr('style', 'z-index: 100000; position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 0.2; background-color: white'),
          element = d3_select('body')
                      .append('div')
                      .attr('id', dlg_id)
                      .attr('class', 'jsroot_dialog')
                      .style('position', 'absolute')
                      .style('width', `${w}px`)
                      .style('left', '50%')
                      .style('top', '50%')
                      .style('z-index', 100001)
                      .attr('tabindex', '0')
                      .html(
         `<div style='position: relative; left: -50%; top: -50%; border: solid green 3px; padding: 5px; display: flex; flex-flow: column; background-color: white'>`+
           `<div style='flex: 0 1 auto; padding: 5px'>${title}</div>`+
           `<div class='jsroot_dialog_content' style='flex: 1 1 auto; padding: 5px'>${main_content}</div>`+
           `<div class='jsroot_dialog_footer' style='flex: 0 1 auto; padding: 5px'>`+
              `<button class='jsroot_dialog_button' style='float: right; margin-right: 1em'>Ok</button>`+
              (args.btns ? `<button class='jsroot_dialog_button' style='float: right; margin-right: 1em'>Cancel</button>` : '') +
         `</div></div>`);

      return new Promise(resolveFunc => {
         element.on('keyup', evnt => {
            if ((evnt.code == 'Enter') || (evnt.code == 'Escape')) {
               evnt.preventDefault();
               evnt.stopPropagation();
               resolveFunc(evnt.code == 'Enter' ? element.node() : null);
               element.remove();
               block.remove();
            }
         });
         element.on('keydown', evnt => {
            if ((evnt.code == 'Enter') || (evnt.code == 'Escape')) {
               evnt.preventDefault();
               evnt.stopPropagation();
            }
         });
         element.selectAll('.jsroot_dialog_button').on('click', evnt => {
            resolveFunc(args.btns && (d3_select(evnt.target).text() == 'Ok') ? element.node() : null);
            element.remove();
            block.remove();
         });

         let f = element.select('.jsroot_dialog_content').select('input');
         if (f.empty()) f = element.select('.jsroot_dialog_footer').select('button');
         if (!f.empty()) f.node().focus();
      });
   }

} // class StandaloneMenu

/** @summary Create JSROOT menu
  * @desc See {@link JSRootMenu} class for detailed list of methods
  * @param {object} [evnt] - event object like mouse context menu event
  * @param {object} [handler] - object with handling function, in this case one not need to bind function
  * @param {string} [menuname] - optional menu name
  * @example
  * import { createMenu } from 'https://root.cern/js/latest/modules/gui/menu.mjs';
  * let menu = await createMenu());
  * menu.add('First', () => console.log('Click first'));
  * let flag = true;
  * menu.addchk(flag, 'Checked', arg => console.log(`Now flag is ${arg}`));
  * menu.show(); */
function createMenu(evnt, handler, menuname) {
   let menu = new StandaloneMenu(handler, menuname || 'root_ctx_menu', evnt);
   return menu.load();
}

/** @summary Close previousely created and shown JSROOT menu
  * @param {string} [menuname] - optional menu name */
function closeMenu(menuname) {
   let element = document.getElementById(menuname || 'root_ctx_menu');
   element?.remove();
   return !!element;
}

/** @summary Fill and show context menu for painter object
  * @private */
function showPainterMenu(evnt, painter, kind) {
   if (isFunc(evnt.stopPropagation)) {
      evnt.stopPropagation(); // disable main context menu
      evnt.preventDefault();  // disable browser context menu
   }

   createMenu(evnt, painter).then(menu => {
      painter.fillContextMenu(menu);
      if ((kind == kToFront) && isFunc(painter.bringToFront)) {
         menu.add('Bring to front', () => painter.bringToFront(true));
         kind = undefined;
      }
      return painter.fillObjectExecMenu(menu, kind);
   }).then(menu => menu.show());
}

/** @summary Assign handler for context menu for painter draw element
  * @private */
function assignContextMenu(painter, kind) {
   if (!painter?.isBatchMode() && painter?.draw_g)
      painter.draw_g.on('contextmenu', settings.ContextMenu ? evnt => showPainterMenu(evnt, painter, kind) : null);
}

export { createMenu, closeMenu, showPainterMenu, assignContextMenu, kToFront };
