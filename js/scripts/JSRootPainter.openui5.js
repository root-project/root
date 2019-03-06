/// @file JSRootPainter.openui5.js
/// Part of JavaScript ROOT graphics, dependent from openui5 functionality
/// Openui5 loaded directly in the script

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['jquery', 'jquery-ui', 'd3', 'JSRootPainter', 'JSRootPainter.hierarchy', 'JSRootPainter.jquery' ], factory );
   } else {

      if (typeof jQuery == 'undefined')
         throw new Error('jQuery not defined', 'JSRootPainter.openui5.js');

      if (typeof jQuery.ui == 'undefined')
         throw new Error('jQuery-ui not defined','JSRootPainter.openui5.js');

      if (typeof d3 != 'object')
         throw new Error('This extension requires d3.v3.js', 'JSRootPainter.openui5.js');

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.openui5.js');

      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter not defined', 'JSRootPainter.openui5.js');

      factory(jQuery, jQuery.ui, d3, JSROOT);
   }
} (function($, myui, d3, JSROOT) {

   "use strict";

   JSROOT.sources.push("openui5");

   var load_callback = JSROOT.complete_script_load;
   delete JSROOT.complete_script_load; // normal callback is intercepted - we need to instantiate openui5

   JSROOT.completeUI5Loading = function() {
      // when running with THttpServer, automatically set "rootui5" folder
      var rootui5sys = undefined;
      if (JSROOT.source_dir.indexOf("jsrootsys") >= 0)
         rootui5sys = JSROOT.source_dir.replace(/jsrootsys/g, "rootui5sys");

      sap.ui.loader.config({
         paths: {
            jsroot: JSROOT.source_dir,
            rootui5: rootui5sys
         }
      });

      JSROOT.CallBack(load_callback);
      load_callback = null;
   }

   function TryOpenOpenUI(sources) {

      // where to take openui5 sources
      var src = sources.shift();

      if ((src.indexOf("roothandler")==0) && (src.indexOf("://")<0)) src = src.replace(/\:\//g,"://");

      var element = document.createElement("script");
      element.setAttribute('type', "text/javascript");
      element.setAttribute('id', "sap-ui-bootstrap");
      // use nojQuery while we are already load jquery and jquery-ui, later one can use directly sap-ui-core.js

      // this is location of openui5 scripts when working with THttpServer or when scripts are installed inside JSROOT
      element.setAttribute('src', src + "resources/sap-ui-core-nojQuery.js"); // latest openui5 version

      element.setAttribute('data-sap-ui-libs', JSROOT.openui5libs || "sap.m, sap.ui.layout, sap.ui.unified, sap.ui.commons");

      element.setAttribute('data-sap-ui-theme', 'sap_belize');
      element.setAttribute('data-sap-ui-compatVersion', 'edge');
      // element.setAttribute('data-sap-ui-bindingSyntax', 'complex');

      element.setAttribute('data-sap-ui-preload', 'async'); // '' to disable Component-preload.js

      // configure path for openui5 scripts
      // element.setAttribute('data-sap-ui-resourceroots', '{ "sap.ui.jsroot": "' + JSROOT.source_dir + 'openui5/" }');

      element.setAttribute('data-sap-ui-evt-oninit', "JSROOT.completeUI5Loading()");

      element.onerror = function() {
         // remove failed element
         element.parentNode.removeChild(element);
         // and try next
         TryOpenOpenUI(sources);
      }

      element.onload = function() {
         console.log('Load openui5 from ' + src);
      }

      document.getElementsByTagName("head")[0].appendChild(element);
   }

   var sources = [],
       openui5_dflt = "https://openui5.hana.ondemand.com/",
       openui5_root = JSROOT.source_dir.replace(/jsrootsys/g, "rootui5sys/distribution");

   if (openui5_root == JSROOT.source_dir) openui5_root = "";

   if (typeof JSROOT.openui5src == 'string') {
      switch (JSROOT.openui5src) {
         case "nodefault": openui5_dflt = ""; break;
         case "default": sources.push(openui5_dflt); openui5_dflt = ""; break;
         case "nojsroot": openui5_root = ""; break;
         case "jsroot": sources.push(openui5_root); openui5_root = ""; break;
         default: sources.push(JSROOT.openui5src); break;
      }

   }

   if (openui5_root && (sources.indexOf(openui5_root)<0)) sources.push(openui5_root);
   if (openui5_dflt && (sources.indexOf(openui5_dflt)<0)) sources.push(openui5_dflt);

   TryOpenOpenUI(sources);

   // function allows to create menu with openui
   // for the moment deactivated - can be used later
   JSROOT.Painter.createMenuNew = function(painter, maincallback) {

      var menu = { painter: painter,  element: null, cnt: 1, stack: [], items: [], separ: false };

      // this is slightly modified version of original MenuItem.render function.
      // need to be updated with any further changes
      function RenderCustomItem(rm, oItem, oMenu, oInfo) {
         var oSubMenu = oItem.getSubmenu();
         rm.write("<li ");

         var sClass = "sapUiMnuItm";
         if (oInfo.iItemNo == 1) {
            sClass += " sapUiMnuItmFirst";
         } else if (oInfo.iItemNo == oInfo.iTotalItems) {
            sClass += " sapUiMnuItmLast";
         }
         if (!oMenu.checkEnabled(oItem)) {
            sClass += " sapUiMnuItmDsbl";
         }
         if (oItem.getStartsSection()) {
            sClass += " sapUiMnuItmSepBefore";
         }

         rm.writeAttribute("class", sClass);
         if (oItem.getTooltip_AsString()) {
            rm.writeAttributeEscaped("title", oItem.getTooltip_AsString());
         }
         rm.writeElementData(oItem);

         // ARIA
         if (oInfo.bAccessible) {
            rm.writeAccessibilityState(oItem, {
               role: "menuitem",
               disabled: !oMenu.checkEnabled(oItem),
               posinset: oInfo.iItemNo,
               setsize: oInfo.iTotalItems,
               labelledby: {value: /*oMenu.getId() + "-label " + */this.getId() + "-txt " + this.getId() + "-scuttxt", append: true}
            });
            if (oSubMenu) {
               rm.writeAttribute("aria-haspopup", true);
               rm.writeAttribute("aria-owns", oSubMenu.getId());
            }
         }

         // Left border
         rm.write("><div class=\"sapUiMnuItmL\"></div>");

         // icon/check column
         rm.write("<div class=\"sapUiMnuItmIco\">");
         if (oItem.getIcon()) {
            rm.writeIcon(oItem.getIcon(), null, {title: null});
         }
         rm.write("</div>");

         // Text column
         rm.write("<div id=\"" + this.getId() + "-txt\" class=\"sapUiMnuItmTxt\">");
         rm.write(oItem.custom_html);
         rm.write("</div>");

         // Shortcut column
         rm.write("<div id=\"" + this.getId() + "-scuttxt\" class=\"sapUiMnuItmSCut\"></div>");

         // Submenu column
         rm.write("<div class=\"sapUiMnuItmSbMnu\">");
         if (oSubMenu) {
            rm.write("<div class=\"sapUiIconMirrorInRTL\"></div>");
         }
         rm.write("</div>");

         // Right border
         rm.write("<div class=\"sapUiMnuItmR\"></div>");

         rm.write("</li>");
      }

      sap.ui.define(['sap/ui/unified/Menu', 'sap/ui/unified/MenuItem', 'sap/ui/unified/MenuItemBase'],
                       function(sapMenu, sapMenuItem, sapMenuItemBase) {

         menu.add = function(name, arg, func) {
            if (name == "separator") { this.separ = true; return; }

            if (name.indexOf("header:")==0)
               return this.items.push(new sapMenuItem("", { text: name.substr(7), enabled: false }));

            if (name=="endsub:") {
               var last = this.stack.pop();
               last._item.setSubmenu(new sapMenu("", { items: this.items }));
               this.items = last._items;
               return;
            }

            var issub = false, checked = null;
            if (name.indexOf("sub:")==0) {
               name = name.substr(4);
               issub = true;
            }

            if (typeof arg == 'function') { func = arg; arg = name;  }

            if (name.indexOf("chk:")==0) { name = name.substr(4); checked = true; } else
            if (name.indexOf("unk:")==0) { name = name.substr(4); checked = false; }

            var item = new sapMenuItem("", { });

            if (!issub && (name.indexOf("<svg")==0)) {
               item.custom_html = name;
               item.render = RenderCustomItem;
            } else {
               item.setText(name);
               if (this.separ) item.setStartsSection(true);
               this.separ = false;
            }

            if (checked) item.setIcon("sap-icon://accept");

            this.items.push(item);

            if (issub) {
               this.stack.push({ _items: this.items, _item: item });
               this.items = [];
            }

            if (typeof func == 'function') {
               item.menu_func = func; // keep call-back function
               item.menu_arg = arg; // keep call-back argument
            }

            this.cnt++;
         }

         menu.addchk = function(flag, name, arg, func) {
            return this.add((flag ? "chk:" : "unk:") + name, arg, func);
         }

         menu.size = function() { return this.cnt-1; }

         menu.addDrawMenu = function(menu_name, opts, call_back) {
            if (!opts) opts = [];
            if (opts.length==0) opts.push("");

            var without_sub = false;
            if (menu_name.indexOf("nosub:")==0) {
               without_sub = true;
               menu_name = menu_name.substr(6);
            }

            if (opts.length === 1) {
               if (opts[0]==='inspect') menu_name = menu_name.replace("Draw", "Inspect");
               return this.add(menu_name, opts[0], call_back);
            }

            if (!without_sub) this.add("sub:" + menu_name, opts[0], call_back);

            for (var i=0;i<opts.length;++i) {
               var name = opts[i];
               if (name=="") name = '&lt;dflt&gt;';

               var group = i+1;
               if ((opts.length>5) && (name.length>0)) {
                  // check if there are similar options, which can be grouped once again
                  while ((group<opts.length) && (opts[group].indexOf(name)==0)) group++;
               }

               if (without_sub) name = menu_name + " " + name;

               if (group < i+2) {
                  this.add(name, opts[i], call_back);
               } else {
                  this.add("sub:" + name, opts[i], call_back);
                  for (var k=i+1;k<group;++k)
                     this.add(opts[k], opts[k], call_back);
                  this.add("endsub:");
                  i = group-1;
               }
            }
            if (!without_sub) this.add("endsub:");
         }

         menu.remove = function() {
            if (this.remove_bind) {
               document.body.removeEventListener('click', this.remove_bind);
               this.remove_bind = null;
            }
            if (this.element) {
               this.element.destroy();
               if (this.close_callback) this.close_callback();
            }
            this.element = null;
         }

         menu.remove_bind = menu.remove.bind(menu);

         menu.show = function(event, close_callback) {
            this.remove();

            if (typeof close_callback == 'function') this.close_callback = close_callback;

            var old = sap.ui.getCore().byId("root_context_menu");
            if (old) old.destroy();

            document.body.addEventListener('click', this.remove_bind);

            this.element = new sapMenu("root_context_menu", { items: this.items });

            // this.element.attachClosed({}, this.remove, this);

            this.element.attachItemSelect(null, this.menu_item_select, this);

            var eDock = sap.ui.core.Popup.Dock;
            // var oButton = oEvent.getSource();
            this.element.open(false, null, eDock.BeginTop, eDock.BeginTop, null, event.clientX + " " + event.clientY);
         }

         menu.menu_item_select = function(oEvent) {
            var item = oEvent.getParameter("item");
            if (!item || !item.menu_func) return;
            // console.log('select item arg', item.menu_arg);
            // console.log('select item', item.getText());
            if (this.painter)
               item.menu_func.bind(this.painter)(item.menu_arg);
            else
               item.menu_func(item.menu_arg);
         }

         JSROOT.CallBack(maincallback, menu);
      });

      return menu;
   }

   return JSROOT;

}));

