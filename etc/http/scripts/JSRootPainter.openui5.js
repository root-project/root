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

   JSROOT.sources.push("openui5");

   var load_callback = JSROOT.complete_script_load;
   JSROOT.complete_script_load = null; // normal callback is intercepted - we need to instantiate openui5

   JSROOT.completeUI5Loading = function() {
      console.log('complete ui5 loading', typeof sap);
      JSROOT.sap = sap;
      JSROOT.CallBack(load_callback);
      load_callback = null;
   }

   console.log('start ui5 loading ', typeof jQuery);

   var element = document.createElement("script");

   element.setAttribute('type', "text/javascript");
   element.setAttribute('id', "sap-ui-bootstrap");
   // use nojQuery while we are already load jquery and jquery-ui, later one can use directly sap-ui-core.js
   element.setAttribute('src', "https://openui5.hana.ondemand.com/resources/sap-ui-core-nojQuery.js");
//   element.setAttribute('data-sap-ui-trace', "true");
   element.setAttribute('data-sap-ui-libs', "sap.m,sap.ui.table,sap.ui.commons,sap.tnt");
//   element.setAttribute('data-sap-ui-areas', "uiArea1");

   element.setAttribute('data-sap-ui-theme', 'sap_belize');
   element.setAttribute('data-sap-ui-compatVersion', 'edge');
   element.setAttribute('data-sap-ui-preload', 'async');

   element.setAttribute('data-sap-ui-evt-oninit', "JSROOT.completeUI5Loading()");

   document.getElementsByTagName("head")[0].appendChild(element);



   JSROOT.Painter.createMenu = function(painter, maincallback) {
      var menuname = 'root_ctx_menu';

      if (!maincallback && typeof painter==='function') { maincallback = painter; painter = null; }

      var menu = { painter: painter,  element: null, code: "", cnt: 1, funcs: {}, separ: false };

      menu.add = function(name, arg, func) {
         if (name == "separator") { this.code += "<li>-</li>"; this.separ = true; return; }

         if (name.indexOf("header:")==0) {
            this.code += "<li class='ui-widget-header' style='padding:3px; padding-left:5px;'>"+name.substr(7)+"</li>";
            return;
         }

         if (name=="endsub:") { this.code += "</ul></li>"; return; }
         var close_tag = "</li>", style = "";
         if (name.indexOf("sub:")==0) { name = name.substr(4); close_tag="<ul>"; /* style += ";padding-right:2em" */}

         if (typeof arg == 'function') { func = arg; arg = name;  }

         // if ((arg==null) || (typeof arg != 'string')) arg = name;

         var item = "";

         if (name.indexOf("chk:")==0) { item = "<span class='ui-icon ui-icon-check' style='margin:1px'></span>"; name = name.substr(4); } else
         if (name.indexOf("unk:")==0) { item = "<span class='ui-icon ui-icon-blank' style='margin:1px'></span>"; name = name.substr(4); }

         // special handling of first versions with menu support
         if (($.ui.version.indexOf("1.10")==0) || ($.ui.version.indexOf("1.9")==0))
            item = '<a href="#">' + item + name + '</a>';
         else
         if ($.ui.version.indexOf("1.11")==0)
            item += name;
         else
            item = '<div>' + item + name + '</div>';

         this.code += "<li cnt='" + this.cnt + "' arg='" + arg + "' style='" + style + "'>" + item + close_tag;
         if (typeof func == 'function') this.funcs[this.cnt] = func; // keep call-back function

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
         if (this.element!==null) {
            this.element.remove();
            if (this.close_callback) this.close_callback();
            document.body.removeEventListener('click', this.remove_bind);
         }
         this.element = null;
      }

      menu.remove_bind = menu.remove.bind(menu);

      menu.show = function(event, close_callback) {
         this.remove();

         if (typeof close_callback == 'function') this.close_callback = close_callback;

         document.body.addEventListener('click', this.remove_bind);

         var oldmenu = document.getElementById(menuname);
         if (oldmenu) oldmenu.parentNode.removeChild(oldmenu);

         $(document.body).append('<ul class="jsroot_ctxmenu">' + this.code + '</ul>');

         this.element = $('.jsroot_ctxmenu');

         var pthis = this;

         this.element
            .attr('id', menuname)
            .css('left', event.clientX + window.pageXOffset)
            .css('top', event.clientY + window.pageYOffset)
//            .css('font-size', '80%')
            .css('position', 'absolute') // this overrides ui-menu-items class property
            .menu({
               items: "> :not(.ui-widget-header)",
               select: function( event, ui ) {
                  var arg = ui.item.attr('arg'),
                      cnt = ui.item.attr('cnt'),
                      func = cnt ? pthis.funcs[cnt] : null;
                  pthis.remove();
                  if (typeof func == 'function') {
                     if ('painter' in menu)
                        func.bind(pthis.painter)(arg); // if 'painter' field set, returned as this to callback
                     else
                        func(arg);
                  }
              }
            });

         var newx = null, newy = null;

         if (event.clientX + this.element.width() > $(window).width()) newx = $(window).width() - this.element.width() - 20;
         if (event.clientY + this.element.height() > $(window).height()) newy = $(window).height() - this.element.height() - 20;

         if (newx!==null) this.element.css('left', (newx>0 ? newx : 0) + window.pageXOffset);
         if (newy!==null) this.element.css('top', (newy>0 ? newy : 0) + window.pageYOffset);
      }

      JSROOT.CallBack(maincallback, menu);

      return menu;
   }


   return JSROOT;

}));

