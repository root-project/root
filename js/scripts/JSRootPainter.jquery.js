/// @file JSRootPainter.jquery.js
/// Part of JavaScript ROOT graphics, dependent from jQuery functionality

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['jquery', 'jquery-ui', 'd3', 'JSRootPainter', 'JSRootPainter.hierarchy'], factory );
   } else {

      if (typeof jQuery == 'undefined')
         throw new Error('jQuery not defined', 'JSRootPainter.jquery.js');

      if (typeof jQuery.ui == 'undefined')
         throw new Error('jQuery-ui not defined','JSRootPainter.jquery.js');

      if (typeof d3 != 'object')
         throw new Error('This extension requires d3.v3.js', 'JSRootPainter.jquery.js');

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.jquery.js');

      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter not defined', 'JSRootPainter.jquery.js');

      factory(jQuery, jQuery.ui, d3, JSROOT);
   }
} (function($, myui, d3, JSROOT) {

   "use strict";

   JSROOT.sources.push("jq2d");

   if ( typeof define === "function" && define.amd )
      JSROOT.loadScript('$$$style/jquery-ui.css');

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

   // =================================================================================================

   var BrowserLayout = JSROOT.BrowserLayout;

   /// set browser title text
   /// Title also used for dragging of the float browser
   BrowserLayout.prototype.SetBrowserTitle = function(title) {
      var main = d3.select("#" + this.gui_div + " .jsroot_browser");
      if (!main.empty())
         main.select(".jsroot_browser_title").text(title);
   }

   BrowserLayout.prototype.ToggleBrowserKind = function(kind) {

      if (!this.gui_div) return;

      if (!kind) {
         if (!this.browser_kind) return;
         kind = (this.browser_kind === "float") ? "fix" : "float";
      }

      var main = d3.select("#"+this.gui_div+" .jsroot_browser"),
          jmain = $(main.node()),
          area = jmain.find(".jsroot_browser_area"),
          pthis = this;

      if (this.browser_kind === "float") {
          area.css('bottom', '0px')
              .css('top', '0px')
              .css('width','').css('height','')
              .toggleClass('jsroot_float_browser', false)
              .resizable("destroy")
              .draggable("destroy");
      } else if (this.browser_kind === "fix") {
         main.select(".jsroot_v_separator").remove();
         area.css('left', '0px');
         d3.select("#"+this.gui_div+"_drawing").style('left','0px'); // reset size
         main.select(".jsroot_h_separator").style('left','0px');
         d3.select("#"+this.gui_div+"_status").style('left','0px'); // reset left
         pthis.CheckResize();
      }

      this.browser_kind = kind;
      this.browser_visible = true;

      if (kind==="float") {
         area.css('bottom', '40px')
           .toggleClass('jsroot_float_browser', true)
           .resizable({
              containment: "parent",
              minWidth: 100,
              resize: function( event, ui ) {
                 pthis.SetButtonsPosition();
              },
              stop: function( event, ui ) {
                 var bottom = $(this).parent().innerHeight() - ui.position.top - ui.size.height;
                 if (bottom<7) $(this).css('height', "").css('bottom', 0);
              }
         })
         .draggable({
             containment: "parent",
             handle : $("#"+this.gui_div).find(".jsroot_browser_title"),
             snap: true,
             snapMode: "inner",
             snapTolerance: 10,
             drag: function( event, ui ) {
                pthis.SetButtonsPosition();
             },
             stop: function( event, ui ) {
                var bottom = $(this).parent().innerHeight() - $(this).offset().top - $(this).outerHeight();
                if (bottom<7) $(this).css('height', "").css('bottom', 0);
             }
          });
         this.AdjustBrowserSize();

     } else {

        area.css('left',0).css('top',0).css('bottom',0).css('height','');

        var vsepar =
           main.append('div')
               .classed("jsroot_separator", true).classed('jsroot_v_separator', true)
               .style('position', 'absolute').style('top',0).style('bottom',0);
        // creation of vertical separator
        $(vsepar.node()).draggable({
           axis: "x" , cursor: "ew-resize",
           containment: "parent",
           helper : function() { return $(this).clone().css('background-color','grey'); },
           drag: function(event,ui) {
              pthis.SetButtonsPosition();
              pthis.AdjustSeparator(ui.position.left, null);
           },
           stop: function(event,ui) {
              pthis.CheckResize();
           }
        });

        this.AdjustSeparator(250, null, true, true);
     }

      this.SetButtonsPosition();
   }

   BrowserLayout.prototype.SetButtonsPosition = function() {
      if (!this.gui_div) return;

      var jmain = $("#"+this.gui_div+" .jsroot_browser"),
          btns = jmain.find(".jsroot_browser_btns"),
          top = 7, left = 7;

      if (!btns.length) return;

      if (this.browser_visible) {
         var area = jmain.find(".jsroot_browser_area"),
             off0 = jmain.offset(), off1 = area.offset();
         top = off1.top - off0.top + 7;
         left = off1.left - off0.left + area.innerWidth() - 27;
      }

      btns.css('left', left+'px').css('top', top+'px');
   }

   BrowserLayout.prototype.AdjustBrowserSize = function(onlycheckmax) {
      if (!this.gui_div || (this.browser_kind !== "float")) return;

      var jmain = $("#" + this.gui_div + " .jsroot_browser");
      if (!jmain.length) return;

      var area = jmain.find(".jsroot_browser_area"),
          cont = jmain.find(".jsroot_browser_hierarchy"),
          chld = cont.children(":first");

      if (onlycheckmax) {
         if (area.parent().innerHeight() - 10 < area.innerHeight())
            area.css('bottom', '0px').css('top','0px');
         return;
      }

      if (!chld.length) return;

      var h1 = cont.innerHeight(),
          h2 = chld.innerHeight();

      if ((h2!==undefined) && (h2<h1*0.7)) area.css('bottom', '');
   }

   BrowserLayout.prototype.ToggleBrowserVisisbility = function(fast_close) {
      if (!this.gui_div || (typeof this.browser_visible==='string')) return;

      var main = d3.select("#" + this.gui_div + " .jsroot_browser"),
          area = main.select('.jsroot_browser_area');

      if (area.empty()) return;

      var vsepar = main.select(".jsroot_v_separator"),
          drawing = d3.select("#" + this.gui_div + "_drawing"),
          tgt = area.property('last_left'),
          tgt_separ = area.property('last_vsepar'),
          tgt_drawing = area.property('last_drawing');

      if (!this.browser_visible) {
         if (fast_close) return;
         area.property('last_left', null).property('last_vsepar',null).property('last_drawing', null);
      } else {
         area.property('last_left', area.style('left'));
         if (!vsepar.empty()) {
            area.property('last_vsepar', vsepar.style('left'));
            area.property('last_drawing', drawing.style('left'));
         }
         tgt = (-$(area.node()).outerWidth(true)-10).toString() + "px";
         var mainw = $(main.node()).outerWidth(true);

         if (vsepar.empty() && ($(area.node()).offset().left > mainw/2)) tgt = (mainw+10) + "px";

         tgt_separ = "-10px";
         tgt_drawing = "0px";
      }

      var pthis = this, visible_at_the_end  = !this.browser_visible, _duration = fast_close ? 0 : 700;

      this.browser_visible = 'changing';

      area.transition().style('left', tgt).duration(_duration).on("end", function() {
         if (fast_close) return;
         pthis.browser_visible = visible_at_the_end;
         if (visible_at_the_end) pthis.SetButtonsPosition();
      });

      if (!visible_at_the_end)
         main.select(".jsroot_browser_btns").transition().style('left', '7px').style('top', '7px').duration(_duration);

      if (!vsepar.empty()) {
         vsepar.transition().style('left', tgt_separ).duration(_duration);
         drawing.transition().style('left', tgt_drawing).duration(_duration).on("end", this.CheckResize.bind(this));
      }

      if (this.status_layout && (this.browser_kind == 'fix')) {
         main.select(".jsroot_h_separator").transition().style('left', tgt_drawing).duration(_duration);
         main.select(".jsroot_status_area").transition().style('left', tgt_drawing).duration(_duration);
      }
   }

   /// used together with browser buttons
   BrowserLayout.prototype.Toggle = function(browser_kind) {
      if (this.browser_visible!=='changing') {
         if (browser_kind === this.browser_kind) this.ToggleBrowserVisisbility();
                                            else this.ToggleBrowserKind(browser_kind);
      }
   }

   BrowserLayout.prototype.DeleteContent = function() {
      var main = d3.select("#" + this.gui_div + " .jsroot_browser");
      if (main.empty()) return;

      this.CreateStatusLine("delete");
      var vsepar = main.select(".jsroot_v_separator");
      if (!vsepar.empty())
         $(vsepar.node()).draggable('destroy');

      this.ToggleBrowserVisisbility(true);

      main.selectAll("*").remove();
      delete this.browser_visible;
      delete this.browser_kind;

      this.CheckResize();
   }

   /// method creates status line
   BrowserLayout.prototype.CreateStatusLine = function(height, mode) {
      var main = d3.select("#"+this.gui_div+" .jsroot_browser");
      if (main.empty()) return '';

      var id = this.gui_div + "_status",
          line = d3.select("#"+id), skip_height_check = false,
          is_visible = !line.empty();

      if (mode==="toggle") { mode = !is_visible; skip_height_check = (height === this.last_hsepar_height); } else
      if (height==="delete") { mode = false; height = 0; delete this.status_layout; } else
      if (mode===undefined) { mode = true; this.status_layout = "app"; }

      if (is_visible) {
         if ((mode === true) || (this.status_layout==="app")) return id;

         var hsepar = main.select(".jsroot_h_separator");

         $(hsepar.node()).draggable("destroy");

         hsepar.remove();
         line.remove();

         delete this.status_layout;

         if (this.status_handler && (JSROOT.Painter.ShowStatus === this.status_handler)) {
            delete JSROOT.Painter.ShowStatus;
            delete this.status_handler;
         }

         this.AdjustSeparator(null, 0, true);
         return "";
      }

      if (mode === false) return "";

      var left_pos = d3.select("#" + this.gui_div + "_drawing").style('left');

      line = main.insert("div",".jsroot_browser_area").attr("id",id)
                 .classed("jsroot_status_area", true)
                 .style('position',"absolute").style('left',left_pos).style('height',"20px").style('bottom',0).style('right',0)
                 .style('margin',0).style('border',0);

      var hsepar = main.insert("div",".jsroot_browser_area")
                       .classed("jsroot_separator", true).classed("jsroot_h_separator", true)
                      .style('position','absolute').style('left',left_pos).style('right',0).style('bottom','20px').style('height','5px');

      var pthis = this;

      $(hsepar.node()).draggable({
         axis: "y" , cursor: "ns-resize", containment: "parent",
         helper: function() { return $(this).clone().css('background-color','grey'); },
         drag: function(event,ui) {
            pthis.AdjustSeparator(null, -ui.position.top);
         },
         stop: function(event,ui) {
            pthis.CheckResize();
         }
      });

      if (!height || (typeof height === 'string')) height = this.last_hsepar_height || 20;

      this.AdjustSeparator(null, height, true);

      if (this.status_layout == "app") return id;

      this.status_layout = new JSROOT.GridDisplay(id, 'horizx4_1213');

      var frame_titles = ['object name','object title','mouse coordinates','object info'];
      for (var k=0;k<4;++k)
         d3.select(this.status_layout.GetFrame(k)).attr('title', frame_titles[k]).style('overflow','hidden')
           .append("label").attr("class","jsroot_status_label");

      this.status_handler = this.ShowStatus.bind(this);

      JSROOT.Painter.ShowStatus = this.status_handler;

      return id;
   }

   BrowserLayout.prototype.AdjustSeparator = function(vsepar, hsepar, redraw, first_time) {

      if (!this.gui_div) return;

      var main = d3.select("#" + this.gui_div + " .jsroot_browser"), w = 5;

      if ((hsepar===null) && first_time && !main.select(".jsroot_h_separator").empty()) {
         // if separator set for the first time, check if status line present
         hsepar = main.select(".jsroot_h_separator").style('bottom');
         if ((typeof hsepar=='string') && (hsepar.indexOf('px')==hsepar.length-2))
            hsepar = hsepar.substr(0,hsepar.length-2);
         else
            hsepar = null;
      }

      if (hsepar!==null) {
         hsepar = parseInt(hsepar);
         var elem = main.select(".jsroot_h_separator"), hlimit = 0;

         if (!elem.empty()) {
            if (hsepar<0) hsepar += ($(main.node()).outerHeight(true) - w);
            if (hsepar<5) hsepar = 5;
            this.last_hsepar_height = hsepar;
            elem.style('bottom', hsepar+'px').style('height', w+'px');
            d3.select("#" + this.gui_div + "_status").style('height', hsepar+'px');
            hlimit = (hsepar+w) + 'px';
         }

         d3.select("#" + this.gui_div + "_drawing").style('bottom',hlimit);
      }

      if (vsepar!==null) {
         vsepar = parseInt(vsepar);
         if (vsepar<50) vsepar = 50;
         main.select(".jsroot_browser_area").style('width',(vsepar-5)+'px');
         d3.select("#" + this.gui_div + "_drawing").style('left',(vsepar+w)+'px');
         main.select(".jsroot_h_separator").style('left', (vsepar+w)+'px');
         d3.select("#" + this.gui_div + "_status").style('left',(vsepar+w)+'px');
         main.select(".jsroot_v_separator").style('left',vsepar+'px').style('width',w+"px");
      }

      if (redraw) this.CheckResize();
   }

   BrowserLayout.prototype.ShowStatus = function(name, title, info, coordinates) {
      if (!this.status_layout) return;

      $(this.status_layout.GetFrame(0)).children('label').text(name || "");
      $(this.status_layout.GetFrame(1)).children('label').text(title || "");
      $(this.status_layout.GetFrame(2)).children('label').text(coordinates || "");
      $(this.status_layout.GetFrame(3)).children('label').text(info || "");

      if (!this.status_layout.first_check) {
         this.status_layout.first_check = true;
         var maxh = 0;
         for (var n=0;n<4;++n)
            maxh = Math.max(maxh, $(this.status_layout.GetFrame(n)).children('label').outerHeight());
         if ((maxh>5) && ((maxh>this.last_hsepar_height) || (maxh<this.last_hsepar_height+5)))
            this.AdjustSeparator(null, maxh, true);
      }
   }

   // =================================================================================================

   var HierarchyPainter = JSROOT.HierarchyPainter;

   HierarchyPainter.prototype.isLastSibling = function(hitem) {
      if (!hitem || !hitem._parent || !hitem._parent._childs) return false;
      var chlds = hitem._parent._childs, indx = chlds.indexOf(hitem);
      if (indx<0) return false;
      while (++indx < chlds.length)
         if (!('_hidden' in chlds[indx])) return false;
      return true;
   }

   HierarchyPainter.prototype.addItemHtml = function(hitem, d3prnt, arg) {

      if (!hitem || ('_hidden' in hitem)) return true;

      var isroot = (hitem === this.h),
          has_childs = ('_childs' in hitem),
          handle = JSROOT.getDrawHandle(hitem._kind),
          img1 = "", img2 = "", can_click = false, break_list = false,
          d3cont, itemname = this.itemFullName(hitem);

      if (handle !== null) {
         if ('icon' in handle) img1 = handle.icon;
         if ('icon2' in handle) img2 = handle.icon2;
         if ((img1.length==0) && (typeof handle.icon_get == 'function'))
            img1 = handle.icon_get(hitem, this);
         if (('func' in handle) || ('execute' in handle) || ('aslink' in handle) ||
             (('expand' in handle) && (hitem._more !== false))) can_click = true;
      }

      if ('_icon' in hitem) img1 = hitem._icon;
      if ('_icon2' in hitem) img2 = hitem._icon2;
      if ((img1.length==0) && ('_online' in hitem))
         hitem._icon = img1 = "img_globe";
      if ((img1.length==0) && isroot)
         hitem._icon = img1 = "img_base";

      if (hitem._more || ('_expand' in hitem) || ('_player' in hitem))
         can_click = true;

      var can_menu = can_click;
      if (!can_menu && (typeof hitem._kind == 'string') && (hitem._kind.indexOf("ROOT.")==0))
         can_menu = can_click = true;

      if (img2.length==0) img2 = img1;
      if (img1.length==0) img1 = (has_childs || hitem._more) ? "img_folder" : "img_page";
      if (img2.length==0) img2 = (has_childs || hitem._more) ? "img_folderopen" : "img_page";

      if (arg === "update") {
         d3prnt.selectAll("*").remove();
         d3cont = d3prnt;
      } else {
         d3cont = d3prnt.append("div");
         if (arg && (arg >= (hitem._parent._show_limit || JSROOT.gStyle.HierarchyLimit))) break_list = true;
      }

      hitem._d3cont = d3cont.node(); // set for direct referencing
      d3cont.attr("item", itemname);

      // line with all html elements for this item (excluding childs)
      var d3line = d3cont.append("div").attr('class','h_line');

      // build indent
      var prnt = isroot ? null : hitem._parent;
      while (prnt && (prnt !== this.h)) {
         d3line.insert("div",":first-child")
               .attr("class", this.isLastSibling(prnt) ? "img_empty" : "img_line");
         prnt = prnt._parent;
      }

      var icon_class = "", plusminus = false;

      if (isroot) {
         // for root node no extra code
      } else
      if (has_childs && !break_list) {
         icon_class = hitem._isopen ? "img_minus" : "img_plus";
         plusminus = true;
      } else
      /*if (hitem._more) {
         icon_class = "img_plus"; // should be special plus ???
         plusminus = true;
      } else */ {
         icon_class = "img_join";
      }

      var h = this;

      if (icon_class.length > 0) {
         if (break_list || this.isLastSibling(hitem)) icon_class += "bottom";
         var d3icon = d3line.append("div").attr('class', icon_class);
         if (plusminus) d3icon.style('cursor','pointer')
                              .on("click", function() { h.tree_click(this, "plusminus"); });
      }

      // make node icons

      if (this.with_icons && !break_list) {
         var icon_name = hitem._isopen ? img2 : img1;

         var d3img;

         if (icon_name.indexOf("img_")==0)
            d3img = d3line.append("div")
                          .attr("class", icon_name)
                          .attr("title", hitem._kind);
         else
            d3img = d3line.append("img")
                          .attr("src", icon_name)
                          .attr("alt","")
                          .attr("title", hitem._kind)
                          .style('vertical-align','top').style('width','18px').style('height','18px');

         if (('_icon_click' in hitem) || (handle && ('icon_click' in handle)))
            d3img.on("click", function() { h.tree_click(this, "icon"); });
      }

      var d3a = d3line.append("a");
      if (can_click || has_childs || break_list)
         d3a.attr("class","h_item")
            .on("click", function() { h.tree_click(this); });

      if (break_list) {
         hitem._break_point = true; // indicate that list was broken here
         d3a.attr('title', 'there are ' + (hitem._parent._childs.length-arg) + ' more items')
            .text("...more...");
         return false;
      }

      if ('disp_kind' in h) {
         if (JSROOT.gStyle.DragAndDrop && can_click)
           this.enable_dragging(d3a.node(), itemname);
         if (JSROOT.gStyle.ContextMenu && can_menu)
            d3a.on('contextmenu', function() { h.tree_contextmenu(this); });

         d3a.on("mouseover", function() { h.tree_mouseover(true, this); })
            .on("mouseleave", function() { h.tree_mouseover(false, this); });
      } else
      if (hitem._direct_context && JSROOT.gStyle.ContextMenu)
         d3a.on('contextmenu', function() { h.direct_contextmenu(this); });

      var element_name = hitem._name, element_title = "";

      if ('_realname' in hitem)
         element_name = hitem._realname;

      if ('_title' in hitem)
         element_title = hitem._title;

      if ('_fullname' in hitem)
         element_title += "  fullname: " + hitem._fullname;

      if (!element_title)
         element_title = element_name;

      d3a.attr('title', element_title)
         .text(element_name + ('_value' in hitem ? ":" : ""))
         .style('background', hitem._background ? hitem._background : null);

      if ('_value' in hitem) {
         var d3p = d3line.append("p");
         if ('_vclass' in hitem) d3p.attr('class', hitem._vclass);
         if (!hitem._isopen) d3p.html(hitem._value);
      }

      if (has_childs && (isroot || hitem._isopen)) {
         var d3chlds = d3cont.append("div").attr("class", "h_childs");
         for (var i=0; i< hitem._childs.length; ++i) {
            var chld = hitem._childs[i];
            chld._parent = hitem;
            if (!this.addItemHtml(chld, d3chlds, i)) break; // if too many items, skip rest
         }
      }

      return true;
   }

   HierarchyPainter.prototype.toggleOpenState = function(isopen, h) {
      var hitem = h ? h : this.h;

      if (!('_childs' in hitem)) {
         if (!isopen || this.with_icons || (!hitem._expand && (hitem._more !== true))) return false;
         this.expand(this.itemFullName(hitem));
         if (hitem._childs) hitem._isopen = true;
         return true;
      }

      if ((hitem != this.h) && isopen && !hitem._isopen) {
         // when there are childs and they are not see, simply show them
         hitem._isopen = true;
         return true;
      }

      var change_child = false;
      for (var i=0; i < hitem._childs.length; ++i)
         if (this.toggleOpenState(isopen, hitem._childs[i])) change_child = true;

      if ((hitem != this.h) && !isopen && hitem._isopen && !change_child) {
         // if none of the childs can be closed, than just close that item
         delete hitem._isopen;
         return true;
       }

      if (!h) this.RefreshHtml();

      return false;
   }

   HierarchyPainter.prototype.RefreshHtml = function(callback) {

      if (!this.divid) return JSROOT.CallBack(callback);

      var d3elem = this.select_main();

      d3elem.html("")
            .style('overflow','hidden') // clear html - most simple way
            .style('display','flex')
            .style('flex-direction','column');

      var h = this, factcmds = [], status_item = null;
      this.ForEach(function(item) {
         delete item._d3cont; // remove html container
         if (('_fastcmd' in item) && (item._kind == 'Command')) factcmds.push(item);
         if (('_status' in item) && !status_item) status_item = item;
      });

      if (!this.h || d3elem.empty())
         return JSROOT.CallBack(callback);

      if (factcmds.length) {
         var fastbtns = d3elem.append("div").attr("class","jsroot");
         for (var n=0;n<factcmds.length;++n) {
            var btn = fastbtns.append("button")
                       .text("")
                       .attr("class",'fast_command')
                       .attr("item", this.itemFullName(factcmds[n]))
                       .attr("title", factcmds[n]._title)
                       .on("click", function() { h.ExecuteCommand(d3.select(this).attr("item"), this); } );

            if ('_icon' in factcmds[n])
               btn.append('img').attr("src", factcmds[n]._icon);
         }
      }

      var d3btns = d3elem.append("p").attr("class", "jsroot").style("margin-bottom","3px").style("margin-top",0);
      d3btns.append("a").attr("class", "h_button").text("open all")
            .attr("title","open all items in the browser").on("click", h.toggleOpenState.bind(h,true));
      d3btns.append("text").text(" | ");
      d3btns.append("a").attr("class", "h_button").text("close all")
            .attr("title","close all items in the browser").on("click", h.toggleOpenState.bind(h,false));

      if ('_online' in this.h) {
         d3btns.append("text").text(" | ");
         d3btns.append("a").attr("class", "h_button").text("reload")
               .attr("title","reload object list from the server").on("click", h.reload.bind(h));
      }

      if ('disp_kind' in this) {
         d3btns.append("text").text(" | ");
         d3btns.append("a").attr("class", "h_button").text("clear")
               .attr("title","clear all drawn objects").on("click", h.clear.bind(h,false));
      }

      var maindiv =
         d3elem.append("div")
               .attr("class", "jsroot")
               .style('font-size', this.with_icons ? "12px" : "15px")
               .style("overflow","auto")
               .style("flex","1");

      if (this.background) // case of object inspector and streamer infos display
         maindiv.style("background-color", this.background)
                .style('margin', '2px').style('padding', '2px');

      this.addItemHtml(this.h, maindiv.append("div").attr("class","h_tree"));

      if (status_item && !this.status_disabled && (JSROOT.GetUrlOption('nostatus')===null)) {
         var func = JSROOT.findFunction(status_item._status);
         var hdiv = (typeof func == 'function') ? this.CreateStatusLine() : null;
         if (hdiv) func(hdiv, this.itemFullName(status_item));
      }

      JSROOT.CallBack(callback);
   }

   HierarchyPainter.prototype.UpdateTreeNode = function(hitem, d3cont) {
      if ((d3cont===undefined) || d3cont.empty())  {
         d3cont = d3.select(hitem._d3cont ? hitem._d3cont : null);
         var name = this.itemFullName(hitem);
         if (d3cont.empty())
            d3cont = this.select_main().select("[item='" + name + "']");
         if (d3cont.empty() && ('_cycle' in hitem))
            d3cont = this.select_main().select("[item='" + name + ";" + hitem._cycle + "']");
         if (d3cont.empty()) return;
      }

      this.addItemHtml(hitem, d3cont, "update");

      if (this.brlayout) this.brlayout.AdjustBrowserSize(true);
   }

   HierarchyPainter.prototype.UpdateBackground = function(hitem, scroll_into_view) {

      if (!hitem || !hitem._d3cont) return;

      var d3cont = d3.select(hitem._d3cont);

      if (d3cont.empty()) return;

      var d3a = d3cont.select(".h_item");

      d3a.style('background', hitem._background ? hitem._background : null);

      if (scroll_into_view && hitem._background)
         d3a.node().scrollIntoView(false);
   }

   HierarchyPainter.prototype.tree_click = function(node, place) {
      if (!node) return;

      var d3cont = d3.select(node.parentNode.parentNode);
      var itemname = d3cont.attr('item');
      if (!itemname) return;

      var hitem = this.Find(itemname);
      if (!hitem) return;

      if (hitem._break_point) {
         // special case of more item

         delete hitem._break_point;

         // update item itself
         this.addItemHtml(hitem, d3cont, "update");

         var prnt = hitem._parent, indx = prnt._childs.indexOf(hitem),
             d3chlds = d3.select(d3cont.node().parentNode);

         if (indx<0) return console.error('internal error');

         prnt._show_limit = (prnt._show_limit || JSROOT.gStyle.HierarchyLimit) * 2;

         for (var n=indx+1;n<prnt._childs.length;++n) {
            var chld = prnt._childs[n];
            chld._parent = prnt;
            if (!this.addItemHtml(chld, d3chlds, n)) break; // if too many items, skip rest
         }

         return;
      }

      var prnt = hitem, dflt = undefined;
      while (prnt) {
         if ((dflt = prnt._click_action) !== undefined) break;
         prnt = prnt._parent;
      }

      if (!place || (place=="")) place = "item";

      var sett = JSROOT.getDrawSettings(hitem._kind), handle = sett.handle;

      if (place == "icon") {
         var func = null;
         if (typeof hitem._icon_click == 'function') func = hitem._icon_click; else
         if (handle && typeof handle.icon_click == 'function') func = handle.icon_click;
         if (func && func(hitem,this))
            this.UpdateTreeNode(hitem, d3cont);
         return;
      }

      // special feature - all items with '_expand' function are not drawn by click
      if ((place=="item") && ('_expand' in hitem) && !d3.event.ctrlKey && !d3.event.shiftKey) place = "plusminus";

      // special case - one should expand item
      if (((place == "plusminus") && !('_childs' in hitem) && hitem._more) ||
          ((place == "item") && (dflt === "expand"))) {
         return this.expand(itemname, null, d3cont);
      }

      if (place == "item") {
         if ('_player' in hitem)
            return this.player(itemname);

         if (handle && handle.aslink)
            return window.open(itemname + "/");

         if (handle && handle.execute)
            return this.ExecuteCommand(itemname, node.parentNode);

         if (handle && handle.ignore_online && this.isOnlineItem(hitem)) return;

         var can_draw = hitem._can_draw,
             can_expand = hitem._more,
             dflt_expand = (this.default_by_click === "expand"),
             drawopt = "";

         if (d3.event.shiftKey) {
            drawopt = (handle && handle.shift) ? handle.shift : "inspect";
            if ((drawopt==="inspect") && handle && handle.noinspect) drawopt = "";
         }
         if (handle && handle.ctrl && d3.event.ctrlKey) drawopt = handle.ctrl;

         if (!drawopt) {
            for (var pitem = hitem._parent; pitem; pitem = pitem._parent) {
               if (pitem._painter) { can_draw = false; if (can_expand===undefined) can_expand = false; break; }
            }
         }

         if (hitem._childs) can_expand = false;

         if (can_draw === undefined) can_draw = sett.draw;
         if (can_expand === undefined) can_expand = sett.expand;

         if (can_draw && can_expand && !drawopt) {
            // if default action specified as expand, disable drawing
            if (dflt_expand || (handle && (handle.dflt === 'expand'))) can_draw = false; else
            if (this.isItemDisplayed(itemname)) can_draw = false; // if already displayed, try to expand
         }

         if (can_draw)
            return this.display(itemname, drawopt);

         if (can_expand || dflt_expand)
            return this.expand(itemname, null, d3cont);

         // cannot draw, but can inspect ROOT objects
         if ((typeof hitem._kind === "string") && (hitem._kind.indexOf("ROOT.")===0) && sett.inspect && (can_draw!==false))
            return this.display(itemname, "inspect");

         if (!hitem._childs || (hitem === this.h)) return;
      }

      if (hitem._isopen)
         delete hitem._isopen;
      else
         hitem._isopen = true;

      this.UpdateTreeNode(hitem, d3cont);
   }

   HierarchyPainter.prototype.tree_mouseover = function(on, elem) {
      var itemname = d3.select(elem.parentNode.parentNode).attr('item');

      var hitem = this.Find(itemname);
      if (!hitem) return;

      var painter, prnt = hitem;
      while (prnt && !painter) {
         painter = prnt._painter;
         prnt = prnt._parent;
      }

      if (painter && typeof painter.MouseOverHierarchy === 'function')
         painter.MouseOverHierarchy(on, itemname, hitem);
   }

   HierarchyPainter.prototype.direct_contextmenu = function(elem) {
      // this is alternative context menu, used in the object inspector

      d3.event.preventDefault();
      var itemname = d3.select(elem.parentNode.parentNode).attr('item');
      var hitem = this.Find(itemname);
      if (!hitem) return;

      if (typeof this.fill_context !== 'function') return;

      JSROOT.Painter.createMenu(this, function(menu) {

         menu.painter.fill_context(menu, hitem);

         if (menu.size() > 0) {
            menu.tree_node = elem.parentNode;
            menu.show(d3.event);
         }
      });
   }

   HierarchyPainter.prototype.tree_contextmenu = function(elem) {
      // this is handling of context menu request for the normal objects browser

      d3.event.preventDefault();

      var itemname = d3.select(elem.parentNode.parentNode).attr('item');

      var hitem = this.Find(itemname);
      if (!hitem) return;

      var painter = this,
          onlineprop = painter.GetOnlineProp(itemname),
          fileprop = painter.GetFileProp(itemname);

      function qualifyURL(url) {
         function escapeHTML(s) {
            return s.split('&').join('&amp;').split('<').join('&lt;').split('"').join('&quot;');
         }
         var el = document.createElement('div');
         el.innerHTML = '<a href="' + escapeHTML(url) + '">x</a>';
         return el.firstChild.href;
      }

      JSROOT.Painter.createMenu(painter, function(menu) {

         if ((itemname == "") && !('_jsonfile' in hitem)) {
            var addr = "", cnt = 0;
            function separ() { return cnt++ > 0 ? "&" : "?"; }

            var files = [];
            painter.ForEachRootFile(function(item) { files.push(item._file.fFullURL); });

            if (!painter.GetTopOnlineItem())
               addr = JSROOT.source_dir + "index.htm";

            if (painter.IsMonitoring())
               addr += separ() + "monitoring=" + painter.MonitoringInterval();

            if (files.length==1)
               addr += separ() + "file=" + files[0];
            else
               if (files.length>1)
                  addr += separ() + "files=" + JSON.stringify(files);

            if (painter['disp_kind'])
               addr += separ() + "layout=" + painter.disp_kind.replace(/ /g, "");

            var items = [];

            if (painter.disp)
               painter.disp.ForEachPainter(function(p) {
                  if (p.GetItemName()!=null)
                     items.push(p.GetItemName());
               });

            if (items.length == 1) {
               addr += separ() + "item=" + items[0];
            } else if (items.length > 1) {
               addr += separ() + "items=" + JSON.stringify(items);
            }

            menu.add("Direct link", function() { window.open(addr); });
            menu.add("Only items", function() { window.open(addr + "&nobrowser"); });
         } else if (onlineprop) {
            painter.FillOnlineMenu(menu, onlineprop, itemname);
         } else {
            var sett = JSROOT.getDrawSettings(hitem._kind, 'nosame');

            // allow to draw item even if draw function is not defined
            if (hitem._can_draw) {
               if (!sett.opts) sett.opts = [""];
               if (sett.opts.indexOf("")<0) sett.opts.unshift("");
            }

            if (sett.opts)
               menu.addDrawMenu("Draw", sett.opts, function(arg) { this.display(itemname, arg); });

            if (fileprop && sett.opts && !fileprop.localfile) {
               var filepath = qualifyURL(fileprop.fileurl);
               if (filepath.indexOf(JSROOT.source_dir) == 0)
                  filepath = filepath.slice(JSROOT.source_dir.length);
               filepath = fileprop.kind + "=" + filepath;
               if (fileprop.itemname.length > 0) {
                  var name = fileprop.itemname;
                  if (name.search(/\+| |\,/)>=0) name = "\'" + name + "\'";
                  filepath += "&item=" + name;
               }

               menu.addDrawMenu("Draw in new tab", sett.opts, function(arg) {
                  window.open(JSROOT.source_dir + "index.htm?nobrowser&"+filepath +"&opt="+arg);
               });
            }

            if (sett.expand && !('_childs' in hitem) && (hitem._more || !('_more' in hitem)))
               menu.add("Expand", function() { painter.expand(itemname); });

            if (hitem._kind === "ROOT.TStyle")
               menu.add("Apply", function() { painter.ApplyStyle(itemname); });
         }

         if (typeof hitem._menu == 'function')
            hitem._menu(menu, hitem, painter);

         if (menu.size() > 0) {
            menu.tree_node = elem.parentNode;
            if (menu.separ) menu.add("separator"); // add separator at the end
            menu.add("Close");
            menu.show(d3.event);
         }

      }); // end menu creation

      return false;
   }

   /** \brief Creates configured JSROOT.MDIDisplay object
   *
   * @param callback - called when mdi object created
   */

   HierarchyPainter.prototype.CreateDisplay = function(callback) {

      if ('disp' in this) {
         if (this.disp.NumDraw() > 0) return JSROOT.CallBack(callback, this.disp);
         this.disp.Reset();
         delete this.disp;
      }

      // check that we can found frame where drawing should be done
      if (document.getElementById(this.disp_frameid) == null)
         return JSROOT.CallBack(callback, null);

      if (this.disp_kind == "tabs")
         this.disp = new TabsDisplay(this.disp_frameid);
      else
      if (this.disp_kind.indexOf("flex")==0)
         this.disp = new FlexibleDisplay(this.disp_frameid);
      else
      if (this.disp_kind.indexOf("coll")==0)
         this.disp = new CollapsibleDisplay(this.disp_frameid);
      else
         this.disp = new JSROOT.GridDisplay(this.disp_frameid, this.disp_kind);

      if (this.disp)
         this.disp.CleanupFrame = this.CleanupFrame.bind(this);

      JSROOT.CallBack(callback, this.disp);
   }

   HierarchyPainter.prototype.enable_dragging = function(element, itemname) {
      $(element).draggable({ revert: "invalid", appendTo: "body", helper: "clone" });
   }

   HierarchyPainter.prototype.enable_dropping = function(frame, itemname) {
      var h = this;
      $(frame).droppable({
         hoverClass : "ui-state-active",
         accept: function(ui) {
            var dropname = ui.parent().parent().attr('item');
            if ((dropname == itemname) || !dropname) return false;

            var ditem = h.Find(dropname);
            if (!ditem || (!('_kind' in ditem))) return false;

            return ditem._kind.indexOf("ROOT.")==0;
         },
         drop: function(event, ui) {
            var dropname = ui.draggable.parent().parent().attr('item');
            if (!dropname) return false;
            return h.dropitem(dropname, $(this).attr("id"));
         }
      });
   }

   HierarchyPainter.prototype.CreateBrowser = function(browser_kind, update_html, call_back) {

      if (!this.gui_div || this.exclude_browser || !this.brlayout) return false;

      var main = d3.select("#" + this.gui_div + " .jsroot_browser"),
          jmain = $(main.node());

      // one requires top-level container
      if (main.empty()) return false;

      if ((browser_kind==="float") && this.float_browser_disabled) browser_kind = "fix";

      if (!main.select('.jsroot_browser_area').empty()) {
         // this is case when browser created,
         // if update_html specified, hidden state will be toggled

         if (update_html) this.brlayout.Toggle(browser_kind);

         JSROOT.CallBack(call_back);

         return true;
      }

      var guiCode = "<p class='jsroot_browser_version'><a href='https://root.cern/js/'>JSROOT</a> version <span style='color:green'><b>" + JSROOT.version + "</b></span></p>";

      if (this.is_online) {
         guiCode +='<p> Hierarchy in <a href="h.json">json</a> and <a href="h.xml">xml</a> format</p>'
                 + '<div style="display:flex;flex-direction:row;">'
                 + '<label style="margin-right:5px; vertical-align:middle;">'
                 + '<input style="vertical-align:middle;" type="checkbox" name="monitoring" class="gui_monitoring"/>'
                 + 'Monitoring</label>';
      } else if (!this.no_select) {
         var myDiv = d3.select("#"+this.gui_div),
             files = myDiv.attr("files") || "../files/hsimple.root",
             path = JSROOT.GetUrlOption("path") || myDiv.attr("path") || "",
             arrFiles = files.split(';');

         guiCode +=
            '<input type="text" value="" style="width:95%; margin:5px;border:2px;" class="gui_urlToLoad" title="input file name"/>'
            +'<div style="display:flex;flex-direction:row;padding-top:5px">'
            +'<select class="gui_selectFileName" style="flex:1;padding:2px;" title="select file name"'
            +'<option value="" selected="selected"></option>';
         for (var i in arrFiles)
            guiCode += '<option value = "' + path + arrFiles[i] + '">' + arrFiles[i] + '</option>';
         guiCode += '</select>'
            +'<input type="file" class="gui_localFile" accept=".root" style="display:none"/><output id="list" style="display:none"></output>'
            +'<input type="button" value="..." class="gui_fileBtn" style="min-width:3em;padding:3px;margin-left:5px;margin-right:5px;" title="select local file for reading"/><br/>'
            +'</div>'
            +'<p id="gui_fileCORS"><small><a href="https://github.com/root-project/jsroot/blob/master/docs/JSROOT.md#reading-root-files-from-other-servers">Read docu</a>'
            +' how to open files from other servers.</small></p>'
            +'<div style="display:flex;flex-direction:row">'
            +'<input style="padding:3px;margin-right:5px;"'
            +'       class="gui_ReadFileBtn" type="button" title="Read the Selected File" value="Load"/>'
            +'<input style="padding:3px;margin-right:5px;"'
            +'       class="gui_ResetUIBtn" type="button" title="Close all opened files and clear drawings" value="Reset"/>'
      } else if (this.no_select == "file") {
         guiCode += '<div style="display:flex;flex-direction:row">';
      }

      if (this.is_online || !this.no_select || this.no_select=="file")
         guiCode += '<select style="padding:2px;margin-right:5px;" title="layout kind" class="gui_layout"></select>'
                  + '</div>';

      guiCode += '<div id="' + this.gui_div+'_browser_hierarchy" class="jsroot_browser_hierarchy"></div>';

      this.brlayout.SetBrowserContent(guiCode);

      this.brlayout.SetBrowserTitle(this.is_online ? 'ROOT online server' : 'Read a ROOT file');

      var hpainter = this, localfile_read_callback = null;

      if (!this.is_online && !this.no_select) {

         this.ReadSelectedFile = function() {
            var filename = main.select(".gui_urlToLoad").property('value').trim();
            if (!filename) return;

            if ((filename.toLowerCase().lastIndexOf(".json") == filename.length-5))
               this.OpenJsonFile(filename);
            else
               this.OpenRootFile(filename);
         }

         jmain.find(".gui_selectFileName").val("").change(function() {
            jmain.find(".gui_urlToLoad").val($(this).val());
         });
         jmain.find(".gui_fileBtn").button().click(function() {
            jmain.find(".gui_localFile").click();
         });

         jmain.find(".gui_ReadFileBtn").button().click(function(){
            hpainter.ReadSelectedFile();
         });

         jmain.find(".gui_ResetUIBtn").button().click(function(){
            hpainter.clear(true);
         });

         jmain.find(".gui_urlToLoad").keyup(function(e) {
            if (e.keyCode == 13) hpainter.ReadSelectedFile();
         });

         jmain.find(".gui_localFile").change(function(evnt) {
            var files = evnt.target.files;

            for (var n=0;n<files.length;++n) {
               var f = files[n];
               main.select(".gui_urlToLoad").property('value', f.name);
               if (hpainter) hpainter.OpenRootFile(f, localfile_read_callback);
            }

            localfile_read_callback = null;
         });

         this.SelectLocalFile = function(read_callback) {
            localfile_read_callback = read_callback;
            $("#" + this.gui_div + " .jsroot_browser").find(".gui_localFile").click();
         }
      }

      var jlayout = jmain.find(".gui_layout");
      if (jlayout.length) {
         var lst = ['simple', 'vert2', 'vert3', 'vert231', 'horiz2', 'horiz32', 'flex',
                     'grid 2x2', 'grid 1x3', 'grid 2x3', 'grid 3x3', 'grid 4x4', 'collapsible',  'tabs'];

         for (var k=0;k<lst.length;++k){
            var opt = document.createElement('option');
            opt.value = lst[k];
            opt.innerHTML = lst[k];
            jlayout.get(0).appendChild(opt);
        }

         jlayout.change(function() {
            hpainter.SetDisplay($(this).val() || 'collapsible', hpainter.gui_div + "_drawing");
         });
      }

      this.SetDivId(this.gui_div + '_browser_hierarchy');

      if (update_html) {
         this.RefreshHtml();
         this.InitializeBrowser();
      }

      this.brlayout.ToggleBrowserKind(browser_kind || "fix");

      JSROOT.CallBack(call_back);

      return true;
   }

   HierarchyPainter.prototype.InitializeBrowser = function() {

      var main = d3.select("#" + this.gui_div + " .jsroot_browser");
      if (main.empty() || !this.brlayout) return;
      var jmain = $(main.node()), hpainter = this;

      if (this.brlayout) this.brlayout.AdjustBrowserSize();

      var selects = main.select(".gui_layout").node();

      if (selects) {
         var found = false;
         for (var i in selects.options) {
            var s = selects.options[i].text;
            if (typeof s !== 'string') continue;
            if ((s == this.GetLayout()) || (s.replace(/ /g,"") == this.GetLayout())) {
               selects.selectedIndex = i; found = true;
               break;
            }
         }
         if (!found) {
            var opt = document.createElement('option');
            opt.innerHTML = opt.value = this.GetLayout();
            selects.appendChild(opt);
            selects.selectedIndex = selects.options.length-1;
         }
      }

      if (this.is_online) {
         if (this.h && this.h._toptitle)
            this.brlayout.SetBrowserTitle(this.h._toptitle);
         jmain.find(".gui_monitoring")
           .prop('checked', this.IsMonitoring())
           .click(function() {
               hpainter.EnableMonitoring(this.checked);
               hpainter.updateAll(!this.checked);
            });
      } else if (!this.no_select) {
         var fname = "";
         this.ForEachRootFile(function(item) { if (!fname) fname = item._fullurl; });
         jmain.find(".gui_urlToLoad").val(fname);
      }
   }

   HierarchyPainter.prototype.EnableMonitoring = function(on) {
      this.SetMonitoring(undefined, on);

      var chkbox = d3.select("#" + this.gui_div + " .jsroot_browser .gui_monitoring");
      if (!chkbox.empty() && (chkbox.property('checked') !== on))
         chkbox.property('checked', on);
   }

   HierarchyPainter.prototype.CreateStatusLine = function(height, mode) {
      if (this.status_disabled || !this.gui_div || !this.brlayout) return '';
      return this.brlayout.CreateStatusLine(height, mode);
   }

   JSROOT.BuildGUI = function() {
      var myDiv = d3.select('#simpleGUI'), online = false;

      if (myDiv.empty()) {
         myDiv = d3.select('#onlineGUI');
         if (myDiv.empty()) return alert('no div for gui found');
         online = true;
      }

      if (myDiv.attr("ignoreurl") === "true")
         JSROOT.gStyle.IgnoreUrlOptions = true;

      if ((JSROOT.GetUrlOption("nobrowser")!==null) || (myDiv.attr("nobrowser") && myDiv.attr("nobrowser")!=="false"))
         return JSROOT.BuildNobrowserGUI();

      JSROOT.Painter.readStyleFromURL();

      var hpainter = new JSROOT.HierarchyPainter('root', null);

      hpainter.is_online = online;

      hpainter.StartGUI(myDiv, hpainter.InitializeBrowser.bind(hpainter));
   }

   // ==================================================

   function CollapsibleDisplay(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0; // use to count newly created frames
   }

   CollapsibleDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   CollapsibleDisplay.prototype.ForEachFrame = function(userfunc,  only_visible) {
      var topid = this.frameid + '_collapsible';

      if (document.getElementById(topid) == null) return;

      if (typeof userfunc != 'function') return;

      $('#' + topid + ' .collapsible_draw').each(function() {

         // check if only visible specified
         if (only_visible && $(this).is(":hidden")) return;

         userfunc($(this).get(0));
      });
   }

   CollapsibleDisplay.prototype.GetActiveFrame = function() {
      var found = JSROOT.MDIDisplay.prototype.GetActiveFrame.call(this);
      if (found && !$(found).is(":hidden")) return found;

      found = null;
      this.ForEachFrame(function(frame) {
         if (!found) found = frame;
      }, true);

      return found;
   }

   CollapsibleDisplay.prototype.ActivateFrame = function(frame) {
      if ($(frame).is(":hidden")) {
         $(frame).prev().toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
                 .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
                 .next().toggleClass("ui-accordion-content-active").slideDown(0);
      }
      $(frame).prev()[0].scrollIntoView();
      // remember title
      this.active_frame_title = d3.select(frame).attr('frame_title');
   }

   CollapsibleDisplay.prototype.CreateFrame = function(title) {

      this.BeforeCreateFrame(title);

      var topid = this.frameid + '_collapsible';

      if (document.getElementById(topid) == null)
         $("#"+this.frameid).append('<div id="'+ topid  + '" class="jsroot ui-accordion ui-accordion-icons ui-widget ui-helper-reset" style="overflow:auto; overflow-y:scroll; height:100%; padding-left: 2px; padding-right: 2px"></div>');

      var mdi = this,
          hid = topid + "_sub" + this.cnt++,
          uid = hid + "h",
          entryInfo = "<h5 id=\"" + uid + "\">" +
                        "<span class='ui-icon ui-icon-triangle-1-e'></span>" +
                        "<a> " + title + "</a>&nbsp; " +
                        "<button type='button' class='jsroot_collaps_closebtn' style='float:right; width:1.4em' title='close canvas'/>" +
                        " </h5>\n" +
                        "<div class='collapsible_draw' id='" + hid + "'></div>\n";

      $("#" + topid).append(entryInfo);

      $('#' + uid)
            .addClass("ui-accordion-header ui-helper-reset ui-state-default ui-corner-top ui-corner-bottom")
            .hover(function() { $(this).toggleClass("ui-state-hover"); })
            .click( function() {
                     $(this).toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
                           .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s")
                           .end().next().toggleClass("ui-accordion-content-active").slideToggle(0);
                     var sub = $(this).next(), hide_drawing = sub.is(":hidden");
                     sub.css('display', hide_drawing ? 'none' : '');
                     if (!hide_drawing) JSROOT.resize(sub.get(0));
                  })
            .next()
            .addClass("ui-accordion-content ui-helper-reset ui-widget-content ui-corner-bottom")
            .hide();

      $('#' + uid).find(" .jsroot_collaps_closebtn")
           .button({ icons: { primary: "ui-icon-close" }, text: false })
           .click(function(){
              mdi.CleanupFrame($(this).parent().next().attr('id'));
              $(this).parent().next().remove(); // remove drawing
              $(this).parent().remove();  // remove header
           });

      $('#' + uid)
            .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
            .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end().next()
            .toggleClass("ui-accordion-content-active").slideToggle(0);

      return $("#" + hid).attr('frame_title', title).css('overflow','hidden')
                         .attr('can_resize','height') // inform JSROOT that it can resize height of the
                         .css('position','relative') // this required for correct positioning of 3D canvas in WebKit
                         .get(0);
    }

   // ================================================

   function TabsDisplay(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0;
   }

   TabsDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   TabsDisplay.prototype.ForEachFrame = function(userfunc, only_visible) {
      var topid = this.frameid + '_tabs';

      if (document.getElementById(topid) == null) return;

      if (typeof userfunc != 'function') return;

      var cnt = -1;
      var active = $('#' + topid).tabs("option", "active");

      $('#' + topid + '> .tabs_draw').each(function() {
         cnt++;
         if (!only_visible || (cnt == active))
            userfunc($(this).get(0));
      });
   }

   TabsDisplay.prototype.GetActiveFrame = function() {
      var found = null;
      this.ForEachFrame(function(frame) {
         if (!found) found = frame;
      }, true);

      return found;
   }

   TabsDisplay.prototype.ActivateFrame = function(frame) {
      var cnt = 0, id = -1;
      this.ForEachFrame(function(fr) {
         if ($(fr).attr('id') == $(frame).attr('id')) id = cnt;
         cnt++;
      });
      $('#' + this.frameid + "_tabs").tabs("option", "active", id);

      this.active_frame_title = d3.select(frame).attr('frame_title');
   }

   TabsDisplay.prototype.CreateFrame = function(title) {

      this.BeforeCreateFrame(title);

      var mdi = this,
          topid = this.frameid + '_tabs',
          hid = topid + "_sub" + this.cnt++,
          li = '<li><a href="#' + hid + '">' + title
                 + '</a><span class="ui-icon ui-icon-close" style="float: left; margin: 0.4em 0.2em 0 0; cursor: pointer;" role="presentation">Remove Tab</span></li>',
         cont = '<div class="tabs_draw" id="' + hid + '"></div>';

      if (document.getElementById(topid) == null) {
         $("#" + this.frameid).append('<div id="' + topid + '" class="jsroot">' + ' <ul>' + li + ' </ul>' + cont + '</div>');

         var tabs = $("#" + topid)
                       .css('overflow','hidden')
                       .tabs({
                          heightStyle : "fill",
                          activate : function (event,ui) {
                             $(ui.newPanel).css('overflow', 'hidden');
                             JSROOT.resize($(ui.newPanel).get(0));
                           }
                        });

         tabs.delegate("span.ui-icon-close", "click", function() {
            var panelId = $(this).closest("li").remove().attr("aria-controls");
            mdi.CleanupFrame(panelId);
            $("#" + panelId).remove();
            tabs.tabs("refresh");
            if ($('#' + topid + '> .tabs_draw').length == 0)
               $("#" + topid).remove();

         });
      } else {
         $("#" + topid).find("> .ui-tabs-nav").append(li);
         $("#" + topid).append(cont);
         $("#" + topid).tabs("refresh");
         $("#" + topid).tabs("option", "active", -1);
      }
      $('#' + hid)
         .empty()
         .css('overflow', 'hidden')
         .attr('frame_title', title);

      return $('#' + hid).get(0);
   }

   TabsDisplay.prototype.CheckMDIResize = function(frame_id, size) {
      $("#" + this.frameid + '_tabs').tabs("refresh");
      JSROOT.MDIDisplay.prototype.CheckMDIResize.call(this, frame_id, size);
   }

   // ==================================================

   function FlexibleDisplay(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0; // use to count newly created frames
   }

   FlexibleDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   FlexibleDisplay.prototype.ForEachFrame = function(userfunc,  only_visible) {
      var topid = this.frameid + '_flex';

      if (document.getElementById(topid) == null) return;
      if (typeof userfunc != 'function') return;

      $('#' + topid + ' .flex_draw').each(function() {
         // check if only visible specified
         if (only_visible && $(this).is(":hidden")) return;

         userfunc($(this).get(0));
      });
   }

   FlexibleDisplay.prototype.GetActiveFrame = function() {
      var found = JSROOT.MDIDisplay.prototype.GetActiveFrame.call(this);
      if (found && !$(found).is(":hidden")) return found;

      found = null;
      this.ForEachFrame(function(frame) {
         if (!found) found = frame;
      }, true);

      return found;
   }

   FlexibleDisplay.prototype.ActivateFrame = function(frame) {
      this.active_frame_title = d3.select(frame).attr('frame_title');
   }

   FlexibleDisplay.prototype.CreateFrame = function(title) {

      this.BeforeCreateFrame(title);

      var topid = this.frameid + '_flex';

      if (document.getElementById(topid) == null)
         $("#" + this.frameid).append('<div id="'+ topid  + '" class="jsroot" style="overflow:none; height:100%; width:100%"></div>');

      var mdi = this,
          top = $("#" + topid),
          w = top.width(),
          h = top.height(),
          subid = topid + "_frame" + this.cnt;

      var entry ='<div id="' + subid + '" class="flex_frame" style="position:absolute">' +
                  '<div class="ui-widget-header flex_header">'+
                    '<p>'+title+'</p>' +
                    '<button type="button" style="float:right; width:1.4em"/>' +
                    '<button type="button" style="float:right; width:1.4em"/>' +
                    '<button type="button" style="float:right; width:1.4em"/>' +
                   '</div>' +
                  '<div id="' + subid + '_cont" class="flex_draw"></div>' +
                 '</div>';

      top.append(entry);

      function PopupWindow(div) {
         if (div === 'first') {
            div = null;
            $('#' + topid + ' .flex_frame').each(function() {
               if (!$(this).is(":hidden") && ($(this).prop('state') != "minimal")) div = $(this);
            });
            if (!div) return;
         }

         div.appendTo(div.parent());

         if (div.prop('state') == "minimal") return;

         div = div.find(".flex_draw").get(0);
         var dummy = new JSROOT.TObjectPainter();
         dummy.SetDivId(div, -1);
         JSROOT.Painter.SelectActivePad({ pp: dummy.canv_painter(), active: true });

         JSROOT.resize(div);
      }

      function ChangeWindowState(main, state) {
         var curr = main.prop('state');
         if (!curr) curr = "normal";
         main.prop('state', state);
         if (state==curr) return;

         if (curr == "normal") {
            main.prop('original_height', main.height());
            main.prop('original_width', main.width());
            main.prop('original_top', main.css('top'));
            main.prop('original_left', main.css('left'));
         }

         main.find(".jsroot_minbutton").find('.ui-icon')
             .toggleClass("ui-icon-triangle-1-s", state!="minimal")
             .toggleClass("ui-icon-triangle-2-n-s", state=="minimal");

         main.find(".jsroot_maxbutton").find('.ui-icon')
             .toggleClass("ui-icon-triangle-1-n", state!="maximal")
             .toggleClass("ui-icon-triangle-2-n-s", state=="maximal");

         switch (state) {
            case "minimal":
               main.height(main.find('.flex_header').height()).width("auto");
               main.find(".flex_draw").css("display","none");
               main.find(".ui-resizable-handle").css("display","none");
               break;
            case "maximal":
               main.height("100%").width("100%").css('left','').css('top','');
               main.find(".flex_draw").css("display","");
               main.find(".ui-resizable-handle").css("display","none");
               break;
            default:
               main.find(".flex_draw").css("display","");
               main.find(".ui-resizable-handle").css("display","");
               main.height(main.prop('original_height'))
                   .width(main.prop('original_width'));
               if (curr!="minimal")
                  main.css('left', main.prop('original_left'))
                      .css('top', main.prop('original_top'));
         }

         if (state !== "minimal")
            PopupWindow(main);
         else
            PopupWindow("first");
      }

      $("#" + subid)
         .css('left', parseInt(w * (this.cnt % 5)/10))
         .css('top', parseInt(h * (this.cnt % 5)/10))
         .width(Math.round(w * 0.58))
         .height(Math.round(h * 0.58))
         .resizable({
            helper: "jsroot-flex-resizable-helper",
            start: function(event, ui) {
               // bring element to front when start resizing
               PopupWindow($(this));
            },
            stop: function(event, ui) {
               var rect = { width : ui.size.width-1, height : ui.size.height - $(this).find(".flex_header").height()-1 };
               JSROOT.resize($(this).find(".flex_draw").get(0), rect);
            }
          })
          .draggable({
            containment: "parent",
            start: function(event, ui) {
               // bring element to front when start dragging
               PopupWindow($(this));

               var ddd = $(this).find(".flex_draw");

               // block dragging when mouse below header
               var elementMouseIsOver = document.elementFromPoint(event.clientX, event.clientY);
               var isparent = false;
               $(elementMouseIsOver).parents().map(function() { if ($(this).get(0) === ddd.get(0)) isparent = true; });
               if (isparent) return false;
            }
         })
       .click(function() { PopupWindow($(this)); })
       .find('.flex_header')
         // .hover(function() { $(this).toggleClass("ui-state-hover"); })
         .click(function() {
            PopupWindow($(this).parent());
         })
         .dblclick(function() {
            var main = $(this).parent();
            if (main.prop('state') == "normal")
               ChangeWindowState(main, "maximal");
            else
               ChangeWindowState(main, "normal");
         })
        .find("button")
           .first()
           .attr('title','close canvas')
           .button({ icons: { primary: "ui-icon-close" }, text: false })
           .click(function() {
              var main = $(this).parent().parent();
              mdi.CleanupFrame(main.find(".flex_draw").get(0));
              main.remove();
              PopupWindow('first'); // set active as first window
           })
           .next()
           .attr('title','maximize canvas')
           .addClass('jsroot_maxbutton')
           .button({ icons: { primary: "ui-icon-triangle-1-n" }, text: false })
           .click(function() {
              var main = $(this).parent().parent();
              var maximize = $(this).find('.ui-icon').hasClass("ui-icon-triangle-1-n");
              ChangeWindowState(main, maximize ? "maximal" : "normal");
           })
           .next()
           .attr('title','minimize canvas')
           .addClass('jsroot_minbutton')
           .button({ icons: { primary: "ui-icon-triangle-1-s" }, text: false })
           .click(function() {
              var main = $(this).parent().parent();
              var minimize = $(this).find('.ui-icon').hasClass("ui-icon-triangle-1-s");
              ChangeWindowState(main, minimize ? "minimal" : "normal");
           });

      // set default z-index to avoid overlap of these special elements
      $("#" + subid).find(".ui-resizable-handle").css('z-index', '');

      this.cnt++;

      return $("#" + subid + "_cont").attr('frame_title', title).get(0);
   }

   // ================== new grid with flexible boundaries ========

   JSROOT.GridDisplay.prototype.CreateSeparator = function(handle, main, group) {
      var separ = $(main.append("div").node());

      separ.toggleClass('jsroot_separator', true)
           .toggleClass(handle.vertical ? 'jsroot_hline' : 'jsroot_vline', true)
           .prop('handle', handle)
           .attr('separator-id', group.id)
           .css('position','absolute')
           .css(handle.vertical ? 'top' : 'left', "calc(" + group.position+"% - 2px)")
           .css(handle.vertical ? 'width' : 'height', (handle.size || 100)+"%")
           .css(handle.vertical ? 'height' : 'width', '5px')
           .css('cursor', handle.vertical ? "ns-resize" : "ew-resize");

      separ.bind('changePosition', function(e, drag_ui) {
         var handle = $(this).prop('handle'),
             id = parseInt($(this).attr('separator-id')),
             pos = handle.groups[id].position;

         if (drag_ui === 'restore') {
            pos = handle.groups[id].position0;
         } else
         if (drag_ui && drag_ui.offset) {
            if (handle.vertical)
               pos = Math.round((drag_ui.offset.top+2-$(this).parent().offset().top)/$(this).parent().innerHeight()*100);
            else
               pos = Math.round((drag_ui.offset.left+2-$(this).parent().offset().left)/$(this).parent().innerWidth()*100);
         }

         var diff = handle.groups[id].position - pos;

         if (Math.abs(diff)<0.3) return; // if no significant change, do nothing

         // do not change if size too small
         if (Math.min(handle.groups[id-1].size-diff, handle.groups[id].size+diff) < 5) return;

         handle.groups[id-1].size -= diff;
         handle.groups[id].size += diff;
         handle.groups[id].position = pos;

         function SetGroupSize(prnt, grid) {
            var name = handle.vertical ? 'height' : 'width',
                size = handle.groups[grid].size+'%';
            prnt.children("[groupid='"+grid+"']").css(name, size)
                .children(".jsroot_separator").css(name, size);
         }

         $(this).css(handle.vertical ? 'top' : 'left', "calc("+pos+"% - 2px)");

         SetGroupSize($(this).parent(), id-1);
         SetGroupSize($(this).parent(), id);

         if (drag_ui === 'restore') {
            $(this).trigger('resizeGroup', id-1);
            $(this).trigger('resizeGroup', id);
         }
      });

      separ.bind('resizeGroup', function(e, grid) {
         var sel = $(this).parent().children("[groupid='"+grid+"']");
         if (!sel.hasClass('jsroot_newgrid')) sel = sel.find(".jsroot_newgrid");
         sel.each(function() { JSROOT.resize($(this).get(0)); });
      });

      separ.dblclick(function() {
         $(this).trigger('changePosition', 'restore');
      });

      separ.draggable({
         axis: handle.vertical ? "y" : "x",
         cursor: handle.vertical ? "ns-resize" : "ew-resize",
         containment: "parent",
         helper : function() { return $(this).clone().css('background-color','grey'); },
         start: function(event,ui) {
            // remember start position
            var handle = $(this).prop('handle'),
                id = parseInt($(this).attr('separator-id'));
            handle.groups[id].startpos = handle.groups[id].position;
         },
         drag: function(event,ui) {
            $(this).trigger('changePosition', ui);
         },
         stop: function(event,ui) {
            // verify if start position was changed
            var handle = $(this).prop('handle'),
               id = parseInt($(this).attr('separator-id'));
            if (Math.abs(handle.groups[id].startpos - handle.groups[id].position)<0.5) return;

            $(this).trigger('resizeGroup', id-1);
            $(this).trigger('resizeGroup', id);
         }
      });
   }

   // ========== performs tree drawing on server ==================

   JSROOT.CreateTreePlayer = function(player) {

      player.draw_first = true;

      player.ConfigureOnline = function(itemname, url, askey, root_version, dflt_expr) {
         this.SetItemName(itemname, "", this);
         this.url = url;
         this.root_version = root_version;
         this.askey = askey;
         this.dflt_expr = dflt_expr;
      }

      player.ConfigureTree = function(tree) {
         this.local_tree = tree;
      }

      player.KeyUp = function(e) {
         if (e.keyCode == 13) this.PerformDraw();
      }

      player.ShowExtraButtons = function(args) {
         var main = $(this.select_main().node());

          main.find(".treedraw_buttons")
             .append(" Cut: <input class='treedraw_cut ui-corner-all ui-widget' style='width:8em;margin-left:5px' title='cut expression'></input>"+
                     " Opt: <input class='treedraw_opt ui-corner-all ui-widget' style='width:5em;margin-left:5px' title='histogram draw options'></input>"+
                     " Num: <input class='treedraw_number' style='width:7em;margin-left:5px' title='number of entries to process (default all)'></input>" +
                     " First: <input class='treedraw_first' style='width:7em;margin-left:5px' title='first entry to process (default first)'></input>" +
                     " <button class='treedraw_clear' title='Clear drawing'>Clear</button>");

          var page = 1000, numentries = undefined, p = this;
          if (this.local_tree) numentries = this.local_tree.fEntries || 0;

          main.find(".treedraw_cut").val(args && args.parse_cut ? args.parse_cut : "").keyup(this.keyup);
          main.find(".treedraw_opt").val(args && args.drawopt ? args.drawopt : "").keyup(this.keyup);
          main.find(".treedraw_number").val(args && args.numentries ? args.numentries : "").spinner({ numberFormat: "n", min: 0, page: 1000, max: numentries }).keyup(this.keyup);
          main.find(".treedraw_first").val(args && args.firstentry ? args.firstentry : "").spinner({ numberFormat: "n", min: 0, page: 1000, max: numentries }).keyup(this.keyup);
          main.find(".treedraw_clear").button().click(function() { JSROOT.cleanup(p.drawid); });
      }

      player.Show = function(divid, args) {
         this.drawid = divid + "_draw";

         this.keyup = this.KeyUp.bind(this);

         var show_extra = args && (args.parse_cut || args.numentries || args.firstentry);

         var main = $("#" + divid);

         main.html("<div class='treedraw_buttons' style='padding-left:0.5em'>" +
               "<button class='treedraw_exe' title='Execute draw expression'>Draw</button>" +
               " Expr:<input class='treedraw_varexp ui-corner-all ui-widget' style='width:12em;margin-left:5px' title='draw expression'></input> " +
               (show_extra ? "" : "<button class='treedraw_more'>More</button>") +
               "</div>" +
               "<hr/>" +
               "<div id='" + this.drawid + "' style='width:100%'></div>");

         // only when main html element created, one can set divid
         this.SetDivId(divid);

         var p = this;

         if (this.local_tree)
            main.find('.treedraw_buttons').attr('title', "Tree draw player for: " + this.local_tree.fName);
         main.find('.treedraw_exe').button().click(function() { p.PerformDraw(); });
         main.find('.treedraw_varexp')
              .val(args && args.parse_expr ? args.parse_expr : (this.dflt_expr || "px:py"))
              .keyup(this.keyup);

         if (show_extra) {
            this.ShowExtraButtons(args);
         } else {
            main.find('.treedraw_more').button().click(function() {
               $(this).remove();
               p.ShowExtraButtons();
            });
         }

         this.CheckResize();
      }

      player.PerformLocalDraw = function() {
         if (!this.local_tree) return;

         var frame = $(this.select_main().node()),
             args = { expr: frame.find('.treedraw_varexp').val() };

         if (frame.find('.treedraw_more').length==0) {
            args.cut = frame.find('.treedraw_cut').val();
            if (!args.cut) delete args.cut;

            args.drawopt = frame.find('.treedraw_opt').val();
            if (args.drawopt === "dump") { args.dump = true; args.drawopt = ""; }
            if (!args.drawopt) delete args.drawopt;

            args.numentries = parseInt(frame.find('.treedraw_number').val());
            if (isNaN(args.numentries)) delete args.numentries;

            args.firstentry = parseInt(frame.find('.treedraw_first').val());
            if (isNaN(args.firstentry)) delete args.firstentry;
         }

         var p = this;

         if (args.drawopt) JSROOT.cleanup(p.drawid);

         p.local_tree.Draw(args, function(histo, hopt, intermediate) {
            JSROOT.redraw(p.drawid, histo, hopt);
         });
      }

      player.PerformDraw = function() {

         if (this.local_tree) return this.PerformLocalDraw();

         var frame = $(this.select_main().node()),
             url = this.url + '/exe.json.gz?compact=3&method=Draw',
             expr = frame.find('.treedraw_varexp').val(),
             hname = "h_tree_draw", option = "",
             pos = expr.indexOf(">>");

         if (pos<0) {
            expr += ">>" + hname;
         } else {
            hname = expr.substr(pos+2);
            if (hname[0]=='+') hname = hname.substr(1);
            var pos2 = hname.indexOf("(");
            if (pos2>0) hname = hname.substr(0, pos2);
         }

         if (frame.find('.treedraw_more').length==0) {
            var cut = frame.find('.treedraw_cut').val(),
                nentries = frame.find('.treedraw_number').val(),
                firstentry = frame.find('.treedraw_first').val();

            option = frame.find('.treedraw_opt').val();

            url += '&prototype="const char*,const char*,Option_t*,Long64_t,Long64_t"&varexp="' + expr + '"&selection="' + cut + '"';

            // provide all optional arguments - default value kMaxEntries not works properly in ROOT6
            if (nentries=="") nentries = (this.root_version >= 394499) ? "TTree::kMaxEntries": "1000000000"; // kMaxEntries available since ROOT 6.05/03
            if (firstentry=="") firstentry = "0";
            url += '&option="' + option + '"&nentries=' + nentries + '&firstentry=' + firstentry;
         } else {
            url += '&prototype="Option_t*"&opt="' + expr + '"';
         }
         url += '&_ret_object_=' + hname;

         var player = this;

         function SubmitDrawRequest() {
            JSROOT.NewHttpRequest(url, 'object', function(res) {
               if (!res) return;
               JSROOT.cleanup(player.drawid);
               JSROOT.draw(player.drawid, res, option);
            }).send();
         }

         if (this.askey) {
            // first let read tree from the file
            this.askey = false;
            JSROOT.NewHttpRequest(this.url + "/root.json", 'text', SubmitDrawRequest).send();
         } else {
            SubmitDrawRequest();
         }
      }

      player.CheckResize = function(arg) {
         var main = $(this.select_main().node());

         $("#" + this.drawid).width(main.width());
         var h = main.height(),
             h0 = main.find(".treedraw_buttons").outerHeight(true),
             h1 = main.find("hr").outerHeight(true);

         $("#" + this.drawid).height(h - h0 - h1 - 2);

         JSROOT.resize(this.drawid);
      }

      return player;
   }

   /// @private
   /// function used with THttpServer to assign player for the TTree object

   JSROOT.drawTreePlayer = function(hpainter, itemname, askey, asleaf) {

      var item = hpainter.Find(itemname),
          top = hpainter.GetTopOnlineItem(item),
          draw_expr = "", leaf_cnt = 0;
      if (!item || !top) return null;

      if (asleaf) {
         draw_expr = item._name;
         while (item && !item._ttree) item = item._parent;
         if (!item) return null;
         itemname = hpainter.itemFullName(item);
      }

      var url = hpainter.GetOnlineItemUrl(itemname);
      if (!url) return null;

      var root_version = top._root_version ? parseInt(top._root_version) : 396545; // by default use version number 6-13-01

      var mdi = hpainter.GetDisplay();
      if (!mdi) return null;

      var frame = mdi.FindFrame(itemname, true);
      if (!frame) return null;

      var divid = d3.select(frame).attr('id'),
          player = new JSROOT.TBasePainter();

      if (item._childs && !asleaf)
         for (var n=0;n<item._childs.length;++n) {
            var leaf = item._childs[n];
            if (leaf && leaf._kind && (leaf._kind.indexOf("ROOT.TLeaf")==0) && (leaf_cnt<2)) {
               if (leaf_cnt++ > 0) draw_expr+=":";
               draw_expr+=leaf._name;
            }
         }

      JSROOT.CreateTreePlayer(player);
      player.ConfigureOnline(itemname, url, askey, root_version, draw_expr);
      player.Show(divid);

      return player;
   }

   /// @private
   /// function used with THttpServer when tree is not yet loaded
   JSROOT.drawTreePlayerKey = function(hpainter, itemname) {
      return JSROOT.drawTreePlayer(hpainter, itemname, true);
   }

   /// @private
   /// function used with THttpServer for when tree is not yet loaded
   JSROOT.drawLeafPlayer = function(hpainter, itemname) {
      return JSROOT.drawTreePlayer(hpainter, itemname, false, true);
   }

   // =======================================================================

   JSROOT.Painter.ConfigureVSeparator = function(handle) {
      // FIXME: obsolete, will be removed
   }

   JSROOT.Painter.AdjustLayout = function(left, height, firsttime) {
      // FIXME: obsolete, will be removed
      if (JSROOT.hpainter && JSROOT.hpainter.brlayout)
         JSROOT.hpainter.brlayout.AdjustSeparator(left, height, true);
   }

   JSROOT.Painter.ConfigureHSeparator = function(height) {
      // FIXME: obsolete, will be removed
      if (!JSROOT.hpainter) return "";

      return JSROOT.hpainter.CreateStatusLine(height);
   }

   return JSROOT;

}));

