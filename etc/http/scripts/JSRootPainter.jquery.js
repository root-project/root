/// @file JSRootPainter.jquery.js
/// Part of JavaScript ROOT graphics, dependent from jQuery functionality

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( ['jquery', 'jquery-ui', 'd3', 'JSRootPainter'], factory );
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

      // Browser globals
      factory(jQuery, jQuery.ui, d3, JSROOT);
   }
} (function($, myui, d3, JSROOT) {

   if ( typeof define === "function" && define.amd )
      JSROOT.loadScript('$$$style/jquery-ui.css');

   JSROOT.Painter.createMenu = function(maincallback, menuname) {
      if (!menuname || (typeof menuname !== 'string')) menuname = 'root_ctx_menu';

      var menu = { element: null, code: "", cnt: 1, funcs: {}, separ: false };

      menu.add = function(name, arg, func) {
         if (name == "separator") { this.code += "<li>-</li>"; this.separ = true; return; }

         if (name.indexOf("header:")==0) {
            this.code += "<li class='ui-widget-header' style='padding-left:5px'>"+name.substr(7)+"</li>";
            return;
         }

         if (name=="endsub:") { this.code += "</ul></li>"; return; }
         var close_tag = "</li>", style = "padding-top:2px;padding-bottom:1px";
         if (name.indexOf("sub:")==0) { name = name.substr(4); close_tag="<ul>"; style += ";padding-right:2em"}

         if (typeof arg == 'function') { func = arg; arg = name;  }

         // if ((arg==null) || (typeof arg != 'string')) arg = name;

         if (name.indexOf("chk:")==0) { name = "<span class='ui-icon ui-icon-check' style='margin:1px'></span>" + name.substr(4); } else
         if (name.indexOf("unk:")==0) { name = "<span class='ui-icon ui-icon-blank' style='margin:1px'></span>" + name.substr(4); }

         // special handling of first versions with menu support
         if (($.ui.version.indexOf("1.10")==0) || ($.ui.version.indexOf("1.9")==0))
            name = '<a href="#">' + name + '</a>';

         this.code += "<li cnt='" + this.cnt + "' arg='" + arg + "' style='" + style + "'>" + name + close_tag;
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

         if (opts.length === 1) {
            if (opts[0]==='inspect') menu_name = menu_name.replace("Draw", "Inspect");
            return this.add(menu_name, opts[0], call_back);
         }

         this.add("sub:" + menu_name, opts[0], call_back);

         for (var i=0;i<opts.length;++i) {
            var name = opts[i];
            if (name=="") name = '&lt;dflt&gt;';

            var group = i+1;
            if ((opts.length>5) && (name.length>0)) {
               // check if there are similar options, which can be grouped once again
               while ((group<opts.length) && (opts[group].indexOf(name)==0)) group++;
            }

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
         this.add("endsub:");
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
            .css('font-size', '80%')
            .css('position', 'absolute') // this overrides ui-menu-items class property
            .menu({
               items: "> :not(.ui-widget-header)",
               select: function( event, ui ) {
                  var arg = ui.item.attr('arg');
                  var cnt = ui.item.attr('cnt');
                  var func = cnt ? pthis.funcs[cnt] : null;
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

   JSROOT.HierarchyPainter.prototype.isLastSibling = function(hitem) {
      if (!hitem || !hitem._parent || !hitem._parent._childs) return false;
      var chlds = hitem._parent._childs, indx = chlds.indexOf(hitem);
      if (indx<0) return false;
      while (++indx < chlds.length)
         if (!('_hidden' in chlds[indx])) return false;
      return true;
   }

   JSROOT.HierarchyPainter.prototype.addItemHtml = function(hitem, d3prnt, doupdate) {

      if (!hitem || ('_hidden' in hitem)) return;

      var isroot = (hitem === this.h),
          has_childs = ('_childs' in hitem),
          handle = JSROOT.getDrawHandle(hitem._kind),
          img1 = "", img2 = "", can_click = false,
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

      if (doupdate) {
         d3prnt.selectAll("*").remove();
         d3cont = d3prnt;
      } else {
         d3cont = d3prnt.append("div");
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
      if (has_childs) {
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
         if (this.isLastSibling(hitem)) icon_class+="bottom";
         var d3icon = d3line.append("div").attr('class', icon_class);
         if (plusminus) d3icon.style('cursor','pointer')
                              .on("click", function() { h.tree_click(this, "plusminus"); });
      }

      // make node icons

      if (this.with_icons) {
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
                          .style('vertical-align','top')
                          .style('width','18px')
                          .style('height','18px');

         if (('_icon_click' in hitem) || (handle && ('icon_click' in handle)))
            d3img.on("click", function() { h.tree_click(this, "icon"); });
      }

      var d3a = d3line.append("a");
      if (can_click || has_childs)
         d3a.attr("class","h_item")
            .on("click", function() { h.tree_click(this); });

      if ('disp_kind' in h) {
         if (JSROOT.gStyle.DragAndDrop && can_click)
           this.enable_dragging(d3a.node(), itemname);
         if (JSROOT.gStyle.ContextMenu && can_menu)
            d3a.on('contextmenu', function() { h.tree_contextmenu(this); });
      }

      var element_name = hitem._name;

      if ('_realname' in hitem)
         element_name = hitem._realname;

      var element_title = "";
      if ('_title' in hitem) element_title = hitem._title;

      if ('_fullname' in hitem)
         element_title += "  fullname: " + hitem._fullname;

      if (element_title.length === 0)
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
         for (var i=0; i< hitem._childs.length;++i) {
            var chld = hitem._childs[i];
            chld._parent = hitem;
            this.addItemHtml(chld, d3chlds);
         }
      }
   }

   JSROOT.HierarchyPainter.prototype.RefreshHtml = function(callback) {

      if (!this.divid) return JSROOT.CallBack(callback);

      var d3elem = this.select_main();

      d3elem.html(""); // clear html - most simple way

      var h = this, factcmds = [], status_item = null;
      this.ForEach(function(item) {
         delete item._d3cont; // remove html container
         if (('_fastcmd' in item) && (item._kind == 'Command')) factcmds.push(item);
         if (('_status' in item) && (status_item==null)) status_item = item;
      });

      if ((this.h == null) || d3elem.empty())
         return JSROOT.CallBack(callback);

      var maindiv =
         d3elem.append("div")
               .attr("class", "jsroot")
               .style("background-color", this.background ? this.background : "")
               .style('font-size', this.with_icons ? "12px" : "15px");

      for (var n=0;n<factcmds.length;++n) {
         var btn = maindiv.append("button")
                    .text("")
                    .attr("class",'fast_command')
                    .attr("item", this.itemFullName(factcmds[n]))
                    .attr("title", factcmds[n]._title)
                    .on("click", function() { h.ExecuteCommand(d3.select(this).attr("item"), this); } );

         if ('_icon' in factcmds[n])
            btn.append('img').attr("src", factcmds[n]._icon);
      }

      var d3p = maindiv.append("p");

      d3p.append("a").attr("href", '#').text("open all").on("click", function() { h.toggle(true); d3.event.preventDefault(); });
      d3p.append("text").text(" | ");
      d3p.append("a").attr("href", '#').text("close all").on("click", function() { h.toggle(false); d3.event.preventDefault(); });

      if ('_online' in this.h) {
         d3p.append("text").text(" | ");
         d3p.append("a").attr("href", '#').text("reload").on("click", function() { h.reload(); d3.event.preventDefault(); });
      }

      if ('disp_kind' in this) {
         d3p.append("text").text(" | ");
         d3p.append("a").attr("href", '#').text("clear").on("click", function() { h.clear(false); d3.event.preventDefault(); });
      }

      if (this.nobrowser) {
         d3p.append("text").text(" | ");
         d3p.append("a").attr("href", '#').text("hide").on("click", function() { h.ToggleFloatBrowser(false); d3.event.preventDefault(); });
      }

      this.addItemHtml(this.h, maindiv.append("div").attr("class","h_tree"));

      if (status_item && (JSROOT.GetUrlOption('nostatus')===null)) {
         var func = JSROOT.findFunction(status_item._status);
         var hdiv = (typeof func == 'function') ? JSROOT.Painter.ConfigureHSeparator(30) : null;
         if (hdiv) func(hdiv, this.itemFullName(status_item));
      }

      JSROOT.CallBack(callback);
   }

   JSROOT.HierarchyPainter.prototype.UpdateTreeNode = function(hitem, d3cont) {
      if ((d3cont===undefined) || d3cont.empty())  {
         d3cont = d3.select(hitem._d3cont ? hitem._d3cont : null);
         var name = this.itemFullName(hitem);
         if (d3cont.empty())
            d3cont = this.select_main().select("[item='" + name + "']");
         if (d3cont.empty() && ('_cycle' in hitem))
            d3cont = this.select_main().select("[item='" + name + ";" + hitem._cycle + "']");
         if (d3cont.empty()) return;
      }

      this.addItemHtml(hitem, d3cont, true);
   }

   JSROOT.HierarchyPainter.prototype.UpdateBackground = function(hitem, scroll_into_view) {

      if (!hitem || !hitem._d3cont) return;

      var d3cont = d3.select(hitem._d3cont);

      if (d3cont.empty()) return;

      var d3a = d3cont.select(".h_item");

      d3a.style('background', hitem._background ? hitem._background : null);

      if (scroll_into_view && hitem._background)
         d3a.node().scrollIntoView(false);
   }

   JSROOT.HierarchyPainter.prototype.tree_click = function(node, place) {
      if (node===null) return;
      var d3cont = d3.select(node.parentNode.parentNode);

      var itemname = d3cont.attr('item');
      if (itemname == null) return;
      
      var hitem = this.Find(itemname);
      if (hitem == null) return;

      var prnt = hitem, dflt = undefined;
      while (prnt) {
         if ((dflt = prnt._click_action) !== undefined) break;
         prnt = prnt._parent;
      }

      if (!place || (place=="")) place = "item";

      var sett = JSROOT.getDrawSettings(hitem._kind),
          handle = sett.handle;

      if (place == "icon") {
         var func = null;
         if (typeof hitem._icon_click == 'function') func = hitem._icon_click; else
         if (handle && typeof handle.icon_click == 'function') func = handle.icon_click;
         if (func && func(hitem,this))
            this.UpdateTreeNode(hitem, d3cont);
         return;
      }

      // special feature - all items with '_expand' function are not drawn by click
      if ((place=="item") && ('_expand' in hitem)) place = "plusminus";

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

         var can_draw = hitem._can_draw,
             can_expand = hitem._more,
             dflt_expand = (this.default_by_click === "expand");

         if (hitem._childs) can_expand = false;

         if (can_draw === undefined) can_draw = sett.draw;
         if (can_expand === undefined) can_expand = sett.expand;

         if (can_draw && can_expand) {
            // if default action specified as expand, disable drawing
            if (dflt_expand || (handle && (handle.dflt === 'expand'))) can_draw = false; else
            if (this.isItemDisplayed(itemname)) can_draw = false; // if already displayed, try to expand
         }

         if (can_draw)
            return this.display(itemname);

         if (can_expand || dflt_expand)
            return this.expand(itemname, null, d3cont);

         // cannot draw, but can inspect ROOT objects
         if ((typeof hitem._kind === "string") && (hitem._kind.indexOf("ROOT.")===0) && sett.inspect)
            return this.display(itemname, "inspect");

         if (!hitem._childs || (hitem === this.h)) return;
      }

      if (hitem._isopen)
         delete hitem._isopen;
      else
         hitem._isopen = true;

      this.UpdateTreeNode(hitem, d3cont);
   }

   JSROOT.HierarchyPainter.prototype.tree_contextmenu = function(elem) {
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

      JSROOT.Painter.createMenu(function(menu) {

         menu.painter = painter;

         if ((itemname == "") && !('_jsonfile' in hitem)) {
            var addr = "", cnt = 0;
            function separ() { return cnt++ > 0 ? "&" : "?"; }

            var files = [];
            painter.ForEachRootFile(function(item) { files.push(item._file.fFullURL); });

            if (painter.GetTopOnlineItem()==null)
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
         } else
         if (onlineprop != null) {
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

            if (fileprop && sett.opts) {
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

   JSROOT.HierarchyPainter.prototype.CreateDisplay = function(callback) {
      if ('disp' in this) {
         if (this.disp.NumDraw() > 0) return JSROOT.CallBack(callback, this.disp);
         this.disp.Reset();
         delete this.disp;
      }

      // check that we can found frame where drawing should be done
      if (document.getElementById(this.disp_frameid) == null)
         return JSROOT.CallBack(callback, null);

      if (this.disp_kind == "tabs")
         this.disp = new JSROOT.TabsDisplay(this.disp_frameid);
      else
      if ((this.disp_kind == "flex") || (this.disp_kind == "flexible"))
         this.disp = new JSROOT.FlexibleDisplay(this.disp_frameid);
      else
      if (this.disp_kind.search("grid") == 0)
         this.disp = new JSROOT.GridDisplay(this.disp_frameid, this.disp_kind);
      else
      if (this.disp_kind == "simple")
         this.disp = new JSROOT.SimpleDisplay(this.disp_frameid);
      else
         this.disp = new JSROOT.CollapsibleDisplay(this.disp_frameid);

      if (this.disp)
         this.disp.CleanupFrame = this.CleanupFrame.bind(this);

      JSROOT.CallBack(callback, this.disp);
   }

   JSROOT.HierarchyPainter.prototype.enable_dragging = function(element, itemname) {
      $(element).draggable({ revert: "invalid", appendTo: "body", helper: "clone" });
   }

   JSROOT.HierarchyPainter.prototype.enable_dropping = function(frame, itemname) {
      var h = this;
      $(frame).droppable({
         hoverClass : "ui-state-active",
         accept: function(ui) {
            var dropname = ui.parent().parent().attr('item');
            if ((dropname == itemname) || (dropname==null)) return false;

            var ditem = h.Find(dropname);
            if ((ditem==null) || (!('_kind' in ditem))) return false;

            return ditem._kind.indexOf("ROOT.")==0;
         },
         drop: function(event, ui) {
            var dropname = ui.draggable.parent().parent().attr('item');
            if (dropname==null) return false;
            return h.dropitem(dropname, $(this).attr("id"));
         }
      });
   }

   // ==================================================

   JSROOT.CollapsibleDisplay = function(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0; // use to count newly created frames
   }

   JSROOT.CollapsibleDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   JSROOT.CollapsibleDisplay.prototype.ForEachFrame = function(userfunc,  only_visible) {
      var topid = this.frameid + '_collapsible';

      if (document.getElementById(topid) == null) return;

      if (typeof userfunc != 'function') return;

      $('#' + topid + ' .collapsible_draw').each(function() {

         // check if only visible specified
         if (only_visible && $(this).is(":hidden")) return;

         userfunc($(this).get(0));
      });
   }

   JSROOT.CollapsibleDisplay.prototype.GetActiveFrame = function() {
      var found = JSROOT.MDIDisplay.prototype.GetActiveFrame.call(this);
      if (found && !$(found).is(":hidden")) return found;

      found = null;
      this.ForEachFrame(function(frame) {
         if (!found) found = frame;
      }, true);

      return found;
   }

   JSROOT.CollapsibleDisplay.prototype.ActivateFrame = function(frame) {
      if ($(frame).is(":hidden")) {
         $(frame).prev().toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
                 .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
                 .next().toggleClass("ui-accordion-content-active").slideDown(0);
      }
      $(frame).prev()[0].scrollIntoView();
      // remember title
      this.active_frame_title = d3.select(frame).attr('frame_title');
   }

   JSROOT.CollapsibleDisplay.prototype.CreateFrame = function(title) {

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
                     var sub = $(this).next();
                     sub.attr('frame_active', sub.is(":hidden") ? "false" : "true");
                     JSROOT.resize(sub.attr('id'));
                     return false;
                  })
            .next()
            .addClass("ui-accordion-content ui-helper-reset ui-widget-content ui-corner-bottom")
            .hide();

      $('#' + uid).find(" .jsroot_collaps_closebtn")
           .button({ icons: { primary: "ui-icon-close" }, text: false })
           .click(function(){
              mdi.CleanupFrame($(this).parent().next().attr('id'));
              $(this).parent().next().andSelf().remove();
           });

      $('#' + uid)
            .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
            .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end().next()
            .toggleClass("ui-accordion-content-active").slideToggle(0);

      return $("#" + hid).attr('frame_title', title).css('overflow','hidden').get(0);
   }

   // ================================================

   JSROOT.TabsDisplay = function(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0;
   }

   JSROOT.TabsDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   JSROOT.TabsDisplay.prototype.ForEachFrame = function(userfunc, only_visible) {
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

   JSROOT.TabsDisplay.prototype.GetActiveFrame = function() {
      var found = null;
      this.ForEachFrame(function(frame) {
         if (!found) found = frame;
      }, true);

      return found;
   }

   JSROOT.TabsDisplay.prototype.ActivateFrame = function(frame) {
      var cnt = 0, id = -1;
      this.ForEachFrame(function(fr) {
         if ($(fr).attr('id') == $(frame).attr('id')) id = cnt;
         cnt++;
      });
      $('#' + this.frameid + "_tabs").tabs("option", "active", id);

      this.active_frame_title = d3.select(frame).attr('frame_title');
   }

   JSROOT.TabsDisplay.prototype.CreateFrame = function(title) {

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
                             console.log('tabs activate');
                             $(ui.oldPanel).attr('frame_active', 'false');
                             $(ui.newPanel).css('overflow', 'hidden').attr('frame_active', 'true');
                             JSROOT.resize($(ui.newPanel).attr('id'));
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
         .attr('frame_active', 'true')
         .attr('frame_title', title);

      return $('#' + hid).get(0);
   }

   JSROOT.TabsDisplay.prototype.CheckMDIResize = function(frame_id, size) {
      $("#" + this.frameid + '_tabs').tabs("refresh");
      JSROOT.MDIDisplay.prototype.CheckMDIResize.call(this, frame_id, size);
   }

   // ==================================================

   JSROOT.FlexibleDisplay = function(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0; // use to count newly created frames
   }

   JSROOT.FlexibleDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   JSROOT.FlexibleDisplay.prototype.ForEachFrame = function(userfunc,  only_visible) {
      var topid = this.frameid + '_flex';

      if (document.getElementById(topid) == null) return;
      if (typeof userfunc != 'function') return;

      $('#' + topid + ' .flex_draw').each(function() {
         // check if only visible specified
         if (only_visible && $(this).is(":hidden")) return;

         userfunc($(this).get(0));
      });
   }

   JSROOT.FlexibleDisplay.prototype.GetActiveFrame = function() {
      var found = JSROOT.MDIDisplay.prototype.GetActiveFrame.call(this);
      if (found && !$(found).is(":hidden")) return found;

      found = null;
      this.ForEachFrame(function(frame) {
         if (!found) found = frame;
      }, true);

      return found;
   }


   JSROOT.FlexibleDisplay.prototype.ActivateFrame = function(frame) {

      this.active_frame_title = d3.select(frame).attr('frame_title');
   }

   JSROOT.FlexibleDisplay.prototype.CreateFrame = function(title) {

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
             .toggleClass("ui-icon-carat-1-s", state!="minimal")
             .toggleClass("ui-icon-carat-2-n-s", state=="minimal");

         main.find(".jsroot_maxbutton").find('.ui-icon')
             .toggleClass("ui-icon-carat-1-n", state!="maximal")
             .toggleClass("ui-icon-carat-2-n-s", state=="maximal");

         switch (state) {
            case "minimal" :
               main.height(main.find('.flex_header').height()).width("auto");
               main.find(".flex_draw").css("display","none");
               main.find(".ui-resizable-handle").css("display","none");
               break;
            case "maximal" :
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

         main.find(".flex_draw").attr('frame_active', state !== "minimal" ? "true" : "false");

         if (state !== "minimal")
            JSROOT.resize(main.find(".flex_draw").get(0));
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
               $(this).appendTo($(this).parent());
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
               $(this).appendTo($(this).parent());
               var ddd = $(this).find(".flex_draw");

               // block dragging when mouse below header
               var elementMouseIsOver = document.elementFromPoint(event.clientX, event.clientY);
               var isparent = false;
               $(elementMouseIsOver).parents().map(function() { if ($(this).get(0) === ddd.get(0)) isparent = true; });
               if (isparent) return false;
            }
         })
       .find('.flex_header')
         // .hover(function() { $(this).toggleClass("ui-state-hover"); })
         .click(function() {
            var div = $(this).parent();
            div.appendTo(div.parent());
         })
        .find("button")
           .first()
           .attr('title','close canvas')
           .button({ icons: { primary: "ui-icon-close" }, text: false })
           .click(function() {
              var main = $(this).parent().parent();
              mdi.CleanupFrame(main.find(".flex_draw").get(0));
              main.remove();
           })
           .next()
           .attr('title','maximize canvas')
           .addClass('jsroot_maxbutton')
           .button({ icons: { primary: "ui-icon-carat-1-n" }, text: false })
           .click(function() {
              var main = $(this).parent().parent();
              var maximize = $(this).find('.ui-icon').hasClass("ui-icon-carat-1-n");
              ChangeWindowState(main, maximize ? "maximal" : "normal");
           })
           .next()
           .attr('title','minimize canvas')
           .addClass('jsroot_minbutton')
           .button({ icons: { primary: "ui-icon-carat-1-s" }, text: false })
           .click(function() {
              var main = $(this).parent().parent();
              var minimize = $(this).find('.ui-icon').hasClass("ui-icon-carat-1-s");
              ChangeWindowState(main, minimize ? "minimal" : "normal");
           });

      // set default z-index to avoid overlap of these special elements
      $("#" + subid).find(".ui-resizable-handle").css('z-index', '');

      //var draw_w = $("#" + subid).width() - 1;
      //var draw_h = $("#" + subid).height() - $("#" + subid).find(".flex_header").height()-1;
      //$("#" + subid).find(".flex_draw").width(draw_w).height(draw_h);
      this.cnt++;

      return $("#" + subid + "_cont").attr('frame_title', title).get(0);
   }

   // ========== performs tree drawing on server ==================

   JSROOT.TTreePlayer = function(itemname, url, askey, root_version) {
      JSROOT.TBasePainter.call(this);
      this.SetItemName(itemname);
      this.url = url;
      this.root_version = root_version;
      this.hist_painter = null;
      this.askey = askey;
      return this;
   }

   JSROOT.TTreePlayer.prototype = Object.create( JSROOT.TBasePainter.prototype );

   JSROOT.TTreePlayer.prototype.Show = function(divid) {
      this.drawid = divid + "_draw";

      $("#" + divid)
        .html("<div class='treedraw_buttons' style='padding-left:0.5em'>" +
            "<button class='treedraw_exe'>Draw</button>" +
            " Expr:<input class='treedraw_varexp' style='width:12em'></input> " +
            "<button class='treedraw_more'>More</button>" +
            "</div>" +
            "<div id='" + this.drawid + "' style='width:100%'></div>");

      var player = this;

      $("#" + divid).find('.treedraw_exe').click(function() { player.PerformDraw(); });
      $("#" + divid).find('.treedraw_varexp')
           .val("px:py")
           .keyup(function(e){
               if(e.keyCode == 13) player.PerformDraw();
            });

      $("#" + divid).find('.treedraw_more').click(function() {
         $(this).remove();
         $("#" + divid).find(".treedraw_buttons")
         .append(" Cut:<input class='treedraw_cut' style='width:8em'></input>"+
                 " Opt:<input class='treedraw_opt' style='width:5em'></input>"+
                 " Num:<input class='treedraw_number' style='width:7em'></input>" +
                 " First:<input class='treedraw_first' style='width:7em'></input>");

         $("#" + divid +" .treedraw_opt").val("");
         $("#" + divid +" .treedraw_number").val("").spinner({ numberFormat: "n", min: 0, page: 1000});
         $("#" + divid +" .treedraw_first").val("").spinner({ numberFormat: "n", min: 0, page: 1000});
      });

      this.SetDivId(divid);

      this.CheckResize();
   }

   JSROOT.TTreePlayer.prototype.PerformDraw = function() {

      var frame = $(this.select_main().node());

      var url = this.url + '/exe.json.gz?compact=3&method=Draw';
      var expr = frame.find('.treedraw_varexp').val();
      var hname = "h_tree_draw";

      var pos = expr.indexOf(">>");
      if (pos<0) {
         expr += ">>" + hname;
      } else {
         hname = expr.substr(pos+2);
         if (hname[0]=='+') hname = hname.substr(1);
         var pos2 = hname.indexOf("(");
         if (pos2>0) hname = hname.substr(0, pos2);
      }

      if (frame.find('.treedraw_more').length==0) {
         var cut = frame.find('.treedraw_cut').val();
         var option = frame.find('.treedraw_opt').val();
         var nentries = frame.find('.treedraw_number').val();
         var firstentry = frame.find('.treedraw_first').val();

         url += '&prototype="const char*,const char*,Option_t*,Long64_t,Long64_t"&varexp="' + expr + '"&selection="' + cut + '"';

         // if any of optional arguments specified, specify all of them
         if ((option!="") || (nentries!="") || (firstentry!="")) {
            if (nentries=="") nentries = (this.root_version >= 394499) ? "TTree::kMaxEntries": "1000000000"; // kMaxEntries available since ROOT 6.05/03
            if (firstentry=="") firstentry = "0";
            url += '&option="' + option + '"&nentries=' + nentries + '&firstentry=' + firstentry;
         }
      } else {
         url += '&prototype="Option_t*"&opt="' + expr + '"';
      }
      url += '&_ret_object_=' + hname;

      var player = this;

      function SubmitDrawRequest() {
         JSROOT.NewHttpRequest(url, 'object', function(res) {
            if (res==null) return;
            $("#"+player.drawid).empty();
            player.hist_painter = JSROOT.draw(player.drawid, res)
         }).send();
      }

      if (this.askey) {
         // first let read tree from the file
         this.askey = false;
         JSROOT.NewHttpRequest(this.url + "/root.json", 'text', SubmitDrawRequest).send();
      } else SubmitDrawRequest();
   }

   JSROOT.TTreePlayer.prototype.CheckResize = function(arg) {
      var main = $(this.select_main().node());

      $("#" + this.drawid).width(main.width());
      var h = main.height();
      var h0 = main.find(".treedraw_buttons").height();
      $("#" + this.drawid).height(h - 1 - h0);

      if (this.hist_painter)
         this.hist_painter.CheckResize(arg);
   }

   JSROOT.drawTreePlayer = function(hpainter, itemname, askey) {

      var url = hpainter.GetOnlineItemUrl(itemname);
      if (!url) return null;

      var top = hpainter.GetTopOnlineItem(hpainter.Find(itemname));
      if (!top) return null;
      var root_version = ('_root_version' in top) ? top._root_version : 336417; // by default use version number 5-34-32

      var mdi = hpainter.GetDisplay();
      if (mdi == null) return null;

      var frame = mdi.FindFrame(itemname, true);
      if (frame==null) return null;

      var divid = d3.select(frame).attr('id');

      var player = new JSROOT.TTreePlayer(itemname, url, askey, root_version);
      player.Show(divid);
      return player;
   }

   JSROOT.drawTreePlayerKey = function(hpainter, itemname) {
      // function used when tree is not yet loaded on the server

      return JSROOT.drawTreePlayer(hpainter, itemname, true);
   }


   // =======================================================================

   JSROOT.Painter.separ = null;

   JSROOT.Painter.AdjustLayout = function(left, height, firsttime) {
      if (this.separ == null) return;

      if (left!=null) {
         var wdiff = $("#"+this.separ.left).outerWidth() - $("#"+this.separ.left).width();
         var w = 5;
         $("#"+this.separ.vertical).css('left', left + "px").width(w).css('top','1px');
         $("#"+this.separ.left).width(left-wdiff-1).css('top','1px');
         $("#"+this.separ.right).css('left',left+w+"px").css('top','1px');
         if (!this.separ.horizontal) {
            $("#"+this.separ.vertical).css('bottom', '1px');
            $("#"+this.separ.left).css('bottom', '1px');
            $("#"+this.separ.right).css('bottom', '1px');
         }
      }

      if ((height!=null) && this.separ.horizontal)  {
         var diff = $("#"+this.separ.bottom).outerHeight() - $("#"+this.separ.bottom).height();
         height -= 2*diff;
         if (height<5) height = 5;
         var bot = height + diff;
         $('#'+this.separ.bottom).height(height);
         var h = 5;
         $("#"+this.separ.horizontal).css('bottom', bot + 'px').height(h);
         bot += h;
         $("#"+this.separ.left).css('bottom', bot + 'px');
      }

      if (this.separ.horizontal)
         if (this.separ.hpart) {
            var ww = $("#"+this.separ.left).outerWidth() - 2;
            $('#'+this.separ.bottom).width(ww);
            $("#"+this.separ.horizontal).width(ww);
         } else {
            var bot = $("#"+this.separ.left).css('bottom');
            $("#"+this.separ.vertical).css('bottom', bot);
            $("#"+this.separ.right).css('bottom', bot);
         }

      if (firsttime || (this.separ.handle==null)) return;

      if (typeof this.separ.handle == 'function') this.separ.handle(); else
      if ((typeof this.separ.handle == 'object') &&
          (typeof this.separ.handle.CheckResize === 'function')) this.separ.handle.CheckResize();
   }

   JSROOT.Painter.ConfigureVSeparator = function(handle) {

      JSROOT.Painter.separ = { handle: handle, left: "left-div", right: "right-div", vertical: "separator-div",
                               horizontal : null, bottom : null, hpart: true };

      $("#separator-div").addClass("separator").draggable({
         axis: "x" , zIndex: 100, cursor: "ew-resize",
         helper : function() { return $("#separator-div").clone().attr('id','separator-clone').css('background-color','grey'); },
         stop: function(event,ui) {
            event.stopPropagation();
            var left = ui.position.left;
            $("#separator-clone").remove();
            JSROOT.Painter.AdjustLayout(left, null, false);
         }
      });

      var w0 = Math.round($(window).width() * 0.2);
      if (w0<300) w0 = Math.min(300, Math.round($(window).width() * 0.5));

      JSROOT.Painter.AdjustLayout(w0, null, true);
   }

   JSROOT.Painter.ConfigureHSeparator = function(height, onlyleft) {

      if ((JSROOT.Painter.separ == null) ||
          (JSROOT.Painter.separ.horizontal != null)) return null;

      JSROOT.Painter.separ['horizontal'] = 'horizontal-separator-div';
      JSROOT.Painter.separ['bottom'] = 'bottom-div';
      JSROOT.Painter.separ.hpart = (onlyleft === true);

      var prnt = $("#"+this.separ.left).parent();

      prnt.append('<div id="horizontal-separator-div" class="separator" style="left:1px; right:1px;  height:4px; bottom:16px; cursor: ns-resize"></div>');
      prnt.append('<div id="bottom-div" class="column" style="left:1px; right:1px; height:15px; bottom:1px"></div>');

      $("#horizontal-separator-div").addClass("separator").draggable({
         axis: "y" , zIndex: 100, cursor: "ns-resize",
         helper : function() { return $("#horizontal-separator-div").clone().attr('id','horizontal-separator-clone').css('background-color','grey'); },
         stop: function(event,ui) {
            event.stopPropagation();
            var top = $(window).height() - ui.position.top;
            $('#horizontal-separator-clone').remove();
            JSROOT.Painter.AdjustLayout(null, top, false);
         }
      });

      JSROOT.Painter.AdjustLayout(null, height, false);

      return JSROOT.Painter.separ.bottom;
   }

   return JSROOT.Painter;

}));

