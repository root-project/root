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
      if (!menuname) menuname = "root_ctx_menu";

      var menu = { divid: menuname, code:"", cnt: 1, funcs : {}, separ : false };

      menu.add = function(name, arg, func) {
         if (name == "separator") { this.code += "<li>-</li>"; this.separ = true; return; }

         if (name.indexOf("header:")==0) {
            this.code += "<li class='ui-widget-header' style='padding-left:5px'>"+name.substr(7)+"</li>";
            return;
         }

         if (name=="endsub:") { this.code += "</ul></li>"; return; }
         var close_tag = "</li>";
         if (name.indexOf("sub:")==0) { name = name.substr(4); close_tag="<ul>"; }

         if (typeof arg == 'function') { func = arg; arg = name; }

         // if ((arg==null) || (typeof arg != 'string')) arg = name;

         if (name.indexOf("chk:")==0) { name = "<span class='ui-icon ui-icon-check'></span>" + name.substr(4); } else
         if (name.indexOf("unk:")==0) { name = "<span class='ui-icon ui-icon-blank'></span>" + name.substr(4); }

         // special handling of first versions with menu support
         if (($.ui.version.indexOf("1.10")==0) || ($.ui.version.indexOf("1.9")==0))
            name = '<a href="#">' + name + '</a>';

         this.code += "<li cnt='" + this.cnt + "' arg='" + arg + "'>" + name + close_tag;
         if (typeof func == 'function') this.funcs[this.cnt] = func; // keep call-back function

         this.cnt++;
      }

      menu.addchk = function(flag, name, arg, func) {
         return this.add((flag ? "chk:" : "unk:") + name, arg, func);
      }

      menu.size = function() { return this.cnt-1; }

      menu.addDrawMenu = function(menu_name, opts, call_back) {
         if (opts==null) opts = new Array;
         if (opts.length==0) opts.push("");

         this.add((opts.length > 1) ? ("sub:" + menu_name) : menu_name, opts[0], call_back);
         if (opts.length<2) return;

         for (var i=0;i<opts.length;++i) {
            var name = opts[i];
            if (name=="") name = '&lt;dflt&gt;';
            this.add(name, opts[i], call_back);
         }
         this.add("endsub:");
      }

      menu.remove = function() { $("#"+menuname).remove(); }

      menu.show = function(event) {
         menu.remove();

         document.body.onclick = function(e) { menu.remove(); }

         $(document.body).append('<ul id="' + menuname + '">' + this.code + '</ul>');

         $("#" + menuname)
            .css('left', event.clientX + window.pageXOffset)
            .css('top', event.clientY + window.pageYOffset)
            .attr('class', 'ctxmenu')
            .css('font-size', '80%')
            .css('position', 'absolute') // this overrides ui-menu-items class property
            .menu({
               items: "> :not(.ui-widget-header)",
               select: function( event, ui ) {
                  var arg = ui.item.attr('arg');
                  var cnt = ui.item.attr('cnt');
                  var func = cnt ? menu.funcs[cnt] : null;
                  menu.remove();
                  if (typeof func == 'function') {
                     if ('painter' in menu)
                        func.bind(menu['painter'])(arg); // if 'painter' field set, returned as this to callback
                     else
                        func(arg);
                  }
              }
         });

         var newx = null, newy = null;

         if (event.clientX + $("#" + menuname).width() > $(window).width()) newx = $(window).width() - $("#" + menuname).width() - 20;
         if (event.clientY + $("#" + menuname).height() > $(window).height()) newy = $(window).height() - $("#" + menuname).height() - 20;

         if (newx!=null) $("#" + menuname).css('left',(newx>0 ? newx : 0) + window.pageXOffset);
         if (newy!=null) $("#" + menuname).css('top',(newy>0 ? newy : 0) + window.pageYOffset);
      }

      JSROOT.CallBack(maincallback, menu);

      return menu;
   }

   JSROOT.HierarchyPainter.prototype.isLastSibling = function(hitem) {
      if (!hitem || !hitem._parent || !hitem._parent._childs) return false;
      var chlds = hitem._parent._childs;
      var indx = chlds.indexOf(hitem);
      if (indx<0) return false;
      while (++indx < chlds.length)
         if (!('_hidden' in chlds[indx])) return false;
      return true;
   }

   JSROOT.HierarchyPainter.prototype.addItemHtml = function(hitem, parent) {
      var isroot = (parent == null);
      var has_childs = '_childs' in hitem;
      if (!isroot) hitem._parent = parent;

      if ('_hidden' in hitem) return;

      var handle = JSROOT.getDrawHandle(hitem._kind);
      var img1 = "", img2 = "";
      var can_click = false;
      if (handle!=null) {
         if ('icon' in handle) img1 = handle.icon;
         if ('icon2' in handle) img2 = handle.icon2;
         if (('func' in handle) || ('execute' in handle) || ('aslink' in handle) || ('expand' in handle)) can_click = true;
      }
      if ('_icon' in hitem) img1 = hitem['_icon'];
      if ('_icon2' in hitem) img2 = hitem['_icon2'];
      if ((img1.length==0) && ('_online' in hitem))
         hitem['_icon'] = img1 = "img_globe";
      if ((img1.length==0) && isroot)
         hitem['_icon'] = img1 = "img_base";

      if (hitem['_more'] || ('_expand' in hitem) || ('_player' in hitem))
         can_click = true;

      if (img2.length==0) img2 = img1;
      if (img1.length==0) img1 = (has_childs || hitem['_more']) ? "img_folder" : "img_page";
      if (img2.length==0) img2 = (has_childs || hitem['_more']) ? "img_folderopen" : "img_page";

      var itemname = this.itemFullName(hitem);

      this['html'] += '<div item="' + itemname + '">';

      // build indent
      var sindent = "";
      var prnt = isroot ? null : hitem._parent;
      while ((prnt != null) && (prnt != this.h)) {
         sindent = '<div class="' + (this.isLastSibling(prnt) ? "img_empty" : "img_line") + '"/>' + sindent;
         prnt = prnt._parent;
      }
      this['html'] += sindent;

      var icon_class = "", plusminus = false;

      if (isroot) {
         // for root node no extra code
      } else
      if (has_childs) {
         icon_class = hitem._isopen ? "img_minus" : "img_plus";
         plusminus = true;
      } else
      /*if (hitem['_more']) {
         icon_class = "img_plus"; // should be special plus ???
         plusminus = true;
      } else */ {
         icon_class = "img_join";
      }

      if (icon_class.length > 0) {
         this['html'] += '<div class="' + icon_class;
         if (this.isLastSibling(hitem)) this['html'] += "bottom";
         if (plusminus) this['html'] += ' plus_minus" style="cursor:pointer';
         this['html'] += '"/>';
      }

      // make node icons

      if (this.with_icons) {
         var icon_name = hitem._isopen ? img2 : img1;

         if (icon_name.indexOf("img_")==0) {
            if ('_icon_click' in hitem) icon_name+= " icon_click";
            this['html'] += '<div class="' + icon_name + '" title="' + hitem._kind + '"/>';
         } else {
           this['html'] += '<img src="' + icon_name + '" alt="" style="vertical-align:top;width:18px;height:18px" title="' + hitem._kind +'"/>';
         }
      }

      this['html'] += '<a';
      if (can_click || has_childs) this['html'] +=' class="h_item"';

      var element_name = hitem._name;

      if ('_realname' in hitem)
         element_name = hitem._realname;

      var element_title = "";
      if ('_title' in hitem) element_title = hitem._title;

      if ('_fullname' in hitem)
         element_title += "  fullname: " + hitem['_fullname'];

      if (element_title.length == 0) element_title = element_name;

      this['html'] += ' title="' + element_title + '"';
      this['html'] += '>' + element_name + ('_value' in hitem ? ":" : "") + '</a>';
      if ('_value' in hitem) {
         this['html'] += "<p";
         if ('_vclass' in hitem) this['html'] += " class='" + hitem._vclass + "'";
         this['html'] += ">";
         if (!hitem['_isopen']) this['html'] += hitem._value;
         this['html'] += "</p>";
      }

      if (has_childs && (isroot || hitem._isopen)) {
         this['html'] += '<div class="h_childs">';
         for (var i in hitem._childs)
            this.addItemHtml(hitem._childs[i], hitem);
         this['html'] += '</div>';
      }

      this['html'] += '</div>';
   }

   JSROOT.HierarchyPainter.prototype.RefreshHtml = function(callback) {

      if (this.divid == null) return JSROOT.CallBack(callback);
      var d3elem = this.select_main();
      if ((this.h == null) || d3elem.empty()) {
         d3elem.html("");
         return JSROOT.CallBack(callback);
      }

      var elem = $(d3elem.node());

      var factcmds = [], status_item = null;
      this.ForEach(function(item) {
         if (('_fastcmd' in item) && (item._kind == 'Command')) factcmds.push(item);
         if (('_status' in item) && (status_item==null)) status_item = item;
      });

      this['html'] = "<div style='overflow:auto; width:100%; height:100%;"
      if (this.background) this['html']+="background-color:"+this.background + ";";
      if (this.with_icons) this['html']+="font-size:12px;";
                      else this['html']+="font-size:15px;";
      this['html']+="'>";
      if (factcmds.length>0) {
         for (var n in factcmds)
            this['html'] += "<button class='fast_command'> </button>";
      }
      this['html'] += "<p>";
      this['html'] += "<a href='#open_all'>open all</a>";
      this['html'] += "| <a href='#close_all'>close all</a>";
      if ('_online' in this.h)
         this['html'] += "| <a href='#reload'>reload</a>";
      else
         this['html'] += "<a/>";

      if ('disp_kind' in this)
         this['html'] += "| <a href='#clear'>clear</a>";
      else
         this['html'] += "<a/>";

      this['html'] += "</p>";

      this['html'] += '<div class="h_tree">';
      this.addItemHtml(this.h, null);
      this['html'] += '</div>';
      this['html'] += '</div>';

      var h = this;

      var items = elem.html(this['html'])
          .find(".h_item")
          .click(function() { h.tree_click($(this)); });

      if ('disp_kind' in h) {
         if (JSROOT.gStyle.DragAndDrop)
            items.draggable({ revert: "invalid", appendTo: "body", helper: "clone" });

         if (JSROOT.gStyle.ContextMenu)
            items.on('contextmenu', function(e) { h.tree_contextmenu($(this), e); });
      }

      // d3elem.selectAll(".plus_minus").on("click", function(d) { h.tree_click($(d),true); });
      elem.find(".plus_minus").click(function() { h.tree_click($(this), "plusminus"); });

      elem.find(".icon_click").click(function() { h.tree_click($(this), "icon"); });

      elem.find("a").first().click(function() { h.toggle(true); return false; })
                    .next().click(function() { h.toggle(false); return false; })
                    .next().click(function() { h.reload(); return false; })
                    .next().click(function() { h.clear(false); return false; });

      if (factcmds.length>0)
         elem.find('.fast_command').each(function(index) {
            if ('_icon' in factcmds[index])
               $(this).text("").append('<img src="' + factcmds[index]['_icon'] + '"/>');
            $(this).button()
                   .attr("item", h.itemFullName(factcmds[index]))
                   .attr("title", factcmds[index]._title)
                   .click(function() { h.ExecuteCommand($(this).attr("item"), $(this)); });
         });

      if ((status_item!=null) && (JSROOT.GetUrlOption('nostatus')==null)) {
         var func = JSROOT.findFunction(status_item._status);
         var hdiv = (typeof func == 'function') ? JSROOT.Painter.ConfigureHSeparator(30) : null;
         if (hdiv != null)
            func(hdiv, this.itemFullName(status_item));
      }

      JSROOT.CallBack(callback);
   }

   JSROOT.HierarchyPainter.prototype.UpdateTreeNode = function(hitem, node, set_attr) {
      if (node==null) {
         var name = this.itemFullName(hitem);
         node = $(this.select_main().node()).find("[item='" + name + "']");
         if ((node.length == 0) && ('_cycle' in hitem))
            node = $(this.select_main().node()).find("[item='" + name + ";" + hitem._cycle + "']");
         if (node.length == 0) return;
      }

      if (set_attr) {
         node.attr('item', this.itemFullName(hitem));
         node.find("a").text(hitem._name);
      }

      // better search for images again, but not store them extra
      var img1 = "", img2 = "", has_childs = ('_childs' in hitem);
      var handle = JSROOT.getDrawHandle(hitem._kind);
      if (handle!=null) {
         if ('icon' in handle) img1 = handle.icon;
         if ('icon2' in handle) img2 = handle.icon2;
      }
      if ('_icon' in hitem) img1 = hitem['_icon'];
      if ('_icon2' in hitem) img2 = hitem['_icon2'];
      if (img2.length==0) img2 = img1;
      if (img1.length==0) img1 = (has_childs || hitem['_more']) ? "img_folder" : "img_page";
      if (img2.length==0) img2 = (has_childs || hitem['_more']) ? "img_folderopen" : "img_page";

      var a_node = node.find("a").first();

      if ('_value' in hitem) {
         var p_node = a_node.next().html(hitem._isopen ? "" : hitem._value);
      }

      var img = a_node.prev();

      if (this.with_icons) {
         var newname = hitem._isopen ? img2 : img1;
         if (newname.indexOf("img_")==0) {
            if ('_icon_click' in hitem) newname += " icon_click";
            img.attr("class", newname);
         } else {
            img.attr("src", newname);
         }
         img = img.prev();
      }

      var h = this;

      var new_class = hitem._isopen ? "img_minus" : "img_plus";
      if (this.isLastSibling(hitem)) new_class += "bottom";

      if (img.hasClass("plus_minus")) {
         img.attr('class', new_class + " plus_minus");
      } else
      if (has_childs) {
         img.attr('class', new_class + " plus_minus");
         img.css('cursor', 'pointer');
         img.click(function() { h.tree_click($(this), "plusminus"); });
      }

      var childs = node.children().last();
      if (childs.hasClass("h_childs")) childs.remove();

      var display_childs = has_childs && hitem._isopen;
      if (!display_childs) return;

      this['html'] = '<div class="h_childs">';
      for (var i in hitem._childs)
         this.addItemHtml(hitem._childs[i], hitem);
      this['html'] += '</div>';
      node.append(this['html']);
      childs = node.children().last();

      var items = childs.find(".h_item")
                   .click(function() { h.tree_click($(this)); });

      if ('disp_kind' in h) {
         if (JSROOT.gStyle.DragAndDrop)
            items.draggable({ revert: "invalid", appendTo: "body", helper: "clone" });

         if (JSROOT.gStyle.ContextMenu)
            items.on('contextmenu', function(e) { h.tree_contextmenu($(this), e); })
      }

      childs.find(".plus_minus").click(function() { h.tree_click($(this), "plusminus"); });

      childs.find(".icon_click").click(function() { h.tree_click($(this), "icon"); });
   }

   JSROOT.HierarchyPainter.prototype.tree_click = function(node, place) {
      var itemname = node.parent().attr('item');

      if (itemname == null) return;

      var hitem = this.Find(itemname);
      if (hitem == null) return;

      if (!place || (place=="")) place = "item";

      if (place == "icon") {
         if (('_icon_click' in hitem) && (typeof hitem['_icon_click'] == 'function'))
            if (hitem['_icon_click'](hitem))
               this.UpdateTreeNode(hitem, node.parent());
         return;
      }

      // special feature - all items with '_expand' function are not drawn by click
      if ((place=="item") && ('_expand' in hitem)) place = "plusminus";

      // special case - one should expand item
      if ((place == "plusminus") && !('_childs' in hitem) && hitem['_more'])
         return this.expand(itemname, null, node.parent());

      if (place == "item") {
         if ('_player' in hitem)
            return this.player(itemname);

         var handle = JSROOT.getDrawHandle(hitem._kind);
         if (handle != null) {
            if ('aslink' in handle)
               return window.open(itemname + "/");

            if ('func' in handle)
               return this.display(itemname);

            if ('execute' in handle)
               return this.ExecuteCommand(itemname, node);

            if (('expand' in handle) && (hitem['_childs'] == null))
               return this.expand(itemname, null, node.parent());
         }

         if ((hitem['_childs'] == null))
            return this.expand(itemname, null, node.parent());

         if (!('_childs' in hitem) || (hitem === this.h)) return;
      }

      if (hitem._isopen)
         delete hitem._isopen;
      else
         hitem._isopen = true;

      this.UpdateTreeNode(hitem, node.parent());
   }

   JSROOT.HierarchyPainter.prototype.tree_contextmenu = function(node, event) {
      event.preventDefault();

      var itemname = node.parent().attr('item');

      var hitem = this.Find(itemname);
      if (hitem==null) return;

      var painter = this;

      var onlineprop = painter.GetOnlineProp(itemname);
      var fileprop = painter.GetFileProp(itemname);

      function qualifyURL(url) {
         function escapeHTML(s) {
            return s.split('&').join('&amp;').split('<').join('&lt;').split('"').join('&quot;');
         }
         var el = document.createElement('div');
         el.innerHTML = '<a href="' + escapeHTML(url) + '">x</a>';
         return el.firstChild.href;
      }

      JSROOT.Painter.createMenu(function(menu) {

         menu['painter'] = painter;

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
               addr += separ() + "layout=" + painter['disp_kind'].replace(/ /g, "");

            var items = [];

            if (painter['disp'] != null)
               painter['disp'].ForEachPainter(function(p) {
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
            var opts = JSROOT.getDrawOptions(hitem._kind, 'nosame');

            if (opts!=null)
               menu.addDrawMenu("Draw", opts, function(arg) { this.display(itemname, arg); });

            if ((fileprop!=null) && (opts!=null)) {
               var filepath = qualifyURL(fileprop.fileurl);
               if (filepath.indexOf(JSROOT.source_dir) == 0)
                  filepath = filepath.slice(JSROOT.source_dir.length);
               menu.addDrawMenu("Draw in new window", opts, function(arg) {
                  window.open(JSROOT.source_dir + "index.htm?nobrowser&file=" + filepath + "&item=" + fileprop.itemname+"&opt="+arg);
               });
            }

            if (!('_childs' in hitem) && (hitem['_more'] || !('_more' in hitem)))
               menu.add("Expand", function() { painter.expand(itemname); });
         }

         if (('_menu' in hitem) && (typeof hitem['_menu'] == 'function'))
            hitem['_menu'](menu, hitem, painter);

         if (menu.size() > 0) {
            menu['tree_node'] = node;
            if (menu.separ) menu.add("separator"); // add separator at the end
            menu.add("Close");
            menu.show(event);
         }

      }); // end menu creation

      return false;
   }

   JSROOT.HierarchyPainter.prototype.CreateDisplay = function(callback) {
      if ('disp' in this) {
         if (this['disp'].NumDraw() > 0) return JSROOT.CallBack(callback, this['disp']);
         this['disp'].Reset();
         delete this['disp'];
      }

      // check that we can found frame where drawing should be done
      if (document.getElementById(this['disp_frameid']) == null)
         return JSROOT.CallBack(callback, null);

      if (this['disp_kind'] == "tabs")
         this['disp'] = new JSROOT.TabsDisplay(this['disp_frameid']);
      else
      if ((this['disp_kind'] == "flex") || (this['disp_kind'] == "flexible"))
         this['disp'] = new JSROOT.FlexibleDisplay(this['disp_frameid']);
      else
      if (this['disp_kind'].search("grid") == 0)
         this['disp'] = new JSROOT.GridDisplay(this['disp_frameid'], this['disp_kind']);
      else
      if (this['disp_kind'] == "simple")
         this['disp'] = new JSROOT.SimpleDisplay(this['disp_frameid']);
      else
         this['disp'] = new JSROOT.CollapsibleDisplay(this['disp_frameid']);

      JSROOT.CallBack(callback, this['disp']);
   }

   JSROOT.HierarchyPainter.prototype.enable_dropping = function(frame, itemname) {
      var h = this;
      if (JSROOT.gStyle.DragAndDrop)
         $(frame).droppable({
            hoverClass : "ui-state-active",
            accept: function(ui) {
               var dropname = ui.parent().attr('item');
               if ((dropname == itemname) || (dropname==null)) return false;

               var ditem = h.Find(dropname);
               if ((ditem==null) || (!('_kind' in ditem))) return false;

               return ditem._kind.indexOf("ROOT.")==0;
            },
            drop: function(event, ui) {
               var dropname = ui.draggable.parent().attr('item');
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

   JSROOT.CollapsibleDisplay.prototype.ActivateFrame = function(frame) {
      if ($(frame).is(":hidden")) {
         $(frame).prev().toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
                 .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
                 .next().toggleClass("ui-accordion-content-active").slideDown(0);
      }
      $(frame).prev()[0].scrollIntoView();
   }

   JSROOT.CollapsibleDisplay.prototype.CreateFrame = function(title) {

      var topid = this.frameid + '_collapsible';

      if (document.getElementById(topid) == null)
         $("#"+this.frameid).append('<div id="'+ topid  + '" class="ui-accordion ui-accordion-icons ui-widget ui-helper-reset" style="overflow:auto; overflow-y:scroll; height:100%; padding-left: 2px; padding-right: 2px"></div>');

      var hid = topid + "_sub" + this.cnt++;
      var uid = hid + "h";

      var entryInfo = "<h5 id=\"" + uid + "\">" +
                        "<span class='ui-icon ui-icon-triangle-1-e'></span>" +
                        "<a> " + title + "</a>&nbsp; " +
                        "<button type='button' class='jsroot_collaps_closebtn' style='float:right; width:1.4em' title='close canvas'/>" +
                        " </h5>\n";
      entryInfo += "<div class='collapsible_draw' id='" + hid + "'></div>\n";
      $("#" + topid).append(entryInfo);

      var pthis = this;

      $('#' + uid)
            .addClass("ui-accordion-header ui-helper-reset ui-state-default ui-corner-top ui-corner-bottom")
            .hover(function() { $(this).toggleClass("ui-state-hover"); })
            .click( function() {
                     $(this).toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
                           .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s")
                           .end().next().toggleClass("ui-accordion-content-active").slideToggle(0);
                     pthis.CheckResize($(this).next().attr('id'));
                     return false;
                  })
            .next()
            .addClass("ui-accordion-content ui-helper-reset ui-widget-content ui-corner-bottom")
            .hide();

      $('#' + uid).find(" .jsroot_collaps_closebtn")
           .button({ icons: { primary: "ui-icon-close" }, text: false })
           .click(function(){ $(this).parent().next().andSelf().remove(); });


      $('#' + uid)
            .toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
            .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end().next()
            .toggleClass("ui-accordion-content-active").slideToggle(0);

      $("#" + hid).prop('title', title);

      return $("#" + hid).get(0);
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

   JSROOT.TabsDisplay.prototype.ActivateFrame = function(frame) {
      var cnt = 0, id = -1;
      this.ForEachFrame(function(fr) {
         if ($(fr).attr('id') == $(frame).attr('id')) id = cnt;
         cnt++;
      });
      $('#' + this.frameid + "_tabs").tabs("option", "active", id);
   }

   JSROOT.TabsDisplay.prototype.CreateFrame = function(title) {
      var topid = this.frameid + '_tabs';

      var hid = topid + "_sub" + this.cnt++;

      var li = '<li><a href="#' + hid + '">' + title
            + '</a><span class="ui-icon ui-icon-close" style="float: left; margin: 0.4em 0.2em 0 0; cursor: pointer;" role="presentation">Remove Tab</span></li>';
      var cont = '<div class="tabs_draw" id="' + hid + '"></div>';

      var pthis = this;

      if (document.getElementById(topid) == null) {
         $("#" + this.frameid).append('<div id="' + topid + '">' + ' <ul>' + li + ' </ul>' + cont + '</div>');

         var tabs = $("#" + topid)
                       .css('overflow','hidden')
                       .tabs({
                          heightStyle : "fill",
                          activate : function (event,ui) {
                             $(ui.newPanel).css('overflow', 'hidden');
                             pthis.CheckResize($(ui.newPanel).attr('id'));
                           }
                          });

         tabs.delegate("span.ui-icon-close", "click", function() {
            var panelId = $(this).closest("li").remove().attr("aria-controls");
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
         .prop('title', title);

      return $('#' + hid).get(0);
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
         //if (only_visible && $(this).is(":hidden")) return;

         userfunc($(this).get(0));
      });
   }

   JSROOT.FlexibleDisplay.prototype.ActivateFrame = function(frame) {
   }

   JSROOT.FlexibleDisplay.prototype.CreateFrame = function(title) {
      var topid = this.frameid + '_flex';

      if (document.getElementById(topid) == null)
         $("#" + this.frameid).append('<div id="'+ topid  + '" style="overflow:none; height:100%; width:100%"></div>');

      var top = $("#" + topid);

      var w = top.width(), h = top.height();

      var subid = topid + "_frame" + this.cnt;

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
               main.height(main.find('.flex_header').height())
                   .width("auto")
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

         if (state!="minimal")
            JSROOT.CheckElementResize(main.find(".flex_draw").get(0));
      }

      $("#" + subid)
         .css('left', parseInt(w * (this.cnt % 5)/10))
         .css('top', parseInt(h * (this.cnt % 5)/10))
         .width(Math.round(w * 0.58))
         .height(Math.round(h * 0.58))
         .resizable({
            helper: "jsroot-resizable-helper",
            start: function(event, ui) {
               // bring element to front when start resizing
               $(this).appendTo($(this).parent());
            },
            stop: function(event, ui) {
               var rect = { width : ui.size.width-1, height : ui.size.height - $(this).find(".flex_header").height()-1 };
               JSROOT.CheckElementResize($(this).find(".flex_draw").get(0), rect);
            }
          })
          .draggable({
            containment: "parent",
            start: function(event, ui) {
               // bring element to front when start dragging
               $(this).appendTo($(this).parent());
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
           .click(function() { $(this).parent().parent().remove(); })
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

      return $("#" + subid + "_cont").prop('title', title).get(0);
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

      this.CheckResize();

      this.SetDivId(divid);
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

   JSROOT.TTreePlayer.prototype.CheckResize = function(force) {
      var main = $(this.select_main().node());

      $("#" + this.drawid).width(main.width());
      var h = main.height();
      var h0 = main.find(".treedraw_buttons").height();
      if (h>h0+30) $("#" + this.drawid).height(h - 1 - h0);

      if (this.hist_painter) {
         this.hist_painter.CheckResize(force);
      }
   }

   JSROOT.drawTreePlayer = function(hpainter, itemname, askey) {

      var url = hpainter.GetOnlineItemUrl(itemname);
      if (url == null) return null;

      var top = hpainter.GetTopOnlineItem(hpainter.Find(itemname));
      if (top == null) return null;
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
         var w = JSROOT.touches ? 10 : 4;
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
         var h = JSROOT.touches ? 10 : 4;
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
          (typeof this.separ.handle['CheckResize'] == 'function')) this.separ.handle.CheckResize();
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

