/// @file JSRootPainter.jquery.js
/// Part of JavaScript ROOT graphics, dependent from jQuery functionality

(function() {

   if (typeof JSROOT != 'object') {
      var e1 = new Error('JSROOT is not defined');
      e1.source = 'JSRootPainter.jquery.js';
      throw e1;
   }

   if (typeof d3 != 'object') {
      var e1 = new Error('This extension requires d3.v3.js');
      e1.source = 'JSRootPainter.jquery.js';
      throw e1;
   }

   if (typeof JSROOT.Painter != 'object') {
      var e1 = new Error('JSROOT.Painter not defined');
      e1.source = 'JSRootPainter.jquery.js';
      throw e1;
   }

   if (typeof jQuery == 'undefined') {
      var e1 = new Error('jQuery not defined ');
      e1.source = 'JSRootPainter.jquery.js';
      throw e1;
   }

   JSROOT.Painter.createMenu = function(maincallback, menuname) {
      if (!menuname) menuname = "root_ctx_menu";

      var menu = { divid: menuname, code:"", cnt: 1, funcs : {} };

      menu.add = function(name, arg, func) {
         if (name.indexOf("header:")==0) {
            this.code += "<li class='ui-widget-header'>"+name.substr(7)+"</li>";
            return;
         }

         if (name=="endsub:") { this.code += "</ul></li>"; return; }
         var close_tag = "</li>";
         if (name.indexOf("sub:")==0) { name = name.substr(4); close_tag="<ul>"; }

         if (typeof arg == 'function') { func = arg; arg = name; }

         if ((arg==null) || (typeof arg != 'string')) arg = name;

         // special handling of first versions with menu support
         if (($.ui.version.indexOf("1.10")==0) || ($.ui.version.indexOf("1.9")==0))
            name = '<a href="#">' + name + '</a>';

         this.code += "<li cnt='" + this.cnt + "' arg='" + arg + "'>" + name + close_tag;
         if (typeof func == 'function') this.funcs[this.cnt] = func; // keep call-back function

         this.cnt++;
      }

      menu.size = function() { return this.cnt-1; }

      menu.addDrawMenu = function(menu_name, opts, call_back) {
         if (opts==null) opts = new Array;
         if (opts.length==0) opts.push("");

         this.add((opts.length > 1) ? ("sub:" + menu_name) : menu_name, opts[0], call_back);
         if (opts.length<2) return;

         for (var i=0;i<opts.length;i++) {
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
            .menu({
               items: "> :not(.ui-widget-header)",
               select: function( event, ui ) {
                  var arg = ui.item.attr('arg');
                  var cnt = ui.item.attr('cnt');
                  var func = cnt ? menu.funcs[cnt] : null;
                  menu.remove();
                  if (typeof func == 'function') func(arg);
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

      if ('_hidden' in hitem) return;

      var cando = this.CheckCanDo(hitem);

      if (!isroot) hitem._parent = parent;

      var can_click = false;

      if (!has_childs || !cando.scan) {
         if (cando.expand) {
            can_click = true;
            if (cando.img1.length == 0) {
               cando.img1 = 'img_folder';
               cando.img2 = 'img_folderopen';
            }
         } else
         if (cando.display || cando.execute) {
            can_click = true;
         } else
         if (cando.html.length > 0) can_click = true;
      }

      hitem['_img1'] = cando.img1;
      hitem['_img2'] = cando.img2;
      if (hitem['_img2']=="") hitem['_img2'] = hitem['_img1'];

      // assign root icon
      if (isroot && (hitem['_img1'].length==0))
         hitem['_img1'] = hitem['_img2'] = "img_base";

      // assign node icons

      if (hitem['_img1'].length==0)
         hitem['_img1'] = has_childs ? "img_folder" : "img_page";

      if (hitem['_img2'].length==0)
         hitem['_img2'] = has_childs ? "img_folderopen" : "img_page";

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
      } else {
         icon_class = "img_join";
      }

      if (icon_class.length > 0) {
         this['html'] += '<div class="' + icon_class;
         if (this.isLastSibling(hitem)) this['html'] += "bottom";
         if (plusminus) this['html'] += ' plus_minus" style="cursor:pointer';
         this['html'] += '"/>';
      }

      // make node icons

      var icon_name = hitem._isopen ? hitem._img2 : hitem._img1;

      if (icon_name.indexOf("img_")==0)
         this['html'] += '<div class="' + icon_name + '"/>';
      else
         this['html'] += '<img src="' + icon_name + '" alt="" style="vertical-align:top;width:18px;height:18px"/>';

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
      this['html'] += '>' + element_name + '</a>';

      if (has_childs && (isroot || hitem._isopen)) {
         this['html'] += '<div class="h_childs">';
         for (var i in hitem._childs)
            this.addItemHtml(hitem._childs[i], hitem);
         this['html'] += '</div>';
      }

      this['html'] += '</div>';
   }

   JSROOT.HierarchyPainter.prototype.RefreshHtml = function(callback) {

      if (this.frameid == null) return JSROOT.CallBack(callback);
      var elem = $("#" + this.frameid);
      if ((this.h == null) || (elem.length == 0)) {
         elem.html("");
         return JSROOT.CallBack(callback);
      }

      var factcmds = [], status_item = null;
      this.ForEach(function(item) {
         if (('_fastcmd' in item) && (item._kind == 'Command')) factcmds.push(item);
         if (('_status' in item) && (status_item==null)) status_item = item;
      });

      this['html'] = "";
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

      elem.find(".plus_minus").click(function() { h.tree_click($(this),true); });

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

   JSROOT.HierarchyPainter.prototype.UpdateTreeNode = function(node, hitem) {
      var has_childs = '_childs' in hitem;

      var newname = hitem._isopen ? hitem._img2 : hitem._img1;
      var oldname = hitem._isopen ? hitem._img1 : hitem._img2;

      var img = node.find("a").first().prev();

      if (newname.indexOf("img_")<0) {
         img.attr("src", newname);
      } else {
         if (newname!=oldname)
            img.switchClass(oldname, newname);
      }

      img = img.prev();

      var h = this;

      var new_class = hitem._isopen ? "img_minus" : "img_plus";
      if (this.isLastSibling(hitem)) new_class += "bottom";

      if (img.hasClass("plus_minus")) {
         img.attr('class', new_class + " plus_minus");
      } else
      if (has_childs) {
         img.attr('class', new_class + " plus_minus");
         img.css('cursor', 'pointer');
         img.click(function() { h.tree_click($(this), true); });
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

      childs.find(".plus_minus").click(function() { h.tree_click($(this), true); });
   }

   JSROOT.HierarchyPainter.prototype.tree_click = function(node, plusminus) {
      var itemname = node.parent().attr('item');

      if (itemname==null) return;

      var hitem = this.Find(itemname);
      if (hitem==null) return;

      if (!plusminus) {
         var cando = this.CheckCanDo(hitem);

         if (cando.open && (cando.html.length>0))
            return window.open(cando.html);

         if (cando.expand && (hitem['_childs'] == null))
            return this.expand(itemname, hitem, node.parent());

         if (cando.display)
            return this.display(itemname);

         if ('_player' in hitem)
            return this.player(itemname);

         if (cando.execute)
            return this.ExecuteCommand(itemname, node);

         if (!('_childs' in hitem) || (hitem === this.h)) return;
      }

      if (hitem._isopen)
         delete hitem._isopen;
      else
         hitem._isopen = true;

      this.UpdateTreeNode(node.parent(), hitem);
   }

   JSROOT.HierarchyPainter.prototype.tree_contextmenu = function(node, event) {
      event.preventDefault();

      var itemname = node.parent().attr('item');

      var hitem = this.Find(itemname);
      if (hitem==null) return;

      var cando = this.CheckCanDo(hitem);

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

         if (itemname == "") {
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
         } else
         if (fileprop != null) {
            var opts = JSROOT.getDrawOptions(cando.typename, 'nosame');

            menu.addDrawMenu("Draw", opts, function(arg) { painter.display(itemname, arg); });

            var filepath = qualifyURL(fileprop.fileurl);
            if (filepath.indexOf(JSROOT.source_dir) == 0)
               filepath = filepath.slice(JSROOT.source_dir.length);

            menu.addDrawMenu("Draw in new window", opts, function(arg) {
               window.open(JSROOT.source_dir + "index.htm?nobrowser&file=" + filepath + "&item=" + fileprop.itemname+"&opt="+arg);
            });
         }

         if (menu.size()>0) {
            menu['tree_node'] = node;
            menu.add("Close");
            menu.show(event);
         }

      }); // end menu creation

      return false;
   }

   JSROOT.HierarchyPainter.prototype.expand = function(itemname, item0, node) {
      var painter = this;

      if (node==null)
         node = $("#" + this.frameid).find("[item='" + itemname + "']");

      if (node.length==0)
         return JSROOT.console("Did not found node with item = " + itemname);

      if (item0==null) item0 = this.Find(itemname);
      if (item0==null) return;
      item0['_doing_expand'] = true;

      this.get(itemname, function(item, obj) {
         delete item0['_doing_expand'];
         if ((item == null) || (obj == null)) return;

         var curr = item;
         while (curr != null) {
            if (('_expand' in curr) && (typeof (curr['_expand']) == 'function')) {
                if (curr['_expand'](item, obj)) {
                   var itemname = painter.itemFullName(item);
                   node.attr('item', itemname);
                   node.find("a").text(item._name);
                   item._isopen = true;
                   painter.UpdateTreeNode(node, item);
                }
                return;
            }
            curr = ('_parent' in curr) ? curr['_parent'] : null;
         }
      });
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
         $("#right-div").append('<div id="'+ topid  + '" class="ui-accordion ui-accordion-icons ui-widget ui-helper-reset" style="overflow:auto; overflow-y:scroll; height:100%; padding-left: 2px; padding-right: 2px"></div>');

      var hid = topid + "_sub" + this.cnt++;
      var uid = hid + "h";

      var entryInfo = "<h5 id=\"" + uid + "\"><a> " + title + "</a>&nbsp; </h5>\n";
      entryInfo += "<div class='collapsible_draw' id='" + hid + "'></div>\n";
      $("#" + topid).append(entryInfo);

      var pthis = this;

      $('#' + uid)
            .addClass("ui-accordion-header ui-helper-reset ui-state-default ui-corner-top ui-corner-bottom")
            .hover(function() { $(this).toggleClass("ui-state-hover"); })
            .prepend('<span class="ui-icon ui-icon-triangle-1-e"></span>')
            .append('<button type="button" class="closeButton" title="close canvas" '+
                    'onclick="javascript: $(this).parent().next().andSelf().remove();">'+
                    '<img class="img_remove" src="" alt=""/></button>')
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
            + '</a><span class="ui-icon ui-icon-close" role="presentation">Remove Tab</span></li>';
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

   // ========== performs tree drawing on server ==================

   JSROOT.TTreePlayer = function(itemname) {
      JSROOT.TBasePainter.call(this);
      this.SetItemName(itemname);
      this.hpainter = null;
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

      var frame = $("#" + this.divid);

      var url = this.GetItemName() + '/exe.json.gz?compact=3&method=Draw';
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
            if (nentries=="") nentries = "1000000000";
            if (firstentry=="") firstentry = "0";
            url += '&option="' + option + '"&nentries=' + nentries + '&firstentry=' + firstentry;
         }
      } else {
         url += '&prototype="Option_t*"&opt="' + expr + '"';
      }
      url += '&_ret_object_=' + hname;

      var player = this;

      var req = JSROOT.NewHttpRequest(url, 'object', function(res) {
         if (res==0) return;
         $("#"+player.drawid).empty();
         player.hpainter = JSROOT.draw(player.drawid, res)
      });
      req.send();
   }

   JSROOT.TTreePlayer.prototype.CheckResize = function(force) {
      $("#" + this.drawid).width($("#" + this.divid).width());
      var h = $("#" + this.divid).height();
      var h0 = $("#" + this.divid +" .treedraw_buttons").height();
      if (h>h0+30) $("#" + this.drawid).height(h - 1 - h0);

      if (this.hpainter) {
         this.hpainter.CheckResize(force);
      }
   }

   JSROOT.drawTreePlayer = function(hpainter, itemname) {
      var mdi = hpainter.GetDisplay();
      if (mdi == null) return null;

      var frame = mdi.FindFrame(itemname, true);
      if (frame==null) return null;

      var divid = d3.select(frame).attr('id');

      var player = new JSROOT.TTreePlayer(itemname);
      player.Show(divid);
      return player;
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


})();
