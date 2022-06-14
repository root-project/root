/**
 * This is submodule produces visualizations for decision trees. The visualization is interactive, and it is made
 * with d3js.
 * Interactions supported:
 *    - Mouseover (node, weight): showing decision path
 *    - Zooming and grab and move supported
 *    - Reset zoomed tree: double click
 *    - Expand all closed subtrees, turn off zoom: button in the bottom of the picture
 *    - Click on node:
 *                    * hiding subtree, if node children are hidden the node will have a green border
 *                    * rescaling: bigger nodes, bigger texts
 *                    * click again to show the subtree
 * Author: Attila Bagoly <battila93@gmail.com>
 * Created:  6/11/16
 */


(function(factory){
    var url = "";
    if (require.s.contexts.hasOwnProperty("_")) {
        url = require.s.contexts._.config.paths["JsMVA"].replace("src/js/JsMVA.min","");
    }

    define(["d3"], function(d3){
        return factory({}, d3, url);
    });

})(function(DecisionTree, d3, url){

    Object.size = function(obj){
        return Object.keys(obj).length;
    };

    var style = {
        margin: {
            x: 20,
            y: 20
        },
        node: {
            padding: 10,
            yspace: 40,
            xspace: 20,
            width: 150,
            height: 40,
            mwidth: 150,
            mheight: 60,
            colors: {
                focus: "#033A00",
                closed: "#00A62B",
                pureBkg: "red",
                pureSig: "blue"
            },
            swidth: "4px"
        },
        link: {
            colors:{
                "default": "#CCC",
                focus: "#033A00"
            },
            width: "4px",
            focus_width: "8px"
        },
        aduration: 1500,
        legend: {
            size: 20,
            rect_width: 100,
            rect_height: 30,
            rect_fucus_width: 115
        },
        text: {
            color: "#DEDEDE",
            padding: "4px"
        },
        buttons:{
            reset:{
                width: "36px",
                height: "36px",
                alpha: "0.5",
                img: url+"img/reset.png",
                background: "white"
            }
        }
    };


    var nodeColor = d3.scale.linear()
        .range([style.node.colors.pureBkg, style.node.colors.pureSig]);

    var canvas, svg, roottree, variables;

    var d3tree = d3.layout.tree();
    var d3path = (function(){
        var diagonal = d3.svg.diagonal();
        var forEachData = function(d, i, hidden){
            if (hidden) return diagonal(d);
            return diagonal({
                source: {
                    x: (d.source.x + style.node.width),
                    y: d.source.y
                },
                target: {
                    x: (d.target.x + style.node.width),
                    y:  d.target.y - style.node.height
                }
            });
        };
        return forEachData;
    })();

    var clickOnNode = function(d){
        if ("children" in d){
            d._children = d.children;
            d.children = null;
        } else {
            d.children = d._children;
            d._children = null;
        }
        drawTree(d);
    };

    var drawLabels = function(nodeContainer) {
        var height = style.node.height;
        nodeContainer.append("text")
            .attr("dy", height* 0.35)
            .attr("class", "label1")
            .attr("dx", style.text.padding)
            .style("fill-opacity", 1e-6)
            .style("font-size", 1e-6+"px")
            .style("cursor", "pointer")
            .style("fill", style.text.color)
            .style("font-weight", "bold")
            .text(function (d) {
                return "S/(S+B)=" + Number(d.info.purity).toFixed(3);
            });
        nodeContainer.append("text")
            .attr("class", "label2")
            .attr("dx", style.text.padding)
            .attr("dy", height*0.75)
            .style("fill-opacity", 1e-6)
            .style("cursor", "pointer")
            .text(function(d){
                return d.info.IVar!=-1
                    ? variables[d.info.IVar]+">"+(Number(d.info.Cut).toFixed(3))
                    : "";
            })
            .style("font-size", 1e-6+"px")
            .style("fill", style.text.color)
            .style("font-weight", "bold");
    };

    var drawNodes = function(nodeSelector, father){
        var nodeContainer = nodeSelector.enter()
            .append("g").attr("class", "nodes")
            .attr("transform", function(d){return "translate("+father.x0+","+father.y0+")";})
            .style("cursor","pointer");

        nodeContainer.filter(function(d){
                return d.parent;
            })
            .on("click", clickOnNode)
            .on("mouseover", path)
            .on("contextmenu", function(d, i){
                d3.event.preventDefault();
                makePathNodesBigger(d);
            })
            .on("mouseleave", function(d, i){
                if (d.bigger) makePathNodesBigger(d, i, 1);
                return path(d, i, 1);
            });

        nodeContainer.append("rect")
            .attr("width", 1e-6)
            .attr("height", 1e-6);

        drawLabels(nodeContainer);

        nodeSelector.transition().duration(style.aduration)
            .attr("transform", function(d){
                return "translate("
                    + (d.x+style.node.width*0.5) + ","
                    + (d.y-style.node.height) + ")";
            });

        nodeSelector.select("rect").transition().duration(style.aduration)
            .attr("width", style.node.width)
            .attr("height", style.node.height)
            .attr("fill", function(d){return nodeColor(Number(d.info.purity));})
            .style("stroke-width", style.node.swidth)
            .style("stroke", function(d){
                return (d._children) ? style.node.colors.closed : "";
            });

        nodeSelector.selectAll("text")
            .transition().duration(style.aduration)
            .style("font-size", function(d) {
                var l1 = "S/(S+B)=" + Number(d.info.purity).toFixed(3);
                var l2 = d.info.IVar!=-1
                    ? variables[d.info.IVar]+">"+(Number(d.info.Cut).toFixed(3))
                    : "";
                d.font_size = 1.5*(style.node.width-2*Number(style.node.swidth.replace("px","")))/Math.max(l1.length, l2.length);
                return d.font_size+"px";
            })
            .attr("dx", style.text.padding)
            .attr("dy", function(d){
                return ((d3.select(this).attr("class")=="label1")? (style.node.height * 0.35) : (style.node.height * 0.75))+"px"; })
            .style("fill-opacity", 1);

        var nodeExit = nodeSelector.exit()
            .transition().duration(style.aduration)
            .attr("transform", function(d){
                return "translate("
                    + (father.x+style.node.width) + ","
                    + father.y + ")";
            })
            .remove();

        nodeExit.select("rect")
            .attr("width", 1e-6)
            .attr("height", 1e-6);

        nodeExit.selectAll("text")
            .style("font-size", 1e-6+"px")
            .style("fill-opacity", 1e-6);
    };

    var drawLinks = function(linkSelector, father){
        linkSelector.enter()
            .insert("path", "g")
            .attr("class", "link")
            .attr("d", function(d, i){
                var o = {x:father.x0, y:father.y0};
                return d3path({source: o, target: o},i, 1);
            });

        linkSelector.transition().duration(style.aduration)
            .attr("d", d3path)
            .style("fill", "none")
            .style("stroke", style.link.colors["default"])
            .style("stroke-width", style.link.width)
            .attr("id", function(d){return "link"+d.target.id;});

        linkSelector.exit()
            .transition().duration(style.aduration)
            .attr("d", function(d, i){
                var o = {x:father.x+style.node.width, y:father.y};
                return d3path({source:o, target:o},i, 1);
            })
            .remove();
    };

    var path = function(node, i, clear){
        svg.selectAll("path.link").filter(function(d){return d.target.id==node.id;})
            .style("stroke-width", (clear) ? style.link.width : style.link.focus_width)
            .style("stroke", (clear) ? style.link.colors["default"] : style.link.colors.focus);

        svg.selectAll("g.nodes rect").filter(function(d){return d.id==node.id;})
            .style("stroke-width", style.node.swidth)
            .style("stroke", function(d){
                return (clear)
                    ? (d._children) ? style.node.colors.closed : nodeColor(d.info.purity)
                    : style.node.colors.focus;
            });

        if (node.parent) path(node.parent, i, clear);
    };

    var makePathNodesBigger = function(node, i, clear){
        var width = (clear) ? style.node.width : 2*style.node.width,
            height = (clear) ? style.node.height : 1.5*style.node.height;
        console.log("anim height:"+String(height));
        svg.selectAll("g.nodes rect").filter(function(d){d.bigger=(clear) ? false : true; return d.id==node.id;})
            .transition().duration(style.aduration/2)
            .attr("width", width+"px")
            .attr("height", height+"px");

        svg.selectAll("g.nodes text").filter(function(d){return d.id==node.id;})
            .transition().duration(style.aduration/2)
            .style("font-size", function(d){
                return ((clear) ? d.font_size : 2*d.font_size)+"px";
            })
            .attr("dx", (clear) ? style.text.padding : (2*Number(style.text.padding.replace("px", ""))+"px"))
            .attr("dy", function(){
                return ((d3.select(this).attr("class")=="label1")? (height * 0.35) : (height * 0.75))+"px";
            });
        if (node.parent) makePathNodesBigger(node.parent, i, clear);
    };

    var drawTree = function(father, nox0y0Calc){
        updateSizesColors();
        var nodes = d3tree.nodes(roottree),
            links = d3tree.links(nodes);

        var maxDepth = 0;
        nodes.forEach(function(d){
           if (maxDepth<Number(d.depth)) maxDepth = Number(d.depth);
        });

        nodes.forEach(function(d){
           d.y = d.depth * canvas.height / maxDepth;
        });

        if (!("x0" in father) || !("y0" in father)){
            father.x0 = nodes[0].x+style.node.width*0.5;
            father.y0 = nodes[0].y-style.node.height;
        }

        var nodeSelector = svg.selectAll("g.nodes")
            .data(nodes, function(d, i ){return d.id || (d.id =  i);});

        drawNodes(nodeSelector, father);

        var linkSelector = svg.selectAll("path.link")
            .data(links, function(d){return d.target.id;});

        drawLinks(linkSelector, father);

        svg.transition().duration(style.aduration).attr("transform", "translate("+(-style.node.width)+", "+style.node.height+")");
        if (nox0y0Calc!==undefined && nox0y0Calc==true) return;
        nodes.forEach(function(d){
            d.x0 = d.x+style.node.width;
            d.y0 = d.y;
        });
    };

    var treeHeight = function(tree){
        if (tree.length!==undefined){
            var sum = 0;
            for(var i in tree){
                sum += treeHeight(tree[i]);
            }
            return sum;
        }
        if (!("children" in tree) || tree["children"].length==0) return 0;
        return 1+treeHeight(tree["children"]);
    };

    var treeWidth = function(nodes){
        var posxs = Array();
        for (var i in nodes) {
            if (posxs.indexOf(Math.round(nodes[i].x))==-1){
                posxs.push(Math.round(nodes[i].x));
            }
        }
        return posxs.length+1;
    };

    var purityToColor = function(nodes){
        var min = 1e6,
            max = -1e6;
        var pur;
        for (var i in nodes) {
            pur = Number(nodes[i].info.purity);
            if (pur<min) min = pur;
            if (pur>max) max = pur;
        }
        return [min, max];
    };

    var updateSizesColors = function(first){
        var nodes = d3tree.nodes(roottree);
        var tree;
        for(var i in nodes){
            if (!nodes[i].parent){
                tree = nodes[i];
                break;
            }
        }
        var height = treeHeight(tree);

        style.node.height = canvas.height/(height+1)-style.node.yspace;
        if (style.node.height>style.node.mheight) style.node.height = style.node.mheight;
        var corr = 0;
        while(height!=0){
            if (!(height%4)) corr++;
            height /= 4;
        }
        style.node.width  = canvas.width/(treeWidth(nodes)+1-corr)-style.node.xspace;
        if (style.node.width>style.node.mwidth) style.node.height = style.node.mwidth;

        d3tree.size([canvas.width, canvas.height]);

        nodeColor.domain(purityToColor(nodes));
    };

    var drawLegend = function(svgOriginal){
        var labels = [
            {text: "Pure Backg.", id: "label1", color: nodeColor(nodeColor.domain()[0]), x:5,y:5},
            {text: "Pure Signal", id: "label2", color: nodeColor(nodeColor.domain()[1]), x:5,y:40}
        ];
        var legend = svgOriginal.append("g")
            .attr("transform", "translate(5,5)");

        var group = legend.selectAll("g")
            .data(labels, function(d){return d.id;})
            .enter()
            .append("g")
            .style("cursor", "pointer")
            .attr("transform", function(d){return "translate("+d.x+","+d.y+")";});

        group.on("mouseover", function(d){
            d3.select("#"+d.id).style("font-weight", "bold");
            d3.select("#"+d.id+"_rect").attr("width", style.legend.rect_fucus_width);
        });
        group.on("mouseout", function(d){
            d3.select("#"+d.id).style("font-weight", "normal");
            d3.select("#"+d.id+"_rect").attr("width", style.legend.rect_width);

        });

        group.append("rect")
            .attr("id", function(d){return d.id+"_rect";})
            .attr("width", style.legend.rect_width)
            .attr("height", style.legend.rect_height)
            .attr("fill", function(d){return d.color;});

        group.append("text")
            .attr("id", function(d){return d.id;})
            .attr("x", function(d){return 5;})
            .attr("y", function(d){return 20;})
            .text(function(d){return d.text;})
            .style("fill", style.text.color);
    };

    var openSubTree = function(node){
        if ("_children" in node && node.children==null){
            node.children = node._children;
            node._children = null;
        }
        if ("children" in node && node.children!=null){
            for(var i in node.children){
                openSubTree(node.children[i]);
            }
        }
    };

    var findFathers = function(start){
        var fathers = [];
        var Q = [];
        Q.push(start);
        while(Q.length>0){
            var node = Q.shift();
            if ("_children" in node && node._children!=null){
                fathers.push(node);
            }
            if (node.children!=null) {
                for (var i = 0; i < node.children.length; i++) {
                    Q.push(node.children[i]);
                }
            }
        }
        return fathers.length>0 ? fathers : start;
    };


    DecisionTree.draw = function(divid, pyobj){
        var div = d3.select("#"+divid);

        roottree  = pyobj["tree"];
        variables = pyobj["variables"];

        if (Object.size(roottree)==0){
            div.innerHTML = "<b style='color:red'>Tree empty...</b>";
            return;
        }

        canvas = {
            width:  div.property("style")["width"],
            height: div.property("style")["height"]
        };

        svg = div.append("svg")
            .attr("width", canvas.width)
            .attr("height", canvas.height);
        var svgOriginal = svg;
        Object.keys(canvas).forEach(function (key) {
            canvas[key] = Number(canvas[key].replace("px",""));
            canvas[key] -= key=="width" ? 2*style.margin.x+style.node.width : 2*style.margin.y+style.node.height;
        });

        updateSizesColors(1);

        var zoom = d3.behavior.zoom()
            .scaleExtent([1, 10])
            .on("zoom", function(){
                svg.attr("transform",
                    "translate("+(-style.node.width)+", "+style.node.height
                    +")translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
            });
        svg = svg
            .on("dblclick", function(){
                zoom.scale(1);
                zoom.translate([0, 0]);
                svg.transition().attr("transform", "translate("+(-style.node.width)+", "+style.node.height+")");
            })
            .append("g").call(zoom).append("g")
            .attr("transform", "translate("+(-style.node.width)+", "+style.node.height+")");

        drawLegend(svgOriginal);

        drawTree(roottree);

        div.append("button")
            .style("position", "relative")
            .style("top", "-"+style.buttons.reset.height)
            .style("width", style.buttons.reset.width)
            .style("height", style.buttons.reset.height)
            .style("opacity", style.buttons.reset.alpha)
            .style("background", style.buttons.reset.background)
            .style("background-size", "contain")
            .style("background-image", "url("+style.buttons.reset.img+")")
            .style("cursor", "pointer")
            .style("border", "none")
            .on("mouseover", function(){
                d3.select(this).style("opacity", "1");
            })
            .on("mouseout", function(){
                d3.select(this).style("opacity", style.buttons.reset.alpha);
            })
            .on("click", function(){
                zoom.scale(1);
                zoom.translate([0, 0]);
                svg.transition().attr("transform", "translate("+(-style.node.width)+", "+style.node.height+")");
                var fathers = findFathers(roottree);
                for(var i=0;i<fathers.length;i++){
                    openSubTree(fathers[i]);
                    drawTree(fathers[i], true);
                }
            });
    };

    Object.seal(DecisionTree);
    return DecisionTree;
});