/**
 * Created by Attila Bagoly <battila93@gmail.com> on 5/14/16.
 */

(function(factory){

    var JSROOT_source_dir = "https://root.cern.ch/js/notebook/scripts/";

    var url = "";
    if (requirejs.s.contexts.hasOwnProperty("_")) {
        url = requirejs.s.contexts._.config.paths["JsMVA"].replace("JsMVA.min","");
    }
    if ((console!==undefined) && (typeof console.log == 'function')) {
        if (url!=""){
            console.log("JsMVA source dir:" + url.substring(0, url.length-1));
        } else {
            console.log("JsMVA source dir can't be resolved, requireJS doesn't have context '_', this will be a problem!");
        }
    }

    require.config({
        paths: {
            'd3': JSROOT_source_dir+'d3.v3.min',
            'JsRootCore': JSROOT_source_dir+'JSRootCore',
            'nn': url+'NeuralNetwork.min',
            'dtree': url+'DecisionTree.min',
            'IChart': url+'IChart'
        }
    });

    define(['JsRootCore'], function(jsroot){
        return factory({}, jsroot);
    });

}(function(JsMVA, JSROOT){

    JsMVA.drawTH2 = function(divid, dat_json){
        var obj = JSROOT.parse(dat_json);
        JSROOT.draw(divid, obj, "colz;PAL50;text");
    };

    JsMVA.draw = function(divid, dat_json){
        var obj = JSROOT.parse(dat_json);
        JSROOT.draw(divid, obj);
    };

    JsMVA.drawNeuralNetwork = function(divid, dat_json){
        var obj = JSON.parse(dat_json);
        require(['nn'], function(nn){
            nn.draw(divid, obj);
        });
    };

    JsMVA.drawDecisionTree = function(divid, dat_json){
        require(['dtree'], function(dtree){
            var obj = JSON.parse(dat_json);
            dtree.draw(divid, obj);
        });
    };

    JsMVA.drawTrainingTestingErrors = function(divid, dat_json){
        var obj = JSROOT.parse(dat_json);
        JSROOT.draw(divid, obj);
        require(['d3'], function(d3){
            var div = d3.select("#"+divid).style("position", "relative");
            var svg = div.append("svg")
                .attr("width", "200px")
                .attr("height", "50px")
                .style({"position":"absolute", "top": "8px", "right": "8px"});
            var attr = {
                "pos": {"x": 150, "y": 10},
                "rect": {"width": 10, "height":10},
                "dy": 20,
                "padding": 10
            };
            canvas = {
                width:  160,
                height: 70
            };
            var container = svg.append("g").attr("id", "legend");
            container.selectAll("g")
                .data(obj.fGraphs.arr)
                .enter()
                .append("g")
                .each(function(d, i){
                    var g = d3.select(this);
                    g.append("rect")
                        .attr("x", canvas.width-attr.pos.x)
                        .attr("y", attr.pos.y+i*attr.dy)
                        .attr("width", attr.rect.width)
                        .attr("height", attr.rect.height)
                        .style("fill", function(d){return JSROOT.Painter.root_colors[d.fFillColor];});
                    g.append("text")
                        .attr("x", canvas.width-attr.pos.x+attr.rect.width+attr.padding)
                        .attr("y", attr.pos.y+i*attr.dy+attr.rect.height)
                        .text(function(d){return d.fTitle;})
                        .style("fill", function(d){return JSROOT.Painter.root_colors[d.fFillColor];});
                });
            div.append("svg").attr("width", "55px").attr("height", "20px")
                .style({"position":"absolute", "bottom": "15px", "right": "40px"})
                .append("text")
                .attr("x", "5px")
                .attr("y", "15px")
                .text(obj.fGraphs.arr[0].fTitle.indexOf("Error on training set")!=-1 ? "Epoch" : "#tree")
                .style({"font-size": "16px"});
        });
    };

    JsMVA.updateTrainingTestingErrors = function(divid, dat_json){
        var obj = JSROOT.parse(dat_json);
        JSROOT.redraw(divid, obj);
    };

    return JsMVA;
}));
