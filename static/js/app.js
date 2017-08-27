/** This will be our global, topmost level namespace */
var NS = NS || { };

NS.ie = function(){ return navigator.userAgent.indexOf('MSIE') != -1; };

"use strict;"

/** This is the skeleton for the module. Copy this to start with a new module */
NS.common = (function () {
    var config = {
        autoInvokeInit: false,
        topics: {},
        news_topics:{},
        files: {},
        doc_topics: {}
    };

     // render pie chart for topic word distribution
    var render_news_topic_pie_chart = function(title) {
        console.log("rendering topic chart for ", title);
        var chart = c3.generate({
            bindto: "#topic-chart",
            data: {
                columns: config.topics[title],
                type : 'pie'
            }
        });
        $("#topic-title").text(title);
    };

    // render pie chart for topic word distribution
    var render_topic_pie_chart = function(title) {
        console.log("rendering topic chart for ", title);
        var chart = c3.generate({
            bindto: "#topic-chart",
            data: {
                columns: config.topics[title],
                type : 'pie'
            }
        });
        $("#topic-title").text(title);
    };


    // render pie chart for document topics
    var render_doc_pie_chart = function(doc_name) {
        console.log("rendering doc chart for ", doc_name);
        var chart = c3.generate({
            bindto: "#doc-chart",
            data: {
                columns: config.doc_topics[config.files[doc_name]],
                type : 'pie',
                onclick: function (d, i) {
                    NS.common.render_topic_pie_chart('topic' + d["id"]);
                },
            }
        });
        $("#doc-title").text(doc_name);
        var topic_no = "topic" + config.doc_topics[config.files[doc_name]][0][0];
        NS.common.render_topic_pie_chart(topic_no)
    }

    var register_topic_pie_chart = function() {
        console.log("register topics for pie chart");
        $(".topic").hover(function(){
            var topic_id = $(this).attr("id");
            NS.common.render_topic_pie_chart(topic_id);
        });
    }

    var register_news_topic_pie_chart = function() {
        console.log("register topics for pie chart");
        $(".topic").hover(function(){
            var topic_id = $(this).attr("id");
            NS.common.render_topic_pie_chart(topic_id);
        });
    }


    //render news topics
    var render_news_topics = function (el,topics) {
        console.log("rendering news topics");
        var table_body = "";
         $.each(topics, function(k, v){
           table_body += "<tr id=" + k + " class='topic'><td>" + k + "</td><td>";
           v.forEach(function(item) {table_body +="<em><strong>" + item[0] + "</strong></em> <small>(" + item[1] + ") </small>"});
           table_body += "</td></tr>";
        });
        $(el).html(table_body);
        NS.common.register_topic_pie_chart();
    }

    // render topics
    var render_topics = function(el, topics) {
        console.log("rendering topics ");
        var table_body = "";
        $.each(topics, function(k, v){
           table_body += "<tr id=" + k + " class='topic'><td>" + k + "</td><td>";
           v.forEach(function(item) {table_body +="<em><strong>" + item[0] + "</strong></em> <small>(" + item[1] + ") </small>"});
           table_body += "</td></tr>";
        });
        $(el).html(table_body);
        NS.common.register_topic_pie_chart();
    };

    var register_dropdown_change = function () {
        $(".wiki-document").on("click", function(){
            var doc_name = $(this).text();
            NS.common.render_doc_pie_chart(doc_name);
        });
    };

    var update_docs_dropdown_menu = function() {
        var li_menu = "";
        var doc_names = [];
        $.each(config.files, function(k, v){
            doc_names.push(k);
        });
        doc_names.sort();
        $.each(doc_names, function(k, v){
            li_menu += "<li class='wiki-document'><a href='#'>" + v + "</a></li>";
        });

        $("#documents-dropdown-menu").html(li_menu);
        register_dropdown_change();
    };

    var init = function (data) {
        var topics_file = data["topics_file"];
        var files_list_file = data["files_list_file"];
        var doc_topics_file = data["doc_topics_file"];
        var news_topics = data["comp_graphics_topics.json"];
        var topics_page = data["topics_page"];

        //new file new by me
        $.getJSON(news_topics,function (topics) {
            console.log("1.rendering news topics.....");
            config.topics = topics;
            if(topics_page){
                NS.common.render_news_topics("#topic1-table thbody",topics);
                NS.common.render_news_topic_pie_chat('topic0');
            }
        });

        $.getJSON(topics_file, function(topics) {
            console.log("1.rendering all topics...");
            config.topics = topics;
            if (topics_page){
                NS.common.render_topics("#topic-table tbody", topics);
                NS.common.render_topic_pie_chart('topic0');
            }
        });

        $.getJSON(files_list_file, function(file_names) {
            console.log("2.rendering all files...");
            config.files = file_names;
        });

        $.getJSON(doc_topics_file, function(doc_topics) {
            console.log("3.rendering all doc topics...");
            config.doc_topics = doc_topics;
            NS.common.update_docs_dropdown_menu();
            if (!topics_page){
                setTimeout(function(){
                    NS.common.render_doc_pie_chart(Object.keys(NS.common.config.files)[0]);
                }, 500);
            }
        });
    };

    return {
        init: init,
        config: config,
        render_news_topics:render_news_topics,
        render_news_topic_pie_chat:register_news_topic_pie_chart,
        register_news_topic_pie_chart:register_news_topic_pie_chart,
        render_topic_pie_chart: render_topic_pie_chart,
        render_doc_pie_chart: render_doc_pie_chart,
        render_topics: render_topics,
        register_topic_pie_chart: register_topic_pie_chart,
        update_docs_dropdown_menu: update_docs_dropdown_menu
    };
})();

NS.wiki = (function () {
    // define some config (Settings)
    var config = {
        autoInvokeInit: false // whether the init function must be invoked automatically when page loads
    };

    // define the init function (Implementation)
    var init = function (topics_page) {
        console.log("wiki init");
        NS.common.init({
            "topics_file": "/static/json/topics.json",
            "files_list_file": "/static/json/wp_files.json",
            "doc_topics_file":"/static/json/doc_topics.json",
            "news_topics":"bbc_topics.json",
            'tnews_topics':"/static/comp_graphics_topics.json",
            "topics_page": topics_page
        });
    };

    // return an object
    return {
        config: config,
        init: init
    };

})();

NS.bbc = (function () {
    // define some config (Settings)
    var config = {
        autoInvokeInit: false // whether the init function must be invoked automatically when page loads
    };

    // define the init function (Implementation)
    var init = function (topics_page) {
        console.log("bbc init");
        NS.common.init({
            "topics_file": "/static/json/bbc_topics.json",
            "files_list_file": "/static/json/bbc_files.json",
            "doc_topics_file":"/static/json/bbc_doc_topics.json",
            "news_topics":"/static/json/bbc_topics.json",
            "topics_page": topics_page
        });
    };

    // return an object
    return {
        config: config,
        init: init
    };



})();

NS.tnews = (function () {
    // define some config (Settings)
    var config = {
        autoInvokeInit: false // whether the init function must be invoked automatically when page loads
    };

    // define the init function (Implementation)
    var init = function (topics_page) {
        console.log("20news iniit");
        NS.common.init({
            // "topics_file": "/static/json/topics.json",
            "topics_file": "/static/json/comp_graphics_topics.json",
            // "files_list_file": "/static/json/wp_files.json",
            // "doc_topics_file":"/static/json/doc_topics.json",
            // "news_topics":"/static/json/comp_graphics_topics.json",
            "topics_page": topics_page
        });
    };

    // return an object
    return {
        config: config,
        init: init
    };

})();