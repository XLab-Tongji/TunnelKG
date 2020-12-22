var SENTENCE_DOM = document.getElementById("sentence");
var RELATIVE_WORDS_DOM = document.getElementById("related-words");
var RELATIONS_DOM = document.getElementById("relations");
function render_word_block(sentence_div, sentence, start, end, type, entity_count) {
    let word_block_div = document.createElement("div");
    let word_div = document.createElement("div");
    let word_cat_div = document.createElement("div");
    let word_cat = null;
    word_block_div.className = "word-block";
    word_div.className = "word";
    word_cat_div.className = "word-cat";
    if (type) {
        word_cat = type;
        word_div.classList.add("entity");
        word_div.id = "entity-" + entity_count.toString();
        word_cat_div.innerHTML = word_cat;
    }
    else {
        word_cat_div.innerHTML = "&nbsp;";
    }
    word_div.innerHTML = sentence.substring(start, end);
    word_block_div.appendChild(word_div);
    word_block_div.appendChild(word_cat_div);
    sentence_div.appendChild(word_block_div);
}
export function render_sentence(data, sentence) {
    let sentence_div = document.createElement("div");
    let most_unrelated_div = document.createElement("div");
    let clear_div = document.createElement("div");
    let entries = data["entities"];
    let most_unrelated = data["MUE"];
    SENTENCE_DOM.innerHTML = "<h1>实体识别</h1>";
    sentence_div.id = "words";
    clear_div.className = "clear";
    if (entries.length > 0) {
        for (let j = 0; j < entries[0]["range"][0]; j++) {
            render_word_block(sentence_div, sentence, j, j + 1);
        }
    }
    for (let index = 0; index < entries.length; index++) {
        const entry = entries[index];
        const next_entry_start = (index < entries.length - 1) ? entries[index + 1]["range"][0] : sentence.length;
        render_word_block(sentence_div, sentence, entry["range"][0], entry["range"][1], entry["type"], index);
        for (let j = entry["range"][1]; j < next_entry_start; j++) {
            render_word_block(sentence_div, sentence, j, j + 1);
        }
    }
    most_unrelated_div.id = "most-unrelated";
    most_unrelated_div.innerHTML = "最不相关实体<div>" + most_unrelated + "</div>";
    SENTENCE_DOM.appendChild(sentence_div);
    SENTENCE_DOM.appendChild(most_unrelated_div);
    SENTENCE_DOM.appendChild(clear_div);
}
export function render_related_words(data) {
    let inHtml = "<h1>相关词</h1>";
    inHtml += "<table>";
    for (var i of data) {
        inHtml += "<tr>" +
            "<td><div class=\"related-word\">" + i[0] + "</div></td>" +
            "<td class=\"relativity\">" + i[1].toString() + "</td>" +
            "</tr>";
    }
    inHtml += "</table>";
    RELATIVE_WORDS_DOM.innerHTML = inHtml;
}
export function render_relations(data, chosen_index) {
    let inHtml = "<h1>关系</h1>";
    let sentence = sessionStorage.getItem("sentence");
    let entities_range = JSON.parse(sessionStorage.getItem("entities"));
    let chosen_entity = sentence.substring(entities_range[chosen_index][0], entities_range[chosen_index][1]);
    inHtml += "<table>";
    for (let index = 0; index < data.length; index++) {
        if (index == chosen_index) {
            continue;
        }
        const relation = data[index];
        let entity = sentence.substring(entities_range[index][0], entities_range[index][1]);
        inHtml += "<tr>" +
            "<td><div class=\"related-word\">" +
            chosen_entity + "</div>▶<div class=\"related-word\">" +
            entity + "</div></td>" +
            "<td class=\"relation\">" + relation + "</td>" +
            "</tr>";
    }
    inHtml += "</table>";
    RELATIONS_DOM.innerHTML = inHtml;
}
