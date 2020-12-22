import { Ajax } from "./ajax.js";
import { render_sentence, render_related_words, render_relations } from "./render.js";
function add_mouseover_event(listener) {
    let entities = document.getElementsByClassName("entity");
    for (var entity of Array.from(entities)) {
        entity.addEventListener("mouseover", listener);
    }
}
function process_sentence_listener(data_json) {
    let sentence = sessionStorage.getItem("sentence");
    let data = JSON.parse(data_json);
    let entity_indexes = [];
    for (const entry of data["entities"]) {
        entity_indexes.push([entry["range"][0], entry["range"][1]]);
    }
    render_sentence(data, sentence);
    sessionStorage.setItem("entities", JSON.stringify(entity_indexes));
    add_mouseover_event(entity_mouseover);
}
function entity_mouseover_listener(data_json) {
    let data = JSON.parse(data_json);
    let chosen_index = data["chosen_index"];
    let related_words = data["related_words"];
    let relations = data["ralations"];
    render_related_words(related_words);
    render_relations(relations, chosen_index);
}
function process_sentence() {
    let form_data = new FormData(document.getElementById("sentence-input"));
    let sentence = form_data.get("input").toString();
    sessionStorage.setItem("sentence", sentence);
    Ajax.post("/sentence/", sentence, process_sentence_listener);
    return false;
}
function entity_mouseover(event) {
    let sentence = sessionStorage.getItem("sentence");
    let entities = JSON.parse(sessionStorage.getItem("entities"));
    let chosen_index = Number(event.target.id.substring(7));
    let chosen_entity_range = entities[chosen_index];
    let chosen_entity = sentence.substring(chosen_entity_range[0], chosen_entity_range[1]);
    let post_data = {
        "entities": entities,
        "chosen_index": chosen_index,
        "sentence": sentence
    };
    document.getElementById("selected-entity").innerHTML =
        chosen_entity +
            "<div>实体词</div>";
    Ajax.post("/entity/", JSON.stringify(post_data), entity_mouseover_listener);
}
//script starts here
window.addEventListener("load", function () {
    document.getElementById("sentence-input").onsubmit =
        process_sentence;
});
