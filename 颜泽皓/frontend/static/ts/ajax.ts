export const Ajax = {
    get: function (url: string, callback: (respText: string) => any) {
        let xhr = new XMLHttpRequest();
        xhr.open("GET", url, true);
        xhr.onreadystatechange = function () {
            if (
                xhr.readyState == 4 &&
                (xhr.status == 200 || xhr.status == 304)
            ) {
                callback(xhr.responseText);
            }
        }
    },

    post: function (url: string, data: string, callback: (respText: string) => any) {
        let xhr = new XMLHttpRequest();
        xhr.open("POST", url, true);
        xhr.setRequestHeader("Content-type", "application/json")
        xhr.onreadystatechange = function () {
            if (
                xhr.readyState == 4 &&
                (xhr.status == 200 || xhr.status == 304)
            ) {
                callback(xhr.responseText);
            }
        }
        xhr.send(data);
    },
};
