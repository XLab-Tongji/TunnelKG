## 接口文档

​                   

---

### baseurl/chineseNER/{input_str}

**Request Method**: GET

**Description**

use model to find the entity in the input sentence

**Parameter**

| Name      | Located in | Type   | Description    | Required |
| --------- | ---------- | ------ | -------------- | -------- |
| input_str | url        | String | input sentence | Yes      |

**Return**

| Code | Description | Type |
| ---- | ----------- | ---- |
| 200  | OK          | JSON |

**Sample Request**

```http
GET baseurl/chineseNER/你好隧道，你好盾构机
```

**Sample Return**

Status: 200 OK

Body:

```JSON
[
    {
        "start": 2,
        "stop": 4,
        "word": "隧道",
        "type": "x"
    },
    {
        "start": 7,
        "stop": 9,
        "word": "盾构",
        "type": "x"
    }
]
```

