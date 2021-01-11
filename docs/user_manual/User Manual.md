# User Manual

For prerequisites and running steps, see [Prerequisites and Running](#Installation and Running).

## User Interface

By opening the [URI provided](#Running) in the browser, a web page like the following will show up.

![image-20210111224629232](.\UI blank.png)

The following figure shows all the functions provided by this project.

![image-20210111224117121](.\UI.png)

To start, **type the sentence** you want to process in the search bar shown as below, and hit `Enter` key. (Only Chinese is supported currently)

![image-20210111225024547](.\inputbox.png)

Then in the Entity Recognition section you will see the result of NER along with the Most Unrelated Entity. Sentence you just inputted will be echoed here with the entities underlined and labeled with the category it belongs to. (Currently all entity will be labeled with the only category "x".) The Most Unrelated Entity is given at the bottom right corner of the section.

![image-20210111230409588](.\NER.png)

To further process an entity in the sentence, **hover the mouse cursor** over the entity (in the Entity Recognition section). The entity you choose, its related words (with relativity value), and the relation between it and the rest of the entities in the sentence will show up at the right part of the web page.

![image-20210111230959781](.\right.png)

## Prerequisites and Running

### Prerequisites

* [Python](https://www.python.org/) `>=3.9.0`
* Python packages
	* [flask](https://palletsprojects.com/p/flask/) `>=1.1.2`
	* [requests](https://requests.readthedocs.io) `>=2.25.1`
	* [gensim](http://radimrehurek.com/gensim) `>=3.8.3`

### Running

In command line, change the directory to the root folder of source code, then:

```
python main.py
```

After a while, you will see the following lines:

```
 * Serving Flask app "main" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Here `http://127.0.0.1:5000/` is **the URL** you are going to open in the browser. (It can be **anything else**)