# Advanced NLP with spaCy

## Chapter 1 
[Walk Through with Exercises](https://course.spacy.io/en/chapter1)

[YouTube Video](https://youtu.be/THduWAnG97k)

### [Introduction to spaCy](https://youtu.be/THduWAnG97k?t=16)

At the center of spaCy is the object containing the processing pipeline. It is common practice to call this variable `nlp`.

To create an English NLP object:
    `from spacy.lang.en import English` imports the English class from spaCy
    `nlp = English()` instantiates the English class and assigns it to a variable. You can use the `nlp` object like a function to analyse text. It contains all the different components in the pipeline. It also contains all the rules to tokenize words into text and punctuation.

When you process text with a `nlp` object, spaCy first tokenizes the text and creates a doc (document) object. The doc allows you to access information about the text in a structured way and no information is lost. The doc behaves like a normal python sequence and lets you iterate over its tokens or get a token by its index. 

Token objects represent a token (word, char) in a document (string). 

Token Attributes:
- `token.text` returns the verbatim token text.
- `span = doc[1:3] \n span.text` returns a slice of the doc as specified. This is only a view of the doc and does not contain any data itself.
- `token.is_alpha` returns a bool value if token char is an alphabetical char.
- `token.is_punct` returns a bool value if token char is a punctuation char.
- `token.like_num` returns a bool value if token char is a like a number char. It will also check if it is a number spelled out, such as "ten".
- `doc[token.i + 1]` will give you the next token in the document when you are in a for loop such as `for token in doc:`.
- `token.pos_` returns the predicted part of speech.
  - `PROPN`: Proper Noun
  - `NOUN`: Noun
  - `VERB`: Verb
  - `ADJ`: Adjective
- `token.dep_` returns the predicted dependency label.
  - Dependency label scheme:
    - `nsubj` nominal subject - the subject attached to the verb.
    - `dobj` direct object - the noun attached to the verb.
    - `det` determiner (article) - attached to the noun.
    - `ROOT` the main part of speech - usually a verb or the like.
- `token.head.text` returns the syntactic head token (parent token)
- `doc.ents` allows you to access the predicted entities.
  - `ent.text` prints the entity's text.
  - `ent.label_` returns the type of entity.
    - `ORG` organization
    - `GPE` Geo Political Entity (i.e. location)
    - `MONEY` currency


- TIP: `spacy.explain("<label or tag>")` to get a quick definition of the most common tags and labels

In spaCy attributes that return strings usually end with `_`. Attributes without the `_` return an integer / id value.

Lexical Attributes are attributes that refer to the entry in the vocabulary and not the text being passed in. These include `.is_alpha`, `.is_punct`, and `.like_num`.

### [Statistical Models](https://youtu.be/THduWAnG97k?t=191)
You can use spaCy to predict linguistic attributes in *context*, such as part-of-speech tags, syntactic dependencies, named entities. For example, if the word is a person's name, a verb, noun, etc.

Models are trained on large labeled example text datasets. They can be updated with more examples to fine-tune predictions. 

Model Packages:
- Each model package includes:
  - Binary weights that enable spaCy to make predictions
  - Vocabulary
  - Meta Information (language, pipeline)
- Not included in the model packages:
  - The labelled data that the model was trained on.
    - Statistical models allow you to generalize based on a set of training examples. Once they're trained, they use binary weights to make predictions. That's why it's not necessary to ship them with their training data.
- `en_core_web_sm` is the small package, trained on web text
  - `python -m spacy download en_core_web_sm` downloads the model to your machine.
  - `import spacy` have to import spacy to use it.
  - `nlp = spacy.load("en_core_web_sm")` to use the model package.
- `en_core_web_md` is the medium package, trained on web text
- `en_core_web_lg` is the large package, trained on web text

Predicting Named Entities:
- Real world objects that are assigned a name, such as a person, organization, or country.

Sometimes the model might predict an entity wrong. In this case you would manually have to create that entity as a span:
    
    ```
    # Get the span for "iPhone X" which is at index 1 and 2
    iphone_x = doc[1:3]
    
    # Print the span text
    print("Missing entity:", iphone_x.text)
    ```

### [Rule-based Matching](https://youtu.be/THduWAnG97k?t=430)
Using spaCy for rule-based matching is similar to reg ex.

spaCy ~vs~ reg ex:
- You can match on `doc` objects and not just strings
- You can match on tokens and token attributes. 
- You can use it to search for text and other lexical attributes.
- You can use the model's predictions to get back parts of speech and other useful information

Match patterns are a list of dictionaries, one per token.

You can match exact token texts:  
    `[{"TEXT": "iPhone"}, {"TEXT": "X"}]`

You can match lexical attributes:  
    `[{"LOWER": "iphone"}, {"LOWER": "x"}]`

You can match any token attributes:
    `[{"LEMMA": "buy"}, {"POS": "NOUN"}]`
    - `LEMMA` is the base form of the word.
    - `POS` is the part of speech

To use the Matcher:  

    ```
    # Import spaCy
    import spacy

    # Import the Matcher
    from spacy.matcher import Matcher
    
    # Load a spaCy model and create the nlp object
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize the matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)

    # Add the patter you want to the matcher
    pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]

    # The first arg ("IPHONE_PATTERN") is a unique id to identify which pattern was matched.
    # The second arg (None) is an optional callback.
    # The third arg (pattern) is the pattern you want to match
    matcher.add("IPHONE_PATTERN", None, pattern)

    # Process some text
    doc = nlp("Upcoming iPhone X release date leaked")

    # Call the matcher on the doc
    matches = matcher(doc)
    ```

When you call the matcher on a doc it returns a list of tuples. Each tuple consist of 3 values:
    - `match_id`: hash value of the pattern name
    - `start`: start index of the matched span
    - `end`: end index of the matched span

Since it is stored as a tuple, we can iterate over the matched values and create a span object:
    
    ```
    # Iterate over the matches
    for match_id, start, end in matches:
        # Get the matched span
        matched_span = doc[start:end]
        print(matched_span.text)

    # Or you can use list comprehension
    print("Matches:", [doc[start:end].text for match_id, start, end in matches])
    ```

Example of a more complex pattern using Lexical Attributes:
    
    ```
    pattern = [
        {"IS_DIGIT": True},  # Is it a number (start of pattern)
        {"LOWER": "fifa"},   # Case insensitive "fifa"
        {"LOWER": "world"},  # Case insensitive "world"
        {"LOWER": "cup"},    # Case insensitive "cup"
        {"IS_PUNCT": True}   # Is punctuation (end of pattern)
    ]
    
    doc = nlp("2018 FIFA World Cup: France won!")

    OUTPUT: 2018 FIFA World Cup:
    ```

Another example for matching 2 patterns:
    
    ```
    pattern = [
        {"LEMMA": "love", "POS": "VERB"},  # Base token is love and a verb (start of the pattern)
        {"POS": "NOUN"}  # followed by a noun (end of pattern)
    ]

    doc = nlp("I loved dogs but now I love cats more.")

    OUTPUT: loved dogs
            love cats
    ```

Example for matching multiple patterns using operators and quantifiers:
    
    ```
    pattern = [
        {"LEMMA": "buy"},  # Base token (start of pattern)
        {"POS": "DET", "OP": "?"},  # OP is optional: match 0 or 1 times, making it optional to match the article "DET" (a, the, an, etc.)
        {"POS": "NOUN"}  # A noun (end of pattern)
    ]

    doc = nlp("I bought a smartphone. Now I'm buying apps.")

    OUTPUT: bought a smartphone
            buying apps
    ```

`OP` can have one of four values:
- `{"OP": "!"}`: Negation: match 0 times
- `{"OP": "?"}`: Optional: match 0 or 1 times
- `{"OP": "+"}`: Match 1 or more times
- `{"OP": "*"}`: Match 0 or more times

## Chapter 2

[Walk Through with Exercises]()

[YouTube Video](https://youtu.be/THduWAnG97k?t=666)

In this chapter, you'll use your new skills to extract specific information from large volumes of text. You'll learn how to make the most of spaCy's data structures, and how to effectively combine statistical and rule-based approaches for text analysis.

### [Data Structures (1): Vocab, Lexemes, and StringStore](https://youtu.be/THduWAnG97k?t=666)

