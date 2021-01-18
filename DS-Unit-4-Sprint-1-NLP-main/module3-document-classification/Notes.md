## Kaggle Submissions:

### Submission 1: Grid Search CV

Created a function to clean the text data before doing a train/val split:

```
def clean_text(df):
    data = []
    
    for i in range(len(df)):
        x = re.sub(r'^\\n', '', df[i])
        x1 = x.lower()
        x2 = re.sub('[^a-z A-Z 0-9]', ' ', x1)
        data.append(x2)
    
    return data
```

```
# Instaniate default vecotorizer
vect = TfidfVectorizer(stop_words='english',
                       ngram_range=(1, 3))

# Instaniate default classifer
clf = LinearSVC()

# Create the pipline
pipe = Pipeline([('vect', vect), ('clf', clf)])
```

```
# Get sparse document term matrix
dtm = vect.fit_transform(X_train)

# Convert DTM to dataframe
dtm = pd.DataFrame(dtm.todense(), columns=vect.get_feature_names())

# Check my work and look at the column names in the DTM
print(dtm.shape)
dtm.columns
```

```
parameters = {
    'vect__max_df': (0.25, 0.30, 0.50, 0.60),
    'vect__min_df': (3, 6, 9, 12, 15),
    'vect__max_features': (6000, 9000, 12000),
    'clf__penalty': ('l1', 'l2'),
    'clf__C': (0.3, 0.5, 1.0, 1.5, 2.0)
}

grid_search = GridSearchCV(pipe, parameters, cv=6, n_jobs=8, verbose=1)
grid_search.fit(X_train, y_train)
```

```
grid_search.best_score_
   0.7512940726749414
```

```
grid_search.best_params_
   {'clf__C': 0.3,
    'clf__penalty': 'l2',
    'vect__max_df': 0.5,
    'vect__max_features': 12000,
    'vect__min_df': 3}
```

```
# Evaluate on the validation data
y_val_pred = grid_search.predict(X_val)
accuracy_score(y_val, y_val_pred)
    0.7555012224938875
```

**Score received on the public leaderboard: 0.76143**

### Submission 2: LSI 
- TruncatedSVD, 
- TdidfVectorizer, 
- RandomforestClassifier, 
- RandomizedSearchCV, 
- GridSearchCV

Used cleaned text that was already cleaned with my function above

```
# Use TruncatedSVD for my LSI model
lsi = TruncatedSVD(algorithm='randomized', n_iter=30)

# Use the best parameters from my grid search for my Tfidf vectorizer
vect = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.5,
    max_features=12000
)

# Instantiate my classifier
clf = RandomForestClassifier(random_state=97)

# Define my pipeline 
#   (make sure to run the vectorizer first or it will throw an error 
#   when fitting about not being able to converting a string to a float)
pipe = Pipeline([
    ('vect', vect),  # TF-IDF Vectorier
    ('lsi', lsi),    # Truncated SVD Dimensionality Reduction
    ('clf', clf)     # RandomForest Classifier
])

# Define my parameters to try
params = {
    'lsi__n_components': stats.randint(100, 3000)
    
}
```

```
# Run it through a Randomized Search Cross Validation
random_search = RandomizedSearchCV(pipe, params, cv=3, n_iter=5, 
                                   n_jobs=10, verbose=1)
# Fit the model
random_search.fit(X_train, y_train)
```

```
Random Search CV Best Score: 0.7063329991603554 

Random Search CV Best Parameters: {'lsi__n_components': 419} 

Random Search CV Test Accuracy 0.7078239608801956
```

```
# Define parameters to try using grid search cv
parameters = {
    'lsi__n_components': [10, 100, 250, 419, 500],
    'vect__max_df': (0.3, 0.5, 0.6, 0.75),
    'clf__max_depth':(5, 10, 15, 20)
}

grid_search = GridSearchCV(pipe, parameters, cv=5, n_jobs=10, verbose=1)
grid_search.fit(X_train, y_train)
```

```
Grid Search Best Score: 0.7295797799851076 

Grid Search Best Parameters: {'clf__max_depth': 10, 'lsi__n_components': 10, 'vect__max_df': 0.3} 

Grid Search Test Accuracy 0.7200488997555012
```

**Score received on the public leaderboard: 0.72549**

### Submission 3: Word Embeddings

```

```

```

```

```

```

**Score received on the public leaderboard: **

### Submission 4: Gradient Boosting Classifier and SpaCy

```


```

```


```

**Score received on the public leaderboard: **

**Highest score received on the public leaderboard: 0.72549**

**Score received on the private leaderboard: 0.77234**