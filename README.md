# Fashion Forward Forecasting

**Udacity Data Scientist Nanodegree — DS Pipelines Project**

## Scenario

You've recently joined **StyleSense**, a rapidly growing online women's clothing
retailer, as a data scientist. StyleSense is known for its trendy and affordable
fashion, and its customer base has exploded in recent months. This influx of new
customers is fantastic for business, but it has created a challenge: a backlog
of product reviews with missing data. Customers are still leaving valuable
feedback in the text of their reviews, but they aren't always indicating whether
they recommend the product.

Your first task at StyleSense is to leverage the existing data — those reviews
with complete information — to build a predictive model. This model will
analyze the text of a review, the customer's age, the product category, and
other relevant information to predict whether or not the customer would
recommend the product. By automating this process, you'll help StyleSense gain
valuable insights into customer satisfaction, identify trending products, and
ultimately provide a better shopping experience for their growing customer base.

## Objective

Build a machine learning pipeline that predicts whether a customer recommends a
product, using numerical, categorical, and text data from product reviews.

## Results

| Metric | Score |
|---|---|
| Best CV Score (fine-tuning) | 85.88% |
| Accuracy (test set) | 85.96% |
| Precision (test set) | 86.12% |
| Recall (test set) | **98.88%** |

**Best hyperparameters:** `n_estimators=100`, `max_depth=None`

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook
```

Place the dataset at `data/raw/reviews.csv`, then open `src/starter.ipynb`.

## Project Layout

```
dsnd-pipeline-project/
  src/
    starter.ipynb
  data/
    raw/          ← place reviews.csv here (not included)
  requirements.txt
  README.md
```

---

## Narrative

### The Task

StyleSense has a backlog of product reviews where customers left written feedback
but didn't fill in the "Recommended" field. The goal: train a model on complete
reviews to predict that missing field — turning unstructured text into a business
signal about customer satisfaction.

This is a classic multi-modal classification problem: the data has numerical
features (Age, Positive Feedback Count), categorical features (Division,
Department, Class Name), and text features (Title, Review Text). A sklearn
pipeline handles all three simultaneously, preprocessing each type appropriately
before passing everything to a classifier.

---

### Data Exploration

The dataset has 18,442 rows with zero nulls. Key findings:

**Class imbalance:** 81.6% of reviews were "Recommended" (1), only 18.4% "Not
Recommended" (0). This means the model naturally leans toward predicting
"recommended" — worth noting when interpreting precision vs. recall tradeoffs.

**Numerical features:**
- Age ranged 18–99, mean 43 — well-behaved, no issues
- Positive Feedback Count was heavily right-skewed (mean 2.7, max 122) — outliers exist but StandardScaler handles them adequately

**Categorical features:**
- Division Name had only 2 unique values (General, General Petite)
- Department Name had 6 values (Tops, Dresses, Bottoms, Jackets, Intimate, Trend)
- Class Name had many values — OneHotEncoder's `handle_unknown='ignore'` is critical

**Text features:**
- Title: avg 3.3 words — short, limited signal on its own
- Review Text: avg 62 words — rich, this is where TF-IDF does its best work

#### Why Explore After the Train/Test Split?

The data exploration section comes *after* the train/test split in the notebook —
and that's intentional, not an accident. If you explore the full dataset before
splitting, you risk **data leakage**: you might unconsciously make modeling
decisions (which features to keep, how to handle outliers, what imputation
strategy to choose) based on patterns that include the test set. The test set is
supposed to simulate truly unseen data. By splitting first, the test set goes into
a "lockbox" and all exploration and modeling decisions are driven only by the
training data.

---

### Pipeline Design

The pipeline has four parallel preprocessing branches combined by a
`ColumnTransformer`, feeding into a `RandomForestClassifier`:

**Numerical pipeline** (`Age`, `Positive Feedback Count`):
`SimpleImputer(mean)` → `StandardScaler`
- SimpleImputer handles any future nulls gracefully, even though the current dataset has none
- StandardScaler normalizes the scale difference between Age (~18–99) and
  Positive Feedback Count (~0–122)

**Categorical pipeline** (`Division Name`, `Department Name`, `Class Name`):
`SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
- `handle_unknown='ignore'` is essential — if a new category appears at prediction time that wasn't in training (a new clothing class, for example), the encoder silently ignores it rather than crashing

**Review Text pipeline** (`Review Text`):
`FunctionTransformer(np.ravel)` → `SpacyLemmatizer` → `TfidfVectorizer(stop_words='english')`
- `np.ravel` flattens the column from a 2D array to 1D (required by TfidfVectorizer)
- `SpacyLemmatizer` is a custom sklearn Transformer that reduces each word to its base form using spaCy (e.g., "running" → "run", "dresses" → "dress") — this consolidates vocabulary so TF-IDF captures meaning more efficiently
- `TfidfVectorizer` converts the lemmatized text to numerical features, weighting rare meaningful words higher than common ones

**Title pipeline** (`Title`):
`FunctionTransformer(np.ravel)` → `TfidfVectorizer(stop_words='english')`
- Titles average only 3.3 words, so spaCy lemmatization adds less value — plain TF-IDF is sufficient here

**Why RandomForestClassifier?**
Random forests handle mixed feature types well, are robust to the class imbalance
(81/19 split), and don't require feature scaling to be perfect. They also
naturally handle the high-dimensional sparse output of TF-IDF without overfitting
as aggressively as a single decision tree.

---

### Training

Training on ~16,500 reviews took approximately 3 minutes. The spaCy lemmatization
step dominates the time — it processes each review word-by-word through the full
English language model.

---

### Fine-Tuning

`RandomizedSearchCV` searched 6 combinations across two hyperparameters:
- `n_estimators`: [50, 100, 200] — number of trees in the forest
- `max_depth`: [None, 10, 20] — how deep each tree can grow

With 5-fold cross-validation, that's 30 total fits. The best parameters were
`n_estimators=100, max_depth=None`, suggesting deeper trees with the default
ensemble size perform best on this data.

---

### Final Evaluation

| Metric | Score |
|---|---|
| Accuracy | 85.96% |
| Precision | 86.12% |
| Recall | **98.88%** |

The recall number is the standout result. The model catches **98.88%** of all
actually-recommended products — almost nothing slips through. For StyleSense this
is exactly the right tradeoff: missing a genuinely recommended product (false
negative) is worse than occasionally flagging a borderline review as recommended
(false positive). The class imbalance (81.6% recommended) naturally drives the
model toward high recall on the majority class.

Precision at 86.12% means about 14% of predicted "recommended" items were
actually not recommended — acceptable given the business context.

---

### AHA Moments

- **Recall vs. Precision tradeoff** — high recall means "don't miss the positives";
  high precision means "don't cry wolf." The right metric depends entirely on the
  business cost of each error type.
- **Class imbalance shapes your metrics** — with 81.6% of data labeled "recommended,"
  a model that always predicts "recommended" would get 81.6% accuracy for free.
  Accuracy alone doesn't tell the whole story; precision and recall together do.
- **Two text pipelines add significant compute cost** — running spaCy through both
  Title and Review Text across 30 fine-tuning folds turned a 3-minute training
  run into a 90-minute search. For future projects: consider whether the shorter
  text column justifies the compute cost, or skip lemmatization on short text.
- **Split before you explore** — the order of operations in the notebook is not
  arbitrary. Exploring only training data keeps the test set as a true lockbox
  and prevents data leakage from influencing modeling decisions.
