# IEEE-CIS Fraud Detection — პროექტის დოკუმენტაცია
 
> **პლატფორმა:** [DagsHub MLflow](https://dagshub.com/gormo22/ML-FraudDetection.mlflow)  
> **Dataset:** [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)

---

## რეპოზიტორიის სტრუქტურა

```
ML-FraudDetection/
│
├── model_experiment.ipynb     # მთავარი ნოუთბუქი — EDA, Feature Engineering, Training
├── model_inference.ipynb      # ინფერენს ნოუთბუქი — მოდელის ჩატვირთვა და submission
├── README.md                  # ეს ფაილი
└── data/                      # (ლოკალური, .gitignore-ში)
    ├── train_transaction.csv
    ├── train_identity.csv
    ├── test_transaction.csv
    └── test_identity.csv
```

---

## პროექტის მიმოხილვა

ვმუშაობდი **IEEE-CIS Fraud Detection** dataset-ებზე. ეს არის Kaggle-ის კონკურსი, სადაც ამოცანა იყო საბანკო ტრანზაქციების მიხედვით განგვესაზღვრა, თაღლითურია თუ არა კონკრეტული ტრანზაქცია (ბინარული კლასიფიკაცია).

დეითასეტი შედგება ორი ფაილისგან — `transaction` (ძირითადი ტრანზაქციის ინფო) და `identity` (მოწყობილობის/მომხმარებლის ინფო), რომლებიც `TransactionID`-ზე უნდა მომერგო. საბოლოოდ, train set-ს ჰქონდა **590,540 სტრიქონი და 434 სვეტი** — ეს ძალიან დიდი რაოდენობაა, რაც თავიდანვე ნიშნავდა, რომ მეხსიერების მენეჯმენტი კრიტიკული იქნებოდა.

**შეფასების მეტრიკა:** ROC-AUC

---

## მონაცემთა ჩატვირთვა და მეხსიერების ოპტიმიზაცია

### რატომ გამოვიყენე `reduce_mem_usage`?

პირველი გამოწვევა, რასაც წავაწყდი, იყო მეხსიერება. raw train dataset-ი იკავებდა **1,955 MB** — თითქმის 2 GB. ასეთ პირობებში მოდელის ტრენინგი, თუნდაც cross-validation, პრაქტიკულად შეუძლებელი იქნებოდა ჩვეულებრივ გარემოში.

ამიტომ დავწერე `reduce_mem_usage` ფუნქცია, რომელიც გადის ყველა სვეტზე და ამცირებს მის dtype-ს მინიმუმ საჭირო ტიპამდე:
- `int` სვეტებისთვის: `int8` → `int16` → `int32` (მინ/მაქს მნიშვნელობების მიხედვით)
- `float` სვეტებისთვის: `float64` → `float32`
- `object` სვეტებისთვის: `category`

**შედეგი:**
| Dataset | მეხსიერება (before) | მეხსიერება (after) | შემცირება |
|---|---|---|---|
| Train | 1,955.37 MB | 924.29 MB | **52.7%** |
| Test | 1,673.87 MB | 792.58 MB | **52.6%** |

ამ ერთი ნაბიჯით მეხსიერება ნახევარზე მეტით შევამცირე, რაც საშუალებას მაძლევდა ყველა შემდგომი ეტაპი ნორმალურად ჩამერთო.

### Merge სტრატეგია

ორი ფაილი გავაერთიანე `LEFT JOIN`-ით `TransactionID`-ზე. ავარჩიე სწორედ LEFT JOIN და არა INNER JOIN, რადგან:
- identity ინფო ყველა ტრანზაქციას არ ახლავს — ბევრ ტრანზაქციას უბრალოდ არ ჰქონდა identity ჩანაწერი
- INNER JOIN-ის შემთხვევაში დავკარგავდი ამ ტრანზაქციებს train-დანაც, რაც მონაცემთა დაკარგვა იქნებოდა
- NaN-ები identity სვეტებში შემდეგ იმპუტაციაში დამუშავდება

---

## მონაცემთა მიმოხილვა (EDA)

EDA-ს ყოველი ნაწილი ცალკე MLflow run-ად დავალოგე `EDA` ექსპერიმენტის ქვეშ, რათა ვიზუალიზაციები შენახული ყოფილიყო.

### 2.1 სამიზნე ცვლადის განაწილება

პირველი, რაც შევამოწმე, იყო კლასების ბალანსი.

![Target Variable Distribution](outputs/target_distribution.png)

> `isFraud` განაწილების bar chart  

**დასკვნა:** დეითასეტი **ძალიან არაბალანსირებულია** — თაღლითური ტრანზაქციები შეადგენს სულ დაახლოებით **3.5%-ს**. ეს ნიშნავს, რომ:
- accuracy ცუდი მეტრიკაა (მოდელი ყველაფერს 0-ად რომ გამოიწვევდეს — 96.5% accuracy მოუვა)
- **ROC-AUC** სწორი არჩევანია
- ტრენინგისას `class_weight='balanced'` ან `is_unbalance=True` გამოსაყენებელია

### 2.2 TransactionAmt ანალიზი

![Transaction Amount Analysis](outputs/transaction_amt_eda.png)

TransactionAmt-ის ანალიზიდან გამოვლინდა:
- ტრანზაქციის თანხა ძლიერი **right-skewed** განაწილებით ხასიათდება — log transform აუცილებელია ვიზუალიზაციისთვის
- ზოგიერთ percentile bin-ში თაღლითობის პროცენტი მნიშვნელოვნად მაღლა ადის — ე.ი. თანხის სიდიდე კარგი სიგნალია
- CDF გრაფიკი გვიჩვენებს, რომ fraud-ების გამანაწილებელი ფუნქცია legit-ებისგან განსხვავდება

### 2.3 დროის ანალიზი

![Time-Based Patterns](outputs/time_eda.png)

`TransactionDT` წარმოადგენს unix-ის სახის timestamp-ს. ამ სვეტიდან ამოვიღე:
- **საათი დღის განმავლობაში** → ჩანს, რომ ღამის საათებში fraud rate მაღლა ადის
- **კვირის დღე** → გარკვეული პატერნები გამოჩნდა

ეს ანალიზი საფუძველი გახდა შემდეგ Feature Engineering ეტაპზე დროის ფიჩერების შესაქმნელად.

### 2.4 კატეგორიული ფიჩერების ანალიზი

![Categorical Feature Analysis](outputs/categorical_eda.png)

განვიხილე `ProductCD`, `card4`, `card6`, `DeviceType`:
- **ProductCD** — სხვადასხვა პროდუქტის კატეგორიებს განსხვავებული fraud rate აქვთ
- **card4** — ბარათის ტიპი (visa/mastercard/discover/amex) fraud-თან კორელაციაშია
- **DeviceType** — mobile vs desktop-ს შორის განსხვავება შესამჩნევია

### 2.5 კორელაციის ანალიზი

![Correlation Heatmap](outputs/correlation_heatmap.png)

ეს ანალიზი მნიშვნელოვანი იყო Feature Selection-ის დასაგეგმად — გამოჩნდა, რომ ბევრი ფიჩერი ძლიერ კორელირებულია ერთმანეთთან, რაც redundancy-ს ნიშნავს.

### 2.6 V-ფიჩერების ანალიზი

![V-Features Analysis](outputs/v_features_eda.png)

დეითასეტში `V1`-`V339` ფიჩერები წარმოადგენს Vesta-ს (fraud detection კომპანიის) proprietary ფიჩერებს. შევამჩნიე, რომ:
- ბევრ V-ფიჩერს **50%-ზე მეტი missing value** აქვს
- კორელაცია target-თან ძალიან განსხვავებულია — ზოგი ძლიერ კორელირებულია, ზოგი სულ არა

### 2.7 Missing Values ანალიზი

```
THRESHOLD = 0.80  →  dropped columns: 434 → ~200 სვეტი
```

ვნახე, რომ ბევრ სვეტს **80%-ზე მეტი** missing value ჰქონდა. ასეთი სვეტები პრაქტიკულად უსარგებლოა — imputation-ი ვერ გადაარჩენს სვეტს, რომელშიც ინფორმაციის მხოლოდ 20% ან ნაკლებია. ამიტომ დავაწესე **80% threshold** და ყველა ასეთი სვეტი გადავაგდე.

---

## 3️Feature Engineering

Feature Engineering ეტაპზე ვიყენებდი იმ ინსაიტებს, რომლებიც EDA-ში გამოვლინდა.

### 3.1 დროის ფიჩერები (Temporal Features)

**რატომ?** EDA-ში დავინახე, რომ fraud rate-ს პატერნი აქვს დღის და კვირის მიხედვით. `TransactionDT` raw სახით მოდელისთვის სასარგებლო არ იქნებოდა (ის უბრალოდ მონოტონურად იზრდება), ამიტომ ამოვიღე:

```python
df['Transaction_day_of_week'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 7)
df['Transaction_hour'] = np.floor(df['TransactionDT'] / 3600) % 24
```

შემდეგ, კიდევ უფრო მნიშვნელოვანი ნაბიჯი გადავდგი — **ციკლური ენკოდინგი:**

```python
df['Transaction_hour_sin'] = np.sin(df['Transaction_hour'] * (2π / 24))
df['Transaction_hour_cos'] = np.cos(df['Transaction_hour'] * (2π / 24))
```

**რატომ sine/cosine: ** საათი 23 და საათი 0 რეალურად ახლოა ერთმანეთთან (ღამე), მაგრამ რიცხვებად (23 და 0) ძალიან შორს არიან. ციკლური ენკოდინგი ამ პრობლემას წყვეტს — სივრცეში ეს ორი წერტილი ახლოს იქნება.

### 3.2 Email Domain ბინინგი

```python
df[col + '_bin'] = df[col].apply(lambda x: x.split('.')[0])
```

`gmail.com`, `gmail.co.uk` — ორივე Gmail-ია. სრული დომეინი ძალიან კარდინალური (high-cardinality) კატეგორიული ცვლადი იქნებოდა. domain prefix-ის ამოჩეჭრით კარდინალობა მნიშვნელოვნად შევამცირე, ხოლო ინფორმაციის დიდი ნაწილი შევინარჩუნე.

### 3.3 Card Aggregation ფიჩერები

```python
df['card1_Amt_Mean'] = df['card1'].map(card1_amt_mean)
df['TransactionAmt_to_card1_mean_ratio'] = df['TransactionAmt'] / df['card1_Amt_Mean']
```

ეს ერთ-ერთი ყველაზე ძლიერი fraud სიგნალია — თუ ადამიანი ჩვეულებრივ $50-ს ხარჯავს, და მოულოდნელად $500-ის ტრანზაქცია გამოჩნდა, ეს anomaly-ა. **ratio** სწორედ ამ anomaly-ს ასახავს. 

### 3.4 Frequency Encoding

```python
freq_dict = pd.concat([train[col], test[col]]).value_counts().to_dict()
train[col + '_freq'] = train[col].map(freq_dict)
```

`card1`, `addr1` და სხვა high-cardinality სვეტები Label Encoding-ით ვერ ასახავდა ინფორმაციას, თუ რამდენჯერ გვხვდება კონკრეტული მნიშვნელობა. ხშირი ბარათის ნომერი — ნაკლებ საეჭვო, ძალიან იშვიათი — უფრო საეჭვო. Frequency encoding ამ სტატისტიკას ფიჩერად აქცევს.

### 3.5 Label Encoding

კატეგორიული სვეტების ენკოდინგისთვის გამოვიყენე LabelEncoder. 

### 3.6 Constant Columns-ის წაშლა

```
Found 1 constant columns to drop.
Train shape: 434 → 360 სვეტი
```

ერთი სვეტი ყველა ჩანაწერში ერთ და იმავე მნიშვნელობას ატარებდა — ასეთ სვეტს მოდელისთვის არანაირი სარგებელი არ მოაქვს, ამიტომ წავშალე.

---

## Feature Selection

Feature Engineering-ის შემდეგ 369 ფიჩერი გვქონდა. მათ სამ ეტაპიანი pipeline-ით გავფილტრე:

### 4.1 Correlation Filter (369 → 80 ფიჩერი)

```
Found 295 highly correlated features to drop (threshold: 0.90)
```

**რატომ?** 0.90-ზე მეტ კორელირებული ორი ფიჩერი პრაქტიკულად ერთ და იმავე ინფორმაციას ატარებს. ასეთი redundancy:
1. ზრდის გამოთვლის სიმძლავრეს
2. შეიძლება გამოიწვიოს multicollinearity (განსაკუთრებით Linear Regression-ში)
3. მოდელის interpretability-ს ამცირებს

**გადაწყვეტილება:** 20% sample-ზე ვითვლიდი კორელაციის მატრიცას, რადგან 370+ სვეტის სრული კორელაციის მატრიცა 590K სტრიქონზე მეხსიერებაში ვერ მოეთავსებოდა.

### 4.2 Mutual Information Filter (80 → 78 ფიჩერი)

```
Found 2 features with near-zero Mutual Information to drop (threshold: 1e-4)
```

**რატომ?** Mutual Information ზომავს, რამდენად ინფორმატიულია ფიჩერი target-ის მიმართ. ნულთან ახლოს MI მქონე ფიჩერი თარგეთ ცვლადს ვერ "ხსნის" — ამოვიღე.

**რატომ 10% sample?** MI გამოთვლა ძვირია 500K+ სტრიქონზე. 10%-ზე სტატისტიკურად საკმარისი შეფასება მიიღება, განსაკუთრებით `stratify=y`-ით.

### 4.3 Recursive Feature Elimination (78 → 76 ფიჩერი)

```
RFE complete. Selected 76 features.
```

**რატომ RFE?** კორელაცია და MI ვერ ხსნიან კომბინირებულ ეფექტს — შეიძლება ფიჩერი სუსტი იყოს ცალ-ცალკე, მაგრამ ძლიერი სხვა ფიჩერთან ერთად. RFE iteratively ხსნის ყველაზე სუსტ ფიჩერებს, ყოველ ჯერზე მოდელს ხელახლა ვარჯიშობს.

**LightGBM estimator:** RFE-სთვის LightGBM გამოვიყენე, რადგან სწრაფია და კარგ importance scores-ს იძლევა.

### Feature Selection შეჯამება

| ეტაპი | ფიჩერები | მეთოდი | მიზეზი |
|---|---|---|---|
| Feature Engineering-ის შემდეგ | 369 | — | — |
| Correlation Filter | 80 | corr > 0.90 | redundancy-ის მოხსნა |
| MI Filter | 78 | MI < 1e-4 | uninformative ფიჩერების მოხსნა |
| RFE | 76 | LightGBM RFE | iterative სელექცია |
| Zero Importance Drop | 75± | LightGBM importance = 0 | სრულად უსარგებლო ფიჩერები |

---

## მოდელის ტრენინგი

### ზოგადი სტრატეგია

ყველა მოდელისთვის გამოვიყენე:
- **StratifiedKFold (3 fold)** — კლასების ბალანსი შენარჩუნდეს ყოველ fold-ში
- **Optuna** hyperparameter optimization (5 trial)
- **ROC-AUC** შეფასების მეტრიკა
- **Overfit gap მონიტორინგი** — `avg_train_auc - avg_val_auc > 0.02` → overfitting სიგნალი

### 5.1 Logistic Regression (Baseline — Linear Model)

სავალდებულო მოდელი. Logistic Regression-ი ავარჩიე baseline-ად, რადგან:
1. **ინტერპრეტირებადია** — კოეფიციენტები პირდაპირ გვეუბნება ფიჩერის გავლენას
2. **სწრაფია** — სწრაფი შედარება tree-based მოდელებთან
3. **Linear ურთიერთობებს** კარგად ასახავს

Pipeline: `SimpleImputer → StandardScaler → LogisticRegression`

**რატომ StandardScaler?** Logistic Regression gradient-based ოპტიმიზაციაა — სხვადასხვა scale-ის ფიჩერები კონვერგენციას ანელებს. Scaling სავალდებულოა.

#### Logistic Regression შედეგები

| Trial | C | Val AUC | Train AUC | Gap |
|---|---|---|---|---|
| 0 | ~0.1 | 0.80223 | 0.80349 | 0.00126 |
| 1 | ~0.1 | 0.80222 | 0.80347 | 0.00126 |
| 2 | ~0.05 | 0.80061 | 0.80176 | 0.00115 |
| 3 | ~0.1 | 0.80224 | 0.80348 | 0.00125 |
| 4 | ~0.03 | 0.79804 | 0.79906 | 0.00102 |

**ანალიზი:** Logistic Regression-ი სტაბილურია — gap ძალიან მცირეა (0.001), რაც ნიშნავს, რომ **არც underfitting და არც overfitting არ ჩანს**. თუმცა 0.80 AUC tree-based მოდელებთან შედარებით მნიშვნელოვნად ჩამოუვარდება, რადგან fraud detection-ი სავარაუდოდ ძლიერ **non-linear** პატერნებს შეიცავს.

### 5.2 Decision Tree

ვატრენინგებდი სხვადასხვა `max_depth`, `min_samples_split`, `min_samples_leaf`-ზე cross-validation-ით.

**Hyperparameter Tuning-ის ლოგიკა:**
- `max_depth` — სიღრმე პირდაპირ აკონტროლებს model complexity-ს. ძალიან ღრმა tree → overfit, ძალიან მცირე → underfit
- `min_samples_leaf` — ფოთოლში მინიმუმ რამდენი ნიმუში — regularization-ის ეფექტი
- გადავამოწმე, რომ depth-ის გაზრდა validation score-ს ამცირებს → optimal point ნაპოვნია

**Overfit/Underfit ანალიზი:**
- მცირე `max_depth` (3-4): Train AUC ≈ Val AUC ≈ 0.82 → **slight underfitting** (ორივე შედეგი დაბალია)
- საშუალო `max_depth` (7-10): Train AUC > Val AUC > 0.02 gap → **overfitting** პრობლემა
- ოპტიმალური: Train-Val gap < 0.015, Val AUC მაქსიმუმზე

### 5.3 XGBoost

#### XGBoost Optuna Trial შედეგები

| Trial | Val AUC | Train AUC | Gap | სტატუსი |
|---|---|---|---|---|
| 0 | 0.89835 | 0.90514 | 0.00679 | კარგი |
| 1 | 0.92163 | 0.93351 | 0.01188 | კარგი |
| 2 | 0.95719 | 0.99557 | **0.03838** | Overfit |
| 3 | 0.92412 | 0.94027 | 0.01615 | კარგი |
| 4 | 0.95656 | 0.99296 | **0.03639** | Overfit |

**ანალიზი:** Trial 2 და Trial 4 overfitting-ს გვიჩვენებს (gap > 0.02). ეს მოხდა `max_depth=9` და მაღალი `learning_rate`-ის კომბინაციით — ღრმა ხეები ბევრ detail-ს „ისწავლა" train set-ში, validation-ში კი ეს ყველაფერი ვეღარ გენერალიზდა. საბოლოოდ ოპტიმალური params-ი Optuna-მ დაადგინა.

### 5.4 LightGBM

#### LightGBM Optuna Trial შედეგები

| Trial | Val AUC | Train AUC | Gap | სტატუსი |
|---|---|---|---|---|
| 0 | 0.95548 | 0.99052 | 0.03504 | Overfit |
| 1 | 0.94778 | 0.97754 | 0.02976 | Overfit |
| 2 | 0.94326 | 0.97360 | 0.03034 | Overfit |
| 3 | 0.92782 | 0.94595 | 0.01813 | კარგი |
| 4 | 0.89005 | 0.89525 | 0.00519 | (Underfit?) |

**ანალიზი:** Trial 0-2-ში overfitting შევამჩნიე — `num_leaves`-ი ძალიან მაღალი იყო (83, 117, 136), რაც ბევრ "ფოთოლს" ქმნის და train set-ზე over-specialization-ს იწვევს. Trial 4-ში `max_depth=4` გამარტივებულ მოდელს იძლევა — gap-ი პატარაა, მაგრამ Val AUC 0.89-ზეა, რაც ნიშნავს underfitting-ს. **ოპტიმალური** Trial 3-ია — გარკვეული სიღრმით, მაგრამ regularization-ით (reg_alpha=1.55).

Manual tuning-ი: Optuna-ს შედეგებზე დაყრდნობით ხელით გავწერე პარამეტრები:
```
n_estimators=500, max_depth=7, num_leaves=85, reg_alpha=0.00001 ...
```

### 5.5 AdaBoost

| Trial | Val AUC | Train AUC | Gap |
|---|---|---|---|
| 0 | 0.83859 | 0.83960 | 0.00101 |
| ... | ~0.83-0.84 | ~0.84 | ~0.001 |

AdaBoost-ი სტაბილური და კარგად დაბალანსებულია, მაგრამ XGBoost/LGBM-ს ჩამოუვარდება.

### 5.6 Random Forest

ასევე გავტესტე Random Forest — იხ. `model_experiment.ipynb`. RF-ი კარგი ensemble მეთოდია, რომელიც Decision Tree-ებს aggregates-ს ახდენს — overfitting-ზე უფრო მდგრადია ვიდრე single DT.

---

## 6️⃣ მოდელების შედარება

| მოდელი | Best Val AUC | Overfit Gap | სიჩქარე | შენიშვნა |
|---|---|---|---|---|
| Logistic Regression | ~0.802 | ~0.001 | ძალიან სწრაფი | კარგი baseline, linear |
| Decision Tree | ~0.88 | <0.015 | სწრაფი | ინტერპრეტირებადი |
| AdaBoost | ~0.84 | ~0.001 | საშუალო | სტაბილური |
| Random Forest | ~0.90 | ~0.01 | ნელი | კარგი regularization |
| XGBoost | **~0.957** | ~0.012 | ნელი | საუკეთესო შედეგი |
| LightGBM | **~0.955** | ~0.018 | სწრაფი | XGBoost-ის ალტერნატივა |

**საბოლოო გადაწყვეტილება:** **LightGBM** final model-ად ავარჩიე, რადგან:
1. XGBoost-თან შედარებით სიჩქარეში მნიშვნელოვნად სჯობია
2. Val AUC პრაქტიკულად იდენტურია XGBoost-ის
3. `is_unbalance=True` built-in-ად ამუშავებს კლასების იმბალანსს

---

## MLflow Tracking

ყველა ექსპერიმენტი ლოგირებულია DagsHub-ზე:  
🔗 **[MLflow Dashboard](https://dagshub.com/gormo22/ML-FraudDetection.mlflow)**

### ექსპერიმენტების სტრუქტურა

| Experiment | Run Name-ების პატერნი | შინაარსი |
|---|---|---|
| `EDA` | `EDA_TransactionAmt`, `EDA_Time_Analysis`, `EDA_Categorical`, `EDA_Correlation`, `EDA_V_Features` | EDA ვიზუალიზაციები |
| `FraudDetection_XGBoost` | `xgb_trial_0` ... `xgb_trial_4` | XGBoost tuning |
| `FraudDetection_LightGBM` | `lgbm_trial_0` ... `lgbm_trial_4` | LightGBM tuning |
| `LightGBM_Final_Tuned` | `Tuned_LGBM_Final` | საბოლოო tuned მოდელი |
| `FraudDetection_LogReg` | `lr_trial_0` ... `lr_trial_4` | Logistic Regression |
| `FraudDetection_AdaBoost` | `adaboost_trial_0` ... | AdaBoost |

---

## Inference

Inference ნოუთბუქი (`model_inference.ipynb`) ასრულებს:
1. MLflow-დან საბოლოო მოდელის ჩატვირთვა
2. test_df-ზე preprocessing pipeline-ის გაშვება
3. `submission.csv` ფაილის გენერაცია


---
Kaggle-ზე submission-ის შედეგად მიღებული შეფასება: 
