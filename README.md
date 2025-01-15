# Phishing URL Detection Project

This project consists of two Python notebooks that focus on detecting phishing URLs using machine learning techniques. The first notebook, `implementation_feature extraction from URL.ipynb`, is responsible for extracting features from URLs, while the second notebook, `Model training and data analysis.ipynb`, handles the training and evaluation of machine learning models using the extracted features.

## Notebook 1: `implementation_feature extraction from URL.ipynb`

### Overview
This notebook is designed to extract various features from URLs that can be used to determine whether a URL is legitimate or phishing. The features extracted include:
- Presence of an IP address in the URL
- URL length
- Use of URL shortening services
- Presence of '@' symbol
- Prefix and suffix in the domain
- Number of subdomains
- SSL certificate details
- Domain registration length
- Favicon presence
- HTTPS token
- Request URL analysis
- URL of anchor tags
- Links in meta, script, and link tags
- Email submission forms
- Age of the domain
- Web traffic and page rank

### Key Functions
- `url_having_ip(url)`: Checks if the URL contains an IP address.
- `url_length(url)`: Determines the length of the URL.
- `url_short(url)`: Checks if the URL is shortened.
- `having_at_symbol(url)`: Checks for the presence of '@' in the URL.
- `prefix_suffix(url)`: Checks for prefix or suffix in the domain.
- `sub_domain(url)`: Counts the number of subdomains.
- `SSL_final_state(url)`: Evaluates the SSL certificate.
- `domain_registration(url)`: Checks the domain registration length.
- `fav(url)`: Checks for the presence of a favicon.
- `https_token(url)`: Checks for the presence of 'https' in the domain.
- `request_url(url)`: Analyzes the request URL.
- `url_of_anchor(url)`: Analyzes the URL of anchor tags.
- `links_in_tags(url)`: Counts links in meta, script, and link tags.
- `email_submit(url)`: Checks for email submission forms.
- `age_of_domain(url)`: Determines the age of the domain.
- `web_traffic(url)`: Analyzes web traffic.
- `page_rank(url)`: Determines the page rank.

### Usage
To use this notebook, simply input a URL, and the notebook will extract the relevant features and output a list of feature values that can be used for further analysis or model training.

## Notebook 2: `Model training and data analysis.ipynb`

### Overview
This notebook focuses on training and evaluating machine learning models using the features extracted from URLs. The models used include:
- XGBoost
- Random Forest
- Support Vector Classifier (SVC)
- Gaussian Naive Bayes
- AdaBoost
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)

### Key Steps
1. **Data Preparation**: The dataset is loaded, and the target variable is prepared. The dataset is split into training and testing sets.
2. **Model Training**: Various models are trained using stratified k-fold cross-validation.
3. **Model Evaluation**: The models are evaluated based on their F1 scores, and the results are visualized.
4. **Feature Selection**: Variance thresholding is applied to select the most relevant features.
5. **Model Saving**: The best-performing model (Random Forest) is saved using `joblib`.

### Key Functions
- `skfold(X, y, model)`: Performs stratified k-fold cross-validation.
- `print_acc(results, model)`: Prints the F1 score and standard deviation of the model.
- `VarianceThreshold`: Applies variance thresholding for feature selection.
- `RandomForestRegressor`: Trains a Random Forest model.

### Usage
To use this notebook, ensure that the dataset is correctly loaded and preprocessed. The notebook will automatically train and evaluate the models, and the best model will be saved for future use.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

## How to Run
1. Ensure all dependencies are installed.
2. Run the `implementation_feature extraction from URL.ipynb` notebook to extract features from URLs.
3. Use the extracted features to train and evaluate models in the `Model training and data analysis.ipynb` notebook.
4. Save the best model for deployment.

## Conclusion
This project provides a comprehensive approach to detecting phishing URLs by extracting relevant features and training machine learning models. The Random Forest model performed the best, achieving an F1 score of 97.21%. The saved model can be used for real-time phishing URL detection.

For any questions or issues, please refer to the documentation or contact the project maintainers.
