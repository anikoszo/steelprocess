import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
import pickle

# read dataset
dataset = pd.read_csv('data/output.csv')
# fill all the None values in order to create usable feature from them
dataset['temp'] = dataset['end_t'].fillna(0)

# set variables
perfect_heat_stop = 1680
# the acceptable stop temperature could be less or more with this value
acceptable_range_value = 9


# check if one variable is within an acceptable range, no further than 9 from the perfect heat
def historical_flag_maker(temp):
    if perfect_heat_stop - acceptable_range_value <= temp <= perfect_heat_stop + acceptable_range_value:
        flag = "Perfect"
    elif temp < perfect_heat_stop - acceptable_range_value:
        flag = "Need more heat"
    else:
        flag = "Too much heat"
    return flag


# apply the flags on historical data
dataset['heat_flag'] = dataset['temp'].apply(historical_flag_maker)

X = dataset[['gas1']]
y = dataset['heat_flag']

# as it is highly imbalanced it needs to be oversampled
over_sampler = RandomOverSampler(random_state=42)
# create oversampled portions for test and train
X_resampled, y_resampled = over_sampler.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# use a tree based model
model = DecisionTreeClassifier(random_state=42)
model_fit = model.fit(X_train, y_train)

# pickle the model for later usage, in the simulator
with open('data/model.pkl', 'wb') as file:
    pickle.dump(model, file)

y_pred = model.predict(X_test)

# create metrics
acc = "{:.2%}".format(accuracy_score(y_test, y_pred))
prec = "{:.2%}".format(precision_score(y_test, y_pred, average="weighted"))
recall = "{:.2%}".format(recall_score(y_test, y_pred, average="weighted"))

print(f" Accuracy :{acc}")
print(f" Precision :{prec}")
print(f" Recall :{recall}")

number_of_test_sample = 14
X_subset = X_test.iloc[:number_of_test_sample]

prediction = model.predict(X_subset)

# use this commented part to see the predictions
"""for i, (input_gas_value, prediction) in enumerate(zip(X_subset['gas1'], prediction)):
    print(f"Row {i + 1}: Input value {input_gas_value},{prediction}")

    if prediction == "Perfect":
        print(f" Input value {input_gas_value} Stop here, it is the perfect round")"""
