import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Create a list to hold lists of evidence and values of labels
    evidence = []
    labels = []

    # Save a dict with month abbreviations and index
    months = {month: (index - 1) for index, month in enumerate(calendar.month_abbr)}

    # Open the file
    with open(filename) as f:
        # Read it as csv
        data = csv.reader(f)
        # Skip the headers
        next(data)
        # Loop over the rows in the reader
        for row in data:
            # Add the label value
            if row[-1] == 'TRUE':
                labels.append(1)
            else:
                labels.append(0)
            # Start editing the evidence values
            evidence_row = row[:-1]
            # Check every column
            for column in range(len(row)):
                # Check columns Administrative, Informational, ProductRelated, OperatingSystems, Browser, Region and TrafficType
                if column in (0, 2, 4, 11, 12, 13, 14):
                    evidence_row[column] = int(evidence_row[column])
                # Check column Weekend
                elif column == 16:
                    if evidence_row[column] == 'TRUE':
                        evidence_row[column] = 1
                    else:
                        evidence_row[column] = 0
                # Check columns Administrative_Duration, Informational_Duration, ProductRelated_Duration, BounceRates, ExitRates, PageValues, and SpecialDay
                elif column in (1, 3, 5, 6, 7, 8, 9):
                    evidence_row[column] = float(evidence_row[column])
                # Check column Month
                elif column == 10:
                    if evidence_row[column] == 'June':
                        evidence_row[column] = 5
                    else:
                        evidence_row[column] = months[evidence_row[column]]
                    continue
                # Check column VisitorType
                elif column == 15:
                    if evidence_row[column] == 'Returning_Visitor':
                        evidence_row[column] = 1
                    else:
                        evidence_row[column] = 0
            # Append the edited row
            evidence.append(evidence_row)
    
    # Return the tuple
    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Keep track of the total positive true values (positive), the total negative true values (negative) and
    # of this positive and negative values, keep track when they were correctly predicted
    sensitivity_count = 0
    specificity_count  = 0
    positive = 0
    negative = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            positive += 1
            if actual == predicted:
                sensitivity_count += 1
        
        if actual == 0:
            negative += 1
            if actual == predicted:
                specificity_count += 1

    # Sensitivity is the proportion of identified positive values in the total positive true values
    sensitivity = sensitivity_count / positive
    # Specificity is the proportion of identified negative values in the total negative true values
    specificity = specificity_count / negative 

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
