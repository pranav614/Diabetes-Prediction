from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def home(request):
    return render(request, 'home.html')


def prediction(request):
    return render(request, 'prediction.html')
def diet(request):
    return render(request,'diet.html')
def twofive(request):
    return render(request,'twofive.html')
def above25(request):
    return render(request,'above25.html')
def above50(request):
    return render(request,'above50.html')

def result(request):
    data = pd.read_csv(r"/Users/pranavreddy/Documents/mini project/diabetes.csv")
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='sag', max_iter=2000)
    model.fit(X_train, y_train)

    val1 = float(request.GET['l1'])
    val2 = float(request.GET['l2'])
    val3 = float(request.GET['l3'])
    val4 = float(request.GET['l4'])
    val5 = float(request.GET['l5'])
    val6 = float(request.GET['l6'])
    val7 = float(request.GET['l7'])
    val8 = float(request.GET['l8'])
    input_features = [[val1, val2, val3, val4, val5, val6, val7, val8]]
    predict = model.predict(input_features)

    output = ""
    if predict == 1:
        output = "positive"
    else:
        output = "negative"

    return render(request, 'prediction.html', {"result2": output})
