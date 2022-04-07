import sys

from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
import os
from wtforms import IntegerField, SubmitField, RadioField, FloatField, StringField
from wtforms.validators import DataRequired, ValidationError, InputRequired
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sdflknvitgjwfpiiph3FG46'
Bootstrap(app)

basedir = os.path.abspath(os.path.dirname(__file__))
data_file = os.path.join(basedir, 'heart.csv')
df = pd.read_csv(data_file)


# -----------------labels encoding--------------
def encode_input(column, input_df):
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    input_df[column] = label_encoder.fit_transform(input_df[column])
    input_df[column].unique()


def encode_label(column):
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    df[column].unique()


encode_label("Sex")
encode_label("ChestPainType")
encode_label("RestingECG")
encode_label("ExerciseAngina")
encode_label("ST_Slope")

scaler = StandardScaler()
df1 = df.drop('HeartDisease', axis=1)
scaler.fit(df1)

scaled_features = scaler.transform(df1)
df_scaled = pd.DataFrame(scaled_features, columns=df1.columns[:])
X = df_scaled
y = df['HeartDisease']
knn = KNeighborsClassifier(n_neighbors=9, p=1)
knn.fit(X, y)


def validate_age(form, field):
    if field.data <= 0:
        raise ValidationError('age must be bigger than 0')


class NameForm(FlaskForm):
    age = IntegerField('Age', validators=[InputRequired(), validate_age])
    sex = RadioField('Sex', choices=['M', 'F'], validators=[InputRequired()])
    chestPainType = StringField('chestPainType (TA/ATA/NAP/ASY)', validators=[InputRequired()])
    restingBP = IntegerField('RestingBP', validators=[InputRequired()])
    cholesterol = IntegerField('Cholesterol', validators=[InputRequired()])
    fastingBS = IntegerField('FastingBS (0/1)', validators=[InputRequired()])
    restingECG = StringField('RestingECG (Normal/ST/LVH)', validators=[InputRequired()])
    maxHR = IntegerField('MaxHR (60-202)', validators=[InputRequired()])
    exerciseAngina = StringField('ExerciseAngina (N/Y)', validators=[InputRequired()])
    oldPeak = FloatField('OldPeak (0.0-4.5)', validators=[InputRequired()])
    st_slope = StringField('ST_Slope (UP/FLAT/DOWN)', validators=[InputRequired()])
    submit = SubmitField('submit')


@app.route('/', methods=['GET', 'POST'])
def hello():
    form = NameForm()
    message = ""
    print("before form validation")
    message = "before validation"

    if form.validate_on_submit():
        data = {}
        data['Age'] = form.age.data
        data['Sex'] = form.sex.data
        data['ChestPainType'] = form.chestPainType.data
        data['RestingBP'] = form.restingBP.data
        data['Cholesterol'] = form.cholesterol.data
        data['FastingBS'] = form.fastingBS.data
        data['RestingECG'] = form.restingECG.data
        data['MaxHR'] = form.maxHR.data
        data['ExerciseAngina'] = form.exerciseAngina.data
        data['Oldpeak'] = form.oldPeak.data
        data['ST_Slope'] = form.st_slope.data

        data_list = [data]
        input_df = pd.DataFrame(data=data_list)

        encode_input("Sex", input_df)
        encode_input("ChestPainType", input_df)
        encode_input("RestingECG", input_df)
        encode_input("ExerciseAngina", input_df)
        encode_input("ST_Slope", input_df)

        input_scaled_features = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(input_scaled_features, columns=input_df.columns[:])
        pred = knn.predict(scaled_input_df)
        print(int(pred[0]), file=sys.stdout)
        if (int(pred[0])) == 1:
            message = "Model predicts Presence of Heart Disease"
        else:
            message = "Model does not predict Presence of Heart Disease"

    return render_template('index.html', form=form, message=message)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
