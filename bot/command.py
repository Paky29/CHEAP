import json
import numpy as np
import pandas as pd
import sklearn
import slack
import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, Response
from slackeventsapi import SlackEventAdapter
from joblib import load


from data_cleaning.cleanSeera import save_dataset, scaling_robust
from ensemble import LinearRegression_MetaModel3

env_path = Path('./bot/') / '.env'
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(os.environ['SIGNING_SECRET'], '/slack/events', app)

client = slack.WebClient(token=os.environ['SLACK_TOKEN'])
BOT_ID = client.api_call("auth.test")['user_id']

@slack_event_adapter.on('message')
def message(payload):
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')

    if user_id != BOT_ID:
        if text == "Info":
            response = "Elenco feature\n-User Manual\n-Project Manager experience\n-Team size\n-H\\W"
            #client.chat_postMessage(channel=channel_id, text=response)

@app.route('/show-params', methods=['POST'])
def show_params():
    data = request.form
    channel_id = data.get('channel_id')
    print(data)
    response = "Elenco feature:\n1. Organization type\n2. Customer organization type\n3. Estimated duration\n4. Application domain\n5. Government policy impact\n6. Organization management structure clarity\n7. Developer hiring policy\n8. Developer training\n9. Development team management\n10. Requirements flexibility\n11. Project manager experience\n12. DBMS expert availability\n13. Precedentedness\n14. Software tool experience\n15. Team size\n16. Daily working hours\n17. Team contracts\n18. Schedule quality\n19. Degree of risk management\n20. Requirement accuracy level\n21. User manual\n22. Required reusability\n23. Product complexity\n24. Security requirements\n25. Specified H/W"
    client.chat_postMessage(channel=channel_id, text=response)
    return Response(), 200

@app.route('/info', methods=['POST'])
def info():
    data = request.form
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    text = data.get('text')

    sentence_mapping = {
        'Organization type': '1 = Public Software company\n2 = University\n3 = Federal Ministry\n4 = Federal directorates\n5 = Private Software company\n6 = Corporate IT department\n7 = Freelancer\n8 = Telecommunication company',
        'Customer organization type': '1 = University department\n2 = Department in a Private company\n3 = Bank\n4 = Federal Ministry\n5 = Department in a federal ministry\n6 = Factory\n7 = Hospital department\n8 = TV Channel\n9 = Non-profit organization\n10 = Private company\n11 = Hospital\n12 = Public company\n13 = In-house development\n14 = Federal directorates\n15 = Private school\n16 = Department in a bank',
        'Estimated duration': 'the estimated time (Months) to complete the project',
        'Application domain': '1 = Banking systems\n2 = ERP\n3 = Mobile applications\n5 = Financial and managerial\n6 = Web applications\n7 = Bespoke applications',
        'Government policy impact': '1 = Very positive impact\n2 = Positive impact\n3 = No impact\n4 = Negative impact\n5 = Very negative impact',
        'Organization management structure clarity': '1 = Organization management structure is clear and all procedures are clear\n2 = Organization management structure isn’t clear\n3 = No organization management structure exists',
        'Developer hiring policy': '1 = There exists hiring standards and applicant evaluations and are applied\n2 = No hiring standards but applicant evaluation is applied\n3 = There exists hiring standards and applicant evaluations but are not implemented – hiring acquaintances\n4 = No specific policy followed',
        'Developer training': '1 = Organization provides periodic training which was utilized\n2 = Developers were trained specifically for this project\n3 = No training provided',
        'Development team management': '(Fixed minimum working hours (0/1) + Time sheet recording (0/1) + Team work commitment (0/1)) + Consequence for lack of work + Absence policy implementation \n - Consequence for lack of work commitment: 1 = Dismissal from work; 2 = Salary deduction; 3 = Warning; 4 = No consequences \n - Absence policy implementation level: 1 = Rules and its applied; 2 = Rules and it’s not applied; 3 = No rules',
        'Requirements flexibility': 'If (Direct automation of the manual system = 1 ) then (Direct automation of the manual system * Impact on scheduleL * 5) \nOR \nIf (Direct automation of the manual system = 0) then (Direct automation of the manual system + Impact on schedule) (Mapped to 1/5)\n - Direct automation of the manual system (0/1) \n - Impact on schedule: 1 = Very positive impact; 2 = Positive impact; 3 = No impact; 4 = Negative impact; 5 = Very negative impact  ',
        'Project manager experience': '1 = Previous experience in similar software systems\n2 = Previous experience in non-similar software systems\n3 = No previous experience',
        'DBMS expert availability': '1 = Yes\n0 = No',
        'Precedentedness': 'Number of new software tools + New architecture (0/1) + Number of new complex algorithms ',
        'Software tool experience': '1 = More than 4 years\n2 = 2 years - 3 years\n3 = 1 year - 2 years\n4 = 6 months - 1 year\n5 = First time in this project',
        'Team size': 'Number of people in the team',
        'Daily working hours': 'The number of hours an individual is expected to work in a day',
        'Team contracts': 'percentage value based on the number of full-time employees, part-time employees, and employees engaged in training or national service. (Mapped to [1,5])',
        'Schedule quality': '1 = Devised a schedule and followed it with periodic evaluation\n2 = Devised a schedule with no periodic evaluation\n3 = Devised a schedule and did not follow it\n4 = No schedule',
        'Degree of risk management': 'Presence of risk plan (0/1) + Risk management tool usage (0/1) + 1',
        'Requirement accuracy level': '1 = Accurate requirements specifications used to develop the software system\n2 = Inaccurate requirements specifications and required the re-analysis of the software requirements\n3 = Inaccurate requirements specifications and required the re-design of the software system\n4 = Inaccurate requirements specifications and required re-programming the software system',
        'User manual': '1 = No user manual\n2 = User manual does not cover all the software system\n3 = Unclear user manual, written in technical terminology\n4 = Clear user manual that covers all the software system',
        'Required reusability': '1 = No reusing required\n2 = Reusing of some modules\n3 = Reusing of the complete software system to develop another software system\n4 = Customizations of the software system to be sold to other customers',
        'Product complexity': '1 = Clear and simple\n2 = Clear with some complexity\n3 = Complex\n4 = Involves very complex algorithms and difficult to understand',
        'Security requirements': 'Code security and encryption (0/1) + Database security (0/1) + Program security and encryption (0/1) + Basic authentication (0/1) + 1',
        'Specified H/W': '1 = Not required\n2 = Required specialized H/W that was available on time and we have prior experience with the H/W\n3 = Required specialized H/W that was available on time but we do not have prior experience with the H/W\n4 = Required specialized H/W that was not available on time'
    }

    if text in sentence_mapping:
        response = text + ":\n" + sentence_mapping[text]
    else:
        response='Invalid sentence'

    client.chat_postMessage(channel=channel_id, text=response)
    return Response(), 200

@app.route('/predict-format', methods=['POST'])
def print_format():
    data = request.form
    channel_id = data.get('channel_id')
    print(data)
    response = 'Organization type: X, Customer organization type: X, Estimated duration: X, Application domain: X, Government policy impact: X, Organization management structure clarity: X, Developer hiring policy: X, Developer training: X, Development team management: X, Requirements flexibility: X, Project manager experience: X, DBMS expert availability: X, Precedentedness: X, Software tool experience: X, Team size: X, Daily working hours: X, Team contracts: X, Schedule quality: X, Degree of risk management: X, Requirement accuracy level: X, User manual: X, Required reusability: X, Product complexity: X, Security requirements: X, Specified H/W: X'
    client.chat_postMessage(channel=channel_id, text=response)
    return Response(), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    print(data)
    text = data.get('text')

    # Split the string by commas
    feature_list = text.split(", ")

    # Extract only the values from the name-value pairs and convert them to floats
    input_data = [float(feature.split(": ")[1].strip()) for feature in feature_list]

    '''
    # Convert the input_data array to JSON string without newlines
    json_data = json.dumps({"input_data": input_data}, separators=(",", ":"))

    # Store the JSON data as a string variable
    json_string = str(json_data)

    # Print the JSON string
    print(json_data)
    client.chat_postMessage(channel=channel_id, text=json_string)'''


    seera = pd.read_csv("datasets/SEERA_retrain_try.csv", delimiter=',', decimal=".")
    print(seera)
    last_row_index = seera['Indice Progetto'].max() + 1
    input_data.insert(0, last_row_index)
    input_data.append("NaN")

    new_row = pd.Series(input_data, index=seera.columns)
    seera = pd.concat([seera, pd.DataFrame([new_row])], ignore_index=True)

    save_dataset(seera, 'SEERA_retrain_try.csv')

    seera = seera.drop(['Indice Progetto', 'Actual effort'], axis=1)
    seera = scaling_robust(seera)

    input_scaled = seera.tail(1)
    rf_regressor = load("models_saved/random_forest.joblib")
    svr = load("models_saved/svr.joblib")
    gb_regressor = load("models_saved/gradient_boosting.joblib")
    knn_regressor = load("models_saved/knn.joblib")
    elasticnet_regressor = load("models_saved/elastic_net.joblib")

    y_pred_rf = rf_regressor.predict(input_scaled)
    y_pred_svr = svr.predict(input_scaled)
    y_pred_gb = gb_regressor.predict(input_scaled)
    y_pred_knn = knn_regressor.predict(input_scaled)
    y_pred_elastic = elasticnet_regressor.predict(input_scaled)

    X_meta = np.column_stack((y_pred_gb, y_pred_rf, y_pred_svr, y_pred_knn, y_pred_elastic))
    meta_regressor = load('models_saved/meta_regressor.joblib')
    prediction = meta_regressor.predict(X_meta)

    print(prediction)

    #response = {'index': last_row_index, 'prediction': prediction.tolist()[0]}

    response = "The estimated effort for the project is: {}.\n The id assigned to you project is {}. \n You can use the id to let me know the actual effort once the project will be completed ".format(round(prediction[0],1), last_row_index)

    client.chat_postMessage(channel=channel_id, text=response)

    return Response(), 200

@app.route('/update', methods=['POST'])
def retraining():
    data = request.form
    user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    print(data)
    text = data.get('text')

    # Split the string by commas
    feature_list = text.split(", ")

    print(feature_list)
    # Extract only the values from the name-value pairs and convert them to integers
    input_data = [(feature.split(":")[1].strip()) for feature in feature_list]

    index = float(input_data[0])
    real_effort = round(float(input_data[1]), 1)


    seera = pd.read_csv("datasets/SEERA_retrain_try.csv", delimiter=',', decimal=".")

    # Retrieve the last row of the dataset
    row = seera.loc[seera['Indice Progetto']==index]

    row['Actual effort'] = real_effort

    # Update the modified last row in the dataset
    seera.loc[seera['Indice Progetto']==index] = row

    save_dataset(seera, "SEERA_retrain_try.csv")

    #seera.drop('Indice Progetto', axis=1)
    seera.dropna(subset=['Actual effort'])
    seera = scaling_robust(seera)

    save_dataset(seera,'SEERA_train_try.csv')
    seera = seera.drop('Indice Progetto',axis =1)
    LinearRegression_MetaModel3.run(seera.drop('Actual effort', axis=1), seera['Actual effort'])

    client.chat_postMessage(channel=channel_id, text="Thank you for your help")

    return Response(), 200


def run_bot():
    app.run(debug=True)