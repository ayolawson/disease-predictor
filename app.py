from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Disease treatments
treatments = {
    'Flu': (
        "1. Rest and drink plenty of fluids.\n"
        "2. Take antiviral medication like oseltamivir if prescribed early.\n"
        "3. Use fever reducers such as paracetamol.\n"
        "4. Isolate to avoid spreading infection.\n"
        "5. Seek medical help if symptoms worsen."
    ),
    'Asthma': (
        "1. Use inhalers: short-acting beta agonists for quick relief.\n"
        "2. Take corticosteroids for long-term control.\n"
        "3. Avoid triggers (dust, smoke, pollen).\n"
        "4. Monitor breathing and follow your asthma action plan.\n"
        "5. Visit a doctor if attacks become frequent."
    ),
    'Pneumonia': (
        "1. Use prescribed antibiotics for bacterial pneumonia.\n"
        "2. Take antipyretics and cough medicine as needed.\n"
        "3. Ensure bed rest and fluid intake.\n"
        "4. Oxygen therapy if breathing is difficult.\n"
        "5. Hospital admission for severe cases."
    ),
    'Anxiety': (
        "1. Cognitive Behavioral Therapy (CBT).\n"
        "2. Relaxation techniques: deep breathing, meditation.\n"
        "3. Regular exercise and good sleep.\n"
        "4. Avoid caffeine and alcohol.\n"
        "5. Medications like SSRIs if prescribed."
    ),
    'Depression': (
        "1. Psychotherapy: CBT or interpersonal therapy.\n"
        "2. Antidepressants such as SSRIs (e.g., fluoxetine).\n"
        "3. Engage in social and physical activities.\n"
        "4. Maintain a regular sleep schedule.\n"
        "5. Emergency support for suicidal thoughts (seek help immediately)."
    )
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = [
            int(request.form.get('symptom1', 0)),
            int(request.form.get('symptom2', 0)),
            int(request.form.get('symptom3', 0)),
            int(request.form.get('symptom4', 0)),
            int(request.form.get('symptom5', 0))
        ]

        prediction = model.predict([np.array(symptoms)])
        disease = prediction[0]
        treatment = treatments.get(disease, "No treatment info available.")
        return render_template('result.html', disease=disease, treatment=treatment)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)

