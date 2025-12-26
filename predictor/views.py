from django.shortcuts import render
import joblib
import numpy as np

# Load the model
model = joblib.load('placement_model.joblib')

def index(request):  
    result = None
    if request.method == 'POST':
        cgpa = float(request.POST.get('cgpa'))
        internships = int(request.POST.get('internships'))
        projects = int(request.POST.get('projects'))

        input_data = np.array([[cgpa, internships, projects]])
        prediction = model.predict(input_data)[0]
        
        result = "Placed! 🎉" if prediction == 1 else "Keep working! 💪"

    
    return render(request, 'predictor/index.html', {'result': result})
