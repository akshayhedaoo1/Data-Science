from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')



@app.route("/")
def index():
    return render_template('home.html')  

@app.route("/predict", methods=['POST', 'GET'])
def predict():
     model = joblib.load('CO2_Emission_Pred.pkl')
     if request.method=='POST':
          veh_class = request.form['vehicle_class']
          transmission = request.form['transmission']
          fuel_type = request.form['fuel_type']   
          consumption_city =  float(request.form['consumption_city'])
   
          if veh_class =='Van Cargo':
               van_cargo = 1
          else:
               van_cargo = 0
    
          if veh_class == 'Van Passenger':
               van_pass = 1
          else:
               van_pass = 0
          if transmission == 'AM9':
               trans = 1
          else:
               trans = 0
          if fuel_type == 'X':
               fuel_X = 1
          else:
               fuel_X = 0
          if fuel_type == 'Z':
               fuel_Z = 1
          else:
               fuel_Z = 0
    
          arr1= np.array([van_cargo, van_pass, trans, fuel_X, fuel_Z, 
                    consumption_city])
          pred = model.predict([arr1])       
          output = round(float(pred), 2)
          return render_template('home.html', prediction_text= 'CO2 Emission : {0}'.format(output))
     
     return render_template('home.html')



if __name__ == "__main__":
    app.run(debug =True)    