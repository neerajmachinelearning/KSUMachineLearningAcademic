from flask import Flask, request, render_template
from flask_restful import Resource, Api
from flask import abort

# from flask import Flask, request, render_template
#create an object of flask
app = Flask(__name__)
# pass app object to the function  API and create an object of API function api
api = Api(app)

class checkinestimation(Resource):
    def post(self):
        # get the value from the request posted.
        data = request.get_json()
        # get teh value of dataset in the request
        seatCapacityCnt = data["seatCapacityCnt"]
        allowedCarryOn = data["allowedCarryOn"]
        bookedPassengerCnt = data["bookedPassengerCnt"]
        bagsExpected = data["bagsExpected"]
        bagsperpassengerpercent = data["bagsperpassengerpercent"]
        X_new = [seatCapacityCnt, allowedCarryOn, bookedPassengerCnt, bagsExpected, bagsperpassengerpercent ]
       #handle exception
        if not seatCapacityCnt or not allowedCarryOn or not bookedPassengerCnt or not bagsExpected or not bagsperpassengerpercent:
            abort(400, 'Please enter all the mandatory inputs seatCapacityCnt, allowedCarryOn, bookedPassengerCnt, bagsExpected, bagsperpassengerpercent')
        else:
            accuracy, provide = mlmodelrf.main(X_new)
            return {'Accuracy': accuracy,
                'provide': provide}, 200

# create the resource path for the api
api.add_resource(checkinestimation, '/estimatecabin')

if __name__ == '__main__':
    app.run(debug=True)