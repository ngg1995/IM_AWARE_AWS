from flask import Flask, request, jsonify, render_template, Response, make_response
from flask_cors import CORS, cross_origin

import time
import base64
from datetime import datetime
import folium
import os
from io import BytesIO

from dam_break.dambreak_sim import DAMBREAK_SIM
from dam_break.dam_break import DAM_BREAK
import directory_manager

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def sim(pondRadius, nObj, tailingsVolume, tailingsDensity, maxTime, timeStep):
    drawOptions={
            'polyline':True,
            'rectangle':True,
            'polygon':True,
            'circle':False,
            'marker':True,
            'circlemarker':False}

    resultsDirectory = directory_manager.get_warehouse_dir()
    demDirectory = directory_manager.get_dem_dir()

    latitude= -20.119722
    longitude= -44.121389


    damID = 'default_dam'
    simID = 'default_sim'
    simulationSettings = {
            'siteLat': latitude,
            'siteLon': longitude,
            'pondRadius': pondRadius,
            'nObj': nObj,
            'tailingsVolume': tailingsVolume,
            'tailingsDensity': tailingsDensity,
            'maxTime': maxTime,
            'timeStep': timeStep,
            'dampingCoeff': 0.04,
            'fileHandler': None
    }

    simulation = None
    fileHandler = None



    ''' Runs the flood simulation for the selected point '''
    simID = f'{damID}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    simulation = DAM_BREAK(**simulationSettings)
    simulation._bVerbose = True
    simulation.run_simulation()

    simRecord = simulation.get_database_record(simID)

    simResultsHandler = DAMBREAK_SIM(simRecord,simulation.results_to_array())
    
    dMask,X,Y,vxMask,vyMask,vzMask,speedMask,altMask,eMask,depthMask = simResultsHandler.fit_all_masks(simResultsHandler.max_time())
    
    results = {}

    results['minLon'],results['maxLon'],results['minLat'],results['maxLat'] = simResultsHandler.get_lon_lat_bounds(maxTime=simResultsHandler.max_time())
    
    masks = {
        'speed': speedMask,
        'alt': altMask,
        'energy': eMask,
        'inundation': dMask > 0,
        'density': dMask,
        'depth': depthMask
    }
    for name, mask in masks.items():
        
        results[name] = simResultsHandler.get_image_data(mask,X,Y)

    
    return results


@app.route('/sim', methods=['POST'])
def start():
    json_data = request.json
    
    try:
        for k,v in json_data.items():
            json_data[k] = float(v)
            assert json_data[k] > 0
    except:
        return jsonify(
            {'status': 1}
        )
    
    data = sim(**json_data)
    data['status'] = 0
    response = make_response(jsonify(data))
    return response
    
@app.route('/test')    
@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"
    
if __name__ == '__main__':
    app.run(debug=True)
