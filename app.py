from flask import Flask, request, jsonify, render_template

from datetime import datetime
import folium
import os

from dam_break.dambreak_sim import DAMBREAK_SIM
from dam_break.dam_break import DAM_BREAK
import directory_manager

app = Flask(__name__)

def sim( latitude, longitude, pondRadius, nObj, tailingsVolume, tailingsDensity, maxTime, timeStep):
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
    pondRadius= 103.0
    nObj= 100
    tailingsVolume= 2685782.0
    tailingsDensity= 1594.0
    maxTime= 180.0
    timeStep= 0.2

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
            'demDirectory': demDirectory,
            'fileHandler': None
    }

    simulation = None
    fileHandler = None

    def get_sim_ID():
        ''' Generates a unique ID for a simulation'''
        time_stamp = datetime.now()
        fileID = '{}-{}'.format(damID,
                                time_stamp.strftime("%Y%m%d-%H%M%S"))
        return fileID


    ''' Runs the flood simulation for the selected point '''
    simID = get_sim_ID()
    simulation = DAM_BREAK(**simulationSettings)
    simulation._bVerbose = True
    simulation.run_simulation()
    (fileName,csvName) = simulation.save_results(damID,simID,fileHandler='')

    simRecord = simulation.get_database_record(simID)
    simRecord['File_Address'] = csvName
    simRecord['File_Handler'] = fileHandler
    simResultsHandler = DAMBREAK_SIM(srcInput=simRecord,bAbsolutePath=True,demDirectory=demDirectory)
    renderPath = os.path.dirname(csvName)
    # mask,maskX,maskY = simResultsHandler.fit_speed_mask(
    #                                             simResultsHandler.max_time(),
    #                                             resolution = mapResolution,
    #                                             skipPoints = mapSkipPoints)
    # maskPath = renderPath + '/speed_%s.png' % simID
    # simResultsHandler.save_mask(maskPath,mask,maskX,maskY)

    # # Create new sim info
    extent = simResultsHandler.get_lon_lat_bounds(maxTime=simResultsHandler.max_time())
    print(latitude, longitude, extent)
    return "done"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the values of a, b, and c from the form
        # a = float(request.form.get('a'))
        # b = float(request.form.get('b'))
        # c = float(request.form.get('c'))
        simSettings = {k:k.get(v,None) for k,v in request.form}
        print(simSettings)
        # Pass the values to the sim() function
        # result = sim()
        return render_template('result.html', result=0)
    else:
        m = folium.Map([-20.119722, -44.121389],zoom_start=15)

        # set the iframe width and height
        m.get_root().width = "100%"
        m.get_root().height = "100%"
        iframe = m.get_root()._repr_html_()
        return render_template('index.html',iframe = iframe)

if __name__ == '__main__':
    app.run(debug=True)
