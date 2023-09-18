from flask import Flask, request, jsonify, render_template, Response, make_response
from flask_cors import CORS, cross_origin

import base64
from datetime import datetime
import io
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm

from dam_break.dambreak_sim import DAMBREAK_SIM
from dam_break.dam_break import DAM_BREAK

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def rgba_to_hex(rgba_color):
    """
    Convert RGBA color to hexadecimal color.

    :param rgba_color: Tuple or list containing RGBA values in the range [0, 1].
    :return: Hexadecimal color string (e.g., '#RRGGBB' or '#RRGGBBAA' for RGB or RGBA).
    """
    if len(rgba_color) < 3 or len(rgba_color) > 4:
        raise ValueError("RGBA color should be a tuple or list of 3 or 4 values.")
    
    rgba_color = [int(value * 255) for value in rgba_color[:3]]  # Convert to 8-bit values (0-255)
    
    if len(rgba_color) == 4:
        alpha = int(rgba_color[3] * 255)
        hex_color = "#{:02X}{:02X}{:02X}{:02X}".format(*rgba_color[0:3], alpha)
    else:
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgba_color)
    
    return hex_color


def get_image_data(mask: np.ndarray, X: np.ndarray, Y: np.ndarray, cbar, **kwargs):

    x_width = np.abs(np.min(X) - np.max(X))
    y_width = np.abs(np.min(Y) - np.max(Y))
    
    cmap = cm.get_cmap(cbar)
    # Define the number of contour levels you want
    # Generate contour levels based on your data

    custom_colors = [rgba_to_hex(cmap(level)) for level in np.linspace(0,1,256)]
    fig = go.Figure(data =
        go.Contour(
            z = mask.T,
            showscale=False,
            opacity=0.8,
            colorscale=custom_colors,
            colorbar=dict(**kwargs)
        ))
    
    
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(
        width=1000, 
        height=int(1000 * y_width/x_width), 
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # colorbar_fig = go.Figure(data=fig.data, layout=fig.layout)
    # colorbar_fig.update_layout(width=50, height=250, margin=dict(l=0, r=0, t=0, b=0))
    # colorbar_fig.update_layout(showlegend=False)
    # colorbar_fig.update_layout(plot_bgcolor='rgba(0,0,0,1)', paper_bgcolor='rgba(0,0,0,1)')
    # colorbar_fig.show()
    
    image_data = fig.to_image(format="png")
    # colorbar_data = colorbar_fig.to_image(format="png")
    
    return base64.b64encode(image_data).decode('utf-8'), f"{np.nanmin(mask):.3f}", f"{np.nanmax(mask):.3f}"


def sim(latitude,longitude, pondRadius, nObj, tailingsVolume, tailingsDensity, dampingFactor, maxTime, timeStep, cbar):

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
            'dampingFactor': dampingFactor,
            'fileHandler': None
    }

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
        'energy': eMask,
        'depth': depthMask
    }
        
    for name, mask in masks.items():
        
        # results[name] =  get_image_data(mask,X,Y,cbar,title = units[name])
        results[name] =  {i:j for i,j in zip(['img','mn','mx'],get_image_data(mask,X,Y,cbar))}
        # results[name], results[f"colorbar-{name}"] = get_image_data(mask,X,Y,title = units[name])

    return results

@app.route('/sim', methods=['POST'])
@cross_origin()
def start():
    json_data = request.json
    
    try:
        for k,v in json_data.items():
            if k != 'cbar':
                json_data[k] = float(v)
                print(k,v)

    except Exception as e:
        print(e)
        return jsonify(
            {'status': 1,'error':e}
        )
    
    data = sim(**json_data)
    data['status'] = 0
    response = make_response(jsonify(data))
    return response
    
@app.route('/')    
@cross_origin()
def check():
  return "IM AWARE BACKEND"
    
if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)