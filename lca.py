import requests
from dash import Dash, dcc, html, Input, Output, callback, State
import plotly.graph_objects as go
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import geopandas as gpd
import dash

def calculate_camera_position(positions, focus_index, distance=1000):
    """ Calculate the camera position to follow closely behind the satellite """
    if positions and len(positions) > 1:
        focus_point = positions[focus_index]
        prev_point = positions[max(focus_index - 1, 0)]
        direction_vector = np.array(focus_point) - np.array(prev_point)
        direction_vector /= np.linalg.norm(direction_vector)
        camera_pos = np.array(focus_point) - direction_vector * distance
        return {
            'eye': {'x': camera_pos[0], 'y': camera_pos[1], 'z': camera_pos[2]},
            'up': {'x': 0, 'y': 0, 'z': 1},
            'center': {'x': 0, 'y': 0, 'z': 0}
        }
    return None

def update_camera_view(fig, camera_position):
    """ Update the camera view of the plot """
    if camera_position:
        fig.update_layout(scene_camera=camera_position, scene_aspectmode='data')


def plot_back(fig):
    """Back half of sphere"""
    clor = 'rgb(220, 220, 220)'
    R = 6371  # Earth's radius in kilometers
    u_angle = np.linspace(0, np.pi, 25)
    v_angle = np.linspace(0, 2*np.pi, 25)
    x_dir = R * np.outer(np.cos(u_angle), np.sin(v_angle))
    y_dir = R * np.outer(np.sin(u_angle), np.sin(v_angle))
    z_dir = R * np.outer(np.ones(np.size(u_angle)), np.cos(v_angle))
    fig.add_surface(z=z_dir, x=x_dir, y=y_dir, colorscale=[[0, clor], [1, clor]], opacity=1.0, showlegend=False)

def plot_polygon(fig, poly):
    """ Add polygon to the plot """
    xy_coords = poly.exterior.coords.xy
    lon = np.array(xy_coords[0])
    lat = np.array(xy_coords[1])
    lon = lon * np.pi / 180
    lat = lat * np.pi / 180
    R = 6371  # Earth's radius in kilometers
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='rgb(0, 0, 0)'), showlegend=False))

def plot_countries(fig):
    """ Plot country boundaries """
    gdf = gpd.read_file("ne_110m_admin_0_countries.shp")
    for i in gdf.index:
        polys = gdf.loc[i].geometry
        if polys.geom_type == 'Polygon':
            plot_polygon(fig, polys)
        elif polys.geom_type == 'MultiPolygon':
            for poly in polys.geoms:
                plot_polygon(fig, poly)

def plot_front(fig):
    """Front half of sphere"""
    clor = 'rgb(220, 220, 220)'
    R = 6371  # Earth's radius in kilometers
    u_angle = np.linspace(-np.pi, 0, 25)
    v_angle = np.linspace(0, 2*np.pi, 25)
    x_dir = R * np.outer(np.cos(u_angle), np.sin(v_angle))
    y_dir = R * np.outer(np.sin(u_angle), np.sin(v_angle))
    z_dir = R * np.outer(np.ones(np.size(u_angle)), np.cos(v_angle))
    fig.add_surface(z=z_dir, x=x_dir, y=y_dir, colorscale=[[0, clor], [1, clor]], opacity=1.0, showlegend=False)

def get_tle(url):
    """ Fetch TLE data from the provided URL """
    try:
        r = requests.get(url)
        tle_data = r.text.split('\n')
        for i, line in enumerate(tle_data):
            if 'ISS' in line:
                tle_line1 = tle_data[i + 1].strip()
                tle_line2 = tle_data[i + 2].strip()
                return tle_line1, tle_line2
    except requests.RequestException as e:
        print("Error fetching TLE data:", e)
    return None, None

def orbital_elements_to_eci(a, e, i, RAAN, omega, nu):
    """ Convert orbital elements to ECI coordinates """
    # Convert angles from degrees to radians
    i = np.radians(i)
    RAAN = np.radians(RAAN)
    omega = np.radians(omega)
    nu = np.radians(nu)

    # Distance from the central body
    r = a * (1 - e**2) / (1 + e * np.cos(nu))

    # Position in orbital plane
    x_orbital = r * np.cos(nu)
    y_orbital = r * np.sin(nu)

    # Convert to 3D coordinates
    x_eci = (np.cos(RAAN) * np.cos(omega) - np.sin(RAAN) * np.sin(omega) * np.cos(i)) * x_orbital \
            + (-np.cos(RAAN) * np.sin(omega) - np.sin(RAAN) * np.cos(omega) * np.cos(i)) * y_orbital
    y_eci = (np.sin(RAAN) * np.cos(omega) + np.cos(RAAN) * np.sin(omega) * np.cos(i)) * x_orbital \
            + (-np.sin(RAAN) * np.sin(omega) + np.cos(RAAN) * np.cos(omega) * np.cos(i)) * y_orbital
    z_eci = np.sin(i) * np.sin(omega) * x_orbital + np.sin(i) * np.cos(omega) * y_orbital

    return np.array([x_eci, y_eci, z_eci])

def update_orbital_position(orbital_elements, t):
    """ Update the true anomaly and calculate the ECI position """
    a, e, i, RAAN, omega, nu = orbital_elements
    nu_updated = nu + t * np.sqrt(a**3) * 360 / (2 * np.pi)
    return orbital_elements_to_eci(a, e, i, RAAN, omega, nu_updated)

if __name__ == '__main__':
    app = Dash(__name__)

    app.layout = html.Div([
        html.H4('Chaser and Target Orbits Over Time'),
        dcc.Graph(id='orbit-graph', style={'width': '90vw', 'height': '90vh'}),
        html.Button('Zoom on Chaser', id='btn-zoom-chaser', n_clicks=0),
        html.Button('Zoom on Target', id='btn-zoom-target', n_clicks=0),
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # in milliseconds
            n_intervals=5
        ),
        dcc.Store(id='camera-store')
    ])

    @app.callback(
        [Output('orbit-graph', 'figure'),
        Output('camera-store', 'data')],
        [Input('interval-component', 'n_intervals'),
        Input('btn-zoom-chaser', 'n_clicks'),
        Input('btn-zoom-target', 'n_clicks')],
        [State('orbit-graph', 'relayoutData'),  # Get the current layout data
        State('camera-store', 'data')]  # Get the stored camera data
    )
    def update_graph_live(n, zoom_chaser, zoom_target, relayoutData, camera_data):
        fig = go.Figure()

        # Set initial camera position for a better overview
        initial_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2, y=2, z=2)  # You can adjust these values as needed
        )


        plot_front(fig)
        plot_back(fig)
        plot_countries(fig)

        orbital_elements_chaser = (6999.8, 0.001, 98.5, 0, 75, 0)  # Example values
        orbital_elements_target = (7000, 0.001, 98.5, 0, 75, 0)  # Example values

        positions_chaser = [update_orbital_position(orbital_elements_chaser, t) for t in range(0, n * 5, 1)]
        positions_target = [update_orbital_position(orbital_elements_target, t) for t in range(0, n * 5, 1)]

        x_chaser, y_chaser, z_chaser = np.transpose(positions_chaser)
        x_target, y_target, z_target = np.transpose(positions_target)

        fig.add_trace(go.Scatter3d(x=x_chaser, y=y_chaser, z=z_chaser, mode='lines', line=dict(color='red'), name='Chaser'))
        fig.add_trace(go.Scatter3d(x=x_target, y=y_target, z=z_target, mode='lines', line=dict(color='blue'), name='Target'))

        ctx = dash.callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'btn-zoom-chaser' and positions_chaser:
                camera_position = calculate_camera_position(positions_chaser, -1,  distance=1500)  # Focus on the last position of the chaser
                update_camera_view(fig, camera_position)
            elif button_id == 'btn-zoom-target' and positions_target:
                camera_position = calculate_camera_position(positions_target, -1, distance=1500)  # Focus on the last position of the target
                update_camera_view(fig, camera_position)

       # Check if the user has interacted with the camera (zoom or pan)
        if relayoutData and 'scene.camera' in relayoutData:
            camera_data = relayoutData['scene.camera']

        # Use the camera data if it's available
        if camera_data:
            fig.update_layout(scene_camera=camera_data)
        else:
            # Set a default camera position
            fig.update_layout(scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=2)
            ))

        return fig, camera_data
    app.run_server(debug=True, port=8051)