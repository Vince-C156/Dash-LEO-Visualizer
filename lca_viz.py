import requests
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import geopandas as gpd
import dash
from dash_canvas import DashCanvas

class SatelliteOrbitApp:
    def __init__(self, spacecraft=None):
        self.app = Dash(__name__)

        self.app.layout = html.Div([
            html.H4('Chaser and Target Orbits Over Time'),
            dcc.Graph(id='orbit-graph', style={'width': '90vw', 'height': '90vh'}),
            html.Button('Zoom on Chaser', id='btn-zoom-chaser', n_clicks=0),
            html.Button('Zoom on Target', id='btn-zoom-target', n_clicks=0),
            dcc.Interval(
                id='interval-component',
                interval=2 * 1000,  # in milliseconds
                n_intervals=3
            ),
            dcc.Store(id='camera-store'),
            DashCanvas(
                id='trajectory-canvas',
                lineWidth=2,
                hide_buttons=['zoom', 'pan', 'reset', 'save'],
                width='90vw',
                height='90vh'
            ),
            html.Div(id='time-step-display', children=0),  # Add time step display
        ])
        # Register the callback for the update_graph_live function
        @self.app.callback(
            [Output('orbit-graph', 'figure'),
             Output('camera-store', 'data'),
             Output('trajectory-canvas', 'json_data'),
             Output('time-step-display', 'children')],  # Update time step display
            [Input('interval-component', 'n_intervals'),
             Input('btn-zoom-chaser', 'n_clicks'),
             Input('btn-zoom-target', 'n_clicks')],
            [State('orbit-graph', 'relayoutData'),  # Get the current layout data
             State('camera-store', 'data'),  # Get the stored camera data
             State('time-step-display', 'children')]  # Get the current time step value
        )
        def update_graph_live_callback(n, zoom_chaser, zoom_target, relayoutData, camera_data, current_time_step):
            current_time_step = int(current_time_step) + 1  # Increment the time step
            return self.update_graph_live(n, zoom_chaser, zoom_target, relayoutData, camera_data, current_time_step)
        self.true_positions_chaser = []
        self.estimated_positions_chaser = []
        self.true_positions_target = []
        self.estimated_positions_target = []
        self.spacecraft = spacecraft


def update_positions(self, position_vector, is_chaser=True, is_true=True):
    if is_chaser:
        if is_true:
            self.true_positions_chaser.append(position_vector)
        else:
            self.estimated_positions_chaser.append(position_vector)
    else:
        if is_true:
            self.true_positions_target.append(position_vector)
        else:
            self.estimated_positions_target.append(position_vector)

    def update_graph_live(self, n, zoom_chaser, zoom_target, relayoutData, camera_data, current_time_step):
        fig = go.Figure()
        initial_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2, y=2, z=2)
        )
        self.plot_front(fig)
        self.plot_back(fig)
        self.plot_countries(fig)
        orbital_elements_chaser = (8000, 0.3, 45, 50, 70, 0)  # Example values (a, e, i, RAAN, omega, nu)

        orbital_elements_target = (8000, 0.2, 45, 30, 60, 0) # Example values (a, e, i, RAAN, omega, nu)

        positions_chaser = [self.update_orbital_position(orbital_elements_chaser, t) for t in range(0, n * 1, 1)]
        positions_target = [self.update_orbital_position(orbital_elements_target, t) for t in range(0, n * 1, 1)]
        x_chaser, y_chaser, z_chaser = np.transpose(positions_chaser)
        x_target, y_target, z_target = np.transpose(positions_target)

        # Add spheres for chaser and target
        sphere_radius = 2.5  # Adjust the radius as needed
        fig.add_trace(go.Scatter3d(x=[x_chaser[-1]], y=[y_chaser[-1]], z=[z_chaser[-1]], mode='markers', marker=dict(size=sphere_radius, color='red'), name='Chaser'))
        fig.add_trace(go.Scatter3d(x=[x_target[-1]], y=[y_target[-1]], z=[z_target[-1]], mode='markers', marker=dict(size=sphere_radius, color='blue'), name='Target'))

        fig.add_trace(go.Scatter3d(x=x_chaser, y=y_chaser, z=z_chaser, mode='lines', line=dict(color='red'), name='Chaser'))
        fig.add_trace(go.Scatter3d(x=x_target, y=y_target, z=z_target, mode='lines', line=dict(color='blue'), name='Target'))

        ctx = dash.callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'btn-zoom-chaser' and positions_chaser:
                camera_position = self.calculate_camera_position(positions_chaser, -1, distance=1500)
                self.update_camera_view(fig, camera_position)
            elif button_id == 'btn-zoom-target' and positions_target:
                camera_position = self.calculate_camera_position(positions_target, -1, distance=1500)
                self.update_camera_view(fig, camera_position)
        if relayoutData and 'scene.camera' in relayoutData:
            camera_data = relayoutData['scene.camera']
        if camera_data:
            fig.update_layout(scene_camera=camera_data)
        else:
            fig.update_layout(scene_camera=initial_camera)

        # Update the trajectory data
        trajectory_data = [{'lineColor': 'red', 'points': positions_chaser},
                           {'lineColor': 'blue', 'points': positions_target}]
        trajectory_canvas_data = {'objects': trajectory_data}

        return fig, camera_data, trajectory_canvas_data, current_time_step  # Return the updated time step

    def calculate_camera_position(self, positions, focus_index, distance=1000):
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

    def update_camera_view(self, fig, camera_position):
        if camera_position:
            fig.update_layout(scene_camera=camera_position, scene_aspectmode='data')

    def plot_back(self, fig):
        clor = 'rgb(220, 220, 220)'
        R = 6371  # Earth's radius in kilometers
        u_angle = np.linspace(0, np.pi, 25)
        v_angle = np.linspace(0, 2 * np.pi, 25)
        x_dir = R * np.outer(np.cos(u_angle), np.sin(v_angle))
        y_dir = R * np.outer(np.sin(u_angle), np.sin(v_angle))
        z_dir = R * np.outer(np.ones(np.size(u_angle)), np.cos(v_angle))
        fig.add_surface(z=z_dir, x=x_dir, y=y_dir, colorscale=[[0, clor], [1, clor]], opacity=1.0, showlegend=False)

    def plot_polygon(self, fig, poly):
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

    def plot_countries(self, fig):
        gdf = gpd.read_file("ne_110m_admin_0_countries.shp")
        for i in gdf.index:
            polys = gdf.loc[i].geometry
            if polys.geom_type == 'Polygon':
                self.plot_polygon(fig, polys)
            elif polys.geom_type == 'MultiPolygon':
                for poly in polys.geoms:
                    self.plot_polygon(fig, poly)

    def plot_front(self, fig):
        clor = 'rgb(220, 220, 220)'
        R = 6371  # Earth's radius in kilometers
        u_angle = np.linspace(-np.pi, 0, 25)
        v_angle = np.linspace(0, 2 * np.pi, 25)
        x_dir = R * np.outer(np.cos(u_angle), np.sin(v_angle))
        y_dir = R * np.outer(np.sin(u_angle), np.sin(v_angle))
        z_dir = R * np.outer(np.ones(np.size(u_angle)), np.cos(v_angle))
        fig.add_surface(z=z_dir, x=x_dir, y=y_dir, colorscale=[[0, clor], [1, clor]], opacity=1.0, showlegend=False)

    def get_tle(self, url):
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

    def orbital_elements_to_eci(self, a, e, i, RAAN, omega, nu):
        i = np.radians(i)
        RAAN = np.radians(RAAN)
        omega = np.radians(omega)
        nu = np.radians(nu)
        r = a * (1 - e ** 2) / (1 + e * np.cos(nu))
        x_orbital = r * np.cos(nu)
        y_orbital = r * np.sin(nu)
        x_eci = (np.cos(RAAN) * np.cos(omega) - np.sin(RAAN) * np.sin(omega) * np.cos(i)) * x_orbital \
                + (-np.cos(RAAN) * np.sin(omega) - np.sin(RAAN) * np.cos(omega) * np.cos(i)) * y_orbital
        y_eci = (np.sin(RAAN) * np.cos(omega) + np.cos(RAAN) * np.sin(omega) * np.cos(i)) * x_orbital \
                + (-np.sin(RAAN) * np.sin(omega) + np.cos(RAAN) * np.cos(omega) * np.cos(i)) * y_orbital
        z_eci = np.sin(i) * np.sin(omega) * x_orbital + np.sin(i) * np.cos(omega) * y_orbital
        return np.array([x_eci, y_eci, z_eci])

    def update_orbital_position(self, orbital_elements, t):
        a, e, i, RAAN, omega, nu = orbital_elements
        nu_updated = nu + t * np.sqrt(a ** 3) * 360 / (2 * np.pi)
        return self.orbital_elements_to_eci(a, e, i, RAAN, omega, nu_updated)

    def run(self):
        self.app.run_server(debug=True, port=8051)


if __name__ == '__main__':
    app = SatelliteOrbitApp()
    app.run()
