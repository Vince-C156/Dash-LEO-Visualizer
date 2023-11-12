# https://geopandas.org/en/stable/docs/user_guide/io.html
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

import requests
import plotly.graph_objects as go

def plot_back(fig):
    """back half of sphere"""
    clor=f'rgb(220, 220, 220)'
    R = np.sqrt(6368.134)
    u_angle = np.linspace(0, np.pi, 25)
    v_angle = np.linspace(0, np.pi, 25)
    x_dir = np.outer(R*np.cos(u_angle), R*np.sin(v_angle))
    y_dir = np.outer(R*np.sin(u_angle), R*np.sin(v_angle))
    z_dir = np.outer(R*np.ones(u_angle.shape[0]), R*np.cos(v_angle))
    fig.add_surface(z=z_dir, x=x_dir, y=y_dir, colorscale=[[0, clor], [1, clor]], opacity=1.0, showlegend=False, lighting=dict(diffuse=0.1)) # opacity=fig.sphere_alpha, colorscale=[[0, fig.sphere_color], [1, fig.sphere_color]])


def plot_front(fig):
    """front half of sphere"""
    clor=f'rgb(220, 220, 220)'
    R = np.sqrt(6368.134)
    u_angle = np.linspace(-np.pi, 0, 25)
    v_angle = np.linspace(0, np.pi, 25)
    x_dir = np.outer(R*np.cos(u_angle), R*np.sin(v_angle))
    y_dir = np.outer(R*np.sin(u_angle), R*np.sin(v_angle))
    z_dir = np.outer(R*np.ones(u_angle.shape[0]), R*np.cos(v_angle))
    fig.add_surface(z=z_dir, x=x_dir, y=y_dir, colorscale=[[0, clor], [1, clor]], opacity=1.0, showlegend=False, lighting=dict(diffuse=0.1)) # opacity=fig.sphere_alpha, colorscale=[[0, fig.sphere_color], [1, fig.sphere_color]])


def plot_polygon(poly):
    
    xy_coords = poly.exterior.coords.xy
    lon = np.array(xy_coords[0])
    lat = np.array(xy_coords[1])
    
    lon = lon * np.pi/180
    lat = lat * np.pi/180
    
    R = 6378.134
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    
    return x, y, z

def plot_orbit() :
    angle = np.linspace(0, 2.0*np.pi, 144)
    R = 6878.134
    x = R*np.cos(angle)
    y = R*np.sin(angle)
    z = np.zeros(144)
    
    return x, y, z

if __name__ == "__main__":
    # Read the shapefile.  Creates a DataFrame object
    gdf = gpd.read_file("ne_110m_admin_0_countries.shp")
    fig = go.Figure()
    plot_front(fig)
    plot_back(fig)

    marker = dict(color=[f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})' for _ in range(25)],
            size=10)

    for i in gdf.index :
        # print(gdf.loc[i].NAME)            # Call a specific attribute
        
        polys = gdf.loc[i].geometry         # Polygons or MultiPolygons 
        
        if polys.geom_type == 'Polygon':
            x, y, z = plot_polygon(polys)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=f'rgb(0, 0,0)'), showlegend=False) ) 
            
        elif polys.geom_type == 'MultiPolygon':
            
            for poly in polys.geoms:
                x, y, z = plot_polygon(poly)
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=f'rgb(0, 0,0)'), showlegend=False) ) 
                

    # Helix equation
    x, y, z = plot_orbit()

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=f'rgb(255, 0,0)'), showlegend=False ) ) 


    fig.write_html("3d_plot.html")
    fig.show()