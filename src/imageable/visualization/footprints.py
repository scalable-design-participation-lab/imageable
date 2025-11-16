import json
import numpy as np
import pydeck as pdk
from typing import List

def _load_geojson(geojson: str) -> dict:
    if isinstance(geojson, str):
        with open(geojson, "r") as f:
            return json.load(f)
    return geojson


def _compute_view_state(geojson: dict) -> pdk.ViewState:
    coords = []
    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        if gtype == "Polygon":
            coords.extend(geom["coordinates"][0])
        elif gtype == "MultiPolygon":
            coords.extend(geom["coordinates"][0][0])
    if not coords:
        return pdk.ViewState(latitude=0, longitude=0, zoom=1, pitch=45, bearing=0)

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return pdk.ViewState(
        longitude=float(np.mean(lons)),
        latitude=float(np.mean(lats)),
        zoom=15,
        pitch=45,
        bearing=0,
    )


def visualize_heights(
    geojson: str,
    height_column: str = "building_heights",
    cmap=None,
    elevation_scale: float = 1.0,
    save_html: str = None,
    map_style="https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json",
    view_state: pdk.ViewState = None,
    default_color: List[int] = [200, 200, 200],
) -> pdk.Deck:
    data = _load_geojson(geojson)

    values = []
    for feature in data.get("features", []):
        v = feature.get("properties", {}).get(height_column)
        if v is not None:
            values.append(v)

    if values:
        min_height = float(np.min(values))
        max_height = float(np.max(values))
    else:
        min_height = 0.0
        max_height = 1.0

    for feature in data.get("features", []):
        properties = feature.setdefault("properties", {})
        v = properties.get(height_column)

        if v is None:
            r, g, b = default_color
            elevation = 0.0
        else:
            v = float(v)
            if cmap is not None and max_height > min_height:
                t = (v - min_height) / (max_height - min_height)
                r_f, g_f, b_f, *_ = cmap(t)
                r = int(255 * r_f)
                g = int(255 * g_f)
                b = int(255 * b_f)
            else:
                t = 0.0 if max_height == min_height else (v - min_height) / (max_height - min_height)
                r = int(255 * (1 - t))
                g = int(255 * t)
                b = 150
            elevation = v * elevation_scale

        properties["r"] = int(r)
        properties["g"] = int(g)
        properties["b"] = int(b)
        properties["_elevation"] = float(elevation)

    if view_state is None:
        view_state = _compute_view_state(data)

    layer = pdk.Layer(
        "GeoJsonLayer",
        data,
        pickable=True,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_fill_color='[properties.r*1, properties.g*1, properties.b*1]',
        get_elevation='properties._elevation*1',
        get_line_color=[0, 0, 0, 80],
    )

    tooltip = {
        "html": "<b>Height:</b> {" + height_column + "}",
        "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=map_style,
        tooltip=tooltip,
    )

    if save_html is not None:
        deck.to_html(save_html, notebook_display=False)

    return deck



def visualize_materials(
    geojson: dict,
    cmap = None,
    material_column_names: List[str] = [
        "mat_asphalt_pct",
        "mat_concrete_pct",
        "mat_metal_pct",
        "mat_road_markiing_pct",
        "mat_fabric_leather_pct",
        "mat_glass_pct",
        "mat_plaster_pct",
        "mat_plastic_pct",
        "mat_rubber_pct",
        "mat_sand_pct",
        "mat_gravel_pct",
        "mat_ceramic_pct",
        "mat_cobblestone_pct",
        "mat_brick_pct",
        "mat_grass_pct",
        "mat_wood_pct",
        "mat_leaf_pct",
        "mat_water_pct",
        "mat_human_body_pct",
        "mat_sky_pct"
    ],
    elevation_scale: float = 1.0,
    height_column: str = "building_heights",
    save_html: str = None,
    map_style="https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json",
    view_state: pdk.ViewState = None,
    default_color: List[int] = [200, 200, 200], 
    default_material_colors: List[List[int]] = [
        [160, 160, 160],        # asphalt
        [220, 220, 220],    # concrete
        [192, 192, 192],    # metal
        [255, 215, 0],      # road marking
        [139, 69, 19],      # fabric/leather
        [135, 206, 235],    # glass
        [245, 245, 220],    # plaster
        [255, 182, 193],    # plastic
        [0, 0, 0],          # rubber
        [210, 180, 140],    # sand
        [169, 169, 169],    # gravel
        [255, 228, 225],    # ceramic
        [112, 128, 144],    # cobblestone
        [178, 34, 34],      # brick
        [34, 139, 34],      # grass
        [160, 82, 45],      # wood
        [34, 139, 34],      # leaf
        [30, 144, 255],     # water
        [255, 224, 189],    # human body
        [135, 206, 250]     # sky
    ]
) -> pdk.Deck:
    #Open the data
    data = _load_geojson(geojson)
    if(cmap is not None):
        material_colors = []
        n = len(material_column_names)
        for i in range(0,n):
            r_f, g_f, b_f, *_ = cmap(i/max(n-1,1))
            material_colors.append([int(255 * r_f), int(255 * g_f), int(255 * b_f)])
    else:
        material_colors = default_material_colors
    

    #At this point we have the material colors, now we need to set some heights
    heights = []
    for feature in data.get("features", []):
        v = feature.get("properties", {}).get(height_column)
        if v is not None:
            heights.append(v)
    
    if heights:
        min_height = float(np.min(heights))
        max_height = float(np.max(heights))
    else:
        min_height = 0.0
        max_height = 1.0
    
    # Assign the values to the features
    for feature in data.get("features", []):
        props = feature.setdefault("properties", {})
        #We will obtain the dominant material
        values = []
        for col in material_column_names:
            v = props.get(col, None)
            values.append(float(v) if v is not None else 0.0)
        
        if all(v == 0.0 for v in values):
            r, g, b = default_color
            props["dominant_material"] = None
        else:
            idx = int(np.argmax(values))
            r, g, b = material_colors[idx]
            props["dominant_material"] = material_column_names[idx]
        
        height = props.get(height_column)
        if height is None:
            elevation = 0.0
        else:
            height = float(height)
            elevation = height * elevation_scale
        
        props["r"] = int(r)
        props["g"] = int(g)
        props["b"] = int(b)
        props["_elevation"] = float(elevation)

    if view_state is None:
        view_state = _compute_view_state(data)
    
    layer = pdk.Layer(
        "GeoJsonLayer",
        data,
        pickable=True,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_fill_color='[properties.r, properties.g, properties.b]',
        get_elevation='properties._elevation',
        get_line_color=[0, 0, 0, 80],
    )

    tooltip = {
        "html": """
        <b>Height:</b> {""" + height_column + """}<br/>
        <b>Dominant material:</b> {dominant_material}<br/>
        <b>building_ID:</b> {building_id}
        """,
        "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"},
    }

    deck = pdk.Deck(
        layers = [layer],
        initial_view_state = view_state,
        map_style = map_style,
        tooltip = tooltip
    )

    if save_html is not None:
        deck.to_html(save_html, notebook_display = False)

    return deck









