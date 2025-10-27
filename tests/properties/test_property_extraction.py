from shapely import Polygon
from imageable.properties.extract import extract_building_properties


def _get_mock_polygon() -> Polygon:
    polygon = Polygon(
        [
            (-99.23446127617437, 18.91870238070048),
            (-99.23445345855458, 18.918635822907078),
            (-99.23441124340603, 18.918644697280627),
            (-99.23442218807382, 18.91871273413274),
            (-99.23446127617437, 18.91870238070048),
        ]
    )

    return polygon


def _get_mock_neighbors() -> list[Polygon]:
    neighbor_polygons = [
        Polygon(
            [
                (-99.23454017114447, 18.918677888483728),
                (-99.23454570643362, 18.918634252881546),
                (-99.2345005015698, 18.918633380170036),
                (-99.23449404373203, 18.91867701577236),
                (-99.23454017114447, 18.918677888483728),
            ]
        ),
        Polygon(
            [
                (-99.23459594657386, 18.918657977469422),
                (-99.23459884019812, 18.918627867054298),
                (-99.23455615923929, 18.91862376108807),
                (-99.23455326561502, 18.91865729314162),
                (-99.23459594657386, 18.918657977469422),
            ]
        ),
        Polygon(
            [
                (-99.23463366664907, 18.918608063816222),
                (-99.23463103557879, 18.918559114654883),
                (-99.23458893845576, 18.9185607739485),
                (-99.23459156952603, 18.91860640452309),
                (-99.23463366664907, 18.918608063816222),
            ]
        ),
        Polygon(
            [
                (-99.23441983485986, 18.91885925769411),
                (-99.23441733095564, 18.91873253541972),
                (-99.23430089942349, 18.918744378626556),
                (-99.23431592284685, 18.918846230174992),
                (-99.23441983485986, 18.91885925769411),
            ]
        ),
    ]

    return neighbor_polygons


def test_footprint_features():
    # Add more tests for footprint features as needed
    index_polygon = 0
    polygon = _get_mock_polygon()
    neighbor_polygons = _get_mock_neighbors()

    footprint_props_only = extract_building_properties(index_polygon, polygon, all_buildings=neighbor_polygons)

    assert footprint_props_only.unprojected_area == polygon.area
    assert footprint_props_only.neighbor_count == len(neighbor_polygons)
    assert footprint_props_only.complexity > 0.0
    assert footprint_props_only.n_vertices == 4
    assert footprint_props_only.building_height == -1.0
    assert footprint_props_only.nearest_neighbor_distance > 0.0
    assert footprint_props_only.material_percentages == {}
