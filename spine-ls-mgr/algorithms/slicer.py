def build_markup_line(a, b, name):
    return {
        "type": "Line",
        "coordinateSystem": "LPS",
        "controlPoints": [
            {
                "label": f"{name}-1", "position": list(a)
            },
            {
                "label": f"{name}-2", "position": list(b)
            }
        ]
    }

def build_markup_fiducial(points, name):
    points_markup = [{
                "label": f"{name}-{i}", "position": list(point)
            } for (i, point) in enumerate(points)]
    return {
        "type": "Fiducial",
        "coordinateSystem": "LPS",
        "controlPoints": points_markup
    }


def build_markup_vector(a, b, name):
  return {
    "type": "Line",
    "coordinateSystem": "LPS",
    "controlPoints": [
      {
        "label": f"{name}-start", "position": list(a)
      },
      {
        "label": f"{name}-end", "position": list(b)
      }
    ]
  }

def build_markups(markups):
    return {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": markups
    }
