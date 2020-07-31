import os

import dash

# Getting the stylesheet file for the app layout
this_dir = os.path.dirname(os.path.abspath(__file__))
css_file = os.path.join(this_dir, "stylesheet.css")

# Intializing the Dash app
app = dash.Dash(
    __name__, external_stylesheets=[css_file], suppress_callback_exceptions=True
)
