#!/bin/bash
cd ~/jenga_perception
PYTHONPATH=/home/esd26/.local/lib/python3.10/site-packages python3 -s -c "import sys, runpy; sys.path.insert(0, '/usr/lib/python3/dist-packages'); runpy.run_path('camera_view.py', run_name='__main__')"
