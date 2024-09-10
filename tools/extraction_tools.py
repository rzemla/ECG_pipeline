'''

'''
from utils.extract_utils.extract_utils import rotate_origin_only, move_along_the_axis, scale_values_based_on_eich_peak, \
    create_measurement_points, adjust_leads_baseline, preprocess_page_content, extract_graphics_string
import PyPDF2 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def extract_PDF_raw(filename):
    #read in PDF
    reader = PyPDF2.PdfFileReader(open(filename, 'rb'))
    #extract # of pages (may not be relevant)
    num_pages = reader.getNumPages()
    #extract objects from PDF
    pg1 = reader.getPage(0).getContents()._data
    #convert pdf hex symbols/notation to string of Latin-1 endoded bytes (each character= 1 byte) 
    pg1 = preprocess_page_content(pg1)
    #split series of PDF converted to strings to list of objects
    pg1_objects = (pg1).split('S')
    return pg1_objects