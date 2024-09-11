'''
for extract_leads_numeric, indices correspond to the following leads:
#411 - calibration marker
#412 - I short
#413 - II short
#414 - III short
#415 - aVR short
#416 - aVL short
#417 - aVF short
#418 - V1 short
#419 - V2 short
#420 - V3 short
#421 - V4 short
#422 - V5 short
#423 - V6 short
#424 - calibration marker
#425 - calibration marker
#426 - calibration marker
#427 - V1 long
#428 - II long
#429 - V5 long

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

def extract_leads_numeric(range_list,pdf_object):
    #unpack the dict with indices
    start_idx = range_list['start_idx']
    end_idx = range_list['end_idx']
    start_idx_long = range_list['start_idx_long']
    end_idx_long = range_list['end_idx_long']
    cal_idx = range_list['cal_idx']

    lead_objects = []
    #extract short-time lead objects (string of points)
    for x in range(start_idx, end_idx+1):
        lead_objects.append(pdf_object[x])

    #extract long-time lead objects (string of points indices)
    for x in range(start_idx_long, end_idx_long+1):
        lead_objects.append(pdf_object[x])

    #extract calibration marker
    lead_objects.append(pdf_object[cal_idx])

    #remove m(mark?) and l (line?) markers in PDF in each lead object and split newlines
    for i,obj in enumerate(lead_objects):
        lead_objects[i] = obj.replace(' l', '').replace(' m', '').split('\n')

    #select only indices with numeric strings (sample points)
    for i, obj in enumerate(lead_objects):
        #set advance index flag to 0 by default
        adv_idx = 0 
        for j in obj:
            if 'BT' in j:
                #advance starting idx by 1 flag
                adv_idx = 1
        #extract the indices according to the flag advancing by 1 if 'BT' present
        if adv_idx == 1:
            lead_objects[i] = lead_objects[i][2:-1]
        else:
            lead_objects[i] = lead_objects[i][1:-1]

    lead_numeric = []    
    #convert string trace sample coordinates into int x,y coordinates
    for x, trace in enumerate(lead_objects):
        tmp_xy_col = []
        for r in trace:
            #get coordinates as string
            tmp_xy = r.split(' ')
            #cast string x,y coordinates as int
            tmp_xy = [int(i) for i in tmp_xy]
            #append to trace array
            tmp_xy_col.append(tmp_xy) 
        #covert int coordinates to numpy array format
        lead_numeric.append(np.array(tmp_xy_col))

    return lead_numeric


def move_trace_to_origin(lead_list,index = 0):
    #input list of x,y samples for given EKG lead
    #shifts all the traces to common origin and shift y coordinates against first sample value
    tmp = 0
    for (x, y), i in zip(lead_list, range(len(lead_list))):
        if x < index:
            tmp = i

    x0, y0 = lead_list[tmp]
    tmp = [(x, y - y0) for x, y in lead_list]

    delta = index - tmp[0][0]

    new_lead_list = []
    for i in tmp:
        new_lead_list.append((i[0] + delta, i[1]))

    return new_lead_list