'''
for extract_leads_numeric, indices correspond to the following leads:
# 411 - calibration marker
# 412 - I short
# 413 - II short
# 414 - III short
# 415 - aVR short
# 416 - aVL short
# 417 - aVF short
# 418 - V1 short
# 419 - V2 short
# 420 - V3 short
# 421 - V4 short
# 422 - V5 short
# 423 - V6 short
# 424 - calibration marker
# 425 - calibration marker
# 426 - calibration marker
# 427 - V1 long
# 428 - II long
# 429 - V5 long

'''
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2

from utils.extract_utils.extract_utils import (adjust_leads_baseline,
                                               create_measurement_points,
                                               extract_graphics_string,
                                               move_along_the_axis,
                                               preprocess_page_content,
                                               rotate_origin_only,
                                               scale_values_based_on_eich_peak)





#extract short lead traces from input PDF path
def extract_traces(filename):
    # reference value for the calibration jag
    cal_ref = 1000

    # extracted height for the calibration jag in PDF (extract pixel height from PDF)
    # orginal provided measurement from codebase 28.34800000000001 (height that is read in via pdf reader)
    # extracted by measuring pixel height in Illustrator (28.35)
    cal_height = 1000  # height that is read in with pdf reader with our EKG pdfs


    # factor for scaling (effectively gamma = 1 based on read in structure from GE EKGs)
    gamma = cal_ref / cal_height

    # name of the leads
    lead_names_short = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    lead_names_long = ["V1", "II", "V5"]


    # Extract raw PDF data from source file
    pdf_object = extract_PDF_raw(filename)


    # define dict with index ranges for graphics object corresponding to leads/cal marker
    range_list = {
        # range of lead extraction (short leads)
        "start_idx": 412,
        "end_idx": 423,
        # range of lead extraction (long leads)
        "start_idx_long": 427,
        "end_idx_long": 429,
        # calibration marker index
        "cal_idx": 424,
    }

    # Extract numeric x,y points for each lead
    lead_numeric = extract_leads_numeric(range_list, pdf_object)

    # Shift traces to common origin
    shift_leads = list(map(move_trace_to_origin, lead_numeric))

    # Scale leads by gamma factor (gamma = 1 for current EKG PDFs)
    scale_shift_leads = [scale_values_using_cal_mark(l, gamma) for l in shift_leads]

    # convert traces to by linear interpolation to timepoints along sampling frequency (500 Hz)
    # for 2.5s strips (500Hz acquisition freq for 2.5s)
    sample_points = 1250
    leads_upsampled = [
        create_measurement_points(l, sample_points) for l in scale_shift_leads[0:12]
    ]

    # for 10s strips (500Hz acquisition frq for 10s)
    sample_points = 5000
    leads_upsampled.extend(
        [create_measurement_points(l, sample_points) for l in scale_shift_leads[13:16]]
    )

    # Convert short lead list to dataframe
    df_leads_short = pd.DataFrame(
        perform_shape_switch(leads_upsampled[0:12]), columns=lead_names_short
    )

    # Adjust baseline position of each lead
    df_leads_short = adjust_leads_baseline(df_leads_short)

    # Extract trace ID for given set of traces

    return df_leads_short



def extract_PDF_raw(filename):
    reader = PyPDF2.PdfReader(open(filename, 'rb'))
    pg1 = reader.pages[0].get_contents()._data
    pg1 = preprocess_page_content(pg1)
    pg1_objects = (pg1).split('S')
    return pg1_objects


def extract_leads_numeric(range_list, pdf_object):
    start_idx = range_list['start_idx']
    end_idx = range_list['end_idx']
    start_idx_long = range_list['start_idx_long']
    end_idx_long = range_list['end_idx_long']
    cal_idx = range_list['cal_idx']

    lead_objects = []
    # extract short-time lead objects (string of points)
    for x in range(start_idx, end_idx+1):
        lead_objects.append(pdf_object[x])

    # extract long-time lead objects (string of points indices)
    for x in range(start_idx_long, end_idx_long+1):
        lead_objects.append(pdf_object[x])

    # extract calibration marker
    lead_objects.append(pdf_object[cal_idx])

    # remove m(mark?) and l (line?) markers in PDF in each lead object and split newlines
    for i,obj in enumerate(lead_objects):
        lead_objects[i] = obj.replace(' l', '').replace(' m', '').split('\n')

    # select only indices with numeric strings (sample points)
    for i, obj in enumerate(lead_objects):
        # set advance index flag to 0 by default
        adv_idx = 0 
        for j in obj:
            if 'BT' in j:
                # advance starting idx by 1 flag
                adv_idx = 1
        # extract the indices according to the flag advancing by 1 if 'BT' present
        if adv_idx == 1:
            lead_objects[i] = lead_objects[i][2:-1]
        else:
            lead_objects[i] = lead_objects[i][1:-1]

    lead_numeric = []    
    # convert string trace sample coordinates into int x,y coordinates
    for x, trace in enumerate(lead_objects):
        tmp_xy_col = []
        for r in trace:
            # get coordinates as string
            tmp_xy = r.split(' ')
            # cast string x,y coordinates as int
            tmp_xy = [int(i) for i in tmp_xy]
            # append to trace array
            tmp_xy_col.append(tmp_xy) 
        # covert int coordinates to numpy array format
        lead_numeric.append(np.array(tmp_xy_col))

    return lead_numeric


def move_trace_to_origin(lead_list,index = 0):
    # input list of x,y samples for given EKG lead
    # shifts all the traces to common origin and shift y coordinates against first sample value
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


def scale_values_using_cal_mark(lead_list, gamma=0.5):
    # apply scaling based on calibration mark (not necessary with current PDF output from EKGs)
    new_lead_list = []
    for xy_pair in lead_list:
        new_y_value = xy_pair[1] * gamma
        new_lead_list.append([xy_pair[0], new_y_value])

    return new_lead_list


def create_measurement_points(lead_list, number_of_points):
    """
    creates measuring points at equidistant intervals from each other
    :param lead_list: list with lead
    :param number_of_points: number of measuring points to be created
    :return: list with measuring points
    """
    measurement_points = []
    max_element = lead_list[-1][0]
    distance = max_element / number_of_points

    x_values = [x[0] for x in lead_list]
    y_values = [y[1] for y in lead_list]

    for i in range(0, number_of_points):
        measurement_points.append(get_y_value(i * distance, x_values, y_values))

    measurement_points = [int(y) for y in measurement_points]
    return measurement_points



def get_y_value(x, list_x, list_y):
    """
    returns the Y value of a transferred X value based on the transferred list of values.
    :param x: x Value
    :param list_x: list of X-values
    :param list_y: list of Y-values
    :return: y value
    """
    x_value, index = find_value1_value2(list_x, x)
    y_value = [list_y[index - 1], list_y[index]]

    m = (y_value[0] - y_value[1]) / (x_value[0] - x_value[1])
    b = (x_value[0] * y_value[1] - x_value[1] * y_value[0]) / (x_value[0] - x_value[1])
    y = m * x + b
    return y


def find_value1_value2(liste, value):
    """
    finds the next smaller and larger value in a list for a passed value.
    :param liste: list to be searched
    :param value: value
    :return: lower value, upper value and index
    """
    tmp_array = np.array(liste)
    index = np.where(tmp_array > value)[0][0]

    value1 = 0 if index == 0 else liste[index - 1]
    value2 = liste[index]

    return [value1, value2], index


def perform_shape_switch(input):
    # shift list representation into numpy array with feed to dataframe
    # columns: leads
    # rows: timepoints

    input = np.asarray(input)
    length = len(input)
    width = len(input[0])

    output = np.zeros((width, length))

    for row in range(width):
        for item in range(length):
            output[row][item] = input[item][row]

    return output


def calc_stddev(df, window_size=124):
    """
        calculates the average using the standard deviation
        Note: the procedure is only executed on the first lead(?)
        Code below calculate interval where std is the smallest in each lead, finds the one with smallest std for each lead, and returns the mean of that interval for each lead 

    :param df: DataFrame which is scanned
    :param window_size: size of the sliding window
    :return: average
    """
    min_dev_sum = np.Inf
    avg = []
    for i in range(0, len(df) - window_size):
        df_tmp = df.loc[i:i + window_size]

        if sum(df_tmp.std()) < min_dev_sum:
            min_dev_sum = sum(df_tmp.std())
            avg = df_tmp.mean()
    return avg


def adjust_leads_baseline(df_leads):
    stddev_tmp = calc_stddev(df_leads)

    for column in df_leads.columns:
        df_leads[column] = df_leads[column] - stddev_tmp[column]

    return df_leads
