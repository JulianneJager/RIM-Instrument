import numpy as np # Mathmatical library
import pandas as pd # data management library
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, AutoDateLocator
import glob
import serial
import time
import os
from datetime import datetime
from scipy.optimize import curve_fit

"""
TO UPDATE
# URGENT UPDATES
- Combine automated 2 panel voltage/res_freq Capacity plotter with main script
- Fix Parabolic_fit to work when the there are insufficient points to either 
  side of the peak


# Other Updates
- Update res_freq plotter to have meaningful axis
# Potential Future Updates
- ADD MINIMUM NUMBER OF SAMPLE POINTS OR EDIT POLYFIT TO WORK FOR LESS THAN 6 SAMPLES
- HAVE 10 HZ tails also perform average
- IMPROVE LORENTZ FITTING FOR LESS SUITABLE DATA SETS
- Add GUI
- Trim down code (functionality is pretty good, work to reduce complexity)
- Fix warnings to disable curve fitting warnings only, not blanket supress all 
  warnings


WARNINGS
    -The read timeout value assigned in the serial_defaults dict below as "timeout"
     is a precise value. If too low, you will outrun the arduino and read in empty
     lines. If too high you will wait too long and the arduino will overwrite itself
     causing truncated data. The data filter will compensate for truncated data, 
     so being set too high is prefirable to being set too low. You will know when
     your timeout is set too high, as the sample time will jump up. Currently 2-3 
     seconds seems to work nicely.

     - Leave write timeout pretty high unless you know what you are doing.
"""
# GLOBAL DEFAULTS
ARDUINO_MICROS_OVERFLOW_VAL = 4294967295

serial_defaults = {"port" : None, "baudrate" : 9600, "bytesize" : serial.EIGHTBITS, 
                    "parity" : serial.PARITY_NONE, "stopbits" : serial.STOPBITS_ONE, 
                    "timeout" : 2 , "xonxoff" : False, "rtscts" : False,
                    "write_timeout" : 4, "dsrdtr" : False, "inter_byte_timeout" : None,
                    "exclusive" : None}

options_defaults = {"lorentz_fit" : True, "parabolic_fit" : True, 
                    "A_fit" : True, "A_avg" : False, "A_max" : False, 
                    "input_directory" : os.getcwd(), "output_directory" : os.getcwd() + "\jiggler_output",
                        "export_data" : True, "export_figure" : True,
                        "verbose" : False, "silent" : False,
                        "x_lims" : None, "y_lims" : None, "warnings" : True,
                        "Sweep_Column_DF_export" : False}

"""                            DEFININING CLASS                              """

class Jiggler():


    def __init__(self, com_port = None, f_interval = [118,130], step_size = 0.1, sample_size = 1500, 
                serial_dict = serial_defaults, options_dict = options_defaults):

        """
        ---Initialization Attributes
        Two dictionaries of options
        self.serial_dict = dictionary of the parameters of serial.Serial which is
                            passed directly to a serial.Serial call in Jiggler.sweep()

        options_dict keys:
            "lorentz_fit" : True 
                Attempts a lorentz curve fit to determine the resonant frequency

            "parabolic_fit" : False
                Attempts a parabolic cap fit to determine the resonant frequency

            "input_directory" : os.getcwd() 
                path to the directory of files you want to import, default is 
                the current working directory

            "output_directory" : os.getcwd() + "\jiggler_output"
                Path to the desired output directory for the jiggler data, default
                is cwd + "\jiggler_output
            
            "verbose" : False,
                Maximizes information printed during operation
            "silent" : False
                Minimizes information printed during operation
            
            "xlims" = Nothing currently

            "ylims" = [min, max] for min and max being floats for the min and 
                max limits of the plots produced by the import plotter
            
            "warnings" = Have to check, unsure what warnings...

            "Sweep_Column_DF_export" = True
                Exports a data file with the import plotter where the data 
                exported is in columns with a label. Only exports the frequency
                and Sin-fit amplitude. 
        }

        """
        self.serial_dict = serial_dict
        self.serial_dict["port"] = com_port
        self.options_dict = options_dict

        # These are just conversions of the parameters into properties
        self.frequency_interval = f_interval
        self.sample_size = sample_size
        self.step_size = step_size

        """
        ---Dynamic Attributes---
        These attributes are regularly overwritten or appended to during operation 
        and are used as variables for various methods. 

        Typically they are only called by the user when there has been a mid-sweep
        crash, and the user needs to explore exactly where it occured.
        
        The values stored here will only be the most recent measurment in most 
        cases
        """
        self.formatted_data = []
        self.loop_count = 0
        self.lorentz_fit_params = []
        self.parabolic_fit_params = []
        self.serial = None
        self.solution_list = []
        self.sweep_data = []
        self.midsample_times = []

        """
        ---Recursive Attributes---
        """
        self.frequency_range = self.frequency_steps()
        self.frequency_byte_list = Jiggler.range_byte_encoder(self)

        """
        ---Data Storage Attributes---
        """
        self.error_log = []
        self.import_times = []
        self.imported_data = []
        self.export_df_jig = []
        self.export_df_final = []
        self.parabolic_res_freq = []
        self.lorentz_res_freq = []
        self.res_freq = [] # This is the res_freq the code uses for all res freq dependencies
        self.res_freq_time = []
        self.res_freq_amp = []
        self.time_list = []
        self.def_df_archive = []
        self.temp1 = []
        self.temp2 = []
  
        """Debugging"""
        self.Debug1 = []
        self.Debug2 = []
        self.Debug3 = []

    """Serial read/write and Data management functions"""
    #---------------------------------------------------------------------------
    # Basic Functions (Simple functions)
    def write_data(self, freq, buffer = 100/1E6):
        """
        Writes frequency values to the serial port with a default time buffer
        of 100 microseconds.

        Also prints a warning when the write timout is reached
        
        Parameters:
            freq: a float encoded as a byte
            buffer = 100/1E6 : time in seconds to delay before and after write
        """
        time.sleep(buffer)
        self.serial.reset_input_buffer()
        time.sleep(buffer)

        output = self.serial.write(freq)
        if self.options_dict["silent"] == False:
            if output == 0:
                error = f"WRITING FREQUENCY {freq} FAILED with buffer {buffer}"
                print(error)
                self.error_log.append(error)


    def read_data(self):
        """Reads in sample_size (usually 1500) data points from the Jiggler and 
        decodes them. The code also includes a safety where if a string is 
        accidently parsed or a float mis-converted then a float error is 
        triggered and the read loop is broken.
        
        Data from the Jiggler is recieved with ascii encoding.

        Parameters:
            self.serial = given during class initialization
            self.sample_size = given during class initialization

        Returns:
            data_list: a list of comma delimited strings the length of the 
                sample_size in the format:
                ["113,12030548658,163", "113,12030548702,163" , ...]
                Where the first value is the frequency, the second is the time 
                in microseconds, and the third is the angle in tenths of a degree
        """
        data_list = []
        for i in range(self.sample_size):
            data = self.serial.read_until() # EXPERIMENTAL CHECK ON THIS
            # data = self.serial.readline()
            # print(data)
            data_decoded = data.decode("ascii")
            #print(data_decoded)
            # Checks if float conversion of the frequency returned a 0 in the arduino
            # If the Arduino gets a float conversion failure it will print out "FLOAT ERROR"
            # And we skip to the next value.
            # if self.options_dict["silent"] == False:
            #     if i == 0:
            #         if data_decoded == "FLOAT ERROR\r\n":
            #             error = f"FLOAT ERROR DETECTED"
            #             print(error)
            #             self.error_log.append(error)
            #             data_list = None
            #             break

            # print(f"value number {i} with data {type(data)} {data}")
            data_list.append(data_decoded)

        return data_list


    def reset_instrument(self):
        """Sends a frequency value of 1, the Arduino firmware uses all frequency 
        values less than 50 as stop commands"""

        stop_byte = str(1).encode()
        self.serial.write(stop_byte)


    def stop_instrument(self):
        """Sends a frequency value of 1, the Arduino firmware uses all frequency 
        values less than 50 as stop commands"""

        self.serial = serial.Serial(
            self.serial_dict["port"], self.serial_dict["baudrate"],
            self.serial_dict["bytesize"], self.serial_dict["parity"], 
            self.serial_dict["stopbits"], self.serial_dict["timeout"],
            self.serial_dict["xonxoff"], self.serial_dict["rtscts"],
            self.serial_dict["write_timeout"], self.serial_dict["dsrdtr"],
            self.serial_dict["inter_byte_timeout"], self.serial_dict["exclusive"]
                            )

        stop_byte = str(2).encode()
        self.serial.write(stop_byte)
        time.sleep(1)
        self.reset_instrument()


    def frequency_steps(self):
        """Uses numpy.linspace to create a linear spacing of frequency values the
        instrument will sample"""\

        # Start stop and step found using values from class initialization
        start = self.frequency_interval[0]
        stop = self.frequency_interval[-1]
        step = self.step_size

        # Linear frequency spacing for sampling
        freq_range = np.linspace(start, stop, int(((np.abs(stop-start))/step))+1)

        return freq_range


    def range_byte_encoder(self):
        """Determines the frequency values to send to Jiggler using the start, stop
        and step values. Then encodes those values using utf8
        
        Parameters:
            self.frequency_range
            self.step_size
            
        Returns:
            freq_byte_list = a list of frequencies encoded in utf8 in byte foramt
        """

        # Iterating through the frequency array and creating a list of utf-8
        # encoded frequency values
        freq_byte_list = []
        for freq in self.frequency_range:
            freq_byte = str(freq).encode()
            freq_byte_list.append(freq_byte)
        
        return freq_byte_list


    def data_filter(self, data_string_list):
        """
        Stupid Over the top data logic filter. Only allows data rows through IF
        1.) They have exactly 3 comma delimited entries
        2.) The first entry is a float in str format
        3.) the 2nd and 3rd entries are integers in str format
        
        Parameters:
        -sweep_data: list of comma delimited strings of the form 
                    ["freq(float), time(int), deflection(int)"]

        Returns:
        -sweep_data_clean: Returns a split list of length 3 of the form
                    [freq, time, deflection] where the entries are in the format 
                    [float, int, int] Also removes all entries which fail to fit the
                    desired format.
        """

        # bad data counter, counts the number of bad samples
        bad_count = 0
        data_clean = []
        for i, data in enumerate(data_string_list):
            split_row = data.split(',')

            # Checks if the row split into a list with 3 components
            if len(split_row) != 5: #changed from 3 to 5 to allow for the temperature data
                # print(f"for row {i} we have {split_row} which is not of length 3")
                bad_count += 1
                continue

            # Checks if the first component of the list is a string representation of a float
            try:
                int(split_row[0])
                bad_count += 1
                continue
                # print(f"Error Frequency value in row {i} of {data} is not a float")
            except:
                pass
            try:
                split_row[0] = float(split_row[0])
            except:
                # print(f"Error Frequency value in row {i} of {data} is not an float")
                bad_count +=1
                continue

            # Checks if the time component of the list is a string representation of an int
            try:
                split_row[1] = int(split_row[1])
            except:
                # print(f"Error Time value in row {i} of {data} is not an int!")
                bad_count += 1
                continue
            
            # Checks if the angular deflection value is a string representation of an int
            try:
                split_row[2] = int(split_row[2])
            except:
                # print(f"Error Ang Deflection value in row {i} of {data} is not an int!") 
                bad_count += 1
                continue

            try:
                split_row[3] = float(split_row[3]) # Check for temp1
            except:
                # print(f"Error Time value in row {i} of {data} is not a float!")
                bad_count += 1
                continue

            try:
                split_row[4] = float(split_row[4]) # Check for temp2
            except:
                # print(f"Error Time value in row {i} of {data} is not a float!")
                bad_count += 1
                continue
            
            # Appending cleaned rows to data list
            data_clean.append(split_row)

        # Provides printed warning for bad rows.
        if bad_count > 0:
            print(f"For loop {self.loop_count} Frequency {data_clean[0][0]} had {bad_count} rows removed for failing data filtering")
        return data_clean, bad_count  # EB added else statement


    #---------------------------------------------------------------------------
    # Composite Functions (Functions composed of basic functions or of greater
    #                       complexity)
    def linear_sweep(self):
        """Samples the frequency values at the points determined by the start,
        stop, and step values.
        
        The function writes the values to the Jiggler, reads the returned samples
        storing them as lists of strings, then stores each list of strings as a
        list of lists of data
        
        I.E [sample_1_data, sample_2_data, sample_3_data,...] where each sample
        is a list of strings of the length of the sample size"""

        # Initializing the list of data lists
        sweep_data = []

        # Iterating through each frequency we wish to sample for
        for freq_byte in self.frequency_byte_list:
            self.write_data(freq_byte)
            data_list = self.read_data()

            # When data_list == none then there was a float conversion error,
            # Currently the code skips to the next frequency and logs an error.
            if data_list == None:
                continue
            
            # Disabled for looping, stability is now high enough that this is spam
            if self.options_dict["silent"] == False:
                 print(f"Freq {freq_byte} data collected")


            # Appending each data set to a list of data_sets
            sweep_data.append(data_list)
            self.sweep_data = sweep_data

        return sweep_data


    def data_importer(self, input_directory = None):
        """
        Imports all csv files found in the input_directory folder and exports their 
        data as a list of lists.

        Parameters:
            input_directory = a filepath where the files to import are found.
            output_directory = desired output folder path. 

        Returns:
            A_sol_list = a list of sublists where each sublist is of the 
                                format returned by A_solver
            fnames = a alphanumerically sorted list of filenames
        """
        # Resetting self.midsample_times for import
        self.midsample_times = []

        # Defaults to the value in the options_dict if no input directory is specified
        if input_directory == None:
            input_directory = self.options_dict["input_directory"]

        # Retrieving all .csv filenames and sorting them
        fnames = sorted(glob.glob(input_directory + '\*.csv'))

        # Iterating through each file and collecting the start times and solution data
        midsample_times_list = []
        A_sol_list_list = []
        for fname in fnames:
            print(f"Importing file {os.path.basename(fname)}")

            # Parsing sample_start time from filename and appending to a list
            base_name = os.path.basename(fname)
            midsample_times = base_name[:base_name.find(".")]
            midsample_times_list.append(midsample_times)
            self.midsample_times.append(midsample_times)

            # The import_times list is padded with Nones to follow the self.times_list format
            self.import_times.append([None, midsample_times, None])

            # Reading in data using pandas
            df = pd.read_csv(fname, index_col = 0)
            df_T = df.T
            formatted_df = df_T.rename(columns = {0 : "Frequency", 1 : "A_fit", 2 : "A_avg", 3 : "A_max", 4: "Temp1_avg", 5: "Temp2_avg"})

            # converting columns into arrays
            freq_array = formatted_df["Frequency"].values
            A_fit_array = formatted_df["A_fit"].values
            A_avg_array = formatted_df["A_avg"].values
            A_max_array = formatted_df["A_max"].values
            temp1_array = formatted_df["Temp1_avg"].values
            temp2_array = formatted_df["Temp2_avg"].values

            # Storing arrays as a list in the style of the original data generation
            A_sol_list = [freq_array, A_fit_array, A_avg_array, A_max_array, temp1_array, temp2_array]
            A_sol_list_list.append(A_sol_list)

        self.imported_data = A_sol_list_list
        self.export_df = formatted_df.iloc[:, 0:4]

        return self.imported_data, self.import_times

        """
    Stupid Over the top data logic filter. Only allows data rows through IF
    1.) They have exactly 3 comma delimited entries
    2.) The first entry is a float in str format
    3.) the 2nd and 3rd entries are integers in str format
    """


    def data_formatter(self, sweep_data = None):

        # If no data is manually supplied uses data stored in the class
        if sweep_data == None:
            sweep_data = self.sweep_data

        formatted_data = []
        for data_list in sweep_data:

            # Filtering bad data
            clean_data_list, bad_count = self.data_filter(data_list)

            # Initializing dummy list
            data = []
            # Retrieving Values and converting to float/arrays
            freq_value = float(clean_data_list[0][0])

            # Retrieving Time converting to 64 bit integers
            time_raw = (np.array([row[1] for row in clean_data_list]).astype('int64'))
            time_vals = time_raw / 1E6 # Unit conversion to seconds
            """
            NOTE V1.01 and earlier contains a script for handling MICROS overflows.
            If the microcontroller firmware is updated no to longer reset between samples
            you will need to re-institute the code below)
            """
            # time_raw = np.array([row[1] for row in clean_data_list]).astype('int64')
            # if time_raw[-1] < time_raw[0]:
            #     min = time_raw.min()
            #     min_ind = np.where(time_raw == min)
            #     time_final = time_raw[min_ind[0][0]:]+ARDUINO_MICROS_OVERFLOW_VAL
            #     time_vals = np.concatenate((time_raw[:min_ind[0][0]], time_final))
            #     time_vals = time_vals /1E6
            # else:
            #     time_vals = time_raw / 1E6

            # Retrieving Angle Values
            angle_vals = np.array([row[2] for row in clean_data_list]).astype(float)

            #Retrieving temp values of RTD#1
            temp_vals1 = np.array([row[3] for row in clean_data_list]).astype(float)

            #Retrieving temp values from RTD#2
            temp_vals2 = np.array([row[4] for row in clean_data_list]).astype(float)

            data.append(freq_value), data.append(time_vals), data.append(angle_vals), data.append(temp_vals1), data.append(temp_vals2)
            formatted_data.append(data)

            # Saving current iteration of formatted data as a class property
            self.formatted_data = formatted_data

        return formatted_data


    def Amplitude_solver(self, formatted_data = None):
        """Takes the data from formatted data, and applies all three methods for 
        calculating Amplitude"""

        # If no data is manually supplied to functions uses data stored in class
        # property from the most recent data_formatter call
        if formatted_data == None:
            formatted_data = self.formatted_data

        # Initializing data lists
        freq_list = []
        A_fit_list = []
        A_avg_list = []
        A_max_list = []
        temp1_avg_list = []
        temp2_avg_list = []
    
        for data in formatted_data:
            # Slicing data from data list
            freq_value = data[0]
            time_vals = data[1]
            angle_vals = data[2]
            temp1_vals = data[3]
            temp2_vals = data[4]

            # Applying Amplitude solving functions
            A_fit = self.Nicks_Sin_fit(time_vals, angle_vals, freq_value)
            A_avg = self.Average_Amplitude(angle_vals)
            A_max = self.Amplitude_max(angle_vals)
            temp1_avg = self.Average_Temp(temp1_vals)
            temp2_avg = self.Average_Temp(temp2_vals)


            # Saving solutions into new lists
            freq_list.append(freq_value)
            A_fit_list.append(A_fit)
            A_avg_list.append(A_avg)
            A_max_list.append(A_max)
            temp1_avg_list.append(temp1_avg)
            temp2_avg_list.append(temp2_avg)


        # Converting solution lists to arrays, converting angle units to degrees 
        # rather than tenths of a degree
        freq_array = np.array(freq_list)
        A_fit_array = np.array(A_fit_list)/10
        A_avg_array = np.array(A_avg_list)/10
        A_max_array = np.array(A_max_list)/10
        temp1_array = np.array(temp1_avg_list)
        temp2_array = np.array(temp2_avg_list)
        
       
        
        # Returning Solution arrays as a single solution list
        A_sol_list = [freq_array, A_fit_array, A_avg_array, A_max_array, temp1_array, temp2_array]

        # Saving A_Sol_list to object properties
        self.solution_list = A_sol_list

        return A_sol_list



    #---------------------------------------------------------------------------
    """Primary Operation Functions:"""
    #    Functions which perform an entire common operation of the instrument
    #    Typically with mostly default settings
    def Jiggler_sweep(self):
        """
        This function performs a linearly spaced sweep of the frequency range
        given during initialization for the step size given during initialization.

        It is a composition of the functions:
        reset_instrument()
        linear_sweep()
        data_formatter()
        Amplitude_solver()
        polyfit()

        Along with pre-built functions from the time, and datetime libraries.

        Parameters: (all parameters are properties defined during class initialization)
            self.frequency
            self.sample_size
            self.serial

        Returns:
            A_sol_list = the return from Amplitude_Solver()
            pfit_A_sol_list = the return from polyfit()
            start_time = start time of the sweep in '%Y_%m_%d %H_%M_%S' format
            end_time = end time of the sweep in '%Y_%m_%d %H_%M_%S' format
        """
        # Intializing serial.Serial object with parameters from self.serial_dict
        self.serial = serial.Serial(
                    self.serial_dict["port"], self.serial_dict["baudrate"],
                    self.serial_dict["bytesize"], self.serial_dict["parity"], 
                    self.serial_dict["stopbits"], self.serial_dict["timeout"],
                    self.serial_dict["xonxoff"], self.serial_dict["rtscts"],
                    self.serial_dict["write_timeout"], self.serial_dict["dsrdtr"],
                    self.serial_dict["inter_byte_timeout"], self.serial_dict["exclusive"]
                                    )

        # Delay to give the Arduino time to initialize
        time.sleep(2)

        # Storing start time of sweep
        start_time = datetime.today()

        """Perform Initial linear sweep of the frequency range"""
        sweep_data = self.linear_sweep()

        # Storing end time of sweep
        end_time = datetime.today()

        # Time at midpoint of sample
        mid_time = (start_time + (start_time-end_time)/2).strftime('%Y_%m_%d %H_%M_%S')

        # Sending reset command to Arduino
        time.sleep(1.5)
        self.reset_instrument()

        """Formatting Collected Data"""
        formatted_data = self.data_formatter(sweep_data)

        """Calculating Amplitude values from measured data"""
        A_sol_list = self.Amplitude_solver(formatted_data)

        """Curve fits"""
        if self.options_dict["parabolic_fit"] == True:
            self.parabolic_fit_params = self.parabolic_fit(A_sol_list)

            # Filtering out failed fits
            if (self.parabolic_fit_params[0] < 200) and ((self.parabolic_fit_params[0] > 50)):
                self.parabolic_res_freq.append(self.parabolic_fit_params[0])
            else:
                self.parabolic_res_freq.append(None)
                print(f"Parabolic_fit_failed for data {mid_time}")

        if self.options_dict["lorentz_fit"] == True:
            self.lorentz_fit_params = self.lorentz_fit(A_sol_list)
            # Filtering out failed fits
            if (self.lorentz_fit_params[0] < 200) and ((self.lorentz_fit_params[0] > 50)):
                self.lorentz_res_freq.append(self.lorentz_fit_params[0])
            else:
                self.lorentz_res_freq.append(None)
                print(f"Lorentz_fit_failed for data {mid_time}")


        # Adding start and end time to the list
        self.time_list.append([start_time, mid_time, end_time])
        time_list = self.time_list[-1]

        return time_list


    def Jiggler_loop(self, duration, time_between_samples = 240):
        """Performs periodic measurements sweeping along the chosen frequency 
        range storing the resonant frequency each sweep and producing a graph 
        and .csv file with the data after each measurement. This function is 
        essentially a loop of the Jiggler_sweep() functions.
        
        Parameters:
            duration = length of time the instrument will be in operation
            time_between_samples = rest time between each sample

        Returns:
            self.res_freq_list = if the loop runs to completion it will return a
                                list of the accumulated resosnant frequency values
            """

        # Start and stop times for the timer
        start = time.time()
        stop = time.time()

        self.midsample_times = []
        # Looping until the duration is reached
        while ((stop - start) <= duration):
            
            # Increasing loop counter
            self.loop_count += 1

            # Sweeping
            time_list = self.Jiggler_sweep()

            # Saving Midpoint times
            self.midsample_times.append(self.time_list[-1][1])

            # plotting
            self.quick_plot()

            # Resting
            time.sleep(time_between_samples)
            stop = time.time()

        print(f"Loop Complete at {stop}")


    #---------------------------------------------------------------------------
    """Exporting Functions"""
    def quick_plot(self, A_sol_list = None, time_list = None, x_lims = None, y_lims = None, export = None):
        """
        Plots the data returned by Jiggler_sweep() as an amplitude vs frequency 
        graph.

        self Parameters
            A_sol_list = return value of Amplitude_Solver()
            fit = Options are "lorentz" or "parabolic" input as string
            parabolic_fit = parabolic fitting parameters (output of parabolic_fit())
            lorentz_fit = output of lorentz_fit)()
            start_time = start time, either given manually or returned by Jiggler_sweep()

        Parameters:
            x_interval = a list [min, max] frequency interval to plot on the xaxis, 
                        if none is given then it will plot the interval provided 
                        on initialization
            y_interval = a list [min, max] amplitude interval to plot on the yaxis, 
                        if none is given then it will plot [0, 1.2]

        Returns:
            Exported figures of the data, and the calculated amplitudes to the 
            output folder.

        """
        
        if A_sol_list == None:
            A_sol_list = self.solution_list
        
        if time_list == None:
            time_list = self.time_list[-1]

        if export == None:
            export = self.options_dict["export_data"]

        # Creating figure and axis objects
        fig, ax = plt.subplots()

        """Plotting Amplitude data"""
        if self.options_dict["A_fit"] == True:
            plt.plot(A_sol_list[0], A_sol_list[1], '.b', label = "Sin Fit")
        if self.options_dict["A_avg"] == True:
            plt.plot(A_sol_list[0], A_sol_list[2], '.k', label = "Adjusted Average")
        if self.options_dict["A_max"] == True:
            plt.plot(A_sol_list[0], A_sol_list[3], '.g', label = "Amax")

        """Plotting curve fits"""
        # if self.options_dict["parabolic_fit"] == True:
        #     # Plotting Parabolic cap
        #     plt.plot(self.parabolic_fit_params[1], self.parabolic_fit_params[2],
        #              'r', label = "Parabolic fit")

        #     # Plotting vertical line at estimated resonant frequency
        #     plt.axvline(self.parabolic_fit_params[0], 0, 1 , color = "black", 
        #                 linestyle = 'dashed', 
        #                 label = f"Vertex is {self.parabolic_fit_params[0]:.2f}")

        # if self.options_dict["lorentz_fit"] == True:
        #     if (self.lorentz_fit_params[0] > 50) and (self.lorentz_fit_params[0] < 200):
        #         # Plotting Lorentz Curve
        #         plt.plot(self.lorentz_fit_params[1], self.lorentz_fit_params[2], 
        #                 "grey", label = "Lorentz fit" )

        #         # Plotting vertical line at estimated resonant frequency
        #         plt.axvline(self.lorentz_fit_params[0], 0, 1 , color = "black", 
        #                     linestyle = '-.', 
        #                     label = f"lorentz peak = {self.lorentz_fit_params[0]:.2f}")
                
        #         # Lorentz curve parameters for plotting FWHM line
        #         x_peak = self.lorentz_fit_params[0]
        #         FWHM = self.lorentz_fit_params[3]
        #         FWHM_yval = self.lorentz_fit_params[4]

        #         # Plotting two arrows on top of each other facing opposite directions
        #         # To create a double headed arrow
        #         plt.arrow(x = x_peak - (FWHM/2), y = FWHM_yval, dx = self.lorentz_fit_params[3], 
        #                     dy = 0, facecolor = "black", head_width = 0.015, 
        #                     label = f"FWHM = {np.abs(FWHM):.2f}",
        #                     length_includes_head = True, zorder = 10, overhang = 3)
        #         plt.arrow(x = x_peak + (FWHM/2), y = FWHM_yval, dx = -self.lorentz_fit_params[3], overhang = 3, 
        #                     dy = 0, facecolor = "black", head_width = 0.015,
        #                     length_includes_head = True, head_starts_at_zero = True, zorder = 11)


        # Configuring dynamic Labels
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (degrees)")
        plt.title(f"{time_list[1]}")
        plt.grid(True)
        plt.legend(loc="upper left")

        # Setting x and y limits (This is a disaster clean this up!)
        if x_lims == None: # Checking if local xlim variable is none
            x_lims = self.options_dict["x_lims"]
            if x_lims != None: # Checking if global dict variable is none
                plt.xlim(x_lims[0], x_lims[1])
            else: # if both are none use freq range
                x_lims = [self.frequency_range[0], self.frequency_range[-1]]
                plt.xlim(x_lims[0], x_lims[1])
        else: # if local is not none plot local
            plt.xlim(x_lims[0], x_lims[1])

        if y_lims == None:
            y_lims = self.options_dict["y_lims"]
            if y_lims != None:
                plt.ylim(y_lims[0], y_lims[1])
        else:
            plt.ylim(y_lims[0], y_lims[1])


        """Saving Figure"""
        if self.options_dict["export_figure"] == True:
            data_path = self.options_dict["output_directory"] + r"\data"
            fig_path = self.options_dict["output_directory"] + r"\figures"

            # Checking if output folder exists, if not, a new folder is created
            if os.path.exists(self.options_dict["output_directory"]) == False:
                os.mkdir(self.options_dict["output_directory"])

            if os.path.exists(data_path) == False:
                os.mkdir(data_path)

            if os.path.exists(fig_path) == False:
                os.mkdir(fig_path)

            # Exporting Figure with desired filename
            plt.savefig(f"{fig_path}\\{time_list[1]}.pdf")

        # Clearing Figure from memory
        plt.close("all")
        plt.clf()

        # Printing Completion statement
        if self.options_dict["silent"] == False:
            print(f"\n\n !!!   GRAPH GENERATED for file {time_list[1]} !!! \n\n")

        # Exporting Column Format dataframe with Pfit data
        if self.options_dict["Sweep_Column_DF_export"] == True:
            df_jig = self.export_df
            df_jig = pd.DataFrame({'Frequency' : df_jig["Frequency"].values , 
                                    "A_fit" : df_jig["A_fit"].values})
            df_pfit = pd.DataFrame({"pfit_xvals" : self.parabolic_fit_params[1],
                        "pfit_yvals" : self.parabolic_fit_params[2]})
            df_resfreq = pd.DataFrame({"Res_Freq" : [self.parabolic_fit_params[0]]})
            df = pd.concat([df_jig, df_pfit, df_resfreq], axis = 1)

            # saving final export df from jiggler as class property
            self.export_df = df
            df.to_csv(f"{data_path}\{time_list[1]}_DF_FORMAT.csv", index = False)
            print(f"\n\n !!!   DF GENERATED for file {time_list[1]} !!! \n\n")


            self.Debug1 = df_pfit
            self.Debug2 = df_resfreq
        #BETTER EXPORT FORMAT UPDATE CODE TO USE COLUMN DATA EXPORTS IN FUTURE!
        # if self.options_dict["Sweep_Column_DF_export"] == True:
        #     df = pd.DataFrame(self.solution_list)
        #     print(df)
        #     df_T= df.iloc[[0, 1], :].transpose()
        #     df_Tr = df_T.rename(columns = {0 : "Frequency" , 1 : 
        #                                     "Amplitude (sin-fit)"})
        #     df.to_csv(f"{data_path}\{time_list[1]}_DF_FORMAT.csv")
        #     print(f"\n\n !!!   DF GENERATED for file {time_list[1]} !!! \n\n")


        # Exporting data and saving figure
        if export == True:
            df = pd.DataFrame(self.solution_list)
            df.to_csv(f"{data_path}\{time_list[1]}.csv")
    
    # def Average_Temp(self, temp):
    #     """
    #     Returns the average of the temp data, as a float.
    #     """
    #     temp_mean = np.mean(temp)
    #     return temp_mean


    def import_plotter(self, imported_data_list = None, midpoint_time_list = None):
        """
        This function takes the output from data_importer and uses the imported
        data to dynamically generate graphs.

        You can change the settings by changing the options in Jiggler.options_dict
        """
        # Checking for manual data input, if no data is given uses the data stored
        # in Jiggler.imported_data from the last call of Jiggler.data_importer()
        if imported_data_list == None:
            imported_data_list = self.imported_data
        
        if midpoint_time_list == None:
            midpoint_time_list = self.import_times

        # Iterating through each imported amplitude data
        resonant_frequency_list = []
        for i, A_sol_list in enumerate(imported_data_list):
            # Calculating Step size for dynamic Parabolic fit
            step_size = A_sol_list[0][1] - A_sol_list[0][0] 

            # Applying Parabolic Curve Fit
            if self.options_dict["parabolic_fit"] == True:
                self.parabolic_fit_params = self.parabolic_fit(A_sol_list,
                                                 step_size = step_size)
                # Filtering bad data
                if (self.parabolic_fit_params[0] < 200) and ((self.parabolic_fit_params[0] > 50)):
                    # Storing resonant frequency found by parabolic fit
                    self.parabolic_res_freq.append(self.parabolic_fit_params[0])
                else:
                    self.parabolic_res_freq.append(None)

            # Applying Lorentzian Curve Fit
            if self.options_dict["lorentz_fit"] == True:
                self.lorentz_fit_params = self.lorentz_fit(A_sol_list)

                # Filtering bad data
                if (self.lorentz_fit_params[0] < 200) and ((self.lorentz_fit_params[0] > 50)):
                    # Storing resonant frequency found by Lorentz fit
                    self.lorentz_res_freq.append(self.lorentz_fit_params[0])
                else:
                    self.lorentz_res_freq.append(None)

            self.quick_plot(A_sol_list = A_sol_list, time_list = midpoint_time_list[i], 
                            x_lims = [A_sol_list[0][0], A_sol_list[0][-1]], 
                            y_lims = self.options_dict["y_lims"], export = False)
            self.temp1.append(self.Average_Temp(A_sol_list[4]))
            self.temp2.append(self.Average_Temp(A_sol_list[5]))

        self.resonance_exporter()


    def resonance_exporter(self):
        # Making a list of res_freq_methods
        res_freq_lists = [self.lorentz_res_freq, self.parabolic_res_freq]
        res_amp_list = self.res_freq_amp
        name_list = ["Lorentz fit", "Parabolic fit"]
        temp1_list = self.temp1
        temp2_list = self.temp2

        # Converting string times to a list of datetime objects
        dt_list = []
        dt_labels = []
        for dt_string in self.midsample_times:
            dt = datetime.strptime(dt_string, '%Y_%m_%d %H_%M_%S')
            dt_list.append(dt)
            dt_labels.append(dt.strftime('%H_%M_%S'))

        # Testing for output folders existance
        output = self.options_dict["output_directory"]
        if os.path.exists(output +"\\data") == False:
            os.mkdir(output + "\\data")

        if os.path.exists(output +"\\figures") == False:
            os.mkdir(output + "\\figures")

        if os.path.exists(output + "\\data\\res_freq") == False:
            os.mkdir(output + "\\data\\res_freq")

        # Creating plots
        fig, ax = plt.subplots(constrained_layout = True)
        for i, res_freq_list in enumerate(res_freq_lists):
            if res_freq_list == []:
                print(f"No {name_list[i]} performed on data")
                continue

            # Plotting resonant frequency vs datetime
            ax.plot(dt_list, res_freq_list, '.', label = f"{name_list[i]} resonant frequencies")

        # Setting Time Loctators and formats
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

        # Labels and Titles
        ax.set_xlabel("Time of Measurement")
        ax.set_ylabel("Res freq (Hz)")
        ax.set_title(f"{name_list[i]} Res Freq vs Sample number")
        ax.grid(True)
        ax.legend(loc="best")

        # Constructing dynamic file names
        output_figures = self.options_dict['output_directory'] + r"\figures"
        output_data = self.options_dict['output_directory'] + r"\data"
        start = self.import_times[0][1]
        end = self.import_times[-1][1]
        plt.savefig(f"{output_figures}\\_res_freq_{start}-{end}.pdf")
        plt.close("all")
        plt.clf()

        # Constructing resonant frequency vs time DataFrame

        """ Lorentz Fit export (Currently not in use)"""
        # df = pd.DataFrame([dt_list, res_freq_lists[0], res_freq_lists[1]]).T

        # df.rename(columns = {0 : "Times", 1 : "lorentz fit", 2 : "parabolic fit"},
        #                     inplace = True)
        # df.to_csv(f"{output_data}\\res_freq\\_res_freq_{start}-{end}.csv")

        """ No Lorentz Fit export"""
        """df = pd.DataFrame([dt_list, res_freq_lists[1]]).T"""
        """df = pd.DataFrame([dt_list, res_freq_lists[1], self.res_freq_amp]).T"""
        """df.rename(columns = {0 : "Datetimes", 1 : "parabolic fit"},
                            inplace = True)
        df.to_csv(f"{output_data}\\res_freq\\_res_freq_{start}-{end}.csv")"""

        df = pd.DataFrame([dt_list, res_freq_lists[1], res_amp_list, temp1_list, temp2_list]).T

        df.rename(columns = {0 : "Datetime", 1 : "parabolic fit", 2 : "peak amplitude",
                             3 : "Temp 1", 4 : "Temp 2"},
                            inplace = True)
        #---------------------------------------------------------------------------
        #This section added by JJ
        # Plot data to identify correlation with temperature
        
        def plot_res_temp_date():

            fig, ax1 = plt.subplots(figsize=(10,6))
            color1 = 'tab:blue'
            color2 = 'tab:orange'
            color3 = 'yellow'

            # Plot the first y-axis
            ax1.set_xlabel('Datetime')
            ax1.set_ylabel('Resonance Frequency (Hz)', color = color1)

            # Setting Time Loctators and formats
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

            #Rotate x-axis labels
            plt.xticks(rotation=45)
            ax1.plot(dt_list, res_freq_lists[1], marker='o', linestyle='', color=color1, label='Resonance Frequency')
            ax1.tick_params(axis='y', labelcolor=color1)

            # Create a second y-axis sharing the same x-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel('Temperature (°C)', color = color2)
            ax2.plot(dt_list, temp1_list, marker='x', linestyle='', color=color2, label='Box Temperature')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.plot(dt_list, temp2_list, marker='x', linestyle='', color=color3, label='Cell Temperature')
             


            # Add a legend
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            # Add grid
            ax1.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.5)
            # Set automatic limits to x axis to ensure too much padding doesn't occur
            #ax1.set_xlim(dt_list.index.min(), dt_list.index.max())


            plt.title('4680 Cell Resonance Frequency and Temperature Fluctuation Over Time')
            fig.tight_layout()

            plt.savefig(f"res_temp_date_{start}-{end}.pdf", format='pdf')


        def plot_res_temp():
            plt.figure(figsize=(8,6))
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Resonance Frequency (Hz)')

            # Calculate equation for trendline
            z = np.polyfit(temp1_list, res_freq_lists[1], deg = 1)
            p = np.poly1d(z)

            plt.scatter(temp1_list, res_freq_lists[1], marker='o', linestyle='', color = 'tab:blue')
            plt.plot(temp1_list, p(temp1_list), color='black', linewidth=3)
            plt.grid(True)
            plt.title('4680 Cell Resonance Frequency vs Temperature')

            #plt.savefig(f"res_temp_{start}-{end}.pdf", format='pdf')
            
        # Export the plots
        plot_res_temp_date()
        plot_res_temp()
        #---------------------------------------------------------------------------
        
        df.to_csv(f"{output_data}\\res_freq\\_res_freq_{start}-{end}.csv")


    def deflection_exporter(self, fdata = None):
        """
        Exports Deflection Data
        
        If no formatted data is provided directly for export, the function will use the
        data saved in the self.formatted_data property.
        """
        # Save data to dummy variable
        if fdata == None:
            fdata = self.formatted_data

        # Creating Directory
        path = f"{self.options_dict['output_directory']}/Deflection vs Time/Deflection Data"
        os.makedirs(path, exist_ok = True)

        # Iterate through measurements and construct an export for each measurement
        for i, item in enumerate(fdata):
            i = i+1
            freq = item[0]
            time_data = item[1]
            def_data = item[2]
            data_dict = {"time(s)" : time_data, "Deflection (0.1 deg)" : def_data}
            df = pd.DataFrame(data_dict)

            name = f"test graph {i} of {len(fdata)}"

            self.def_df_archive.append(df)
            # Export data
            df.to_csv(f"{path}/{name}.csv")
            print(f"Data {name} exported")
    #-------------------------------------------------------------------------------



    """Amplitude Calculation Functions"""
    #---------------------------------------------------------------------------
    def Nicks_Sin_fit(self, time, angle, frequency, return_fit = False):
        """
        Takes the time and angle data, and drive frequency, and returns a sin curve 
        function fit to the time and angle data.

        Parameters:
            time: A 1d numpy array of time values, must be the same shape as angle
            angle: a 1d numpy array of angle values, must be the same shape as time
            frequency: a float value for the driving frequency
        
        Returns:
            sin_fit: a sin curve fit to the time array passed to the function
        """
        # Mean angle value
        mean = np.mean(angle)

        # Setting up X array
        X = np.ones([len(time), 3])
        X[:,1] = np.cos(2*np.pi*frequency*(time))
        X[:,2] = np.sin(2*np.pi*frequency*(time))

        # X transpose
        XT = X.T

        # (XT)X matrix multiplication
        XT_X = np.matmul(XT, X)
        inv_XT_X = np.linalg.inv(XT_X)

        # Calculating XTb
        b = angle[:, np.newaxis]
        XTb = np.matmul(XT, b)

        # Calculating a
        a = np.matmul(inv_XT_X, XTb)

        # Calculating Amplitude and phase
        A = np.sqrt(a[1,0]**2 + a[2,0]**2)
        phase_angle = np.arctan(-a[2,0]/a[1,0])

        sin_fit = mean + A*np.cos(2*np.pi*frequency*time+np.pi/1.75+phase_angle)

        if return_fit == True:
            return A, sin_fit
        else:
            return A


    def Average_Amplitude(self, angle):
        """
        Returns the average of the absolute value of the normed angle data as a float
        """
        mean = np.mean(angle)
        angle_normed = angle - mean
        abs_angle_normed = np.abs(angle_normed)
        A_average = np.mean(abs_angle_normed)*np.pi/2

        return A_average
    

    def Average_Temp(self, temp):
        """
        Returns the average of the temp data, as a float.
        """
        temp_mean = np.mean(temp)
        return temp_mean


    def Amplitude_max(self, angle):
        """
        Finds the maximum amplitude value for which at least 2% or more of the sample
        data is the same value.

        Returns maximum amplitude value as a float
        """
        mean = np.round(np.mean(angle))
        angle_normed = angle - mean
        abs_angle_normed = (np.abs(angle_normed)).astype(int)

        unique_angles = np.unique((np.abs(abs_angle_normed)))
        frequency_count = np.bincount(abs_angle_normed)

        for i in range(1,len(unique_angles)+1):
            # print(frequency_count[unique_angles[-i]], unique_angles[-i])
            if len(angle)*0.02 < frequency_count[unique_angles[-i]]:
                A_max = unique_angles[-i]
                break
        return A_max



    """Curve Fitting Functions"""
    #---------------------------------------------------------------------------
    def parabolic_fit(self, A_sol_list, peak_width = 10, step_size = None):
        """First we need to dynamically find the xdata and ydata values for the 
        peak from the data"""
        
        if step_size == None:
            step_size = self.step_size
        # Calculating number of index values for half the width of the peak
        half_width_in_steps = int(peak_width/(2*step_size))
        
        # Retrieve Amplitude and frequency values
        sinfit_amplitudes = A_sol_list[1]
        frequency = A_sol_list[0]

        # Getting indices of the maximum amplitude value
        Amax = sinfit_amplitudes.max()
        Amax_ind = np.where(sinfit_amplitudes == Amax)[0][0]

        # Number of data points between Amax and each end of the sampling interval
        steps_A_to_start = len(frequency[:Amax_ind])
        steps_A_to_end = len(frequency[Amax_ind:])

        # Getting a value for 10% of the sample points
        sample_size_10_percent = int(len(frequency) /10)

        """Noise Finder
        Counts data within half_width of the curve on either side of Amax as the
        peak. If the data set doesn't have large enough tails for this to work 
        then we simply use the average of the first and last 10% of the data in
        the sample set presuming the peak is in between.

        This needs to be improved.
        1. What if the peak is on the edge of the data set?
        2. this is currently complex and hard to read. Clarify it
        """
        if half_width_in_steps >  steps_A_to_end:
            b_vals = []
            for i in range(1, sample_size_10_percent+1):
                b_vals.append(sinfit_amplitudes[-i])
            b = np.mean(np.array(b_vals))
        else:
            b = sinfit_amplitudes[Amax_ind + half_width_in_steps-1]

        if half_width_in_steps >  steps_A_to_start:
            a_vals = []
            for i in range(1, sample_size_10_percent + 1):
                a_vals.append(sinfit_amplitudes[i])
            a = np.mean(np.array(a_vals))
        else:
            a = sinfit_amplitudes[Amax_ind - half_width_in_steps]

        offset = (b+a)/2

        A_fw_hm = (0.5)*Amax + (0.5)*offset

        # Slicing data above full width half max Amp
        ind_fit = np.where(sinfit_amplitudes >= A_fw_hm)

        p_fit_y = sinfit_amplitudes[ind_fit]
        p_fit_x = frequency[ind_fit]

        # Solving 2nd order Polyfit (Parabolic fit)
        C = np.polyfit(p_fit_x, p_fit_y, 2)
        fit_x = np.roots([C[0], C[1], C[2]-A_fw_hm])

        x_coords = np.linspace(fit_x[0], fit_x[1], 1000)
        y_fit = C[0]*(x_coords**2) + C[1]*x_coords + C[2]
        # print(f"x_coords are {x_coords}")
        resonant_freq = np.mean(fit_x)

        resonant_amplitude = C[0]*resonant_freq**2 + C[1]*resonant_freq + C[2]
        self.res_freq_amp.append(resonant_amplitude)


        return [resonant_freq, x_coords, y_fit]


    def lorentz_fit(self, A_sol_list, start_tail = 5, end_tail = 5, peak_width = 15):
        
        """Defining Lorentz Function"""
        def CF_lorentz_curve(x, x0, gamma, z):
            """
            x0 = location parameter, specifies location of the peak of the distribution
            gamma/y = scale parameter, specifies the half-width at half-max
                        2y would be FWHM
            """
            y = gamma

            lorentz = z * ((y**2) / ((x-x0)**2 + y**2))
            return lorentz


        # """Dynamically estimating the start and end of the peak from the data"""
        # # Retrieve Amplitude (yvals) and frequency (xvals) values
        ydata = A_sol_list[1]
        xdata = A_sol_list[0]

        # # Find the frequencies of the 5 largest amplitude values, take the mean 
        # # as the estimate for the location of the center of the peak of the 
        # # dataset.
        # y_index_sorted = np.argsort(np.array(ydata))
        # x_sorted = xdata[y_index_sorted]
        # length = len(x_sorted)

        # freq_of_A5_largest = []
        # for i in range(length-5, length):
        #     freq_of_A5_largest.append(x_sorted[i])

        # # Taking the mean as estimated peak frequency
        # freq_of_A5_largest_array = np.array(freq_of_A5_largest)
        # estimated_peak_freq = np.mean(freq_of_A5_largest_array)

        # """All x values +-  (1/2*peak_width) hz from the estimated peak_freq
        # are used for the curve fitting"""
        # x0 = estimated_peak_freq
        # HWHM = peak_width/2

        # where = np.where((xdata>(x0-HWHM)) & (xdata<(x0+HWHM)))
        # xfit = xdata[where]
        # yfit = ydata[where]
        # print(HWHM)
        # print(nf)
        
        #TEMP
        xfit = xdata
        yfit = ydata

        """Estimating background noise"""
        avg_start_noise = np.mean(ydata[0:start_tail])
        avg_end_noise = np.mean(ydata[-end_tail: -1])
        nf = (avg_start_noise+avg_end_noise)/2 # noise factor
        # print(nf)
        # print(f"len x is {len(xfit)} array is {xfit}")
        # print(f"len x is {len(xother)} array is {xother}")
        # print(f"len y is {len(yfit)} array is {yfit}")

        # Calculating least squares fitting of an optimized lorentz curve, 
        # subtracting background noise from the amplitude values.
        try:
            popt, pcov = curve_fit(CF_lorentz_curve, xfit, yfit-nf)
        except:
            print("Lorentz fit failed for current file")
            return [0, 0, 0, 0, 0]

        # Defining output values
        resonant_frequency = popt[0] # This is the center of  the peak
        FWHM = 2*popt[1] # 2x gamma value from lorentz curve is FWHM
        xdata = np.linspace(xfit[0],xfit[-1], 1000)
        ydata = CF_lorentz_curve(xdata, *popt) + nf # Adding back noise for fit
        height_FWHM = ((np.max(ydata)-nf)/2) + nf

        return resonant_frequency, xdata, ydata, FWHM, height_FWHM


    # Unfinished Peak Search Function
    # def Amax_finder(freq_array, angle_array, step_size=1):
    #     Amax = np.max(angle_array) 
    #-------------------------------------------------------------------------------