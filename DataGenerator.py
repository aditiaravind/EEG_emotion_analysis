import glob
import pickle
import numpy as np
import pandas as pd


#CLASS to Split, Concatenate and return the Dataset according to the defined "Mode" of Splitting
#Mode of splitting can be one of the following:
#     's_dep': Subject Dependent - requires a patient ID to be specified
#     's_indep': Subject Independent 
#     's_male': By Male Gender 
#     's_fem': By Female Gender 
#     's02': Split into 2 random groups 
#     's08': Split into 8 random groups 
#     's16': Split into 16 random groups


class DataGenerator():
    
    def __init__(self, datapath = r"C:\Users\Aditi\Documents\Machine Learning\Upwork Project\data_preprocessed_python\data_preprocessed_python\Final_features\feats", 
                 metapath = r"C:\Users\Aditi\Documents\Machine Learning\Upwork Project\metadata_csv\participant_questionnaire.csv", reshape=True):
        
        #Initialization function - sets variables and their default values for use in the class functions.
        
        self.data_path = datapath #path to the pickled feature files
        self.metadata_path = metapath #path to the participant_uestionnair.csv file
        self.reshape = reshape #Flag denotng if data should be reshaped before returning
        self.modes = ['s_dep', 's_indep', 's_male', 's_fem', 's08', 's16'] #Defined modes for splitting and collecting data
        self.pid = '01' #Default patient ID value
        self.rand = False #Random splitting flag
    
    
    def load_data(self, pid):
    #Base loading function. 
    #Returns the features and labels from the "feats-xx.pickle" files 
    
        with open(self.data_path + pid + '.pickle', 'rb') as handle:
            data = pickle.load(handle)
        return data['features'], data['labels']
    
    def reshape_toggle(self):
    #Function to toggle the reshape variable.
    #To be called if you want to avoid reshaping the data and only return the features as they were
        self.reshape = not self.reshape
        
    def check_mode(self, mode):
    #Checks if the given mode is one of the defined modes (to avoid errors)
        if mode not in self.modes:
            raise ModeError(mode)
        else:
            return mode
    
    def get_meta(self, gender=None):
    #Returns the list of patient IDs, male/female, or all as required from the metadata - patient questionnaire
        meta = pd.read_csv(self.metadata_path, usecols=['Participant_id','Gender'])
        meta['Participant_id'] = meta['Participant_id'].apply(lambda x: x[1:])
        if gender:
            return (meta['Participant_id'][meta['Gender']==gender]).to_numpy(dtype='str')
        else:
            return meta['Participant_id'].to_numpy(dtype='str')
        
    def concat_data(self, pids):
    #Concatenates all the data together that is defined in the pids variable and returns data, labels
        data = []
        labels = []
        
        #Iterates over all the patient IDs in 
        for ID in pids:
            d, l = self.load_data(ID)
            data.append(d)
            labels.append(l)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        #Sanity Check
        if data.shape[1:] != (32,5,18):
            print("Error: Shape mismatch in Data Variable")
        if labels.shape[1:] != (3,):
            print("Error: Shape mismatch in Labels Variable")
        
        #If required to reshape into (n_sample, n_features)
        if self.reshape:
            data = data.reshape(data.shape[0]*32, 5*18)
            labels = np.repeat(labels, 32, axis=0)
            
        return data, labels

    ## MAIN CALL FUNCTION
    def gen_data(self, mode = 's_indep', pid='01'):
    #Main Function to collect required data from the feature files in the defined mode and concatenate them
    #Reshape if necessary into (n_samples, n_features)
    #Return data, labels respectively
        
        self.rand=False #Set the custom random flag false - for random splits - the data generated is in the form of a list of data from each random                            split. Else, it only returns a single data and label set 
        
        self.mode = self.check_mode(mode) #Ensures that the mode is from the list of pre-defined modes (to avoid errors)
        
        #Generate a list of patient IDs respective to the mode of data splitting
        
        #s_male returns all the IDs of male patients
        if self.mode == 's_male':
            self.pid = self.get_meta("Male")
        #s_fem returns all the IDs of female patients
        elif self.mode == 's_fem':
            self.pid = self.get_meta("Female")
        
        #s_dep can be used for a single patient ID or a list of patient IDs
        elif self.mode == 's_dep':
            if type(pid)==list:
                self.pid = pid
            else:
                self.pid = [pid]
        
        #s_indep returns all the IDs of all patients
        elif self.mode == 's_indep':
            self.pid = self.get_meta()
        
        #for the random splits ("sN") - returns "N" array of arrays, each having 2, 8 or 16 patient IDs 
        else:
            self.rand = True
            pids = self.get_meta()
            groups = int(self.mode[-2:])
            np.random.shuffle(pids)
            self.pid = np.split(pids, groups)
            print("The Split Groupings are: ")
            for p in self.pid:
                print(p)
        
        #Now collect data for the respective patient IDs defined in the variable self.pid
        
        #Random group flag is true - returns a list of data
        if self.rand:
            data = []
            labels = []
            for p in self.pid:
                d,l = self.concat_data(p)
                data.append(d)
                labels.append(l)
        
        #Random group flag is false - returns just data and label for defined mode
        else:
            data, labels = self.concat_data(self.pid)
        #Return
        return data, labels
#___________________________________________________________________#

#Custom error function, not important 
# It creates a custom error message to be displayed if the mode input given is not from one of the pre-defined modes of splitting data
class ModeError(Exception):
    def __init__(self, mode, 
                 message="is not present in the defined options:{'s_dep', 's_indep', 's_male', 's_fem', 's02', 's08', 's16'} "):
        self.mode = mode
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.mode} -> {self.message}'

