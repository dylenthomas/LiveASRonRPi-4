import numpy as np

class makeArrayFromTextGrid():
    def __init__(self, sample_rate:int, dict_path:str):
        """Parse a directory of TextGrid files and encodes as a numpy array where each sample corresponds to a word spoken 

        Args:
            file_path (str): path to TextGrid file
            sample_rate (int): sample rate the of the audio files
            dict_path (str): the path to the dictionary used for encoding
        """
        self.sample_rate = sample_rate
        self.dict_path = dict_path
        self.load_dict()
        
    def processTextGrid(self, filename:str):
        """Process a given TextGrid file and return its contents in a numpy array

        Args:
            filename (str): path to the TextGrid file
        """
        data = self.read_TextGrid(filename)
        return self.create_array(data)
                    
    def create_array(self, data:dict):
        """Create a numpy array with extracted TextGrid data

        Args:
            data (dict): dictionary containing text grid data

        Raises:
            Exception: the TextGrid file contains non IntervalTier classes
        """
        for chunk in data:
            if chunk['class'] == 'IntervalTier' and chunk['name'] == 'words':
                data_len = float(chunk['xmax']) * self.sample_rate #Length of data in samples
                data_len = int(data_len)
                num_items = int(chunk['intervals: size']) #Number of intervals found
                word_array = np.empty([1, data_len], dtype=int)
                
                for i in range(num_items):
                    i += 1
                    interval_data = chunk[f'intervals[{i}]']

                    #Convert interval period to samples instead of seconds
                    interval_start = float(self.extract_info(interval_data[0], 'xmin')) * self.sample_rate
                    interval_end = float(self.extract_info(interval_data[1], 'xmax')) * self.sample_rate
                    
                    interval_start = int(interval_start)
                    interval_end = int(interval_end)
                 
                    #insert encoded words into numpy array to be stored in dataframe
                    info = self.extract_info(interval_data[2], 'text')
                    word_array[0, interval_start:interval_end] = self.encode(info)

            elif chunk['class'] != 'IntervalTier':
                raise Exception(f'Unknown class type:"{chunk['class']}"')
            
        return word_array
            
    def encode(self, text:str):
        try:
            return self.dictionary[text] if text != '' else self.dictionary['<eps>'] 
        except KeyError: #If there is an appostrophe in the word that shouldn't be
            text = text.replace("'", '')
            try:
                return self.dictionary[text]
            except KeyError: #If the word is just not in the dictionary
                self.add_to_dict(text)
                return self.dictionary[text]
            
    def add_to_dict(self, text):
        """Add an unknown word to the dictionary
        """
        print(f"{text} wasn't found in the provided dictionary, adding...")
        #Add the term to the dictionary
        with open(self.dict_path, 'a') as f:
            f.write(f"{text}\t{len(self.dictionary)}\n")
            f.close() 
            
        #Reload the dictionary
        del(self.dictionary)
        self.load_dict()
                 
    def load_dict(self):
        """Load the dictionary and put it into a dict
        """
        with open(self.dict_path, 'r') as f:
            dictionary_list = f.read().replace(' ', '').split('\n')
            del(dictionary_list[len(dictionary_list) - 1]) #remove the empty string at the end
            f.close()
            
        #Create python dictionary for the dictionary
        self.dictionary = {}
        for entry in dictionary_list:
            entry = entry.split('\t')
            self.dictionary[entry[0]] = entry[1]
            
    def read_TextGrid(self, path:str):
        """Extract data from TextGrid and put into dictionary

        Args:
            path (str): path to individual TextGrid file

        Returns:
            dict: dictionary containing data of TextGrid file
        """
        with open(path, 'r') as textgrid:
            data = textgrid.read().split('\n')
            textgrid.close()

        item_loc = []
        for i, line in enumerate(data):
            if 'item [' in line:
                item_loc.append(i)
            
        del(item_loc[0]) #remove the item class identifier line
        item_loc.append(len(data) - 1) #add the length of the file
        
        items = []
        for i in range(len(item_loc) - 1):
            items.append(data[item_loc[i] + 1: item_loc[i + 1]])
            
        #Parse items
        all_item_info = []
        for item in items:
            item = [_.replace(' ', '') for _ in item] #remove redundant spaces
            
            #Get item parameters
            params = item[0:5]
            item_info = {}
            item_info['class'] = self.extract_info(params[0], 'class')
            item_info['name'] = self.extract_info(params[1], 'name')
            item_info['xmin'] = self.extract_info(params[2], 'xmin')
            item_info['xmax'] = self.extract_info(params[3], 'xmax')
            item_info['intervals: size'] = self.extract_info(params[4], 'intervals:size')
            
            #Extract item intevervals
            chunk = item[5:len(item)]
            interval = len(chunk) // int(item_info['intervals: size'])
            for i in range(int(item_info['intervals: size'])):
                interval_data = item[(i*interval + 5):(i*interval + 5 + interval)]
                item_info[interval_data[0].replace(':', '')] = interval_data[1:len(interval_data)]
            
            all_item_info.append(item_info)
            
        return all_item_info
    
    def extract_info(self, data, entry_type):
        for thing in [entry_type, '"', '=']:
            if thing in data: data = data.replace(thing, '')
        
        return data
    

if __name__ == '__main__':
    TG_processor = makeArrayFromTextGrid(441000, 'forced_alignment_test/', 'dictionary/words.txt')
    print(TG_processor.processTextGrid('forced_alignment_test/19-198-0001.TextGrid'))