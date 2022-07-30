import pandas as pd
from sklearn.model_selection import train_test_split
import re

class DataLoader():
    def __init__(self, data_list:list):
        self.raw_data = pd.DataFrame(data_list)  
        if self.raw_data.size > 0:      
            self.raw_data.columns=[x.lower() for x in self.raw_data.columns]
            self.raw_data = self.raw_data.rename(
                columns={
                    "abstract":"article",
                    "ab":"article",
                    "ti":"title",
                    "relevance":"relevance_label"
                }
            )
            if "relevance_label" not in self.raw_data.columns: self.raw_data["relevance_label"] = ""
        
            #Remove weird characters        
            self.raw_data['title'] = self.raw_data['title'].map(self.remove_weird_characters)
            self.raw_data['article'] = self.raw_data['article'].map(self.remove_weird_characters)  

            self.raw_data["abstract"] = '[' + self.raw_data.index.astype(str) + '] ' + \
                                            self.raw_data['title'] + ' --- ' + \
                                            self.raw_data['article'].fillna('')
                                            
            self.raw_data = self.raw_data[self.raw_data["abstract"].notna()]
            self.raw_data = self.raw_data[self.raw_data["relevance_label"].notna()]
            self.raw_data.reset_index(inplace=True)

    def remove_weird_characters(self,text):
        text = str(text)
        text = text.encode("ascii","ignore")
        text = text.decode("latin-1")
        text = re.sub(r'\r\n',' ',text)
        return text

    def load_train_test(self):
        y = self.raw_data['relevance_label'].tolist()
        x_cols = self.raw_data.columns
        X = self.raw_data[x_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=.25,random_state=1973)
        return X_train, X_test, y_train, y_test

    def load_full_test(self):
        y_test = self.raw_data['relevance_label'].tolist()
        x_cols = self.raw_data.columns
        y_train = self.raw_data[x_cols]
        return y_train, y_test
    
    def load_full_train(self):
        return self.raw_data



if __name__ == "__main__":
    pass
