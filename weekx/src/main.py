import pandas as pd
import data_process
import numpy as np
import fst




def submission(flag_y, df_test, file = '../data/submission.csv'):
	sub = pd.DataFrame()
	sub['id'] = df_test['id']
	print "please implement!"
	print "Submission samples:%d, file:%s"%(len(df_test), file)
	sub.to_csv(file, index=False)



def experiment():
	df_train = pd.read_csv("../data/en_train.csv")
	df_test = pd.read_csv("../data/en_test.csv")
	x_train, y_train, x_test, _ = data_process.gen_data(df_train, df_test)
	print "please implement!"


def loadModel(path):
	return None


def classify(df):
	model = loadModel('../model/xgb_model')
	data = data_process.gen_features(df)



if __name__ == "__main__":
# 	experiment()
# 	t = fst.Transducer()
# 	t.add_arc(0, 1, 'a', 'A')
# 	t.add_arc(0, 1, 'b', 'B')
# 	t.add_arc(1, 2, 'c', 'C')
# 	
# 	t[2].final = True
# 	
# 	print t.shortest_path()
# 	t.write("data/fst.txt")
	
	
	

