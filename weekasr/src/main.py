import pandas as pd
import data_process
import numpy as np





def submission(flag_y, df_test, file = '../data/submission.csv'):
	sub = pd.DataFrame()
	sub['id'] = df_test['id']
	print "please implement!"
	print "Submission samples:%d, file:%s"%(len(df_test), file)
	sub.to_csv(file, index=False)



def experiment():
	df_train = pd.read_csv("../data/train.csv")
	df_test = pd.read_csv("../data/test.csv")
	x_train, y_train, x_test, _ = data_process.gen_data(df_train, df_test)
	print "please implement!"







if __name__ == "__main__":
	experiment()



